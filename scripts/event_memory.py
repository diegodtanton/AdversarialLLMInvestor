from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "news.db"

LATEST_PATH = DATA_DIR / "latest.json"
ENRICHED_LATEST_PATH = DATA_DIR / "enriched_latest.json"

DATA_DIR.mkdir(exist_ok=True)

load_dotenv(ROOT / ".env")

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

client = OpenAI(
    api_key=OPENAI_API_KEY,
    max_retries=3,
    timeout=180.0,
)

EVENT_MODEL = os.getenv(
    "EVENT_MODEL",
    "gpt-5.6-terra",
)

EVENT_REASONING = os.getenv(
    "EVENT_REASONING",
    "medium",
)

EMBEDDING_MODEL = "text-embedding-3-small"

# Using shortened embeddings keeps the persistent database and
# in-memory event index small.
EMBEDDING_DIMENSIONS = 512

BATCH_SIZE = 10
MAX_EVENT_CANDIDATES = 6
TIMELINE_ITEMS_PER_EVENT = 4


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# DATABASE
# ============================================================

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    canonical_title TEXT NOT NULL,
    summary TEXT NOT NULL,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS stories (
    story_id TEXT PRIMARY KEY,
    story_date TEXT NOT NULL,
    story_rank INTEGER NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,

    event_id TEXT NOT NULL,

    update_type TEXT NOT NULL,
    relationship TEXT NOT NULL,
    novelty REAL NOT NULL,

    historical_note TEXT NOT NULL,
    related_event_id TEXT,

    key_prior_story_ids_json TEXT NOT NULL,

    created_at TEXT NOT NULL,

    FOREIGN KEY(event_id)
        REFERENCES events(event_id),

    FOREIGN KEY(related_event_id)
        REFERENCES events(event_id)
);

CREATE INDEX IF NOT EXISTS idx_stories_date
ON stories(story_date);

CREATE INDEX IF NOT EXISTS idx_stories_event
ON stories(event_id);

CREATE INDEX IF NOT EXISTS idx_events_last_seen
ON events(last_seen);

CREATE TABLE IF NOT EXISTS event_relationships (
    from_event_id TEXT NOT NULL,
    to_event_id TEXT NOT NULL,
    relationship TEXT NOT NULL,
    created_at TEXT NOT NULL,

    PRIMARY KEY(
        from_event_id,
        to_event_id,
        relationship
    ),

    FOREIGN KEY(from_event_id)
        REFERENCES events(event_id),

    FOREIGN KEY(to_event_id)
        REFERENCES events(event_id)
);
"""


def connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)

    conn.row_factory = sqlite3.Row

    conn.executescript(SCHEMA_SQL)

    return conn


# ============================================================
# GENERAL HELPERS
# ============================================================

def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def normalized_title(value: str) -> str:
    value = value.casefold()

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    return " ".join(
        value.split()
    )


def make_story_id(
    story_date: str,
    story: dict,
) -> str:
    """
    Stable ID based on date + canonical headline.

    We do not use rank because rank can change if the collector
    is rerun on the same day.
    """

    if story.get("story_id"):
        return story["story_id"]

    digest = hashlib.sha1(
        normalized_title(
            story["title"]
        ).encode("utf-8")
    ).hexdigest()[:12]

    compact_date = story_date.replace(
        "-",
        "",
    )

    return f"story_{compact_date}_{digest}"


def story_embedding_text(
    story: dict,
) -> str:
    return (
        f"Headline: {story['title']}\n"
        f"Description: {story.get('description', '')}"
    )


def event_embedding_text(
    title: str,
    summary: str,
) -> str:
    return (
        f"Event: {title}\n"
        f"History: {summary}"
    )


# ============================================================
# EMBEDDINGS
# ============================================================

def embed_texts(
    texts: list[str],
) -> np.ndarray:

    if not texts:
        return np.empty(
            (0, EMBEDDING_DIMENSIONS),
            dtype=np.float32,
        )

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        dimensions=EMBEDDING_DIMENSIONS,
    )

    ordered = sorted(
        response.data,
        key=lambda item: item.index,
    )

    matrix = np.asarray(
        [
            item.embedding
            for item in ordered
        ],
        dtype=np.float32,
    )

    # Normalize so dot product = cosine similarity.
    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True,
    )

    norms[norms == 0] = 1

    return matrix / norms


def vector_to_blob(
    vector: np.ndarray,
) -> bytes:
    return np.asarray(
        vector,
        dtype=np.float32,
    ).tobytes()


def blob_to_vector(
    blob: bytes,
) -> np.ndarray:
    vector = np.frombuffer(
        blob,
        dtype=np.float32,
    ).copy()

    norm = np.linalg.norm(vector)

    if norm:
        vector /= norm

    return vector


# ============================================================
# LEXICAL RETRIEVAL
# ============================================================

STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "at",
    "by",
    "with",
    "from",
    "after",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "has",
    "have",
    "had",
    "says",
    "say",
    "amid",
    "over",
    "into",
    "new",
}


def lexical_tokens(
    value: str,
) -> set[str]:

    words = re.findall(
        r"[a-z0-9][a-z0-9.-]+",
        value.casefold(),
    )

    return {
        word
        for word in words
        if (
            len(word) >= 3
            and word not in STOP_WORDS
        )
    }


def lexical_similarity(
    query: str,
    document: str,
) -> float:

    q = lexical_tokens(query)
    d = lexical_tokens(document)

    if not q or not d:
        return 0.0

    intersection = len(
        q.intersection(d)
    )

    # More useful than pure Jaccard for proper nouns because
    # we care how much of the query is explained by the event.
    return intersection / len(q)


# ============================================================
# EVENT INDEX
# ============================================================

def load_event_index(
    conn: sqlite3.Connection,
) -> list[dict]:

    rows = conn.execute(
        """
        SELECT
            event_id,
            canonical_title,
            summary,
            first_seen,
            last_seen,
            embedding
        FROM events
        """
    ).fetchall()

    events = []

    for row in rows:
        vector = blob_to_vector(
            row["embedding"]
        )

        # Protect against changing EMBEDDING_DIMENSIONS later.
        if len(vector) != EMBEDDING_DIMENSIONS:
            continue

        events.append(
            {
                "event_id": row["event_id"],
                "canonical_title": row["canonical_title"],
                "summary": row["summary"],
                "first_seen": row["first_seen"],
                "last_seen": row["last_seen"],
                "vector": vector,
            }
        )

    return events


def get_recent_event_timeline(
    conn: sqlite3.Connection,
    event_id: str,
    limit: int = TIMELINE_ITEMS_PER_EVENT,
) -> list[dict]:

    rows = conn.execute(
        """
        SELECT
            story_id,
            story_date,
            story_rank,
            title,
            description,
            update_type,
            novelty
        FROM stories
        WHERE event_id = ?
        ORDER BY story_date DESC, story_rank ASC
        LIMIT ?
        """,
        (
            event_id,
            limit,
        ),
    ).fetchall()

    timeline = [
        {
            "story_id": row["story_id"],
            "date": row["story_date"],
            "title": row["title"],
            "description": row["description"],
            "update_type": row["update_type"],
            "novelty": row["novelty"],
        }
        for row in rows
    ]

    timeline.reverse()

    return timeline


def retrieve_event_candidates(
    conn: sqlite3.Connection,
    event_index: list[dict],
    story: dict,
    story_vector: np.ndarray,
) -> list[dict]:

    if not event_index:
        return []

    query_text = (
        f"{story['title']} "
        f"{story.get('description', '')}"
    )

    scored = []

    for event in event_index:
        semantic = float(
            np.dot(
                story_vector,
                event["vector"],
            )
        )

        lexical = lexical_similarity(
            query_text,
            (
                f"{event['canonical_title']} "
                f"{event['summary']}"
            ),
        )

        # Semantic retrieval carries most of the weight.
        combined = (
            0.87 * semantic
            + 0.13 * lexical
        )

        scored.append(
            (
                combined,
                semantic,
                lexical,
                event,
            )
        )

    scored.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    results = []

    for (
        combined,
        semantic,
        lexical,
        event,
    ) in scored[:MAX_EVENT_CANDIDATES]:

        results.append(
            {
                "event_id": event["event_id"],
                "canonical_title": (
                    event["canonical_title"]
                ),
                "summary": event["summary"],
                "first_seen": event["first_seen"],
                "last_seen": event["last_seen"],
                "retrieval_score": round(
                    combined,
                    4,
                ),
                "semantic_similarity": round(
                    semantic,
                    4,
                ),
                "lexical_similarity": round(
                    lexical,
                    4,
                ),
                "recent_timeline": (
                    get_recent_event_timeline(
                        conn,
                        event["event_id"],
                    )
                ),
            }
        )

    return results


# ============================================================
# STRUCTURED OUTPUT
# ============================================================

EVENT_LINK_SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "story_id": {
                        "type": "string",
                    },
                    "matched_event_id": {
                        "anyOf": [
                            {
                                "type": "string",
                            },
                            {
                                "type": "null",
                            },
                        ]
                    },
                    "relationship": {
                        "type": "string",
                        "enum": [
                            "SAME_EVENT",
                            "CAUSE_OF",
                            "CONSEQUENCE_OF",
                            "RELATED_CONTEXT",
                            "UNRELATED",
                        ],
                    },
                    "update_type": {
                        "type": "string",
                        "enum": [
                            "NEW_EVENT",
                            "MAJOR_DEVELOPMENT",
                            "MINOR_DEVELOPMENT",
                            "REACTION_OR_COMMENTARY",
                            "RECAP_OR_BACKGROUND",
                            "RESOLUTION",
                        ],
                    },
                    "novelty": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "canonical_event_title": {
                        "type": "string",
                    },
                    "event_summary": {
                        "type": "string",
                    },
                    "historical_note": {
                        "type": "string",
                    },
                    "key_prior_story_ids": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                    },
                },
                "required": [
                    "story_id",
                    "matched_event_id",
                    "relationship",
                    "update_type",
                    "novelty",
                    "canonical_event_title",
                    "event_summary",
                    "historical_note",
                    "key_prior_story_ids",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "decisions",
    ],
    "additionalProperties": False,
}


EVENT_MEMORY_INSTRUCTIONS = """
You maintain a persistent temporal memory of major news events.

You receive up to 10 CURRENT STORIES. For each current story you also
receive zero or more CANDIDATE HISTORICAL EVENTS retrieved from a
database.

Your job is to decide how the current story relates to history.

Be conservative about SAME_EVENT.

Two stories are SAME_EVENT when they concern the same persistent
real-world occurrence, case, transaction, investigation, conflict,
policy action, disaster, corporate situation, legal proceeding, etc.

Merely sharing:
- a person,
- company,
- country,
- industry,
- war,
- broad political issue,
- product,
or theme

does NOT make two stories the same event.

Examples:

Original:
"Boeing loses $20 billion lawsuit"

Later:
"Boeing CEO discusses $20 billion lawsuit"

=> SAME_EVENT
=> likely REACTION_OR_COMMENTARY
=> novelty should be relatively low.

Original:
"Russia cuts gas supplies to Europe"

Later:
"European fertilizer plants shut as gas prices surge"

=> not SAME_EVENT.
The second may be CONSEQUENCE_OF the first.

For every current story:

1. Select matched_event_id ONLY from its supplied candidates, or null.
2. Decide relationship.
3. Classify update_type.
4. Estimate novelty from 0 to 1.
5. Produce a compact canonical event title.
6. Produce an UPDATED rolling event summary.
7. Write a historical_note explaining what today's headline adds
   relative to what was already known.
8. Identify any especially important prior story IDs.

If SAME_EVENT:
- matched_event_id must be populated.
- event_summary must update that existing event summary with the new
  information while preserving important prior chronology.

If the story deserves its own new event:
- relationship may be UNRELATED, CAUSE_OF, CONSEQUENCE_OF, or
  RELATED_CONTEXT.
- canonical_event_title and event_summary describe the NEW event.
- If there is an important historical relationship, matched_event_id
  points to that older event.

Do not overstate novelty.
A dramatic headline can still contain little new information.
"""


# ============================================================
# EVENT MATCHING
# ============================================================

def link_story_batch(
    batch_payload: list[dict],
) -> list[dict]:

    response = client.responses.create(
        model=EVENT_MODEL,
        reasoning={
            "effort": EVENT_REASONING,
        },
        instructions=EVENT_MEMORY_INSTRUCTIONS,
        input=json.dumps(
            {
                "current_stories": batch_payload,
            },
            ensure_ascii=False,
        ),
        text={
            "format": {
                "type": "json_schema",
                "name": "event_memory_decisions",
                "strict": True,
                "schema": EVENT_LINK_SCHEMA,
            }
        },
        max_output_tokens=12000,
        store=False,
    )

    if not response.output_text:
        raise RuntimeError(
            "Event Memory returned no output."
        )

    result = json.loads(
        response.output_text
    )

    return result["decisions"]


def create_event_id() -> str:
    return (
        "evt_"
        + uuid.uuid4().hex[:16]
    )


def refresh_event_embeddings(
    conn: sqlite3.Connection,
    event_ids: set[str],
) -> None:

    if not event_ids:
        return

    rows = []

    for event_id in event_ids:
        row = conn.execute(
            """
            SELECT
                event_id,
                canonical_title,
                summary
            FROM events
            WHERE event_id = ?
            """,
            (event_id,),
        ).fetchone()

        if row:
            rows.append(row)

    if not rows:
        return

    vectors = embed_texts(
        [
            event_embedding_text(
                row["canonical_title"],
                row["summary"],
            )
            for row in rows
        ]
    )

    for row, vector in zip(
        rows,
        vectors,
    ):
        conn.execute(
            """
            UPDATE events
            SET
                embedding = ?,
                updated_at = ?
            WHERE event_id = ?
            """,
            (
                vector_to_blob(vector),
                utc_now(),
                row["event_id"],
            ),
        )


def apply_decision(
    conn: sqlite3.Connection,
    story_date: str,
    story: dict,
    story_id: str,
    story_vector: np.ndarray,
    candidates: list[dict],
    decision: dict,
) -> str:

    supplied_event_ids = {
        candidate["event_id"]
        for candidate in candidates
    }

    matched_event_id = (
        decision["matched_event_id"]
    )

    if (
        matched_event_id
        and matched_event_id
        not in supplied_event_ids
    ):
        logger.warning(
            "Model returned event not supplied for %s: %s",
            story_id,
            matched_event_id,
        )

        matched_event_id = None

    relationship = decision[
        "relationship"
    ]

    # --------------------------------------------------------
    # SAME EVENT
    # --------------------------------------------------------

    if (
        relationship == "SAME_EVENT"
        and matched_event_id
    ):
        event_id = matched_event_id

        existing = conn.execute(
            """
            SELECT first_seen
            FROM events
            WHERE event_id = ?
            """,
            (event_id,),
        ).fetchone()

        if not existing:
            raise RuntimeError(
                f"Missing event {event_id}"
            )

        conn.execute(
            """
            UPDATE events
            SET
                canonical_title = ?,
                summary = ?,
                last_seen = ?,
                updated_at = ?
            WHERE event_id = ?
            """,
            (
                decision[
                    "canonical_event_title"
                ],
                decision[
                    "event_summary"
                ],
                story_date,
                utc_now(),
                event_id,
            ),
        )

        related_event_id = None

    # --------------------------------------------------------
    # NEW EVENT
    # --------------------------------------------------------

    else:
        event_id = create_event_id()

        related_event_id = (
            matched_event_id
            if (
                matched_event_id
                and relationship
                in {
                    "CAUSE_OF",
                    "CONSEQUENCE_OF",
                    "RELATED_CONTEXT",
                }
            )
            else None
        )

        conn.execute(
            """
            INSERT INTO events (
                event_id,
                canonical_title,
                summary,
                first_seen,
                last_seen,
                embedding,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                decision[
                    "canonical_event_title"
                ],
                decision[
                    "event_summary"
                ],
                story_date,
                story_date,
                vector_to_blob(
                    story_vector
                ),
                utc_now(),
                utc_now(),
            ),
        )

        if related_event_id:
            conn.execute(
                """
                INSERT OR IGNORE INTO
                event_relationships (
                    from_event_id,
                    to_event_id,
                    relationship,
                    created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    event_id,
                    related_event_id,
                    relationship,
                    utc_now(),
                ),
            )

    conn.execute(
        """
        INSERT INTO stories (
            story_id,
            story_date,
            story_rank,
            title,
            description,
            event_id,
            update_type,
            relationship,
            novelty,
            historical_note,
            related_event_id,
            key_prior_story_ids_json,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            story_id,
            story_date,
            story["rank"],
            story["title"],
            story.get(
                "description",
                "",
            ),
            event_id,
            decision["update_type"],
            relationship,
            float(
                decision["novelty"]
            ),
            decision[
                "historical_note"
            ],
            related_event_id,
            json.dumps(
                decision[
                    "key_prior_story_ids"
                ]
            ),
            utc_now(),
        ),
    )

    return event_id


# ============================================================
# PROCESS A DAILY NEWS FILE
# ============================================================

def process_news_payload(
    conn: sqlite3.Connection,
    payload: dict,
) -> None:

    story_date = payload["date"]

    pending = []

    for story in payload["stories"]:
        story_id = make_story_id(
            story_date,
            story,
        )

        exists = conn.execute(
            """
            SELECT 1
            FROM stories
            WHERE story_id = ?
            """,
            (story_id,),
        ).fetchone()

        if exists:
            continue

        pending.append(
            {
                "story_id": story_id,
                "story": story,
            }
        )

    logger.info(
        "%s: %d unprocessed stories",
        story_date,
        len(pending),
    )

    for start in range(
        0,
        len(pending),
        BATCH_SIZE,
    ):
        batch = pending[
            start:start + BATCH_SIZE
        ]

        logger.info(
            "Event Memory: stories %d-%d of %d for %s",
            start + 1,
            start + len(batch),
            len(pending),
            story_date,
        )

        story_vectors = embed_texts(
            [
                story_embedding_text(
                    item["story"]
                )
                for item in batch
            ]
        )

        # Refresh between batches so the next ten stories can
        # retrieve events created by previous batches.
        event_index = load_event_index(
            conn
        )

        batch_payload = []
        candidate_map = {}

        for item, vector in zip(
            batch,
            story_vectors,
        ):
            candidates = (
                retrieve_event_candidates(
                    conn,
                    event_index,
                    item["story"],
                    vector,
                )
            )

            candidate_map[
                item["story_id"]
            ] = candidates

            batch_payload.append(
                {
                    "story_id": (
                        item["story_id"]
                    ),
                    "date": story_date,
                    "rank": (
                        item["story"]["rank"]
                    ),
                    "title": (
                        item["story"]["title"]
                    ),
                    "description": (
                        item["story"].get(
                            "description",
                            "",
                        )
                    ),
                    "candidate_historical_events": (
                        candidates
                    ),
                }
            )

        decisions = link_story_batch(
            batch_payload
        )

        decisions_by_id = {
            decision["story_id"]: decision
            for decision in decisions
        }

        expected_ids = {
            item["story_id"]
            for item in batch
        }

        if set(
            decisions_by_id
        ) != expected_ids:
            raise RuntimeError(
                "Event Memory did not return exactly "
                "one decision for every story."
            )

        touched_event_ids = set()

        for (
            item,
            vector,
        ) in zip(
            batch,
            story_vectors,
        ):
            story_id = item[
                "story_id"
            ]

            decision = (
                decisions_by_id[
                    story_id
                ]
            )

            event_id = apply_decision(
                conn=conn,
                story_date=story_date,
                story=item["story"],
                story_id=story_id,
                story_vector=vector,
                candidates=(
                    candidate_map[
                        story_id
                    ]
                ),
                decision=decision,
            )

            touched_event_ids.add(
                event_id
            )

        refresh_event_embeddings(
            conn,
            touched_event_ids,
        )

        conn.commit()


# ============================================================
# BACKFILL
# ============================================================

DATE_FILE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}\.json$"
)


def dated_news_files() -> list[Path]:
    files = [
        path
        for path in DATA_DIR.iterdir()
        if (
            path.is_file()
            and DATE_FILE_RE.match(
                path.name
            )
        )
    ]

    return sorted(files)


def load_json(
    path: Path,
) -> dict:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


# ============================================================
# ENRICHED OUTPUT
# ============================================================

def related_event_info(
    conn: sqlite3.Connection,
    event_id: str | None,
) -> dict | None:

    if not event_id:
        return None

    row = conn.execute(
        """
        SELECT
            event_id,
            canonical_title,
            summary
        FROM events
        WHERE event_id = ?
        """,
        (event_id,),
    ).fetchone()

    if not row:
        return None

    return {
        "event_id": row["event_id"],
        "canonical_title": (
            row["canonical_title"]
        ),
        "summary": row["summary"],
    }


def build_enriched_output(
    conn: sqlite3.Connection,
    latest_payload: dict,
) -> dict:

    story_date = latest_payload[
        "date"
    ]

    enriched_stories = []

    for original_story in latest_payload[
        "stories"
    ]:
        story_id = make_story_id(
            story_date,
            original_story,
        )

        story_row = conn.execute(
            """
            SELECT *
            FROM stories
            WHERE story_id = ?
            """,
            (story_id,),
        ).fetchone()

        if not story_row:
            raise RuntimeError(
                f"Story {story_id} has not "
                "been processed by Event Memory."
            )

        event_row = conn.execute(
            """
            SELECT *
            FROM events
            WHERE event_id = ?
            """,
            (
                story_row[
                    "event_id"
                ],
            ),
        ).fetchone()

        timeline = (
            get_recent_event_timeline(
                conn,
                story_row[
                    "event_id"
                ],
                limit=8,
            )
        )

        enriched_stories.append(
            {
                **original_story,
                "story_id": story_id,
                "event_context": {
                    "event_id": (
                        event_row[
                            "event_id"
                        ]
                    ),
                    "canonical_title": (
                        event_row[
                            "canonical_title"
                        ]
                    ),
                    "first_seen": (
                        event_row[
                            "first_seen"
                        ]
                    ),
                    "last_seen": (
                        event_row[
                            "last_seen"
                        ]
                    ),
                    "rolling_summary": (
                        event_row[
                            "summary"
                        ]
                    ),
                    "update_type": (
                        story_row[
                            "update_type"
                        ]
                    ),
                    "relationship": (
                        story_row[
                            "relationship"
                        ]
                    ),
                    "novelty": (
                        story_row[
                            "novelty"
                        ]
                    ),
                    "historical_note": (
                        story_row[
                            "historical_note"
                        ]
                    ),
                    "related_event": (
                        related_event_info(
                            conn,
                            story_row[
                                "related_event_id"
                            ],
                        )
                    ),
                    "timeline": timeline,
                },
            }
        )

    return {
        **latest_payload,
        "memory_generated_at": (
            utc_now()
        ),
        "stories": enriched_stories,
    }


def write_enriched_latest(
    conn: sqlite3.Connection,
) -> None:

    latest_payload = load_json(
        LATEST_PATH
    )

    enriched = build_enriched_output(
        conn,
        latest_payload,
    )

    date_str = enriched["date"]

    dated_path = (
        DATA_DIR
        / f"{date_str}.enriched.json"
    )

    dated_path.write_text(
        json.dumps(
            enriched,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    shutil.copyfile(
        dated_path,
        ENRICHED_LATEST_PATH,
    )

    logger.info(
        "Wrote %s",
        dated_path,
    )

    logger.info(
        "Wrote %s",
        ENRICHED_LATEST_PATH,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backfill",
        action="store_true",
        help=(
            "Process all historical YYYY-MM-DD.json "
            "files before today's latest.json."
        ),
    )

    args = parser.parse_args()

    conn = connect_db()

    try:
        if args.backfill:
            files = dated_news_files()

            logger.info(
                "Backfilling %d dated news files",
                len(files),
            )

            for path in files:
                logger.info(
                    "Processing historical file: %s",
                    path.name,
                )

                payload = load_json(
                    path
                )

                process_news_payload(
                    conn,
                    payload,
                )

        else:
            payload = load_json(
                LATEST_PATH
            )

            process_news_payload(
                conn,
                payload,
            )

        write_enriched_latest(
            conn
        )

    finally:
        conn.close()

    logger.info(
        "Event Memory complete"
    )


if __name__ == "__main__":
    main()