from __future__ import annotations

import calendar
import html
import json
import logging
import math
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import feedparser
import numpy as np
import requests
from dotenv import load_dotenv
from openai import OpenAI
from requests.adapters import HTTPAdapter
from sklearn.cluster import AgglomerativeClustering
from urllib3.util.retry import Retry


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs"

DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

load_dotenv(ROOT / ".env")

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

LOOKBACK_HOURS = 26
TARGET_STORIES = 100

EMBEDDING_MODEL = "text-embedding-3-small"
SUMMARY_MODEL = "gpt-5.6-luna"

# Cosine distance.
#
# Lower = stricter clustering.
# Higher = more aggressive merging.
#
# Suggested starting range:
#   0.20 = strict
#   0.24 = good default
#   0.30 = aggressive
CLUSTER_DISTANCE_THRESHOLD = 0.24

SUMMARY_BATCH_SIZE = 20

# Prevent one huge feed from completely dominating the input.
MAX_ITEMS_PER_FEED = 150

# Skip articles whose RSS feed provides no timestamp.
# This avoids accidentally including old evergreen content.
ALLOW_UNDATED_ARTICLES = False

REQUEST_TIMEOUT = 30


# ============================================================
# RSS FEEDS
# ============================================================

def google_news_url(topic: str | None = None) -> str:
    """
    Return the Google News US-English RSS URL.

    topic=None gives the main Top Stories feed.
    """

    if topic is None:
        return (
            "https://news.google.com/rss"
            "?hl=en-US&gl=US&ceid=US:en"
        )

    return (
        "https://news.google.com/rss/headlines/section/topic/"
        f"{topic}"
        "?hl=en-US&gl=US&ceid=US:en"
    )


RSS_FEEDS = [
    # --------------------------------------------------------
    # GOOGLE NEWS
    #
    # Google News is especially valuable here because each
    # article carries its original publication as the source.
    # That gives us broad coverage across many publishers.
    # --------------------------------------------------------
    {
        "name": "Google News - Top Stories",
        "url": google_news_url(),
        "source": "Google News",
        "category": "general",
        "region": "global",
        "aggregator": True,
    },
    {
        "name": "Google News - World",
        "url": google_news_url("WORLD"),
        "source": "Google News",
        "category": "world",
        "region": "global",
        "aggregator": True,
    },
    {
        "name": "Google News - US",
        "url": google_news_url("NATION"),
        "source": "Google News",
        "category": "us",
        "region": "us",
        "aggregator": True,
    },
    {
        "name": "Google News - Business",
        "url": google_news_url("BUSINESS"),
        "source": "Google News",
        "category": "business",
        "region": "global",
        "aggregator": True,
    },
    {
        "name": "Google News - Technology",
        "url": google_news_url("TECHNOLOGY"),
        "source": "Google News",
        "category": "technology",
        "region": "global",
        "aggregator": True,
    },
    {
        "name": "Google News - Science",
        "url": google_news_url("SCIENCE"),
        "source": "Google News",
        "category": "science",
        "region": "global",
        "aggregator": True,
    },
    {
        "name": "Google News - Health",
        "url": google_news_url("HEALTH"),
        "source": "Google News",
        "category": "health",
        "region": "global",
        "aggregator": True,
    },
    {
        "name": "Google News - Sports",
        "url": google_news_url("SPORTS"),
        "source": "Google News",
        "category": "sports",
        "region": "global",
        "aggregator": True,
    },
    {
        "name": "Google News - Entertainment",
        "url": google_news_url("ENTERTAINMENT"),
        "source": "Google News",
        "category": "entertainment",
        "region": "global",
        "aggregator": True,
    },

    # --------------------------------------------------------
    # BBC
    # --------------------------------------------------------
    {
        "name": "BBC - Top Stories",
        "url": "https://feeds.bbci.co.uk/news/rss.xml",
        "source": "BBC News",
        "category": "general",
        "region": "gb",
        "aggregator": False,
    },
    {
        "name": "BBC - World",
        "url": "https://feeds.bbci.co.uk/news/world/rss.xml",
        "source": "BBC News",
        "category": "world",
        "region": "global",
        "aggregator": False,
    },
    {
        "name": "BBC - Business",
        "url": "https://feeds.bbci.co.uk/news/business/rss.xml",
        "source": "BBC News",
        "category": "business",
        "region": "global",
        "aggregator": False,
    },
    {
        "name": "BBC - Technology",
        "url": "https://feeds.bbci.co.uk/news/technology/rss.xml",
        "source": "BBC News",
        "category": "technology",
        "region": "global",
        "aggregator": False,
    },
    {
        "name": "BBC - Science",
        "url": (
            "https://feeds.bbci.co.uk/news/"
            "science_and_environment/rss.xml"
        ),
        "source": "BBC News",
        "category": "science",
        "region": "global",
        "aggregator": False,
    },
    {
        "name": "BBC - Health",
        "url": "https://feeds.bbci.co.uk/news/health/rss.xml",
        "source": "BBC News",
        "category": "health",
        "region": "global",
        "aggregator": False,
    },

    # --------------------------------------------------------
    # NPR
    # --------------------------------------------------------
    {
        "name": "NPR - News",
        "url": "https://feeds.npr.org/1001/rss.xml",
        "source": "NPR",
        "category": "general",
        "region": "us",
        "aggregator": False,
    },

    # --------------------------------------------------------
    # NEW YORK TIMES
    # --------------------------------------------------------
    {
        "name": "NYT - Home Page",
        "url": (
            "https://rss.nytimes.com/services/xml/rss/"
            "nyt/HomePage.xml"
        ),
        "source": "The New York Times",
        "category": "general",
        "region": "us",
        "aggregator": False,
    },
    {
        "name": "NYT - World",
        "url": (
            "https://rss.nytimes.com/services/xml/rss/"
            "nyt/World.xml"
        ),
        "source": "The New York Times",
        "category": "world",
        "region": "global",
        "aggregator": False,
    },
    {
        "name": "NYT - US",
        "url": (
            "https://rss.nytimes.com/services/xml/rss/"
            "nyt/US.xml"
        ),
        "source": "The New York Times",
        "category": "us",
        "region": "us",
        "aggregator": False,
    },
    {
        "name": "NYT - Business",
        "url": (
            "https://rss.nytimes.com/services/xml/rss/"
            "nyt/Business.xml"
        ),
        "source": "The New York Times",
        "category": "business",
        "region": "global",
        "aggregator": False,
    },
    {
        "name": "NYT - Technology",
        "url": (
            "https://rss.nytimes.com/services/xml/rss/"
            "nyt/Technology.xml"
        ),
        "source": "The New York Times",
        "category": "technology",
        "region": "global",
        "aggregator": False,
    },
    {
        "name": "NYT - Science",
        "url": (
            "https://rss.nytimes.com/services/xml/rss/"
            "nyt/Science.xml"
        ),
        "source": "The New York Times",
        "category": "science",
        "region": "global",
        "aggregator": False,
    },
    {
        "name": "NYT - Health",
        "url": (
            "https://rss.nytimes.com/services/xml/rss/"
            "nyt/Health.xml"
        ),
        "source": "The New York Times",
        "category": "health",
        "region": "global",
        "aggregator": False,
    },

    # --------------------------------------------------------
    # THE GUARDIAN
    # --------------------------------------------------------
    {
        "name": "Guardian - World",
        "url": "https://www.theguardian.com/world/rss",
        "source": "The Guardian",
        "category": "world",
        "region": "global",
        "aggregator": False,
    },
    {
        "name": "Guardian - US",
        "url": "https://www.theguardian.com/us-news/rss",
        "source": "The Guardian",
        "category": "us",
        "region": "us",
        "aggregator": False,
    },
    {
        "name": "Guardian - Business",
        "url": "https://www.theguardian.com/business/rss",
        "source": "The Guardian",
        "category": "business",
        "region": "global",
        "aggregator": False,
    },
    {
        "name": "Guardian - Technology",
        "url": "https://www.theguardian.com/technology/rss",
        "source": "The Guardian",
        "category": "technology",
        "region": "global",
        "aggregator": False,
    },
    {
        "name": "Guardian - Science",
        "url": "https://www.theguardian.com/science/rss",
        "source": "The Guardian",
        "category": "science",
        "region": "global",
        "aggregator": False,
    },

    # --------------------------------------------------------
    # CNN
    #
    # CNN RSS endpoints have changed over time. These are
    # intentionally non-fatal: if CNN changes one, the rest
    # of the job continues.
    # --------------------------------------------------------
    {
        "name": "CNN - Top Stories",
        "url": "https://rss.cnn.com/rss/cnn_topstories.rss",
        "source": "CNN",
        "category": "general",
        "region": "us",
        "aggregator": False,
    },
    {
        "name": "CNN - World",
        "url": "https://rss.cnn.com/rss/edition_world.rss",
        "source": "CNN",
        "category": "world",
        "region": "global",
        "aggregator": False,
    },

    # --------------------------------------------------------
    # ASSOCIATED PRESS
    # --------------------------------------------------------
    {
        "name": "AP - Top News",
        "url": "https://apnews.com/hub/ap-top-news?format=rss",
        "source": "AP News",
        "category": "general",
        "region": "global",
        "aggregator": False,
    },

    # --------------------------------------------------------
    # AL JAZEERA
    # --------------------------------------------------------
    {
        "name": "Al Jazeera",
        "url": "https://www.aljazeera.com/xml/rss/all.xml",
        "source": "Al Jazeera",
        "category": "world",
        "region": "global",
        "aggregator": False,
    },
]


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# CLIENTS
# ============================================================

openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    max_retries=3,
    timeout=60.0,
)

session = requests.Session()

retry_strategy = Retry(
    total=3,
    connect=3,
    read=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)

adapter = HTTPAdapter(
    max_retries=retry_strategy,
    pool_connections=20,
    pool_maxsize=20,
)

session.mount("https://", adapter)
session.mount("http://", adapter)

session.headers.update(
    {
        "User-Agent": (
            "daily-news-ranking/1.0 "
            "(personal RSS news aggregator)"
        ),
        "Accept": (
            "application/rss+xml, "
            "application/atom+xml, "
            "application/xml, "
            "text/xml, */*"
        ),
    }
)


# ============================================================
# TEXT / URL HELPERS
# ============================================================

def clean_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def strip_html(value: str | None) -> str:
    """
    Turn an RSS HTML description into plain text.
    """

    if not value:
        return ""

    value = re.sub(
        r"<script\b[^>]*>.*?</script>",
        " ",
        value,
        flags=re.IGNORECASE | re.DOTALL,
    )

    value = re.sub(
        r"<style\b[^>]*>.*?</style>",
        " ",
        value,
        flags=re.IGNORECASE | re.DOTALL,
    )

    value = re.sub(
        r"<[^>]+>",
        " ",
        value,
    )

    value = html.unescape(value)

    return clean_whitespace(value)


def clean_url(url: str) -> str:
    """
    Remove fragments and common tracking query strings.

    Google News links are intentionally retained as Google News URLs;
    resolving every redirect is not necessary for clustering.
    """

    if not url:
        return ""

    parts = urlsplit(url)

    hostname = parts.netloc.lower()

    # Google News article URLs often require their query information,
    # so don't strip it aggressively.
    if "news.google.com" in hostname:
        return urlunsplit(
            (
                parts.scheme,
                hostname,
                parts.path.rstrip("/"),
                parts.query,
                "",
            )
        )

    return urlunsplit(
        (
            parts.scheme,
            hostname,
            parts.path.rstrip("/"),
            "",
            "",
        )
    )


def normalized_title(title: str) -> str:
    """
    Normalize a title for within-source duplicate detection.
    """

    title = html.unescape(title)
    title = title.casefold()

    title = re.sub(
        r"[^a-z0-9]+",
        " ",
        title,
    )

    return clean_whitespace(title)


def canonicalize_source(source: str) -> str:
    """
    Avoid counting small naming variations as separate outlets.
    """

    source = clean_whitespace(source)

    aliases = {
        "bbc": "BBC News",
        "bbc news": "BBC News",
        "the guardian": "The Guardian",
        "guardian": "The Guardian",
        "new york times": "The New York Times",
        "the new york times": "The New York Times",
        "nyt": "The New York Times",
        "npr": "NPR",
        "cnn": "CNN",
        "ap": "AP News",
        "ap news": "AP News",
        "associated press": "AP News",
        "al jazeera": "Al Jazeera",
        "al jazeera english": "Al Jazeera",
    }

    return aliases.get(
        source.casefold(),
        source,
    )


def remove_source_suffix(
    title: str,
    source: str,
) -> str:
    """
    Google News commonly gives titles like:

        Something happened - Reuters

    Since Google also exposes Reuters as the source field, remove that
    suffix before embedding.
    """

    title = clean_whitespace(title)

    if not source:
        return title

    for separator in (" - ", " – ", " — "):
        suffix = f"{separator}{source}"

        if title.casefold().endswith(suffix.casefold()):
            title = title[:-len(suffix)].strip()
            break

    return title


# ============================================================
# DATE HELPERS
# ============================================================

def struct_time_to_datetime(
    value,
) -> datetime | None:

    if value is None:
        return None

    try:
        timestamp = calendar.timegm(value)
        return datetime.fromtimestamp(
            timestamp,
            tz=timezone.utc,
        )
    except (TypeError, ValueError, OverflowError):
        return None


def entry_datetime(entry) -> datetime | None:
    """
    feedparser normalizes RSS/Atom timestamps into *_parsed values.
    """

    published = struct_time_to_datetime(
        entry.get("published_parsed")
    )

    if published:
        return published

    updated = struct_time_to_datetime(
        entry.get("updated_parsed")
    )

    if updated:
        return updated

    created = struct_time_to_datetime(
        entry.get("created_parsed")
    )

    return created


# ============================================================
# RSS COLLECTION
# ============================================================

def get_entry_source(
    entry,
    feed_config: dict,
) -> str:

    if not feed_config["aggregator"]:
        return canonicalize_source(
            feed_config["source"]
        )

    # Google News RSS normally provides:
    #
    # <source url="...">Reuters</source>
    #
    source = entry.get("source")

    if isinstance(source, dict):
        source_title = source.get("title")

        if source_title:
            return canonicalize_source(
                strip_html(source_title)
            )

    return canonicalize_source(
        feed_config["source"]
    )


def fetch_feed(
    feed_config: dict,
) -> list[dict]:

    logger.info(
        "Fetching RSS: %s",
        feed_config["name"],
    )

    response = session.get(
        feed_config["url"],
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    parsed = feedparser.parse(
        response.content
    )

    if parsed.bozo:
        logger.warning(
            "RSS parser warning for %s: %s",
            feed_config["name"],
            parsed.bozo_exception,
        )

    articles = []

    entries = parsed.entries[
        :MAX_ITEMS_PER_FEED
    ]

    for entry in entries:
        raw_title = strip_html(
            entry.get("title")
        )

        if not raw_title:
            continue

        source = get_entry_source(
            entry,
            feed_config,
        )

        title = raw_title

        if feed_config["aggregator"]:
            title = remove_source_suffix(
                title,
                source,
            )

        description = strip_html(
            entry.get("summary")
            or entry.get("description")
            or ""
        )

        url = (
            entry.get("link")
            or entry.get("id")
            or ""
        )

        published_dt = entry_datetime(
            entry
        )

        articles.append(
            {
                "title": title,
                "description": description,
                "url": url,
                "clean_url": clean_url(url),
                "source": source,
                "published_at": (
                    published_dt.isoformat()
                    if published_dt
                    else None
                ),
                "_published_dt": published_dt,
                "feed_name": feed_config["name"],
                "feed_url": feed_config["url"],
                "category": feed_config["category"],
                "region": feed_config["region"],
                "aggregator": feed_config["aggregator"],
            }
        )

    logger.info(
        "Fetched %d entries from %s",
        len(articles),
        feed_config["name"],
    )

    return articles


def collect_articles() -> list[dict]:
    """
    Fetch all RSS feeds.

    A failed feed does NOT kill the daily job. It is logged and the
    remaining feeds continue.
    """

    raw_articles = []

    successful_feeds = 0
    failed_feeds = 0

    for feed_config in RSS_FEEDS:
        try:
            articles = fetch_feed(
                feed_config
            )

            raw_articles.extend(
                articles
            )

            successful_feeds += 1

        except Exception:
            failed_feeds += 1

            logger.exception(
                "Failed RSS feed: %s",
                feed_config["name"],
            )

    logger.info(
        "RSS collection complete: "
        "%d successful feeds, "
        "%d failed feeds, "
        "%d raw entries",
        successful_feeds,
        failed_feeds,
        len(raw_articles),
    )

    if successful_feeds == 0:
        raise RuntimeError(
            "Every RSS feed failed."
        )

    cutoff = (
        datetime.now(timezone.utc)
        - timedelta(hours=LOOKBACK_HOURS)
    )

    # --------------------------------------------------------
    # Deduplication
    #
    # IMPORTANT:
    #
    # We don't globally dedupe identical titles because identical
    # coverage from two DIFFERENT outlets is exactly the signal we
    # want for measuring story importance.
    #
    # Instead we remove duplicates from the SAME publisher.
    # --------------------------------------------------------

    by_source_and_title: dict[
        tuple[str, str],
        dict,
    ] = {}

    skipped_old = 0
    skipped_undated = 0

    for article in raw_articles:
        title = article["title"].strip()

        if not title:
            continue

        published_dt = article["_published_dt"]

        if published_dt is None:
            if not ALLOW_UNDATED_ARTICLES:
                skipped_undated += 1
                continue

        elif published_dt < cutoff:
            skipped_old += 1
            continue

        source_key = article[
            "source"
        ].casefold()

        title_key = normalized_title(
            title
        )

        if not title_key:
            continue

        dedupe_key = (
            source_key,
            title_key,
        )

        existing = by_source_and_title.get(
            dedupe_key
        )

        if existing is None:
            article["categories"] = {
                article["category"]
            }

            article["regions"] = {
                article["region"]
            }

            article["feed_names"] = {
                article["feed_name"]
            }

            by_source_and_title[
                dedupe_key
            ] = article

            continue

        # Same publisher/title found in several RSS feeds.
        # Merge metadata rather than treating it as separate coverage.
        existing["categories"].add(
            article["category"]
        )

        existing["regions"].add(
            article["region"]
        )

        existing["feed_names"].add(
            article["feed_name"]
        )

        # Prefer whichever record has a richer description.
        if (
            len(article["description"])
            > len(existing["description"])
        ):
            existing["description"] = (
                article["description"]
            )

        # Prefer a direct publisher link over a Google News link.
        if (
            existing["aggregator"]
            and not article["aggregator"]
        ):
            existing["url"] = article["url"]
            existing["clean_url"] = (
                article["clean_url"]
            )
            existing["aggregator"] = False

        # Keep newest timestamp.
        existing_dt = existing[
            "_published_dt"
        ]

        article_dt = article[
            "_published_dt"
        ]

        if (
            article_dt is not None
            and (
                existing_dt is None
                or article_dt > existing_dt
            )
        ):
            existing["_published_dt"] = (
                article_dt
            )

            existing["published_at"] = (
                article_dt.isoformat()
            )

    articles = list(
        by_source_and_title.values()
    )

    for article in articles:
        article["categories"] = sorted(
            article["categories"]
        )

        article["regions"] = sorted(
            article["regions"]
        )

        article["feed_names"] = sorted(
            article["feed_names"]
        )

        # Internal datetime is no longer needed.
        article.pop(
            "_published_dt",
            None,
        )

        # Remove redundant single-value fields.
        article.pop(
            "category",
            None,
        )

        article.pop(
            "region",
            None,
        )

        article.pop(
            "feed_name",
            None,
        )

    logger.info(
        "%d unique recent publisher articles remain "
        "after filtering/deduplication",
        len(articles),
    )

    logger.info(
        "Skipped %d old entries and %d undated entries",
        skipped_old,
        skipped_undated,
    )

    return articles


# ============================================================
# EMBEDDINGS
# ============================================================

def text_for_embedding(
    article: dict,
) -> str:

    title = article["title"]

    description = (
        article.get("description")
        or ""
    )

    # Enough context to distinguish similar headlines while
    # preventing giant RSS descriptions from dominating.
    description = description[:500]

    return (
        f"Headline: {title}\n"
        f"Description: {description}"
    )


def create_embeddings(
    articles: list[dict],
    batch_size: int = 200,
) -> np.ndarray:

    texts = [
        text_for_embedding(article)
        for article in articles
    ]

    vectors = []

    for start in range(
        0,
        len(texts),
        batch_size,
    ):
        batch = texts[
            start:start + batch_size
        ]

        logger.info(
            "Embedding articles %d-%d of %d",
            start + 1,
            start + len(batch),
            len(texts),
        )

        response = (
            openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch,
            )
        )

        ordered = sorted(
            response.data,
            key=lambda item: item.index,
        )

        vectors.extend(
            item.embedding
            for item in ordered
        )

    return np.asarray(
        vectors,
        dtype=np.float32,
    )


# ============================================================
# CLUSTERING
# ============================================================

def parse_iso_datetime(
    value: str | None,
) -> datetime | None:

    if not value:
        return None

    try:
        return datetime.fromisoformat(
            value.replace(
                "Z",
                "+00:00",
            )
        )

    except ValueError:
        return None


def cluster_articles(
    articles: list[dict],
    vectors: np.ndarray,
) -> list[dict]:

    if len(articles) == 1:
        labels = np.array([0])

    else:
        logger.info(
            "Clustering %d articles",
            len(articles),
        )

        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=(
                CLUSTER_DISTANCE_THRESHOLD
            ),
            metric="cosine",
            linkage="average",
        )

        labels = clusterer.fit_predict(
            vectors
        )

    grouped = defaultdict(list)

    for article, label in zip(
        articles,
        labels,
    ):
        grouped[
            int(label)
        ].append(article)

    clusters = []

    now = datetime.now(
        timezone.utc
    )

    for (
        cluster_id,
        cluster_articles_,
    ) in grouped.items():

        sources = sorted(
            {
                article["source"]
                for article
                in cluster_articles_
            }
        )

        regions = sorted(
            {
                region
                for article
                in cluster_articles_
                for region
                in article["regions"]
            }
        )

        categories = sorted(
            {
                category
                for article
                in cluster_articles_
                for category
                in article["categories"]
            }
        )

        publication_times = [
            parse_iso_datetime(
                article[
                    "published_at"
                ]
            )
            for article
            in cluster_articles_
        ]

        publication_times = [
            dt
            for dt
            in publication_times
            if dt is not None
        ]

        newest = (
            max(publication_times)
            if publication_times
            else now
        )

        age_hours = max(
            0,
            (
                now - newest
            ).total_seconds()
            / 3600,
        )

        # ----------------------------------------------------
        # IMPORTANCE SCORE
        #
        # The biggest signal is independent publisher coverage.
        #
        # A story appearing at Reuters/AP/BBC/CNN/etc. should beat
        # something appearing twenty times from one publisher.
        #
        # Freshness matters, but not so much that a trivial story
        # from 10 minutes ago automatically beats major news from
        # 8 hours ago.
        # ----------------------------------------------------

        source_score = (
            7.0
            * math.log1p(
                len(sources)
            )
        )

        article_score = (
            1.0
            * math.log1p(
                len(
                    cluster_articles_
                )
            )
        )

        category_score = (
            1.0
            * math.log1p(
                len(categories)
            )
        )

        region_score = (
            0.75
            * math.log1p(
                len(regions)
            )
        )

        freshness_score = (
            3.0
            * math.exp(
                -age_hours / 18.0
            )
        )

        score = (
            source_score
            + article_score
            + category_score
            + region_score
            + freshness_score
        )

        clusters.append(
            {
                "cluster_id": (
                    cluster_id
                ),
                "score": round(
                    score,
                    4,
                ),
                "source_count": len(
                    sources
                ),
                "article_count": len(
                    cluster_articles_
                ),
                "sources": sources,
                "regions": regions,
                "categories": (
                    categories
                ),
                "newest_published_at": (
                    newest.isoformat()
                ),
                "articles": (
                    cluster_articles_
                ),
            }
        )

    clusters.sort(
        key=lambda cluster: (
            cluster["score"]
        ),
        reverse=True,
    )

    logger.info(
        "%d distinct story clusters found",
        len(clusters),
    )

    return clusters


# ============================================================
# GPT SUMMARIZATION
# ============================================================

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "stories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cluster_id": {
                        "type": "integer",
                    },
                    "title": {
                        "type": "string",
                    },
                    "description": {
                        "type": "string",
                    },
                },
                "required": [
                    "cluster_id",
                    "title",
                    "description",
                ],
                "additionalProperties": (
                    False
                ),
            },
        }
    },
    "required": [
        "stories"
    ],
    "additionalProperties": False,
}


def cluster_for_prompt(
    cluster: dict,
) -> dict:

    # Prefer articles with descriptions and recent timestamps.
    articles = sorted(
        cluster["articles"],
        key=lambda x: (
            bool(
                x.get(
                    "description"
                )
            ),
            x.get(
                "published_at"
            )
            or "",
        ),
        reverse=True,
    )

    evidence = []

    # Eight different articles is generally more than enough
    # evidence to summarize one event.
    for article in articles[:8]:
        evidence.append(
            {
                "source": (
                    article["source"]
                ),
                "headline": (
                    article["title"]
                ),
                "description": (
                    article[
                        "description"
                    ][:500]
                ),
                "published_at": (
                    article[
                        "published_at"
                    ]
                ),
            }
        )

    return {
        "cluster_id": (
            cluster[
                "cluster_id"
            ]
        ),
        "source_count": (
            cluster[
                "source_count"
            ]
        ),
        "sources": (
            cluster["sources"]
        ),
        "articles": evidence,
    }


def summarize_batch(
    clusters: list[dict],
) -> list[dict]:

    payload = {
        "stories": [
            cluster_for_prompt(
                cluster
            )
            for cluster
            in clusters
        ]
    }

    instructions = """
You are an editor producing a daily list of major news stories.

Each input cluster represents RSS headlines that an automated
clustering system believes concern the same underlying news event.

For every cluster:

1. Produce one neutral canonical headline.
2. Produce one concise one-sentence description explaining what
   happened and why the development is notable.
3. Base the output ONLY on facts supported by the supplied headlines
   and descriptions.
4. Do not invent facts, numbers, quotations, names, motives, causes,
   consequences, dates, or background.
5. When evidence is incomplete, describe only what is clearly supported.
6. When sources disagree, use cautious wording and do not resolve the
   disagreement yourself.
7. Do not use sensational or clickbait wording.
8. Do not mention the RSS feed, cluster, ranking system, or source count
   in the finished headline or description.
9. Headline should preferably contain 8-18 words.
10. Description should preferably contain 20-45 words.
11. Preserve cluster_id exactly.
12. Return every supplied cluster exactly once.
"""

    response = (
        openai_client.responses.create(
            model=SUMMARY_MODEL,
            instructions=instructions,
            input=json.dumps(
                payload,
                ensure_ascii=False,
            ),
            reasoning={
                "effort": "low",
            },
            text={
                "format": {
                    "type": (
                        "json_schema"
                    ),
                    "name": (
                        "daily_news_summaries"
                    ),
                    "strict": True,
                    "schema": (
                        SUMMARY_SCHEMA
                    ),
                }
            },
            max_output_tokens=8000,
            store=False,
        )
    )

    if not response.output_text:
        raise RuntimeError(
            "OpenAI returned no output text."
        )

    result = json.loads(
        response.output_text
    )

    return result[
        "stories"
    ]


def summarize_clusters(
    clusters: list[dict],
) -> dict[int, dict]:

    results = {}

    for start in range(
        0,
        len(clusters),
        SUMMARY_BATCH_SIZE,
    ):
        batch = clusters[
            start:
            start + SUMMARY_BATCH_SIZE
        ]

        logger.info(
            "Summarizing stories %d-%d of %d",
            start + 1,
            start + len(batch),
            len(clusters),
        )

        summaries = (
            summarize_batch(
                batch
            )
        )

        for summary in summaries:
            results[
                summary[
                    "cluster_id"
                ]
            ] = summary

    return results


# ============================================================
# OUTPUT
# ============================================================

def write_outputs(
    top_clusters: list[dict],
    summaries: dict[int, dict],
) -> None:

    generated_at = datetime.now(
        timezone.utc
    )

    stories = []
    debug_stories = []

    for rank, cluster in enumerate(
        top_clusters,
        start=1,
    ):
        summary = summaries.get(
            cluster[
                "cluster_id"
            ]
        )

        if summary is None:
            logger.warning(
                "Missing summary for cluster %s",
                cluster[
                    "cluster_id"
                ],
            )

            continue

        public_story = {
            "rank": rank,
            "title": (
                summary["title"]
            ),
            "description": (
                summary[
                    "description"
                ]
            ),
        }

        stories.append(
            public_story
        )

        debug_stories.append(
            {
                **public_story,
                "score": (
                    cluster["score"]
                ),
                "source_count": (
                    cluster[
                        "source_count"
                    ]
                ),
                "article_count": (
                    cluster[
                        "article_count"
                    ]
                ),
                "sources": (
                    cluster[
                        "sources"
                    ]
                ),
                "regions": (
                    cluster[
                        "regions"
                    ]
                ),
                "categories": (
                    cluster[
                        "categories"
                    ]
                ),
                "articles": [
                    {
                        "title": (
                            article[
                                "title"
                            ]
                        ),
                        "source": (
                            article[
                                "source"
                            ]
                        ),
                        "url": (
                            article[
                                "url"
                            ]
                        ),
                        "published_at": (
                            article[
                                "published_at"
                            ]
                        ),
                        "feed_names": (
                            article[
                                "feed_names"
                            ]
                        ),
                    }
                    for article
                    in cluster[
                        "articles"
                    ]
                ],
            }
        )

    public_output = {
        "date": (
            generated_at
            .date()
            .isoformat()
        ),
        "generated_at": (
            generated_at.isoformat()
        ),
        "window_hours": (
            LOOKBACK_HOURS
        ),
        "story_count": len(
            stories
        ),
        "stories": stories,
    }

    debug_output = {
        **public_output,
        "stories": debug_stories,
    }

    date_str = (
        generated_at
        .date()
        .isoformat()
    )

    dated_path = (
        DATA_DIR
        / f"{date_str}.json"
    )

    latest_path = (
        DATA_DIR
        / "latest.json"
    )

    debug_path = (
        DATA_DIR
        / f"{date_str}.debug.json"
    )

    dated_path.write_text(
        json.dumps(
            public_output,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    debug_path.write_text(
        json.dumps(
            debug_output,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    shutil.copyfile(
        dated_path,
        latest_path,
    )

    logger.info(
        "Wrote %s",
        dated_path,
    )

    logger.info(
        "Wrote %s",
        latest_path,
    )

    logger.info(
        "Wrote %s",
        debug_path,
    )


# ============================================================
# DIAGNOSTICS
# ============================================================

def print_top_clusters(
    clusters: list[dict],
    limit: int = 20,
) -> None:

    logger.info(
        "Top %d clusters before GPT summarization:",
        min(
            limit,
            len(clusters),
        ),
    )

    for rank, cluster in enumerate(
        clusters[:limit],
        start=1,
    ):
        representative = max(
            cluster["articles"],
            key=lambda article: (
                len(
                    article[
                        "description"
                    ]
                )
            ),
        )

        logger.info(
            "#%d | score=%.2f | sources=%d | articles=%d | %s",
            rank,
            cluster["score"],
            cluster[
                "source_count"
            ],
            cluster[
                "article_count"
            ],
            representative[
                "title"
            ],
        )


# ============================================================
# MAIN
# ============================================================

def main():
    logger.info(
        "Starting daily RSS news update"
    )

    logger.info(
        "Looking back %d hours",
        LOOKBACK_HOURS,
    )

    logger.info(
        "Configured RSS feeds: %d",
        len(RSS_FEEDS),
    )

    # --------------------------------------------------------
    # 1. COLLECT RSS
    # --------------------------------------------------------

    articles = collect_articles()

    if (
        len(articles)
        < TARGET_STORIES
    ):
        raise RuntimeError(
            f"Only {len(articles)} recent "
            "candidate articles found; "
            f"need at least {TARGET_STORIES}."
        )

    # --------------------------------------------------------
    # 2. EMBED
    # --------------------------------------------------------

    vectors = create_embeddings(
        articles
    )

    # --------------------------------------------------------
    # 3. CLUSTER
    # --------------------------------------------------------

    clusters = cluster_articles(
        articles,
        vectors,
    )

    print_top_clusters(
        clusters
    )

    if (
        len(clusters)
        < TARGET_STORIES
    ):
        raise RuntimeError(
            f"Only {len(clusters)} distinct "
            "story clusters found. "
            "Try lowering "
            "CLUSTER_DISTANCE_THRESHOLD "
            "or increasing LOOKBACK_HOURS."
        )

    # --------------------------------------------------------
    # 4. SELECT TOP 100
    # --------------------------------------------------------

    top_clusters = clusters[
        :TARGET_STORIES
    ]

    # --------------------------------------------------------
    # 5. GPT SUMMARIES
    # --------------------------------------------------------

    summaries = summarize_clusters(
        top_clusters
    )

    # --------------------------------------------------------
    # 6. WRITE JSON
    # --------------------------------------------------------

    write_outputs(
        top_clusters,
        summaries,
    )

    logger.info(
        "Daily RSS news update complete"
    )


if __name__ == "__main__":
    main()