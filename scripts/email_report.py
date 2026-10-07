from __future__ import annotations

import html
import json
import logging
import os
import smtplib
import hashlib
from . import trade_store
from .trade_report import summary as trading_summary
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from zoneinfo import ZoneInfo


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

load_dotenv(
    ROOT / ".env"
)

NEWS_INPUT_PATH = (
    DATA_DIR
    / "enriched_latest.json"
)

TRENDS_INPUT_PATH = Path(
    os.getenv(
        "MORNING_BRIEF_PATH",
        (
            "/Users/diegotanton/Desktop/"
            "Youtube/data/morning_brief.json"
        ),
    )
)

OUTPUT_PATH = (
    DATA_DIR
    / "email_report_latest.json"
)

REPORT_TIMEZONE = ZoneInfo(
    os.getenv(
        "REPORT_TIMEZONE",
        "America/Chicago",
    )
)


# ------------------------------------------------------------
# GMAIL
# ------------------------------------------------------------

GMAIL_ADDRESS = os.environ[
    "GMAIL_ADDRESS"
]

GMAIL_APP_PASSWORD = os.environ[
    "GMAIL_APP_PASSWORD"
]

EMAIL_TO = os.getenv(
    "EMAIL_TO",
    GMAIL_ADDRESS,
)

EMAIL_REPORT_LIMIT = int(
    os.getenv(
        "EMAIL_REPORT_LIMIT",
        "50",
    )
)


# ------------------------------------------------------------
# OPENAI
# ------------------------------------------------------------

TRENDS_MODEL = os.getenv(
    "TRENDS_MODEL",
    "gpt-5.6-luna",
)

TRENDS_REASONING = os.getenv(
    "TRENDS_REASONING",
    "low",
)

TRENDS_PROMPT = """
Write one concise paragraph explaining the most important online trends
today across the supplied YouTube and X data.

Prioritize the 3-5 most meaningful themes rather than mechanically
listing individual trends.

The reader should NOT be assumed to know internet slang, hashtags,
sports shorthand, entertainment events, usernames, fandom terminology,
or unexplained proper nouns.

Whenever you mention an unfamiliar or ambiguous term, briefly explain
what it is in natural language.

For example, do not simply write:
"#AEWAllIn is trending."

Instead write something like:
"#AEWAllIn, referring to All Elite Wrestling's All In event, is trending."

Likewise, never include cryptic phrases such as "4 QBs" unless you can
determine what they specifically refer to and explain them.

Use web search when necessary to identify or contextualize an unclear
trend. If you still cannot confidently determine what a trend means,
omit it rather than repeating an unexplained label.

Please do not include citations in your output, because they will show
up to the user as a long URL string.

Use specific examples and noteworthy metrics when they genuinely help
explain the trend.

Do not speculate about why something is trending unless that explanation
is supported by the supplied data or web research.

Write roughly 140-200 words.

Return plain text only:
- no Markdown
- no headings
- no bullets
- no bold or italic markers
""".strip()


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def report_date(
    payload: dict,
) -> str:

    generated_at = payload.get(
        "generated_at"
    )

    if generated_at:
        dt = datetime.fromisoformat(
            generated_at.replace(
                "Z",
                "+00:00",
            )
        )
    else:
        dt = datetime.now(
            timezone.utc
        )

    return (
        dt.astimezone(
            REPORT_TIMEZONE
        )
        .date()
        .isoformat()
    )


def load_json(
    path: Path,
) -> dict:

    if not path.exists():
        raise RuntimeError(
            f"Expected input does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


# ============================================================
# TRENDS
# ============================================================

def generate_trends_digest(
    trends_payload: dict,
) -> str:

    logger.info(
        "Generating Trends digest with %s",
        TRENDS_MODEL,
    )

    openai_client = OpenAI(max_retries=2, timeout=120.0)
    response = (
        openai_client.responses.create(
            model=TRENDS_MODEL,

            reasoning={
                "effort": (
                    TRENDS_REASONING
                ),
            },

            instructions=(
                TRENDS_PROMPT
            ),

            input=json.dumps(
                trends_payload,
                ensure_ascii=False,
            ),

            tools=[
                {
                    "type": "web_search",
                }
            ],

            tool_choice="auto",

            # Explicitly use a stable cache key for this
            # particular prompt/application.
            prompt_cache_key=(
                "daily-email-trends-v1"
            ),

            # Keep eligible cached prompt prefixes around
            # for up to 24 hours.
            prompt_cache_retention="24h",

            max_output_tokens=1200,

            store=False,
        )
    )

    digest = clean_trends_digest(
        response.output_text
        or ""
    )

    if not digest:
        raise RuntimeError(
            "Trends model returned no output."
        )

    logger.info(
        "Trends digest generated"
    )

    return digest


def clean_trends_digest(
    value: str,
) -> str:

    value = value.strip()

    # Defensive cleanup if the model emits Markdown despite
    # being explicitly asked for plain text.
    value = value.replace(
        "**",
        "",
    )

    value = value.replace(
        "__",
        "",
    )

    value = value.replace(
        "*",
        "",
    )

    return " ".join(
        value.split()
    )


# ============================================================
# PLAIN-TEXT REPORT
# ============================================================

def build_plain_text(
    news_payload: dict,
    stories: list[dict],
    trends_digest: str,
) -> str:

    date_str = report_date(
        news_payload
    )

    lines = [
        news_payload.get("trade_summary", "Trade status unconfirmed."),
        "",
        f"DAILY BRIEF — {date_str}",
        news_payload.get("news_notice", ""),
        "",
        "NEWS",
        "====",
        "",
    ]

    for story in stories:
        lines.extend(
            [
                (
                    f"{story['rank']}. "
                    f"{story['title']}"
                ),
                story.get(
                    "description",
                    "",
                ),
                "",
            ]
        )

    lines.extend(
        [
            "",
            "TRENDS",
            "======",
            "",
            trends_digest,
            "",
        ]
    )

    return "\n".join(
        lines
    )


# ============================================================
# HTML REPORT
# ============================================================

def build_html(
    news_payload: dict,
    stories: list[dict],
    trends_digest: str,
) -> str:

    story_blocks = []

    for story in stories:
        rank = html.escape(
            str(
                story["rank"]
            )
        )

        title = html.escape(
            story["title"]
        )

        description = html.escape(
            story.get(
                "description",
                "",
            )
        )

        story_blocks.append(
            f"""
            <div style="
                margin: 0 0 26px 0;
                padding: 0 0 22px 0;
                border-bottom: 1px solid #e5e7eb;
            ">
                <div style="
                    font-size: 12px;
                    font-weight: 700;
                    color: #9ca3af;
                    margin-bottom: 5px;
                ">
                    #{rank}
                </div>

                <div style="
                    font-size: 18px;
                    line-height: 1.4;
                    font-weight: 700;
                    color: #111827;
                    margin-bottom: 7px;
                ">
                    {title}
                </div>

                <div style="
                    font-size: 15px;
                    line-height: 1.6;
                    color: #374151;
                ">
                    {description}
                </div>
            </div>
            """
        )

    date_str = html.escape(
        report_date(
            news_payload
        )
    )
    escaped_trends = html.escape(
        trends_digest
    )

    return f"""
    <!doctype html>
    <html>
    <body style="
        margin: 0;
        padding: 0;
        background: #f3f4f6;
        font-family: Arial, Helvetica, sans-serif;
    ">
        <div style="
            max-width: 760px;
            margin: 0 auto;
            padding: 32px 18px;
        ">
            <div style="
                background: #ffffff;
                border-radius: 12px;
                padding: 34px;
            ">

                <!-- HEADER -->

                <div style="
                    font-size: 12px;
                    font-weight: 700;
                    letter-spacing: 1.5px;
                    color: #6b7280;
                    margin-bottom: 8px;
                ">
                    DAILY BRIEF
                </div>

                <h1 style="
                    margin: 0 0 6px 0;
                    font-size: 30px;
                    color: #111827;
                ">
                    News & Trends
                </h1>

                <div style="
                    color: #6b7280;
                    font-size: 14px;
                    margin-bottom: 42px;
                ">
                    {date_str}
                </div>


                <div style="padding:18px;background:#eff6ff;margin-bottom:24px;white-space:pre-wrap;">{html.escape(news_payload.get('trade_summary', 'Trade status unconfirmed.'))}</div>
                <p>{html.escape(news_payload.get('news_notice', ''))}</p>
                <!-- NEWS -->

                <div style="
                    font-size: 13px;
                    font-weight: 700;
                    letter-spacing: 1.5px;
                    color: #6b7280;
                    margin-bottom: 8px;
                ">
                    NEWS
                </div>

                <h2 style="
                    margin: 0 0 28px 0;
                    font-size: 24px;
                    color: #111827;
                ">
                    Top {len(stories)} Stories
                </h2>

                {''.join(story_blocks)}


                <!-- TRENDS -->

                <div style="
                    margin-top: 46px;
                    padding-top: 32px;
                    border-top: 3px solid #111827;
                ">

                    <div style="
                        font-size: 13px;
                        font-weight: 700;
                        letter-spacing: 1.5px;
                        color: #6b7280;
                        margin-bottom: 8px;
                    ">
                        TRENDS
                    </div>

                    <h2 style="
                        margin: 0 0 18px 0;
                        font-size: 24px;
                        color: #111827;
                    ">
                        What's Trending Today
                    </h2>

                    <div style="
                        font-size: 16px;
                        line-height: 1.7;
                        color: #374151;
                    ">
                        {escaped_trends}
                    </div>

                </div>


                <!-- FOOTER -->

                <div style="
                    margin-top: 42px;
                    padding-top: 20px;
                    border-top: 1px solid #e5e7eb;
                    color: #9ca3af;
                    font-size: 12px;
                ">
                    Generated automatically by the News pipeline.
                </div>

            </div>
        </div>
    </body>
    </html>
    """


# ============================================================
# EMAIL
# ============================================================

def send_report(
    news_payload: dict,
    trends_digest: str,
) -> int:

    date_str = report_date(
        news_payload
    )

    stories = news_payload.get(
        "stories",
        [],
    )[
        :EMAIL_REPORT_LIMIT
    ]


    subject = f"Daily Brief — {date_str} — {news_payload['trade_summary'].splitlines()[0]}"

    message = EmailMessage()

    message[
        "Subject"
    ] = subject

    message[
        "From"
    ] = GMAIL_ADDRESS

    message[
        "To"
    ] = EMAIL_TO

    message.set_content(
        build_plain_text(
            news_payload,
            stories,
            trends_digest,
        )
    )

    message.add_alternative(
        build_html(
            news_payload,
            stories,
            trends_digest,
        ),
        subtype="html",
    )

    logger.info(
        "Sending daily brief to %s",
        EMAIL_TO,
    )

    report_key = news_payload['report_key']
    message['Message-ID'] = '<' + hashlib.sha256(report_key.encode()).hexdigest() + '@news-paper-report>'
    outbox = DATA_DIR / 'reports'
    outbox.mkdir(exist_ok=True)
    (outbox / (hashlib.sha256(report_key.encode()).hexdigest() + '.eml')).write_bytes(message.as_bytes())
    if not trade_store.claim_report(report_key):
        raise RuntimeError('This report was sent or has an ambiguous prior send; inspect reports before retrying')
    send_started = False
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com',465,timeout=30) as smtp:
            smtp.login(GMAIL_ADDRESS,GMAIL_APP_PASSWORD)
            send_started = True
            smtp.send_message(message)
        trade_store.report_status(report_key,'SENT')
    except Exception as exc:
        trade_store.report_status(report_key,'UNKNOWN' if send_started else 'FAILED',str(exc))
        raise

    logger.info(
        "Email sent successfully"
    )

    return len(
        stories
    )


# ============================================================
# OUTPUT
# ============================================================

def write_output(
    news_payload: dict,
    trends_payload: dict,
    trends_digest: str,
    story_count: int,
) -> None:

    result = {
        "run_id": os.getenv("NEWS_RUN_ID"),
        "trade_summary": news_payload.get("trade_summary"),
        "date": report_date(
            news_payload
        ),
        "source_date_utc": news_payload[
            "date"
        ],
        "generated_at": utc_now(),
        "status": "SENT",
        "recipient": EMAIL_TO,
        "story_count": story_count,
        "trends_source": str(
            TRENDS_INPUT_PATH
        ),
        "trends_source_generated_at": (
            trends_payload.get(
                "generated_at_utc"
            )
        ),
        "trends_model": TRENDS_MODEL,
        "trends_digest": (
            trends_digest
        ),
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main():
    now = datetime.now(timezone.utc)
    run_id = os.getenv('NEWS_RUN_ID')
    with trade_store.connect() as db:
        summary = trading_summary(db, run_id, now, str(REPORT_TIMEZONE))
    try:
        news = load_json(NEWS_INPUT_PATH)
        original_generated = news.get('generated_at')
        source_dt = datetime.fromisoformat(original_generated.replace('Z','+00:00'))
        if source_dt.tzinfo is None or not -60 <= (now-source_dt).total_seconds() <= 36*3600:
            raise ValueError('News source is stale')
        news['news_notice'] = 'News source generated at ' + original_generated
    except Exception as exc:
        news = {'date':now.date().isoformat(),'stories':[], 'news_notice':'News unavailable: '+str(exc)}
    # Report date is delivery date, not a reused news file's date.
    news['generated_at'] = now.isoformat()
    news['trade_summary'] = summary
    news['report_key'] = run_id or ('standalone-' + now.astimezone(REPORT_TIMEZONE).date().isoformat())
    trends = {}
    try:
        trends = load_json(TRENDS_INPUT_PATH)
        source = datetime.fromisoformat(trends['generated_at_utc'].replace('Z','+00:00'))
        if source.tzinfo is None or not -60 <= (now-source).total_seconds() <= 36*3600:
            raise ValueError('Trends source is stale')
        digest = generate_trends_digest(trends)
    except Exception as exc:
        logger.warning('Trends unavailable: %s',exc)
        digest = 'Trends unavailable for this report.'
    # Persist the factual summary before any SMTP side effect.
    trade_store.atomic_json(DATA_DIR / 'trade_summary_latest.json',
                            {'generated_at':now.isoformat(),'run_id':run_id,'summary':summary})
    count = send_report(news,digest)
    write_output(news,trends,digest,count)


if __name__ == "__main__":
    main()