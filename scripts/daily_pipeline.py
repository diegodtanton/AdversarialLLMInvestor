from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
from alpaca.trading.client import TradingClient
from dotenv import load_dotenv


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

load_dotenv(
    ROOT / ".env"
)

ALPACA_API_KEY = os.environ[
    "ALPACA_API_KEY"
]

ALPACA_SECRET_KEY = os.environ[
    "ALPACA_SECRET_KEY"
]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# These stages maintain the daily news/event history even when
# financial markets are closed.
ALWAYS_RUN_STAGES = [
    "scripts.news_collect",
    "scripts.event_memory",
    "scripts.email_report",
]


# These stages only run while the regular US equity market is open.
MARKET_OPEN_STAGES = [
    "scripts.position_manager",

    "scripts.gilman",
    "scripts.market_context",
    "scripts.brody",

    "scripts.alpaca_execute",
]


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the daily news → Gilman → "
            "Brody → Alpaca pipeline."
        ),
    )

    parser.add_argument(
        "--no_news",
        action="store_true",
        help=(
            "Skip news collection, Event Memory, and the "
            "email report. Reuse the existing "
            "data/enriched_latest.json and run only the "
            "market/trading portion of the pipeline."
        ),
    )

    return parser.parse_args()


def validate_no_news_input() -> None:
    """
    Ensure a previous enriched news dataset exists before running
    Gilman without refreshing news/Event Memory.
    """

    path = (
        DATA_DIR
        / "enriched_latest.json"
    )

    data = load_json(
        path
    )

    stories = data.get(
        "stories",
        [],
    )

    if not stories:
        raise RuntimeError(
            "Cannot run --no_news because "
            "enriched_latest.json contains no stories."
        )

    missing_context = [
        story.get(
            "story_id",
            story.get("rank"),
        )
        for story in stories
        if not story.get(
            "event_context"
        )
    ]

    if missing_context:
        raise RuntimeError(
            "Cannot run --no_news because "
            "enriched_latest.json contains stories "
            "without Event Memory context."
        )

    logger.info(
        "✓ Reusing existing enriched news: %d stories",
        len(stories),
    )

# ------------------------------------------------------------
# CONNECTIVITY / RETRY
# ------------------------------------------------------------

MAX_CONNECTIVITY_ATTEMPTS = 5

# 15 minutes.
CONNECTIVITY_RETRY_SECONDS = 15 * 60

# If a normal stage starts but dies from a transient API/network
# problem, give that stage a few chances.
MAX_STAGE_ATTEMPTS = 3
STAGE_RETRY_SECONDS = 15 * 60

CONNECTIVITY_URL = (
    "https://www.google.com/generate_204"
)

CONNECTIVITY_TIMEOUT_SECONDS = 10


# This stage is stateful: it can submit brokerage orders.
#
# Do NOT generically rerun it just because the process returned
# an error. alpaca_execute.py should handle order reconciliation
# and idempotency internally.
NON_RETRYABLE_STAGES = {
    "scripts.position_manager",
    "scripts.alpaca_execute",
}


# ============================================================
# ALPACA
# ============================================================

# We only use this client here to ask Alpaca whether the regular
# US equity market is currently open.
#
# Execution remains isolated inside alpaca_execute.py.
trading_client = TradingClient(
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    paper=True,
)


# ============================================================
# CONNECTIVITY
# ============================================================

def has_internet() -> bool:
    """
    Test actual HTTPS connectivity.

    Google's generate_204 endpoint normally returns exactly 204.

    A coffee-shop captive portal will often redirect this request
    instead, which we deliberately treat as "not online yet".
    """

    try:
        response = requests.get(
            CONNECTIVITY_URL,
            timeout=CONNECTIVITY_TIMEOUT_SECONDS,
            allow_redirects=False,
        )

        return (
            response.status_code
            == 204
        )

    except requests.RequestException:
        return False


def wait_for_internet() -> None:
    for attempt in range(
        1,
        MAX_CONNECTIVITY_ATTEMPTS + 1,
    ):
        if has_internet():
            logger.info(
                "✓ Internet connection is available"
            )

            return

        logger.warning(
            "No usable internet connection yet "
            "(attempt %d/%d)",
            attempt,
            MAX_CONNECTIVITY_ATTEMPTS,
        )

        if (
            attempt
            >= MAX_CONNECTIVITY_ATTEMPTS
        ):
            raise RuntimeError(
                "No usable internet connection "
                "after all connectivity attempts."
            )

        logger.info(
            "Checking again in %d minutes...",
            CONNECTIVITY_RETRY_SECONDS // 60,
        )

        time.sleep(
            CONNECTIVITY_RETRY_SECONDS
        )


# ============================================================
# MARKET CLOCK
# ============================================================

def is_market_open() -> bool:
    """
    Ask Alpaca whether the regular US equity market is open now.

    This naturally handles:
    - weekends
    - exchange holidays
    - normal trading days
    - early-close sessions

    We deliberately use the brokerage's own market state rather
    than maintaining a separate holiday calendar.
    """

    clock = (
        trading_client
        .get_clock()
    )

    if clock.is_open:
        logger.info(
            "✓ US equity market is open | "
            "next close: %s",
            clock.next_close,
        )

        return True

    logger.info(
        "US equity market is closed | "
        "next open: %s",
        clock.next_open,
    )

    return False


# ============================================================
# FILE HELPERS
# ============================================================

def load_json(
    path: Path,
) -> dict:

    if not path.exists():
        raise RuntimeError(
            f"Expected output does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


# ============================================================
# STAGE EXECUTION
# ============================================================

def run_stage_once(
    module: str,
) -> None:

    logger.info(
        "============================================================"
    )

    logger.info(
        "Starting stage: %s",
        module,
    )

    subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            module,
        ],
        cwd=ROOT,
        check=True,
    )

    logger.info(
        "Completed stage: %s",
        module,
    )


def run_stage(
    module: str,
) -> None:
    """
    Run one stage.

    Normal data/LLM stages can retry after transient failures.

    alpaca_execute is deliberately run only once because a failed
    brokerage request may have succeeded remotely even if our local
    process did not receive the response.
    """

    if module in NON_RETRYABLE_STAGES:
        wait_for_internet()

        run_stage_once(
            module
        )

        return

    for attempt in range(
        1,
        MAX_STAGE_ATTEMPTS + 1,
    ):
        try:
            wait_for_internet()

            logger.info(
                "%s attempt %d/%d",
                module,
                attempt,
                MAX_STAGE_ATTEMPTS,
            )

            run_stage_once(
                module
            )

            return

        except subprocess.CalledProcessError as exc:
            logger.error(
                "%s failed on attempt %d/%d "
                "with exit code %s",
                module,
                attempt,
                MAX_STAGE_ATTEMPTS,
                exc.returncode,
            )

            if (
                attempt
                >= MAX_STAGE_ATTEMPTS
            ):
                raise

            logger.info(
                "Retrying %s in %d minutes...",
                module,
                STAGE_RETRY_SECONDS // 60,
            )

            time.sleep(
                STAGE_RETRY_SECONDS
            )


# ============================================================
# VALIDATION
# ============================================================

def validate_news_collect() -> None:
    data = load_json(
        DATA_DIR / "latest.json"
    )

    stories = data.get(
        "stories",
        [],
    )

    if len(stories) != 100:
        raise RuntimeError(
            f"news_collect produced "
            f"{len(stories)} stories; expected 100."
        )

    logger.info(
        "✓ NEWS COLLECT: %d stories",
        len(stories),
    )


def validate_event_memory() -> None:
    data = load_json(
        DATA_DIR / "enriched_latest.json"
    )

    stories = data.get(
        "stories",
        [],
    )

    if len(stories) != 100:
        raise RuntimeError(
            f"event_memory produced "
            f"{len(stories)} enriched stories; "
            f"expected 100."
        )

    missing = [
        story.get(
            "story_id",
            story.get("rank"),
        )
        for story in stories
        if not story.get(
            "event_context"
        )
    ]

    if missing:
        raise RuntimeError(
            f"event_memory is missing context "
            f"for {len(missing)} stories."
        )

    logger.info(
        "✓ EVENT MEMORY: %d enriched stories",
        len(stories),
    )

def validate_email_report() -> None:
    data = load_json(
        DATA_DIR
        / "email_report_latest.json"
    )

    if (
        data.get("status")
        != "SENT"
    ):
        raise RuntimeError(
            "Email report was not successfully sent."
        )

    story_count = data.get(
        "story_count"
    )

    if (
        not isinstance(
            story_count,
            int,
        )
        or story_count != 25
    ):
        raise RuntimeError(
            f"Email report contained "
            f"{story_count} stories; expected 25."
        )

    logger.info(
        "✓ EMAIL REPORT: %d stories sent",
        story_count,
    )


def validate_position_manager() -> None:
    data = load_json(
        DATA_DIR
        / "position_manager_latest.json"
    )

    managed = data.get(
        "managed_count"
    )

    closed = data.get(
        "closed_count"
    )

    holding = data.get(
        "holding_count"
    )

    pending = data.get(
        "pending_exit_count"
    )

    for name, value in {
        "managed_count": managed,
        "closed_count": closed,
        "holding_count": holding,
        "pending_exit_count": pending,
    }.items():
        if not isinstance(
            value,
            int,
        ):
            raise RuntimeError(
                f"position_manager has invalid {name}."
            )

    logger.info(
        "✓ POSITION MANAGER: "
        "%d managed | "
        "%d closed | "
        "%d holding | "
        "%d pending exit",
        managed,
        closed,
        holding,
        pending,
    )


def validate_gilman() -> None:
    data = load_json(
        DATA_DIR / "gilman_latest.json"
    )

    story_count = data.get(
        "story_count"
    )

    trade_count = data.get(
        "trade_count"
    )

    no_trade_count = data.get(
        "no_trade_count"
    )

    if story_count != 100:
        raise RuntimeError(
            f"Gilman analyzed "
            f"{story_count} stories; expected 100."
        )

    if not isinstance(
        trade_count,
        int,
    ):
        raise RuntimeError(
            "Gilman output has no valid trade_count."
        )

    if not isinstance(
        no_trade_count,
        int,
    ):
        raise RuntimeError(
            "Gilman output has no valid no_trade_count."
        )

    if (
        trade_count
        + no_trade_count
        != story_count
    ):
        raise RuntimeError(
            "Gilman counts do not add up."
        )

    trades = load_json(
        DATA_DIR
        / "gilman_trades_latest.json"
    )

    if (
        trades.get(
            "trade_count"
        )
        != trade_count
    ):
        raise RuntimeError(
            "Gilman output files disagree "
            "about trade_count."
        )

    logger.info(
        "✓ GILMAN: %d TRADE | %d NO_TRADE",
        trade_count,
        no_trade_count,
    )


def validate_market_context() -> None:
    gilman = load_json(
        DATA_DIR
        / "gilman_trades_latest.json"
    )

    market = load_json(
        DATA_DIR
        / "market_context_latest.json"
    )

    expected = gilman.get(
        "trade_count"
    )

    actual = market.get(
        "candidate_count"
    )

    if actual != expected:
        raise RuntimeError(
            "market_context candidate count "
            f"does not match Gilman: "
            f"{actual} != {expected}"
        )

    candidates = market.get(
        "candidates",
        [],
    )

    if len(candidates) != actual:
        raise RuntimeError(
            "market_context candidate_count "
            "does not match candidates array."
        )

    logger.info(
        "✓ MARKET CONTEXT: %d candidates",
        actual,
    )


def validate_brody() -> None:
    market = load_json(
        DATA_DIR
        / "market_context_latest.json"
    )

    brody = load_json(
        DATA_DIR
        / "brody_latest.json"
    )

    expected_candidates = (
        market.get(
            "candidate_count",
            0,
        )
    )

    actual_candidates = (
        brody.get(
            "candidate_count"
        )
    )

    if (
        actual_candidates
        != expected_candidates
    ):
        raise RuntimeError(
            "Brody candidate count does not "
            "match market_context."
        )

    decision = brody.get(
        "decision"
    )

    if decision not in {
        "TRADE",
        "NO_TRADE",
    }:
        raise RuntimeError(
            f"Invalid Brody decision: {decision}"
        )

    assessments = brody.get(
        "candidate_assessments",
        [],
    )

    if (
        len(assessments)
        != expected_candidates
    ):
        raise RuntimeError(
            "Brody did not assess every candidate."
        )

    approved = [
        item
        for item in assessments
        if item.get(
            "verdict"
        ) == "APPROVE"
    ]

    if decision == "NO_TRADE":
        if approved:
            raise RuntimeError(
                "Brody says NO_TRADE but "
                "approved a candidate."
            )

        if (
            brody.get(
                "selected_trade"
            )
            is not None
        ):
            raise RuntimeError(
                "Brody says NO_TRADE but "
                "selected_trade is populated."
            )

        logger.info(
            "✓ BRODY: NO_TRADE"
        )

        return

    # TRADE
    if len(approved) != 1:
        raise RuntimeError(
            "Brody TRADE must contain "
            "exactly one approved candidate."
        )

    selected = brody.get(
        "selected_trade"
    )

    if not selected:
        raise RuntimeError(
            "Brody says TRADE but "
            "selected_trade is missing."
        )

    logger.info(
        "✓ BRODY: APPROVED %s %s",
        selected["direction"],
        selected["ticker"],
    )


def validate_alpaca_execute() -> None:
    brody = load_json(
        DATA_DIR
        / "brody_latest.json"
    )

    execution = load_json(
        DATA_DIR
        / "alpaca_execution_latest.json"
    )

    status = execution.get(
        "status"
    )

    allowed_statuses = {
        "NO_TRADE",
        "NO_FILL",
        "POSITION_OPEN",
        "ALREADY_EXECUTED",
    }

    if status not in allowed_statuses:
        raise RuntimeError(
            f"Unexpected Alpaca execution "
            f"status: {status}"
        )

    if (
        brody["decision"]
        == "NO_TRADE"
        and status != "NO_TRADE"
    ):
        raise RuntimeError(
            "Brody said NO_TRADE but Alpaca "
            "execution did something else."
        )

    if (
        brody["decision"]
        == "TRADE"
        and status == "NO_TRADE"
    ):
        raise RuntimeError(
            "Brody approved a trade but Alpaca "
            "reported NO_TRADE."
        )

    if status == "POSITION_OPEN":
        details = execution.get(
            "details",
            {},
        )

        logger.info(
            "✓ ALPACA: POSITION OPEN | "
            "%s %s | qty=%s | entry=%.4f | "
            "target=%.4f",
            details.get(
                "direction"
            ),
            details.get(
                "ticker"
            ),
            details.get(
                "qty"
            ),
            details.get(
                "avg_entry_price"
            ),
            details.get(
                "take_profit_price"
            ),
        )

    elif status == "NO_FILL":
        logger.info(
            "✓ ALPACA: approved trade did not "
            "fill within execution window"
        )

    elif status == "ALREADY_EXECUTED":
        logger.info(
            "✓ ALPACA: trade was already executed; "
            "duplicate prevented"
        )

    else:
        logger.info(
            "✓ ALPACA: NO_TRADE"
        )


VALIDATORS = {
    "scripts.news_collect": (
        validate_news_collect
    ),
    "scripts.event_memory": (
        validate_event_memory
    ),
    "scripts.email_report": (
        validate_email_report
    ),
    "scripts.position_manager": (
        validate_position_manager
    ),
    "scripts.gilman": (
        validate_gilman
    ),
    "scripts.market_context": (
        validate_market_context
    ),
    "scripts.brody": (
        validate_brody
    ),
    "scripts.alpaca_execute": (
        validate_alpaca_execute
    ),
}


# ============================================================
# PIPELINE HELPERS
# ============================================================

def run_and_validate(
    module: str,
) -> None:

    run_stage(
        module
    )

    validator = VALIDATORS.get(
        module
    )

    if validator:
        # Validation failures deliberately DO NOT retry.
        #
        # If a script exited successfully but produced bad or
        # inconsistent output, that probably isn't a Wi-Fi
        # problem. Fail loudly instead of paying to rerun it
        # repeatedly.
        validator()


# ============================================================
# PIPELINE
# ============================================================

def main():
    args = parse_args()

    logger.info(
        "============================================================"
    )

    if args.no_news:
        logger.info(
            "Starting no_news pipeline"
        )
    else:
        logger.info(
            "Starting daily news → Gilman → "
            "Brody → Alpaca pipeline"
        )

    # Important for the "opened laptop at coffee shop" case.
    wait_for_internet()

    # --------------------------------------------------------
    # DAILY NEWS / EVENT MEMORY
    #
    # Normally these run every calendar day, including weekends
    # and exchange holidays.
    #
    # --no_news deliberately skips them and reuses the
    # existing enriched_latest.json so Gilman/Brody can be
    # tested without paying for another news/Event Memory run.
    # --------------------------------------------------------

    if args.no_news:
        logger.info(
            "Skipping news collection, Event Memory, "
            "and email report"
        )

        validate_no_news_input()

    else:
        for module in ALWAYS_RUN_STAGES:
            run_and_validate(
                module
            )

    # --------------------------------------------------------
    # MARKET GATE
    #
    # Gilman, market context, Brody, and execution only run while
    # regular US equity trading is actually open.
    #
    # This applies to --no_news as well. The flag changes
    # where the pipeline starts; it does not bypass trading-hour
    # safety.
    # --------------------------------------------------------

    wait_for_internet()

    if not is_market_open():
        logger.info(
            "============================================================"
        )

        logger.info(
            "DAILY PIPELINE COMPLETE: "
            "MARKET CLOSED"
        )

        if args.no_news:
            logger.info(
                "no_news mode requested, but the "
                "regular US equity market is closed. "
                "Skipping Gilman → Market Context → "
                "Brody → Alpaca."
            )
        else:
            logger.info(
                "News collection and Event Memory completed. "
                "Skipping Gilman → Market Context → "
                "Brody → Alpaca."
            )

        return

    # --------------------------------------------------------
    # INVESTMENT / EXECUTION PIPELINE
    # --------------------------------------------------------

    for module in MARKET_OPEN_STAGES:
        run_and_validate(
            module
        )

    logger.info(
        "============================================================"
    )

    if args.no_news:
        logger.info(
            "no_news PIPELINE SUCCESS"
        )
    else:
        logger.info(
            "DAILY PIPELINE SUCCESS"
        )


if __name__ == "__main__":
    main()