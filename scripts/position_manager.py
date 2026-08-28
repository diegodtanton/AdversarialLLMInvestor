from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
)
from alpaca.trading.requests import (
    MarketOrderRequest,
)
from dotenv import load_dotenv


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

DB_PATH = (
    DATA_DIR
    / "trading.db"
)

OUTPUT_PATH = (
    DATA_DIR
    / "position_manager_latest.json"
)

load_dotenv(
    ROOT / ".env"
)

ALPACA_API_KEY = os.environ[
    "ALPACA_API_KEY"
]

ALPACA_SECRET_KEY = os.environ[
    "ALPACA_SECRET_KEY"
]


def env_bool(
    name: str,
    default: bool,
) -> bool:

    raw = os.getenv(
        name,
        str(default),
    )

    return (
        raw.strip().casefold()
        in {
            "1",
            "true",
            "yes",
            "y",
        }
    )


ALPACA_PAPER = env_bool(
    "ALPACA_PAPER",
    True,
)

if not ALPACA_PAPER:
    raise RuntimeError(
        "position_manager.py is intentionally "
        "PAPER-ONLY. ALPACA_PAPER must be true."
    )


# How long to wait for a time-expiry market exit
# to fill before leaving it for reconciliation.
EXIT_FILL_TIMEOUT_SECONDS = int(
    os.getenv(
        "EXIT_FILL_TIMEOUT_SECONDS",
        "120",
    )
)

EXIT_POLL_SECONDS = int(
    os.getenv(
        "EXIT_POLL_SECONDS",
        "5",
    )
)

CANCEL_TIMEOUT_SECONDS = int(
    os.getenv(
        "CANCEL_TIMEOUT_SECONDS",
        "30",
    )
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# ALPACA
# ============================================================

trading_client = TradingClient(
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    paper=True,
)


# ============================================================
# DATABASE
# ============================================================

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS executions (
    decision_key TEXT PRIMARY KEY,

    trade_date TEXT NOT NULL,
    story_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,

    status TEXT NOT NULL,

    requested_notional REAL,

    entry_client_order_id TEXT,
    entry_order_id TEXT,
    requested_qty INTEGER,
    entry_limit_price REAL,

    filled_qty REAL,
    avg_entry_price REAL,
    filled_at TEXT,

    take_profit_price REAL,
    take_profit_client_order_id TEXT,
    take_profit_order_id TEXT,

    exit_deadline TEXT,

    exit_client_order_id TEXT,
    exit_order_id TEXT,

    closed_at TEXT,
    exit_price REAL,
    exit_reason TEXT,
    realized_return_pct REAL,

    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


EXTRA_COLUMNS = {
    "exit_client_order_id": "TEXT",
    "exit_order_id": "TEXT",
    "closed_at": "TEXT",
    "exit_price": "REAL",
    "exit_reason": "TEXT",
    "realized_return_pct": "REAL",
}


def connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(
        DB_PATH
    )

    conn.row_factory = sqlite3.Row

    conn.executescript(
        SCHEMA_SQL
    )

    # Existing databases created by alpaca_execute.py
    # won't automatically acquire new columns from
    # CREATE TABLE IF NOT EXISTS, so migrate them here.
    existing_columns = {
        row["name"]
        for row in conn.execute(
            "PRAGMA table_info(executions)"
        ).fetchall()
    }

    for (
        column_name,
        column_type,
    ) in EXTRA_COLUMNS.items():

        if (
            column_name
            not in existing_columns
        ):
            logger.info(
                "Adding trading.db column: %s",
                column_name,
            )

            conn.execute(
                f"""
                ALTER TABLE executions
                ADD COLUMN {column_name} {column_type}
                """
            )

    conn.commit()

    return conn


# ============================================================
# HELPERS
# ============================================================

def utc_now_dt() -> datetime:
    return datetime.now(
        timezone.utc
    )


def utc_now() -> str:
    return utc_now_dt().isoformat()


def parse_datetime(
    value: str | None,
) -> datetime | None:

    if not value:
        return None

    result = datetime.fromisoformat(
        value.replace(
            "Z",
            "+00:00",
        )
    )

    if result.tzinfo is None:
        result = result.replace(
            tzinfo=timezone.utc
        )

    return result.astimezone(
        timezone.utc
    )


def enum_value(
    value,
) -> str:

    if value is None:
        return ""

    return str(
        getattr(
            value,
            "value",
            value,
        )
    ).casefold()


OPEN_ORDER_STATUSES = {
    "new",
    "accepted",
    "pending_new",
    "accepted_for_bidding",
    "partially_filled",
    "pending_replace",
    "pending_cancel",
    "pending_review",
    "held",
}


TERMINAL_ORDER_STATUSES = {
    "filled",
    "canceled",
    "cancelled",
    "expired",
    "rejected",
    "done_for_day",
    "stopped",
    "suspended",
    "replaced",
}


def is_order_open(
    order,
) -> bool:

    return (
        enum_value(
            order.status
        )
        in OPEN_ORDER_STATUSES
    )


def is_order_filled(
    order,
) -> bool:

    return (
        enum_value(
            order.status
        )
        == "filled"
    )


def order_filled_qty(
    order,
) -> float:

    return float(
        order.filled_qty
        or 0
    )


def order_avg_price(
    order,
) -> float | None:

    if not order.filled_avg_price:
        return None

    return float(
        order.filled_avg_price
    )


def order_filled_at(
    order,
) -> str:

    value = (
        order.filled_at
        or utc_now_dt()
    )

    if value.tzinfo is None:
        value = value.replace(
            tzinfo=timezone.utc
        )

    return (
        value
        .astimezone(
            timezone.utc
        )
        .isoformat()
    )


# ============================================================
# DATABASE HELPERS
# ============================================================

def update_execution(
    conn: sqlite3.Connection,
    decision_key: str,
    **fields,
) -> None:

    if not fields:
        return

    fields[
        "updated_at"
    ] = utc_now()

    assignments = ", ".join(
        f"{name} = ?"
        for name in fields
    )

    values = list(
        fields.values()
    )

    values.append(
        decision_key
    )

    conn.execute(
        f"""
        UPDATE executions
        SET {assignments}
        WHERE decision_key = ?
        """,
        values,
    )

    conn.commit()


def active_executions(
    conn: sqlite3.Connection,
) -> list[sqlite3.Row]:

    return conn.execute(
        """
        SELECT *
        FROM executions
        WHERE status IN (
            'TAKE_PROFIT_OPEN',
            'POSITION_OPEN',
            'TIME_EXIT_SUBMITTING',
            'TIME_EXIT_SUBMITTED'
        )
        ORDER BY filled_at ASC
        """
    ).fetchall()


# ============================================================
# POSITION HELPERS
# ============================================================

def get_position(
    ticker: str,
):

    positions = (
        trading_client
        .get_all_positions()
    )

    for position in positions:
        if (
            position.symbol.upper()
            == ticker.upper()
        ):
            return position

    return None


def validate_position_side(
    position,
    direction: str,
) -> None:

    side = enum_value(
        position.side
    )

    if (
        direction == "FOR"
        and side != "long"
    ):
        raise RuntimeError(
            f"Expected long position in "
            f"{position.symbol}, found {side}."
        )

    if (
        direction == "AGAINST"
        and side != "short"
    ):
        raise RuntimeError(
            f"Expected short position in "
            f"{position.symbol}, found {side}."
        )


# ============================================================
# ORDER HELPERS
# ============================================================

def get_order(
    order_id: str | None,
):

    if not order_id:
        return None

    return (
        trading_client
        .get_order_by_id(
            order_id
        )
    )


def api_error_status_code(
    exc: APIError,
) -> int | None:

    error = getattr(
        exc,
        "error",
        None,
    )

    if error is None:
        return None

    if isinstance(
        error,
        dict,
    ):
        return error.get(
            "status_code"
        )

    return getattr(
        error,
        "status_code",
        None,
    )


def get_order_by_client_id_or_none(
    client_order_id: str,
):

    try:
        return (
            trading_client
            .get_order_by_client_id(
                client_order_id
            )
        )

    except APIError as exc:
        if (
            api_error_status_code(
                exc
            )
            == 404
        ):
            return None

        raise


def cancel_order_and_wait(
    order,
):

    if order is None:
        return None

    if not is_order_open(
        order
    ):
        return order

    logger.info(
        "Canceling order %s",
        order.id,
    )

    trading_client.cancel_order_by_id(
        order.id
    )

    deadline = (
        time.monotonic()
        + CANCEL_TIMEOUT_SECONDS
    )

    while (
        time.monotonic()
        < deadline
    ):
        time.sleep(1)

        current = get_order(
            str(order.id)
        )

        if not is_order_open(
            current
        ):
            return current

    raise RuntimeError(
        f"Order {order.id} did not cancel "
        f"within {CANCEL_TIMEOUT_SECONDS} seconds."
    )


# ============================================================
# RETURN CALCULATION
# ============================================================

def realized_return_pct(
    direction: str,
    entry_price: float | None,
    exit_price: float | None,
) -> float | None:

    if (
        entry_price is None
        or exit_price is None
        or entry_price <= 0
        or exit_price <= 0
    ):
        return None

    if direction == "FOR":
        result = (
            exit_price
            / entry_price
            - 1
        )

    else:
        result = (
            entry_price
            / exit_price
            - 1
        )

    return round(
        result * 100,
        4,
    )


def combined_exit_price(
    row: sqlite3.Row,
    take_profit_order,
    final_exit_order,
) -> float | None:
    """
    If some shares hit the +10% order and the remaining
    shares are liquidated at day 28, calculate a weighted
    average exit price.
    """

    final_qty = order_filled_qty(
        final_exit_order
    )

    final_price = order_avg_price(
        final_exit_order
    )

    if (
        final_qty <= 0
        or final_price is None
    ):
        return None

    tp_qty = 0.0
    tp_price = None

    if take_profit_order:
        tp_qty = order_filled_qty(
            take_profit_order
        )

        tp_price = order_avg_price(
            take_profit_order
        )

    if (
        tp_qty <= 0
        or tp_price is None
    ):
        return final_price

    total_qty = (
        tp_qty
        + final_qty
    )

    if total_qty <= 0:
        return None

    return (
        (
            tp_qty
            * tp_price
        )
        + (
            final_qty
            * final_price
        )
    ) / total_qty


# ============================================================
# CLOSED POSITION RECORD
# ============================================================

def mark_closed(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    exit_price: float | None,
    closed_at: str,
    exit_reason: str,
) -> None:

    entry_price = (
        float(
            row["avg_entry_price"]
        )
        if row[
            "avg_entry_price"
        ] is not None
        else None
    )

    return_pct = (
        realized_return_pct(
            row["direction"],
            entry_price,
            exit_price,
        )
    )

    update_execution(
        conn,
        row["decision_key"],
        status="CLOSED",
        closed_at=closed_at,
        exit_price=exit_price,
        exit_reason=exit_reason,
        realized_return_pct=(
            return_pct
        ),
    )

    logger.info(
        "CLOSED %s %s | reason=%s | "
        "entry=%s | exit=%s | return=%s%%",
        row["direction"],
        row["ticker"],
        exit_reason,
        entry_price,
        exit_price,
        return_pct,
    )


# ============================================================
# TIME EXIT
# ============================================================

def time_exit_client_id(
    row: sqlite3.Row,
) -> str:
    """
    Deterministic within a given day.

    If Wi-Fi dies after Alpaca accepts the order but before
    Python receives the response, rerunning today will retrieve
    the same order instead of submitting another.
    """

    digest = hashlib.sha1(
        row[
            "decision_key"
        ].encode(
            "utf-8"
        )
    ).hexdigest()[:14]

    today = (
        utc_now_dt()
        .date()
        .strftime(
            "%Y%m%d"
        )
    )

    return (
        f"pm-{digest}-{today}"
    )


def submit_or_recover_time_exit(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    position,
):

    validate_position_side(
        position,
        row["direction"],
    )

    client_order_id = (
        time_exit_client_id(
            row
        )
    )

    # --------------------------------------------------------
    # RECOVERY FIRST
    #
    # Maybe yesterday/earlier today we submitted this order
    # but lost the network response.
    # --------------------------------------------------------

    existing = (
        get_order_by_client_id_or_none(
            client_order_id
        )
    )

    if existing:
        logger.info(
            "Recovered existing time-exit "
            "order %s for %s",
            existing.id,
            row["ticker"],
        )

        update_execution(
            conn,
            row["decision_key"],
            status=(
                "TIME_EXIT_SUBMITTED"
            ),
            exit_client_order_id=(
                client_order_id
            ),
            exit_order_id=str(
                existing.id
            ),
        )

        return existing

    # --------------------------------------------------------
    # SUBMIT
    # --------------------------------------------------------

    qty = abs(
        float(
            position.qty
        )
    )

    if qty <= 0:
        raise RuntimeError(
            f"Invalid position quantity "
            f"for {row['ticker']}: {qty}"
        )

    if row["direction"] == "FOR":
        side = OrderSide.SELL
    else:
        side = OrderSide.BUY

    # Write the deterministic ID locally BEFORE sending
    # the network request.
    update_execution(
        conn,
        row["decision_key"],
        status="TIME_EXIT_SUBMITTING",
        exit_client_order_id=(
            client_order_id
        ),
    )

    logger.info(
        "Submitting 28-day market exit: "
        "%s | qty=%s",
        row["ticker"],
        qty,
    )

    order = (
        trading_client
        .submit_order(
            order_data=(
                MarketOrderRequest(
                    symbol=(
                        row["ticker"]
                    ),
                    qty=qty,
                    side=side,
                    time_in_force=(
                        TimeInForce.DAY
                    ),
                    extended_hours=False,
                    client_order_id=(
                        client_order_id
                    ),
                )
            )
        )
    )

    update_execution(
        conn,
        row["decision_key"],
        status=(
            "TIME_EXIT_SUBMITTED"
        ),
        exit_order_id=str(
            order.id
        ),
    )

    return order


def wait_for_exit_fill(
    order,
):

    deadline = (
        time.monotonic()
        + EXIT_FILL_TIMEOUT_SECONDS
    )

    while (
        time.monotonic()
        < deadline
    ):
        current = get_order(
            str(
                order.id
            )
        )

        logger.info(
            "Time-exit order %s status=%s "
            "filled=%s/%s",
            current.id,
            current.status,
            current.filled_qty,
            current.qty,
        )

        if is_order_filled(
            current
        ):
            return current

        status = enum_value(
            current.status
        )

        if (
            status
            in TERMINAL_ORDER_STATUSES
            and status != "filled"
        ):
            return current

        time.sleep(
            EXIT_POLL_SECONDS
        )

    return get_order(
        str(
            order.id
        )
    )


# ============================================================
# ONE POSITION
# ============================================================

def manage_execution(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
) -> dict:

    ticker = row[
        "ticker"
    ]

    logger.info(
        "------------------------------------------------------------"
    )

    logger.info(
        "Managing %s %s",
        row["direction"],
        ticker,
    )

    # --------------------------------------------------------
    # 1. RECONCILE A PREVIOUSLY SUBMITTED TIME EXIT
    # --------------------------------------------------------

    if row[
        "exit_order_id"
    ]:
        exit_order = get_order(
            row[
                "exit_order_id"
            ]
        )

        if (
            exit_order
            and is_order_filled(
                exit_order
            )
        ):
            exit_price = order_avg_price(
                exit_order
            )

            mark_closed(
                conn,
                row,
                exit_price=exit_price,
                closed_at=order_filled_at(
                    exit_order
                ),
                exit_reason=(
                    "TIME_LIMIT"
                ),
            )

            return {
                "ticker": ticker,
                "action": "CLOSED_TIME_LIMIT",
                "exit_price": exit_price,
            }

        if (
            exit_order
            and is_order_open(
                exit_order
            )
        ):
            logger.info(
                "%s already has an active "
                "time-exit order.",
                ticker,
            )

            return {
                "ticker": ticker,
                "action": "WAITING_FOR_TIME_EXIT",
            }

    # --------------------------------------------------------
    # 2. RECONCILE TAKE PROFIT
    # --------------------------------------------------------

    take_profit_order = (
        get_order(
            row[
                "take_profit_order_id"
            ]
        )
        if row[
            "take_profit_order_id"
        ]
        else None
    )

    if (
        take_profit_order
        and is_order_filled(
            take_profit_order
        )
    ):
        exit_price = order_avg_price(
            take_profit_order
        )

        mark_closed(
            conn,
            row,
            exit_price=exit_price,
            closed_at=order_filled_at(
                take_profit_order
            ),
            exit_reason=(
                "TAKE_PROFIT"
            ),
        )

        return {
            "ticker": ticker,
            "action": "CLOSED_TAKE_PROFIT",
            "exit_price": exit_price,
        }

    # --------------------------------------------------------
    # 3. VERIFY POSITION STILL EXISTS
    # --------------------------------------------------------

    position = get_position(
        ticker
    )

    if position is None:
        # Critical cleanup:
        #
        # If somebody manually closed the position but left our
        # take-profit order alive, that sell/buy order must be
        # canceled or it could create a NEW position later.
        if (
            take_profit_order
            and is_order_open(
                take_profit_order
            )
        ):
            take_profit_order = (
                cancel_order_and_wait(
                    take_profit_order
                )
            )

            # Maybe it filled during the cancellation race.
            if (
                take_profit_order
                and is_order_filled(
                    take_profit_order
                )
            ):
                exit_price = (
                    order_avg_price(
                        take_profit_order
                    )
                )

                mark_closed(
                    conn,
                    row,
                    exit_price=(
                        exit_price
                    ),
                    closed_at=(
                        order_filled_at(
                            take_profit_order
                        )
                    ),
                    exit_reason=(
                        "TAKE_PROFIT"
                    ),
                )

                return {
                    "ticker": ticker,
                    "action": (
                        "CLOSED_TAKE_PROFIT"
                    ),
                    "exit_price": (
                        exit_price
                    ),
                }

        mark_closed(
            conn,
            row,
            exit_price=None,
            closed_at=utc_now(),
            exit_reason=(
                "POSITION_NOT_FOUND"
            ),
        )

        return {
            "ticker": ticker,
            "action": "POSITION_NOT_FOUND",
        }

    validate_position_side(
        position,
        row["direction"],
    )

    # --------------------------------------------------------
    # 4. CHECK TIME LIMIT
    # --------------------------------------------------------

    deadline = parse_datetime(
        row[
            "exit_deadline"
        ]
    )

    if deadline is None:
        raise RuntimeError(
            f"{ticker} has no exit_deadline."
        )

    now = utc_now_dt()

    if now < deadline:
        # We expect the take-profit order to remain alive.
        if take_profit_order is None:
            raise RuntimeError(
                f"{ticker} is open but has no "
                "take-profit order record."
            )

        if not is_order_open(
            take_profit_order
        ):
            raise RuntimeError(
                f"{ticker} position is open but "
                f"take-profit order status is "
                f"{take_profit_order.status}. "
                "Manual attention required."
            )

        remaining = (
            deadline
            - now
        )

        logger.info(
            "%s still inside holding period | "
            "deadline=%s | %.2f days remain",
            ticker,
            deadline.isoformat(),
            (
                remaining.total_seconds()
                / 86400
            ),
        )

        return {
            "ticker": ticker,
            "action": "HOLD",
            "exit_deadline": (
                deadline.isoformat()
            ),
        }

    # --------------------------------------------------------
    # 5. 28 DAYS REACHED
    #
    # First cancel the +10% order.
    # --------------------------------------------------------

    logger.info(
        "%s reached its 28-day deadline.",
        ticker,
    )

    if (
        take_profit_order
        and is_order_open(
            take_profit_order
        )
    ):
        take_profit_order = (
            cancel_order_and_wait(
                take_profit_order
            )
        )

        # It may have filled just as we canceled.
        if is_order_filled(
            take_profit_order
        ):
            exit_price = order_avg_price(
                take_profit_order
            )

            mark_closed(
                conn,
                row,
                exit_price=exit_price,
                closed_at=order_filled_at(
                    take_profit_order
                ),
                exit_reason=(
                    "TAKE_PROFIT"
                ),
            )

            return {
                "ticker": ticker,
                "action": "CLOSED_TAKE_PROFIT",
                "exit_price": exit_price,
            }

    # Recheck the actual position AFTER canceling the TP.
    position = get_position(
        ticker
    )

    if position is None:
        mark_closed(
            conn,
            row,
            exit_price=None,
            closed_at=utc_now(),
            exit_reason=(
                "POSITION_NOT_FOUND_AFTER_TP_CANCEL"
            ),
        )

        return {
            "ticker": ticker,
            "action": "POSITION_NOT_FOUND",
        }

    # --------------------------------------------------------
    # 6. SUBMIT / RECOVER TIME EXIT
    # --------------------------------------------------------

    exit_order = (
        submit_or_recover_time_exit(
            conn,
            row,
            position,
        )
    )

    exit_order = wait_for_exit_fill(
        exit_order
    )

    if is_order_filled(
        exit_order
    ):
        weighted_exit_price = (
            combined_exit_price(
                row,
                take_profit_order,
                exit_order,
            )
        )

        had_partial_tp = (
            take_profit_order is not None
            and order_filled_qty(
                take_profit_order
            ) > 0
        )

        exit_reason = (
            "TIME_LIMIT_AFTER_PARTIAL_TAKE_PROFIT"
            if had_partial_tp
            else "TIME_LIMIT"
        )

        mark_closed(
            conn,
            row,
            exit_price=(
                weighted_exit_price
            ),
            closed_at=(
                order_filled_at(
                    exit_order
                )
            ),
            exit_reason=(
                exit_reason
            ),
        )

        return {
            "ticker": ticker,
            "action": "CLOSED_TIME_LIMIT",
            "exit_price": (
                weighted_exit_price
            ),
        }

    # If it didn't fill immediately, leave it alone.
    #
    # We have a live exit order at Alpaca and will reconcile
    # it on the next run.
    status = enum_value(
        exit_order.status
    )

    if is_order_open(
        exit_order
    ):
        update_execution(
            conn,
            row[
                "decision_key"
            ],
            status=(
                "TIME_EXIT_SUBMITTED"
            ),
        )

        logger.warning(
            "%s time-exit order remains open: %s",
            ticker,
            exit_order.status,
        )

        return {
            "ticker": ticker,
            "action": (
                "TIME_EXIT_SUBMITTED"
            ),
            "order_status": (
                status
            ),
        }

    # Rejected/canceled/etc.
    #
    # Fail loudly rather than pretending the position was closed.
    raise RuntimeError(
        f"{ticker} 28-day exit order ended "
        f"with status {exit_order.status} "
        "while the position may still be open."
    )


# ============================================================
# OUTPUT
# ============================================================

def write_output(
    actions: list[dict],
) -> None:

    closed = [
        action
        for action in actions
        if action[
            "action"
        ].startswith(
            "CLOSED_"
        )
    ]

    holding = [
        action
        for action in actions
        if action[
            "action"
        ] == "HOLD"
    ]

    pending_exit = [
        action
        for action in actions
        if action[
            "action"
        ]
        in {
            "TIME_EXIT_SUBMITTED",
            "WAITING_FOR_TIME_EXIT",
        }
    ]

    output = {
        "generated_at": utc_now(),
        "managed_count": len(
            actions
        ),
        "closed_count": len(
            closed
        ),
        "holding_count": len(
            holding
        ),
        "pending_exit_count": len(
            pending_exit
        ),
        "actions": actions,
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    logger.info(
        "Wrote %s",
        OUTPUT_PATH,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info(
        "Starting position manager"
    )

    conn = connect_db()

    try:
        rows = active_executions(
            conn
        )

        logger.info(
            "Managed open positions: %d",
            len(rows),
        )

        actions = []

        for row in rows:
            action = manage_execution(
                conn,
                row,
            )

            actions.append(
                action
            )

        write_output(
            actions
        )

    finally:
        conn.close()

    logger.info(
        "Position manager complete"
    )


if __name__ == "__main__":
    main()