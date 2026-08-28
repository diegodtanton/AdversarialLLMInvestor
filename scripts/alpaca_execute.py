from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    OrderSide,
    OrderStatus,
    TimeInForce,
)
from alpaca.trading.requests import (
    LimitOrderRequest,
    ReplaceOrderRequest,
)
from dotenv import load_dotenv


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

INPUT_PATH = (
    DATA_DIR
    / "brody_latest.json"
)

DB_PATH = (
    DATA_DIR
    / "trading.db"
)

EXECUTION_LATEST_PATH = (
    DATA_DIR
    / "alpaca_execution_latest.json"
)

load_dotenv(ROOT / ".env")


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

# Intentionally fail hard in this version if anyone points this
# script at a live brokerage account.
if not ALPACA_PAPER:
    raise RuntimeError(
        "alpaca_execute.py is intentionally PAPER-ONLY. "
        "ALPACA_PAPER must be true."
    )


TRADE_NOTIONAL_USD = float(
    os.getenv(
        "TRADE_NOTIONAL_USD",
        "1000",
    )
)

ENTRY_CUSHION_BPS = float(
    os.getenv(
        "ENTRY_CUSHION_BPS",
        "25",
    )
)

# Refuse truly awful quotes but don't be overly restrictive.
MAX_SPREAD_BPS = float(
    os.getenv(
        "MAX_SPREAD_BPS",
        "150",
    )
)

MAX_ENTRY_WAIT_MINUTES = int(
    os.getenv(
        "MAX_ENTRY_WAIT_MINUTES",
        "180",
    )
)

ENTRY_REPRICE_MINUTES = int(
    os.getenv(
        "ENTRY_REPRICE_MINUTES",
        "15",
    )
)

ENTRY_POLL_SECONDS = int(
    os.getenv(
        "ENTRY_POLL_SECONDS",
        "30",
    )
)

TAKE_PROFIT_PCT = float(
    os.getenv(
        "TAKE_PROFIT_PCT",
        "0.10",
    )
)

MAX_HOLD_DAYS = int(
    os.getenv(
        "MAX_HOLD_DAYS",
        "28",
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
# CLIENTS
# ============================================================

trading_client = TradingClient(
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    paper=True,
)

market_data_client = (
    StockHistoricalDataClient(
        ALPACA_API_KEY,
        ALPACA_SECRET_KEY,
    )
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

    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_executions_status
ON executions(status);

CREATE INDEX IF NOT EXISTS idx_executions_ticker
ON executions(ticker);
"""


def connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(
        DB_PATH
    )

    conn.row_factory = sqlite3.Row

    conn.executescript(
        SCHEMA_SQL
    )

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


def load_brody() -> dict:
    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def decision_key(
    payload: dict,
) -> str:

    trade = payload[
        "selected_trade"
    ]

    return (
        f"{payload['date']}:"
        f"{trade['story_id']}:"
        f"{trade['ticker']}:"
        f"{trade['direction']}"
    )


def client_id_prefix(
    payload: dict,
) -> str:

    trade = payload[
        "selected_trade"
    ]

    date_part = payload[
        "date"
    ].replace(
        "-",
        "",
    )

    story_part = (
        trade[
            "story_id"
        ][-12:]
    )

    ticker = (
        trade[
            "ticker"
        ]
        .lower()
        .replace(
            ".",
            "-",
        )
    )

    return (
        f"brody-{date_part}-"
        f"{story_part}-{ticker}"
    )


def enum_value(value) -> str:
    if value is None:
        return ""

    return str(
        getattr(
            value,
            "value",
            value,
        )
    ).casefold()


def price_increment(
    asset,
) -> Decimal:

    raw = getattr(
        asset,
        "price_increment",
        None,
    )

    if raw:
        return Decimal(
            str(raw)
        )

    # Conventional fallback for listed US stocks.
    return Decimal(
        "0.01"
    )


def round_to_increment(
    price: float,
    increment: Decimal,
    *,
    up: bool,
) -> float:

    p = Decimal(
        str(price)
    )

    units = p / increment

    rounding = (
        ROUND_CEILING
        if up
        else ROUND_FLOOR
    )

    units = units.to_integral_value(
        rounding=rounding
    )

    result = (
        units
        * increment
    )

    return float(result)


def is_terminal_status(
    status,
) -> bool:

    return status in {
        OrderStatus.FILLED,
        OrderStatus.CANCELED,
        OrderStatus.EXPIRED,
        OrderStatus.REJECTED,
        OrderStatus.DONE_FOR_DAY,
        OrderStatus.STOPPED,
        OrderStatus.SUSPENDED,
    }


def is_order_open(
    status,
) -> bool:

    return status in {
        OrderStatus.NEW,
        OrderStatus.ACCEPTED,
        OrderStatus.PENDING_NEW,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.ACCEPTED_FOR_BIDDING,
        OrderStatus.PENDING_REPLACE,
        OrderStatus.PENDING_REVIEW,
    }


# ============================================================
# QUOTES
# ============================================================

def latest_quote(
    ticker: str,
) -> dict:

    request = StockLatestQuoteRequest(
        symbol_or_symbols=ticker
    )

    result = (
        market_data_client
        .get_stock_latest_quote(
            request
        )
    )

    quote = result[
        ticker
    ]

    bid = float(
        quote.bid_price
        or 0
    )

    ask = float(
        quote.ask_price
        or 0
    )

    if (
        bid <= 0
        or ask <= 0
        or ask < bid
    ):
        raise RuntimeError(
            f"Invalid quote for {ticker}: "
            f"bid={bid}, ask={ask}"
        )

    mid = (
        bid + ask
    ) / 2

    spread_bps = (
        (ask - bid)
        / mid
        * 10000
    )

    return {
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "spread_bps": (
            spread_bps
        ),
    }


def entry_limit_from_quote(
    quote: dict,
    direction: str,
    increment: Decimal,
) -> float:

    cushion = (
        ENTRY_CUSHION_BPS
        / 10000
    )

    if direction == "FOR":
        # Buy limit slightly ABOVE ask.
        #
        # This is marketable but still provides a maximum price.
        raw = (
            quote["ask"]
            * (1 + cushion)
        )

        return round_to_increment(
            raw,
            increment,
            up=True,
        )

    # Short sale.
    #
    # Sell limit slightly BELOW bid.
    # It can execute at the bid or anything better.
    raw = (
        quote["bid"]
        * (1 - cushion)
    )

    return round_to_increment(
        raw,
        increment,
        up=False,
    )


# ============================================================
# PRE-FLIGHT
# ============================================================

def get_borrow_status(
    asset,
) -> str:

    # New Alpaca field.
    current = getattr(
        asset,
        "borrow_status",
        None,
    )

    if current is not None:
        return enum_value(
            current
        )

    # Compatibility with current alpaca-py versions while
    # easy_to_borrow is being phased out.
    legacy = getattr(
        asset,
        "easy_to_borrow",
        None,
    )

    if legacy is True:
        return "easy_to_borrow"

    if legacy is False:
        return "not_easy_to_borrow"

    return "unknown"


def check_existing_position(
    ticker: str,
) -> None:

    positions = (
        trading_client
        .get_all_positions()
    )

    for position in positions:
        if (
            position.symbol.upper()
            == ticker.upper()
        ):
            raise RuntimeError(
                f"Existing Alpaca position already "
                f"open in {ticker}."
            )


def preflight(
    ticker: str,
    direction: str,
):

    account = (
        trading_client
        .get_account()
    )

    if account.trading_blocked:
        raise RuntimeError(
            "Alpaca account is blocked from trading."
        )

    if (
        float(
            account.buying_power
            or 0
        )
        < TRADE_NOTIONAL_USD
    ):
        raise RuntimeError(
            "Insufficient Alpaca buying power."
        )

    asset = (
        trading_client
        .get_asset(
            ticker
        )
    )

    if not asset.tradable:
        raise RuntimeError(
            f"{ticker} is not tradable on Alpaca."
        )

    if direction == "AGAINST":

        if not asset.shortable:
            raise RuntimeError(
                f"{ticker} is not shortable."
            )

        shorting_enabled = getattr(
            account,
            "shorting_enabled",
            None,
        )

        if shorting_enabled is False:
            raise RuntimeError(
                "Shorting is disabled on this account."
            )

        borrow_status = (
            get_borrow_status(
                asset
            )
        )

        logger.info(
            "%s borrow status: %s",
            ticker,
            borrow_status,
        )

        # First implementation deliberately avoids HTB locates.
        if (
            borrow_status
            != "easy_to_borrow"
        ):
            raise RuntimeError(
                f"{ticker} is not confirmed "
                f"easy-to-borrow: "
                f"{borrow_status}"
            )

    check_existing_position(
        ticker
    )

    clock = (
        trading_client
        .get_clock()
    )

    if not clock.is_open:
        raise RuntimeError(
            "US equity market is currently closed. "
            "Executor will not queue a stale entry overnight."
        )

    return (
        account,
        asset,
    )


# ============================================================
# DATABASE WRITES
# ============================================================

def upsert_execution(
    conn: sqlite3.Connection,
    key: str,
    payload: dict,
    *,
    status: str,
    entry_client_order_id: str | None = None,
    entry_order_id: str | None = None,
    requested_qty: int | None = None,
    entry_limit_price: float | None = None,
    filled_qty: float | None = None,
    avg_entry_price: float | None = None,
    filled_at: str | None = None,
    take_profit_price: float | None = None,
    take_profit_client_order_id: str | None = None,
    take_profit_order_id: str | None = None,
    exit_deadline: str | None = None,
) -> None:

    trade = payload[
        "selected_trade"
    ]

    existing = conn.execute(
        """
        SELECT decision_key
        FROM executions
        WHERE decision_key = ?
        """,
        (key,),
    ).fetchone()

    if existing:
        conn.execute(
            """
            UPDATE executions
            SET
                status = ?,
                entry_client_order_id = COALESCE(?, entry_client_order_id),
                entry_order_id = COALESCE(?, entry_order_id),
                requested_qty = COALESCE(?, requested_qty),
                entry_limit_price = COALESCE(?, entry_limit_price),
                filled_qty = COALESCE(?, filled_qty),
                avg_entry_price = COALESCE(?, avg_entry_price),
                filled_at = COALESCE(?, filled_at),
                take_profit_price = COALESCE(?, take_profit_price),
                take_profit_client_order_id = COALESCE(?, take_profit_client_order_id),
                take_profit_order_id = COALESCE(?, take_profit_order_id),
                exit_deadline = COALESCE(?, exit_deadline),
                updated_at = ?
            WHERE decision_key = ?
            """,
            (
                status,
                entry_client_order_id,
                entry_order_id,
                requested_qty,
                entry_limit_price,
                filled_qty,
                avg_entry_price,
                filled_at,
                take_profit_price,
                take_profit_client_order_id,
                take_profit_order_id,
                exit_deadline,
                utc_now(),
                key,
            ),
        )

    else:
        conn.execute(
            """
            INSERT INTO executions (
                decision_key,
                trade_date,
                story_id,
                ticker,
                direction,
                status,
                requested_notional,
                entry_client_order_id,
                entry_order_id,
                requested_qty,
                entry_limit_price,
                filled_qty,
                avg_entry_price,
                filled_at,
                take_profit_price,
                take_profit_client_order_id,
                take_profit_order_id,
                exit_deadline,
                created_at,
                updated_at
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                key,
                payload["date"],
                trade["story_id"],
                trade["ticker"],
                trade["direction"],
                status,
                TRADE_NOTIONAL_USD,
                entry_client_order_id,
                entry_order_id,
                requested_qty,
                entry_limit_price,
                filled_qty,
                avg_entry_price,
                filled_at,
                take_profit_price,
                take_profit_client_order_id,
                take_profit_order_id,
                exit_deadline,
                utc_now(),
                utc_now(),
            ),
        )

    conn.commit()


# ============================================================
# ENTRY
# ============================================================

def submit_entry(
    ticker: str,
    direction: str,
    qty: int,
    limit_price: float,
    client_order_id: str,
):

    side = (
        OrderSide.BUY
        if direction == "FOR"
        else OrderSide.SELL
    )

    request = LimitOrderRequest(
        symbol=ticker,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.DAY,
        limit_price=limit_price,
        extended_hours=False,
        client_order_id=(
            client_order_id
        ),
    )

    return (
        trading_client
        .submit_order(
            order_data=request
        )
    )


def manage_entry_order(
    order,
    ticker: str,
    direction: str,
    increment: Decimal,
):

    started = time.monotonic()

    next_reprice = (
        started
        + ENTRY_REPRICE_MINUTES
        * 60
    )

    max_wait_seconds = (
        MAX_ENTRY_WAIT_MINUTES
        * 60
    )

    while True:

        current = (
            trading_client
            .get_order_by_id(
                order.id
            )
        )

        logger.info(
            "Entry %s status=%s "
            "filled=%s/%s "
            "limit=%s",
            ticker,
            current.status,
            current.filled_qty,
            current.qty,
            current.limit_price,
        )

        if (
            current.status
            == OrderStatus.FILLED
        ):
            return current

        if (
            is_terminal_status(
                current.status
            )
            and current.status
            != OrderStatus.FILLED
        ):
            return current

        elapsed = (
            time.monotonic()
            - started
        )

        if (
            elapsed
            >= max_wait_seconds
        ):
            logger.info(
                "Maximum entry wait reached."
            )

            break

        clock = (
            trading_client
            .get_clock()
        )

        if not clock.is_open:
            logger.info(
                "Market closed before "
                "entry completed."
            )

            break

        if (
            time.monotonic()
            >= next_reprice
        ):
            quote = latest_quote(
                ticker
            )

            if (
                quote[
                    "spread_bps"
                ]
                <= MAX_SPREAD_BPS
            ):
                new_limit = (
                    entry_limit_from_quote(
                        quote,
                        direction,
                        increment,
                    )
                )

                current_limit = float(
                    current.limit_price
                    or 0
                )

                if (
                    new_limit
                    != current_limit
                ):
                    logger.info(
                        "Repricing %s entry "
                        "%.4f -> %.4f",
                        ticker,
                        current_limit,
                        new_limit,
                    )

                    current = (
                        trading_client
                        .replace_order_by_id(
                            current.id,
                            ReplaceOrderRequest(
                                limit_price=(
                                    new_limit
                                )
                            ),
                        )
                    )

                    # Replacement creates/returns the
                    # currently active order object.
                    order = current

            else:
                logger.warning(
                    "%s spread %.1f bps exceeds "
                    "limit; not repricing right now.",
                    ticker,
                    quote[
                        "spread_bps"
                    ],
                )

            next_reprice = (
                time.monotonic()
                + ENTRY_REPRICE_MINUTES
                * 60
            )

        time.sleep(
            ENTRY_POLL_SECONDS
        )

    # Cancel remaining quantity.
    current = (
        trading_client
        .get_order_by_id(
            order.id
        )
    )

    if is_order_open(
        current.status
    ):
        try:
            trading_client.cancel_order_by_id(
                current.id
            )

            time.sleep(2)

        except Exception:
            logger.exception(
                "Entry cancellation returned an error."
            )

    return (
        trading_client
        .get_order_by_id(
            order.id
        )
    )


# ============================================================
# TAKE PROFIT
# ============================================================

def calculate_take_profit(
    avg_entry_price: float,
    direction: str,
    increment: Decimal,
) -> float:

    if direction == "FOR":
        raw = (
            avg_entry_price
            * (
                1
                + TAKE_PROFIT_PCT
            )
        )

        # Long needs sell price >= target.
        return round_to_increment(
            raw,
            increment,
            up=True,
        )

    raw = (
        avg_entry_price
        * (
            1
            - TAKE_PROFIT_PCT
        )
    )

    # Short needs buy price <= target.
    return round_to_increment(
        raw,
        increment,
        up=False,
    )


def submit_take_profit(
    ticker: str,
    direction: str,
    qty: int,
    take_profit_price: float,
    client_order_id: str,
):

    side = (
        OrderSide.SELL
        if direction == "FOR"
        else OrderSide.BUY
    )

    request = LimitOrderRequest(
        symbol=ticker,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.GTC,
        limit_price=(
            take_profit_price
        ),
        extended_hours=False,
        client_order_id=(
            client_order_id
        ),
    )

    return (
        trading_client
        .submit_order(
            order_data=request
        )
    )


# ============================================================
# OUTPUT
# ============================================================

def write_execution_output(
    brody_payload: dict,
    *,
    status: str,
    details: dict | None = None,
) -> None:

    output = {
        "date": brody_payload[
            "date"
        ],
        "generated_at": utc_now(),
        "paper": True,
        "brody_decision": (
            brody_payload[
                "decision"
            ]
        ),
        "status": status,
        "details": (
            details
            or {}
        ),
    }

    dated_path = (
        DATA_DIR
        / (
            f"{brody_payload['date']}"
            ".alpaca-execution.json"
        )
    )

    dated_path.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    EXECUTION_LATEST_PATH.write_text(
        json.dumps(
            output,
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

    logger.info(
        "Starting Alpaca PAPER executor"
    )

    brody = load_brody()

    if (
        brody["decision"]
        == "NO_TRADE"
    ):
        logger.info(
            "Brody selected NO_TRADE. "
            "No order will be submitted."
        )

        write_execution_output(
            brody,
            status="NO_TRADE",
        )

        return

    trade = brody.get(
        "selected_trade"
    )

    if not trade:
        raise RuntimeError(
            "Brody says TRADE but selected_trade is missing."
        )

    ticker = (
        trade[
            "ticker"
        ].upper()
    )

    if ticker == "STRA":
        raise RuntimeError(
            "Trading STRA is prohibited."
        )

    direction = trade[
        "direction"
    ]

    if direction not in {
        "FOR",
        "AGAINST",
    }:
        raise RuntimeError(
            f"Unsupported direction: {direction}"
        )

    key = decision_key(
        brody
    )

    conn = connect_db()

    try:
        existing = conn.execute(
            """
            SELECT *
            FROM executions
            WHERE decision_key = ?
            """,
            (key,),
        ).fetchone()

        if (
            existing
            and existing["status"]
            in {
                "ENTRY_SUBMITTED",
                "PARTIALLY_FILLED",
                "POSITION_OPEN",
                "TAKE_PROFIT_OPEN",
                "CLOSED",
            }
        ):
            logger.info(
                "Decision already executed or active. "
                "Refusing duplicate execution."
            )

            write_execution_output(
                brody,
                status=(
                    "ALREADY_EXECUTED"
                ),
                details={
                    "decision_key": key,
                    "existing_status": (
                        existing[
                            "status"
                        ]
                    ),
                },
            )

            return

        account, asset = preflight(
            ticker,
            direction,
        )

        increment = (
            price_increment(
                asset
            )
        )

        quote = latest_quote(
            ticker
        )

        logger.info(
            "%s quote: bid=%.4f ask=%.4f "
            "spread=%.1f bps",
            ticker,
            quote["bid"],
            quote["ask"],
            quote["spread_bps"],
        )

        if (
            quote["spread_bps"]
            > MAX_SPREAD_BPS
        ):
            raise RuntimeError(
                f"{ticker} spread is "
                f"{quote['spread_bps']:.1f} bps, "
                f"above MAX_SPREAD_BPS="
                f"{MAX_SPREAD_BPS:.1f}."
            )

        entry_limit = (
            entry_limit_from_quote(
                quote,
                direction,
                increment,
            )
        )

        qty = math.floor(
            TRADE_NOTIONAL_USD
            / entry_limit
        )

        if qty < 1:
            raise RuntimeError(
                f"TRADE_NOTIONAL_USD="
                f"${TRADE_NOTIONAL_USD:.2f} "
                f"is too small to purchase one "
                f"whole share of {ticker}."
            )

        actual_notional = (
            qty
            * entry_limit
        )

        prefix = (
            client_id_prefix(
                brody
            )
        )

        entry_client_id = (
            prefix
            + "-entry"
        )

        logger.info(
            "Submitting %s %s | "
            "qty=%d | limit=%.4f | "
            "approx notional=$%.2f",
            direction,
            ticker,
            qty,
            entry_limit,
            actual_notional,
        )

        upsert_execution(
            conn,
            key,
            brody,
            status="PLANNED",
            entry_client_order_id=(
                entry_client_id
            ),
            requested_qty=qty,
            entry_limit_price=(
                entry_limit
            ),
        )

        order = submit_entry(
            ticker=ticker,
            direction=direction,
            qty=qty,
            limit_price=(
                entry_limit
            ),
            client_order_id=(
                entry_client_id
            ),
        )

        logger.info(
            "Entry order submitted: %s",
            order.id,
        )

        upsert_execution(
            conn,
            key,
            brody,
            status="ENTRY_SUBMITTED",
            entry_order_id=str(
                order.id
            ),
        )

        final_order = (
            manage_entry_order(
                order=order,
                ticker=ticker,
                direction=direction,
                increment=increment,
            )
        )

        filled_qty = float(
            final_order.filled_qty
            or 0
        )

        if filled_qty <= 0:
            logger.info(
                "No shares filled. "
                "No position opened."
            )

            upsert_execution(
                conn,
                key,
                brody,
                status="NO_FILL",
                filled_qty=0,
            )

            write_execution_output(
                brody,
                status="NO_FILL",
                details={
                    "ticker": ticker,
                    "direction": (
                        direction
                    ),
                    "entry_order_id": (
                        str(
                            final_order.id
                        )
                    ),
                },
            )

            return

        avg_entry_price = float(
            final_order.filled_avg_price
        )

        # Entry was an integer-share limit order, therefore
        # filled quantity should also be whole shares.
        exit_qty = int(
            round(
                filled_qty
            )
        )

        if exit_qty < 1:
            raise RuntimeError(
                "Filled quantity could not be "
                "converted into a whole-share exit."
            )

        filled_at_dt = (
            final_order.filled_at
            or utc_now_dt()
        )

        if (
            filled_at_dt.tzinfo
            is None
        ):
            filled_at_dt = (
                filled_at_dt.replace(
                    tzinfo=timezone.utc
                )
            )

        exit_deadline_dt = (
            filled_at_dt
            + timedelta(
                days=MAX_HOLD_DAYS
            )
        )

        tp_price = (
            calculate_take_profit(
                avg_entry_price,
                direction,
                increment,
            )
        )

        tp_client_id = (
            prefix
            + "-tp"
        )

        logger.info(
            "%s filled: qty=%s "
            "avg=$%.4f",
            ticker,
            filled_qty,
            avg_entry_price,
        )

        logger.info(
            "Submitting take-profit at %.4f",
            tp_price,
        )

        take_profit_order = (
            submit_take_profit(
                ticker=ticker,
                direction=direction,
                qty=exit_qty,
                take_profit_price=(
                    tp_price
                ),
                client_order_id=(
                    tp_client_id
                ),
            )
        )

        upsert_execution(
            conn,
            key,
            brody,
            status="TAKE_PROFIT_OPEN",
            filled_qty=filled_qty,
            avg_entry_price=(
                avg_entry_price
            ),
            filled_at=(
                filled_at_dt
                .astimezone(
                    timezone.utc
                )
                .isoformat()
            ),
            take_profit_price=(
                tp_price
            ),
            take_profit_client_order_id=(
                tp_client_id
            ),
            take_profit_order_id=str(
                take_profit_order.id
            ),
            exit_deadline=(
                exit_deadline_dt
                .astimezone(
                    timezone.utc
                )
                .isoformat()
            ),
        )

        write_execution_output(
            brody,
            status="POSITION_OPEN",
            details={
                "story_id": (
                    trade[
                        "story_id"
                    ]
                ),
                "ticker": ticker,
                "direction": (
                    direction
                ),
                "qty": exit_qty,
                "avg_entry_price": (
                    avg_entry_price
                ),
                "take_profit_price": (
                    tp_price
                ),
                "entry_order_id": (
                    str(
                        final_order.id
                    )
                ),
                "take_profit_order_id": (
                    str(
                        take_profit_order.id
                    )
                ),
                "filled_at": (
                    filled_at_dt
                    .astimezone(
                        timezone.utc
                    )
                    .isoformat()
                ),
                "exit_deadline": (
                    exit_deadline_dt
                    .astimezone(
                        timezone.utc
                    )
                    .isoformat()
                ),
            },
        )

        logger.info(
            "POSITION OPEN: %s %s",
            direction,
            ticker,
        )

        logger.info(
            "Take profit: %.4f | "
            "deadline: %s",
            tp_price,
            exit_deadline_dt,
        )

    finally:
        conn.close()

    logger.info(
        "Alpaca executor complete"
    )


if __name__ == "__main__":
    main()