from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

INPUT_PATH = DATA_DIR / "gilman_trades_latest.json"
OUTPUT_PATH = DATA_DIR / "market_context_latest.json"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_number(value):
    if value is None:
        return None

    try:
        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return None

        return value

    except (TypeError, ValueError):
        return None


def pct(value):
    value = clean_number(value)

    if value is None:
        return None

    return round(value * 100, 3)


def return_pct(series: pd.Series, periods: int):
    series = series.dropna()

    if len(series) <= periods:
        return None

    old = float(series.iloc[-periods - 1])
    new = float(series.iloc[-1])

    if old == 0:
        return None

    return round(
        (new / old - 1) * 100,
        3,
    )


def sma_distance(close: pd.Series, window: int):
    close = close.dropna()

    if len(close) < window:
        return None

    current = float(close.iloc[-1])
    average = float(close.iloc[-window:].mean())

    if average == 0:
        return None

    return round(
        (current / average - 1) * 100,
        3,
    )


def realized_volatility(close: pd.Series, window: int = 20):
    close = close.dropna()

    returns = close.pct_change().dropna()

    if len(returns) < window:
        return None

    vol = (
        returns.iloc[-window:].std()
        * math.sqrt(252)
    )

    return round(
        float(vol) * 100,
        3,
    )


def get_daily_context(ticker: str) -> dict:
    data = yf.download(
        ticker,
        period="1y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
        multi_level_index=False,
    )

    if data.empty:
        raise RuntimeError(
            f"No daily yfinance data for {ticker}"
        )

    close = data["Close"].dropna()
    volume = data["Volume"].dropna()

    current = float(close.iloc[-1])

    high_52w = float(
        data["High"].dropna().max()
    )

    low_52w = float(
        data["Low"].dropna().min()
    )

    if high_52w > low_52w:
        range_position = (
            (current - low_52w)
            / (high_52w - low_52w)
        ) * 100
    else:
        range_position = None

    volume_ratio = None

    if len(volume) >= 21:
        current_volume = float(volume.iloc[-1])
        normal_volume = float(
            volume.iloc[-21:-1].mean()
        )

        if normal_volume:
            volume_ratio = (
                current_volume
                / normal_volume
            )

    gap_pct = None

    if len(data) >= 2:
        previous_close = float(
            data["Close"].iloc[-2]
        )

        today_open = float(
            data["Open"].iloc[-1]
        )

        if previous_close:
            gap_pct = (
                today_open
                / previous_close
                - 1
            ) * 100

    return {
        "last_price": round(current, 4),

        "returns_pct": {
            "1d": return_pct(close, 1),
            "5d": return_pct(close, 5),
            "20d": return_pct(close, 20),
            "60d": return_pct(close, 60),
            "120d": return_pct(close, 120),
            "252d": return_pct(close, 252),
        },

        "gap_today_pct": (
            round(gap_pct, 3)
            if gap_pct is not None
            else None
        ),

        "volume_vs_20d_average": (
            round(volume_ratio, 3)
            if volume_ratio is not None
            else None
        ),

        "realized_volatility_20d_pct": (
            realized_volatility(
                close,
                20,
            )
        ),

        "moving_average_distance_pct": {
            "20d": sma_distance(
                close,
                20,
            ),
            "50d": sma_distance(
                close,
                50,
            ),
            "200d": sma_distance(
                close,
                200,
            ),
        },

        "52_week": {
            "high": round(
                high_52w,
                4,
            ),
            "low": round(
                low_52w,
                4,
            ),
            "below_high_pct": round(
                (
                    current
                    / high_52w
                    - 1
                )
                * 100,
                3,
            ),
            "range_position_pct": (
                round(
                    range_position,
                    3,
                )
                if range_position is not None
                else None
            ),
        },
    }


def get_intraday_context(ticker: str) -> dict:
    try:
        data = yf.download(
            ticker,
            period="5d",
            interval="5m",
            auto_adjust=True,
            prepost=False,
            progress=False,
            threads=False,
            multi_level_index=False,
        )

        if data.empty:
            return {}

        # Use the most recent trading session in the dataset.
        last_date = data.index[-1].date()

        today = data[
            data.index.date == last_date
        ]

        if today.empty:
            return {}

        session_open = float(
            today["Open"].iloc[0]
        )

        latest = float(
            today["Close"].iloc[-1]
        )

        high = float(
            today["High"].max()
        )

        low = float(
            today["Low"].min()
        )

        move = None

        if session_open:
            move = (
                latest
                / session_open
                - 1
            ) * 100

        return {
            "session_date": (
                last_date.isoformat()
            ),
            "session_open": round(
                session_open,
                4,
            ),
            "latest_price": round(
                latest,
                4,
            ),
            "session_high": round(
                high,
                4,
            ),
            "session_low": round(
                low,
                4,
            ),
            "from_open_pct": (
                round(move, 3)
                if move is not None
                else None
            ),
        }

    except Exception:
        logger.exception(
            "Intraday data failed for %s",
            ticker,
        )

        return {}


INFO_FIELDS = [
    "sector",
    "industry",
    "marketCap",
    "enterpriseValue",
    "trailingPE",
    "forwardPE",
    "pegRatio",
    "priceToSalesTrailing12Months",
    "priceToBook",
    "enterpriseToRevenue",
    "enterpriseToEbitda",
    "grossMargins",
    "operatingMargins",
    "profitMargins",
    "revenueGrowth",
    "earningsGrowth",
    "returnOnEquity",
    "debtToEquity",
    "freeCashflow",
    "operatingCashflow",
    "totalCash",
    "totalDebt",
    "beta",
    "shortPercentOfFloat",
    "sharesShort",
    "sharesShortPriorMonth",
    "heldPercentInstitutions",
    "heldPercentInsiders",
    "targetLowPrice",
    "targetMeanPrice",
    "targetMedianPrice",
    "targetHighPrice",
    "recommendationMean",
    "numberOfAnalystOpinions",
]


PERCENT_INFO_FIELDS = {
    "grossMargins",
    "operatingMargins",
    "profitMargins",
    "revenueGrowth",
    "earningsGrowth",
    "returnOnEquity",
    "shortPercentOfFloat",
    "heldPercentInstitutions",
    "heldPercentInsiders",
}


def get_fundamentals(ticker: str) -> dict:
    obj = yf.Ticker(ticker)

    try:
        info = obj.get_info()
    except Exception:
        logger.exception(
            "Ticker.info failed for %s",
            ticker,
        )
        info = {}

    result = {}

    for field in INFO_FIELDS:
        value = info.get(field)

        if field in {
            "sector",
            "industry",
        }:
            result[field] = value
            continue

        if field in PERCENT_INFO_FIELDS:
            result[field] = pct(value)
        else:
            result[field] = clean_number(
                value
            )

    return result


def get_spy_returns() -> dict:
    data = yf.download(
        "SPY",
        period="1y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
        multi_level_index=False,
    )

    if data.empty:
        return {}

    close = data["Close"].dropna()

    return {
        "1d": return_pct(close, 1),
        "5d": return_pct(close, 5),
        "20d": return_pct(close, 20),
        "60d": return_pct(close, 60),
    }


def relative_return(
    stock_return,
    benchmark_return,
):
    if (
        stock_return is None
        or benchmark_return is None
    ):
        return None

    return round(
        stock_return - benchmark_return,
        3,
    )


def build_context(
    trade: dict,
    spy_returns: dict,
) -> dict:

    ticker = trade["ticker"].upper()

    logger.info(
        "Collecting market context for %s",
        ticker,
    )

    daily = get_daily_context(
        ticker
    )

    intraday = get_intraday_context(
        ticker
    )

    fundamentals = get_fundamentals(
        ticker
    )

    stock_returns = daily[
        "returns_pct"
    ]

    relative = {}

    for horizon in [
        "1d",
        "5d",
        "20d",
        "60d",
    ]:
        relative[horizon] = (
            relative_return(
                stock_returns.get(
                    horizon
                ),
                spy_returns.get(
                    horizon
                ),
            )
        )

    return {
        "gilman_trade": trade,

        "market_data": {
            "daily": daily,
            "intraday": intraday,
            "relative_to_spy_pct": (
                relative
            ),
        },

        "fundamentals": (
            fundamentals
        ),
    }


def main():
    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        gilman = json.load(f)

    trades = gilman.get(
        "trades",
        []
    )

    logger.info(
        "Gilman supplied %d trades",
        len(trades),
    )

    spy_returns = get_spy_returns()

    candidates = []

    for trade in trades:
        try:
            candidates.append(
                build_context(
                    trade,
                    spy_returns,
                )
            )

        except Exception as exc:
            logger.exception(
                "Market context failed for %s",
                trade.get(
                    "ticker"
                ),
            )

            candidates.append(
                {
                    "gilman_trade": trade,
                    "market_data_error": (
                        str(exc)
                    ),
                }
            )

    output = {
        "date": gilman["date"],
        "generated_at": utc_now(),
        "candidate_count": len(
            candidates
        ),
        "benchmark": {
            "ticker": "SPY",
            "returns_pct": (
                spy_returns
            ),
        },
        "candidates": candidates,
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


if __name__ == "__main__":
    main()