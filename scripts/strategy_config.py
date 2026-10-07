"""One effective policy for model prompts and deterministic execution."""
from __future__ import annotations
import math
import os
from pathlib import Path


def policy():
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / '.env')
    result = {
        'max_hold_days': int(os.getenv('MAX_HOLD_DAYS', '180')),
        'take_profit_pct': float(os.getenv('TAKE_PROFIT_PCT', '0.10')),
        'trade_notional_usd': float(os.getenv('TRADE_NOTIONAL_USD', '1000')),
    }
    if not 1 <= result['max_hold_days'] <= 3650:
        raise ValueError('MAX_HOLD_DAYS must be between 1 and 3650')
    if not math.isfinite(result['take_profit_pct']) or not 0 < result['take_profit_pct'] < 1:
        raise ValueError('TAKE_PROFIT_PCT must be between 0 and 1')
    if not math.isfinite(result['trade_notional_usd']) or result['trade_notional_usd'] <= 0:
        raise ValueError('TRADE_NOTIONAL_USD must be positive')
    return result
