"""Human-readable trading facts, sourced only from the broker fill ledger."""
from __future__ import annotations
from datetime import datetime, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo
from . import trade_store as store
from .trade_engine import dt


def operational_summary(db, run_id=None, now=None, zone='America/Chicago'):
    """Broker fills and open executions only (no research narrative)."""
    now = now or datetime.now(timezone.utc)
    tz = ZoneInfo(zone)
    start = now.astimezone(tz).replace(hour=0,minute=0,second=0,microsecond=0).astimezone(timezone.utc)
    synced = store.meta(db,'fills_synced_at')
    fresh = bool(synced and 0 <= (now-dt(synced)).total_seconds() <= 600)
    stage = db.execute("SELECT status FROM stage_runs WHERE run_id=? AND stage='position_manager' ORDER BY id DESC LIMIT 1", (run_id,)).fetchone() if run_id else None
    if run_id and (not stage or stage['status'] != 'SUCCEEDED'):
        fresh = False
    cutoff = min(now, dt(synced)) if synced else now
    rows = db.execute('''SELECT f.*,o.role FROM fills f LEFT JOIN broker_orders o ON o.broker_order_id=f.order_id
        WHERE transaction_time>=? AND transaction_time<=? ORDER BY transaction_time''',
        (start.isoformat(), cutoff.isoformat())).fetchall()
    groups = {}
    for row in rows:
        g = groups.setdefault(row['order_id'],{'ticker':row['ticker'],'side':row['side'],
                             'role':row['role'] or 'unclassified','qty':Decimal(0),'cash':Decimal(0),'last':None})
        g['qty'] += Decimal(row['qty'])
        g['cash'] += Decimal(row['qty'])*Decimal(row['price'])
        g['last'] = row['transaction_time']
    stamp = cutoff.astimezone(tz).strftime('%Y-%m-%d %I:%M:%S %p %Z')
    if not fresh:
        opening = 'Trade status unconfirmed: broker reconciliation is incomplete or stale.'
    elif groups:
        opening = f'Trades executed today: {len(groups)} order(s) received fills.'
    else:
        opening = 'No trades executed today through the report cutoff.'
    lines = [opening, f'PAPER account. Broker fill cutoff: {stamp}' if synced else 'PAPER account. No successful broker fill sync recorded.']
    for g in groups.values():
        when = dt(g['last']).astimezone(tz).strftime('%I:%M:%S %p %Z')
        lines.append(f"{g['ticker']} — {g['role']} — {g['side']} {g['qty']} shares at average ${g['cash']/g['qty']:.4f}; latest fill {when}.")
    pending = db.execute("SELECT ticker,status,last_error FROM executions WHERE status NOT IN ('CLOSED','NO_FILL') ORDER BY created_at").fetchall()
    for p in pending:
        lines.append(f"{p['ticker']}: {p['status']}" + (f" — needs attention: {p['last_error']}" if p['last_error'] else ''))
    return '\n'.join(lines)


def summary(db, run_id=None, now=None, zone='America/Chicago'):
    lines = [operational_summary(db, run_id, now, zone)]
    if run_id:
        run = db.execute('SELECT status,details_json FROM runs WHERE run_id=?',(run_id,)).fetchone()
        if run:
            import json
            lines.append('Current run: '+run['status'])
            lines.extend(json.loads(run['details_json']).get('errors',[]))
        decision = db.execute("SELECT payload_json FROM decisions WHERE run_id=? AND source='brody' ORDER BY id DESC LIMIT 1",(run_id,)).fetchone()
        if decision:
            import json
            d=json.loads(decision[0])
            lines.append('Brody: '+d['decision']+'. '+d.get('portfolio_reason',''))
    return '\n'.join(lines)
