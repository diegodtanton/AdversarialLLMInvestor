"""Read-only SQLite lookup: python -m scripts.trade_history --ticker CSTE."""
from __future__ import annotations
import argparse
import csv
import json
import sqlite3
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from .trade_store import DB_PATH


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--view', choices=['fills','positions','decisions','runs','errors'], default='fills')
    parser.add_argument('--ticker')
    parser.add_argument('--since', help='Local calendar date YYYY-MM-DD')
    parser.add_argument('--timezone', default='America/Chicago')
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--format', choices=['table','json','csv'], default='table')
    args=parser.parse_args()
    if not 1 <= args.limit <= 10000:
        parser.error('--limit must be between 1 and 10000')
    if not DB_PATH.exists():
        parser.error('No trading.db; run python -m scripts.upgrade_trading first')
    db=sqlite3.connect(DB_PATH.resolve().as_uri()+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    definitions={
        'fills':("SELECT f.transaction_time,f.ticker,f.side,f.qty,f.price,COALESCE(o.role,'unclassified') AS role,f.order_id FROM fills f LEFT JOIN broker_orders o ON o.broker_order_id=f.order_id",'f.transaction_time','f.ticker'),
        'positions':('SELECT ticker,direction,status,filled_qty,avg_entry_price,filled_at,exit_deadline,exit_price,realized_pnl_usd,CASE WHEN accounting_version=1 THEN realized_return_pct ELSE NULL END AS gross_return_pct,accounting_version,last_error FROM executions','created_at','ticker'),
        'errors':("SELECT ticker,status,last_error,updated_at,entry_client_order_id,entry_order_id FROM executions",'updated_at','ticker'),
        'decisions':('SELECT generated_at,run_id,source,decision,payload_json FROM decisions','generated_at',None),
        'runs':('SELECT * FROM runs','started_at',None),
    }
    query,stamp,ticker=definitions[args.view]
    clauses,params=[],[]
    if args.view=='errors': clauses.append('last_error IS NOT NULL')
    if args.ticker:
        if not ticker: parser.error('--ticker is supported for fills, positions and errors')
        clauses.append(ticker+'=?');params.append(args.ticker.upper())
    if args.since:
        start=datetime.strptime(args.since,'%Y-%m-%d').replace(tzinfo=ZoneInfo(args.timezone)).astimezone(timezone.utc)
        clauses.append(stamp+'>=?');params.append(start.isoformat())
    if clauses: query+=' WHERE '+' AND '.join(clauses)
    query+=' ORDER BY '+stamp+' DESC LIMIT ?';params.append(args.limit)
    try:
        rows=[dict(r) for r in db.execute(query,params)]
    finally:
        db.close()
    # Preserve UTC timestamps in export and render local timestamps in table.
    if args.format=='json': print(json.dumps(rows,indent=2));return
    if not rows: print('No matching records.');return
    if args.format=='csv':
        writer=csv.DictWriter(sys.stdout,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows);return
    for row in rows:
        for key,value in row.items():
            if key.endswith('_at') or key in {'transaction_time','exit_deadline'}:
                if value:
                    row[key]=datetime.fromisoformat(value.replace('Z','+00:00')).astimezone(ZoneInfo(args.timezone)).strftime('%Y-%m-%d %H:%M:%S %Z')
    keys=list(rows[0]);widths={k:min(70,max(len(k),*(len(str(r[k] or '')) for r in rows))) for k in keys}
    print(' | '.join(k.ljust(widths[k]) for k in keys))
    for row in rows: print(' | '.join(str(row[k] if row[k] is not None else '')[:widths[k]].ljust(widths[k]) for k in keys))


if __name__=='__main__': main()
