from __future__ import annotations

# Run-aware orchestration. Trading maintenance and reporting survive research errors.
import argparse
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from . import trade_store as store


def stage(db, run_id, name, *args):
    started = store.utc_now()
    with db:
        cursor = db.execute('INSERT INTO stage_runs(run_id,stage,started_at,status) VALUES (?,?,?,?)',
                            (run_id, name, started, 'RUNNING'))
    try:
        subprocess.run([sys.executable, '-u', '-m', 'scripts.' + name, *args], cwd=store.ROOT, check=True)
    except Exception as exc:
        with db:
            db.execute('UPDATE stage_runs SET finished_at=?,status=?,error=? WHERE id=?',
                       (store.utc_now(), 'FAILED', str(exc), cursor.lastrowid))
        raise
    else:
        with db:
            db.execute('UPDATE stage_runs SET finished_at=?,status=? WHERE id=?',
                       (store.utc_now(), 'SUCCEEDED', cursor.lastrowid))


def main():
    from dotenv import load_dotenv
    load_dotenv(store.ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_news', '--no-news', dest='no_news', action='store_true',
                        help='Reuse recent enriched news; still reconcile trades and send the report')
    parser.add_argument('--research-only', action='store_true',
                        help='Run research without submitting or canceling any brokerage orders')
    args = parser.parse_args()
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:10]
    os.environ['NEWS_RUN_ID'] = run_id
    os.environ['NEWS_RESEARCH_ONLY'] = '1' if args.research_only else '0'
    errors, outcome = [], 'STARTED'
    with store.process_lock('pipeline'), store.connect() as db:
        with db:
            db.execute('INSERT INTO runs(run_id,started_at,status) VALUES (?,?,?)', (run_id, store.utc_now(), 'RUNNING'))
        try:
            # The broker lock lives in each trading worker. Research never holds it.
            stage(db, run_id, 'position_manager', *(['--sync-only'] if args.research_only else []))
        except Exception as exc:
            errors.append('Position management: ' + str(exc))
        try:
            if not args.no_news:
                stage(db, run_id, 'news_collect')
                stage(db, run_id, 'event_memory')
            payload = json.loads((store.DATA_DIR / 'enriched_latest.json').read_text())
            generated = datetime.fromisoformat(payload['generated_at'].replace('Z', '+00:00'))
            age = (datetime.now(timezone.utc) - generated).total_seconds()
            if not -60 <= age <= int(os.getenv('NEWS_MAX_AGE_HOURS', '36')) * 3600:
                raise RuntimeError('Enriched news is stale; collect fresh news')
            if not payload.get('stories') or not all(s.get('event_context') for s in payload['stories']):
                raise RuntimeError('Missing stories or event context')
            from .paper_broker import PaperBroker
            if not args.research_only and not PaperBroker().clock()['is_open']:
                outcome = 'MARKET_CLOSED'
            else:
                stage(db, run_id, 'gilman')
                stage(db, run_id, 'brody')
                if args.research_only:
                    outcome = 'RESEARCH_ONLY'
                elif errors:
                    outcome = 'EXECUTION_BLOCKED_BY_RECONCILIATION'
                else:
                    stage(db, run_id, 'alpaca_execute')
                    outcome = 'COMPLETE'
        except Exception as exc:
            errors.append('Research/execution: ' + str(exc))
            outcome = 'FAILED'
        finally:
            try:
                stage(db, run_id, 'position_manager', '--sync-only')
            except Exception as exc:
                errors.append('Final fill reconciliation: ' + str(exc))
            with db:
                db.execute('UPDATE runs SET finished_at=?,status=?,details_json=? WHERE run_id=?',
                           (store.utc_now(), 'FAILED' if errors else outcome,
                            store.dumps({'outcome': outcome, 'errors': errors}), run_id))
            # Report failures never prevent portfolio maintenance or hide a run error.
            try:
                stage(db, run_id, 'email_report')
            except Exception as exc:
                errors.append('Email: ' + str(exc))
                with db:
                    db.execute('UPDATE runs SET status=?,details_json=? WHERE run_id=?',
                               ('FAILED', store.dumps({'outcome': outcome, 'errors': errors}), run_id))
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'{run_id}: {outcome}')


if __name__ == "__main__":
    main()