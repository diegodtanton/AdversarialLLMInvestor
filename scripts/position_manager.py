from __future__ import annotations

# Reconcile every nonterminal paper execution independently of research.
import argparse
from . import trade_store as store
from .paper_broker import PaperBroker
from .strategy_config import policy
from .trade_engine import Engine, gross_return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sync-only', action='store_true', help='Reconcile broker records; never submit or cancel orders')
    parser.add_argument('--full-history', action='store_true', help='Fetch all available fill history')
    args = parser.parse_args()
    output = {'run_id': __import__('os').environ.get('NEWS_RUN_ID'), 'generated_at': store.utc_now(),
              'actions': [], 'sync_ok': False}
    with store.process_lock('broker'), store.connect() as db:
        try:
            engine = Engine(db, PaperBroker(), policy())
            engine.sync_fills(full=args.full_history)
            if args.sync_only:
                output['actions'] = engine.refresh_all()
            else:
                output['actions'] = engine.manage_all()
                engine.sync_fills()
            output['sync_ok'] = True
        except Exception as exc:
            output['error'] = str(exc)
            raise
        finally:
            actions = output['actions']
            output.update(generated_at=store.utc_now(), managed_count=len(actions),
                          closed_count=sum(a['action'] == 'CLOSED' for a in actions),
                          holding_count=sum(a['action'] == 'HOLD' for a in actions),
                          pending_exit_count=sum(a['action'] in {'TIME_EXIT_SUBMITTED','EXIT_DUE_MARKET_CLOSED'} for a in actions),
                          error_count=sum(a['action'] == 'ERROR' for a in actions))
            store.atomic_json(store.DATA_DIR / 'position_manager_latest.json', output)
    if output['error_count']:
        raise RuntimeError('One or more executions require reconciliation; see trade lookup')


if __name__ == "__main__":
    main()