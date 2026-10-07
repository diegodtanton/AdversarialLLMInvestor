from __future__ import annotations

# Execute a fresh Brody decision in the paper account only.
import json
from . import trade_store as store
from .paper_broker import PaperBroker
from .strategy_config import policy
from .trade_engine import Engine


def main():
    payload = json.loads((store.DATA_DIR / 'brody_latest.json').read_text())
    output = {'date': payload['date'], 'run_id': payload.get('run_id'), 'generated_at': store.utc_now(),
              'paper': True, 'brody_decision': payload['decision']}
    with store.process_lock('broker'), store.connect() as db:
        try:
            engine = Engine(db, PaperBroker(), policy())
            engine.sync_fills()
            output.update(engine.execute(payload))
            engine.sync_fills()
        except Exception as exc:
            output.update(status='ERROR', details={'error': str(exc)})
            raise
        finally:
            output['generated_at'] = store.utc_now()
            store.atomic_json(store.DATA_DIR / 'alpaca_execution_latest.json', output)
            suffix = payload.get('run_id', payload['date'])
            store.atomic_json(store.DATA_DIR / 'runs' / f'{suffix}.execution.json', output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()