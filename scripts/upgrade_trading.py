"""Local-only upgrade: back up SQLite/.env and configure the requested models.

Never contacts Alpaca/OpenAI and never places orders. Existing deadlines are
preserved unless --extend-open-positions is explicitly supplied.
"""
from __future__ import annotations
import argparse
import os
import shutil
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from . import trade_store as store


def configure_env(path):
    values={'MAX_HOLD_DAYS':'180','GILMAN_MODEL':'gpt-6-astra','BRODY_MODEL':'gpt-6-astra','EVENT_MODEL':'gpt-6-sol'}
    lines=path.read_text().splitlines() if path.exists() else []
    result=[]
    seen=set()
    for line in lines:
        stripped=line.strip()
        candidate=stripped[7:].lstrip() if stripped.startswith('export ') else stripped
        key=candidate.split('=',1)[0].strip() if '=' in candidate and not candidate.startswith('#') else None
        if key in values:
            if key not in seen:
                result.append(f'{key}={values[key]}');seen.add(key)
        else:
            result.append(line)
    result += [f'{k}={v}' for k,v in values.items() if k not in seen]
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.env-upgrade-')
    try:
        with os.fdopen(fd,'w') as f:
            f.write('\n'.join(result)+'\n');f.flush();os.fsync(f.fileno())
        os.chmod(tmp,0o600)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extend-open-positions',action='store_true',
                        help='Extend existing open-position deadlines to at least 180 days after their recorded entry fill')
    args=parser.parse_args()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    with store.process_lock('pipeline'),store.process_lock('broker'):
        backup=store.DATA_DIR/'backups'/stamp
        backup.mkdir(parents=True,exist_ok=False)
        os.chmod(backup,0o700)
        if store.DB_PATH.exists():
            # SQLite backup includes committed WAL pages; copying only .db does not.
            with sqlite3.connect(store.DB_PATH) as source,sqlite3.connect(backup/'trading.db') as target:
                source.backup(target)
            os.chmod(backup/'trading.db',0o600)
        env=store.ROOT/'.env'
        if env.exists():
            shutil.copyfile(env,backup/'.env')
            os.chmod(backup/'.env',0o600)
        with store.connect() as db:
            if args.extend_open_positions:
                for row in db.execute("SELECT * FROM executions WHERE status NOT IN ('CLOSED','NO_FILL') AND filled_at IS NOT NULL").fetchall():
                    first=datetime.fromisoformat(row['filled_at'].replace('Z','+00:00'))
                    if first.tzinfo is None:
                        raise ValueError('Existing fill time lacks timezone; deadline not changed')
                    proposed=first+timedelta(days=180)
                    previous=datetime.fromisoformat(row['exit_deadline'].replace('Z','+00:00')) if row['exit_deadline'] else None
                    if previous and previous.tzinfo is None:
                        raise ValueError('Existing deadline lacks timezone; deadline not changed')
                    # Already submitted exits must be reconciled, never canceled by migration.
                    if row['exit_client_order_id'] or row['exit_order_id']:
                        print(f"Skipped {row['ticker']}: an exit attempt already exists")
                        continue
                    deadline=max(previous,proposed) if previous else proposed
                    store.update_execution(db,row['decision_key'],hold_days=180,exit_deadline=deadline.isoformat())
        configure_env(env)
    print('Upgrade complete: 180 days; Gilman/Brody gpt-6-astra; Event Memory gpt-6-sol.')
    print('Backups:',backup)
    print('Existing deadlines '+('extended where eligible.' if args.extend_open_positions else 'preserved.'))
    print('Shell/launchd environment overrides .env; remove conflicting model or MAX_HOLD_DAYS overrides.')


if __name__=='__main__':main()
