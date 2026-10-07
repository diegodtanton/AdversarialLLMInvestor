"""SQLite audit log shared by research, execution, reporting and lookup.

The legacy executions table is migrated in place. JSON files remain exports.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
DB_PATH = DATA_DIR / 'trading.db'


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def dumps(value):
    return json.dumps(value, ensure_ascii=False, default=str, allow_nan=False)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(json.dumps(value, indent=2, ensure_ascii=False, default=str, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@contextlib.contextmanager
def process_lock(name):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with (DATA_DIR / (name + '.lock')).open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another {name} process is running') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def connect(path=None):
    path = Path(path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.row_factory = sqlite3.Row
    db.executescript('''
        PRAGMA journal_mode=WAL;
        PRAGMA foreign_keys=ON;
        CREATE TABLE IF NOT EXISTS executions (
            decision_key TEXT PRIMARY KEY, trade_date TEXT NOT NULL,
            story_id TEXT NOT NULL, ticker TEXT NOT NULL, direction TEXT NOT NULL,
            status TEXT NOT NULL, requested_notional REAL,
            entry_client_order_id TEXT, entry_order_id TEXT, requested_qty INTEGER,
            entry_limit_price REAL, filled_qty REAL, avg_entry_price REAL, filled_at TEXT,
            take_profit_price REAL, take_profit_client_order_id TEXT, take_profit_order_id TEXT,
            exit_deadline TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS audit_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, finished_at TEXT,
            status TEXT NOT NULL, details_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS stage_runs (
            id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, stage TEXT NOT NULL,
            started_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL, error TEXT
        );
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, source TEXT NOT NULL,
            generated_at TEXT NOT NULL, decision TEXT, payload_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS broker_orders (
            client_order_id TEXT PRIMARY KEY, decision_key TEXT NOT NULL, role TEXT NOT NULL,
            broker_order_id TEXT UNIQUE, status TEXT NOT NULL,
            request_json TEXT NOT NULL, response_json TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS fills (
            activity_id TEXT PRIMARY KEY, order_id TEXT NOT NULL, ticker TEXT NOT NULL,
            side TEXT NOT NULL, qty TEXT NOT NULL, price TEXT NOT NULL,
            transaction_time TEXT NOT NULL, payload_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS fills_time ON fills(transaction_time);
        CREATE INDEX IF NOT EXISTS fills_order ON fills(order_id);
        CREATE TABLE IF NOT EXISTS execution_events (
            id INTEGER PRIMARY KEY, decision_key TEXT NOT NULL, at TEXT NOT NULL,
            event TEXT NOT NULL, details_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS tool_calls (
            id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, call_id TEXT,
            tool TEXT NOT NULL, arguments_json TEXT NOT NULL,
            result_json TEXT NOT NULL, at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS reports (
            report_key TEXT PRIMARY KEY, status TEXT NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL, error TEXT
        );
    ''')
    columns = {r['name'] for r in db.execute('PRAGMA table_info(executions)')}
    additions = {
        'exit_client_order_id': 'TEXT', 'exit_order_id': 'TEXT', 'closed_at': 'TEXT',
        'exit_price': 'REAL', 'exit_reason': 'TEXT', 'realized_return_pct': 'REAL',
        'realized_pnl_usd': 'TEXT', 'run_id': 'TEXT', 'event_id': 'TEXT',
        'hold_days': 'INTEGER', 'take_profit_pct': 'REAL', 'last_error': 'TEXT',
        'accounting_version': 'INTEGER NOT NULL DEFAULT 0',
    }
    for name, kind in additions.items():
        if name not in columns:
            db.execute(f'ALTER TABLE executions ADD COLUMN {name} {kind}')
    db.execute('CREATE INDEX IF NOT EXISTS executions_status ON executions(status)')
    db.execute("INSERT OR REPLACE INTO audit_meta VALUES ('schema_version','1')")
    db.commit()
    return db


def run_id():
    return os.environ.get('NEWS_RUN_ID') or 'standalone-' + uuid.uuid4().hex


def record_decision(source, payload):
    with connect() as db:
        db.execute('INSERT INTO decisions(run_id,source,generated_at,decision,payload_json) VALUES (?,?,?,?,?)',
                   (payload.setdefault('run_id', run_id()), source, payload.get('generated_at', utc_now()),
                    payload.get('decision'), dumps(payload)))


def record_tool(call_id, name, arguments, result):
    with connect() as db:
        db.execute('INSERT INTO tool_calls(run_id,call_id,tool,arguments_json,result_json,at) VALUES (?,?,?,?,?,?)',
                   (os.environ.get('NEWS_RUN_ID', 'standalone'), call_id, name, dumps(arguments), dumps(result), utc_now()))


def update_execution(db, key, **fields):
    allowed = {r['name'] for r in db.execute('PRAGMA table_info(executions)')}
    if not fields.keys() <= allowed:
        raise ValueError('Unknown execution field')
    fields['updated_at'] = utc_now()
    with db:
        db.execute('UPDATE executions SET ' + ','.join(f'{k}=?' for k in fields) + ' WHERE decision_key=?',
                   (*fields.values(), key))
        db.execute('INSERT INTO execution_events(decision_key,at,event,details_json) VALUES (?,?,?,?)',
                   (key, utc_now(), fields.get('status', 'UPDATE'), dumps(fields)))


def meta(db, key, value=None):
    if value is not None:
        with db:
            db.execute('INSERT OR REPLACE INTO audit_meta VALUES (?,?)', (key, str(value)))
        return value
    row = db.execute('SELECT value FROM audit_meta WHERE key=?', (key,)).fetchone()
    return row[0] if row else None


def claim_report(key):
    with connect() as db:
        # SUBMITTING is deliberately not automatically retried after a crash.
        row = db.execute('SELECT status FROM reports WHERE report_key=?', (key,)).fetchone()
        if row and row['status'] in {'SENT', 'SUBMITTING', 'UNKNOWN'}:
            return False
        db.execute('INSERT OR REPLACE INTO reports VALUES (?,?,?,?,NULL)',
                   (key, 'SUBMITTING', utc_now(), utc_now()))
    return True


def report_status(key, status, error=None):
    with connect() as db:
        db.execute('UPDATE reports SET status=?,updated_at=?,error=? WHERE report_key=?',
                   (status, utc_now(), error, key))
