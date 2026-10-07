"""Deterministic paper execution with persisted intent and broker reconciliation.

New entries use one fixed DAY limit. This deliberately removes automatic price
chasing and replacement races. Legacy replacements are followed during recovery.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR

from . import trade_store as store

TERMINAL = {'filled', 'canceled', 'cancelled', 'expired', 'rejected'}
CLOSED = {'CLOSED', 'NO_FILL'}


def dt(value):
    if not value:
        return None
    result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Broker and decision timestamps must be timezone-aware')
    return result.astimezone(timezone.utc)


def decimal(value):
    result = Decimal(str(value or '0'))
    if not result.is_finite():
        raise ValueError('Nonfinite financial value')
    return result


def order_price(value, *, up):
    value = decimal(value)
    increment = Decimal('0.01') if value >= 1 else Decimal('0.0001')
    return str((value / increment).to_integral_value(rounding=ROUND_CEILING if up else ROUND_FLOOR) * increment)


def gross_return(direction, entry, exit_price):
    entry, exit_price = decimal(entry), decimal(exit_price)
    if entry <= 0:
        return None
    sign = 1 if direction == 'FOR' else -1
    return float(sign * (exit_price - entry) / entry * 100)


class Engine:
    def __init__(self, db, broker, policy, *, now=None, sleep=time.sleep):
        self.db, self.broker, self.policy = db, broker, policy
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.sleep = sleep
        self.account = broker.account()
        scope = 'paper:' + str(self.account['id'])
        old = store.meta(db, 'account_scope')
        if old and old != scope:
            raise RuntimeError('trading.db belongs to a different paper account')
        store.meta(db, 'account_scope', scope)

    def sync_fills(self, full=False):
        started = self.now()
        params = {'direction': 'asc', 'page_size': 100, 'until': started.isoformat()}
        previous = store.meta(self.db, 'fills_synced_at')
        if previous and not full:
            params['after'] = (dt(previous) - timedelta(days=7)).isoformat()
        seen_tokens = set()
        for _ in range(10000):
            rows = self.broker.fills(**params)
            if not isinstance(rows, list):
                raise RuntimeError('Malformed broker activities response')
            with self.db:
                for row in rows:
                    if row.get('activity_type') != 'FILL':
                        raise RuntimeError('Unexpected non-fill activity')
                    stamp = dt(row['transaction_time']).isoformat()
                    self.db.execute('INSERT OR REPLACE INTO fills VALUES (?,?,?,?,?,?,?,?)',
                                    (row['id'], row['order_id'], row['symbol'], row['side'],
                                     str(decimal(row['qty'])), str(decimal(row['price'])), stamp, store.dumps(row)))
            if len(rows) < 100:
                store.meta(self.db, 'fills_synced_at', started.isoformat())
                return
            token = rows[-1]['id']
            if token in seen_tokens:
                raise RuntimeError('Broker fill pagination repeated a cursor')
            seen_tokens.add(token)
            params['page_token'] = token
        raise RuntimeError('Broker fill pagination limit reached; sync incomplete')

    def row(self, key):
        return dict(self.db.execute('SELECT * FROM executions WHERE decision_key=?', (key,)).fetchone())

    def save_order(self, key, role, order, request=None):
        client = order['client_order_id']
        with self.db:
            self.db.execute('''INSERT INTO broker_orders VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(client_order_id) DO UPDATE SET
                broker_order_id=excluded.broker_order_id,status=excluded.status,
                response_json=excluded.response_json,updated_at=excluded.updated_at''',
                (client, key, role, order['id'], order['status'], store.dumps(request or {}),
                 store.dumps(order), store.utc_now(), store.utc_now()))
        prefix = {'entry': 'entry', 'tp': 'take_profit', 'exit': 'exit'}[role]
        store.update_execution(self.db, key, **{prefix + '_order_id': order['id'],
                                               prefix + '_client_order_id': client})
        return order

    def recover(self, row, role):
        prefix = {'entry': 'entry', 'tp': 'take_profit', 'exit': 'exit'}[role]
        oid, cid = row.get(prefix + '_order_id'), row.get(prefix + '_client_order_id')
        if not oid and not cid:
            return None
        order = self.broker.order(order_id=oid, client_id=cid)
        if order is None:
            raise RuntimeError(f'{role} outcome unresolved: saved broker/client ID not found')
        seen = set()
        while True:
            if order['id'] in seen:
                raise RuntimeError('Cyclic broker replacement chain')
            seen.add(order['id'])
            self.save_order(row['decision_key'], role, order)
            replacement = order.get('replaced_by')
            if not replacement:
                if order['status'] == 'replaced':
                    raise RuntimeError('Replaced order has no replacement ID yet')
                return order
            order = self.broker.order(order_id=replacement)
            if order is None:
                raise RuntimeError('Replacement order not found')

    def submit(self, key, role, request):
        attempt = self.db.execute('SELECT count(*) FROM broker_orders WHERE decision_key=? AND role=?',
                                  (key, role)).fetchone()[0]
        client_id = 'news-' + hashlib.sha256(key.encode()).hexdigest()[:24] + f'-{role}-{attempt}'
        request = dict(request, client_order_id=client_id)
        prefix = {'entry': 'entry', 'tp': 'take_profit', 'exit': 'exit'}[role]
        with self.db:
            self.db.execute('INSERT INTO broker_orders VALUES (?,?,?,?,?,?,?,?,?)',
                            (client_id, key, role, None, 'SUBMITTING', store.dumps(request), None,
                             store.utc_now(), store.utc_now()))
            self.db.execute(f'UPDATE executions SET {prefix}_client_order_id=?,{prefix}_order_id=NULL WHERE decision_key=?',
                            (client_id, key))
        # On any exception the intent remains. Recovery looks up this SAME ID;
        # it never guesses a new ID or reports that no order exists.
        order = self.broker.submit(request)
        return self.save_order(key, role, order, request)

    def cancel_terminal(self, key, role, order):
        if order['status'] in TERMINAL:
            return order
        try:
            self.broker.cancel(order['id'])
        except Exception:
            # Cancel rejection can be a concurrent fill; establish its outcome.
            current = self.recover(self.row(key), role)
            if current['status'] in TERMINAL:
                return current
            raise
        deadline = time.monotonic() + int(os.getenv('CANCEL_TIMEOUT_SECONDS', '30'))
        while True:
            current = self.recover(self.row(key), role)
            if current['status'] in TERMINAL:
                return current
            if time.monotonic() >= deadline:
                break
            self.sleep(1)
        raise RuntimeError('Cancellation unconfirmed; remaining entry/exit may still fill')

    def quantities(self, key):
        rows = self.db.execute('''SELECT f.*,o.role FROM fills f JOIN broker_orders o
            ON o.broker_order_id=f.order_id WHERE o.decision_key=? ORDER BY transaction_time''', (key,)).fetchall()
        entry_qty = exit_qty = entry_cash = exit_cash = Decimal(0)
        first = last = None
        for row in rows:
            q, cash = decimal(row['qty']), decimal(row['qty']) * decimal(row['price'])
            if row['role'] == 'entry':
                entry_qty += q
                entry_cash += cash
                first = first or row['transaction_time']
            else:
                exit_qty += q
                exit_cash += cash
                last = row['transaction_time']
        return entry_qty, exit_qty, entry_cash, exit_cash, first, last

    def persist_entry_fills(self, key):
        row = self.row(key)
        eq, _, ec, _, first, _ = self.quantities(key)
        if eq > 0:
            deadline = row['exit_deadline'] or (dt(first) + timedelta(days=row['hold_days'] or self.policy['max_hold_days'])).isoformat()
            store.update_execution(self.db, key, filled_qty=float(eq), avg_entry_price=float(ec / eq),
                                   filled_at=first, exit_deadline=deadline)

    def reconcile_position(self, key, *, allow_cancel=True):
        row = self.row(key)
        eq, xq, ec, xc, first, last = self.quantities(key)
        if eq <= 0:
            raise RuntimeError('Entry fill activities not yet available; retry reconciliation')
        if xq > eq:
            raise RuntimeError('Exit fills exceed owned entry quantity')
        deadline = row['exit_deadline'] or (dt(first) + timedelta(days=row['hold_days'] or self.policy['max_hold_days'])).isoformat()
        store.update_execution(self.db, key, filled_qty=float(eq), avg_entry_price=float(ec / eq),
                               filled_at=first, exit_deadline=deadline)
        positions = {p['symbol']: p for p in self.broker.positions()}
        position = positions.get(row['ticker'])
        actual = decimal(position['qty']) if position else Decimal(0)
        expected = (eq - xq) * (1 if row['direction'] == 'FOR' else -1)
        if actual != expected:
            raise RuntimeError(f'Position mismatch: broker={actual}, strategy={expected}; manual activity or delayed fills')
        if xq == eq:
            # Clean up outstanding exits BEFORE marking CLOSED. A cancellation
            # failure must leave the row in the reconciliation workload.
            for role in ('tp', 'exit'):
                current = self.recover(self.row(key), role)
                if current and current['status'] not in TERMINAL:
                    if not allow_cancel:
                        raise RuntimeError('Filled position still has an open exit; execution maintenance required')
                    self.cancel_terminal(key, role, current)
            self.sync_fills()
            eq2, xq2, _, _, _, _ = self.quantities(key)
            actual_positions = {p['symbol']: p for p in self.broker.positions()}
            if eq2 != eq or xq2 != xq or row['ticker'] in actual_positions:
                raise RuntimeError('Broker state changed during closing reconciliation')
            pnl = (xc - ec) * (1 if row['direction'] == 'FOR' else -1)
            store.update_execution(self.db, key, status='CLOSED', closed_at=last,
                                   exit_price=float(xc / xq), realized_pnl_usd=str(pnl),
                                   realized_return_pct=float(pnl / ec * 100),
                                   exit_reason='BROKER_FILLS', accounting_version=1, last_error=None)
            return None
        return position

    def manage(self, key, *, wait_entry=False):
        row = self.row(key)
        if row['status'] in CLOSED:
            return {'ticker': row['ticker'], 'action': row['status']}
        entry = self.recover(row, 'entry')
        if entry is None:
            raise RuntimeError('Execution has no entry order intent')
        if entry['status'] not in TERMINAL:
            if decimal(entry.get('filled_qty')) > 0:
                self.sync_fills()
                self.persist_entry_fills(key)
            if wait_entry:
                stop = time.monotonic() + int(os.getenv('MAX_ENTRY_WAIT_MINUTES', '180')) * 60
                while entry['status'] not in TERMINAL and time.monotonic() < stop:
                    if not self.broker.clock()['is_open']:
                        break
                    self.sleep(max(1, int(os.getenv('ENTRY_POLL_SECONDS', '30'))))
                    entry = self.recover(self.row(key), 'entry')
                    if decimal(entry.get('filled_qty')) > 0:
                        self.sync_fills()
                        self.persist_entry_fills(key)
            # Interrupted/overnight entries are canceled and reconciled before exits.
            entry = self.cancel_terminal(key, 'entry', entry)
        self.sync_fills()
        eq, _, _, _, _, _ = self.quantities(key)
        if eq == 0:
            # A replacement ancestor may have fills not present in the latest order.
            known = self.db.execute('SELECT response_json FROM broker_orders WHERE decision_key=? AND role=?',
                                    (key, 'entry')).fetchall()
            if any(decimal(json.loads(r[0] or '{}').get('filled_qty')) > 0 for r in known):
                raise RuntimeError('Broker reports a fill but activity ledger has not caught up')
            store.update_execution(self.db, key, status='NO_FILL', filled_qty=0, last_error=None)
            return {'ticker': row['ticker'], 'action': 'NO_FILL'}
        # Register legacy exit orders before looking up their fill activities.
        tp = self.recover(self.row(key), 'tp')
        exit_order = self.recover(self.row(key), 'exit')
        position = self.reconcile_position(key)
        if position is None:
            for role, order in [('tp', tp), ('exit', exit_order)]:
                if order and order['status'] not in TERMINAL:
                    self.cancel_terminal(key, role, order)
            return {'ticker': row['ticker'], 'action': 'CLOSED'}
        row = self.row(key)
        if exit_order and exit_order['status'] not in TERMINAL:
            return {'ticker': row['ticker'], 'action': 'TIME_EXIT_SUBMITTED'}
        if self.now() >= dt(row['exit_deadline']):
            if tp and tp['status'] not in TERMINAL:
                self.cancel_terminal(key, 'tp', tp)
                self.sync_fills()
                position = self.reconcile_position(key)
                if position is None:
                    return {'ticker': row['ticker'], 'action': 'CLOSED'}
            if not self.broker.clock()['is_open']:
                return {'ticker': row['ticker'], 'action': 'EXIT_DUE_MARKET_CLOSED'}
            order = self.submit(key, 'exit', {'symbol': row['ticker'], 'qty': str(abs(decimal(position['qty']))),
                                'side': 'sell' if row['direction'] == 'FOR' else 'buy',
                                'type': 'market', 'time_in_force': 'day', 'extended_hours': False})
            store.update_execution(self.db, key, status='TIME_EXIT_SUBMITTED', last_error=None)
            if order['status'] in TERMINAL:
                self.sync_fills()
                self.reconcile_position(key)
            return {'ticker': row['ticker'], 'action': self.row(key)['status']}
        qty = abs(decimal(position['qty']))
        if tp and tp['status'] not in TERMINAL:
            remaining = decimal(tp['qty']) - decimal(tp.get('filled_qty'))
            if remaining != qty:
                raise RuntimeError('Take-profit remaining quantity does not match owned position')
        else:
            target = decimal(row['avg_entry_price']) * (1 + decimal(row['take_profit_pct'] or self.policy['take_profit_pct']) *
                       (1 if row['direction'] == 'FOR' else -1))
            price = order_price(target, up=row['direction'] == 'FOR')
            self.submit(key, 'tp', {'symbol': row['ticker'], 'qty': str(qty),
                        'side': 'sell' if row['direction'] == 'FOR' else 'buy', 'type': 'limit',
                        'time_in_force': 'gtc', 'limit_price': price, 'extended_hours': False})
            store.update_execution(self.db, key, take_profit_price=float(price))
        store.update_execution(self.db, key, status='TAKE_PROFIT_OPEN', last_error=None)
        return {'ticker': row['ticker'], 'action': 'HOLD', 'exit_deadline': row['exit_deadline']}

    def refresh_all(self):
        """Read-only broker reconciliation, including fills arriving after submission."""
        actions = []
        rows = self.db.execute("SELECT * FROM executions WHERE status NOT IN ('CLOSED','NO_FILL')").fetchall()
        for row in rows:
            key = row['decision_key']
            try:
                entry = self.recover(dict(row), 'entry')
                tp = self.recover(self.row(key), 'tp')
                exit_order = self.recover(self.row(key), 'exit')
                if entry is None:
                    raise RuntimeError('Entry intent is incomplete; execution maintenance required')
                eq, _, _, _, _, _ = self.quantities(key)
                if eq > 0:
                    position = self.reconcile_position(key, allow_cancel=False)
                    if position:
                        if exit_order and exit_order['status'] not in TERMINAL:
                            store.update_execution(self.db, key, status='TIME_EXIT_SUBMITTED', last_error=None)
                        elif tp and tp['status'] not in TERMINAL and entry['status'] in TERMINAL:
                            remaining = decimal(tp['qty']) - decimal(tp.get('filled_qty'))
                            if remaining != abs(decimal(position['qty'])):
                                raise RuntimeError('Take-profit remaining quantity does not match position')
                            store.update_execution(self.db, key, status='TAKE_PROFIT_OPEN', last_error=None)
                        else:
                            raise RuntimeError('Position requires execution maintenance; no confirmed active exit coverage')
                actions.append({'ticker': row['ticker'], 'action': self.row(key)['status']})
            except Exception as exc:
                store.update_execution(self.db, key, last_error=str(exc))
                actions.append({'ticker': row['ticker'], 'action': 'ERROR', 'error': str(exc)})
        return actions

    def manage_all(self):
        actions = []
        for row in self.db.execute("SELECT * FROM executions WHERE status NOT IN ('CLOSED','NO_FILL')").fetchall():
            try:
                actions.append(self.manage(row['decision_key']))
            except Exception as exc:
                store.update_execution(self.db, row['decision_key'], last_error=str(exc))
                actions.append({'ticker': row['ticker'], 'action': 'ERROR', 'error': str(exc)})
        return actions

    def execute(self, payload):
        if payload['decision'] == 'NO_TRADE':
            return {'status': 'NO_TRADE', 'details': {}}
        if payload.get('research_only'):
            raise RuntimeError('Research-only decisions cannot be executed; run the normal pipeline')
        made = dt(payload['generated_at'])
        age = (self.now() - made).total_seconds()
        if age < -60 or age > int(os.getenv('DECISION_MAX_AGE_MINUTES', '60')) * 60:
            raise RuntimeError('Brody decision is stale or future-dated')
        expected_run = os.getenv('NEWS_RUN_ID')
        if expected_run and payload.get('run_id') != expected_run:
            raise RuntimeError('Brody output belongs to another pipeline run')
        if payload.get('strategy_policy') != self.policy:
            raise RuntimeError('Brody policy does not match execution policy; rerun research')
        trade = payload['selected_trade']
        ticker, direction = trade['ticker'].upper(), trade['direction']
        if ticker == 'STRA' or direction not in {'FOR', 'AGAINST'}:
            raise RuntimeError('Prohibited ticker or invalid direction')
        key = ':'.join((trade.get('event_id') or trade['story_id'], ticker, direction))
        existing = self.db.execute('SELECT * FROM executions WHERE decision_key=?', (key,)).fetchone()
        if existing:
            if existing['status'] not in CLOSED:
                self.manage(key)
            return {'status': 'ALREADY_EXECUTED', 'details': {'decision_key': key, 'existing_status': self.row(key)['status']}}
        unresolved = self.db.execute("SELECT 1 FROM executions WHERE status NOT IN ('CLOSED','NO_FILL') AND (last_error IS NOT NULL OR ticker=?)", (ticker,)).fetchone()
        if unresolved:
            raise RuntimeError('Resolve pending execution errors or same-symbol exposure before a new entry')
        if not self.broker.clock()['is_open']:
            raise RuntimeError('Market closed; entry will not be queued overnight')
        account = self.broker.account()
        if account.get('trading_blocked') or decimal(account['buying_power']) < decimal(self.policy['trade_notional_usd']):
            raise RuntimeError('Account blocked or insufficient buying power')
        asset = self.broker.asset(ticker)
        if not asset.get('tradable'):
            raise RuntimeError('Asset is not tradable')
        if direction == 'AGAINST':
            borrow = asset.get('borrow_status')
            easy = borrow == 'easy_to_borrow' if borrow is not None else asset.get('easy_to_borrow') is True
            if not asset.get('shortable') or not easy or account.get('shorting_enabled') is False:
                raise RuntimeError('Short requires shortability and confirmed easy-to-borrow availability')
        if any(p['symbol'] == ticker for p in self.broker.positions()) or self.broker.open_orders(ticker):
            raise RuntimeError('Existing position or open order in this ticker')
        q = self.broker.quote(ticker)
        if not -5 <= (self.now() - dt(q['t'])).total_seconds() <= int(os.getenv('MAX_QUOTE_AGE_SECONDS', '60')):
            raise RuntimeError('Stale or future-dated execution quote')
        bid, ask = decimal(q['bp']), decimal(q['ap'])
        if bid <= 0 or ask < bid:
            raise RuntimeError('Invalid execution quote')
        spread = (ask - bid) / ((ask + bid) / 2) * 10000
        if spread > decimal(os.getenv('MAX_SPREAD_BPS', '150')):
            raise RuntimeError('Execution spread exceeds policy')
        cushion = decimal(os.getenv('ENTRY_CUSHION_BPS', '25')) / 10000
        if not 0 <= cushion < Decimal('0.1'):
            raise ValueError('ENTRY_CUSHION_BPS out of range')
        price = order_price(ask * (1 + cushion) if direction == 'FOR' else bid * (1 - cushion), up=direction == 'FOR')
        qty = math.floor(decimal(self.policy['trade_notional_usd']) / decimal(price))
        if qty < 1:
            raise RuntimeError('Configured notional cannot buy one share')
        now = store.utc_now()
        with self.db:
            self.db.execute('''INSERT INTO executions(decision_key,trade_date,story_id,ticker,direction,status,
                requested_notional,requested_qty,entry_limit_price,created_at,updated_at,run_id,event_id,hold_days,take_profit_pct)
                VALUES (?,?,?,?,?,'PLANNED',?,?,?,?,?,?,?,?,?)''',
                (key, payload['date'], trade['story_id'], ticker, direction, self.policy['trade_notional_usd'], qty,
                 float(price), now, now, payload.get('run_id'), trade.get('event_id'),
                 self.policy['max_hold_days'], self.policy['take_profit_pct']))
        self.submit(key, 'entry', {'symbol': ticker, 'qty': str(qty), 'side': 'buy' if direction == 'FOR' else 'sell',
                    'type': 'limit', 'time_in_force': 'day', 'limit_price': price, 'extended_hours': False})
        store.update_execution(self.db, key, status='ENTRY_SUBMITTED')
        self.manage(key, wait_entry=True)
        row = self.row(key)
        return {'status': 'NO_FILL' if row['status'] == 'NO_FILL' else 'POSITION_OPEN',
                'details': {**row, 'qty': row['filled_qty']}}
