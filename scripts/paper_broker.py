"""Small paper-only REST adapter; no automatic retries of order writes."""
from __future__ import annotations
import json
import os
from urllib.error import HTTPError
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen


class BrokerError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(f'Alpaca HTTP {status}: {message}')
        self.status_code = status


class PaperBroker:
    def __init__(self):
        from dotenv import load_dotenv
        from .trade_store import ROOT
        load_dotenv(ROOT / '.env')
        if os.getenv('ALPACA_PAPER', 'true').lower() not in {'true', '1', 'yes', 'y'}:
            raise RuntimeError('This application remains PAPER-ONLY')
        self.headers = {'APCA-API-KEY-ID': os.environ['ALPACA_API_KEY'],
                        'APCA-API-SECRET-KEY': os.environ['ALPACA_SECRET_KEY'],
                        'Content-Type': 'application/json'}

    def request(self, method, path, *, params=None, body=None, data=False):
        base = 'https://data.alpaca.markets' if data else 'https://paper-api.alpaca.markets'
        url = base + path + (('?' + urlencode(params)) if params else '')
        request = Request(url, data=json.dumps(body).encode() if body is not None else None,
                          headers=self.headers, method=method)
        try:
            with urlopen(request, timeout=30) as response:
                raw = response.read()
                return json.loads(raw) if raw else None
        except HTTPError as exc:
            # Broker error body has no request credentials.
            raise BrokerError(exc.code, exc.read().decode()[:1000]) from exc

    def account(self):
        return self.request('GET', '/v2/account')

    def clock(self):
        return self.request('GET', '/v2/clock')

    def positions(self):
        return self.request('GET', '/v2/positions')

    def open_orders(self, ticker):
        return self.request('GET', '/v2/orders', params={'status': 'open', 'symbols': ticker, 'limit': 500})

    def asset(self, ticker):
        return self.request('GET', '/v2/assets/' + quote(ticker, safe=''))

    def order(self, order_id=None, client_id=None):
        try:
            if order_id:
                return self.request('GET', '/v2/orders/' + quote(str(order_id), safe=''))
            return self.request('GET', '/v2/orders:by_client_order_id', params={'client_order_id': client_id})
        except BrokerError as exc:
            if exc.status_code == 404:
                return None
            raise

    def submit(self, request):
        return self.request('POST', '/v2/orders', body=request)

    def cancel(self, order_id):
        return self.request('DELETE', '/v2/orders/' + quote(str(order_id), safe=''))

    def quote(self, ticker):
        params = {}
        if os.getenv('ALPACA_DATA_FEED'):
            params['feed'] = os.environ['ALPACA_DATA_FEED']
        result = self.request('GET', '/v2/stocks/' + quote(ticker, safe='') + '/quotes/latest',
                              params=params, data=True)
        return result['quote']

    def fills(self, **params):
        return self.request('GET', '/v2/account/activities/FILL', params=params)
