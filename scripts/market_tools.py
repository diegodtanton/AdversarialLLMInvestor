"""Bounded research tools. Daily metrics always exclude the unfinished session."""
from __future__ import annotations
import json
import math
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

NY = ZoneInfo('America/New_York')


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def change(a, b):
    return round((float(b) / float(a) - 1) * 100, 4) if a else None


def definition(name, description, properties):
    return {'type': 'function', 'name': name, 'description': description, 'strict': True,
            'parameters': {'type': 'object', 'properties': properties, 'required': list(properties),
                           'additionalProperties': False}}


TICKER = {'type': 'string', 'description': 'Exactly one of the supplied candidate tickers'}
TOOLS = [
    definition('get_price_trend', 'Completed daily sessions only: trend, volatility, dollar volume and SPY-relative returns.',
               {'ticker': TICKER}),
    definition('get_event_price_reaction', 'Measure a verified announcement window. Returns evidence, never a priced-in conclusion.',
               {'ticker': TICKER, 'event_date': {'type': 'string', 'description': 'Verified YYYY-MM-DD announcement date'},
                'timing': {'type': 'string', 'enum': ['before_open','during_session','after_close','unknown']},
                'before_sessions': {'type': 'integer', 'minimum': 1, 'maximum': 60},
                'after_sessions': {'type': 'integer', 'minimum': 1, 'maximum': 20}}),
    definition('get_fundamentals', 'Yahoo summary fundamentals and quarterly cash-flow observations with explicit period metadata.',
               {'ticker': TICKER}),
    definition('get_corporate_calendar', 'Available earnings estimates and recent split/dividend history; unknown dates remain unknown.',
               {'ticker': TICKER}),
    definition('get_execution_context', 'Read-only paper broker quote, asset eligibility, account exposure and pending orders.',
               {'ticker': TICKER}),
]


class MarketTools:
    def __init__(self, trades, *, cutoff=None, yf_module=None, broker=None):
        self.trades = {t['ticker'].upper(): t for t in trades}
        self.event_dates = {ticker: {(t.get('event_date'), t.get('event_timing', 'unknown')) for t in trades if t['ticker'].upper() == ticker} for ticker in self.trades}
        self.cutoff = cutoff or datetime.now(timezone.utc)
        self.yf = yf_module
        self.broker = broker
        self.cache, self.histories, self.objects = {}, {}, {}
        self.successful = set()
        self.successful_events = set()

    def obj(self, ticker):
        if self.yf is None:
            import yfinance
            self.yf = yfinance
        if ticker not in self.objects:
            self.objects[ticker] = self.yf.Ticker(ticker)
        return self.objects[ticker]

    def history(self, ticker):
        import pandas as pd
        if ticker not in self.histories:
            data = self.obj(ticker).history(period='2y', interval='1d', auto_adjust=True, actions=True, timeout=15)
            if data is None or data.empty:
                raise ValueError(f'No daily data for {ticker}')
            data = data.copy()
            index = pd.DatetimeIndex(data.index)
            if index.tz is not None:
                index = index.tz_convert(NY)
            data.index = index.date
            # Deliberately conservative even after close: current date excluded.
            data = data[data.index < self.cutoff.astimezone(NY).date()]
            data = data[~data.index.duplicated(keep='last')].sort_index()
            data = data.dropna(subset=['Close'])
            data = data[data['Close'] > 0]
            if data.empty:
                raise ValueError('No completed daily observations')
            self.histories[ticker] = data
        return self.histories[ticker]

    def get_price_trend(self, ticker):
        import pandas as pd
        data = self.history(ticker)
        close = data['Close']
        age = (self.cutoff.astimezone(NY).date() - data.index[-1]).days
        if age > 7:
            raise ValueError(f'Daily price history is stale: last observation {data.index[-1]}')
        if len(data) < 21:
            raise ValueError('Fewer than 21 completed sessions')
        spy = self.history('SPY')
        joined = pd.concat([close.rename('stock'), spy['Close'].rename('spy')], axis=1).dropna()
        windows = {}
        for n in [1,5,20,60,120,252]:
            if len(joined) <= n:
                windows[str(n)] = None
                continue
            a,b = joined.iloc[-n-1], joined.iloc[-1]
            sr, br = change(a['stock'],b['stock']), change(a['spy'],b['spy'])
            windows[str(n)] = {'start': str(joined.index[-n-1]), 'end': str(joined.index[-1]),
                               'stock_return_pct': sr, 'spy_return_pct': br,
                               'relative_return_percentage_points': round(sr-br,4)}
        volume = data['Volume']
        avg = number(volume.iloc[-21:-1].mean())
        return {'last_completed_session': str(data.index[-1]), 'adjusted_close': number(close.iloc[-1]),
                'current_session_excluded': True, 'adjustment': 'split/dividend adjusted OHLC',
                'returns_by_aligned_session_count': windows,
                'realized_volatility_20_sessions_pct': number(close.pct_change().dropna().iloc[-20:].std()*math.sqrt(252)*100),
                'average_dollar_volume_20_sessions': number((close*volume).iloc[-20:].mean()),
                'completed_session_volume_ratio': number(volume.iloc[-1]/avg) if avg else None,
                'moving_average_distance_pct': {str(n): change(close.iloc[-n:].mean(),close.iloc[-1]) if len(close)>=n else None for n in [20,50,200]},
                'warnings': ['Returns are aligned on available stock/SPY dates; missing sessions can lengthen calendar windows.']}

    def get_event_price_reaction(self, ticker, event_date, timing, before_sessions, after_sessions):
        import pandas as pd
        event = datetime.strptime(event_date, '%Y-%m-%d').date()
        if event >= self.cutoff.astimezone(NY).date():
            raise ValueError('No completed-session reaction available yet for this event date')
        if (self.cutoff.date() - event).days > 700:
            raise ValueError('Event is outside the supported history window')
        if (event_date, timing) not in self.event_dates[ticker]:
            raise ValueError('Event date and timing must match a source-verified Gilman candidate')
        if timing not in {'before_open','during_session','after_close','unknown'}:
            raise ValueError('Invalid event timing')
        for n, upper in [(before_sessions,60),(after_sessions,20)]:
            if type(n) is not int or not 1 <= n <= upper:
                raise ValueError('Event window outside allowed bounds')
        data, spy = self.history(ticker), self.history('SPY')
        joined = pd.concat([data['Close'].rename('stock'),spy['Close'].rename('spy')],axis=1).dropna()
        candidates = [i for i,d in enumerate(joined.index) if d > event or (d == event and timing != 'after_close')]
        if not candidates:
            raise ValueError('No completed post-announcement session')
        i = candidates[0]
        if i < 1:
            raise ValueError('No pre-announcement close available')
        baseline = joined.iloc[i-1]
        reactions = []
        for n in sorted(set([1,after_sessions])):
            j = i+n-1
            if j >= len(joined):
                reactions.append({'sessions':n,'available':False})
                continue
            last = joined.iloc[j]
            sr,br=change(baseline['stock'],last['stock']),change(baseline['spy'],last['spy'])
            reactions.append({'sessions':n,'available':True,'start_close_date':str(joined.index[i-1]),
                              'end_date':str(joined.index[j]),'stock_return_pct':sr,'spy_return_pct':br,
                              'relative_return_percentage_points':round(sr-br,4)})
        prior = data.loc[data.index < joined.index[i],'Volume'].iloc[-20:]
        avg = number(prior.mean())
        return {'event_date':event_date,'timing':timing,'mapped_reaction_session':str(joined.index[i]),
                'pre_event_return_pct':change(joined.iloc[i-before_sessions-1]['stock'],baseline['stock']) if i>before_sessions else None,
                'reaction_windows':reactions,
                'event_session_volume_ratio':number(data.loc[joined.index[i],'Volume']/avg) if avg and len(prior)==20 else None,
                'warnings':['Price movement is not evidence of causation or complete pricing-in.',
                            'Unknown/intraday announcement times make daily event attribution ambiguous.',
                            'Missing stock/benchmark dates can widen the measured window.']}

    def get_fundamentals(self,ticker):
        obj = self.obj(ticker)
        info = obj.get_info()
        fields = ['marketCap','enterpriseValue','trailingPE','forwardPE','priceToSalesTrailing12Months',
                  'priceToBook','enterpriseToRevenue','enterpriseToEbitda','grossMargins','operatingMargins',
                  'profitMargins','revenueGrowth','earningsGrowth','totalCash','totalDebt','operatingCashflow',
                  'freeCashflow','shortPercentOfFloat','sharesShort','dateShortInterest','lastFiscalYearEnd','mostRecentQuarter']
        values={k:number(info.get(k)) for k in fields}
        if not any(v is not None for v in values.values()):
            raise ValueError('No usable fundamentals')
        cashflow=[]
        warnings=['Summary fields may mix reporting periods. Margins/growth/short float are fractions, not percentages.',
                  'Summary cash-flow fields are not assumed comparable without matching statement periods.']
        try:
            frame=obj.quarterly_cashflow
            for period in frame.columns[:4]:
                cashflow.append({'period_end':str(period.date()),'period_type':'quarterly',
                                 **{name:number(frame.loc[name,period]) if name in frame.index else None
                                    for name in ['Operating Cash Flow','Capital Expenditure','Free Cash Flow']}})
        except Exception as exc:
            warnings.append('Quarterly cash-flow data unavailable: '+str(exc)[:160])
        return {'currency':info.get('currency'),'financial_currency':info.get('financialCurrency'),
                'sector':info.get('sector'),'industry':info.get('industry'),'summary':values,
                'quarterly_cashflow':cashflow,'warnings':warnings}

    def get_corporate_calendar(self,ticker):
        calendar=self.obj(ticker).calendar
        data=self.history(ticker)
        actions=[]
        for date,row in data.tail(252).iterrows():
            dividend,split=number(row.get('Dividends')),number(row.get('Stock Splits'))
            if dividend or split:
                actions.append({'date':str(date),'dividend':dividend,'split_ratio':split})
        return {'calendar':json.loads(json.dumps(calendar,default=str)), 'actions':actions,
                'warnings':['Earnings dates may be estimates; absence is not confirmation that no event is scheduled.']}

    def get_execution_context(self,ticker):
        if self.broker is None:
            from .paper_broker import PaperBroker
            self.broker=PaperBroker()
        account=self.broker.account()
        asset=self.broker.asset(ticker)
        quote=self.broker.quote(ticker)
        return {'paper':True,'quote':quote,
                'asset':{k:asset.get(k) for k in ['tradable','shortable','easy_to_borrow','borrow_status','exchange']},
                'buying_power':account.get('buying_power'),'equity':account.get('equity'),
                'positions':[{k:p.get(k) for k in ['symbol','qty','side','market_value','unrealized_pl']} for p in self.broker.positions()],
                'open_orders':[{k:o.get(k) for k in ['symbol','side','qty','filled_qty','status']} for o in self.broker.open_orders(ticker)]}

    def call(self,name,arguments):
        if name not in {t['name'] for t in TOOLS}:
            return {'ok':False,'error':'Unknown tool'}
        if not isinstance(arguments,dict) or arguments.get('ticker') not in self.trades:
            return {'ok':False,'error':'Ticker must be one of the supplied candidates'}
        key=json.dumps([name,arguments],sort_keys=True)
        if key not in self.cache:
            try:
                result=getattr(self,name)(**arguments)
                self.cache[key]={'ok':True,'source':'Alpaca paper' if name=='get_execution_context' else 'Yahoo Finance via yfinance',
                                 'fetched_at':datetime.now(timezone.utc).isoformat(),'cutoff':self.cutoff.isoformat(),'data':result}
                self.successful.add((arguments['ticker'],name))
                if name == 'get_event_price_reaction':
                    self.successful_events.add((arguments['ticker'], arguments['event_date'], arguments['timing']))
            except Exception as exc:
                self.cache[key]={'ok':False,'error':str(exc)[:800], 'cutoff':self.cutoff.isoformat()}
        return self.cache[key]
