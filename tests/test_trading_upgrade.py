"""Offline regression tests. Fake broker only; no credentials or network required."""
from __future__ import annotations
import ast
import importlib
import json
import os
import sqlite3
import tempfile
import types
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from scripts import trade_store as store
from scripts.trade_engine import Engine, gross_return
from scripts.trade_report import summary
from scripts.upgrade_trading import configure_env

NOW=datetime(2026,10,6,15,tzinfo=timezone.utc)
POLICY={'max_hold_days':180,'take_profit_pct':0.1,'trade_notional_usd':1000.0}


class FakeBroker:
    def __init__(self):
        self.orders={};self.activities=[];self.position_qty=Decimal(0)
        self.submissions=[];self.cancellations=[];self.timeout_role=None;self.cancel_pending=False
        self.market_open=True;self.fill_entry=True;self.fill_exits=False;self.qtime=NOW
    def account(self):return {'id':'paper-test','buying_power':'100000','equity':'100000','trading_blocked':False}
    def clock(self):return {'is_open':self.market_open}
    def asset(self,ticker):return {'tradable':True,'shortable':True,'easy_to_borrow':True}
    def quote(self,ticker):return {'bp':99.9,'ap':100,'t':self.qtime.isoformat()}
    def positions(self):
        return [{'symbol':'ABC','qty':str(self.position_qty)}] if self.position_qty else []
    def open_orders(self,ticker):return [o for o in self.orders.values() if o['symbol']==ticker and o['status'] not in {'filled','canceled','expired','rejected','replaced'}]
    def order(self,order_id=None,client_id=None):
        o=self.orders.get(order_id) if order_id else next((o for o in self.orders.values() if o['client_order_id']==client_id),None)
        return dict(o) if o else None
    def submit(self,request):
        self.submissions.append(dict(request))
        oid='o'+str(len(self.orders)+1)
        o={**request,'id':oid,'status':'new','filled_qty':'0','filled_avg_price':None,
           'filled_at':None,'created_at':NOW.isoformat()}
        self.orders[oid]=o
        role=request['client_order_id'].split('-')[-2]
        if role=='entry' and self.fill_entry:self.fill(oid,request['qty'],100)
        if role=='exit' and self.fill_exits:self.fill(oid,request['qty'],90)
        if role==self.timeout_role:raise TimeoutError('Accepted remotely; response lost')
        return dict(o)
    def fill(self,oid,qty,price,when=NOW):
        o=self.orders[oid];q=Decimal(str(qty));p=Decimal(str(price));old=Decimal(o['filled_qty'])
        prior=Decimal(o['filled_avg_price'] or '0')*old
        o['filled_qty']=str(old+q);o['filled_avg_price']=str((prior+q*p)/(old+q))
        o['status']='filled' if old+q==Decimal(o['qty']) else 'partially_filled'
        o['filled_at']=when.isoformat() if o['status']=='filled' else None
        self.position_qty+=q*(1 if o['side']=='buy' else -1)
        self.activities.append({'activity_type':'FILL','id':'f'+str(len(self.activities)+1).zfill(5),
             'order_id':oid,'symbol':o['symbol'],'side':o['side'],'qty':str(q),'price':str(p),'transaction_time':when.isoformat()})
    def cancel(self,oid):
        self.cancellations.append(oid)
        self.orders[oid]['status']='pending_cancel' if self.cancel_pending else 'canceled'
    def fills(self,**params):
        result=[a for a in self.activities if (not params.get('after') or a['transaction_time']>params['after']) and a['transaction_time']<=params['until']]
        if params.get('page_token'):result=result[next(i+1 for i,a in enumerate(result) if a['id']==params['page_token']):]
        return result[:int(params.get('page_size',100))]


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.db=store.connect(Path(self.tmp.name)/'trading.db')
        self.broker=FakeBroker();self.engine=Engine(self.db,self.broker,POLICY,now=lambda:NOW,sleep=lambda _:None)
        self.payload={'date':'2026-10-06','generated_at':NOW.isoformat(),'decision':'TRADE','strategy_policy':POLICY,
                      'selected_trade':{'story_id':'story','event_id':'event','ticker':'ABC','direction':'FOR'}}
        self.env=patch.dict(os.environ,{'NEWS_RUN_ID':'','MAX_ENTRY_WAIT_MINUTES':'0','CANCEL_TIMEOUT_SECONDS':'0'})
        self.env.start()
    def tearDown(self):self.env.stop();self.db.close();self.tmp.cleanup()
    def key(self):return self.db.execute('SELECT decision_key FROM executions').fetchone()[0]
    def test_short_return_uses_entry_notional(self):
        self.assertEqual(gross_return('AGAINST',100,90),10)
        self.assertEqual(gross_return('AGAINST',100,110),-10)
    def test_entry_fill_and_180_day_deadline(self):
        self.engine.execute(self.payload);r=self.engine.row(self.key())
        self.assertEqual(r['status'],'TAKE_PROFIT_OPEN')
        self.assertEqual(datetime.fromisoformat(r['exit_deadline']),NOW+timedelta(days=180))
        self.assertEqual(len(self.broker.submissions),2)
    def test_entry_timeout_recovers_same_order(self):
        self.broker.timeout_role='entry'
        with self.assertRaises(TimeoutError):self.engine.execute(self.payload)
        self.broker.timeout_role=None
        self.engine.manage_all()
        self.assertEqual(len([r for r in self.broker.submissions if '-entry-' in r['client_order_id']]),1)
        self.assertEqual(self.engine.row(self.key())['status'],'TAKE_PROFIT_OPEN')
    def test_tp_timeout_preserves_fill_and_recovers(self):
        self.broker.timeout_role='tp'
        with self.assertRaises(TimeoutError):self.engine.execute(self.payload)
        r=self.engine.row(self.key());self.assertIsNotNone(r['filled_at']);self.assertIsNotNone(r['exit_deadline'])
        self.broker.timeout_role=None;self.engine.manage_all()
        self.assertEqual(len(self.broker.submissions),2)
        self.assertEqual(self.engine.row(self.key())['status'],'TAKE_PROFIT_OPEN')
    def test_uncertain_submission_is_not_resubmitted(self):
        self.broker.timeout_role='entry'
        with self.assertRaises(TimeoutError):self.engine.execute(self.payload)
        self.broker.orders.clear();count=len(self.broker.submissions)
        actions=self.engine.manage_all()
        self.assertEqual(actions[0]['action'],'ERROR');self.assertEqual(len(self.broker.submissions),count)
    def test_pending_cancel_is_not_no_fill(self):
        self.broker.fill_entry=False;self.broker.cancel_pending=True
        with self.assertRaisesRegex(RuntimeError,'Cancellation unconfirmed'):self.engine.execute(self.payload)
        self.assertNotEqual(self.engine.row(self.key())['status'],'NO_FILL')
        self.assertEqual(len(self.broker.submissions),1)
    def test_terminal_cancel_without_fill(self):
        self.broker.fill_entry=False
        result=self.engine.execute(self.payload)
        self.assertEqual(result['status'],'NO_FILL')
        self.assertEqual(len(self.broker.submissions),1)
    def test_repeat_decision_does_not_submit_again(self):
        self.engine.execute(self.payload);self.engine.execute(self.payload)
        self.assertEqual(len(self.broker.submissions),2)
    def test_filled_tp_closes_using_fill_ledger(self):
        self.engine.execute(self.payload);key=self.key();r=self.engine.row(key)
        self.broker.fill(r['take_profit_order_id'],r['filled_qty'],110)
        self.engine.manage(key)
        self.assertEqual(self.engine.row(key)['realized_return_pct'],10)
        self.assertEqual(self.engine.row(key)['status'],'CLOSED')
    def test_partial_tp_and_delayed_exit_weighted_on_restart(self):
        self.engine.execute(self.payload);key=self.key();r=self.engine.row(key)
        qty=Decimal(str(r['filled_qty']));self.broker.fill(r['take_profit_order_id'],qty/2,110)
        store.update_execution(self.db,key,exit_deadline=(NOW-timedelta(seconds=1)).isoformat())
        self.engine.manage(key);r=self.engine.row(key)
        self.broker.fill(r['exit_order_id'],qty/2,80)
        restarted=Engine(self.db,self.broker,POLICY,now=lambda:NOW,sleep=lambda _:None)
        restarted.manage(key)
        r=restarted.row(key);self.assertEqual(r['exit_price'],95);self.assertEqual(r['realized_return_pct'],-5)
    def test_exit_timeout_recovers_persisted_id_next_day(self):
        self.engine.execute(self.payload);key=self.key()
        store.update_execution(self.db,key,exit_deadline=(NOW-timedelta(days=1)).isoformat())
        self.broker.timeout_role='exit'
        with self.assertRaises(TimeoutError):self.engine.manage(key)
        cid=self.engine.row(key)['exit_client_order_id'];count=len(self.broker.submissions)
        self.broker.timeout_role=None
        restarted=Engine(self.db,self.broker,POLICY,now=lambda:NOW+timedelta(days=1),sleep=lambda _:None)
        restarted.manage(key)
        self.assertEqual(restarted.row(key)['exit_client_order_id'],cid);self.assertEqual(len(self.broker.submissions),count)
    def test_external_close_is_unresolved_not_fabricated(self):
        self.engine.execute(self.payload);self.broker.position_qty=0
        actions=self.engine.manage_all()
        self.assertEqual(actions[0]['action'],'ERROR');self.assertIsNone(self.engine.row(self.key())['closed_at'])
    def test_stale_quote_blocks_submission(self):
        self.broker.qtime=NOW-timedelta(minutes=5)
        with self.assertRaisesRegex(RuntimeError,'quote'):self.engine.execute(self.payload)
        self.assertEqual(self.broker.submissions,[])
    def test_stale_decision_blocks_submission(self):
        self.payload['generated_at']=(NOW-timedelta(days=1)).isoformat()
        with self.assertRaisesRegex(RuntimeError,'stale'):self.engine.execute(self.payload)
        self.assertEqual(self.broker.submissions,[])
    def test_policy_mismatch_blocks_submission(self):
        self.payload['strategy_policy']={**POLICY,'max_hold_days':28}
        with self.assertRaisesRegex(RuntimeError,'policy'):self.engine.execute(self.payload)
    def test_sync_paginates_and_is_idempotent(self):
        for i in range(105):
            self.broker.activities.append({'activity_type':'FILL','id':str(i).zfill(4),'order_id':'manual'+str(i),'symbol':'ABC',
                                         'side':'buy','qty':'1','price':'1','transaction_time':NOW.isoformat()})
        self.engine.sync_fills();self.engine.sync_fills()
        self.assertEqual(self.db.execute('SELECT count(*) FROM fills').fetchone()[0],105)
    def test_account_scope_mismatch(self):
        self.broker.account=lambda:{'id':'other'}
        with self.assertRaisesRegex(RuntimeError,'different'):Engine(self.db,self.broker,POLICY)
    def test_report_distinguishes_unknown_from_no_fills(self):
        self.assertIn('unconfirmed',summary(self.db,now=NOW))
        self.engine.sync_fills()
        self.assertIn('No trades executed',summary(self.db,now=NOW))
    def test_report_counts_orders_not_partial_fills(self):
        self.engine.execute(self.payload)
        self.assertIn('1 order(s)',summary(self.db,now=NOW))
    def test_current_failed_sync_overrides_recent_sync(self):
        self.engine.sync_fills()
        self.db.execute("INSERT INTO stage_runs(run_id,stage,started_at,status) VALUES ('run','position_manager',?,'FAILED')",(NOW.isoformat(),));self.db.commit()
        self.assertIn('unconfirmed',summary(self.db,run_id='run',now=NOW))
    def test_migration_is_idempotent(self):
        path=Path(self.tmp.name)/'trading.db';db2=store.connect(path);db2.close()
        self.assertEqual(store.meta(self.db,'schema_version'),'1')

    def test_partial_entry_is_canceled_before_sizing_tp(self):
        self.broker.fill_entry=False
        original=self.broker.submit
        def submit(request):
            order=original(request)
            if '-entry-' in request['client_order_id']:
                self.broker.fill(order['id'],2,100)
                return self.broker.order(order_id=order['id'])
            return order
        self.broker.submit=submit
        self.engine.execute(self.payload)
        row=self.engine.row(self.key())
        self.assertEqual(row['filled_qty'],2)
        self.assertEqual(self.broker.orders[row['take_profit_order_id']]['qty'],'2')
        self.assertEqual(self.broker.orders[row['entry_order_id']]['status'],'canceled')
    def test_legacy_replacement_chain_updates_current_order(self):
        self.engine.execute(self.payload);key=self.key();row=self.engine.row(key)
        oid=row['entry_order_id'];old=dict(self.broker.orders[oid])
        successor={**old,'id':'replacement','client_order_id':'legacy-replacement'}
        self.broker.orders['replacement']=successor
        self.broker.orders[oid]['status']='replaced';self.broker.orders[oid]['replaced_by']='replacement'
        recovered=self.engine.recover(row,'entry')
        self.assertEqual(recovered['id'],'replacement')
        self.assertEqual(self.engine.row(key)['entry_order_id'],'replacement')
        self.assertEqual(self.db.execute("SELECT count(*) FROM broker_orders WHERE role='entry'").fetchone()[0],2)
    def test_read_only_refresh_closes_filled_tp_without_broker_writes(self):
        self.engine.execute(self.payload);row=self.engine.row(self.key())
        self.broker.fill(row['take_profit_order_id'],row['filled_qty'],110)
        self.engine.sync_fills();before=(len(self.broker.submissions),len(self.broker.cancellations))
        self.engine.refresh_all()
        self.assertEqual(self.engine.row(self.key())['status'],'CLOSED')
        self.assertEqual(before,(len(self.broker.submissions),len(self.broker.cancellations)))
    def test_closed_position_with_active_exit_is_not_silently_finalized(self):
        self.engine.execute(self.payload);key=self.key();row=self.engine.row(key)
        self.broker.fill(row['take_profit_order_id'],row['filled_qty'],110)
        self.broker.orders[row['take_profit_order_id']]['status']='pending_cancel'
        self.broker.cancel_pending=True
        actions=self.engine.manage_all()
        self.assertEqual(actions[0]['action'],'ERROR')
        self.assertNotEqual(self.engine.row(key)['status'],'CLOSED')
    def test_research_only_decision_cannot_be_executed(self):
        self.payload['research_only']=True
        with self.assertRaisesRegex(RuntimeError,'Research-only'):self.engine.execute(self.payload)
        self.assertEqual(self.broker.submissions,[])


class MarketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pandas as pd
        cls.pd=pd
    def provider(self):
        from scripts.market_tools import MarketTools
        p=MarketTools([{'ticker':'ABC','event_date':'2026-10-02','event_timing':'after_close'}],cutoff=NOW)
        index=self.pd.date_range('2026-07-01','2026-10-06',freq='B',tz='America/New_York')
        values=list(range(100,100+len(index)))
        data=self.pd.DataFrame({'Open':values,'High':values,'Low':values,'Close':values,'Volume':[1000]*(len(index)-1)+[28]},index=index)
        p.obj=lambda ticker:types.SimpleNamespace(history=lambda **kwargs:data.copy())
        return p
    def test_partial_session_is_excluded(self):
        p=self.provider();r=p.call('get_price_trend',{'ticker':'ABC'})
        self.assertTrue(r['ok']);self.assertEqual(r['data']['last_completed_session'],'2026-10-05')
        self.assertEqual(r['data']['completed_session_volume_ratio'],1)
    def test_unknown_ticker_is_rejected(self):
        self.assertFalse(self.provider().call('get_price_trend',{'ticker':'UNSUPPLIED'})['ok'])
    def test_after_close_maps_to_next_available_session(self):
        p=self.provider();r=p.call('get_event_price_reaction',{'ticker':'ABC','event_date':'2026-10-02','timing':'after_close','before_sessions':5,'after_sessions':5})
        self.assertTrue(r['ok']);self.assertEqual(r['data']['mapped_reaction_session'],'2026-10-05')
        self.assertFalse(r['data']['reaction_windows'][-1]['available'])
    def test_event_date_must_be_verified(self):
        r=self.provider().call('get_event_price_reaction',{'ticker':'ABC','event_date':'2026-09-30','timing':'unknown','before_sessions':5,'after_sessions':1})
        self.assertFalse(r['ok'])


class ConfigTests(unittest.TestCase):
    def test_env_update_keeps_unrelated_values_and_removes_duplicate_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'.env';path.write_text('OTHER=value\nMAX_HOLD_DAYS=28\nexport MAX_HOLD_DAYS=40\n')
            configure_env(path);text=path.read_text()
            self.assertIn('OTHER=value',text);self.assertEqual(text.count('MAX_HOLD_DAYS='),1)
            self.assertIn('MAX_HOLD_DAYS=180',text);self.assertIn('BRODY_MODEL=gpt-6-astra',text)



def extracted_function(filename,name,namespace):
    """Load a pure boundary function without constructing live API clients."""
    source=Path(__file__).resolve().parents[1]/'scripts'/filename
    tree=ast.parse(source.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(source),'exec'),namespace)
    return namespace[name]


class BoundaryTests(unittest.TestCase):
    def test_brody_preserves_reasoning_and_matches_tool_call_ids(self):
        import copy,time
        observed=[];recorded=[]
        result={'decision':'NO_TRADE','selected_story_id':None,'candidate_assessments':[{'ticker':'ABC','verdict':'REJECT'}]}
        def item(value):return types.SimpleNamespace(model_dump=lambda **kwargs:value)
        replies=[types.SimpleNamespace(id='r1',model='gpt-6-astra',usage=None,status='completed',output_text='',output=[
                  item({'type':'reasoning','id':'reason','encrypted_content':'opaque-test-state'}),
                  item({'type':'function_call','name':'get_price_trend','arguments':'{"ticker":"ABC"}','call_id':'call1'})]),
                 types.SimpleNamespace(id='r2',model='gpt-6-astra',usage=None,status='completed',output_text=json.dumps(result),output=[])]
        def create(**kwargs):observed.append(copy.deepcopy(kwargs));return replies.pop(0)
        provider=types.SimpleNamespace(call=lambda *args:{'ok':True,'data':{}},trades={'ABC':{}},successful=set())
        ns={'json':json,'time':time,'policy':lambda:POLICY,'MarketTools':lambda trades:provider,
            'BRODY_INSTRUCTIONS':'180 days','BRODY_MODEL':'gpt-6-astra','BRODY_REASONING':'medium','BRODY_SCHEMA':{},'TOOLS':[],
            'client':types.SimpleNamespace(responses=types.SimpleNamespace(create=create)),
            'trade_store':types.SimpleNamespace(record_tool=lambda *args:recorded.append(args))}
        f=extracted_function('brody.py','analyze',ns)
        answer=f({'candidates':[{'gilman_trade':{'ticker':'ABC','story_id':'s'}}]})
        self.assertEqual(answer['decision'],'NO_TRADE')
        history=observed[1]['input']
        self.assertTrue(any(i.get('encrypted_content')=='opaque-test-state' for i in history))
        self.assertTrue(any(i.get('type')=='function_call_output' and i.get('call_id')=='call1' for i in history))
        self.assertEqual(len(recorded),1)
    def test_brody_approval_requires_tool_evidence(self):
        import time
        reply=types.SimpleNamespace(id='r1',model='gpt-6-astra',usage=None,status='completed',output=[],
              output_text=json.dumps({'candidate_assessments':[{'ticker':'ABC','story_id':'s','verdict':'APPROVE'}]}))
        ns={'json':json,'time':time,'policy':lambda:POLICY,
            'MarketTools':lambda trades:types.SimpleNamespace(trades={'ABC':{}},successful=set()),
            'BRODY_INSTRUCTIONS':'','BRODY_MODEL':'gpt-6-astra','BRODY_REASONING':'medium','BRODY_SCHEMA':{},'TOOLS':[],
            'client':types.SimpleNamespace(responses=types.SimpleNamespace(create=lambda **kw:reply))}
        f=extracted_function('brody.py','analyze',ns)
        with self.assertRaisesRegex(RuntimeError,'tool evidence'):f({'candidates':[{'gilman_trade':{'ticker':'ABC','story_id':'s'}}]})
    def test_brody_duplicate_assessments_rejected(self):
        from collections import Counter
        f=extracted_function('brody.py','validate_result',{'Counter':Counter})
        trade={'story_id':'s','ticker':'ABC','direction':'FOR'}
        assessment={**trade,'verdict':'REJECT'}
        with self.assertRaisesRegex(RuntimeError,'exactly one'):
            f({'candidates':[{'gilman_trade':trade}]},{'candidate_assessments':[assessment,assessment], 'decision':'NO_TRADE','selected_story_id':None})
    def test_report_email_allows_zero_news_and_does_not_repeat_unknown_send(self):
        import hashlib,logging
        from email.message import EmailMessage
        attempts=[];states={}
        class SMTP:
            def __init__(self,*args,**kw):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def login(self,*args):pass
            def send_message(self,message):attempts.append(message);raise TimeoutError('delivery response lost')
        def claim(key):
            if key in states:return False
            states[key]='SUBMITTING';return True
        with tempfile.TemporaryDirectory() as tmp:
            ns={'report_date':lambda p:'2026-10-06','EMAIL_REPORT_LIMIT':25,'EmailMessage':EmailMessage,
                'GMAIL_ADDRESS':'sender@example.invalid','EMAIL_TO':'recipient@example.invalid','GMAIL_APP_PASSWORD':'fake',
                'build_plain_text':lambda p,*args:p['trade_summary'],'build_html':lambda p,*args:'<p>'+p['trade_summary']+'</p>',
                'logger':logging.getLogger('test'),'hashlib':hashlib,'DATA_DIR':Path(tmp),
                'smtplib':types.SimpleNamespace(SMTP_SSL=SMTP),
                'trade_store':types.SimpleNamespace(claim_report=claim,report_status=lambda key,status,error=None:states.update({key:status}))}
            f=extracted_function('email_report.py','send_report',ns)
            payload={'stories':[],'trade_summary':'No trades executed.','report_key':'test'}
            with self.assertRaises(TimeoutError):f(payload,'Unavailable')
            self.assertEqual(states['test'],'UNKNOWN')
            with self.assertRaisesRegex(RuntimeError,'ambiguous'):f(payload,'Unavailable')
            self.assertEqual(len(attempts),1)
    def test_gilman_handoff_keeps_event_identity_and_timing(self):
        import logging
        written={};recorded=[]
        fake_store=types.SimpleNamespace(run_id=lambda:'run',record_decision=lambda *a:recorded.append(a),
                                        atomic_json=lambda path,payload:written.update({str(path):payload}))
        ns={'utc_now':lambda:NOW.isoformat(),'GILMAN_MODEL':'gpt-6-astra','GILMAN_REASONING':'high','BATCH_SIZE':10,
            'policy':lambda:POLICY,'trade_store':fake_store,'DATA_DIR':Path('data'),
            'GILMAN_LATEST_PATH':Path('data/gilman_latest.json'),'GILMAN_TRADES_LATEST_PATH':Path('data/gilman_trades_latest.json'),
            'logger':logging.getLogger('test')}
        f=extracted_function('gilman.py','write_outputs',ns)
        source={'date':'2026-10-06','stories':[{'story_id':'s','event_context':{'event_id':'event','timeline':[]}}]}
        f(source,[{'story_id':'s','decision':'TRADE','investment_relevance':0.8,
                  'trade':{'ticker':'ABC','event_date':'2026-10-02','event_timing':'after_close'}}],[])
        handoff=written['data/gilman_trades_latest.json']
        self.assertEqual(handoff['run_id'],'run');self.assertEqual(handoff['trades'][0]['event_id'],'event')
        self.assertEqual(handoff['trades'][0]['event_date'],'2026-10-02');self.assertEqual(len(recorded),1)

if __name__=='__main__':unittest.main()
