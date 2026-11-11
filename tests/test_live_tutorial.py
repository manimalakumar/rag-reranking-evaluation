"""Hermetic tests of Live/Replay/Plan, SDK wire calls and preserved history."""
import ast
from copy import deepcopy
import json,os
from pathlib import Path
import tempfile,unittest
import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from openai import OpenAI
import pandas as pd
from shared_scoring.live_tutorial import TutorialRun,NOTEBOOKS
from shared_scoring.live_client import TutorialTransport,request_scope
from shared_scoring.io import write_json,read_json

PROJECT=Path(__file__).resolve().parents[1]


class LiveTutorialTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        for name in ['execution.json','tutorial_inputs.json','model_limits.json']:
            write_json(self.root/'configs'/name,read_json(PROJECT/'configs'/name))
        write_json(self.root/'experiments/v3-answer-quality/config.json',read_json(PROJECT/'experiments/v3-answer-quality/config.json'))
        self.calls=[]
        def wire(request):
            body=json.loads(request.read());self.calls.append((str(request.url),body))
            if request.url.path.endswith('/embeddings'):
                response=dict(object='list',model=body['model'],data=[dict(index=i,object='embedding',embedding=[1.,0.,0.,0.]) for i in range(len(body['input']))])
            elif request.url.path.endswith('/rerank'):
                order=[5]+[i for i in range(30) if i!=5]
                response=dict(results=[dict(index=i,relevance_score=30-r) for r,i in enumerate(order)])
            else:response=dict(id='chat-unit',object='chat.completion',created=0,model=body['model'],provider='unit-provider',choices=[dict(index=0,finish_reason='stop',message=dict(role='assistant',content='saved unit answer'))])
            return httpx.Response(200,json=response,request=request)
        self.wire=wire
        self.env=patch.dict(os.environ,{'MODEL_BUDGET_PATH':str(self.root/'budget.sqlite'),'TUTORIAL_TESTS_PASSED':'1','OPENROUTER_API_KEY':'unit-test-secret'})
        self.env.start()
    def tearDown(self):self.env.stop();self.temp.cleanup()
    def client(self,run_id,resume=False,wire=None,retry_in_flight=False,retry_authorizations=None):
        transport=TutorialTransport(self.root/'runs'/run_id/'requests',run_id,resume=resume,retry_in_flight=retry_in_flight,retry_authorizations=retry_authorizations,inner=httpx.MockTransport(wire or self.wire))
        return OpenAI(api_key='unit-test-secret',base_url='https://openrouter.ai/api/v1',max_retries=0,http_client=httpx.Client(transport=transport))
    def test_default_live_is_new_identity_not_cached_execute(self):
        a=TutorialRun(self.root,'answers');b=TutorialRun(self.root,'answers')
        self.assertEqual(a.mode,'live');self.assertNotEqual(a.run_id,b.run_id)
        self.assertFalse(a.folder.exists());self.assertFalse(b.folder.exists())
    def test_plan_needs_no_key_or_cache_and_writes_nothing(self):
        with patch.dict(os.environ,{},clear=True):
            run=TutorialRun(self.root,'answers','plan')
            self.assertTrue(run.preflight()['missing_inputs'])
            self.assertIsNone(run.generate(lambda *args:self.fail('Plan called a generator')))
        self.assertFalse(run.folder.exists())
    def test_missing_inputs_fail_before_creating_run_or_client(self):
        run=TutorialRun(self.root,'baseline')
        with self.assertRaisesRegex(FileNotFoundError,'setup.md'):run.preflight()
        self.assertIsNone(run._client);self.assertFalse(run.folder.exists())
    def test_replay_requires_an_explicit_run_and_never_fills_missing_stages(self):
        with self.assertRaises(ValueError):TutorialRun(self.root,'answers','replay')
        with self.assertRaises(ValueError):TutorialRun(self.root,'answers','replay','reference')
        original=TutorialRun(self.root,'answers')
        write_json(original.folder/'manifest.json',{})
        replay=TutorialRun(self.root,'answers','replay',original.run_id)
        with self.assertRaises(FileNotFoundError):
            replay.generate(lambda *args:self.fail('Replay invoked model'))
    def test_replay_reads_saved_questions_and_rejects_a_different_input_profile(self):
        from shared_scoring.io import sha256_file
        run=TutorialRun(self.root,'demo')
        saved=[dict(query_id='saved-id',question='the saved question',reference_answers=['answer'],gold_ids=[],subgroup='warmup',diagnostic=False)]
        write_json(run.folder/'questions.json',saved)
        write_json(run.folder/'manifest.json',{'contract':{'tutorial_inputs':sha256_file(run.input_configuration)}})
        alternate=read_json(run.input_configuration);alternate['demo_question']['question']='a different active question'
        write_json(self.root/'alternate-inputs.json',alternate)
        with patch.dict(os.environ,{'TUTORIAL_INPUTS':str(self.root/'alternate-inputs.json'),'OPENROUTER_API_KEY':''}):
            replay=TutorialRun(self.root,'demo',mode='replay',run_id=run.run_id)
            with patch('pandas.read_parquet',side_effect=AssertionError('Replay must not read upstream questions')):
                self.assertEqual(replay.questions(),saved)
            with self.assertRaisesRegex(ValueError,'input profile'):replay.preflight()
        self.assertFalse(self.calls)
    def test_real_sdk_fresh_runs_make_new_embedding_calls(self):
        for name in ['first','second']:
            with self.client(name) as client,request_scope('query_embedding','q'):
                r=client.embeddings.create(model='test-model',input=['question'],dimensions=4,encoding_format='float')
                self.assertEqual(len(r.data[0].embedding),4)
        self.assertEqual(len(self.calls),2)
    def test_only_explicit_same_run_resume_can_reuse_a_response(self):
        with self.client('same') as client,request_scope('query_embedding','q'):
            client.embeddings.create(model='m',input=['q'],dimensions=4,encoding_format='float')
        with self.client('same') as client,request_scope('query_embedding','q'):
            with self.assertRaises(Exception):client.embeddings.create(model='m',input=['q'],dimensions=4,encoding_format='float')
        with self.client('same',resume=True) as client,request_scope('query_embedding','q'):
            client.embeddings.create(model='m',input=['q'],dimensions=4,encoding_format='float')
        self.assertEqual(len(self.calls),1)
    def test_chat_and_rerank_use_real_sdk_and_never_log_credentials(self):
        with self.client('sdk') as client:
            with request_scope('generation','q','baseline',1):
                result=client.chat.completions.create(model='m',messages=[dict(role='user',content='q')])
                self.assertEqual(result.choices[0].message.content,'saved unit answer')
            with request_scope('reranking','q'):
                result=client.post('/rerank',cast_to=dict[str, object],body=dict(model='r',query='q',documents=['x']*30,top_n=30))
                self.assertEqual(result['results'][0]['index'],5)
        for p in (self.root/'runs').rglob('*.json'):self.assertNotIn('unit-test-secret',p.read_text())
    def test_no_unscoped_or_passage_embedding_call(self):
        with self.client('blocked') as client:
            with self.assertRaises(Exception):client.embeddings.create(model='m',input=['x'])
            with request_scope('passage_embedding','q'):
                with self.assertRaises(Exception):client.embeddings.create(model='m',input=['x'])
        self.assertFalse(self.calls)
    def test_http_retry_cap_is_enforced_by_transport_not_sdk(self):
        attempts=[]
        def unavailable(request):attempts.append(1);return httpx.Response(503,json={'error':{'message':'temporary'}},request=request)
        with self.client('retry',wire=unavailable) as client,request_scope('query_embedding','q',max_attempts=3),patch('shared_scoring.live_client.time.sleep'):
            with self.assertRaises(Exception):client.embeddings.create(model='m',input=['q'])
        self.assertEqual(len(attempts),3)
    def test_temporary_in_flight_402_honors_cooldown_and_existing_cap(self):
        calls=[]
        def blocked(request):
            calls.append(1)
            return httpx.Response(402,json={'error':{'metadata':{'reason':'in_flight_budget_exhausted','headers':{'Retry-After':'120'}}}},request=request)
        with self.client('inflight',wire=blocked) as client,request_scope('reranking','q',max_attempts=3),patch('shared_scoring.live_client.time.sleep') as sleep:
            with self.assertRaises(Exception):client.post('/rerank',cast_to=dict[str,object],body={'query':'q'})
            self.assertEqual([c.args[0] for c in sleep.call_args_list],[120.,120.])
        with self.client('inflight',resume=True,wire=blocked,retry_in_flight=True) as client,request_scope('reranking','q',max_attempts=3):
            with self.assertRaises(Exception):client.post('/rerank',cast_to=dict[str,object],body={'query':'q'})
        self.assertEqual(len(calls),3)
        self.assertEqual(len(list((self.root/'runs/inflight/requests').glob('*/attempt-*.json'))),3)
    def test_generic_credit_402_is_not_retried(self):
        calls=[]
        def credits(request):
            calls.append(1);return httpx.Response(402,json={'error':{'message':'Insufficient credits'}},request=request)
        with self.client('credits',wire=credits) as client,request_scope('reranking','q'),patch('shared_scoring.live_client.time.sleep') as sleep:
            with self.assertRaises(Exception):client.post('/rerank',cast_to=dict[str,object],body={'query':'q'})
            sleep.assert_not_called()
        self.assertEqual(len(calls),1)
    def test_inspected_resume_keeps_failed_attempt_and_reuses_success_afterward(self):
        def blocked(request):
            return httpx.Response(402,json={'error':{'metadata':{'reason':'in_flight_budget_exhausted','headers':{'Retry-After':'120'}}}},request=request)
        with self.client('resume402',wire=blocked) as client,request_scope('reranking','q'),patch.object(TutorialTransport,'in_flight_delay',return_value=None):
            with self.assertRaises(Exception):client.post('/rerank',cast_to=dict[str,object],body={'query':'q'})
        old=next((self.root/'runs/resume402/requests').glob('*/attempt-01.json'))
        before=read_json(old)['raw_response_text']
        with self.client('resume402',resume=True,retry_in_flight=True) as client,request_scope('reranking','q'),patch('shared_scoring.live_client.time.sleep'):
            client.post('/rerank',cast_to=dict[str,object],body={'query':'q'})
        self.assertEqual(read_json(old)['raw_response_text'],before)
        self.assertEqual(len(list(old.parent.glob('attempt-*.json'))),2)
        with self.client('resume402',resume=True) as client,request_scope('reranking','q'):
            client.post('/rerank',cast_to=dict[str,object],body={'query':'q'})
        self.assertEqual(len(self.calls),1)
    def test_one_request_retry_authorization_preserves_identity_and_counter(self):
        import sqlite3
        def reset(request):raise httpx.ReadError('simulated connection reset',request=request)
        def call(client):return client.chat.completions.create(model='m',messages=[dict(role='user',content='q')])
        with self.client('grant',wire=reset) as client,request_scope('generation','q','baseline',1):
            with self.assertRaises(Exception):call(client)
        old=next((self.root/'runs/grant/requests').glob('*/attempt-01.json'));before=old.read_bytes()
        with self.client('grant',resume=True) as client,request_scope('generation','q','baseline',1):
            with self.assertRaises(Exception):call(client)
        grant={old.parent.name:{'max_total_attempts':2,'reason':'Unit-test approval of exactly one extra attempt'}}
        with self.client('grant',resume=True,retry_authorizations=grant) as client,request_scope('generation','q','baseline',1):call(client)
        self.assertEqual(old.read_bytes(),before)
        self.assertEqual(len(list(old.parent.glob('attempt-*.json'))),2)
        self.assertEqual(len(self.calls),1)
        with sqlite3.connect(self.root/'budget.sqlite') as db:
            self.assertEqual(db.execute('SELECT COUNT(*),SUM(count) FROM attempts').fetchone(),(1,2))
    def test_one_request_retry_grant_cannot_repeat_or_affect_other_requests(self):
        calls=[]
        def reset(request):calls.append(1);raise httpx.ReadError('simulated connection reset',request=request)
        def call(client):return client.chat.completions.create(model='m',messages=[dict(role='user',content='q')])
        with self.client('grant-cap',wire=reset) as client,request_scope('generation','q','baseline',1):
            with self.assertRaises(Exception):call(client)
        old=next((self.root/'runs/grant-cap/requests').glob('*/attempt-01.json'))
        grant={old.parent.name:{'max_total_attempts':2,'reason':'Unit-test single exception'}}
        for _ in range(2):
            with self.client('grant-cap',resume=True,wire=reset,retry_authorizations=grant) as client,request_scope('generation','q','baseline',1):
                with self.assertRaises(Exception):call(client)
        self.assertEqual(len(calls),2)
        with self.client('grant-cap',resume=True,wire=reset,retry_authorizations=grant) as client,request_scope('generation','different','baseline',1):
            with self.assertRaises(Exception):call(client)
        self.assertEqual(len(calls),3)
    def test_targeted_stage_uses_frozen_candidates_but_new_ranks(self):
        declared=read_json(self.root/'configs/tutorial_inputs.json');declared['question_counts']['targeted']=1;declared['targeted_answer_examples']=['q']
        write_json(self.root/'configs/tutorial_inputs.json',declared)
        run=TutorialRun(self.root,'targeted');target=run.upstream('targeted')
        (target/'input').mkdir(parents=True,exist_ok=True)
        pd.DataFrame([dict(query_id='q',question='what is q',reference_answers=['answer'],known_supporting_passage_ids=['p5'],cohort='opportunity')]).to_parquet(target/'input/questions.parquet',index=False)
        docs=pd.DataFrame([dict(query_id='q',passage_id=f'p{i}',text=str(i),rank=i+1,is_known_support=i==5,rerank_rank=30-i) for i in range(30)])
        source=target/'input/frozen_ranked_top30.parquet';docs.to_parquet(source,index=False);before=source.read_bytes()
        with patch('shared_scoring.live_client.httpx.HTTPTransport',return_value=httpx.MockTransport(self.wire)):
            run.preflight()
            run.rerank(lambda q,d:run.client.post('/rerank',cast_to=dict[str, object],body=dict(model='r',query=q,documents=[r['text'] for r in d],top_n=30)))
            run.diagnose()
            run.generate(lambda messages,settings,routing:run.client.chat.completions.create(model='m',messages=messages))
            self.assertEqual(len(read_json(run.folder/'generate_inputs.json')),2)
            self.assertEqual({r['query_id'] for r in read_json(run.folder/'generate_inputs.json')},{'q'})
            with self.assertRaisesRegex(RuntimeError,'not all complete'):run.finalize()
            answers=read_json(run.folder/'generate_inputs.json')
            score_rows=[dict(query_id=r['query_id'],arm=r['arm'],squad_token_f1_pct=50.,squad_exact_match_pct=0.,faithfulness=1.,faithfulness_status='ok',answer_relevancy=1.,answer_relevancy_status='ok') for r in answers]
            score_rows[-1]['faithfulness_status']='missing'
            (run.folder/'scores').mkdir()
            pd.DataFrame(score_rows).to_csv(run.folder/'scores/per_answer_scores.csv',index=False)
            write_json(run.folder/'scores.json',{'answers':2})
            state=run.finalize()
            self.assertEqual(state['state'],'complete_with_metric_failures')
            self.assertEqual(state['failed_or_missing_metrics'],1)
            self.assertEqual(read_json(run.folder/'manifest.json')['state'],state['state'])
            run.close()
        measured=pd.read_csv(run.folder/'retrieval_metrics.csv')
        self.assertEqual(measured.iloc[0].embedding_recall_at_5,0.)
        self.assertEqual(measured.iloc[0].reranker_recall_at_5,1.)
        self.assertEqual(source.read_bytes(),before)
    def test_partial_latency_resume_cannot_report_cached_http_as_new_timing(self):
        first=TutorialRun(self.root,'latency');write_json(first.folder/'manifest.json',{})
        write_json(first.folder/'requests/request/attempt-01.json',{'status':'ok'})
        resumed=TutorialRun(self.root,'latency',run_id=first.run_id,resume=True)
        with self.assertRaisesRegex(RuntimeError,'Partial benchmark'):
            resumed.benchmark(lambda *a:None,lambda *a:None,lambda *a:None)
    def test_demo_stages_make_four_fresh_sdk_calls_and_score_current_answers(self):
        import faiss
        import numpy as np
        from shared_scoring.provider import embedding_payload,rerank_payload
        execution=read_json(self.root/'configs/execution.json');execution['retrieval']['dimensions']=4
        write_json(self.root/'configs/execution.json',execution)
        run=TutorialRun(self.root,'demo');large=run.upstream('prepared_retrieval');large.mkdir(parents=True)
        rng=np.random.default_rng(42);vectors=rng.normal(size=(64,4)).astype('float32');vectors/=np.linalg.norm(vectors,axis=1,keepdims=True)
        faiss.omp_set_num_threads(1)
        index=faiss.IndexIVFFlat(faiss.IndexFlatIP(4),4,2,faiss.METRIC_INNER_PRODUCT)
        index.train(vectors);index.add(vectors)
        faiss.write_index(index,str(large/'index.faiss'));write_json(large/'assets_manifest.json',dict(nprobe=2))
        pd.DataFrame([dict(passage_id=f'p{i}',text=f'evidence {i}',source_url='https://example.test',truncated=False) for i in range(64)]).to_parquet(large/'corpus.parquet',index=False)
        targeted=run.upstream('targeted');(targeted/'input').mkdir(parents=True)
        pd.DataFrame([dict(query_id='1053665',question='what is unit evidence',reference_answers=['saved unit answer'],known_supporting_passage_ids=['p5'],cohort='opportunity')]).to_parquet(targeted/'input/questions.parquet',index=False)
        metric=SimpleNamespace(compute=lambda **kwargs:dict(exact_match=100.,f1=100.))
        with patch('shared_scoring.live_client.httpx.HTTPTransport',return_value=httpx.MockTransport(self.wire)),patch('shared_scoring.answers.load_squad_metric',return_value=metric):
            run.preflight()
            run.retrieve(lambda qs:run.client.embeddings.create(**embedding_payload('m',qs,4)))
            run.generate(lambda msgs,settings,routing:run.client.chat.completions.create(model='m',messages=msgs),arm='baseline')
            self.assertFalse((run.folder/'generate.json').exists())
            run.rerank(lambda q,docs:run.client.post('/rerank',cast_to=dict[str,object],body=rerank_payload('r',q,docs)))
            run.generate(lambda msgs,settings,routing:run.client.chat.completions.create(model='m',messages=msgs),arm='reranked')
            self.assertEqual(read_json(run.folder/'generate.json')['answers'],2)
            asyncio.run(run.score())
            self.assertEqual(run.finalize()['state'],'complete')
            run.close()
        self.assertEqual(len(self.calls),4)
        scores=pd.read_csv(run.folder/'scores/per_answer_scores.csv')
        self.assertEqual(scores.squad_token_f1_pct.tolist(),[100.,100.])
        self.assertTrue(all(r['run_id']==run.run_id for r in read_json(run.folder/'generate_inputs.json')))
        # Exercise the actual warm-up pipeline cells in Replay, in memory only.
        # No private cache, credentials, upstream query reads, or hosted client.
        previous=Path.cwd()
        try:
            os.chdir(self.root)
            namespace={'__name__':'__main__'}
            with patch.dict(os.environ,{'NOTEBOOK_MODE':'replay','TUTORIAL_RUN_ID':run.run_id,
                    'TUTORIAL_RESUME':'0','TUTORIAL_INPUTS':'','OPENROUTER_API_KEY':''}),patch('IPython.display.display'):
                for cell in read_json(PROJECT/NOTEBOOKS['demo'])['cells']:
                    if cell['cell_type']!='code':continue
                    result=eval(compile(''.join(cell['source']),'warmup-replay','exec',flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT),namespace)
                    if __import__('inspect').isawaitable(result):asyncio.run(result)
            self.assertEqual(namespace['question'],read_json(run.folder/'questions.json')[0]['question'])
            self.assertIsNone(namespace['client'])
            self.assertEqual(len(self.calls),4)
        finally:os.chdir(previous)
    def test_baseline_uses_declared_max_output_tokens_and_separate_diagnostic(self):
        run=TutorialRun(self.root,'baseline')
        source=run.upstream('baseline')/'runs/baseline-fresh-5413aa92e9c1375fa06f'
        write_json(source/'config.json',dict(models_and_dataset=dict(generator_model='m',generator_base='https://example.test',temperature=0.,max_output_tokens=400),system_prompt='Use evidence',generation_extras={'stream':False}))
        write_json(source/'questions.json',[dict(query_id=str(i),question='q',scope='baseline_31',reference_answers=['a'],gold_ids=[]) for i in range(31)])
        target=run.upstream('targeted')/'input';target.mkdir(parents=True)
        pd.DataFrame([dict(query_id='1053665',question='what is IMEI',reference_answers=['a'],known_supporting_passage_ids=['p'])]).to_parquet(target/'questions.parquet',index=False)
        questions=run.questions(persist=False)
        self.assertEqual(len(questions),32)
        self.assertEqual(run.generation['max_tokens'],400)
        self.assertEqual(sum(q['diagnostic'] for q in questions),1)
        self.assertFalse(run.folder.exists())

    def test_baseline_uses_the_notebook_retrieval_callback(self):
        run=TutorialRun(self.root,'baseline')
        run._assets=('haystack',None,{'p':{'id':'p','text':'evidence'}})
        seen=[]
        def retrieve_top30(vector):
            seen.append(vector)
            return [SimpleNamespace(id='p',score=0.8)]
        rows=run._search([[1.,0.]], [{'query_id':'q','question':'query','gold_ids':['p']}],retrieve_api=retrieve_top30)
        self.assertEqual(seen,[[1.,0.]])
        self.assertEqual(rows[0]['passage_id'],'p')
        self.assertEqual(rows[0]['rank'],1)
        self.assertTrue(rows[0]['is_known_support'])

    def test_notebooks_have_visible_sdk_calls_and_active_sections(self):
        for study,relative in NOTEBOOKS.items():
            n=read_json(PROJECT/relative);active=[c for c in n['cells'] if c['metadata'].get('live_tutorial')]
            code='\n'.join(''.join(c['source']) for c in active if c['cell_type']=='code')
            self.assertIn('"live"',code);self.assertNotIn('ensure_baseline_study(',code)
            if study in {'demo','baseline','diagnosis','latency'}:self.assertIn('client.embeddings.create(',code)
            if study in {'demo','baseline','answers','latency'}:self.assertIn('client.chat.completions.create(',code)
            for cell in n['cells']:
                if cell['cell_type']=='code':compile(''.join(cell['source']),relative,'exec',flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
                if cell['cell_type']=='code':
                    self.assertTrue(cell['metadata'].get('live_tutorial'))
                    self.assertNotIn('skip-execution',cell['metadata'].get('tags',[]))
                    self.assertNotIn('if False:', ''.join(cell['source']))
