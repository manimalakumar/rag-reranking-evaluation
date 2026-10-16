"""Offline checks for saved-first execution and real small-corpus reconstruction."""
import asyncio
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from shared_scoring import load_configuration
from shared_scoring.execution import (
    ensure_baseline_study, ensure_targeted_study, ensure_answer_quality_study,
    ensure_latency_study, ensure_diagnostic_data, select_cohorts,
)
from shared_scoring.fresh_retrieval import (
    RecordedOpenRouterClient, execution_settings, prepare_dataset,
    ensure_retrieval_assets, retrieve_full_validation, fingerprint,
)
from shared_scoring.io import read_json, write_json, sha256_file
from shared_scoring.opportunity_study import load_opportunity_configuration, refresh_opportunity_results

PROJECT = Path(__file__).resolve().parents[1]


class Response:
    status_code = 200
    ok = True
    def __init__(self, data):
        self.data, self.text = data, json.dumps(data)
    def json(self):
        return self.data
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError('Simulated HTTP failure')


class Session:
    """Deterministic test transport; never connects to a network."""
    def __init__(self):
        self.calls = []
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def post(self, url, json, headers, timeout=None):
        self.calls.append((url, copy.deepcopy(json)))
        if url.endswith('/embeddings'):
            vectors = []
            for index, text in enumerate(json['input']):
                seed = int(fingerprint(text)[:8], 16)
                vectors.append({'index': index, 'embedding': np.random.default_rng(seed).normal(size=json['dimensions']).tolist()})
            return Response({'data': vectors, 'model': json['model']})
        if url.endswith('/rerank'):
            return Response({'results': [{'index': i, 'relevance_score': float(-i)} for i in range(len(json['documents']))]})
        return Response({'choices':[{'message':{'content':'A saved test answer [p_00000001]'},'finish_reason':'length'}],
                         'model':json['model']})


class AsyncSession(Session):
    def __init__(self, *args, **kwargs):
        super().__init__()
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        pass
    async def post(self, *args, **kwargs):
        return super().post(*args, **kwargs)


def raw_rows():
    return [{'query_id':i, 'query':f'question {i}', 'answers':[f'answer {i}'],
        'wellFormedAnswers':[f'well formed answer {i}',f'answer {i}'],
        'passages':{'passage_text':[f'passage {i}-{j} '+('x'*80) for j in range(8)],
                    'url':['https://example.test']*8,'is_selected':[0,0,0,0,0,1,0,0]}} for i in range(8)]


class FreshExecutionTests(unittest.TestCase):
    def setUp(self):
        allow = patch.dict(os.environ, {'ALLOW_MODEL_CALLS': '1'})
        allow.start()
        self.addCleanup(allow.stop)
        self.guard = patch.object(socket.socket,'connect',side_effect=AssertionError('Network forbidden'))
        self.guard.start()
        self.addCleanup(self.guard.stop)


    def test_dataset_download_is_pinned_and_preserves_every_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            settings = execution_settings(PROJECT)
            with patch('shared_scoring.fresh_retrieval.load_raw_dataset',return_value=raw_rows()) as download:
                prepare_dataset(settings,Path(temp))
            download.assert_called_once()
            frame = pd.read_parquet(Path(temp)/'queries.parquet')
            self.assertEqual(list(frame.iloc[0].reference_answers),['answer 0','well formed answer 0'])
            with patch('shared_scoring.fresh_retrieval.load_raw_dataset',side_effect=AssertionError('Download forbidden')):
                prepare_dataset(settings,Path(temp))
            with (Path(temp)/'queries.parquet').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'dataset changed'):
                prepare_dataset(settings,Path(temp))

    def test_request_resume_retains_raw_traces_without_credentials(self):
        with tempfile.TemporaryDirectory() as temp:
            settings = execution_settings(PROJECT)['retrieval']
            client = RecordedOpenRouterClient(settings,temp,Session())
            with patch.dict(os.environ,{'OPENROUTER_API_KEY':'test-credential'}):
                first,_ = client.embed(['test'],'query',4)
            with patch.dict(os.environ,{},clear=True):
                second,_ = client.embed(['test'],'query',4)
            np.testing.assert_array_equal(first,second)
            self.assertEqual(len(client.session.calls),1)
            trace = next(Path(temp).glob('*/attempt-*.json')).read_text()
            self.assertNotIn('test-credential',trace)
            self.assertIn('raw_response_text',trace)
            self.assertIn('elapsed_seconds',trace)

    def test_failed_attempts_are_retained_and_never_silently_retried(self):
        class Broken(Session):
            def post(self,*args,**kwargs):
                raise TimeoutError('simulated timeout')
        with tempfile.TemporaryDirectory() as temp:
            settings = {**execution_settings(PROJECT)['retrieval'],'max_attempts':2,'backoff_seconds':[0]}
            client = RecordedOpenRouterClient(settings,temp,Broken())
            with patch.dict(os.environ,{'OPENROUTER_API_KEY':'test-credential'}):
                with self.assertRaisesRegex(RuntimeError,'exhausted'):
                    client.embed(['test'],'query',4)
                self.assertEqual(len(list(Path(temp).glob('*/attempt-*.json'))),2)
                with self.assertRaisesRegex(RuntimeError,'retained'):
                    client.embed(['test'],'query',4)

    @unittest.skipUnless(importlib.util.find_spec('faiss'),'Install requirements-runtime.txt to test fresh FAISS preparation')
    def test_empty_cache_builds_real_index_and_resumes_without_hosted_calls(self):
        import faiss
        with tempfile.TemporaryDirectory() as temp:
            settings = execution_settings(PROJECT)
            settings['retrieval'].update(dimensions=4,nlist=2,passage_batch_size=16,query_batch_size=3)
            client = RecordedOpenRouterClient(settings['retrieval'],Path(temp)/'requests',Session())
            with patch.dict(os.environ,{'OPENROUTER_API_KEY':'test-credential'}):
                metrics,shards,assets = retrieve_full_validation(settings,temp,client,raw_rows())
            count = len(client.session.calls)
            self.assertEqual(len(metrics),8)
            self.assertEqual(sum(len(pd.read_parquet(p)) for p in shards),8*30)
            index=faiss.read_index(str(assets['index_file']))
            self.assertEqual((index.d,index.ntotal,index.nprobe),(4,64,2))
            with patch.dict(os.environ,{},clear=True):
                repeated,_,_=retrieve_full_validation(settings,temp,client)
            pd.testing.assert_frame_equal(metrics,repeated)
            self.assertEqual(count,len(client.session.calls))
            with assets['index_file'].open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):
                ensure_retrieval_assets(settings,temp,client)

    def test_opportunity_count_is_observed_and_control_draw_is_seeded(self):
        metrics=pd.DataFrame([{'query_row':i,'query_id':str(i),'has_known_support_labels':True,
            'embedding_recall_at_5':0 if i<7 else 1,'embedding_recall_at_30':1} for i in range(20)])
        settings=execution_settings(PROJECT)
        settings['cohorts'].update(opportunity_questions=7,control_questions=4,pilot_controls=2,pilot_opportunities=3,expansion_batches=3)
        first=select_cohorts(metrics,settings)
        self.assertEqual(first.cohort.value_counts().to_dict(),{'opportunity':7,'control':4})
        pd.testing.assert_frame_equal(first,select_cohorts(metrics,settings))



    def test_fresh_targeted_generation_and_latency_share_real_helpers(self):
        from shared_scoring.latency_study import validate_saved_latency_study
        from shared_scoring.paired_tutorial import load_answer_quality_study, verify_answer_quality_sample
        from shared_scoring.latency_tutorial import load_latency_tutorial
        from fakes import FakeIndex, FakeCorpus
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()
            (root/'configs').mkdir()
            settings=read_json(PROJECT/'configs/execution.json')
            settings['retrieval'].update(dimensions=4,max_attempts=1,backoff_seconds=[])
            settings['cohorts'].update(opportunity_questions=3,control_questions=2,pilot_controls=1,pilot_opportunities=1,expansion_batches=3)
            write_json(root/'configs/execution.json',settings)
            for version in ['v2-targeted-reranking','v3-answer-quality','v4-latency-tradeoff']:
                folder=root/'experiments'/version
                folder.mkdir(parents=True)
                config=read_json(PROJECT/'experiments'/version/'config.json')
                config['squad']['metric_source']=str(PROJECT/'scoring-study/output/squad_metric_sources/squad.py')
                config['generation']['request_min_interval_seconds']=0
                config['inspect_query_id']='0'
                if version!='v2-targeted-reranking':
                    config['sampling']['questions_per_stratum']=1
                if version=='v4-latency-tradeoff':
                    config['benchmark']['embedding_settings']['dimensions']=4
                write_json(folder/'config.json',config)
            metrics=[];passages=[]
            for qid,name in enumerate(['gain','unchanged','loss','all-control','absent-control']):
                gold=['p_00000008'] if qid<2 else ['p_00000001','p_00000008'] if qid==2 else ['p_00000001'] if qid==3 else ['p_00000001','p_absent']
                r5=0 if qid<2 else .5 if qid in [2,4] else 1
                r30=.5 if qid==4 else 1
                row={'query_row':qid,'query_id':str(qid),'question':name,'known_supporting_passage_ids':gold,
                    'has_known_support_labels':True,'reference_answers':['A saved test answer'],
                    'original_answers':['A saved test answer'],'well_formed_answers':[],
                    'embedding_recall_at_5':r5,'embedding_recall_at_30':r30,'recall_gap_30_minus_5':r30-r5}
                metrics.append(row)
                for rank in range(1,31):
                    pid=f'p_{rank:08x}'
                    passages.append({**row,'passage_id':pid,'rank':rank,'text':f'passage {rank}',
                        'source_url':'https://example.test','truncated':False,
                        'similarity_score':1-rank/100,'is_known_support':pid in gold})
            shard=root/'retrieval.parquet'
            pd.DataFrame(passages).to_parquet(shard,index=False)
            assets={'contract':{'dataset':{'validation_questions':5,'unique_passages':32}},'actual_nlist':2,'nprobe':2}
            class RerankingClient:
                def rerank(self,question,candidates):
                    order=list(range(30))
                    if question=='gain':order=[7]+[i for i in order if i!=7]
                    if question=='loss':order=[i for i in order if i not in [0,7]]+[0,7]
                    return [{'index':i,'logit':float(-rank)} for rank,i in enumerate(order)],{'elapsed_seconds':.1,'attempt':1}
            with patch('shared_scoring.execution.retrieve_full_validation',return_value=(pd.DataFrame(metrics),[shard],assets)):
                parent=ensure_targeted_study(root/'experiments/v2-targeted-reranking/config.json',root,client=RerankingClient())
                diagnosis=ensure_diagnostic_data(root/'diagnosis-retrieval-opportunity',root)
            self.assertEqual(parent['expected_cohort_counts'],{'opportunity':3,'control':2})
            self.assertEqual(read_json(diagnosis/'reported_full_validation_summary.json')['opportunity_questions'],3)
            self.assertIn('Computed',read_json(diagnosis/'reported_full_validation_summary.json')['source'])
            with patch.dict(os.environ,{'OPENROUTER_API_KEY':'test-credential'}),patch('httpx.AsyncClient',AsyncSession):
                answer=asyncio.run(ensure_answer_quality_study(root/'experiments/v3-answer-quality/config.json',root))
            study=load_answer_quality_study(answer)
            self.assertEqual(len(study['complete_ids']),5)
            self.assertEqual(len(study['records']),10)
            self.assertEqual(sum(row['truncated'] for row in study['inputs']),10)
            verify_answer_quality_sample(study)
            from shared_scoring import paired_tutorial
            from IPython.utils.capture import capture_output
            with capture_output():
                paired_tutorial.show_answer_quality_sample(study)
                paired_tutorial.show_generation_coverage(study)
            calls=[]
            class CountingAsyncSession(AsyncSession):
                async def post(self,*args,**kwargs):
                    calls.append(args)
                    return await super().post(*args,**kwargs)
            with patch.dict(os.environ,{'OPENROUTER_API_KEY':'test-credential'}),patch('httpx.AsyncClient',CountingAsyncSession):
                repeated=asyncio.run(ensure_answer_quality_study(root/'experiments/v3-answer-quality/config.json',root))
            self.assertEqual(calls,[])
            self.assertEqual(answer['checkpoint_file'],repeated['checkpoint_file'])
            corpus=root/'corpus.parquet';index=root/'index.faiss'
            corpus.write_bytes(b'test-corpus');index.write_bytes(b'test-index')
            def prepared(config,project):
                config['benchmark']['corpus_rows'] = 30
                config['benchmark']['asset_sizes'] = {'corpus': corpus.stat().st_size, 'index': index.stat().st_size}
                return config,corpus,index
            with patch.dict(os.environ,{'OPENROUTER_API_KEY':'test-credential'}), \
                 patch('shared_scoring.execution.ensure_latency_assets',prepared), \
                 patch('shared_scoring.latency_benchmark._load_assets',return_value=(FakeIndex(),FakeCorpus())), \
                 patch('requests.Session',Session):
                latency=asyncio.run(ensure_latency_study(root/'experiments/v4-latency-tradeoff/config.json',root))
            measured=load_latency_tutorial(latency)
            self.assertEqual(len(measured['complete_ids']),5)
            self.assertEqual(len(measured['latency']['attempts']),25)
            validate_saved_latency_study(latency)
            with patch('shared_scoring.execution.ensure_latency_assets',side_effect=AssertionError('Asset loading forbidden')), \
                 patch('requests.Session',side_effect=AssertionError('Hosted session forbidden')):
                again=asyncio.run(ensure_latency_study(root/'experiments/v4-latency-tradeoff/config.json',root))
            self.assertEqual(again['checkpoint_file'],latency['checkpoint_file'])

    @unittest.skipUnless(importlib.util.find_spec('haystack'),'Install requirements-runtime.txt to test fresh baseline preparation')
    def test_fresh_baseline_uses_curated_ids_and_independent_generation_calls(self):
        from shared_scoring.fresh_baseline import build_baseline_run, prepare_baseline_scores
        from shared_scoring.records import load_saved_scores
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();source=root/'source';(source/'input').mkdir(parents=True)
            config=load_configuration(PROJECT/'configs/experiment1.json')
            config.update(source_directory=source,diagnostic_ids=['4'])
            settings=read_json(PROJECT/'experiments/v1-baseline/config.json')
            settings['models_and_dataset'].update(embedding_dimensions=4,embedding_batch_size=16,distractor_questions=0)
            settings['diagnostic_query_id']='4'
            write_json(source/'config.json',settings)
            for name in ['experiment_helpers.py','scoring.py']:
                shutil.copy2(PROJECT/'experiments/v1-baseline'/name,source/name)
            rows=raw_rows()
            write_json(source/'input/baseline_ids.json',[str(i) for i in range(4)])
            write_json(source/'input/golden_reviewed.json',[{'query id':i,'query':rows[i]['query'],'answer':json.dumps(rows[i]['answers']),
                'decision':'keep','why':'Test scientific input',
                'related passages':f"Passage 1\n{rows[i]['passages']['passage_text'][5]}\nSource: https://example.test"} for i in range(4)])
            transport=Session()
            client=RecordedOpenRouterClient(execution_settings(PROJECT)['retrieval'],root/'request-cache',transport)
            with patch.dict(os.environ,{'OPENROUTER_API_KEY':'test-credential'}):
                fresh=build_baseline_run(config,PROJECT,root/'fresh',client,rows)
            generation_calls=[payload for url,payload in transport.calls if url.endswith('/chat/completions')]
            self.assertEqual(len(generation_calls),10)
            self.assertEqual(fresh['run_id'],'baseline-fresh-fresh')
            fresh['score_directory']=root/'fresh/standard_scoring'
            prepare_baseline_scores(fresh)
            saved=load_saved_scores(fresh)
            self.assertEqual(saved['audit']['complete_baseline_pairs'],4)
            self.assertEqual(len(saved['diagnostic_scores']),2)
            self.assertTrue(saved['scores'].truncated.all())
            before=len(transport.calls)
            with patch.dict(os.environ,{},clear=True):
                build_baseline_run(config,PROJECT,root/'fresh',client,rows)
            self.assertEqual(before,len(transport.calls))


if __name__ == '__main__':
    unittest.main()
