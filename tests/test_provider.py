import asyncio
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from shared_scoring.fresh_retrieval import RecordedOpenRouterClient, execution_settings
from shared_scoring.provider import embedding_payload, rerank_payload, rerank_scores
from shared_scoring.provenance import evaluation_fingerprint
from test_fresh_execution import Session

PROJECT = Path(__file__).resolve().parents[1]


class ProviderTests(unittest.TestCase):
    def test_all_active_model_settings_use_requested_qwen_models(self):
        paths = [PROJECT / 'configs/execution.json', PROJECT / 'configs/experiment1.json',
                 *PROJECT.glob('experiments/*/config.json')]
        for path in paths:
            cfg = json.loads(path.read_text())
            def check(value):
                if isinstance(value, dict):
                    for key, item in value.items():
                        if key in {'embedding_model'}:
                            self.assertEqual(item, 'qwen/qwen3-embedding-4b', str(path))
                        if key == 'reranker_model':
                            self.assertEqual(item, 'qwen/qwen3-reranker-8b')
                        if key in {'generator_model', 'judge_model'}:
                            self.assertEqual(item, 'qwen/qwen3-30b-a3b-instruct-2507')
                        if key in {'dimensions', 'embedding_dimensions'}:
                            self.assertEqual(item, 2560)
                        self.assertNotIn(key, {'truncate', 'embedding_truncate', 'chat_template_kwargs'})
                        check(item)
                elif isinstance(value, list):
                    for item in value: check(item)
                elif isinstance(value, str):
                    self.assertNotIn('nvidia', value.lower(), str(path))
            check(cfg)

    def test_embedding_contract_formats_queries_and_leaves_passages_unchanged(self):
        cfg = execution_settings(PROJECT)['retrieval']
        query = embedding_payload(cfg['embedding_model'], ['How?'], 2560, 'query', 'Find evidence')
        self.assertEqual(query, {'model': cfg['embedding_model'], 'input': ['Instruct: Find evidence\nQuery: How?'],
                                 'dimensions': 2560, 'encoding_format': 'float'})
        self.assertEqual(embedding_payload(cfg['embedding_model'], ['Raw text'], 2560, 'passage')['input'], ['Raw text'])

    def test_reranker_contract_orders_relevance_scores_and_rejects_bad_permutations(self):
        payload = rerank_payload('qwen/qwen3-reranker-8b', 'Question', [{'text': 'A'}, {'text': 'B'}])
        self.assertEqual(payload, {'model': 'qwen/qwen3-reranker-8b', 'query': 'Question', 'documents': ['A', 'B'], 'top_n': 2})
        response = {'results': [{'index': 1, 'relevance_score': .8}, {'index': 0, 'relevance_score': .2}]}
        self.assertEqual([row['index'] for row in rerank_scores(response, 2)], [1, 0])
        for results in [[{'index': 0, 'relevance_score': .2}] * 2,
                        [{'index': 0, 'relevance_score': float('nan')}, {'index': 1, 'relevance_score': .8}],
                        [{'index': True, 'relevance_score': .2}, {'index': 0, 'relevance_score': .8}]]:
            with self.assertRaises(ValueError): rerank_scores({'results': results}, 2)

    def test_cache_invalidates_for_preprocessing_model_dimensions_and_endpoint(self):
        settings = execution_settings(PROJECT)['retrieval']
        session = Session()
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-key', 'ALLOW_MODEL_CALLS': '1'}):
            for change in [{}, {'query_instruction': 'Different instruction'}, {'embedding_model': 'qwen/another'},
                           {'api_base': 'https://other.example/api/v1'}, {'dimensions': 5}]:
                cfg = {**settings, **change}
                RecordedOpenRouterClient(cfg, folder, session).embed(['Question'], 'query', cfg.get('dimensions', 4))
            self.assertEqual(len(session.calls), 5)
            RecordedOpenRouterClient(settings, folder, session).embed(['Question'], 'query', settings['dimensions'])
            self.assertEqual(len(session.calls), 5)
            for path in Path(folder).rglob('*.json'): self.assertNotIn('test-key', path.read_text())

    def test_model_calls_default_off_even_with_a_key(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'OPENROUTER_API_KEY': 'unused'}):
            os.environ.pop('ALLOW_MODEL_CALLS', None)
            session = Session()
            client = RecordedOpenRouterClient(execution_settings(PROJECT)['retrieval'], folder, session)
            with self.assertRaisesRegex(PermissionError, 'disabled'): client.embed(['test'], 'query', 4)
            self.assertEqual(session.calls, [])

    def test_judge_fingerprint_includes_query_preprocessing(self):
        cfg = json.loads((PROJECT / 'configs/experiment1.json').read_text())['judge']
        row = {'question': 'Q', 'answer': 'A', 'retrieved_contexts': ['P']}
        first = evaluation_fingerprint(row, 'answer_relevancy', cfg, '0.4.3')
        second = evaluation_fingerprint(row, 'answer_relevancy', {**cfg, 'query_instruction': 'Other'}, '0.4.3')
        self.assertNotEqual(first, second)

    def test_notebooks_preserve_outputs_without_errors_or_hardcoded_result_tables(self):
        for path in [*PROJECT.glob('experiments/*/*.ipynb'), *PROJECT.glob('diagnosis-retrieval-opportunity/*.ipynb')]:
            if any(part.startswith('.') for part in path.relative_to(PROJECT).parts): continue
            nb = json.loads(path.read_text())
            for cell in nb['cells']:
                if cell['cell_type'] == 'code':
                    self.assertFalse(any(output['output_type'] == 'error' for output in cell['outputs']))
                    source = ''.join(cell['source'])
                    self.assertNotIn('recorded_pairs =', source)
                    self.assertNotIn("'truncate': 'NONE'", source)
            self.assertIn('Execution and replay', ''.join(nb['cells'][0]['source']))

    def test_real_ragas_components_construct_without_network(self):
        from openai import AsyncOpenAI
        import httpx
        from shared_scoring.judge import create_ragas_metrics
        from ragas.embeddings.base import BaseRagasEmbedding
        class Embeddings(BaseRagasEmbedding):
            def embed_text(self, text, **kwargs): return [1., 0.]
            async def aembed_text(self, text, **kwargs): return [1., 0.]
            def embed_query(self, text): return [1., 0.]
            def embed_documents(self, texts): return [[1., 0.] for _ in texts]
            async def aembed_query(self, text): return [1., 0.]
            async def aembed_documents(self, texts): return [[1., 0.] for _ in texts]
        async def check():
            client = AsyncOpenAI(api_key='test', base_url='https://openrouter.ai/api/v1',
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda request: (_ for _ in ()).throw(AssertionError('No requests')))))
            cfg = json.loads((PROJECT / 'configs/experiment1.json').read_text())['judge']
            try:
                metrics = create_ragas_metrics(client, Embeddings(), cfg)
                self.assertEqual(set(metrics), {'faithfulness', 'answer_relevancy'})
            finally: await client.close()
        asyncio.run(check())

    def test_real_judge_embedding_adapter_records_openrouter_wire_format(self):
        import httpx
        from shared_scoring.judge import _make_components
        calls = []
        class Transport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                payload = json.loads(await request.aread())
                calls.append(payload)
                return httpx.Response(200, json={'data': [{'index': 0, 'embedding': [1., 0., 0., 0.]}],
                    'model': payload['model'], 'usage': {'prompt_tokens': 1, 'total_tokens': 1}}, request=request)
            async def aclose(self): pass
        async def check(folder):
            settings = json.loads((PROJECT / 'configs/experiment1.json').read_text())['judge']
            settings.update(embedding_dimensions=4, embedding_question_policy='evaluate', request_min_interval_seconds=0)
            with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test', 'ALLOW_MODEL_CALLS': '1'}), \
                 patch('httpx.AsyncHTTPTransport', Transport):
                client, metrics = _make_components(PROJECT, 'test', folder, settings)
                try:
                    embeddings = metrics['answer_relevancy'].embeddings
                    self.assertEqual(await embeddings.aembed_text('Question'), [1., 0., 0., 0.])
                    self.assertEqual(await embeddings.aembed_text('Question'), [1., 0., 0., 0.])
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(calls[0], embedding_payload(settings['embedding_model'], ['Question'], 4,
                        instruction=settings['query_instruction']))
                    # Independent SDK invocations may repeat a RAGAS prompt;
                    # they are not retries of the original transport request.
                    for _ in range(4):
                        await client.embeddings.create(**embedding_payload(settings['embedding_model'], ['Other question'], 4,
                            instruction=settings['query_instruction']))
                    self.assertEqual(len(calls), 5)
                finally: await client.close()
        with tempfile.TemporaryDirectory() as temp: asyncio.run(check(Path(temp)))
