"""Exercise adapters with synthetic API-shaped fixtures; no provider recordings."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
import numpy as np
from openai import AsyncOpenAI

from shared_scoring.answers import remove_passage_citations, score_custom_answer
from shared_scoring.fresh_retrieval import RecordedOpenRouterClient, execution_settings
from shared_scoring.provider import QUERY_INSTRUCTION, embedding_payload, rerank_payload, rerank_scores
from test_fresh_execution import Response

PROJECT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / 'fixtures/openrouter'


class SyntheticTransport:
    def __init__(self, fixture):
        self.fixture, self.calls = fixture, 0

    def post(self, url, json, headers, timeout=None):
        if url != self.fixture['url'] or json != self.fixture['request_body']:
            raise AssertionError('Adapter changed the synthetic endpoint or request body')
        self.calls += 1
        response = Response(self.fixture['raw_response_json'])
        response.text = self.fixture['raw_response_text']
        return response


class SyntheticContractTests(unittest.TestCase):
    def fixture(self, stage):
        return json.loads((FIXTURES / f'{stage}.json').read_text())

    def test_fixture_checksums_and_exact_response_text(self):
        manifest = json.loads((FIXTURES / 'manifest.json').read_text())
        self.assertEqual(manifest['fixture_kind'], 'synthetic API-shaped fixtures')
        self.assertEqual(len(manifest['fixtures']), 3)
        for entry in manifest['fixtures']:
            path = FIXTURES / entry['fixture']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry['fixture_sha256'])
            fixture = json.loads(path.read_text())
            self.assertEqual(fixture['fixture_kind'], 'synthetic API-shaped fixture')
            self.assertEqual(fixture['http_status'], 200)
            self.assertEqual(json.loads(fixture['raw_response_text']), fixture['raw_response_json'])

    def test_embedding_adapter_orders_synthetic_vectors_and_replays_cache(self):
        fixture = self.fixture('query_embedding')
        body = fixture['request_body']
        prefix = f'Instruct: {QUERY_INSTRUCTION}\nQuery: '
        self.assertTrue(all(text.startswith(prefix) for text in body['input']))
        texts = [text[len(prefix):] for text in body['input']]
        self.assertEqual(embedding_payload(body['model'], texts, body['dimensions']), body)
        transport = SyntheticTransport(fixture)
        with tempfile.TemporaryDirectory() as folder:
            client = RecordedOpenRouterClient(execution_settings(PROJECT)['retrieval'], folder, transport)
            with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'fixture', 'ALLOW_MODEL_CALLS': '1'}):
                actual, trace = client.embed(texts, 'query', body['dimensions'])
            expected = [row['embedding'] for row in sorted(fixture['raw_response_json']['data'], key=lambda r: r['index'])]
            np.testing.assert_array_equal(actual, np.asarray(expected, dtype='float32'))
            self.assertFalse(trace['cache_hit'])
            with patch.dict(os.environ, {}, clear=True):
                replayed, trace = client.embed(texts, 'query', body['dimensions'])
            np.testing.assert_array_equal(actual, replayed)
            self.assertTrue(trace['cache_hit'])
            self.assertEqual(transport.calls, 1)

    def test_rerank_adapter_orders_synthetic_permutation_and_replays_cache(self):
        fixture = self.fixture('reranking')
        body = fixture['request_body']
        candidates = [{'text': text} for text in body['documents']]
        self.assertEqual(rerank_payload(body['model'], body['query'], candidates), body)
        transport = SyntheticTransport(fixture)
        with tempfile.TemporaryDirectory() as folder:
            client = RecordedOpenRouterClient(execution_settings(PROJECT)['retrieval'], folder, transport)
            with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'fixture', 'ALLOW_MODEL_CALLS': '1'}):
                ranked, trace = client.rerank(body['query'], candidates)
            self.assertFalse(trace['cache_hit'])
            with patch.dict(os.environ, {}, clear=True):
                replayed, trace = client.rerank(body['query'], candidates)
            self.assertEqual(replayed, ranked)
            self.assertTrue(trace['cache_hit'])
        self.assertEqual(ranked, rerank_scores(fixture['raw_response_json'], len(candidates)))
        self.assertEqual([row['index'] for row in ranked], [1, 0, 2])
        self.assertEqual(candidates[ranked[0]['index']]['text'], "The Luma sensor's reset button is beneath its battery cover.")
        self.assertEqual(sorted(row['index'] for row in ranked), list(range(len(candidates))))
        self.assertEqual(transport.calls, 1)

    def test_synthetic_answer_citation_scoring_and_cached_generation(self):
        fixture = self.fixture('generation')
        body = fixture['request_body']
        transport = SyntheticTransport(fixture)
        with tempfile.TemporaryDirectory() as folder:
            client = RecordedOpenRouterClient(execution_settings(PROJECT)['retrieval'], folder, transport)
            with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'fixture', 'ALLOW_MODEL_CALLS': '1'}):
                raw, trace = client.request('generation', fixture['url'], body)
            self.assertFalse(trace['cache_hit'])
            with patch.dict(os.environ, {}, clear=True):
                repeated, trace = client.request('generation', fixture['url'], body)
            self.assertEqual(raw, repeated)
            self.assertTrue(trace['cache_hit'])
        self.assertEqual(transport.calls, 1)
        answer = raw['choices'][0]['message']['content']
        self.assertIn('[p_luma]', answer)
        self.assertIn('[p_luma]', body['messages'][1]['content'])
        reference = "The Luma sensor's reset button is beneath its battery cover."
        pattern = r'\[p_[A-Za-z0-9_]+\]'
        self.assertEqual(remove_passage_citations(answer, pattern).strip(), reference)
        self.assertEqual(score_custom_answer(answer, [reference], pattern), {'normalized_em': 1., 'token_f1': 1.})
        self.assertLess(score_custom_answer("The Vela button is beside its display. [p_vela]", [reference], pattern)['token_f1'], 1.)
        self.assertEqual(remove_passage_citations('[note] ' + answer, pattern).strip(), '[note] ' + reference)

    def test_real_openai_sdk_serializes_and_parses_synthetic_chat_and_embeddings(self):
        async def check():
            for stage, endpoint in [('generation', '/chat/completions'), ('query_embedding', '/embeddings')]:
                fixture = self.fixture(stage)
                calls = []
                def respond(request):
                    self.assertEqual(str(request.url), fixture['url'])
                    payload = json.loads(request.content)
                    self.assertEqual(payload, fixture['request_body'])
                    calls.append(payload)
                    return httpx.Response(fixture['http_status'], text=fixture['raw_response_text'],
                                          headers={'Content-Type': 'application/json'})
                async with AsyncOpenAI(api_key='fixture', base_url=fixture['url'][:-len(endpoint)],
                    max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))) as client:
                    body = dict(fixture['request_body'])
                    if stage == 'generation':
                        extras = {key: body.pop(key) for key in ['provider'] if key in body}
                        result = await client.chat.completions.create(**body, extra_body=extras)
                        expected = fixture['raw_response_json']
                        self.assertEqual(result.choices[0].message.content, expected['choices'][0]['message']['content'])
                        self.assertEqual(result.choices[0].finish_reason, expected['choices'][0]['finish_reason'])
                        self.assertEqual(result.usage.completion_tokens, expected['usage']['completion_tokens'])
                    else:
                        result = await client.embeddings.create(**body)
                        self.assertEqual(result.data[0].embedding, fixture['raw_response_json']['data'][0]['embedding'])
                        self.assertEqual(len(result.data[0].embedding), body['dimensions'])
                self.assertEqual(len(calls), 1)
        asyncio.run(check())
