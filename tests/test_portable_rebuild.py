"""Rebuild retrieval, reranking and scored answer pairs from an empty project cache."""
import asyncio
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import faiss
import pandas as pd

from shared_scoring.execution import ensure_targeted_study, ensure_answer_quality_study
from shared_scoring.fresh_retrieval import RecordedOpenRouterClient, execution_settings
from shared_scoring.io import read_json, write_json
from shared_scoring.opportunity_study import refresh_opportunity_results, validate_frozen_opportunity_inputs
from test_fresh_execution import Session, AsyncSession, raw_rows

PROJECT = Path(__file__).resolve().parents[1]


class PortableRebuildTests(unittest.TestCase):
    def test_empty_project_builds_real_retrieval_reranking_and_scored_answers_then_replays(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = read_json(PROJECT / 'configs/execution.json')
            settings['retrieval'].update(dimensions=4, nlist=2, passage_batch_size=16, query_batch_size=3)
            settings['cohorts'].update(opportunity_questions=1, control_questions=1,
                pilot_opportunities=1, pilot_controls=1, expansion_batches=0)
            write_json(root / 'configs/execution.json', settings)
            configs = {}
            for version in ['v2-targeted-reranking', 'v3-answer-quality']:
                config = read_json(PROJECT / 'experiments' / version / 'config.json')
                config['squad']['metric_source'] = str(PROJECT / 'scoring-study/output/squad_metric_sources/squad.py')
                config['generation']['request_min_interval_seconds'] = 0
                if 'sampling' in config:
                    config['sampling']['questions_per_stratum'] = 1
                configs[version] = root / 'experiments' / version / 'config.json'
                write_json(configs[version], config)
            runtime = execution_settings(root)['runtime_directory']
            self.assertFalse(runtime.exists())
            transport = Session()
            client = RecordedOpenRouterClient(execution_settings(root)['retrieval'], root / 'request-cache', transport)
            with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'fixture', 'ALLOW_MODEL_CALLS': '1'}):
                targeted = ensure_targeted_study(configs['v2-targeted-reranking'], root, client=client, rows=raw_rows())
                with patch('httpx.AsyncClient', AsyncSession):
                    answers = asyncio.run(ensure_answer_quality_study(configs['v3-answer-quality'], root))
            validation = validate_frozen_opportunity_inputs(targeted)
            self.assertEqual(validation['questions'], 2)
            self.assertEqual(validation['candidate_rows'], 60)
            index_path, = runtime.glob('full-validation/*/index.faiss')
            self.assertEqual(faiss.read_index(str(index_path)).ntotal, 64)
            self.assertTrue(any(url.endswith('/embeddings') for url, _ in transport.calls))
            self.assertEqual(sum(url.endswith('/rerank') for url, _ in transport.calls), 2)
            saved = refresh_opportunity_results(answers)
            self.assertEqual(len(saved['complete_ids']), 2)
            self.assertEqual(len(saved['inputs']), 4)
            self.assertEqual(len(saved['standard']['scores']), 4)
            self.assertTrue((answers['score_directory'] / 'squad_per_answer.csv').is_file())
            before = len(transport.calls)
            with patch.dict(os.environ, {}, clear=True), \
                 patch('requests.Session', side_effect=AssertionError('Replay created a network transport')), \
                 patch('httpx.AsyncClient', side_effect=AssertionError('Replay created a generation transport')), \
                 patch('shared_scoring.fresh_retrieval.load_raw_dataset', side_effect=AssertionError('Replay downloaded data')):
                replay = asyncio.run(ensure_answer_quality_study(configs['v3-answer-quality'], root))
            repeated = refresh_opportunity_results(replay)
            self.assertEqual(len(transport.calls), before)
            self.assertEqual(saved['inputs'], repeated['inputs'])
            pd.testing.assert_frame_equal(saved['standard']['scores'], repeated['standard']['scores'])
