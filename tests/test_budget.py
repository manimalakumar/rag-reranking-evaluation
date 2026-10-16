import json
import os
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from shared_scoring.budget import ModelBudgetExceeded, reserve_request
from shared_scoring.fresh_retrieval import RecordedOpenRouterClient, execution_settings
from test_fresh_execution import PROJECT, Session


class BudgetTests(unittest.TestCase):
    def test_manual_restart_cannot_reset_logical_request_cap(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'MODEL_BUDGET_PATH': folder + '/ledger.sqlite'}):
            reserve_request('url', {'input': 'text'}, 'answer-1', 2)
            reserve_request('url', {'input': 'text'}, 'answer-1', 2)
            with self.assertRaisesRegex(ModelBudgetExceeded, 'lifetime cap of 2'):
                reserve_request('url', {'input': 'text'}, 'answer-1', 2)
            self.assertEqual(reserve_request('url', {'input': 'text'}, 'answer-2', 2), 3)

    def test_global_cap_counts_failed_attempts_and_is_atomic(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'MODEL_BUDGET_PATH': folder + '/ledger.sqlite'}):
            limits = json.dumps({'max_total_requests': 5, 'ledger_path': 'unused'})
            actual_read = Path.read_text
            def read(path, *args, **kwargs):
                return limits if path.name == 'model_limits.json' else actual_read(path, *args, **kwargs)
            def reserve(number):
                try:
                    reserve_request('url', {}, str(number), 3)
                    return True
                except ModelBudgetExceeded:
                    return False
            with patch.object(Path, 'read_text', read), ThreadPoolExecutor(max_workers=8) as pool:
                self.assertEqual(sum(pool.map(reserve, range(20))), 5)

    def test_budget_stops_before_transport_and_cache_hits_are_free(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {
            'MODEL_BUDGET_PATH': folder + '/ledger.sqlite', 'ALLOW_MODEL_CALLS': '1', 'OPENROUTER_API_KEY': 'test'}):
            settings = execution_settings(PROJECT)['retrieval']
            settings.update(max_attempts=1)
            session = Session()
            client = RecordedOpenRouterClient(settings, folder + '/cache', session)
            client.embed(['test'], 'query', 4)
            client.embed(['test'], 'query', 4)
            self.assertEqual(len(session.calls), 1)
            other = RecordedOpenRouterClient(settings, folder + '/different-cache', session)
            with self.assertRaises(ModelBudgetExceeded):
                other.embed(['test'], 'query', 4)
            self.assertEqual(len(session.calls), 1)
