"""Deduplication preserves exact response text, vectors and cache-only replay."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from shared_scoring.io import read_json,write_json,sha256_file
from shared_scoring.matrix_storage import load_matrix,write_reference,npy_sha256
from shared_scoring.fresh_retrieval import RecordedOpenRouterClient,execution_settings
from test_fresh_execution import PROJECT,Session


class DeduplicationTests(unittest.TestCase):
    def test_exact_embedding_response_is_stored_once_and_replayed_without_transport(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);transport=Session()
            client=RecordedOpenRouterClient(execution_settings(PROJECT)['retrieval'],root,transport)
            with patch.dict('os.environ',{'OPENROUTER_API_KEY':'test','ALLOW_MODEL_CALLS':'1','MODEL_BUDGET_PATH':str(root/'budget.sqlite')}):
                vectors,first=client.embed(['question'],'query',4)
            path=next(root.glob('*/attempt-*.json'))
            stored=json.loads(path.read_text())
            self.assertNotIn('raw_response_json',stored)
            self.assertEqual(stored['raw_response_text'],first['raw_response_text'])
            self.assertEqual(read_json(path)['raw_response_json'],first['raw_response_json'])
            with patch.dict('os.environ',{},clear=True):
                again,trace=client.embed(['question'],'query',4)
            np.testing.assert_array_equal(vectors,again)
            self.assertTrue(trace['cache_hit']);self.assertEqual(len(transport.calls),1)
            # Notebook-facing records retain their existing schema.
            write_json(root/'notebook-request.json',first)
            self.assertIn('raw_response_json',json.loads((root/'notebook-request.json').read_text()))

    def test_matrix_reference_has_the_identical_original_file_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source.npy';duplicate=root/'subset.npy'
            values=np.random.default_rng(42).normal(size=(64,4)).astype('float32')
            np.save(source,values);np.save(duplicate,values[:48])
            original_hash=sha256_file(duplicate)
            self.assertEqual(npy_sha256(values[:48]),original_hash)
            self.assertEqual(write_reference(duplicate,source,values[:48]),original_hash)
            duplicate.unlink()
            restored=load_matrix(duplicate)
            np.testing.assert_array_equal(restored,values[:48])
            self.assertEqual(npy_sha256(restored),original_hash)
