"""Verify frozen question eligibility and zero passage calls on the reduced corpus."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from shared_scoring.cached_scope import assert_eligible
from shared_scoring.fresh_retrieval import execution_settings, prepare_dataset, ensure_retrieval_assets, retrieve_full_validation, passage_id
from shared_scoring.io import write_json, sha256_file
from test_fresh_execution import PROJECT, Session, raw_rows


class QueryOnlyClient:
    def __init__(self):
        self.modes = []
    def embed(self, texts, mode, dimensions):
        if mode != 'query':
            raise AssertionError('Passage embedding must never run for the cached scope.')
        self.modes.append(mode)
        return np.ones((len(texts), dimensions), dtype='float32'), {}


class CachedScopeTests(unittest.TestCase):
    def test_saved_study_resolves_eligible_ids_and_rejects_outside_questions(self):
        from shared_scoring.opportunity_study import load_opportunity_configuration, load_opportunity_population
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            config=json.loads((PROJECT/'experiments/v2-targeted-reranking/config.json').read_text())
            config.update(questions_file='questions.parquet',eligible_question_ids_file='eligible.txt',
                          expected_cohort_counts={'opportunity':1},diagnostic_ids=[])
            write_json(root/'config.json',config)
            (root/'eligible.txt').write_text('allowed\n')
            pd.DataFrame([dict(query_id='outside',cohort='opportunity',embedding_recall_at_5=0.,embedding_recall_at_30=1.)]).to_parquet(root/'questions.parquet')
            loaded=load_opportunity_configuration(root/'config.json')
            self.assertEqual(loaded['eligible_question_ids_file'],(root/'eligible.txt').resolve())
            with self.assertRaisesRegex(ValueError,'outside'):
                load_opportunity_population(loaded)

    def setup_scope(self, root):
        settings = execution_settings(PROJECT)
        settings['retrieval'].update(dimensions=4,nlist=2,index_add_batch_size=16,query_batch_size=2,training_sample_size=48)
        source = root/'source'
        prepare_dataset(settings, source, raw_rows())
        matrix = np.random.default_rng(42).normal(size=(64,4)).astype('float32')
        matrix /= np.linalg.norm(matrix,axis=1,keepdims=True)
        matrix[48:] = np.nan  # Unfilled rows must NEVER enter the reduced index.
        np.save(source/'embeddings.npy', matrix)
        for start in range(0,48,16):
            write_json(source/'embedding_batches'/f'{start}.json',dict(start=start,stop=start+16,vector_sha256='source-checkpoint'))
        write_json(source/'assets_manifest.json',dict(contract={'retrieval':settings['retrieval']},status='building'))
        ids = root/'eligible.txt'; ids.write_text('0\n1\n2\n3\n')
        coverage = root/'coverage.jsonl'
        with coverage.open('w') as f:
            for row in raw_rows()[:4]:
                f.write(json.dumps(dict(query_id=str(row['query_id']),all_associated_passages_embedded=True,
                                       not_yet_embedded_passage_ids=[],embedded_passage_ids=[passage_id(t) for t in row['passages']['passage_text']]))+'\n')
        settings['scope'] = dict(mode='checkpointed_passages',source_directory=source,eligible_question_ids=ids,
                                coverage_file=coverage,eligible_ids_sha256=sha256_file(ids),coverage_sha256=sha256_file(coverage),
                                eligible_question_count=4,passage_count=48)
        return settings

    def test_only_checkpointed_passages_and_eligible_questions_are_used(self):
        import faiss
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = self.setup_scope(root)
            client = QueryOnlyClient()
            metrics,_,assets = retrieve_full_validation(settings, root/'subset', client)
            self.assertEqual(metrics.query_id.tolist(), ['0','1','2','3'])
            self.assertEqual(client.modes, ['query','query'])
            self.assertEqual(faiss.read_index(str(assets['index_file'])).ntotal, 48)
            from shared_scoring.matrix_storage import load_matrix
            self.assertFalse((root/'subset/embeddings.npy').exists())
            self.assertEqual(load_matrix(root/'subset/embeddings.npy').shape, (48,4))
            before = len(client.modes)
            retrieve_full_validation(settings, root/'subset', client)
            self.assertEqual(len(client.modes), before)

    def test_changed_eligibility_missing_checkpoints_and_outside_ids_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); settings=self.setup_scope(root)
            with self.assertRaisesRegex(ValueError,'outside'):
                assert_eligible(pd.DataFrame({'query_id':['7']}),settings)
            (root/'source/embedding_batches/16.json').unlink()
            with self.assertRaisesRegex(ValueError,'checkpointed'):
                ensure_retrieval_assets(settings,root/'subset')
            (root/'eligible.txt').write_text('7\n')
            with self.assertRaisesRegex(ValueError,'list changed'):
                assert_eligible(pd.DataFrame({'query_id':['7']}),settings)
