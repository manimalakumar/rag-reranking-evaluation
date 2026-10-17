"""Offline IMEI diagnostic checks; never execute notebooks or call models."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from shared_scoring import imei_diagnostic as diagnostic
from shared_scoring.io import read_json,write_json
from fixture_studies import imei_project

PROJECT=Path(__file__).resolve().parents[1]

class ImeiDiagnosticTests(unittest.TestCase):
    def test_synthetic_corpus_support_and_saved_gain(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        project = imei_project(temporary.name)
        _,runtime,baseline,_,_=diagnostic.paths(project)
        q=pd.read_parquet(runtime/diagnostic.TARGETED/'input/questions.parquet',filters=[('query_id','==',diagnostic.QUERY_ID)]).iloc[0]
        gold=set(q.known_supporting_passage_ids)
        docs=read_json(baseline/'input/corpus.json')
        self.assertEqual(len(docs),10210)
        self.assertEqual(len(gold),2)
        self.assertFalse(gold&{d['id'] for d in docs})
        self.assertEqual((q.embedding_recall_at_5,q.reranker_recall_at_5),(.5,1.))
        ranked=pd.read_parquet(runtime/diagnostic.TARGETED/'input/frozen_ranked_top30.parquet',filters=[('query_id','==',diagnostic.QUERY_ID)])
        self.assertEqual(set(ranked.loc[ranked.is_known_support,'passage_id']),gold)
        self.assertEqual(len(ranked[(ranked['rank']==6)&(ranked.rerank_rank==3)&ranked.is_known_support]),1)
        with patch('IPython.display.display') as display, \
             patch.object(diagnostic, 'RecordedOpenRouterClient', side_effect=AssertionError('Saved display made a call')):
            diagnostic.show_large_retrieval(project)
        tables = [args[0] for args, _ in display.call_args_list if isinstance(args[0], pd.DataFrame)]
        self.assertEqual(len(tables), 1)
        self.assertEqual(set(tables[0].passage_id), gold)
        # The real display must reject inconsistent saved recall rather than
        # printing the expected historical narrative over changed evidence.
        questions_path = runtime/diagnostic.TARGETED/'input/questions.parquet'
        changed = pd.read_parquet(questions_path)
        changed['reranker_recall_at_5'] = .5
        changed.to_parquet(questions_path)
        with self.assertRaises(AssertionError), patch('IPython.display.display'):
            diagnostic.show_large_retrieval(project)

    def test_cosine_retrieval_uses_existing_vectors(self):
        docs=[dict(id=str(i),text=str(i),urls=[]) for i in range(3)]
        found=diagnostic.retrieve_small(docs,np.array([[1.,0.],[0.,1.],[-1.,0.]]),[1.,0.],top_k=3)
        self.assertEqual([d['id'] for d in found],['0','1','2'])
        self.assertEqual([d['embedding_rank'] for d in found],[1,2,3])

    def test_completed_small_diagnostic_replays_without_client(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            expected=dict(inputs=[{'arm':'baseline'},{'arm':'reranked'}],excluded_from_baseline_averages=True)
            write_json(folder/'result.json',expected)
            with patch.object(diagnostic,'paths',return_value=(PROJECT,folder,folder,{},folder)),patch.object(diagnostic,'RecordedOpenRouterClient',side_effect=AssertionError('No model client during replay')):
                self.assertEqual(diagnostic.prepare_small(PROJECT,False),expected)
