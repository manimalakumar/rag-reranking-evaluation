"""Current-run report checks with synthetic candidates and failed judge outcomes."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from shared_scoring.io import write_json
from shared_scoring.tutorial_views import TutorialViews


class TutorialViewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.questions = [dict(query_id='q', question='define q', gold_ids=['p4', 'p5'],
                               reference_answers=['q'], subgroup='gain', diagnostic=False)]
        self.run = SimpleNamespace(mode='replay', study='targeted', folder=self.root, project=self.root,
                                   run_id='new', plan={'questions': 1})
        write_json(self.root/'questions.json',self.questions)
        write_json(self.root/'manifest.json', {'contract': {'upstream': {}}})
        rows = [dict(query_id='q', passage_id=f'p{i}', text=f'evidence {i}', rank=i+1,
                     rerank_rank=3 if i==5 else 6 if i==2 else i+1, is_known_support=i in [4,5]) for i in range(30)]
        pd.DataFrame(rows).to_parquet(self.root/'reranked_top30.parquet',index=False)
        self.view=TutorialViews(self.run)
    def tearDown(self): self.temp.cleanup()

    def test_coverage_and_haystack_crosscheck_count_all_supports(self):
        row=self.view.retrieval().iloc[0]
        self.assertEqual(row.embedding_recall_at_5,0.5)
        self.assertEqual(row.reranker_recall_at_5,1.)
        self.assertEqual(row.outcome,'gain')
        with patch('IPython.display.display'):
            self.view.show('haystack_crosscheck')

    def test_paired_summary_never_replaces_missing_judge_outcome_with_zero(self):
        self.run.study='answers'
        (self.root/'scores').mkdir()
        pd.DataFrame([dict(query_id=q, subgroup='gain', arm=arm, faithfulness=value)
                      for q,arm,value in [('a','baseline',0.5),('a','reranked',1.),
                                         ('b','baseline',0.8),('b','reranked',float('nan'))]]).to_csv(self.root/'scores/per_answer_scores.csv',index=False)
        row=self.view.paired().iloc[0]
        self.assertEqual(row.matched_pairs,1)
        self.assertEqual(row.mean_delta,0.5)

    def test_baseline_diagnostic_is_excluded_from_primary_score_means(self):
        self.run.study='baseline';(self.root/'scores').mkdir()
        pd.DataFrame([dict(query_id=q,subgroup='baseline',diagnostic=diagnostic,arm=arm,squad_token_f1_pct=value)
                      for q,diagnostic,arm,value in [('a',False,'baseline',10.),('a',False,'reranked',20.),
                                                    ('q',True,'baseline',0.),('q',True,'reranked',100.)]]).to_csv(self.root/'scores/per_answer_scores.csv',index=False)
        row=self.view.paired().iloc[0]
        self.assertEqual(row.matched_pairs,1)
        self.assertEqual(row.mean_delta,10.)

    def test_audit_rejects_answer_context_from_different_rank_order(self):
        self.run.study='answers'
        records=[dict(query_id='q',arm='reranked',run_id='new',context_passage_ids=[f'p{i}' for i in range(5)],
                      retrieved_contexts=[f'[p{i}]\nevidence {i}' for i in range(5)])]
        records[0]['context']='\n\n'.join(records[0]['retrieved_contexts'])
        write_json(self.root/'generate_inputs.json',records)
        with self.assertRaises(AssertionError):self.view.audit()
