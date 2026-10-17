"""Offline checks for the frozen generation-only extension."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,MagicMock
import pandas as pd
from shared_scoring import generation_extension as extension
from shared_scoring.judge import create_ragas_metrics
from fixture_studies import extension_project

PROJECT=Path(__file__).resolve().parents[1]

class GenerationExtensionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = extension_project(temporary.name)

    def test_frozen_subset_and_only_instruction_changes(self):
        frozen=extension.freeze(self.project)
        _,plan,_,_=extension.location(self.project)
        questions=frozen['contract']['questions']
        self.assertEqual(len(questions),12)
        self.assertEqual(len({r['query_id'] for r in questions}),12)
        self.assertIn('1053665',[q['query_id'] for q in questions])
        self.assertTrue((pd.Series([q['subgroup'] for q in questions]).value_counts()==3).all())
        q=deepcopy(questions[0]);before=deepcopy(q)
        base=extension.payload_for(q,'original',plan,frozen['contract']['generation'])
        for variant in ['evidence_check','concise']:
            actual=extension.payload_for(q,variant,plan,frozen['contract']['generation'])
            self.assertEqual(actual['messages'][1],base['messages'][1])
            self.assertEqual(actual['messages'][0]['content'],base['messages'][0]['content']+'\n\n'+plan['variants'][variant])
            a=deepcopy(actual);b=deepcopy(base);a.pop('messages');b.pop('messages');self.assertEqual(a,b)
            self.assertEqual(actual['max_tokens'],400)
            self.assertEqual(actual['provider']['only'],['nebius/fp8'])
            self.assertFalse(actual['provider']['allow_fallbacks'])
        self.assertEqual(q,before)
        repeated = extension.freeze(self.project)
        self.assertEqual(frozen, repeated)
        _, _, parent, _ = extension.location(self.project)
        changed = pd.read_parquet(parent['questions_file'])
        changed.loc[0, 'question'] = 'Changed after answers were recorded'
        changed.to_parquet(parent['questions_file'])
        with self.assertRaisesRegex(ValueError, 'saved generation inputs/settings differ'):
            extension.freeze(self.project)

    def test_optional_judge_routing_keeps_legacy_behavior(self):
        _,plan,parent,_=extension.location(self.project)
        settings=deepcopy(parent['judge'])
        with patch('ragas.llms.llm_factory') as factory,patch('ragas.metrics.collections.Faithfulness'),patch('ragas.metrics.collections.AnswerRelevancy'):
            create_ragas_metrics(MagicMock(),MagicMock(),settings)
            self.assertNotIn('extra_body',factory.call_args.kwargs)
            settings['judge_provider_routing']=plan['provider_routing']
            create_ragas_metrics(MagicMock(),MagicMock(),settings)
            self.assertEqual(factory.call_args.kwargs['extra_body'],{'provider':plan['provider_routing']})

    def test_metric_means_use_same_matching_questions(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'standard_scoring').mkdir()
            rows=[]
            for q in ['a','b']:
                for variant in ['original','evidence_check','concise']:
                    rows.append(dict(query_id=q,arm=variant,squad_token_f1_pct=50.,faithfulness_0_1=float('nan') if q=='b' and variant=='concise' else 1.,answer_relevancy_0_1=.8,answer_words=20))
            pd.DataFrame(rows).to_csv(folder/'standard_scoring/per_answer_scores.csv',index=False)
            with patch.object(extension,'location',return_value=(PROJECT,{'variants':{'original':'','evidence_check':'','concise':''}},{},folder)):
                table,_=extension.summary(PROJECT)
            self.assertTrue(table['Faithfulness n'].eq(1).all())
            self.assertTrue(table['Token F1 (%) n'].eq(2).all())

    def test_examples_include_f1_regression_when_faithfulness_does_not_drop(self):
        rows=[]
        for q in ['1053665','regression','gain','neutral']:
            for variant in ['original','evidence_check','concise']:
                f1=50.
                if variant=='concise' and q=='regression':f1=20.
                if variant=='concise' and q=='gain':f1=80.
                rows.append(dict(query_id=q,arm=variant,squad_token_f1_pct=f1,faithfulness_0_1=1.))
        with patch.object(extension,'summary',return_value=(pd.DataFrame(),pd.DataFrame(rows))):
            chosen=extension.examples(self.project)
        self.assertEqual(chosen,['1053665','regression','gain'])
