"""Common diagnostic is cached once and never changes the sampled cohorts."""
import asyncio
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from shared_scoring.fresh_retrieval import RecordedOpenRouterClient, execution_settings
from shared_scoring.opportunity_study import load_opportunity_configuration
from shared_scoring import shared_diagnostic as diagnostic
from test_fresh_execution import PROJECT, Session


class SharedDiagnosticTests(unittest.TestCase):
    def test_first_requests_are_reused_and_diagnostic_stays_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'retrieval';source.mkdir()
            ids=root/'eligible.txt';ids.write_text('1100870\n')
            pd.DataFrame([dict(query_id='1100870',question='effervescent meaning',known_supporting_passage_ids=['p_0'],
                               reference_answers=['bubbling'],embedding_recall_at_5=1.)]).to_parquet(source/'per_query_metrics.parquet')
            shard=source/'retrieval_batches/0';shard.mkdir(parents=True)
            pd.DataFrame([dict(query_id='1100870',passage_id=f'p_{i}',rank=i+1,text=f'evidence {i}') for i in range(30)]).to_parquet(shard/'top30.parquet')
            settings=execution_settings(PROJECT)
            settings['scope']={'eligible_question_ids':ids}
            session=SimpleNamespace(settings=settings,retrieval_directory=lambda:source,enable_calls=lambda:None)
            config=load_opportunity_configuration(PROJECT/'experiments/v3-answer-quality/config.json')
            folder=root/'diagnostic'
            transport=Session()
            def factory(settings,path):return RecordedOpenRouterClient(settings,path,transport)
            metric=SimpleNamespace(compute=lambda **kwargs:dict(exact_match=0.,f1=0.))
            with patch.object(diagnostic,'location',return_value=(session,config,folder)), \
                 patch.object(diagnostic,'RecordedOpenRouterClient',side_effect=factory), \
                 patch('shared_scoring.answers.load_squad_metric',return_value=metric), \
                 patch.dict('os.environ',{'OPENROUTER_API_KEY':'test','ALLOW_MODEL_CALLS':'1','MODEL_BUDGET_PATH':str(root/'budget.sqlite')}):
                result=diagnostic.prepare_retrieval(PROJECT,True)
                inputs=asyncio.run(diagnostic.prepare_answers(PROJECT,True,False))
                self.assertEqual(len(transport.calls),3)
                self.assertEqual({r['query_id'] for r in inputs},{'1100870'})
                self.assertTrue(result['excluded_from_cohort_aggregates'])
                asyncio.run(diagnostic.prepare_answers(PROJECT,False,False))
                self.assertEqual(len(transport.calls),3)
                # Recover an interrupted paired-answer checkpoint from exact cached responses.
                diagnostic.write_json(folder/'answers.json',inputs[:1])
                resumed=asyncio.run(diagnostic.prepare_answers(PROJECT,True,False))
                self.assertEqual({r['arm'] for r in resumed},{'baseline','reranked'})
                self.assertEqual(len(transport.calls),3)
                ids.write_text('other\n')
                with self.assertRaisesRegex(ValueError,'outside'):
                    diagnostic.prepare_retrieval(PROJECT,False)
