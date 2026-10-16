import unittest

import pandas as pd

from shared_scoring.execution import select_cohorts
from shared_scoring.fresh_retrieval import execution_settings
from shared_scoring.paired_generation import select_generation_sample
from test_fresh_execution import PROJECT


class SamplingTests(unittest.TestCase):
    def test_targeted_sample_is_exactly_500_opportunities_and_200_controls(self):
        metrics = pd.DataFrame([{'query_row': i, 'query_id': str(i), 'has_known_support_labels': True,
            'embedding_recall_at_5': 0 if i < 800 else 1, 'embedding_recall_at_30': 1} for i in range(1200)])
        settings = execution_settings(PROJECT)
        selected = select_cohorts(metrics, settings)
        self.assertEqual(selected.cohort.value_counts().to_dict(), {'opportunity': 500, 'control': 200})
        self.assertEqual(set(selected.batch_id), {1})
        self.assertTrue(selected.query_id.is_unique)
        self.assertEqual(len(selected), 700)
        pd.testing.assert_frame_equal(selected, select_cohorts(metrics, settings))
        self.assertLess(len(set(selected.query_id) & set(map(str, range(500)))), 500)

    def test_targeted_sample_reports_insufficient_eligible_population(self):
        metrics = pd.DataFrame([{'query_row': i, 'query_id': str(i), 'has_known_support_labels': True,
            'embedding_recall_at_5': 0 if i < 499 else 1, 'embedding_recall_at_30': 1} for i in range(699)])
        with self.assertRaisesRegex(ValueError, '499 eligible opportunities'):
            select_cohorts(metrics, execution_settings(PROJECT))

    def test_answer_quality_caps_each_subgroup_and_retains_smaller_groups(self):
        strata = ['opportunity_gain', 'opportunity_unchanged', 'opportunity_loss',
                  'control_all_support_already_present', 'control_some_support_absent_top30']
        rows = []
        for index, count in enumerate([40, 10, 0, 30, 5]):
            for position in range(count):
                rows.append({'query_id': f'{index}-{position:03}',
                    'cohort': 'opportunity' if index < 3 else 'control',
                    'outcome': ['gain', 'unchanged', 'loss', 'unchanged', 'unchanged'][index],
                    'embedding_recall_at_5': 1 if index == 3 else .5,
                    'known_supports_absent_frozen_top30': 1 if index == 4 else 0})
        population = pd.DataFrame(rows)
        selected = select_generation_sample(population, strata, 25, 42, 1e-12)
        self.assertEqual(selected.generation_stratum.value_counts().to_dict(),
            {strata[0]: 25, strata[1]: 10, strata[3]: 25, strata[4]: 5})
        self.assertTrue(selected.query_id.is_unique)
        self.assertEqual(set(selected.loc[selected.generation_stratum.eq(strata[1]), 'query_id']),
                         set(population.loc[population.cohort.eq('opportunity') & population.outcome.eq('unchanged'), 'query_id']))
        pd.testing.assert_frame_equal(selected, select_generation_sample(population, strata, 25, 42, 1e-12))
