"""Expandable tables preserve every value without truncation or network access."""
from html.parser import HTMLParser
import unittest
import pandas as pd
from shared_scoring.notebook_display import table_html


class NotebookDisplayTests(unittest.TestCase):
    def test_long_table_keeps_first_row_visible_and_all_others_once(self):
        frame=pd.DataFrame({'answer':[f'answer-marker-{i}' for i in range(12)]})
        html=table_html(frame)
        self.assertIn('Show remaining 11 rows (12 total)',html)
        self.assertLess(html.index('answer-marker-0'),html.index('<details>'))
        for text in frame.answer:self.assertEqual(html.count('>'+text+'<'),1)

    def test_answer_pair_can_expand_second_row_with_full_safe_text(self):
        text='<script>alert(1)</script> '+'evidence '*100
        html=table_html(pd.DataFrame({'answer':[text,'second answer']}))
        self.assertIn('Show remaining 1 rows',html)
        self.assertIn('&lt;script&gt;',html)
        self.assertNotIn('<script>',html)
        self.assertIn(('evidence '*100).rstrip(),html)
        self.assertNotIn('...',html)

    def test_small_summary_table_stays_open(self):
        html=table_html(pd.DataFrame({'metric':['F1','faithfulness'],'value':[10.,1.]}))
        self.assertNotIn('<details>',html)

    def test_integrity_digests_are_not_displayed_or_removed_from_source(self):
        from shared_scoring.notebook_display import details_html
        value={'model':'qwen','contract_sha256':'hidden-digest','nested':{'checksum':'hidden-checksum','questions':31},'shadow_mode':True}
        html=details_html('Configuration',value)
        self.assertNotIn('hidden-digest',html)
        self.assertNotIn('hidden-checksum',html)
        self.assertIn('qwen',html)
        self.assertIn('shadow_mode',html)
        self.assertEqual(value['contract_sha256'],'hidden-digest')

    def test_extension_preview_leaves_one_row_outside_collapsed_results(self):
        from shared_scoring.notebook_display import preview_panel_html
        body=pd.DataFrame({'version':['original','evidence_check','concise'],'score':[1.,2.,3.]}).to_html(index=False)+'<p>Evidence and qualitative review</p>'
        html=preview_panel_html(body,'More results',1)
        before,after=html.split('<details>',1)
        self.assertIn('original',before)
        self.assertNotIn('evidence_check',before)
        self.assertIn('evidence_check',after)
        self.assertIn('concise',after)
        self.assertIn('Evidence and qualitative review',after)
        for version in ['original','evidence_check','concise']:self.assertEqual(html.count('>'+version+'<'),1)

    def test_extension_output_without_table_is_still_collapsible(self):
        from shared_scoring.notebook_display import preview_panel_html
        self.assertEqual(preview_panel_html('No scores yet','Details'),'<details><summary>Details</summary>No scores yet</details>')

    def test_scoring_log_preview_collapses_later_logs_and_coverage_table(self):
        from shared_scoring.notebook_display import preview_panel_html
        body='<pre>first entry\nsecond entry\n</pre>'+pd.DataFrame({'outcomes':[32]}).to_html(index=False)
        html=preview_panel_html(body,'More',1,prefer_logs=True)
        before,after=html.split('<details>',1)
        self.assertIn('first entry',before)
        self.assertNotIn('second entry',before)
        self.assertNotIn('<table',before)
        self.assertIn('second entry',after)
        self.assertIn('<table',after)
