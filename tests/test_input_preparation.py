"""Public preparation is portable, bounded, and never needs private runtime artifacts."""
from copy import deepcopy
import hashlib,json,os
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from shared_scoring.input_preparation import collect_public_inputs,corpus_digest,prepare_inputs
from shared_scoring.io import read_json,write_json
from shared_scoring.live_tutorial import TutorialRun,NOTEBOOKS

PROJECT=Path(__file__).resolve().parents[1]

class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        for name in ['execution.json','tutorial_inputs.json','model_limits.json']:
            write_json(self.root/'configs'/name,read_json(PROJECT/'configs'/name))
        cfg=read_json(self.root/'configs/execution.json');cfg['retrieval']['dimensions']=4
        write_json(self.root/'configs/execution.json',cfg)
        for name in ['v1-baseline','v3-answer-quality']:
            write_json(self.root/f'experiments/{name}/config.json',read_json(PROJECT/f'experiments/{name}/config.json'))
        self.rows=[dict(query_id=q,query='question '+q,answers=['answer'],wellFormedAnswers=[],passages={
            'passage_text':[f'Passage {q} {i}' for i in range(16)],'url':['https://example.test']*16,
            'is_selected':[1]+[0]*15}) for q in ['1053665','1031584','qbaseline','qextra']]
        self.calls=[]
        owner=self
        class Client:
            def embed(self,texts,input_type,dimensions):
                owner.calls.append((list(texts),input_type,dimensions))
                return np.arange(1,len(texts)*dimensions+1,dtype='float32').reshape(len(texts),dimensions),{}
        self.client=Client()
    def tearDown(self):self.temp.cleanup()
    def recipe(self):
        corpus,queries=collect_public_inputs(self.rows,64,1800)
        ids_file=self.root/'eligible.txt';ids_file.write_text('\n'.join(queries.query_id)+'\n')
        q=[dict(query_id=r.query_id,question=r.question,reference_answers=list(r.reference_answers),
            gold_ids=list(r.known_supporting_passage_ids),subgroup='opportunity' if i==0 else 'control',diagnostic=i==0)
           for i,r in queries.iterrows()]
        return dict(passage_count=64,corpus_id_text_sha256=corpus_digest(corpus),eligible_ids_file='eligible.txt',
            eligible_ids_sha256=hashlib.sha256(ids_file.read_bytes()).hexdigest(),baseline_passage_ids=corpus.passage_id.tolist(),
            baseline_questions=[q[2]],targeted_questions=q[:2],answer_questions=q[:2],latency_questions=q[:1],
            targeted_candidates={r['query_id']:dict(passage_ids=corpus.passage_id.head(30).tolist(),rerank_order=list(range(1,31))) for r in q[:2]},
            latency_assignments=[dict(query_id=q[0]['query_id'],latency_sample_order=0,arm_order=['baseline','reranked'])])
    def test_small_profile_is_compatible_and_reuses_verified_assets(self):
        before=(self.root/'configs/tutorial_inputs.json').read_bytes()
        path=prepare_inputs(self.root,'small',64,rows=self.rows,client=self.client)
        declarations=read_json(path);self.assertEqual(declarations['allowed_studies'],['demo'])
        self.assertEqual(declarations['demo_question']['query_id'],'1053665')
        self.assertTrue(all(t=='passage' for _,t,_ in self.calls))
        with patch.dict(os.environ,{'TUTORIAL_INPUTS':str(path)}):
            run=TutorialRun(self.root,'demo',mode='plan');self.assertFalse(run.plan['missing_inputs'])
            with self.assertRaisesRegex(ValueError,'only demo'):TutorialRun(self.root,'baseline',mode='plan')
        count=len(self.calls)
        self.assertEqual(prepare_inputs(self.root,'small',64,rows=iter(()),client=self.client),path)
        self.assertEqual(len(self.calls),count)
        self.assertEqual((self.root/'configs/tutorial_inputs.json').read_bytes(),before)
        asset=path.parent/'retrieval/embeddings.npy';matrix=np.load(asset);matrix[0]=0;np.save(asset,matrix)
        with self.assertRaisesRegex(ValueError,'Prepared input changed'):prepare_inputs(self.root,'small',64,client=self.client)
    def test_reference_profile_supplies_all_six_notebook_inputs(self):
        recipe=self.recipe();path=prepare_inputs(self.root,'reference',rows=self.rows,recipe=recipe,client=self.client)
        with patch.dict(os.environ,{'TUTORIAL_INPUTS':str(path)}):
            for study in NOTEBOOKS:
                run=TutorialRun(self.root,study,mode='plan');self.assertFalse(run.plan['missing_inputs'],study)
        base=np.load(path.parent/'baseline/document_embeddings.npy');full=np.load(path.parent/'retrieval/embeddings.npy')
        np.testing.assert_array_equal(base,full)
        frozen=pd.read_parquet(path.parent/'targeted/input/frozen_ranked_top30.parquet')
        self.assertEqual(len(frozen),60)
        self.assertEqual(frozen[frozen.query_id.eq('1053665')].passage_id.tolist(),recipe['targeted_candidates']['1053665']['passage_ids'])
        self.assertFalse(list(path.parent.rglob('generate_inputs.json')))
    def test_wrong_source_fails_before_embedding_and_outputs_stay_ignored(self):
        recipe=self.recipe();recipe['corpus_id_text_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'no model call'):prepare_inputs(self.root,'reference',rows=self.rows,recipe=recipe,client=self.client)
        self.assertFalse(self.calls)
        invalid=self.recipe();invalid['targeted_candidates']['1053665']['rerank_order']=[1]*30
        with self.assertRaisesRegex(ValueError,'no model call'):prepare_inputs(self.root,'reference',rows=self.rows,recipe=invalid,client=self.client)
        unlabeled=deepcopy(self.rows)
        for row in unlabeled:row['passages']['is_selected']=[0]*16
        with self.assertRaisesRegex(ValueError,'No labeled'):prepare_inputs(self.root,'small',64,rows=unlabeled,client=self.client)
        self.assertFalse(self.calls)
        with self.assertRaisesRegex(ValueError,'ignored'):prepare_inputs(self.root,'small',64,output='tracked-output',rows=self.rows,client=self.client)
        with self.assertRaisesRegex(ValueError,'At least 30'):prepare_inputs(self.root,'small',2,rows=self.rows,client=self.client)
