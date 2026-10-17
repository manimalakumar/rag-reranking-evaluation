"""Execute the baseline walkthrough against newly generated mocked Qwen evidence."""
import ast
import asyncio
import contextlib
import inspect
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from shared_scoring import load_configuration
from shared_scoring.fresh_baseline import build_baseline_run, prepare_baseline_scores
from shared_scoring.fresh_retrieval import RecordedOpenRouterClient, execution_settings
from shared_scoring.io import write_json
from test_fresh_execution import Session
from fixture_studies import imei_project
from shared_scoring import imei_diagnostic

PROJECT = Path(__file__).resolve().parents[1]


class NotebookTests(unittest.TestCase):
    def test_demo_runs_both_haystack_paths_with_only_four_mocked_requests(self):
        import faiss
        import numpy as np
        import pandas as pd
        from shared_scoring.notebook_runtime import NotebookSession
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            corpus = pd.DataFrame([dict(passage_id=f'p_{i:020x}',text=f'passage {i}',source_url='https://example.test',truncated=False) for i in range(64)])
            corpus.to_parquet(folder/'corpus.parquet')
            write_json(folder/'dataset_manifest.json',dict(validation_questions=8,unique_passages=64))
            write_json(folder/'assets_manifest.json',dict(nprobe=2))
            vectors=np.random.default_rng(42).normal(size=(64,4)).astype('float32')
            vectors/=np.linalg.norm(vectors,axis=1,keepdims=True)
            faiss.omp_set_num_threads(1)
            index=faiss.IndexIVFFlat(faiss.IndexFlatIP(4),4,2,faiss.METRIC_INNER_PRODUCT)
            index.train(vectors);index.add(vectors)
            faiss.write_index(index,str(folder/'index.faiss'))
            transport=Session()
            namespace={'__name__':'__main__'}
            path=PROJECT/'demos/00-rag-with-and-without-reranking/RAG-Retriever-Embedding-Reranker.ipynb'
            with patch.dict(os.environ,dict(OPENROUTER_API_KEY='test',ALLOW_MODEL_CALLS='1',NOTEBOOK_MODE='execute',MODEL_BUDGET_PATH=str(folder/'budget.sqlite'))), \
                 patch.object(NotebookSession,'require_complete_index',return_value=folder), \
                 patch.object(NotebookSession,'enable_calls',return_value=None), \
                 patch('IPython.display.display',lambda *args,**kwargs:None), contextlib.redirect_stdout(io.StringIO()):
                for position,cell in enumerate(json.loads(path.read_text())['cells']):
                    if cell['cell_type']!='code':continue
                    exec(compile(''.join(cell['source']),f'demo:{position}','exec'),namespace)
                    if position==3:
                        namespace['ARTIFACTS']=folder
                        namespace['DEMO_CACHE']=folder/'demo'
                        namespace['api'].directory=folder/'demo/requests'
                        namespace['api'].session=transport
                        namespace['settings']['embedding_dimensions']=4
            self.assertEqual(len(transport.calls),4)
            self.assertEqual(sum(url.endswith('/chat/completions') for url,_ in transport.calls),2)
            self.assertEqual(len(namespace['baseline_top_five']),5)
            self.assertEqual(len(namespace['reranked_top_five']),5)

    def test_all_notebook_cells_compile_with_current_python(self):
        paths = [*PROJECT.glob('experiments/*/*.ipynb'),
                 *PROJECT.glob('diagnosis-retrieval-opportunity/*.ipynb'), *PROJECT.glob('demos/*/*.ipynb')]
        for path in paths:
            for index, cell in enumerate(json.loads(path.read_text())['cells']):
                if cell['cell_type'] == 'code':
                    compile(''.join(cell['source']), f'{path.name}:{index}', 'exec', flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)

    def test_baseline_notebook_replays_new_mocked_run_without_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp).resolve()
            source = folder / 'source'
            source.mkdir()
            original = PROJECT / 'experiments/v1-baseline'
            shutil.copytree(original / 'input', source / 'input')
            for name in ['experiment_helpers.py', 'scoring.py']:
                shutil.copy2(original / name, source / name)
            settings = json.loads((original / 'config.json').read_text())
            settings['models_and_dataset'].update(embedding_dimensions=4, embedding_batch_size=512)
            write_json(source / 'config.json', settings)
            config = load_configuration(PROJECT / 'configs/experiment1.json')
            config.update(source_directory=source, score_directory=folder / 'scores')
            transport = Session()
            client = RecordedOpenRouterClient(execution_settings(PROJECT)['retrieval'], folder / 'requests', transport)
            with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test', 'ALLOW_MODEL_CALLS': '1'}):
                fresh = build_baseline_run(config, PROJECT, folder / 'run', client)
            prepare_baseline_scores(fresh)
            diagnostic_project = imei_project(folder / "diagnostic-fixture")
            diagnostic_paths = imei_diagnostic.paths(diagnostic_project)
            before = len(transport.calls)
            namespace = {'__name__': '__main__'}
            path = original / 'Experiment-1-Baseline.ipynb'
            previous = Path.cwd()
            try:
                os.chdir(PROJECT)
                with patch('shared_scoring.imei_diagnostic.paths', return_value=diagnostic_paths), \
                     patch('shared_scoring.execution.ensure_baseline_study', return_value=fresh), \
                     patch('shared_scoring.notebook_runtime.NotebookSession.saved_configuration', return_value=fresh), \
                     patch('IPython.display.display', lambda *args, **kwargs: None), \
                     patch('requests.Session.post', side_effect=AssertionError('Replay made a hosted call')), \
                     contextlib.redirect_stdout(io.StringIO()):
                    for index, cell in enumerate(json.loads(path.read_text())['cells']):
                        if cell['cell_type'] != 'code': continue
                        code = compile(''.join(cell['source']), f'{path.name}:{index}', 'exec', flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
                        result = eval(code, namespace)
                        if inspect.isawaitable(result): asyncio.run(result)
            finally:
                os.chdir(previous)
            self.assertEqual(len(transport.calls), before)
            self.assertEqual(len(namespace['results']), 32)
