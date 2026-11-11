"""Compile and Plan the tutorials; preserve recorded outputs without model calls."""
import ast,asyncio,hashlib,inspect,json,os
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
import nbformat

from shared_scoring.live_tutorial import NOTEBOOKS
from shared_scoring.io import read_json,write_json

PROJECT=Path(__file__).resolve().parents[1]


class NotebookTests(unittest.TestCase):
    def test_all_notebook_cells_compile_with_current_python(self):
        for relative in NOTEBOOKS.values():
            for i,c in enumerate(read_json(PROJECT/relative)['cells']):
                if c['cell_type']=='code':compile(''.join(c['source']),f'{relative}:{i}','exec',flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)

    def test_all_active_tutorial_cells_plan_without_cache_or_credentials(self):
        previous=Path.cwd()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name in ['execution.json','tutorial_inputs.json','model_limits.json']:
                write_json(root/'configs'/name,read_json(PROJECT/'configs'/name))
            write_json(root/'experiments/v3-answer-quality/config.json',read_json(PROJECT/'experiments/v3-answer-quality/config.json'))
            try:
                os.chdir(root)
                with patch.dict(os.environ,{'NOTEBOOK_MODE':'plan','OPENROUTER_API_KEY':''}),patch('IPython.display.display',lambda *args,**kwargs:None):
                    for relative in NOTEBOOKS.values():
                        namespace={'__name__':'__main__'}
                        for c in read_json(PROJECT/relative)['cells']:
                            if c['cell_type']!='code' or not c['metadata'].get('live_tutorial'):continue
                            result=eval(compile(''.join(c['source']),relative,'exec',flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT),namespace)
                            if inspect.isawaitable(result):asyncio.run(result)
                        self.assertEqual(namespace['MODE'],'plan')
                        self.assertFalse(namespace['tutorial'].folder.exists())
            finally:os.chdir(previous)

    def test_original_recorded_outputs_match_the_preservation_audit(self):
        audit=read_json(PROJECT/'reports/tutorial-reference-audit.json')
        for relative,entry in audit['notebooks'].items():
            nb=nbformat.read(PROJECT/relative,as_version=4)
            if nb.metadata['live_tutorial']['hosted_execution_state'] != 'pending':
                continue  # A deliberately executed revision has its own real outputs.
            cells={c['id']:c for c in nb['cells']}
            for item in entry['code_cells']:
                if item['id'] in nb.metadata['live_tutorial'].get('user_removed_original_cells',[]):
                    self.assertNotIn(item['id'],cells)
                    continue
                current=cells[item['id']]
                digest=hashlib.sha256(json.dumps(current['outputs'],sort_keys=True).encode()).hexdigest()
                presentation=current.metadata.get('display_refresh',{})
                expected=presentation.get('outputs_sha256',item['outputs_sha256'])
                if presentation:self.assertEqual(presentation['model_calls'],0)
                self.assertEqual(digest,expected,relative)
                self.assertEqual(current['execution_count'],presentation.get('execution_count',item['execution_count']))

    def test_sections_are_active_and_markdown_is_preserved(self):
        for relative in NOTEBOOKS.values():
            nb=nbformat.read(PROJECT/relative,as_version=4)
            markdown=[(c.id,c.source) for c in nb.cells if c.cell_type=='markdown']
            digest=hashlib.sha256(json.dumps(markdown).encode()).hexdigest()
            self.assertEqual(digest,nb.metadata['live_tutorial']['markdown_sha256'])
            for cell in nb.cells:
                if cell.cell_type!='code':continue
                self.assertTrue(cell.metadata.get('live_tutorial'))
                self.assertNotIn('skip-execution',cell.metadata.get('tags',[]))
                tree=ast.parse(cell.source)
                self.assertFalse(any(isinstance(n,ast.If) and isinstance(n.test,ast.Constant) and n.test.value is False for n in ast.walk(tree)))
