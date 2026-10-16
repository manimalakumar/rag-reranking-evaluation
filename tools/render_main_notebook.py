"""Verify main-notebook replay offline and render its HTML companion."""
import json
import os
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

project = Path(__file__).resolve().parents[1]
cache = project / '.cache/notebook'
kernel = cache / 'kernels/offline-replay'
kernel.mkdir(parents=True, exist_ok=True)
(kernel / 'kernel.json').write_text(json.dumps({'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'], 'display_name': 'Offline verification', 'language': 'python'}))
os.environ['JUPYTER_PATH'] = str(cache)
os.environ['JUPYTER_RUNTIME_DIR'] = str(cache / 'runtime')
os.environ['IPYTHONDIR'] = str(cache / 'ipython')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
path = project / 'experiments/v1-baseline/Experiment-1-Baseline.ipynb'
notebook = nbformat.read(path, as_version=4)
# Verification-only guard: executes after the kernel starts, before notebook replay.
guard = nbformat.v4.new_code_cell("import socket\nfrom unittest.mock import patch\n_network_guard = patch.object(socket.socket, 'connect', side_effect=AssertionError('No network calls allowed during saved notebook replay'))\n_network_guard.start()")
notebook.cells.insert(0, guard)
NotebookClient(notebook, timeout=120, kernel_name='offline-replay', resources={'metadata': {'path': str(path.parent)}}).execute()
notebook.cells.pop(0)
# Remove workstation-specific execution timestamps and the verifier-only cell.
for cell in notebook.cells:
    cell.metadata.pop('execution', None)
    if cell.cell_type == 'code':
        cell.execution_count -= 1
        if any(output.output_type == 'error' for output in cell.outputs):
            raise AssertionError('Notebook replay contains a failed cell.')
# Keep rendered displays portable while preserving original trace files.
for cell in notebook.cells:
    for output in cell.get('outputs', []):
        if isinstance(output.get('text'), str):
            output['text'] = output['text'].replace(str(project), '.')
        for mime, content in output.get('data', {}).items():
            if isinstance(content, str):
                output['data'][mime] = content.replace(str(project), '.')
nbformat.validate(notebook)
nbformat.write(notebook, path)
html, _ = HTMLExporter(template_name='lab').from_notebook_node(notebook)
path.with_suffix('.html').write_text(html)
print(f'Executed {sum(c.cell_type == "code" for c in notebook.cells)} code cells with network blocked; zero notebook errors.')
