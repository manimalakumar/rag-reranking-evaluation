"""Execute notebooks and preserve outputs, retaining the first completed notebook outputs.

Default execution blocks outbound network calls. --allow-model-calls runs offline
tests first. Neither mode resumes incomplete full-corpus embedding preparation.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebooks', nargs='*', type=Path)
    parser.add_argument('--mode', choices=['execute', 'replay'], default='execute')
    parser.add_argument('--allow-model-calls', action='store_true')
    parser.add_argument('--judge', action='store_true', help='Run the explicitly enabled RAGAS cells on first execution.')
    args = parser.parse_args()
    if args.allow_model_calls and args.mode == 'replay':
        parser.error('Replay cannot enable model calls.')
    if args.allow_model_calls:
        import subprocess
        subprocess.run([sys.executable, str(PROJECT/'tools/verify_offline.py')], cwd=PROJECT, check=True)
        os.environ['NOTEBOOK_TESTS_PASSED'] = '1'
    cache = PROJECT / '.cache/notebook'
    kernel = cache / 'kernels/qwen-notebooks'
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / 'kernel.json').write_text(json.dumps({'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'], 'display_name': 'Python 3.14 Qwen study', 'language': 'python'}))
    os.environ.update(JUPYTER_PATH=str(cache), JUPYTER_RUNTIME_DIR=str(cache/'runtime'), IPYTHONDIR=str(cache/'ipython'),
                      HAYSTACK_TELEMETRY_ENABLED='False', RAGAS_DO_NOT_TRACK='true', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    paths = args.notebooks or [PROJECT/'demos/00-rag-with-and-without-reranking/RAG-Retriever-Embedding-Reranker.ipynb',
                               PROJECT/'experiments/v1-baseline/Experiment-1-Baseline.ipynb',
                               PROJECT/'diagnosis-retrieval-opportunity/Diagnosis-Retrieval-Opportunity.ipynb',
                               PROJECT/'experiments/v2-targeted-reranking/Experiment-2-Targeted-Reranking.ipynb',
                               PROJECT/'experiments/v3-answer-quality/Experiment-3-Answer-Quality.ipynb',
                               PROJECT/'experiments/v4-latency-tradeoff/Experiment-4-Latency-Tradeoff.ipynb']
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    outcomes = []
    for path in paths:
        path = path.resolve()
        relative = path.relative_to(PROJECT)
        (PROJECT/'.cache/live-rebuild-status.json').write_text(json.dumps(dict(
            state='running_notebooks',pid=os.getpid(),notebook=str(relative),
            eligible_questions=15585,cached_passages=152576,new_passage_embedding_calls=0,
            mode=args.mode),indent=2)+'\n')
        notebook = nbformat.read(path, as_version=4)
        previous = notebook.metadata.get('qwen_execution', {})
        if previous.get('state') == 'complete':
            print(json.dumps(dict(notebook=str(relative), state='kept_first_execution')), flush=True)
            outcomes.append(dict(notebook=str(relative), state='kept_first_execution'))
            continue
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                cell.outputs = []
                cell.execution_count = None
        setup = "import os\nos.environ['NOTEBOOK_MODE'] = " + repr(args.mode) + "\nos.environ['NOTEBOOK_ALLOW_MODEL_CALLS'] = " + repr('1' if args.allow_model_calls else '0') + "\nos.environ.pop('ALLOW_MODEL_CALLS', None)\n"
        setup += "os.environ['NOTEBOOK_RUN_JUDGE'] = " + repr('1' if args.judge and args.allow_model_calls else '0') + "\n"
        if not args.allow_model_calls:
            setup += "os.environ.pop('OPENROUTER_API_KEY', None)\nimport socket\nfrom unittest.mock import patch\n_network_guard = patch.object(socket.socket, 'connect', side_effect=AssertionError('Notebook verification blocks outbound network'))\n_network_guard.start()\n"
        notebook.cells.insert(0, nbformat.v4.new_code_cell(setup))
        state, error = 'complete', None
        try:
            def save_progress(cell, cell_index, execute_reply):
                import copy
                progress = copy.deepcopy(notebook)
                progress.cells.pop(0)
                for item in progress.cells:
                    if item.cell_type == 'code' and item.execution_count is not None:
                        item.execution_count -= 1
                progress.metadata['qwen_execution'] = dict(state='running',mode=args.mode,timestamp_utc=stamp)
                nbformat.write(progress,path)
            NotebookClient(notebook, timeout=7200, kernel_name='qwen-notebooks',
                           on_cell_executed=save_progress,
                           resources={'metadata': {'path': str(path.parent)}}).execute()
        except CellExecutionError:
            state = 'failed'
            for cell in notebook.cells:
                for output in cell.get('outputs', []):
                    if output.output_type == 'error':
                        error = output.evalue
                        if output.ename == 'StudyNotReady' or 'StudyNotReady' in '\n'.join(output.get('traceback', [])):
                            state = 'pending'
                            cell.outputs = [nbformat.v4.new_output('stream', name='stdout', text='Pending first model execution: ' + error + '\n')]
                            cell.execution_count = None
        notebook.cells.pop(0)
        for cell in notebook.cells:
            cell.metadata.pop('execution', None)
            if cell.cell_type == 'code' and cell.execution_count is not None:
                cell.execution_count -= 1
        record = dict(notebook=str(relative), state=state, mode=args.mode, new_model_calls_enabled=args.allow_model_calls,
                      executed_code_cells=sum(c.cell_type == 'code' and c.execution_count is not None for c in notebook.cells),
                      total_code_cells=sum(c.cell_type == 'code' for c in notebook.cells), reason=error)
        notebook.metadata['qwen_execution'] = dict(timestamp_utc=stamp, **record)
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        outcomes.append(record)
        print(json.dumps(record), flush=True)
        if state == 'failed':
            break  # Fix a failed notebook before spending on subsequent stages.
    report = PROJECT/'reports/notebook-execution-summary.json'
    report.write_text(json.dumps(outcomes, indent=2)+'\n')
    state = 'failed' if any(row['state']=='failed' for row in outcomes) else 'pending' if any(row['state']=='pending' for row in outcomes) else 'complete'
    (PROJECT/'.cache/live-rebuild-status.json').write_text(json.dumps(dict(state=state,
        pid=os.getpid(),process_running=False,eligible_questions=15585,cached_passages=152576,
        new_passage_embedding_calls=0,notebooks=outcomes),indent=2)+'\n')
    return int(any(row['state'] == 'failed' for row in outcomes))


if __name__ == '__main__':
    raise SystemExit(main())
