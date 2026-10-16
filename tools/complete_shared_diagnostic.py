"""Execute only separate diagnostic cells; preserve all existing experiment results."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import sys

import nbformat

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def execute_cell(source):
    """Capture one real IPython cell execution, with no notebook setup/replay."""
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    shell = InteractiveShell.instance()
    with capture_output() as captured:
        result = shell.run_cell(source, store_history=False)
    if result.error_before_exec or result.error_in_exec:
        if captured.stdout:
            print(captured.stdout, flush=True)
        if captured.stderr:
            print(captured.stderr, file=sys.stderr, flush=True)
        raise RuntimeError(str(result.error_before_exec or result.error_in_exec))
    outputs = []
    if captured.stdout:
        outputs.append(nbformat.v4.new_output('stream', name='stdout', text=captured.stdout))
    if captured.stderr:
        outputs.append(nbformat.v4.new_output('stream', name='stderr', text=captured.stderr))
    for output in captured.outputs:
        outputs.append(nbformat.v4.new_output('display_data', data=output.data, metadata=output.metadata))
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-model-calls', action='store_true')
    args = parser.parse_args()
    os.chdir(PROJECT)
    local = PROJECT/'.cache/notebook'
    (local/'ipython').mkdir(parents=True, exist_ok=True)
    os.environ.update(IPYTHONDIR=str(local/'ipython'), HAYSTACK_TELEMETRY_ENABLED='False',
                      RAGAS_DO_NOT_TRACK='true', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      NOTEBOOK_MODE='execute' if args.allow_model_calls else 'replay',
                      NOTEBOOK_ALLOW_MODEL_CALLS='1' if args.allow_model_calls else '0')
    if args.allow_model_calls and os.environ.get('NOTEBOOK_TESTS_PASSED') != '1':
        raise RuntimeError('Run the targeted offline tests before enabling diagnostic calls.')
    shared_setup = '''from pathlib import Path
import os, sys
PROJECT = next(p for p in [Path.cwd(), *Path.cwd().parents]
               if (p / "configs/experiment1.json").is_file())
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
allow_diagnostic_calls = (os.environ.get("NOTEBOOK_MODE", "execute") == "execute"
                          and os.environ.get("NOTEBOOK_ALLOW_MODEL_CALLS") == "1")
'''
    retrieval_source = shared_setup + '''# Reuse the cached larger-corpus top 30; make only the missing diagnostic reranker call.
from shared_scoring.shared_diagnostic import prepare_retrieval, show_retrieval
larger_corpus_diagnostic = prepare_retrieval(PROJECT, allow_model_calls=allow_diagnostic_calls)
show_retrieval(larger_corpus_diagnostic)
'''
    answer_source = shared_setup + '''# Two diagnostic-only generation calls plus explicit judging; successful caches are reused.
# No sampled question is regenerated and no subgroup result is recomputed.
from shared_scoring.shared_diagnostic import prepare_answers, show_answers
larger_corpus_answers = await prepare_answers(PROJECT, allow_model_calls=allow_diagnostic_calls,
                                             judge=allow_diagnostic_calls)
show_answers(PROJECT, larger_corpus_answers)
'''
    stages = [
        ('experiments/v2-targeted-reranking/Experiment-2-Targeted-Reranking.ipynb', 25, 26,
         '### Separate diagnostic—larger corpus\n\nRun **1100870 ("effervescent meaning")** separately over the **152,576 cached passages**. '
         'Reuse its frozen embedding top 30 and rerank them. The **500 opportunities + 200 controls** and their averages stay unchanged.',
         retrieval_source, ()),
        ('experiments/v3-answer-quality/Experiment-3-Answer-Quality.ipynb', 15, 16,
         '## 8. Separate diagnostic—larger corpus\n\n**79 sampled questions + 1 deliberately included diagnostic:** '
         '**1100870 ("effervescent meaning")**. Display its own larger-corpus contexts, two new answers and saved scores. '
         'The diagnostic is excluded from all subgroup averages. These are separate from the earlier 10,210-passage baseline answers.',
         answer_source, (17,)),
    ]
    # Verify frozen experiment artifacts are not changed by the diagnostic workflow.
    protected = {}
    for topic in ('targeted-reranking', 'answer-quality'):
        for path in (PROJECT/'.cache/qwen-runs'/topic).rglob('*'):
            if path.is_file():
                protected[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    for relative, heading, index, markdown, source, clear in stages:
        path = PROJECT/relative
        original = nbformat.read(path, as_version=4)
        modified = deepcopy(original)
        print('Executing only separate diagnostic cell:', relative, flush=True)
        outputs = execute_cell(source)
        modified.cells[heading].source = markdown
        modified.cells[index].source = source
        modified.cells[index].outputs = outputs
        modified.cells[index].execution_count = 1
        modified.cells[index].metadata.pop('saved_baseline_diagnostic', None)
        modified.cells[index].metadata['diagnostic_execution'] = 'isolated cell; core experiment cells not rerun'
        for extra in clear:
            modified.cells[extra].source = '# Separate larger-corpus diagnostic is shown above; excluded from all sample aggregates.'
            modified.cells[extra].outputs = []
            modified.cells[extra].execution_count = None
        from shared_scoring.shared_diagnostic import location
        from shared_scoring.io import read_json
        _, config, folder = location(PROJECT)
        outcomes = [read_json(p) for p in (folder/'standard_scoring/ragas_results').glob('*.json')]
        if index == 16 and len(outcomes) != len(config['arms'])*len(config['judge']['metrics']):
            raise RuntimeError('Some diagnostic judge outcomes are missing; cached answers retained.')
        modified.metadata['qwen_shared_diagnostic'] = dict(query_id='1100870', state='complete',
            corpus_passages=152576, excluded_from_aggregates=True,
            completed_utc=datetime.now(timezone.utc).isoformat(),
            judge_successes=sum(r.get('status')=='ok' for r in outcomes),
            judge_failures=sum(r.get('status')!='ok' for r in outcomes))
        changed = {heading, index, *clear}
        for i, (old, new) in enumerate(zip(original.cells, modified.cells)):
            if i not in changed:
                assert old == new, ('Unexpected changed notebook cell', relative, i)
        assert nbformat.read(path, as_version=4) == original, 'Concurrent notebook edit; refusing to overwrite.'
        for artifact, digest in protected.items():
            assert hashlib.sha256(artifact.read_bytes()).hexdigest() == digest, ('Frozen artifact changed', artifact)
        nbformat.validate(modified)
        temporary = path.with_suffix('.ipynb.tmp')
        nbformat.write(modified, temporary)
        temporary.replace(path)
        print('Saved separate diagnostic outputs:', relative, flush=True)
    print('Separate larger-corpus diagnostic finished; frozen cohorts and results unchanged.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
