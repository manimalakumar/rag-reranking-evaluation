"""Explicit notebook execution/replay policy; never silently resume stopped embedding."""
import os
from pathlib import Path
import subprocess
import sys

from .fresh_retrieval import execution_settings, fingerprint
from .io import load_configuration, read_json, sha256_file
from .opportunity_study import load_opportunity_configuration
from .execution import _portable, _retrieval_directory


class StudyNotReady(RuntimeError):
    """A notebook's next stage has no completed Qwen artifacts yet."""


class NotebookSession:
    def __init__(self, project):
        self.project = Path(project)
        self.mode = os.environ.get('NOTEBOOK_MODE', 'execute')
        if self.mode not in {'execute', 'replay'}:
            raise ValueError('NOTEBOOK_MODE must be execute or replay.')
        self.settings = execution_settings(self.project, include_scope=True)
        self.tested = os.environ.get('NOTEBOOK_TESTS_PASSED') == '1'

    def retrieval_directory(self):
        settings = self.settings
        return _retrieval_directory(settings)

    def require_complete_index(self):
        folder = self.retrieval_directory()
        path = folder / 'assets_manifest.json'
        if not path.exists() or read_json(path).get('status') != 'complete' or not (folder / 'index.faiss').is_file():
            # Do not turn a notebook run into an implicit full embedding restart.
            raise StudyNotReady('Full-corpus Qwen retrieval index is unfinished. The saved embedding cache is preserved. '
                                'Embedding was stopped by the user; this notebook will not resume it. '
                                'Complete or explicitly redesign the retrieval scope before running this stage.')
        return folder

    def enable_calls(self):
        if self.mode != 'execute':
            raise StudyNotReady('Replay mode only reads saved results; no model calls are permitted.')
        if os.environ.get('NOTEBOOK_ALLOW_MODEL_CALLS') != '1':
            raise StudyNotReady('This stage needs its first Qwen model execution. Set NOTEBOOK_ALLOW_MODEL_CALLS=1 '
                                'and OPENROUTER_API_KEY in the kernel environment to run it explicitly.')
        if not os.environ.get('OPENROUTER_API_KEY'):
            raise StudyNotReady('Set OPENROUTER_API_KEY in the kernel environment; never put credentials in notebook cells.')
        if not self.tested:
            subprocess.run([sys.executable, str(self.project / 'tools/verify_offline.py')], cwd=self.project, check=True)
            self.tested = True
        os.environ['ALLOW_MODEL_CALLS'] = '1'

    def saved_configuration(self, stage):
        """Resolve the exact current configuration's cache, never an arbitrary old run."""
        settings = self.settings
        if stage == 'baseline':
            config = load_configuration(self.project / 'configs/experiment1.json')
            if settings.get('scope'):
                from .scoped_baseline import prepare_scoped_baseline
                config = prepare_scoped_baseline(config, self.project, settings)
            contract = {'execution': _portable(settings, settings['runtime_directory']),
                        'baseline': read_json(config['source_directory'] / 'config.json'),
                        'membership_sha256': sha256_file(config['source_directory'] / 'input/baseline_ids.json'),
                        'reviewed_input_sha256': sha256_file(config['source_directory'] / 'input/golden_reviewed.json')}
            folder = settings['runtime_directory'] / 'baseline' / fingerprint(contract)[:20]
            run_id = 'baseline-fresh-' + folder.name
            if not (folder / 'runs' / run_id / 'results.json').is_file():
                return None
            config.update(source_directory=folder, run_id=run_id, score_directory=folder / 'standard_scoring',
                          fresh_run=True, historical_reference_directory=self.project / 'experiments/v1-baseline/reference',
                          document_embedding_example=os.path.relpath(folder / 'document_embedding_example.json', self.project))
            return config
        version, topic = {'targeted': ('v2-targeted-reranking', 'targeted-reranking'),
                          'answers': ('v3-answer-quality', 'answer-quality'),
                          'latency': ('v4-latency-tradeoff', 'latency-tradeoff')}[stage]
        path = self.project / 'experiments' / version / 'config.json'
        identity = read_json(path)
        folder = settings['runtime_directory'] / topic / fingerprint(
            {'config': identity, 'execution': _portable(settings, settings['runtime_directory'])})[:20]
        if not (folder / 'config.json').is_file():
            return None
        config = load_opportunity_configuration(folder / 'config.json')
        required = [config['questions_file'], config['passages_file']]
        if stage in {'answers', 'latency'}:
            required.append(config['generation_manifest_file'])
        if stage == 'targeted':
            required.append(config['retrieval_summary_file'])
        return config if all(p.is_file() for p in required) else None

    def preflight(self, stage):
        if stage != 'baseline':
            self.require_complete_index()
        self.enable_calls()
