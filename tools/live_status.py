"""Read local rebuild progress without making network requests."""
import json
import math
import os
from pathlib import Path
import sqlite3

PROJECT = Path(__file__).resolve().parents[1]
CACHE = PROJECT / '.cache'


def main():
    status = CACHE / 'live-rebuild-status.json'
    result = json.loads(status.read_text()) if status.exists() else {'state': 'not_started'}
    pidfile = CACHE / 'live-rebuild.pid'
    if pidfile.exists():
        result['pid'] = int(pidfile.read_text())
        try:
            os.kill(result['pid'], 0)
            result['process_running'] = True
        except ProcessLookupError:
            result['process_running'] = False
        except PermissionError:
            result['process_running'] = None
    limits = json.loads((PROJECT / 'configs/model_limits.json').read_text())
    ledger = CACHE / 'model-request-budget.sqlite'
    result['request_cap'] = limits['max_total_requests']
    if ledger.exists():
        with sqlite3.connect(f'file:{ledger}?mode=ro', uri=True) as connection:
            result['reserved_attempts'] = connection.execute('SELECT COALESCE(SUM(count),0) FROM attempts').fetchone()[0]
    for folder in sorted((CACHE / 'qwen-runs').glob('baseline/*')):
        paths = list((folder / 'requests').glob('*/attempt-*.json'))
        if not paths:
            continue
        latest = max(paths, key=lambda p: p.stat().st_mtime)
        try:
            trace = json.loads(latest.read_text())
        except (OSError, ValueError):
            continue
        result['baseline'] = {'latest_stage': trace['stage'], 'latest_status': trace['status'],
                              'cached_attempt_files': len(paths)}
        if trace['stage'] == 'passage_embedding':
            settings = json.loads((PROJECT / 'experiments/v1-baseline/config.json').read_text())['models_and_dataset']
            corpus = json.loads((PROJECT / 'experiments/v1-baseline/input/corpus.json').read_text())
            result['baseline']['document_batches_total'] = math.ceil(len(corpus) / settings['embedding_batch_size'])
            result['baseline']['document_batches_complete'] = len(list((folder / 'requests').iterdir())) - (trace['status'] != 'ok')
    for folder in sorted((CACHE / 'qwen-runs/full-validation').glob('*')):
        manifest = folder / 'assets_manifest.json'
        if not manifest.exists():
            prepared = folder / 'dataset_manifest.json'
            if prepared.exists():
                data = json.loads(prepared.read_text())
                result['full_validation'] = {'asset_status': 'dataset_ready',
                    'unique_passages': data['unique_passages'], 'validation_questions': data['validation_questions'],
                    'cached_request_groups': len(list((folder / 'requests').glob('*')))}
            continue
        data = json.loads(manifest.read_text())
        dataset = data['contract']['dataset']
        retrieval = data['contract']['retrieval']
        result['full_validation'] = {'asset_status': data['status'],
            'document_batches_complete': len(list((folder / 'embedding_batches').glob('*.json'))),
            'document_batches_total': math.ceil(dataset['unique_passages'] / retrieval['passage_batch_size']),
            'retrieval_batches_complete': len(list((folder / 'retrieval_batches').glob('*/manifest.json')))}
    if (PROJECT/'configs/retrieval_scope.json').is_file():
        import sys
        sys.path.insert(0,str(PROJECT))
        from shared_scoring.fresh_retrieval import execution_settings
        from shared_scoring.execution import _retrieval_directory
        settings=execution_settings(PROJECT,include_scope=True)
        folder=_retrieval_directory(settings)
        result['cached_scope']={
            'eligible_questions':settings['scope']['eligible_question_count'],
            'passages':settings['scope']['passage_count'],
            'passage_embedding_calls_allowed':False,
            'asset_status':json.loads((folder/'assets_manifest.json').read_text()).get('status') if (folder/'assets_manifest.json').exists() else 'not_prepared',
            'retrieval_batches_complete':len(list((folder/'retrieval_batches').glob('*/manifest.json'))),
        }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
