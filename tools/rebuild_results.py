"""Always pass local tests before permitting fresh OpenRouter inference."""
import argparse
import asyncio
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=['baseline', 'diagnosis', 'targeted', 'answers', 'latency', 'all'], default='baseline')
    parser.add_argument('--allow-model-calls', action='store_true')
    parser.add_argument('--judge', action='store_true', help='Also run RAGAS on rebuilt answers')
    args = parser.parse_args()
    subprocess.run([sys.executable, str(PROJECT / 'tools/verify_offline.py')], cwd=PROJECT, check=True)
    if not args.allow_model_calls:
        print('Offline tests passed. No model calls enabled; add --allow-model-calls to rebuild.')
        return
    if not os.environ.get('OPENROUTER_API_KEY'):
        raise SystemExit('Tests passed. Export OPENROUTER_API_KEY to rebuild results.')
    os.environ['ALLOW_MODEL_CALLS'] = '1'
    os.environ['HAYSTACK_TELEMETRY_ENABLED'] = 'False'
    from shared_scoring.execution import (ensure_baseline_study, ensure_diagnostic_data,
        ensure_targeted_study, ensure_answer_quality_study, ensure_latency_study)
    from shared_scoring.io import load_configuration, write_json
    from shared_scoring.judge import run_ragas_evaluation
    from shared_scoring.opportunity_study import run_opportunity_judge
    selected = ['baseline', 'diagnosis', 'targeted', 'answers', 'latency'] if args.study == 'all' else [args.study]
    prepared = {}
    notebook_paths = {
        'baseline': 'experiments/v1-baseline/Experiment-1-Baseline.ipynb',
        'diagnosis': 'diagnosis-retrieval-opportunity/Diagnosis-Retrieval-Opportunity.ipynb',
        'targeted': 'experiments/v2-targeted-reranking/Experiment-2-Targeted-Reranking.ipynb',
        'answers': 'experiments/v3-answer-quality/Experiment-3-Answer-Quality.ipynb',
        'latency': 'experiments/v4-latency-tradeoff/Experiment-4-Latency-Tradeoff.ipynb',
    }
    def save_notebook(study):
        # Inference has already been recorded by this stage. Populate outputs from
        # those exact saved results; the notebook runner blocks outbound sockets.
        subprocess.run([sys.executable, str(PROJECT / 'tools/execute_notebooks.py'),
                        '--mode', 'replay', str(PROJECT / notebook_paths[study])], cwd=PROJECT, check=True)
    for study in selected:
        if study == 'baseline':
            config = ensure_baseline_study(load_configuration(PROJECT / 'configs/experiment1.json'), PROJECT)
        elif study == 'diagnosis':
            prepared[study] = str(ensure_diagnostic_data(PROJECT / 'diagnosis-retrieval-opportunity', PROJECT))
            write_json(PROJECT / '.cache/qwen-runs/prepared-runs.json', prepared)
            save_notebook(study)
            continue
        else:
            version, function = {'targeted': ('v2-targeted-reranking', ensure_targeted_study),
                'answers': ('v3-answer-quality', ensure_answer_quality_study),
                'latency': ('v4-latency-tradeoff', ensure_latency_study)}[study]
            value = function(PROJECT / 'experiments' / version / 'config.json', PROJECT)
            config = asyncio.run(value) if asyncio.iscoroutine(value) else value
        if args.judge:
            if study == 'baseline':
                from shared_scoring import load_saved_inputs
                inputs, _, _, _ = load_saved_inputs(config)
                asyncio.run(run_ragas_evaluation(inputs, config['source_directory'], config['run_id'],
                    config['score_directory'], config['judge'], allow_model_calls=True))
                from shared_scoring.fresh_baseline import prepare_baseline_scores
                prepare_baseline_scores(config)
            elif study in {'answers', 'latency'}:
                asyncio.run(run_opportunity_judge(config, allow_model_calls=True))
        prepared[study] = {'run_id': config['run_id'], 'score_directory': str(config['score_directory'])}
        write_json(PROJECT / '.cache/qwen-runs/prepared-runs.json', prepared)
        save_notebook(study)
        print(f"Prepared {study}: {config['run_id']}", flush=True)
    write_json(PROJECT / '.cache/qwen-runs/prepared-runs.json', prepared)


if __name__ == '__main__':
    main()
