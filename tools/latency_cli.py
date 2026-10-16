"""Replay the saved latency study; new benchmark and judge calls are explicit."""
import argparse
import asyncio
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from shared_scoring.opportunity_study import load_opportunity_configuration, run_opportunity_squad, run_opportunity_judge
from shared_scoring.latency_study import validate_saved_latency_study
from shared_scoring.latency_tutorial import load_latency_tutorial, compact_latency_comparison, export_latency_quality_plot
from shared_scoring.latency_benchmark import run_new_latency_benchmark


def main():
    """Dispatch one clearly named saved-data or explicitly enabled evaluation action."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['replay', 'validate', 'squad', 'judge', 'benchmark', 'plot'])
    parser.add_argument('--config', type=Path, default=PROJECT / 'experiments/v4-latency-tradeoff/config.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--corpus-file', type=Path)
    parser.add_argument('--index-file', type=Path)
    parser.add_argument('--allow-model-calls', action='store_true')
    parser.add_argument('--retry-failures', action='store_true')
    args = parser.parse_args()
    config = load_opportunity_configuration(args.config)
    if args.action == 'validate':
        print(validate_saved_latency_study(config))
    elif args.action == 'replay':
        study = load_latency_tutorial(config)
        print(compact_latency_comparison(study).to_string(index=False))
        print(f"Answer scoring: {study['standard']['status']}; complete pairs: {len(study['complete_ids'])}")
    elif args.action == 'squad':
        print(run_opportunity_squad(config).to_string(index=False))
    elif args.action == 'judge':
        asyncio.run(run_opportunity_judge(config, args.allow_model_calls, args.retry_failures))
    elif args.action == 'benchmark':
        output = args.output or config['study_directory'] / config['benchmark']['output_directory']
        print(run_new_latency_benchmark(config, output, args.corpus_file, args.index_file,
                                       args.allow_model_calls, args.retry_failures))
    else:
        output = args.output or config['latency_plot_file']
        print(export_latency_quality_plot(load_latency_tutorial(config), output))


if __name__ == '__main__':
    main()
