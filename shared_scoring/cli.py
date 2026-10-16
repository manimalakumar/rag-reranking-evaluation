"""Command-line entry points distinguish saved replay from explicit scoring."""
import argparse
import asyncio
from pathlib import Path
from .io import load_configuration
from .records import load_saved_inputs, load_saved_scores
from .actions import run_squad_scoring, rebuild_saved_comparison
from .judge import run_ragas_evaluation


def main(default_action=None):
    """Require configured paths and an explicit flag before judge evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['replay', 'inputs', 'squad', 'judge', 'summary'], nargs='?', default=default_action or 'replay')
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--allow-model-calls', action='store_true')
    parser.add_argument('--retry-failures', action='store_true')
    args = parser.parse_args()
    config = load_configuration(args.config)
    if args.action == 'squad':
        if not args.output:
            parser.error('SQuAD scoring requires --output pointing to a new directory.')
        print(run_squad_scoring(config, args.output).to_string(index=False))
    elif args.action == 'judge':
        if not args.allow_model_calls:
            parser.error('Judge calls require --allow-model-calls. Use replay to display recorded scores.')
        inputs, _, _, _ = load_saved_inputs(config)
        output = args.output or config['score_directory']
        results = asyncio.run(run_ragas_evaluation(inputs, config['source_directory'], config['run_id'], output,
            config['judge'], allow_model_calls=True, retry_failures=args.retry_failures))
        print(results[['query_id', 'arm', 'metric', 'status', 'score']].to_string(index=False))
    elif args.action == 'inputs':
        _, _, failures, audit = load_saved_inputs(config)
        print(f"{config['experiment_name']}: {audit['baseline_questions']} questions; {audit['complete_baseline_pairs']} complete pairs; {len(failures)} original failures")
    elif args.action == 'summary':
        print(rebuild_saved_comparison(config)[1].to_string(index=False))
    else:
        print(load_saved_scores(config)['summary'].to_string(index=False))

if __name__ == '__main__':
    main()
