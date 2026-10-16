"""Saved replay and explicitly enabled actions for Experiment 2."""
import argparse
import asyncio
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from shared_scoring.opportunity_study import (
    load_opportunity_configuration, load_opportunity_population,
    refresh_opportunity_results, validate_frozen_opportunity_inputs,
    run_opportunity_generation, run_opportunity_squad, run_opportunity_judge,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['replay', 'validate', 'generate', 'squad', 'judge'])
    parser.add_argument('--config', type=Path,
                        default=PROJECT / 'experiments/v2-targeted-reranking/config.json')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--query-id', action='append')
    selection.add_argument('--batch-id', type=int)
    selection.add_argument('--all', action='store_true')
    parser.add_argument('--allow-model-calls', action='store_true')
    parser.add_argument('--retry-failures', action='store_true')
    args = parser.parse_args()
    config = load_opportunity_configuration(args.config)
    if args.action == 'validate':
        print(validate_frozen_opportunity_inputs(config))
    elif args.action == 'replay':
        snapshot = refresh_opportunity_results(config)
        print(f"Complete saved answer pairs: {len(snapshot['complete_ids'])}")
        print(f"Answer scoring: {snapshot['standard']['status']}")
        print(snapshot['standard']['summary'].to_string(index=False))
    elif args.action == 'squad':
        print(run_opportunity_squad(config).to_string(index=False))
    elif args.action == 'judge':
        asyncio.run(run_opportunity_judge(config, args.allow_model_calls, args.retry_failures))
    else:
        if not (args.query_id or args.batch_id is not None or args.all):
            parser.error('Choose --query-id, --batch-id or --all explicitly for generation.')
        query_ids = args.query_id
        if args.batch_id is not None:
            population = load_opportunity_population(config)
            query_ids = population.loc[population.batch_id.eq(args.batch_id), 'query_id'].tolist()
            if not query_ids:
                parser.error('That batch has no questions.')
        asyncio.run(run_opportunity_generation(config, query_ids,
                                              args.allow_model_calls, args.retry_failures))


if __name__ == '__main__':
    main()
