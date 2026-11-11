"""Prepare public tutorial inputs; default prints a plan without network or model calls."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from shared_scoring.input_preparation import prepare_inputs,read_recipe
from shared_scoring.io import read_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=['small','reference'],default='small')
    parser.add_argument('--passages',type=int,default=512,help='Small profile passage limit; at least 30')
    parser.add_argument('--output',help='New output directory under .cache; tracked configs are never overwritten')
    parser.add_argument('--allow-model-calls',action='store_true',help='Run offline tests, then explicitly enable paid document embedding preparation')
    args=parser.parse_args()
    if args.passages<30:parser.error('--passages must be at least 30')
    if not args.allow_model_calls:
        count=read_recipe(PROJECT/'configs/reference_input_recipe.json.gz')['passage_count'] if args.profile=='reference' else args.passages
        dimensions=read_json(PROJECT/'configs/execution.json')['retrieval']['dimensions']
        print(f'PLAN: {args.profile}: {count:,} passages, {dimensions} dimensions. Matrix plus index vectors: about {2*count*dimensions*4/1024**2:.1f} MiB, plus datasets and request traces.')
        print('No downloads, model calls or writes. Add --allow-model-calls to prepare. Small supports warm-up; reference supports all six notebooks.')
        return
    subprocess.run([sys.executable,str(PROJECT/'tools/verify_offline.py')],cwd=PROJECT,check=True)
    if not os.environ.get('OPENROUTER_API_KEY'):raise SystemExit('Export OPENROUTER_API_KEY before preparation. No dataset download or request sent.')
    os.environ.update(ALLOW_MODEL_CALLS='1',HAYSTACK_TELEMETRY_ENABLED='False',RAGAS_DO_NOT_TRACK='true')
    path=prepare_inputs(PROJECT,args.profile,args.passages,args.output)
    print('Prepared inputs:',path.relative_to(PROJECT))
    print('Set TUTORIAL_INPUTS to this path before launching the notebook kernel. See setup.md.')

if __name__=='__main__':main()
