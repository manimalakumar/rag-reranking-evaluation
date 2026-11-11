"""Compatibility entry point for the staged Live/Replay/Plan tutorial runner.

Live is the default and always creates a fresh run identity. Choose notebook paths
explicitly; a completed historical qwen_execution never silently skips new work.
"""
import argparse
import subprocess,sys
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from shared_scoring.live_tutorial import NOTEBOOKS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebooks',nargs='+',type=Path,help='Explicit main notebook paths; use tools/run_tutorial.py --study for one study')
    parser.add_argument('--mode',choices=['live','replay','plan'],default='live')
    parser.add_argument('--run-id')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--allow-model-calls',action='store_true',help='Compatibility only: choosing Live already authorizes its declared calls')
    parser.add_argument('--judge',action='store_true',help='Compatibility only: judging is part of the declared Live workflow')
    args=parser.parse_args()
    if args.mode!='live' and args.allow_model_calls:parser.error('Replay/Plan cannot enable model calls')
    lookup={(PROJECT/relative).resolve():study for study,relative in NOTEBOOKS.items()}
    if args.run_id and len(args.notebooks)!=1:parser.error('A specific RUN_ID applies to one notebook at a time')
    for item in args.notebooks:
        path=item.resolve()
        if path not in lookup:parser.error(f'Not a declared main tutorial notebook: {item}')
        command=[sys.executable,str(PROJECT/'tools/run_tutorial.py'),'--study',lookup[path],'--mode',args.mode]
        if args.run_id:command.extend(['--run-id',args.run_id])
        if args.resume:command.append('--resume')
        subprocess.run(command,cwd=PROJECT,check=True)

if __name__=='__main__':main()
