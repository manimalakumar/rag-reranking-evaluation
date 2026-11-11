"""Run main tutorial notebooks in Live (default), Replay or Plan mode.

Original tutorial sections execute in order; outputs are written to the main notebook.
No completed notebook is silently treated as a completed new Live run.
"""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess,sys

import nbformat
from nbclient import NotebookClient

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from shared_scoring.live_tutorial import NOTEBOOKS,TutorialRun


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',choices=list(NOTEBOOKS),required=True)
    parser.add_argument('--mode',choices=['live','replay','plan'],default='live')
    parser.add_argument('--run-id',help='Explicit tutorial run ID for Replay or same-run resume')
    parser.add_argument('--resume',action='store_true',help='Explicitly reuse only the named Live run')
    parser.add_argument('--inputs',help='Prepared tutorial_inputs.json; otherwise use TUTORIAL_INPUTS or the recorded reference profile')
    parser.add_argument('--retry-in-flight',action='store_true',help='After inspection, retry a temporary in-flight-budget 402 within the existing attempt cap; requires --resume')
    args=parser.parse_args()
    if args.inputs:os.environ['TUTORIAL_INPUTS']=args.inputs
    if args.retry_in_flight and (not args.resume or args.mode!='live'):
        parser.error('--retry-in-flight requires --mode live and --resume')
    run_id=args.run_id
    run=TutorialRun(PROJECT,args.study,args.mode,run_id,resume=args.resume)
    if args.mode=='live':
        if run.plan['missing_inputs']:raise SystemExit('Missing frozen inputs; see setup.md:\n'+'\n'.join(run.plan['missing_inputs']))
        if not os.environ.get('OPENROUTER_API_KEY'):raise SystemExit('Set OPENROUTER_API_KEY in the launch environment. No request sent.')
    subprocess.run([sys.executable,str(PROJECT/'tools/verify_tutorial.py')],cwd=PROJECT,check=True)
    if args.mode=='plan':
        print(__import__('json').dumps(run.plan,indent=2))
        return  # Inspect the plan without overwriting notebook outputs.
    cache=PROJECT/'.cache/notebook';kernel=cache/'kernels/qwen-notebooks';kernel.mkdir(parents=True,exist_ok=True)
    (kernel/'kernel.json').write_text(__import__('json').dumps(dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],display_name='Qwen RAG tutorials',language='python')))
    os.environ.update(JUPYTER_PATH=str(cache),JUPYTER_RUNTIME_DIR=str(cache/'runtime'),IPYTHONDIR=str(cache/'ipython'),
        NOTEBOOK_MODE=args.mode,TUTORIAL_RUN_ID=run.run_id,TUTORIAL_RESUME='1' if args.resume else '0',
        TUTORIAL_RETRY_IN_FLIGHT='1' if args.retry_in_flight else '0',
        TUTORIAL_TESTS_PASSED='1',HAYSTACK_TELEMETRY_ENABLED='False',RAGAS_DO_NOT_TRACK='true')
    path=PROJECT/NOTEBOOKS[args.study];nb=nbformat.read(path,as_version=4)
    for c in nb.cells:
        if c.cell_type=='code' and c.metadata.get('live_tutorial'):
            c.outputs=[];c.execution_count=None
    # Network is blocked at the kernel level for modes that promise no inference.
    if args.mode!='live':
        guard=nbformat.v4.new_code_cell("import socket\nfrom unittest.mock import patch\n_network_guard = patch.object(socket.socket, 'connect', side_effect=AssertionError('Replay/Plan blocks all network'))\n_network_guard.start()\n")
        nb.cells.insert(0,guard)
    state='complete'
    try:
        NotebookClient(nb,timeout=14400,kernel_name='qwen-notebooks',skip_cells_with_tag='skip-execution',
            resources={'metadata':{'path':str(path.parent)}}).execute()
    except Exception:
        state='failed';raise
    finally:
        if args.mode!='live':nb.cells.pop(0)
        for c in nb.cells:
            if c.cell_type=='code':
                executed=c.execution_count is not None
                c.metadata['source_changed_since_execution']=not executed
                c.metadata['execution_state']='executed' if executed else 'not_executed'
                if executed:c.metadata['output_provenance']={'run_id':run.run_id,'mode':args.mode,'revised_source_executed':True}
        nb.metadata['live_tutorial']['hosted_execution_state']=state if args.mode=='live' else 'replayed'
        nb.metadata['live_tutorial']['last_execution']=dict(mode=args.mode,run_id=run.run_id,state=state,
            timestamp_utc=datetime.now(timezone.utc).isoformat())
        nbformat.write(nb,path)
    print(f'Saved {args.mode} outputs in the main notebook: {path.relative_to(PROJECT)}')

if __name__=='__main__':main()
