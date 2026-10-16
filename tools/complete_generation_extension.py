"""Execute only new exploratory generation cells; preserve every original cell."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib,os,socket,sys
from pathlib import Path
from unittest.mock import patch
import nbformat

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from tools.complete_shared_diagnostic import execute_cell
from shared_scoring.generation_extension import location,freeze
from shared_scoring.io import read_json,write_json

RELATIVE='experiments/v3-answer-quality/Experiment-3-Answer-Quality.ipynb'
SETUP='''from pathlib import Path
import os, sys
PROJECT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "configs/generation_extension.json").is_file())
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from shared_scoring.generation_extension import freeze, generate, score, show_contract, show_results
allow_extension_calls = os.environ.get("GENERATION_EXTENSION_ALLOW_CALLS") == "1"
'''
GEN=SETUP+'''# First execution: 12 frozen questions, unchanged reranked contexts, 3 pinned-provider prompts.
# Successful first responses are cached; this does not run any original experiment cell.
frozen_extension = freeze(PROJECT)
show_contract(PROJECT)
extension_answers = await generate(PROJECT, allow_model_calls=allow_extension_calls)
print("Saved extension answers:", len(extension_answers))
'''
SCORE=SETUP+'''# Explicit extension-only evaluation: SQuAD plus faithfulness and answer relevancy.
# Original query vectors are reused. Only evaluation synthetic questions may need embeddings.
extension_scores = await score(PROJECT, allow_model_calls=allow_extension_calls)
print("Saved extension score rows:", len(extension_scores))
'''
VIEW=SETUP+'''# Read extension results only; no generation, retrieval, reranking or judging.
show_results(PROJECT)
'''


def check_integrity(integrity):
    for relative,digest in integrity['parent_files'].items():
        assert hashlib.sha256((PROJECT/relative).read_bytes()).hexdigest()==digest,('Original study file changed',relative)
    assert hashlib.sha256((PROJECT/RELATIVE).read_bytes()).hexdigest()==integrity['notebook_sha256'],'Original notebook changed during extension preparation'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-model-calls',action='store_true')
    parser.add_argument('--finalize',action='store_true')
    args=parser.parse_args()
    os.chdir(PROJECT)
    local=PROJECT/'.cache/notebook/ipython';local.mkdir(parents=True,exist_ok=True)
    os.environ.update(IPYTHONDIR=str(local),HAYSTACK_TELEMETRY_ENABLED='False',RAGAS_DO_NOT_TRACK='true',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
        GENERATION_EXTENSION_ALLOW_CALLS='1' if args.allow_model_calls else '0')
    if args.allow_model_calls:
        if os.environ.get('NOTEBOOK_TESTS_PASSED')!='1':raise RuntimeError('Targeted offline tests must pass first')
        os.environ['ALLOW_MODEL_CALLS']='1'
    _,_,parent,folder=location(PROJECT)
    freeze(PROJECT)
    guard=folder/'integrity.json'
    if not guard.exists():
        protected={}
        for topic in ['baseline','targeted-reranking','answer-quality']:
            for f in (PROJECT/'.cache/qwen-runs'/topic).rglob('*'):
                if f.is_file():protected[str(f.relative_to(PROJECT))]=hashlib.sha256(f.read_bytes()).hexdigest()
        write_json(guard,dict(notebook_sha256=hashlib.sha256((PROJECT/RELATIVE).read_bytes()).hexdigest(),parent_files=protected))
    integrity=read_json(guard)
    check_integrity(integrity)
    pending=folder/'pending_executed_cells.json'
    if not args.finalize:
        cells=read_json(pending) if pending.exists() else []
        for index,source in enumerate([GEN,SCORE]):
            if len(cells)>index:continue
            print('Execute only new generation-extension cell',index+1,flush=True)
            outputs=execute_cell(source)
            cell=nbformat.v4.new_code_cell(source,outputs=outputs,execution_count=index+1,
                metadata={'generation_extension':True,'first_execution_outputs':True})
            cells.append(cell)
            write_json(pending,cells)
            check_integrity(integrity)
            print('Saved first outputs for extension cell',index+1,flush=True)
        write_json(folder/'status.json',dict(state='awaiting_qualitative_review',generated_answers=36,
            scored_answers=36,updated_utc=datetime.now(timezone.utc).isoformat()))
        print('Generation and scoring complete. Original study and notebook unchanged; ready for qualitative review and final insertion.',flush=True)
        return
    if not (folder/'assistant_review.json').exists():raise RuntimeError('Complete the short qualitative check before final insertion')
    cells=[nbformat.from_dict(c) for c in read_json(pending)]
    assert len(cells)==2
    with patch.object(socket.socket,'connect',side_effect=AssertionError('Final display forbids network')):
        outputs=execute_cell(VIEW)
    cells.append(nbformat.v4.new_code_cell(VIEW,outputs=outputs,execution_count=3,
        metadata={'generation_extension':True,'extension_execution':'saved-result display only'}))
    original=nbformat.read(PROJECT/RELATIVE,as_version=4)
    updated=deepcopy(original)
    assert 'show_answer_quality_results' in original.cells[14].source
    intro=nbformat.v4.new_markdown_cell('### Can generation use the evidence better?\n\n'
        '**Exploratory generation extension:** retrieve evidence → select evidence → use evidence well. '
        'Compare three prompts on the same frozen reranked contexts; keep the original 79-pair study unchanged.')
    additions=[intro,*cells]
    updated.cells[15:15]=additions
    assert original.cells==updated.cells[:15]+updated.cells[19:]
    updated.metadata['qwen_generation_extension']=dict(state='complete',questions=12,new_answers=36,
        serving_provider='Nebius',provider_fallbacks=False,reranked_contexts_only=True,
        original_79_pairs_unchanged=True,completed_utc=datetime.now(timezone.utc).isoformat(),
        artifacts=str(folder.relative_to(PROJECT)))
    check_integrity(integrity)
    nbformat.validate(updated)
    temp=(PROJECT/RELATIVE).with_suffix('.ipynb.tmp');nbformat.write(updated,temp);temp.replace(PROJECT/RELATIVE)
    write_json(folder/'status.json',dict(state='complete',questions=12,new_answers=36,
        updated_utc=datetime.now(timezone.utc).isoformat()))
    pending.unlink()  # Execution outputs now live only in the notebook, not a duplicate archive.
    print('Inserted four new extension cells after the current answer-quality results. All original cells and outputs preserved.',flush=True)

if __name__=='__main__':main()
