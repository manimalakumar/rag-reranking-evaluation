"""Update only IMEI diagnostic sections; never execute a full notebook."""
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

SETUP='''from pathlib import Path
import sys
PROJECT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "configs/experiment1.json").is_file())
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from shared_scoring.imei_diagnostic import prepare_small, show_small, show_large_retrieval, show_large_answers, paths
from IPython.display import display, Markdown
import pandas as pd
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-model-calls',action='store_true')
    args=parser.parse_args()
    os.chdir(PROJECT)
    local=PROJECT/'.cache/notebook/ipython';local.mkdir(parents=True,exist_ok=True)
    os.environ.update(IPYTHONDIR=str(local),HAYSTACK_TELEMETRY_ENABLED='False',RAGAS_DO_NOT_TRACK='true',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    if args.allow_model_calls:
        if os.environ.get('NOTEBOOK_TESTS_PASSED')!='1':raise RuntimeError('Offline tests must pass first.')
        os.environ['ALLOW_MODEL_CALLS']='1'
    protected={}
    for topic in ('baseline','targeted-reranking','answer-quality'):
        for f in (PROJECT/'.cache/qwen-runs'/topic).rglob('*'):
            if f.is_file():protected[f]=hashlib.sha256(f.read_bytes()).hexdigest()
    plans=[
      ('experiments/v1-baseline/Experiment-1-Baseline.ipynb',{
       26:'### IMEI diagnostic scoring — separate from the 31-question baseline averages\n\nCompare both paths for **1053665, “what is imei and where do i find it”**, using the unchanged **10,210-passage corpus**. Neither annotated support is present. The diagnostic’s new answers and deterministic scores are stored separately; the original 31-question averages are unchanged.',
       30:'## 11. Inspect the separate IMEI diagnostic\n\nTrace **1053665** through the unchanged **10,210-passage baseline corpus**. It contains **0 of 2 annotated supports**. Compare each path’s passages, prompts, answers and diagnostic scores. No human correctness labels are assigned. Later notebooks show the saved larger-corpus recovery.',
       32:'### Reused query embedding → exact baseline top 30\n\nThe existing Qwen query embedding is reused for local Haystack cosine retrieval. No passage or query embedding request is made. Inspect the thirty candidates below.',
       34:'### IMEI: embedding top five and reranked top five\n\nThese are the separate diagnostic’s saved contexts from the unchanged 10,210-passage corpus. Neither contains an annotated support passage.',
       36:'### IMEI: exact contexts, prompts and two new diagnostic answers\n\nThe generator receives only each path’s own five passages. These diagnostic-only requests are cached separately from the original baseline run.',
       38:'### IMEI: evaluation of the recorded diagnostic answers\n\nSQuAD measures reference overlap, not evidence support. Annotated-support Recall@5 is zero in both paths because both annotated supports are absent from the corpus. These scores do not enter the 31-question averages.'
      },{
       27:SETUP+'''import os
imei_small = prepare_small(PROJECT, allow_model_calls=os.environ.get("IMEI_ALLOW_MODEL_CALLS") == "1")
show_small(PROJECT, imei_small)
''',
       31:SETUP+'''imei_small = prepare_small(PROJECT, allow_model_calls=False)
display({"query_id": "1053665", "question": imei_small["retrieval"]["question"],
         "corpus_passages": 10210, "annotated_supports_present": 0, "annotated_supports_total": 2,
         "excluded_from_baseline_averages": True})
''',
       33:SETUP+'''imei_small = prepare_small(PROJECT, allow_model_calls=False)
display({"query_embedding_origin": imei_small["retrieval"]["query_embedding_origin"],
         "embedding_dimensions": 2560, "new_embedding_calls": 0})
display(pd.DataFrame(imei_small["retrieval"]["top30"])[["embedding_rank", "passage_id", "embedding_score", "text"]])
''',
       35:SETUP+'''show_small(PROJECT, prepare_small(PROJECT, allow_model_calls=False), part="retrieval")
''',
       37:SETUP+'''show_small(PROJECT, prepare_small(PROJECT, allow_model_calls=False), part="answers")
''',
       39:SETUP+'''imei_small = prepare_small(PROJECT, allow_model_calls=False)
display(pd.read_csv(PROJECT / imei_small["score_file"])[["path", "squad_token_f1_pct", "squad_exact_match_pct", "annotated_support_recall_at_5"]])
print("Separate IMEI diagnostic; original 31-question averages unchanged. This display makes no model calls.")
'''
      }),
      ('experiments/v2-targeted-reranking/Experiment-2-Targeted-Reranking.ipynb',{
       23:'### Separate diagnostic: IMEI support recovered in the larger corpus\n\nThe **10,210-passage baseline corpus has 0/2 annotated supports** for **1053665**. In the **152,576-passage corpus**, both supports are available. Reranking promotes one from **rank 6 → 3**, improving **Recall@5 from 0.5 → 1.0**.\n\nIMEI already belongs to the frozen 500-opportunity + 200-control cohort. This walkthrough reuses its saved results and does not add another observation or change the averages.'
      },{24:SETUP+'show_large_retrieval(PROJECT)\n'}),
      ('experiments/v3-answer-quality/Experiment-3-Answer-Quality.ipynb',{
       15:'## 8. Separate diagnostic: IMEI answer quality\n\nInspect **1053665** using the original saved **152,576-passage corpus** contexts, answers and scores. The recovered support moves **rank 6 → 3**; **Token F1 rises 33.66% → 51.16%**, and **judged faithfulness rises 0.77 → 1.00**.\n\nIMEI already belongs to the frozen **79-question sample**. Its diagnostic display does not add an extra question, change membership, or recompute subgroup averages.'
      },{16:SETUP+'show_large_answers(PROJECT)\n'})
    ]
    # Load all notebooks up front, so any concurrent user edit stops the final save.
    originals={rel:nbformat.read(PROJECT/rel,as_version=4) for rel,_,_ in plans}
    modified={}
    for relative,markdown,code in plans:
        original=originals[relative];updated=deepcopy(original)
        for i,s in markdown.items():
            assert updated.cells[i].cell_type=='markdown'
            updated.cells[i].source=s
        for i,s in code.items():
            assert updated.cells[i].cell_type=='code'
            print('Execute only IMEI diagnostic cell:',relative,i,flush=True)
            if args.allow_model_calls and relative.startswith('experiments/v1') and i==27:
                os.environ['IMEI_ALLOW_MODEL_CALLS']='1'
                outputs=execute_cell(s)
                os.environ['IMEI_ALLOW_MODEL_CALLS']='0'
            else:
                with patch.object(socket.socket,'connect',side_effect=AssertionError('Saved diagnostic display forbids network')):
                    outputs=execute_cell(s)
            updated.cells[i].source=s;updated.cells[i].outputs=outputs
            updated.cells[i].execution_count=1
            updated.cells[i].metadata.pop('saved_baseline_diagnostic',None)
            updated.cells[i].metadata['diagnostic_execution']='isolated IMEI diagnostic cell; experiment cells not rerun'
        updated.metadata['qwen_shared_diagnostic']=dict(query_id='1053665',state='complete',
            corpus_passages=10210 if relative.startswith('experiments/v1') else 152576,
            adds_sample_observation=False,baseline_excluded_from_31_question_averages=True,
            original_cohorts_and_aggregates_unchanged=True,completed_utc=datetime.now(timezone.utc).isoformat())
        for i,(a,b) in enumerate(zip(original.cells,updated.cells)):
            if i not in set(markdown)|set(code):assert a==b,('Unexpected changed cell',relative,i)
        nbformat.validate(updated);modified[relative]=updated
    for f,digest in protected.items():
        assert hashlib.sha256(f.read_bytes()).hexdigest()==digest,('Original study artifact changed',f)
    for relative,original in originals.items():
        assert nbformat.read(PROJECT/relative,as_version=4)==original,('Concurrent notebook edit',relative)
    for relative,updated in modified.items():
        path=PROJECT/relative;temp=path.with_suffix('.ipynb.tmp')
        nbformat.write(updated,temp);temp.replace(path)
        print('Saved diagnostic sections:',relative,flush=True)
    print('IMEI walkthrough complete; original experiment artifacts and all non-diagnostic cells unchanged.',flush=True)

if __name__=='__main__':main()
