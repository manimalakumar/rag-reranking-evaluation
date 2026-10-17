"""Small synthetic studies built inside each test's temporary directory.

These fixtures exercise storage and replay contracts, not historical study claims.
"""
import asyncio
import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from shared_scoring.io import read_json, write_json
from shared_scoring import imei_diagnostic as diagnostic
from shared_scoring.opportunity_study import load_opportunity_configuration, run_opportunity_generation

PROJECT = Path(__file__).resolve().parents[1]


def extension_project(root):
    """Generate parent answers through the real pipeline with an in-memory transport."""
    from test_fresh_execution import AsyncSession

    root = Path(root)
    write_json(root / 'configs/execution.json', read_json(PROJECT / 'configs/execution.json'))
    plan = read_json(PROJECT / 'configs/generation_extension.json')
    plan['parent_run'] = 'fixture-parent'
    parent_dir = root / '.cache/qwen-runs/fixture-parent'
    config = read_json(PROJECT / 'experiments/v2-targeted-reranking/config.json')
    config.update(questions_file='questions.parquet', passages_file='passages.parquet',
                  checkpoint_file='answers.sqlite', score_directory='scores',
                  retrieval_summary_file='retrieval.csv', expected_cohort_counts={'opportunity': 12, 'control': 4})
    config['generation']['request_min_interval_seconds'] = 0
    questions, passages = [], []
    for number, subgroup in enumerate(plan['query_ids_by_subgroup']):
        ids = [f'{number}-{i}' for i in range(4)]
        if subgroup == 'opportunity_gain':
            ids[0] = plan['required_query_id']
        selected = pd.Series(sorted(ids)).sample(n=3, random_state=plan['seed']).tolist()
        if subgroup == 'opportunity_gain' and plan['required_query_id'] not in selected:
            selected[-1] = plan['required_query_id']
        plan['query_ids_by_subgroup'][subgroup] = selected
        control = subgroup.startswith('control')
        for qid in ids:
            questions.append(dict(query_id=qid, question=f'Synthetic question {qid}',
                generation_stratum=subgroup, cohort='control' if control else 'opportunity',
                batch_id='fixture', reference_answers=['A saved test answer'],
                known_supporting_passage_ids=[f'{qid}-1'], embedding_recall_at_5=1. if control else 0.,
                embedding_recall_at_30=1., reranker_recall_at_5=1.))
            for rank in range(1, 31):
                passages.append(dict(query_id=qid, passage_id=f'{qid}-{rank}',
                    text=f'Synthetic evidence {qid} {rank}', rank=rank, rerank_rank=rank))
    write_json(root / 'configs/generation_extension.json', plan)
    write_json(parent_dir / 'config.json', config)
    pd.DataFrame(questions).to_parquet(parent_dir / 'questions.parquet', index=False)
    pd.DataFrame(passages).to_parquet(parent_dir / 'passages.parquet', index=False)
    parent = load_opportunity_configuration(parent_dir / 'config.json')
    with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'fixture', 'ALLOW_MODEL_CALLS': '1'}), \
         patch('httpx.AsyncClient', AsyncSession):
        asyncio.run(run_opportunity_generation(parent, allow_model_calls=True))
    return root


def imei_project(root):
    """Materialize synthetic IMEI artifacts, including a completed diagnostic."""
    root = Path(root)
    write_json(root / 'configs/execution.json', read_json(PROJECT / 'configs/execution.json'))
    runtime = root / '.cache/qwen-runs'
    baseline = runtime / diagnostic.BASELINE
    run = baseline / 'runs/baseline-fresh-5413aa92e9c1375fa06f'
    write_json(run / 'config.json', {})
    write_json(run / 'document_embedding_origin.json', {'matrix_sha256': 'synthetic-matrix'})
    write_json(baseline / 'input/corpus.json', [dict(id=f'fixture-{i}') for i in range(10210)])
    folder = runtime / diagnostic.TARGETED / 'input'
    folder.mkdir(parents=True)
    gold = ['support-1', 'support-2']
    pd.DataFrame([dict(query_id=diagnostic.QUERY_ID, question='Synthetic IMEI question',
        known_supporting_passage_ids=gold, embedding_recall_at_5=.5, reranker_recall_at_5=1.)]).to_parquet(folder / 'questions.parquet')
    docs = []
    for rank in range(1, 31):
        rerank_rank = 3 if rank == 6 else 6 if rank == 3 else rank
        pid = gold[0] if rank == 1 else gold[1] if rank == 6 else f'candidate-{rank}'
        docs.append(dict(query_id=diagnostic.QUERY_ID, passage_id=pid, rank=rank,
            rerank_rank=rerank_rank, is_known_support=pid in gold, text=f'Synthetic passage {rank}',
            embedding_rank=rank, embedding_score=1. / rank, id=pid))
    pd.DataFrame(docs).to_parquet(folder / 'frozen_ranked_top30.parquet')
    _, _, _, _, output = diagnostic.paths(root)
    scores = output / 'diagnostic_squad.csv'
    scores.parent.mkdir(parents=True)
    pd.DataFrame([dict(path=arm, squad_token_f1_pct=0., squad_exact_match_pct=0.,
        annotated_support_recall_at_5=0.) for arm in ['baseline', 'reranked']]).to_csv(scores, index=False)
    retrieval = dict(question='Synthetic IMEI question', top30=docs,
        embedding_top5=docs[:5], reranked_top5=sorted(docs, key=lambda d: d['rerank_rank'])[:5],
        query_embedding_origin='synthetic fixture')
    inputs = [dict(arm=arm, path=arm, answer='Synthetic answer', messages=[], context='Synthetic context')
              for arm in ['baseline', 'reranked']]
    write_json(output / 'result.json', dict(retrieval=retrieval, inputs=inputs,
        score_file=str(scores), excluded_from_baseline_averages=True))
    return root
