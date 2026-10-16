"""The same eligible diagnostic in Experiments 1–3, outside cohort aggregates."""
from copy import deepcopy
import json

import pandas as pd

from .fresh_retrieval import RecordedOpenRouterClient, fingerprint
from .io import read_json, write_json
from .notebook_runtime import NotebookSession, StudyNotReady
from .opportunity_study import load_opportunity_configuration, generation_messages

QUERY_ID = '1100870'


def location(project):
    session = NotebookSession(project)
    config = load_opportunity_configuration(session.project/'experiments/v3-answer-quality/config.json')
    contract = {'query_id':QUERY_ID,'eligible_ids':session.settings['scope']['eligible_ids_sha256'],
                'passages':session.settings['scope']['passage_count'],
                'retrieval':session.settings['retrieval'],'generation':config['generation'],'judge':config['judge']}
    folder = session.settings['runtime_directory']/'shared-diagnostic'/fingerprint(contract)[:20]
    return session, config, folder


def prepare_retrieval(project, allow_model_calls=False):
    session, config, folder = location(project)
    allowed = set(session.settings['scope']['eligible_question_ids'].read_text().splitlines())
    if QUERY_ID not in allowed:
        raise ValueError('Shared diagnostic is outside the frozen eligible pool.')
    path = folder/'retrieval.json'
    if path.exists():
        return read_json(path)
    if not allow_model_calls:
        raise StudyNotReady('The separate common diagnostic needs its first reranking request.')
    session.enable_calls()
    source = session.retrieval_directory()
    metrics = pd.read_parquet(source/'per_query_metrics.parquet',filters=[('query_id','==',QUERY_ID)])
    if len(metrics)!=1:
        raise ValueError('Shared diagnostic is absent from eligible retrieval metrics.')
    candidates = None
    for shard in (source/'retrieval_batches').glob('*/top30.parquet'):
        frame = pd.read_parquet(shard,filters=[('query_id','==',QUERY_ID)])
        if not frame.empty:
            candidates = frame.sort_values('rank')
            break
    if candidates is None or len(candidates)!=30:
        raise ValueError('The shared diagnostic needs its exact saved 30 candidates.')
    docs = json.loads(candidates.to_json(orient='records'))
    question = str(metrics.iloc[0].question)
    client = RecordedOpenRouterClient(dict(session.settings['retrieval']),folder/'requests')
    ranks, trace = client.rerank(question,docs)
    gold = list(metrics.iloc[0].known_supporting_passage_ids)
    reranked = [docs[item['index']] for item in ranks]
    result = dict(query_id=QUERY_ID,question=question,reference_answers=list(metrics.iloc[0].reference_answers),
                  known_supporting_passage_ids=gold,top30=docs,embedding_top5=docs[:5],reranked_top5=reranked[:5],
                  embedding_recall_at_5=float(metrics.iloc[0].embedding_recall_at_5),
                  reranker_recall_at_5=len(set(gold)&{d['passage_id'] for d in reranked[:5]})/len(gold),
                  rerank_seconds=trace['elapsed_seconds'],excluded_from_cohort_aggregates=True,
                  corpus_passages=session.settings['scope'].get('passage_count',152576))
    write_json(path,result)
    return result


async def prepare_answers(project, allow_model_calls=False, judge=False):
    session, config, folder = location(project)
    retrieval = prepare_retrieval(project,allow_model_calls)
    path = folder/'answers.json'
    inputs = read_json(path) if path.exists() else []
    if len(inputs) < 2:
        if not allow_model_calls:
            raise StudyNotReady('The common diagnostic needs its first paired-answer execution.')
        session.enable_calls()
        settings = config['generation']
        client_settings = dict(session.settings['retrieval'])
        client_settings['max_attempts'] = 1  # First generation attempt only, as in the main study.
        client = RecordedOpenRouterClient(client_settings,folder/'requests')
        from .answers import score_custom_answer
        for arm, passages in [('baseline',retrieval['embedding_top5']),('reranked',retrieval['reranked_top5'])]:
            if any(row['arm']==arm for row in inputs):
                continue
            blocks = [f"[{d['passage_id']}]\n{d['text']}" for d in passages]
            context = '\n\n'.join(blocks)
            payload = dict(model=settings['model'],messages=generation_messages(retrieval['question'],context,settings['system_prompt']),
                           temperature=settings['temperature'],max_tokens=settings['max_tokens'],**settings['extras'])
            raw, trace = client.request('generation',settings['api_base']+'/chat/completions',payload,
                                        cache_key=f'shared-diagnostic-{QUERY_ID}-{arm}')
            choice = raw['choices'][0]
            answer = choice['message']['content']
            old = score_custom_answer(answer,retrieval['reference_answers'],config['legacy_citation_pattern'])
            inputs.append(dict(query_id=QUERY_ID,question=retrieval['question'],arm=arm,path=config['arms'][arm],answer=answer,
                               reference_answers=retrieval['reference_answers'],retrieved_contexts=blocks,context=context,
                               context_passage_ids=[d['passage_id'] for d in passages],finish_reason=choice.get('finish_reason'),
                               truncated=choice.get('finish_reason')=='length',
                               annotated_support_recall_at_5=retrieval['embedding_recall_at_5' if arm=='baseline' else 'reranker_recall_at_5'],
                               existing_token_f1_0_1=old['token_f1'],existing_normalized_em_0_1=old['normalized_em']))
            write_json(path,inputs)  # Preserve each first answer even if the other arm fails.
    scores = folder/'standard_scoring'
    scores.mkdir(parents=True,exist_ok=True)
    from .answers import load_squad_metric, score_squad_answers
    if not (scores/'squad_per_answer.csv').exists():
        metric = load_squad_metric(config['squad']['metric_source'],config['squad']['cache_directory'])
        score_squad_answers(inputs,metric,config['citation_pattern'],config['legacy_citation_pattern']).to_csv(scores/'squad_per_answer.csv',index=False)
    if judge and allow_model_calls:
        expected=[scores/'ragas_results'/f'{QUERY_ID}-{arm}-{metric}.json'
                  for arm in config['arms'] for metric in config['judge']['metrics']]
        if all(p.exists() and read_json(p).get('status')!='started' for p in expected):
            return inputs  # Keep first metric outcomes; no key or API calls for replay.
        session.enable_calls()
        # Reuse the already recorded Experiment 1 embedding of this exact question.
        baseline = session.saved_configuration('baseline')
        run = baseline['source_directory']/'runs'/baseline['run_id']
        record = next(r for r in read_json(run/'results.json') if r['query_id']==QUERY_ID)
        run_id = 'shared-diagnostic-'+folder.name
        seed = folder/'runs'/run_id
        write_json(seed/'results.json',[dict(query_id=QUERY_ID,query=record['query'],query_embedding=record['query_embedding'])])
        write_json(seed/'requests'/f'{QUERY_ID}-query_embedding.json',read_json(run/'requests'/f'{QUERY_ID}-query_embedding.json'))
        judge_settings = deepcopy(config['judge'])
        judge_settings['embedding_question_policy']='reuse_saved'
        from .judge import run_ragas_evaluation
        await run_ragas_evaluation(inputs,folder,run_id,scores,judge_settings,allow_model_calls=True)
    return inputs


def show_retrieval(result):
    from html import escape
    from IPython.display import display, Markdown, HTML
    display(Markdown(f"### Separate diagnostic—larger corpus\n\n**{QUERY_ID}: {result['question']}**\n\n"
        "Retrieval over **152,576 cached passages**; exact saved top 30 reused, then reranked. "
        "Separate from the **500 opportunities + 200 controls** and excluded from their aggregates."))
    display(pd.DataFrame([dict(arm='Embedding RAG',recall_at_5=result['embedding_recall_at_5']),
                          dict(arm='Embedding + reranking RAG',recall_at_5=result['reranker_recall_at_5'])]))
    for key in ['embedding_top5','reranked_top5']:
        display(Markdown('#### '+('Embedding top five' if key=='embedding_top5' else 'Reranked top five')))
        display(pd.DataFrame(result[key])[['passage_id','rank','text']])
        for doc in result[key]:
            display(HTML('<details><summary>'+escape(doc['passage_id'])+'</summary><pre style="white-space:pre-wrap">'
                         +escape(doc['text'])+'</pre></details>'))


def show_answers(project, inputs):
    from html import escape
    from IPython.display import display, Markdown, HTML
    _,_,folder=location(project)
    display(Markdown(f'### Separate diagnostic—larger corpus\n\n**{QUERY_ID}: effervescent meaning**\n\n'
        '**79 sampled questions + 1 deliberately included diagnostic.** '
        'These new answers use each arm’s own five passages retrieved from the **152,576-passage corpus**. '
        'The diagnostic is excluded from subgroup averages and paired aggregate results.'))
    display(Markdown('**Reference answers:** '+'; '.join(inputs[0]['reference_answers'])))
    for row in inputs:
        display(Markdown('#### '+row['path']))
        display(HTML('<pre style="white-space:pre-wrap">'+escape(row['answer'])+'</pre>'))
        display(HTML('<details><summary>Exact larger-corpus context for this answer</summary>'
                     '<pre style="white-space:pre-wrap">'+escape(row['context'])+'</pre></details>'))
        display(Markdown(f"Finish reason: `{row['finish_reason']}` · Truncated: **{row['truncated']}**"))
    display(Markdown('**Saved diagnostic scores** — SQuAD percentages and annotated-support Recall@5 on 0–1.'))
    frame=pd.read_csv(folder/'standard_scoring/squad_per_answer.csv')
    display(frame[['query_id','path','squad_exact_match_pct','squad_token_f1_pct','annotated_support_recall_at_5']])
    outcomes=[read_json(p) for p in (folder/'standard_scoring/ragas_results').glob('*.json')]
    if outcomes:
        display(pd.DataFrame(outcomes)[['query_id','arm','metric','status','score']])
        failed=[r for r in outcomes if r.get('status')!='ok']
        if failed:
            display(Markdown(f'**{len(failed)} judge outcomes failed or are incomplete; these are not successful scores.**'))
    else:
        display(Markdown('Judge outcomes are not yet saved for this diagnostic.'))
