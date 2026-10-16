"""Frozen prompt experiment on unchanged reranked evidence; no parent-study writes."""
import asyncio
from copy import deepcopy
from datetime import datetime,timezone
from html import escape
import json
from pathlib import Path
import re

import pandas as pd

from .answers import load_squad_metric,score_custom_answer,score_squad_answers
from .fresh_retrieval import RecordedOpenRouterClient,execution_settings,fingerprint
from .io import read_json,write_json,sha256_file
from .opportunity_study import load_opportunity_configuration,refresh_opportunity_results,generation_messages
from .provider import embedding_payload


def location(project):
    project=Path(project)
    plan=read_json(project/'configs/generation_extension.json')
    runtime=execution_settings(project)['runtime_directory']
    parent=load_opportunity_configuration(runtime/plan['parent_run']/'config.json')
    identity=fingerprint({'plan':plan,'parent_configuration':sha256_file(runtime/plan['parent_run']/'config.json')})[:20]
    folder=project/'experiments/v3-answer-quality/output/generation-extension'/identity
    return project,plan,parent,folder


def freeze(project):
    project,plan,parent,folder=location(project)
    path=folder/'frozen_subset.json'
    snapshot=refresh_opportunity_results(parent)
    population=pd.read_parquet(parent['questions_file'])
    chosen=[]
    for subgroup,rows in population.sort_values('query_id').groupby('generation_stratum',sort=True):
        ids=rows.sample(n=plan['questions_per_subgroup'],random_state=plan['seed']).query_id.tolist()
        if subgroup=='opportunity_gain' and plan['required_query_id'] not in ids:
            ids[-1]=plan['required_query_id']
        assert ids==plan['query_ids_by_subgroup'][subgroup],'Frozen selection policy no longer matches parent'
        for qid in ids:
            row=next(r for r in snapshot['inputs'] if r['query_id']==qid and r['arm']=='reranked')
            original=snapshot['current'][(qid,'reranked')]
            expected=generation_messages(row['question'],row['context'],parent['generation']['system_prompt'])
            assert original['request_body']['messages']==expected
            assert len(row['retrieved_contexts'])==len(row['context_passage_ids'])==5
            variants=list(plan['variants']);offset=len(chosen)%len(variants)
            chosen.append(dict(query_id=qid,question=row['question'],subgroup=subgroup,
                reference_answers=row['reference_answers'],context=row['context'],retrieved_contexts=row['retrieved_contexts'],
                context_passage_ids=row['context_passage_ids'],context_sha256=fingerprint(row['retrieved_contexts']),
                original_messages=expected,original_answer=row['answer'],
                original_provider=original.get('raw_response_json',{}).get('provider','unknown'),
                annotated_support_recall_at_5=row['annotated_support_recall_at_5'],
                generation_order=variants[offset:]+variants[:offset]))
    assert len(chosen)==len({r['query_id'] for r in chosen})==12
    contract=dict(plan=plan,generation=parent['generation'],questions=chosen,
                  parent_questions_sha256=sha256_file(parent['questions_file']))
    if path.exists():
        existing=read_json(path)
        assert existing['contract']==contract,'Frozen prompts, contexts, or selection changed'
        return existing
    frozen=dict(contract=contract,frozen_before_generation_utc=datetime.now(timezone.utc).isoformat(),
                contract_sha256=fingerprint(contract),study='exploratory; not part of original 79-pair aggregates')
    write_json(path,frozen)
    return frozen


def payload_for(question,variant,plan,generation):
    messages=deepcopy(question['original_messages'])
    if plan['variants'][variant]:
        messages[0]['content']+='\n\n'+plan['variants'][variant]
    payload=dict(model=generation['model'],messages=messages,temperature=generation['temperature'],
                 max_tokens=generation['max_tokens'],**generation['extras'],provider=plan['provider_routing'])
    assert messages[1]==question['original_messages'][1]
    return payload


async def generate(project,allow_model_calls=False):
    if allow_model_calls and __import__('os').environ.get('NOTEBOOK_TESTS_PASSED')!='1':
        raise RuntimeError('Offline tests must pass before extension model calls.')
    project,plan,parent,folder=location(project)
    frozen=freeze(project)
    generation=frozen['contract']['generation']
    semaphore=asyncio.Semaphore(plan['generation_concurrency'])
    pacing=asyncio.Lock()
    next_request=0.
    async def one(q,variant):
        path=folder/'answers'/f"{q['query_id']}-{variant}.json"
        payload=payload_for(q,variant,plan,generation)
        if path.exists():
            saved=read_json(path)
            assert saved['request_sha256']==fingerprint(payload)
            return saved
        if not allow_model_calls:
            raise RuntimeError('Exploratory first generation needs explicitly enabled calls.')
        async with semaphore:
            nonlocal next_request
            async with pacing:
                loop=asyncio.get_running_loop()
                await asyncio.sleep(max(0.,next_request-loop.time()))
                next_request=loop.time()+generation['request_min_interval_seconds']
            settings=dict(execution_settings(project)['retrieval'],max_attempts=plan['generation_attempts'])
            client=RecordedOpenRouterClient(settings,folder/'requests')
            def validate(raw):
                assert raw.get('provider')==plan['expected_serving_provider'],'Unexpected serving provider; stop comparison'
                assert isinstance(raw['choices'][0]['message']['content'],str)
            raw,trace=await asyncio.to_thread(client.request,'generation',generation['api_base']+'/chat/completions',
                payload,validate,f"extension-{q['query_id']}-{variant}")
            choice=raw['choices'][0];answer=choice['message']['content']
            old=score_custom_answer(answer,q['reference_answers'],parent['legacy_citation_pattern'])
            saved=dict(query_id=q['query_id'],question=q['question'],arm=variant,path=variant,
                subgroup=q['subgroup'],answer=answer,context=q['context'],retrieved_contexts=q['retrieved_contexts'],
                context_passage_ids=q['context_passage_ids'],context_sha256=q['context_sha256'],
                reference_answers=q['reference_answers'],finish_reason=choice.get('finish_reason'),
                truncated=choice.get('finish_reason')=='length',provider=raw['provider'],
                annotated_support_recall_at_5=q['annotated_support_recall_at_5'],
                existing_token_f1_0_1=old['token_f1'],existing_normalized_em_0_1=old['normalized_em'],
                answer_words=len(answer.split()),completion_tokens=raw.get('usage',{}).get('completion_tokens'),
                request_sha256=fingerprint(payload),frozen_contract_sha256=frozen['contract_sha256'])
            write_json(path,saved)
            print(f"Saved extension answer {q['query_id']} {variant}: {saved['answer_words']} words, provider {saved['provider']}",flush=True)
            return saved
    # Counterbalance which prompt is requested first; order was frozen before generation.
    first=frozen['contract']['questions'][0]
    await one(first,first['generation_order'][0])  # Verify actual provider before issuing the rest.
    tasks=[one(q,variant) for q in frozen['contract']['questions'] for variant in q['generation_order']]
    inputs=await asyncio.gather(*tasks,return_exceptions=True)
    errors=[r for r in inputs if isinstance(r,BaseException)]
    if errors:
        raise RuntimeError(f'{len(errors)} generation outcomes failed; successful first answers are saved. First error: {errors[0]}')
    assert len(inputs)==36
    write_json(folder/'generation_complete.json',dict(answers=36,provider=plan['expected_serving_provider'],
        frozen_contract_sha256=frozen['contract_sha256'],completed_utc=datetime.now(timezone.utc).isoformat()))
    return inputs


def _seed_question_vectors(project,questions,folder,parent):
    from .execution import _retrieval_directory
    settings=execution_settings(project,include_scope=True)
    source=_retrieval_directory(settings)
    required={q['question']:q['query_id'] for q in questions}
    found={}
    for path in (source/'requests').glob('*/attempt-*.json'):
        text=path.read_text()
        if not any(q in text for q in required if q not in found):continue
        trace=read_json(path)
        if trace['status']!='ok' or trace['stage']!='query_embedding':continue
        body=trace['request_body'];data={r['index']:r['embedding'] for r in trace['raw_response_json']['data']}
        for question,qid in required.items():
            expected=embedding_payload(parent['judge']['embedding_model'],[question],parent['judge']['embedding_dimensions'],
                instruction=parent['judge']['query_instruction'])
            if expected['input'][0] not in body['input']:continue
            assert body['model']==expected['model'] and body['dimensions']==expected['dimensions']
            vector=data[body['input'].index(expected['input'][0])]
            found[question]=(qid,vector,expected,str(path.relative_to(project)))
        if len(found)==len(required):break
    assert len(found)==12,'Missing saved query vectors; never re-embed the retrieval questions'
    run_id='generation-extension-'+folder.name
    records=[]
    for question,(qid,vector,body,origin) in found.items():
        records.append(dict(query_id=qid,query=question,query_embedding=vector))
        write_json(folder/'runs'/run_id/'requests'/f'{qid}-query_embedding.json',
            dict(request_body=body,raw_response_json={'data':[dict(index=0,embedding=vector)]},
                 provenance='Exact single-row projection from saved batched query embedding; no API call',source=origin))
    write_json(folder/'runs'/run_id/'results.json',records)
    return run_id


async def score(project,allow_model_calls=False):
    if allow_model_calls and __import__('os').environ.get('NOTEBOOK_TESTS_PASSED')!='1':
        raise RuntimeError('Offline tests must pass before extension judge calls.')
    project,plan,parent,folder=location(project)
    frozen=freeze(project)
    inputs=[read_json(folder/'answers'/f"{q['query_id']}-{variant}.json")
            for q in frozen['contract']['questions'] for variant in plan['variants']]
    standard=folder/'standard_scoring';standard.mkdir(parents=True,exist_ok=True)
    squad=standard/'squad_per_answer.csv'
    if not squad.exists():
        metric=load_squad_metric(parent['squad']['metric_source'],standard/'metric-cache')
        score_squad_answers(inputs,metric,parent['citation_pattern'],parent['legacy_citation_pattern']).to_csv(squad,index=False)
    expected=[standard/'ragas_results'/f"{r['query_id']}-{r['arm']}-{m}.json" for r in inputs for m in plan['judge_metrics']]
    complete=all(p.exists() and read_json(p).get('status') not in ('started','interrupted') for p in expected)
    if not complete:
        if not allow_model_calls:raise RuntimeError('Exploratory judge results are not complete.')
        from .judge import run_ragas_evaluation
        settings=deepcopy(parent['judge'])
        settings.update(embedding_question_policy='reuse_saved',judge_provider_routing=plan['provider_routing'])
        run_id=_seed_question_vectors(project,frozen['contract']['questions'],folder,parent)
        await run_ragas_evaluation(inputs,folder,run_id,standard,settings,allow_model_calls=True)
    frame=pd.read_csv(squad,dtype={'query_id':str})
    metadata={(r['query_id'],r['arm']):r for r in inputs}
    for i,row in frame.iterrows():
        saved=metadata[(row.query_id,row.arm)]
        for key in ('subgroup','answer_words','completion_tokens','provider','context_sha256'):frame.loc[i,key]=saved[key]
        for name,column in [('faithfulness','faithfulness_0_1'),('answer_relevancy','answer_relevancy_0_1')]:
            outcome=read_json(standard/'ragas_results'/f'{row.query_id}-{row.arm}-{name}.json')
            frame.loc[i,column]=outcome.get('score') if outcome['status']=='ok' else float('nan')
            frame.loc[i,name+'_status']=outcome['status']
    frame.to_csv(standard/'per_answer_scores.csv',index=False)
    return frame


def summary(project):
    _,plan,_,folder=location(project)
    scores=pd.read_csv(folder/'standard_scoring/per_answer_scores.csv',dtype={'query_id':str})
    columns=['squad_token_f1_pct','faithfulness_0_1','answer_relevancy_0_1','answer_words']
    common={c:set(scores.pivot(index='query_id',columns='arm',values=c).dropna().index) for c in columns}
    rows=[]
    for variant in plan['variants']:
        selected=scores[scores.arm.eq(variant)]
        row={'Version':variant,'Answers':len(selected)}
        for column,label in [('squad_token_f1_pct','Token F1 (%)'),('faithfulness_0_1','Faithfulness'),
                              ('answer_relevancy_0_1','Answer relevancy'),('answer_words','Answer words')]:
            row[label]=selected[selected.query_id.isin(common[column])][column].mean()
            row[label+' n']=len(common[column])
        rows.append(row)
    return pd.DataFrame(rows),scores


def examples(project):
    _,_,_,folder=location(project)
    _,scores=summary(project)
    original=scores[scores.arm.eq('original')].set_index('query_id')
    variants=scores[~scores.arm.eq('original')].copy()
    for metric in ['faithfulness_0_1','squad_token_f1_pct']:
        variants[metric+'_delta']=variants.apply(lambda r:r[metric]-original.loc[r.query_id,metric],axis=1)
    selected=['1053665']
    regressions=variants[variants.faithfulness_0_1_delta.lt(0)]
    regression_metric='faithfulness_0_1_delta'
    if regressions.empty:
        regressions=variants[variants.squad_token_f1_pct_delta.lt(0)]
        regression_metric='squad_token_f1_pct_delta'
    candidates=regressions[~regressions.query_id.isin(selected)].sort_values([regression_metric,'query_id'])
    if len(candidates):selected.append(candidates.iloc[0].query_id)
    for metric,ascending in [('squad_token_f1_pct_delta',False),('faithfulness_0_1_delta',True)]:
        if len(selected)==3:break
        candidates=variants[~variants.query_id.isin(selected)].dropna(subset=[metric]).sort_values([metric,'query_id'],ascending=[ascending,True])
        if len(candidates):selected.append(candidates.iloc[0].query_id)
    return selected


def prompt_display_data(frozen):
    """Full system prompts from the frozen run, with no table truncation."""
    contract=frozen['contract']
    original=contract['generation']['system_prompt']
    labels={'original':'Original prompt', 'evidence_check':'Evidence-check prompt',
            'concise':'Concise, question-focused prompt'}
    blocks=[]
    plain=[]
    for variant,instruction in contract['plan']['variants'].items():
        full=original+('\n\n'+instruction if instruction else '')
        note='Unchanged original system prompt.' if not instruction else 'Original system prompt plus the added instruction below.'
        blocks.append('<section><h4>'+escape(labels[variant])+'</h4><p>'+escape(note)+'</p>'
                      '<pre style="white-space:pre-wrap;overflow-wrap:anywhere">'+escape(full)+'</pre></section>')
        plain.append(labels[variant]+'\n'+note+'\n\n'+full)
    note='These are the complete system prompts sent to the generator. The question and exact five-passage user message are identical across versions.'
    return {'text/html':'<p>'+escape(note)+'</p>'+''.join(blocks),
            'text/plain':note+'\n\n'+'\n\n'.join(plain)}


def show_contract(project):
    from IPython.display import display,Markdown
    _,plan,_,folder=location(project);frozen=read_json(folder/'frozen_subset.json')
    display(Markdown('### Can generation use the evidence better?\n\n'
        '**Exploratory generation extension — retrieve evidence → select evidence → use evidence well.**\n\n'
        '**12 frozen questions × 3 prompts = 36 new answers**, using reranked contexts only. '
        'Three questions per populated retrieval subgroup; IMEI was deliberately included before generation. '
        'The original prompt is regenerated alongside the two variants on **Nebius**, with provider fallbacks disabled. '
        'Model, exact five passages and their order, question, 400-token output limit, temperature, top-p and seed stay fixed. '
        'Only one appended system-instruction block changes at a time. This is a small exploratory sample, not a subgroup-wide effect estimate. '
        '**The original 79-pair study and all its aggregates remain unchanged.**'))
    display(pd.DataFrame([dict(query_id=q['query_id'],subgroup=q['subgroup'],question=q['question']) for q in frozen['contract']['questions']]))
    display(prompt_display_data(frozen),raw=True)


def qualitative_review_display_data(review):
    """Full-width review blocks, preserving every word of each evidence check."""
    labels={'original':'Original prompt', 'evidence_check':'Evidence-check prompt',
            'concise':'Concise prompt'}
    html=[]
    plain=[]
    current=None
    for row in review['checks']:
        if row['query_id']!=current:
            current=row['query_id']
            html.append('<h4>Question '+escape(current)+'</h4>')
            plain.append('Question '+current)
        label=labels.get(row['version'],row['version'])
        html.append('<section><h5>'+escape(label)+'</h5><dl>')
        plain.append(label)
        for key,title in [('support','Evidence support'),('completeness','Completeness'),
                          ('abstention','Appropriate abstention')]:
            html.append('<dt><strong>'+title+'</strong></dt>'
                        '<dd style="white-space:pre-wrap;overflow-wrap:anywhere;margin-bottom:0.7em">'
                        +escape(row[key])+'</dd>')
            plain.append(title+': '+row[key])
        html.append('</dl></section>')
    return {'text/html':''.join(html),'text/plain':'\n\n'.join(plain)}


def show_results(project):
    from IPython.display import display,Markdown,HTML
    _,_,_,folder=location(project)
    table,scores=summary(project)
    display(Markdown('#### Prompt comparison on the same frozen evidence\n\n'
        'All three versions below are newly generated through the pinned provider. '
        'Metric means use the same available three-way question matches per metric; n is shown. '
        'Answer length is whitespace-delimited word count. Judge scores are estimates from the same Qwen model family, not human correctness.'))
    display(table[['Version','Answers','Token F1 (%)','Faithfulness','Answer relevancy','Answer words']].round(3))
    display(Markdown('Matched-question denominators: '+', '.join(
        f"{name}: {int(table.iloc[0][name+' n'])}/12"
        for name in ['Token F1 (%)','Faithfulness','Answer relevancy','Answer words'])+'.'))
    failed=scores[(scores.faithfulness_status!='ok')|(scores.answer_relevancy_status!='ok')]
    if len(failed):display(Markdown(f'**{len(failed)} answers have a failed or missing judge metric; affected metrics are not treated as successful.**'))
    for qid in examples(project):
        rows=scores[scores.query_id.eq(qid)].set_index('arm')
        display(Markdown('#### '+qid+' — '+rows.iloc[0].question))
        display(rows[['squad_token_f1_pct','faithfulness_0_1','answer_relevancy_0_1','answer_words']].round(3))
        answer_rows=[read_json(folder/'answers'/f'{qid}-{variant}.json') for variant in ['original','evidence_check','concise']]
        headers=''.join('<th>'+escape(r['arm'])+'</th>' for r in answer_rows)
        cells=''.join('<td style="vertical-align:top;white-space:pre-wrap">'+escape(r['answer'])+'</td>' for r in answer_rows)
        display(HTML('<table><thead><tr>'+headers+'</tr></thead><tbody><tr>'+cells+'</tr></tbody></table>'))
        display(HTML('<details><summary>Same five evidence passages for all three answers</summary><pre style="white-space:pre-wrap">'+escape(answer_rows[0]['context'])+'</pre></details>'))
    review=folder/'assistant_review.json'
    if review.exists():
        display(Markdown('#### Short qualitative check\n\nChecks support in the fixed passages, completeness of requested parts and appropriate abstention.'))
        display(qualitative_review_display_data(read_json(review)),raw=True)
        display(Markdown(read_json(review)['interpretation']))
    else:
        display(Markdown('Qualitative review is pending; automated metrics alone do not establish correctness or completeness.'))
