"""IMEI walkthrough: new small-corpus diagnostic, immutable larger-corpus results."""
from html import escape
from pathlib import Path
import json
import numpy as np
import pandas as pd

from .io import read_json, write_json, load_configuration, sha256_file
from .fresh_retrieval import RecordedOpenRouterClient, execution_settings, fingerprint
from .provider import embedding_payload
from .opportunity_study import generation_messages, load_opportunity_configuration, refresh_opportunity_results

QUERY_ID='1053665'
BASELINE='baseline/5413aa92e9c1375fa06f'
TARGETED='targeted-reranking/e27113d6ccebbf8a88ad'
ANSWERS='answer-quality/8d072cc0a7b92478a716'


def paths(project):
    project=Path(project)
    runtime=execution_settings(project,include_scope=True)['runtime_directory']
    baseline=runtime/BASELINE
    settings=read_json(baseline/'runs/baseline-fresh-5413aa92e9c1375fa06f/config.json')
    identity=fingerprint({'query_id':QUERY_ID,'settings':settings,
        'corpus_sha256':sha256_file(baseline/'input/corpus.json'),
        'matrix_sha256':read_json(baseline/'runs/baseline-fresh-5413aa92e9c1375fa06f/document_embedding_origin.json')['matrix_sha256']})[:20]
    return project,runtime,baseline,settings,runtime/'imei-diagnostic'/identity


def saved_query_vector(project, question):
    """Reuse the exact original query embedding from the completed 15,585-question run."""
    settings=execution_settings(project,include_scope=True)
    from .execution import _retrieval_directory
    directory=_retrieval_directory(settings)
    wanted=embedding_payload(settings['retrieval']['embedding_model'],[question],
        settings['retrieval']['dimensions'],instruction=settings['retrieval']['query_instruction'])['input'][0]
    for path in (directory/'requests').glob('*/attempt-*.json'):
        text=path.read_text()
        if question not in text:
            continue
        trace=read_json(path)
        body=trace['request_body']
        if trace['status']!='ok' or trace['stage']!='query_embedding' or wanted not in body['input']:
            continue
        assert body['model']==settings['retrieval']['embedding_model']
        assert body['dimensions']==2560
        index=body['input'].index(wanted)
        vector=next(r['embedding'] for r in trace['raw_response_json']['data'] if r['index']==index)
        assert len(vector)==2560
        return vector,str(path.relative_to(project)),index
    raise ValueError('Exact saved IMEI query embedding not found; no new embedding call is permitted.')


def retrieve_small(documents, matrix, vector, top_k=30):
    """Use the same Haystack cosine retriever as the baseline, entirely locally."""
    from haystack import Document
    from haystack.document_stores.in_memory import InMemoryDocumentStore
    from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
    store=InMemoryDocumentStore(embedding_similarity_function='cosine')
    store.write_documents([Document(id=d['id'],content=d['text'],embedding=v.tolist()) for d,v in zip(documents,matrix)])
    found=InMemoryEmbeddingRetriever(document_store=store,top_k=top_k).run(query_embedding=vector)['documents']
    lookup={d['id']:d for d in documents}
    return [dict(lookup[d.id],passage_id=d.id,embedding_rank=i,embedding_score=float(d.score)) for i,d in enumerate(found,1)]


def prepare_small(project, allow_model_calls=False):
    project,runtime,baseline,settings,folder=paths(project)
    result_path=folder/'result.json'
    if result_path.exists():
        return read_json(result_path)
    if not allow_model_calls:
        raise RuntimeError('IMEI small-corpus diagnostic is not saved; explicit diagnostic model calls are required.')
    if __import__('os').environ.get('NOTEBOOK_TESTS_PASSED')!='1':
        raise RuntimeError('Offline tests must pass before the new diagnostic calls.')
    cfg=settings['models_and_dataset']
    question=pd.read_parquet(runtime/TARGETED/'input/questions.parquet',filters=[('query_id','==',QUERY_ID)]).iloc[0]
    gold=list(question.known_supporting_passage_ids)
    documents=read_json(baseline/'input/corpus.json')
    assert len(documents)==10210
    assert not set(gold)&{d['id'] for d in documents},'Expected no annotated IMEI support in small corpus'
    client_settings=dict(execution_settings(project)['retrieval'])
    client=RecordedOpenRouterClient(client_settings,folder/'requests')
    retrieval_path=folder/'retrieval.json'
    if retrieval_path.exists():
        retrieval=read_json(retrieval_path)
    else:
        vector,origin,index=saved_query_vector(project,str(question.question))
        matrix=np.load(baseline/'document_embeddings.npy',mmap_mode='r')
        assert matrix.shape==(10210,2560)
        docs=retrieve_small(documents,matrix,vector)
        assert len(docs)==30
        ranks,trace=client.rerank(str(question.question),docs)
        reranked=[dict(docs[item['index']],rerank_score=item['logit']) for item in ranks]
        retrieval=dict(query_id=QUERY_ID,question=str(question.question),reference_answers=list(question.reference_answers),
            known_supporting_passage_ids=gold,corpus_passages=10210,supports_in_corpus=[],
            query_embedding_origin=origin,query_embedding_index=index,embedding_calls=0,
            top30=docs,embedding_top5=docs[:5],reranked_top5=reranked[:5],
            embedding_recall_at_5=0.,reranker_recall_at_5=0.,excluded_from_baseline_averages=True)
        write_json(retrieval_path,retrieval)
    answers_path=folder/'answers.json'
    inputs=read_json(answers_path) if answers_path.exists() else []
    from .answers import score_custom_answer,load_squad_metric,score_squad_answers
    scoring=load_configuration(project/'configs/experiment1.json')
    for arm,passages in [('baseline',retrieval['embedding_top5']),('reranked',retrieval['reranked_top5'])]:
        if any(r['arm']==arm for r in inputs):
            continue
        blocks=[f"[{d['id']}]\n{d['text']}" for d in passages]
        context='\n\n'.join(blocks)
        messages=generation_messages(retrieval['question'],context,settings['system_prompt'])
        payload=dict(model=cfg['generator_model'],messages=messages,temperature=cfg['temperature'],
            max_tokens=cfg['max_output_tokens'],**settings['generation_extras'])
        client.settings['max_attempts']=1
        raw,trace=client.request('generation',cfg['generator_base']+'/chat/completions',payload,
            cache_key=f'imei-small-{QUERY_ID}-{arm}')
        choice=raw['choices'][0];answer=choice['message']['content']
        old=score_custom_answer(answer,retrieval['reference_answers'],scoring['legacy_citation_pattern'])
        inputs.append(dict(query_id=QUERY_ID,question=retrieval['question'],arm=arm,path=scoring['arms'][arm],
            answer=answer,context=context,retrieved_contexts=blocks,context_passage_ids=[d['id'] for d in passages],
            reference_answers=retrieval['reference_answers'],finish_reason=choice.get('finish_reason'),
            truncated=choice.get('finish_reason')=='length',annotated_support_recall_at_5=0.,
            existing_token_f1_0_1=old['token_f1'],existing_normalized_em_0_1=old['normalized_em'],messages=messages))
        write_json(answers_path,inputs)
    metric=load_squad_metric(scoring['squad']['metric_source'],folder/'metric-cache')
    score_path=folder/'diagnostic_squad.csv'
    if not score_path.exists():
        score_squad_answers(inputs,metric,scoring['citation_pattern'],scoring['legacy_citation_pattern']).to_csv(score_path,index=False)
    result=dict(retrieval=retrieval,inputs=inputs,score_file=str(score_path.relative_to(project)),
        excluded_from_baseline_averages=True)
    write_json(result_path,result)
    return result


def _details(label,text):
    from IPython.display import display,HTML
    display(HTML('<details><summary>'+escape(label)+'</summary><pre style="white-space:pre-wrap">'+escape(text)+'</pre></details>'))


def show_small(project, result, part='all'):
    from IPython.display import display,Markdown
    r=result['retrieval']
    display(Markdown('### IMEI diagnostic — 10,210-passage baseline corpus\n\n'
        '**1053665: what is imei and where do i find it**\n\n'
        '**0 of 2 annotated supporting passages are in this corpus.** '
        'Both retrieval paths therefore have annotated-support Recall@5 = 0. '
        'This separate diagnostic is excluded from the **31-question baseline averages**.'))
    if part in ('all','retrieval'):
        for label,key in [('Embedding RAG','embedding_top5'),('Embedding + reranking RAG','reranked_top5')]:
            display(Markdown('#### '+label))
            for d in r[key]:_details(d['id'],d['text'])
    if part in ('all','answers'):
        for row in result['inputs']:
            display(Markdown('#### '+row['path']))
            _details('Exact saved context',row['context'])
            display(Markdown('**Answer:**'))
            from IPython.display import HTML
            display(HTML('<pre style="white-space:pre-wrap">'+escape(row['answer'])+'</pre>'))
            _details('Exact generator messages',json.dumps(row['messages'],indent=2))
        display(Markdown('**Saved diagnostic scores:** SQuAD percentages. No new judge scores were requested for this small-corpus diagnostic.'))
        display(pd.read_csv(Path(project)/result['score_file'])[['path','squad_token_f1_pct','squad_exact_match_pct','annotated_support_recall_at_5']])


def show_large_retrieval(project):
    from IPython.display import display,Markdown
    _,runtime,_,_,_=paths(project)
    q=pd.read_parquet(runtime/TARGETED/'input/questions.parquet',filters=[('query_id','==',QUERY_ID)]).iloc[0]
    docs=pd.read_parquet(runtime/TARGETED/'input/frozen_ranked_top30.parquet',filters=[('query_id','==',QUERY_ID)])
    assert q.embedding_recall_at_5==.5 and q.reranker_recall_at_5==1.
    supports=docs[docs.is_known_support].sort_values('rank')
    assert ((supports['rank']==6)&(supports.rerank_rank==3)).any()
    display(Markdown('### IMEI diagnostic — 152,576-passage corpus\n\n'
        '**1053665: what is imei and where do i find it**\n\n'
        'The 10,210-passage baseline corpus contains **0/2** annotated supports. '
        'The larger corpus contains **2/2**. One support moves **embedding rank 6 → reranker rank 3**; '
        '**Recall@5 improves 0.5 → 1.0**.\n\n'
        'This question already belongs to the frozen 700-question cohort. The walkthrough does not add another observation or change any averages.'))
    display(supports[['passage_id','rank','rerank_rank']])
    for label,column in [('Embedding top five','rank'),('Reranked top five','rerank_rank')]:
        display(Markdown('#### '+label))
        for row in docs[docs[column]<=5].sort_values(column).itertuples():
            _details(f'{row.passage_id} — annotated support: {row.is_known_support}',row.text)


def show_large_answers(project):
    from IPython.display import display,Markdown
    from .paired_tutorial import show_saved_answer_pair
    _,runtime,_,_,_=paths(project)
    config=load_opportunity_configuration(runtime/ANSWERS/'config.json')
    snapshot=refresh_opportunity_results(config)
    assert QUERY_ID in snapshot['complete_ids']
    display(Markdown('### IMEI diagnostic — saved larger-corpus answers\n\n'
        '**1053665: what is imei and where do i find it**\n\n'
        'The recovered annotated passage moved **rank 6 → 3** in retrieval over **152,576 passages**. '
        'Inspect the reranked answer against that passage below.\n\n'
        '**Token F1: 33.66% → 51.16%; judged faithfulness: 0.77 → 1.00.** '
        'These are the original saved answers and scores. IMEI is already one of the **79 sampled questions**; '
        'this diagnostic display adds no observation and changes no subgroup averages.'))
    show_saved_answer_pair({'config':config,**snapshot},QUERY_ID)
    scores=snapshot['standard']['scores']
    display(scores.loc[scores.query_id.eq(QUERY_ID),['path','squad_token_f1_pct','faithfulness_0_1','answer_relevancy_0_1']])
    docs=pd.read_parquet(runtime/TARGETED/'input/frozen_ranked_top30.parquet',filters=[('query_id','==',QUERY_ID)])
    recovered=docs[(docs['rank']==6)&(docs.rerank_rank==3)&docs.is_known_support]
    assert len(recovered)==1
    for row in recovered.itertuples():_details('Recovered annotated supporting passage — rank 6 → 3: '+row.passage_id,row.text)
