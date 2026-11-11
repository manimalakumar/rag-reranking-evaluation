"""Rebuild portable tutorial inputs from a pinned public dataset, never old answers."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .fresh_retrieval import RecordedOpenRouterClient, execution_settings, normalized_vectors, passage_id
from .io import read_json, write_json, sha256_file


def read_recipe(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def corpus_digest(frame):
    digest=hashlib.sha256()
    for row in frame[['passage_id','text']].itertuples(index=False,name=None):
        digest.update((json.dumps(list(row),ensure_ascii=False,separators=(',',':'))+'\n').encode())
    return digest.hexdigest()


def collect_public_inputs(rows,count,char_budget,eligible=None):
    """Keep a dataset-order document prefix; verify every selected query's passages."""
    documents={};questions=[];found=set();wanted=set(eligible) if eligible is not None else None
    for position,row in enumerate(rows):
        associated=[];supports=[];passages=row['passages']
        for text,url,selected in zip(passages['passage_text'],passages['url'],passages['is_selected']):
            text=text.strip()
            if not text:continue
            pid=passage_id(text);associated.append(pid)
            if pid not in documents and len(documents)<count:
                documents[pid]=dict(corpus_index=len(documents),passage_id=pid,text=text[:char_budget],source_url=url,
                                    original_char_count=len(text),truncated=len(text)>char_budget)
            if selected and pid not in supports:supports.append(pid)
        qid=str(row['query_id'])
        if wanted is None or qid in wanted:
            formed=row.get('wellFormedAnswers',[])
            if isinstance(formed,str):formed=json.loads(formed)
            original=list(row.get('answers',[]));references=list(dict.fromkeys(original+list(formed)))
            questions.append(dict(query_id=qid,question=row['query'],query_row=position,
                reference_answers=references,known_supporting_passage_ids=supports,
                original_answers=original,well_formed_answers=list(formed),has_known_support_labels=bool(supports),
                all_associated_passage_ids=list(dict.fromkeys(associated))))
            found.add(qid)
        if len(documents)==count and (wanted is None or found==wanted):break
    if len(documents)!=count:raise ValueError('Dataset ended before the requested passage prefix.')
    included=set(documents)
    complete=[q for q in questions if set(q['all_associated_passage_ids'])<=included]
    if wanted is not None and {q['query_id'] for q in complete}!=wanted:
        raise ValueError('Pinned source did not reproduce every fully covered eligible question.')
    if len({q['query_id'] for q in complete})!=len(complete):raise ValueError('Duplicate question IDs.')
    return pd.DataFrame(documents.values()),pd.DataFrame(complete)


def _question_row(question):
    return dict(query_id=question['query_id'],question=question['question'],
        reference_answers=question['reference_answers'],evaluation_references=question['reference_answers'],
        known_supporting_passage_ids=question['gold_ids'],cohort=question['subgroup'])


def validate_reference_recipe(recipe,corpus,queries):
    available=set(corpus.passage_id);eligible=set(queries.query_id)
    memberships={}
    for key in ['baseline_questions','targeted_questions','answer_questions','latency_questions']:
        ids=[q['query_id'] for q in recipe[key]]
        if len(ids)!=len(set(ids)) or not set(ids)<=eligible:
            raise ValueError('Frozen question membership is invalid; no model call sent.')
        memberships[key]=set(ids)
    if not memberships['answer_questions']<=memberships['targeted_questions'] or not memberships['latency_questions']<=memberships['answer_questions']:
        raise ValueError('Frozen downstream questions are outside their parent cohort; no model call sent.')
    if not set(recipe['baseline_passage_ids'])<=available:
        raise ValueError('Frozen baseline passages are absent; no model call sent.')
    if set(recipe['targeted_candidates'])!=memberships['targeted_questions']:
        raise ValueError('Frozen candidate membership differs; no model call sent.')
    for value in recipe['targeted_candidates'].values():
        ids=value['passage_ids'];ranks=value['rerank_order']
        if len(ids)!=30 or len(set(ids))!=30 or not set(ids)<=available or sorted(ranks)!=list(range(1,31)):
            raise ValueError('Invalid frozen candidate IDs/rank permutation; no model call sent.')
    assignments=recipe['latency_assignments']
    if {a['query_id'] for a in assignments}!=memberships['latency_questions'] or len(assignments)!=len(memberships['latency_questions']):
        raise ValueError('Frozen latency membership differs; no model call sent.')
    if sorted(a['latency_sample_order'] for a in assignments)!=list(range(len(assignments))) or any(sorted(a['arm_order'])!=['baseline','reranked'] for a in assignments):
        raise ValueError('Invalid frozen latency order; no model call sent.')


def write_reference_tables(project,root,recipe,corpus,matrix,declarations):
    """Restore frozen membership, candidate order and contexts from the public recipe."""
    by_id=corpus.set_index('passage_id');positions={pid:i for i,pid in enumerate(corpus.passage_id)}
    baseline=root/'baseline';target=root/'targeted';answers=root/'answers';latency=root/'latency'
    for folder in [baseline,target,answers,latency]:(folder/'input').mkdir(parents=True,exist_ok=True)
    ids=recipe['baseline_passage_ids']
    if not set(ids)<=set(positions):raise ValueError('Baseline passages are outside the prepared prefix.')
    write_json(baseline/'input/corpus.json',[dict(id=pid,text=by_id.loc[pid,'text']) for pid in ids])
    np.save(baseline/'document_embeddings.npy',np.asarray(matrix[[positions[pid] for pid in ids]]))
    run=baseline/'runs/baseline-fresh-5413aa92e9c1375fa06f'
    write_json(run/'questions.json',[dict(_question_row(q),scope='baseline_31') for q in recipe['baseline_questions']])
    write_json(run/'config.json',read_json(project/'experiments/v1-baseline/config.json'))
    target_rows=[_question_row(q) for q in recipe['targeted_questions']]
    pd.DataFrame(target_rows).to_parquet(target/'input/questions.parquet',index=False)
    questions={q['query_id']:q for q in recipe['targeted_questions']};ranked=[]
    for qid,value in recipe['targeted_candidates'].items():
        ids=value['passage_ids'];order=value['rerank_order'];gold=set(questions[qid]['gold_ids'])
        if len(ids)!=30 or len(set(ids))!=30 or sorted(order)!=list(range(1,31)):
            raise ValueError('Recipe must contain 30 unique candidates and a complete rank permutation.')
        for rank,(pid,rerank) in enumerate(zip(ids,order),1):
            ranked.append(dict(query_id=qid,question=questions[qid]['question'],passage_id=pid,text=by_id.loc[pid,'text'],
                               rank=rank,rerank_rank=rerank,is_known_support=pid in gold))
    pd.DataFrame(ranked).to_parquet(target/'input/frozen_ranked_top30.parquet',index=False)
    pd.DataFrame([dict(_question_row(q),generation_stratum=q['subgroup'],sample_order=i)
                  for i,q in enumerate(recipe['answer_questions'])]).to_parquet(answers/'input/questions.parquet',index=False)
    assignments={a['query_id']:a for a in recipe['latency_assignments']}
    pd.DataFrame([dict(_question_row(q),generation_stratum=q['subgroup'],**{k:v for k,v in assignments[q['query_id']].items() if k!='query_id'})
                  for q in recipe['latency_questions']]).to_parquet(latency/'input/questions.parquet',index=False)
    for key,folder in [('baseline',baseline),('targeted',target),('answers',answers),('latency',latency)]:
        declarations[key]=str(folder.relative_to(project))
    declarations['question_counts'].update(baseline=len(recipe['baseline_questions'])+1,diagnosis=len(pd.read_parquet(root/'retrieval/queries.parquet')),
        targeted=len(target_rows),answers=len(recipe['answer_questions']),latency=len(recipe['latency_questions']))


def prepare_inputs(project,profile='small',passages=512,output=None,rows=None,client=None,recipe=None):
    project=Path(project).resolve();root=(project/(output or f'.cache/tutorial-inputs/{profile}')).resolve()
    if project/'.cache' not in root.parents:raise ValueError('Prepared assets must stay under the ignored project .cache directory.')
    if profile not in {'small','reference'}:raise ValueError('Unknown profile')
    settings=execution_settings(project);cfg=deepcopy(settings['retrieval']);cfg['passage_batch_size']=min(128,cfg['passage_batch_size'])
    source_recipe=project/'configs/reference_input_recipe.json.gz'
    if profile=='reference':recipe=recipe or read_recipe(source_recipe);passages=recipe['passage_count']
    if recipe and recipe.get('dataset',settings['dataset'])!=settings['dataset']:
        raise ValueError('Dataset revision/configuration differs from the frozen reference recipe.')
    if recipe and recipe.get('passage_char_budget',cfg['passage_char_budget'])!=cfg['passage_char_budget']:
        raise ValueError('Passage clipping differs from the frozen reference recipe.')
    if passages<30:raise ValueError('At least 30 passages are needed for the tutorial candidate budget.')
    contract=dict(profile=profile,passages=passages,dataset=settings['dataset'],retrieval=cfg,
                  recipe_sha256=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest() if recipe else None)
    manifest=root/'preparation.json'
    if manifest.exists():
        old=read_json(manifest)
        if old['contract']!=contract:raise ValueError('Preparation settings changed; choose a new --output under .cache. Existing inputs were preserved.')
        if old['state']=='complete':
            for name,digest in old['files'].items():
                if sha256_file(root/name)!=digest:raise ValueError('Prepared input changed: '+name)
            return root/'tutorial_inputs.json'
    eligible=None
    if profile=='reference':
        eligible_file=project/recipe['eligible_ids_file']
        if sha256_file(eligible_file)!=recipe['eligible_ids_sha256']:raise ValueError('Eligible ID list changed.')
        eligible=eligible_file.read_text().splitlines()
    if rows is None:
        from datasets import load_dataset
        data=settings['dataset']
        rows=load_dataset(data['name'],data['configuration'],split=data['split'],revision=data['revision'],streaming=True,
                          cache_dir=str(root/'downloads'))
    corpus,questions=collect_public_inputs(rows,passages,cfg['passage_char_budget'],eligible)
    if recipe and corpus_digest(corpus)!=recipe['corpus_id_text_sha256']:
        raise ValueError('Public dataset did not reproduce the recorded passage prefix; no model call sent.')
    if questions.empty:raise ValueError('No fully covered question in this corpus.')
    small_question=None
    if recipe:validate_reference_recipe(recipe,corpus,questions)
    else:
        labeled=questions[questions.has_known_support_labels & questions.reference_answers.map(bool)]
        if labeled.empty:raise ValueError('No labeled tutorial question; choose a larger passage budget. No model call sent.')
        small_question=labeled.iloc[0]
    write_json(manifest,dict(contract=contract,state='building'))
    retrieval=root/'retrieval';retrieval.mkdir(parents=True,exist_ok=True)
    corpus.to_parquet(retrieval/'corpus.parquet',index=False);questions.to_parquet(retrieval/'queries.parquet',index=False)
    vector_file=retrieval/'embeddings.npy'
    matrix=np.lib.format.open_memmap(vector_file,mode='r+' if vector_file.exists() else 'w+',dtype='float32',shape=(passages,cfg['dimensions']))
    if matrix.shape!=(passages,cfg['dimensions']):raise ValueError('Embedding matrix dimensions changed.')
    client=client or RecordedOpenRouterClient(cfg,root/'embedding-requests')
    for start in range(0,passages,cfg['passage_batch_size']):
        stop=min(passages,start+cfg['passage_batch_size']);checkpoint=root/'batches'/f'{start}.json'
        if checkpoint.exists():
            saved=read_json(checkpoint)
            if saved['stop']!=stop or saved['sha256']!=hashlib.sha256(matrix[start:stop].tobytes()).hexdigest():
                raise ValueError('Embedding checkpoint changed.')
            continue
        vectors,_=client.embed(corpus.text.iloc[start:stop].tolist(),'passage',cfg['dimensions'])
        matrix[start:stop]=normalized_vectors(vectors,stop-start,cfg['dimensions']);matrix.flush()
        write_json(checkpoint,dict(stop=stop,sha256=hashlib.sha256(matrix[start:stop].tobytes()).hexdigest()))
    import faiss
    faiss.omp_set_num_threads(cfg['threads'])
    # One inverted list provides exhaustive cosine search without k-means sampling.
    index=faiss.IndexIVFFlat(faiss.IndexFlatIP(cfg['dimensions']),cfg['dimensions'],1,faiss.METRIC_INNER_PRODUCT)
    index.train(np.asarray(matrix));index.add(np.asarray(matrix));index.nprobe=1
    faiss.write_index(index,str(retrieval/'index.faiss'))
    write_json(retrieval/'assets_manifest.json',dict(status='complete',corpus_rows=passages,dimensions=cfg['dimensions'],nprobe=1,
        corpus_id_text_sha256=corpus_digest(corpus),embedding_model=cfg['embedding_model'],dataset=settings['dataset'],index='IVFFlat, one list, exhaustive cosine'))
    declarations=deepcopy(read_json(project/'configs/tutorial_inputs.json'))
    declarations.update(profile=profile,prepared_retrieval=str(retrieval.relative_to(project)),
        allowed_studies=['demo'] if profile=='small' else ['demo','baseline','diagnosis','targeted','answers','latency'])
    if profile=='small':
        q=small_question
        declarations['demo_question']=dict(query_id=q.query_id,question=q.question,reference_answers=list(q.reference_answers),
            known_supporting_passage_ids=list(q.known_supporting_passage_ids),cohort='warmup')
    else:write_reference_tables(project,root,recipe,corpus,matrix,declarations)
    write_json(root/'tutorial_inputs.json',declarations)
    files={str(f.relative_to(root)):sha256_file(f) for f in root.rglob('*') if f.is_file() and
           f!=manifest and not any(part in {'embedding-requests','downloads','batches'} for part in f.relative_to(root).parts)}
    write_json(manifest,dict(contract=contract,state='complete',files=files,model_calls='document embedding only; no generated answers or judge scores'))
    return root/'tutorial_inputs.json'
