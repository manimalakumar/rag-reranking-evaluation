from .provider import embedding_payload, rerank_payload, rerank_scores, QUERY_INSTRUCTION
"""Fresh baseline execution and deterministic scoring, preserving published runs."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd

from .answers import load_squad_metric, remove_passage_citations, score_custom_answer, score_squad_answers
from .fresh_retrieval import RecordedOpenRouterClient, execution_settings, fingerprint, load_raw_dataset, passage_id
from .io import read_json, write_json, sha256_file
from .records import combine_answer_scores, load_saved_judge_scores
from .retrieval import score_retrieval
from .summaries import summarize_paired_scores


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_baseline_inputs(source, destination, settings, baseline_scope, rows=None):
    """Keep curated question membership; rebuild missing source rows from pinned data."""
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ['baseline_ids.json','golden_reviewed.json']:
        if not (source/name).is_file():
            raise FileNotFoundError(f'Keep the small scientific input {source/name} in Git; human-reviewed membership cannot be regenerated.')
        if source/name != destination/name:
            shutil.copy2(source/name, destination/name)
    cfg = settings['models_and_dataset']
    diagnostic_file = f"diagnostic_{settings['diagnostic_query_id']}.json"
    names = ['dataset_rows.json',diagnostic_file,'source_metadata.json','corpus.json']
    if all((source/name).is_file() for name in names):
        for name in names:
            if source/name != destination/name:
                shutil.copy2(source/name, destination/name)
    else:
        if rows is None:
            rows = load_raw_dataset({'dataset': {'name':cfg['dataset'],'configuration':cfg['dataset_config'],
                'split':cfg['split'],'revision':cfg['revision']}}, destination/'downloads')
        by_id = {str(row['query_id']): row for row in rows}
        ids = read_json(destination/'baseline_ids.json')
        missing = set(ids+[str(settings['diagnostic_query_id'])])-by_id.keys()
        if missing:
            raise ValueError(f'Pinned dataset is missing required baseline/diagnostic IDs: {sorted(missing)}')
        distractors = sorted((qid for qid in by_id if qid not in set(ids)), key=lambda qid:fingerprint([cfg['seed'],qid]))
        chosen = [by_id[qid] for qid in ids+distractors[:cfg['distractor_questions']]]
        write_json(destination/'dataset_rows.json', chosen)
        write_json(destination/diagnostic_file, by_id[str(settings['diagnostic_query_id'])])
        write_json(destination/'source_metadata.json', {'Source revision': cfg['revision'], 'source': 'Pinned raw dataset downloaded for a fresh run'})
        documents = {}
        for row in chosen:
            for text,url in zip(row['passages']['passage_text'],row['passages']['url']):
                text = text.strip()
                if not text:
                    continue
                pid = passage_id(text)
                documents.setdefault(pid, {'id':pid,'text':text,'urls':[]})
                if url not in documents[pid]['urls']:
                    documents[pid]['urls'].append(url)
        budget = cfg['passage_char_budget']
        write_json(destination/'corpus.json', [dict(doc, original_char_count=len(doc['text']),
            truncated=len(doc['text'])>budget, text=doc['text'][:budget]) for doc in sorted(documents.values(),key=lambda doc:doc['id'])])
    by_id = {str(row['query_id']): row for row in read_json(destination/'dataset_rows.json')}
    questions = []
    for record in read_json(destination/'golden_reviewed.json'):
        if str(record['decision']).strip().lower() != 'keep':
            continue
        qid = str(record['query id'])
        reviewed = []
        for block in re.split(r'(?:^|\n\n)Passage \d+\n',record['related passages'])[1:]:
            text,url = block.rsplit('\nSource: ',1)
            reviewed.append({'text':text.strip(),'url':url.strip()})
        question = {'query_id':qid,'query':record['query'],'answers':json.loads(record['answer']),
            'scope':baseline_scope,'review_notes':record['why'],'reviewed_passages':reviewed,
            'gold_ids':sorted({passage_id(p['text']) for p in reviewed})}
        row = by_id[qid]
        selected = {passage_id(t) for t,flag in zip(row['passages']['passage_text'],row['passages']['is_selected']) if flag}
        if question['query'].strip()!=row['query'].strip() or question['answers']!=row['answers'] or not set(question['gold_ids'])<=selected:
            raise ValueError(f'Curated question {qid} differs from the pinned dataset.')
        questions.append(question)
    row = read_json(destination/diagnostic_file)
    supports = [{'text':t.strip(),'url':u.strip()} for t,u,selected in zip(
        row['passages']['passage_text'],row['passages']['url'],row['passages']['is_selected']) if selected]
    questions.append({'query_id':str(row['query_id']),'query':row['query'],'answers':row['answers'],
        'gold_ids':sorted({passage_id(p['text']) for p in supports}), 'scope':'diagnostic',
        'review_notes':'Dataset support; separate from baseline aggregates.', 'reviewed_passages':supports})
    return read_json(destination/'corpus.json'), questions


def repair_baseline_artifacts(config, project, directory):
    """Recover local display files from saved answers without any hosted execution."""
    original=config['source_directory'];source_run=original/'runs'/config['run_id']
    directory=Path(directory)
    run=directory/'runs'/config['run_id']
    if not run.exists():
        shutil.copytree(source_run,run)
    shutil.copytree(original/'input',directory/'input',dirs_exist_ok=True)
    shutil.copy2(original/'batch_identity_v1.json',directory/'batch_identity_v1.json')
    settings=read_json(run/'config.json')
    docs,prepared=prepare_baseline_inputs(directory/'input',directory/'input',settings,config['baseline_scope'])
    if not(run/'questions.json').exists():
        write_json(run/'questions.json',prepared)
    questions=read_json(run/'questions.json');results=read_json(run/'results.json')
    if not(run/'candidates.json').exists():
        qmap={q['query_id']:q for q in questions}
        write_json(run/'candidates.json',[dict(qmap[r['query_id']],candidates=r['top30'],
            retrieval_seconds=r.get('retrieval_seconds')) for r in results if r.get('retrieval_status')=='ok'])
    manifest=read_json(run/'run_manifest.json');contract=manifest['contract']
    if not(run/'document_embedding_origin.json').exists():
        write_json(run/'document_embedding_origin.json',{'cache_id':contract.get('document_cache_id'),
            'matrix_sha256':contract['document_matrix_sha256'],
            'origin':'Recovered from saved manifest; original document request timings may be unavailable.'})
    if not(run/'document_embedding_attempts.csv').exists():
        pd.DataFrame(columns=['request_id','stage','attempt','status','elapsed_seconds','error']).to_csv(run/'document_embedding_attempts.csv',index=False)
    for result in results:
        for arm,value in result['arms'].items():
            path=run/'contexts_prompts'/f"{result['query_id']}-{arm}.json"
            if 'messages' in value and not path.exists():
                write_json(path,value)
    if not(run/'manual_review.csv').exists():
        pd.DataFrame(columns=['query_id','scope','arm','correct','rationale','reviewer']).to_csv(run/'manual_review.csv',index=False)
    if not(run/'haystack_crosscheck.csv').exists():
        os.environ.setdefault('HAYSTACK_TELEMETRY_ENABLED','False')
        scorer=_module(original/'scoring.py','baseline_repair_metrics')
        scorer.haystack_crosscheck(read_json(run/'candidates.json'),[r for r in results if r.get('retrieval_status')=='ok']).to_csv(run/'haystack_crosscheck.csv',index=False)
    if not(run/'audit.json').exists():
        write_json(run/'audit.json',{'origin':'Reconstructed local audit from saved answers; no model calls',
            'questions':len(questions),'failures':sum(v['status']!='ok' for r in results for v in r['arms'].values())})
    current=deepcopy(config)
    current.update(source_directory=directory,fresh_run=True,historical_reference_directory=original/'reference')
    return current


def build_baseline_run(original_config, project, directory, client=None, rows=None):
    """Create or resume the original two-path Haystack baseline in a separate directory."""
    os.environ.setdefault('HAYSTACK_TELEMETRY_ENABLED','False')
    try:
        from haystack import Document
        from haystack.document_stores.in_memory import InMemoryDocumentStore
        from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
    except ImportError as error:
        raise ImportError('A fresh baseline needs: python -m pip install -r requirements-runtime.txt') from error
    directory, project = Path(directory), Path(project)
    source = original_config['source_directory']
    settings = read_json(source/'config.json')
    cfg = settings['models_and_dataset']
    helpers = _module(source/'experiment_helpers.py','baseline_execution_helpers')
    scoring = _module(source/'scoring.py','baseline_original_metrics')
    docs, questions = prepare_baseline_inputs(source/'input',directory/'input',settings,original_config['baseline_scope'],rows)
    run_id = 'baseline-fresh-'+directory.name
    run = directory/'runs'/run_id
    run.mkdir(parents=True, exist_ok=True)
    client_settings = {**execution_settings(project)['retrieval'],
        'dimensions':cfg['embedding_dimensions'],'embedding_model':cfg['embedding_model'],
        'api_base':cfg['api_base'],'reranker_model':cfg['reranker_model'],'reranker_url':cfg['reranker_url'],
        'max_attempts':cfg['document_embedding_max_attempts'],'backoff_seconds':cfg['document_embedding_backoff_seconds']}
    client = client or RecordedOpenRouterClient(client_settings,directory/'requests')
    contract = {'settings':settings,'questions':questions,'documents':docs}
    contract_hash = fingerprint(contract)
    identity_path = directory/'batch_identity_v1.json'
    if identity_path.exists() and read_json(identity_path).get('contract_sha256')!=contract_hash:
        raise ValueError('Fresh baseline inputs changed; choose another runtime directory.')
    write_json(identity_path, {'run_id':run_id,'contract_sha256':contract_hash})
    example = directory/'document_embedding_example.json'
    scope_settings = execution_settings(project, include_scope=True)
    if original_config.get('eligible_baseline') and scope_settings.get('scope'):
        from .cached_scope import source_prefix
        source_cache, count, saved_matrix = source_prefix(scope_settings)
        source_corpus = pd.read_parquet(source_cache/'corpus.parquet').iloc[:count]
        positions = {key:i for i,key in enumerate(source_corpus.passage_id)}
        if any(doc['id'] not in positions for doc in docs):
            raise ValueError('An eligible baseline passage has no cached vector.')
        if any(doc['text'] != source_corpus.iloc[positions[doc['id']]].text for doc in docs):
            raise ValueError('Baseline text differs from the cached embedded passage.')
        matrix = np.asarray(saved_matrix[[positions[doc['id']] for doc in docs]]).copy()
        print(f'Reused {len(docs):,} cached passage embeddings for the eligible baseline; no passage embedding calls.',flush=True)
    else:
        vectors = []
        for start in range(0,len(docs),cfg['embedding_batch_size']):
            values,trace = client.embed([doc['text'] for doc in docs[start:start+cfg['embedding_batch_size']]], 'passage',cfg['embedding_dimensions'])
            vectors.append(values)
            if start==0:
                write_json(example,trace)
        matrix = np.concatenate(vectors)
    matrix_path = directory/'document_embeddings.npy'
    with matrix_path.open('wb') as handle:
        np.save(handle,matrix)
    doc_contract = fingerprint({'documents':docs,'model':cfg['embedding_model'],'dimensions':cfg['embedding_dimensions']})
    write_json(run/'document_embedding_origin.json', {'cache_id':doc_contract,'origin':'Fresh baseline document embeddings',
        'creator_run_id':run_id,'matrix_sha256':sha256_file(matrix_path),'matrix_shape':list(matrix.shape)})
    attempt_rows = [read_json(path) for path in client.directory.glob('*/attempt-*.json') if read_json(path)['stage']=='passage_embedding']
    pd.DataFrame(attempt_rows, columns=None if attempt_rows else ['request_id','stage','attempt','status','elapsed_seconds','error']).to_csv(run/'document_embedding_attempts.csv',index=False)
    write_json(run/'run_manifest.json', {'run_id':run_id,'contract_sha256':contract_hash,
        'contract':{'document_matrix_sha256':sha256_file(matrix_path),'scoring_source_sha256':sha256_file(source/'scoring.py'),
            'config':cfg,'system_prompt':settings['system_prompt'],'generation_extras':settings['generation_extras'],
            'input_sha256':{path.name:sha256_file(path) for path in (directory/'input').glob('*.json')}},
        'origin':'Fresh baseline; published saved run preserved'})
    write_json(run/'questions.json',questions)
    write_json(run/'config.json',settings)
    write_json(run/original_config['baseline_ids_file'],read_json(directory/'input/baseline_ids.json'))
    store = InMemoryDocumentStore(embedding_similarity_function='cosine')
    store.write_documents([Document(id=doc['id'],content=doc['text'],embedding=vector.tolist()) for doc,vector in zip(docs,matrix)])
    retriever = InMemoryEmbeddingRetriever(document_store=store,top_k=cfg['top_candidates'])
    lookup = {doc['id']:doc for doc in docs}
    def save_trace(trace,qid,stage,request_id,arm=None):
        value = {**trace,'query_id':qid,'stage':stage,'request_id':request_id}
        if arm:
            value['arm']=arm
        write_json(run/'requests'/(request_id+'.json'),value)
        return value
    def retain_failure(stage,url,payload,qid,request_id,arm=None,cache_key=None):
        contract={'stage':stage,'url':url,'payload':payload}
        if cache_key is not None:
            contract['cache_key']=cache_key
        identity=fingerprint(contract)
        paths=sorted((client.directory/identity).glob('attempt-*.json'))
        if paths:
            save_trace(read_json(paths[-1]),qid,stage,request_id,arm)
    def embed(q):
        try:
            values,trace=client.embed([q['query']],'query',cfg['embedding_dimensions'])
        except Exception:
            retain_failure('query_embedding',cfg['api_base']+'/embeddings',
                embedding_payload(cfg['embedding_model'], [q['query']], cfg['embedding_dimensions'],
                                  instruction=client.settings.get('query_instruction', QUERY_INSTRUCTION)),q['query_id'],q['query_id']+'-query_embedding')
            raise
        return values[0].tolist(), save_trace(trace,q['query_id'],'query_embedding',q['query_id']+'-query_embedding'), trace.get('cache_hit',False)
    def retrieve(vector):
        found=retriever.run(query_embedding=vector)['documents']
        if len(found)!=cfg['top_candidates']:
            raise ValueError('Baseline retrieval returned fewer candidates than configured.')
        return [dict(lookup[doc.id],embedding_score=float(doc.score),embedding_rank=rank) for rank,doc in enumerate(found,1)]
    def rerank(q,candidates):
        try:
            rankings,trace=client.rerank(q['query'],candidates)
        except Exception:
            retain_failure('reranking',cfg['reranker_url'],
                rerank_payload(cfg['reranker_model'], q['query'], candidates),q['query_id'],q['query_id']+'-rerank','reranked')
            raise
        ranked=[dict(candidates[row['index']],rerank_score=row['logit']) for row in rankings]
        return ranked, save_trace(trace,q['query_id'],'rerank_top30',q['query_id']+'-rerank','reranked'),trace.get('cache_hit',False)
    def messages(question,passages):
        from .opportunity_study import generation_messages
        context='\n\n'.join(f"[{doc['id']}]\n{doc['text']}" for doc in passages)
        return context,generation_messages(question,context,settings['system_prompt'])
    def generate(q,arm,msgs):
        payload={'model':cfg['generator_model'],'messages':msgs,'temperature':cfg['temperature'],
            'max_tokens':cfg['max_output_tokens'],**settings['generation_extras']}
        def validate(raw):
            if not isinstance(raw['choices'][0]['message']['content'],str):
                raise ValueError('The generator must return an answer string, including when empty.')
        # The baseline retains its one-attempt generation policy.
        previous=client.settings['max_attempts']
        client.settings['max_attempts']=1
        try:
            raw,trace=client.request('generation',cfg['generator_base']+'/chat/completions',payload,validate,
                                     cache_key=q['query_id']+'-'+arm)
        except Exception:
            retain_failure('generation',cfg['generator_base']+'/chat/completions',payload,q['query_id'],q['query_id']+'-'+arm+'-generation',arm,
                           cache_key=q['query_id']+'-'+arm)
            raise
        finally:
            client.settings['max_attempts']=previous
        return raw,save_trace(trace,q['query_id'],'generation',q['query_id']+'-'+arm+'-generation',arm),trace.get('cache_hit',False)
    def execute(q,position):
        return helpers.execute_question(q,position,embed_query=embed,retrieve_candidates=retrieve,
            rerank_candidates=rerank,build_messages=messages,generate_answer=generate,arms=tuple(original_config['arms']))
    results,candidates=helpers.run_question_batch(questions,run=run,contract_hash=contract_hash,
        complete=(run/'results.json').is_file(),execute_one=execute)
    review_columns=['query_id','scope','arm','question','reference_answers','answer','context','correct',
                    'evidence_used_correctly','same_answer_as_baseline','supporting_source_ids','rationale','reviewer']
    review_rows=[]
    for result in results:
        question=next(q for q in questions if q['query_id']==result['query_id'])
        for arm,value in result['arms'].items():
            if value['status']=='ok':
                review_rows.append({'query_id':result['query_id'],'scope':result['scope'],'arm':arm,'question':question['query'],
                    'reference_answers':json.dumps(question['answers']),'answer':value['answer'],'context':value['context']})
    if not(run/'manual_review.csv').exists():
        pd.DataFrame(review_rows,columns=review_columns).to_csv(run/'manual_review.csv',index=False)
    scoring.haystack_crosscheck(candidates,[r for r in results if r.get('retrieval_status')=='ok']).to_csv(run/'haystack_crosscheck.csv',index=False)
    write_json(run/'audit.json', {'batch_question_count':len(questions),
        'baseline_attempted_questions':len(questions)-len(original_config['diagnostic_ids']),
        'baseline_successful_pairs':sum(r['scope']==original_config['baseline_scope'] and all(v['status']=='ok' for v in r['arms'].values()) for r in results),
        'human_review':'pending','judge_evaluation':'explicit action; pending','original_run_preserved':True})
    config=deepcopy(original_config)
    config.update(source_directory=directory,run_id=run_id,fresh_run=True,
        historical_reference_directory=source/'reference',
        document_embedding_example=os.path.relpath(example,project))
    return config


def prepare_baseline_scores(config):
    """Score existing answers; use their exact complete pairs and keep RAGAS pending."""
    source,directory=Path(config['source_directory']),Path(config['score_directory'])
    run=source/'runs'/config['run_id']
    questions={q['query_id']:q for q in read_json(run/'questions.json')}
    results=read_json(run/'results.json')
    baseline_ids=read_json(run/config['baseline_ids_file'])
    complete=[qid for qid in baseline_ids if all(next(r for r in results if r['query_id']==qid)['arms'][arm]['status']=='ok' for arm in config['arms'])]
    baseline,diagnostic,failures=[],[],[]
    for result in results:
        qid=result['query_id'];q=questions[qid]
        for arm,value in result['arms'].items():
            if value['status']!='ok':
                failures.append({'query_id':qid,'arm':arm,'status':value['status'],'error':value.get('error'),'failed_stage':value.get('failed_stage')})
                continue
            if qid not in complete and qid not in config['diagnostic_ids']:
                continue
            blocks=[f"[{p['id']}]\n{p['text']}" for p in value['top_five']]
            old=score_custom_answer(value['answer'],q['answers'],config['legacy_citation_pattern'])
            row={'query_id':qid,'question':q['query'],'arm':arm,'path':config['arms'][arm],'answer':value['answer'],
                'squad_answer':remove_passage_citations(value['answer'],config['citation_pattern']),
                'reference_answers':q['answers'],'context':value['context'],'retrieved_contexts':blocks,
                'context_passage_ids':[p['id'] for p in value['top_five']],
                'finish_reason':value.get('finish_reason'),'truncated':value.get('finish_reason')=='length',
                'annotated_support_recall_at_5':score_retrieval([p['id'] for p in value['top_five']],q['gold_ids'])['recall'],
                'existing_token_f1_0_1':old['token_f1'],'existing_normalized_em_0_1':old['normalized_em'],
                'generation_request_sha256':sha256_file(run/'requests'/f'{qid}-{arm}-generation.json'),
                'context_record_sha256':sha256_file(run/'contexts_prompts'/f'{qid}-{arm}.json')}
            (baseline if qid in complete else diagnostic).append(row)
    directory.mkdir(parents=True,exist_ok=True)
    for name,data in [('evaluation_inputs.json',baseline),('diagnostic_inputs.json',diagnostic),('original_run_failures.json',failures),
        ('input_audit.json',{'complete_baseline_ids':complete,'complete_baseline_pairs':len(complete),'run_id':config['run_id']})]:
        if (directory/name).exists() and read_json(directory/name)!=data:
            raise ValueError('Baseline scoring inputs changed; choose a new score directory.')
        write_json(directory/name,data)
    if not baseline:
        raise ValueError('The fresh baseline has no complete answer pairs. Failures are retained in its run directory.')
    metric=load_squad_metric(config['squad']['metric_source'],config['squad']['cache_directory'])
    squad=score_squad_answers(baseline,metric,config['citation_pattern'],config['legacy_citation_pattern'])
    diagnostic_scores=score_squad_answers(diagnostic,metric,config['citation_pattern'],config['legacy_citation_pattern']) if diagnostic else squad.iloc[:0]
    squad.to_csv(directory/'squad_per_answer.csv',index=False)
    diagnostic_scores.to_csv(directory/'diagnostic_squad.csv',index=False)
    judge=pd.DataFrame(columns=['query_id','arm','metric','score','status','elapsed_seconds'])
    if (directory/'judge_manifest.json').exists():
        judge=load_saved_judge_scores(directory,baseline,config['judge']['metrics'],config['judge'])
    scores=combine_answer_scores(squad,judge,config['judge']['metrics'])
    scores.to_csv(directory/'per_question_scores.csv',index=False)
    settings=config['summary']
    summary,_=summarize_paired_scores(scores,config['arms'],settings['metrics'],settings['bootstrap_samples'],
        settings['bootstrap_seed'],settings['tie_tolerance'],settings['delta_label'],settings['confidence_quantiles'])
    summary.to_csv(directory/'paired_comparison.csv',index=False)
    squad[squad.token_f1_delta_0_1.abs().gt(settings['tie_tolerance']) | squad.normalized_em_delta_0_1.abs().gt(settings['tie_tolerance'])].to_csv(directory/'scorer_discrepancies.csv',index=False)
    write_json(directory/'scorer_edge_cases.json',[{'case':'empty answer and empty reference','SQuAD Token F1 (0–100)':0,'custom Token F1 (0–1)':1}])
    if not(directory/'evaluation_attempt_history.csv').exists():
        pd.DataFrame(columns=['query_id','arm','metric','status','error']).to_csv(directory/'evaluation_attempt_history.csv',index=False)
    if not(directory/'evaluation_audit.json').exists():
        write_json(directory/'evaluation_audit.json',{'judge_status':'saved outcomes replayed' if len(judge) else 'pending; explicit action required','new_judge_calls':0})
    from importlib.metadata import version
    write_json(directory/'squad_provenance.json',{'framework':'Hugging Face evaluate.load("squad")',
        'evaluate_version':version('evaluate'),'input_sha256':sha256_file(directory/'evaluation_inputs.json'),
        'score_sha256':sha256_file(directory/'squad_per_answer.csv'),'citation_pattern':config['citation_pattern'],
        'multiple_references':'all saved references','scale':'0–100 percent','model_calls':0})
