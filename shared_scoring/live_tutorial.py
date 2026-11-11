"""Stage-by-stage Live / Replay / Plan tutorials over explicitly frozen RAG inputs.

Importing or constructing a Plan never downloads data or calls a model. A new Live
run never reads an older run's inference outputs. Resume is explicitly same-run.
"""
from datetime import datetime,timezone
from copy import deepcopy
from pathlib import Path
import hashlib,json,math,os,time,uuid

import numpy as np
import pandas as pd

from .io import read_json,write_json,sha256_file
from .fresh_retrieval import execution_settings,fingerprint,normalized_vectors
from .provider import embedding_payload,rerank_payload,rerank_scores
from .live_client import TutorialTransport,request_scope
from .opportunity_study import generation_messages

STAGES={'demo':['retrieve','rerank','generate','score'],
        'baseline':['retrieve','rerank','generate','score'],
        'diagnosis':['retrieve','diagnose'],
        'targeted':['rerank','diagnose','generate','score'],
        'answers':['generate','score','extension'],
        'latency':['benchmark','score']}
NOTEBOOKS={'demo':'demos/00-rag-with-and-without-reranking/RAG-Retriever-Embedding-Reranker.ipynb',
 'baseline':'experiments/v1-baseline/Experiment-1-Baseline.ipynb',
 'diagnosis':'diagnosis-retrieval-opportunity/Diagnosis-Retrieval-Opportunity.ipynb',
 'targeted':'experiments/v2-targeted-reranking/Experiment-2-Targeted-Reranking.ipynb',
 'answers':'experiments/v3-answer-quality/Experiment-3-Answer-Quality.ipynb',
 'latency':'experiments/v4-latency-tradeoff/Experiment-4-Latency-Tradeoff.ipynb'}


def question_record(row):
    return dict(query_id=str(row['query_id']),question=str(row.get('question',row.get('query',''))),
        reference_answers=list(row.get('evaluation_references',row.get('reference_answers',row.get('answers',[])))),
        gold_ids=list(row.get('known_supporting_passage_ids',row.get('gold_ids',[]))),
        subgroup=str(row.get('generation_stratum',row.get('cohort',row.get('scope','')))),
        diagnostic=str(row['query_id'])=='1053665')


def as_dict(response):
    return response.model_dump(mode='json') if hasattr(response,'model_dump') else response


class TutorialRun:
    def __init__(self,project,study,mode='live',run_id=None,resume=False):
        self.project=Path(project).resolve();self.study=study;self.mode=str(mode).lower();self.resume=resume
        if study not in STAGES:raise ValueError('Unknown tutorial study')
        if self.mode not in {'live','replay','plan'}:raise ValueError('MODE must be live, replay or plan')
        if resume and (self.mode!='live' or not run_id):raise ValueError('Resume needs MODE=live and an explicit existing RUN_ID')
        if self.mode=='replay' and (not run_id or run_id=='reference'):raise ValueError('Replay needs an explicit tutorial RUN_ID; read notebook outputs directly without executing to inspect them.')
        self.declarations=read_json(self.project/'configs/tutorial_inputs.json')
        self.settings=execution_settings(self.project,include_scope=True)
        self.root=self.project/self.declarations['tutorial_run_directory']/study
        self.run_id=run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-')+uuid.uuid4().hex[:12]
        self.folder=self.root/self.run_id
        if self.mode=='live' and self.folder.exists() and not resume:raise FileExistsError('A Live run must have a new RUN_ID. Set RESUME=True explicitly to continue this same run.')
        if self.mode=='live' and resume and not (self.folder/'manifest.json').is_file():raise FileNotFoundError('Cannot resume a missing run')
        if self.mode=='replay' and self.run_id!='reference' and not (self.folder/'manifest.json').is_file():raise FileNotFoundError('Tutorial run is absent; choose an existing tutorial RUN_ID')
        self._client=None;self._assets=None;self._questions=None
        self.retrieval=self.settings['retrieval']
        self.generation=read_json(self.project/'experiments/v3-answer-quality/config.json')['generation']
        if study=='baseline':
            # The actual recorded baseline prompt/settings are declared upstream.
            self.generation=None
        self.plan=self.describe()

    def upstream(self,key):return self.project/self.declarations[key]

    def requirements(self):
        large=self.upstream('prepared_retrieval');base=self.upstream('baseline')
        required=[]
        if self.study in {'demo','diagnosis','latency'}:required += [large/'corpus.parquet',large/'index.faiss',large/'assets_manifest.json']
        if self.study=='diagnosis':required += [large/'queries.parquet']
        if self.study=='baseline':required += [base/'document_embeddings.npy',base/'input/corpus.json',base/'runs/baseline-fresh-5413aa92e9c1375fa06f/questions.json',base/'runs/baseline-fresh-5413aa92e9c1375fa06f/config.json']
        if self.study in {'baseline','targeted'}:required += [self.upstream('targeted')/'input/questions.parquet']
        if self.study=='targeted':required += [self.upstream('targeted')/'input/frozen_ranked_top30.parquet']
        if self.study in {'answers','latency'}:
            required += [self.upstream(self.study)/'input/questions.parquet']
            if self.study=='answers':required += [self.upstream('targeted')/'input/frozen_ranked_top30.parquet']
        return required

    def describe(self):
        sizes={'demo':1,'baseline':32,'diagnosis':15585,'targeted':700,'answers':79,'latency':8}
        return dict(mode=self.mode,study=self.study,run_id=self.run_id,stages=STAGES[self.study],
            questions=self.declarations.get('question_counts',sizes)[self.study],new_passage_embeddings=0,
            inference_policy='new responses for a new Live run; successful same-run reuse only with RESUME=True',
            upstream_policy='explicit prepared inputs; no automatic parent execution',
            missing_inputs=[str(p.relative_to(self.project)) for p in self.requirements() if not p.is_file()],
            judge_policy='RAGAS evaluates baseline, answer-quality, latency, and the two targeted worked examples; evaluation-only embeddings may be requested',
            targeted_example_ids=self.declarations.get('targeted_answer_examples',[]) if self.study=='targeted' else [],
            extension_policy='answers also runs the frozen 12-question three-prompt extension (36 extra answers)',
            caps=read_json(self.project/'configs/model_limits.json'))

    def preflight(self,create_client=True):
        self.plan=self.describe()
        if self.mode!='live':return self.plan
        if self.plan['missing_inputs']:
            raise FileNotFoundError('Missing frozen upstream inputs:\n'+'\n'.join(self.plan['missing_inputs'])+
                '\nSee setup.md: restore/prepare these assets explicitly. No document embedding or parent study was started.')
        if not os.environ.get('OPENROUTER_API_KEY'):raise RuntimeError('Set OPENROUTER_API_KEY in the kernel environment; see setup.md. No request sent.')
        if os.environ.get('TUTORIAL_TESTS_PASSED')!='1':
            import subprocess,sys
            checks=subprocess.run([sys.executable,str(self.project/'tools/verify_tutorial.py')],cwd=self.project,capture_output=True,text=True)
            if checks.returncode:raise RuntimeError('Offline checks failed before any model call:\n'+checks.stdout[-3000:]+checks.stderr[-3000:])
            os.environ['TUTORIAL_TESTS_PASSED']='1'
        self.questions(persist=False)  # Validate declared cohort before any request or output overwrite.
        if self.study in {'demo','baseline','diagnosis','latency'}:self._load_assets()
        from openai import OpenAI
        import httpx
        identity={'study':self.study,'upstream':{str(p.relative_to(self.project)):sha256_file(p) for p in self.requirements()},
                  'retrieval':self.retrieval,'generation':self.generation,
                  'tutorial_inputs':sha256_file(self.project/'configs/tutorial_inputs.json'),
                  'scoring':sha256_file(self.project/'experiments/v3-answer-quality/config.json'),
                  'extension':sha256_file(self.project/'configs/generation_extension.json') if self.study=='answers' else None}
        manifest=self.folder/'manifest.json'
        if manifest.exists():
            if read_json(manifest)['contract']!=identity:raise ValueError('Inputs/settings changed; use a new Live run rather than resume')
        else:write_json(manifest,dict(run_id=self.run_id,mode='live',created_utc=datetime.now(timezone.utc).isoformat(),
                                     contract=identity,state='ready',model_calls_started=False))
        write_json(self.folder/'questions.json',self._questions)
        os.environ.update(HAYSTACK_TELEMETRY_ENABLED='False',RAGAS_DO_NOT_TRACK='true')
        if create_client:
            self.bind_client(OpenAI(api_key=os.environ['OPENROUTER_API_KEY'],base_url=self.retrieval['api_base'],max_retries=0,
                timeout=self.retrieval['timeout_seconds'],http_client=httpx.Client(transport=self.transport())))
        return self.plan

    def transport(self):
        if self.mode!='live' or not (self.folder/'manifest.json').is_file():
            raise RuntimeError('A hosted SDK transport requires a preflighted Live run.')
        return TutorialTransport(self.folder/'requests',self.run_id,resume=self.resume,backoff=self.retrieval['backoff_seconds'])

    def bind_client(self,client):
        if self.mode!='live':raise RuntimeError('Plan/Replay never bind a hosted model client')
        self._client=client

    @property
    def client(self):
        if self.mode!='live':raise RuntimeError('Plan/Replay never construct a hosted inference client')
        if self._client is None:raise RuntimeError('Call tutorial.preflight() before executing a model stage')
        return self._client

    def _stage(self,name,action):
        if self.mode=='plan':print(f'PLAN: {name}; no execution, model calls or cached answers loaded.');return None
        path=self.folder/f'{name}.json'
        if path.exists():
            if self.mode=='replay' or self.resume:return read_json(path)
            raise RuntimeError(f'{name} already completed for this run. Start a new Live run or explicitly resume.')
        if self.mode=='replay':raise FileNotFoundError(f'Missing {name} for the requested run; replay never fills it with paid calls')
        result=action();write_json(path,result)
        print(f'{name}: saved for CURRENT LIVE RUN {self.run_id}')
        return result

    def questions(self,persist=True):
        if self._questions is not None:return self._questions
        if self.mode!='live':
            raise RuntimeError('Only Live stages resolve upstream questions; Replay reads the selected run and Plan does not load results.')
        if self.study=='baseline':
            base=self.upstream('baseline');run=base/'runs/baseline-fresh-5413aa92e9c1375fa06f'
            settings=read_json(run/'config.json');cfg=settings['models_and_dataset']
            self.generation=dict(model=cfg['generator_model'],api_base=cfg['generator_base'],temperature=cfg['temperature'],
                max_tokens=cfg['max_output_tokens'],system_prompt=settings['system_prompt'],extras=settings['generation_extras'])
            rows=[r for r in read_json(run/'questions.json') if r['scope']=='baseline_31']
            diagnostic=pd.read_parquet(self.upstream('targeted')/'input/questions.parquet',filters=[('query_id','==','1053665')]).iloc[0].to_dict()
            rows.append(diagnostic)
        elif self.study=='demo':rows=[self.declarations['demo_question']]
        elif self.study=='diagnosis':rows=pd.read_parquet(self.upstream('prepared_retrieval')/'queries.parquet').to_dict('records')
        else:
            frame=pd.read_parquet(self.upstream(self.study)/'input/questions.parquet')
            order_column='sample_order' if self.study=='answers' else 'latency_sample_order'
            if order_column in frame:frame=frame.sort_values(order_column)
            rows=frame.to_dict('records')
        self._questions=[question_record(r) for r in rows]
        if len(self._questions)!=self.plan['questions'] or len({q['query_id'] for q in self._questions})!=len(self._questions):
            raise ValueError('Question count or uniqueness differs from the declared frozen tutorial scope.')
        if self.settings.get('scope') and self.study!='demo':
            from .cached_scope import eligible_ids
            if not {q['query_id'] for q in self._questions}<=set(eligible_ids(self.settings)):
                raise ValueError('A question is outside the frozen fully covered eligible pool.')
        if self.study=='targeted':
            ids=self.declarations['targeted_answer_examples']
            if not ids or len(ids)!=len(set(ids)) or not set(ids)<={q['query_id'] for q in self._questions}:
                raise ValueError('Targeted answer examples must be distinct members of the frozen cohort.')
        if self.study=='answers':
            extension=read_json(self.project/'configs/generation_extension.json')
            ids=sum(extension['query_ids_by_subgroup'].values(),[])
            if len(ids)!=12 or len(set(ids))!=12 or not set(ids)<={q['query_id'] for q in self._questions}:
                raise ValueError('The declared 12-question extension is not a frozen subset of this answer-quality cohort.')
        if persist:write_json(self.folder/'questions.json',self._questions)
        return self._questions

    def _load_assets(self):
        if self._assets is not None:return self._assets
        if self.study=='baseline':
            from haystack import Document
            from haystack.document_stores.in_memory import InMemoryDocumentStore
            from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
            base=self.upstream('baseline');docs=read_json(base/'input/corpus.json');matrix=np.load(base/'document_embeddings.npy',mmap_mode='r')
            if matrix.shape!=(len(docs),self.retrieval['dimensions']):raise ValueError('Prepared baseline matrix does not match corpus/dimensions')
            store=InMemoryDocumentStore(embedding_similarity_function='cosine')
            store.write_documents([Document(id=d['id'],content=d['text'],embedding=v.tolist()) for d,v in zip(docs,matrix)])
            self._assets=('haystack',InMemoryEmbeddingRetriever(document_store=store,top_k=30),{d['id']:d for d in docs})
        else:
            import faiss
            base=self.upstream('prepared_retrieval');docs=pd.read_parquet(base/'corpus.parquet');index=faiss.read_index(str(base/'index.faiss'))
            manifest=read_json(base/'assets_manifest.json');index.nprobe=manifest['nprobe']
            if index.ntotal!=len(docs) or index.d!=self.retrieval['dimensions']:raise ValueError('Prepared index/corpus dimensions mismatch')
            self._assets=('faiss',index,docs)
        return self._assets

    def document_retriever(self):
        backend,retriever,_ = self._load_assets()
        if backend!='haystack':raise ValueError('This study does not use the baseline Haystack retriever.')
        return retriever

    def _search(self,vectors,questions,retrieve_api=None):
        backend,index,docs=self._load_assets();all_rows=[]
        if backend=='faiss':
            scores,positions=index.search(normalized_vectors(vectors,len(questions),self.retrieval['dimensions']),30)
            if (positions<0).any():raise ValueError('Index returned missing candidates')
        for i,q in enumerate(questions):
            if backend=='haystack':
                found=retrieve_api(vectors[i]) if retrieve_api else index.run(query_embedding=list(vectors[i]))['documents']
                selected=[dict(docs[d.id],passage_id=d.id,similarity_score=float(d.score)) for d in found]
            else:selected=[dict(docs.iloc[int(pos)],similarity_score=float(score)) for pos,score in zip(positions[i],scores[i])]
            for rank,d in enumerate(selected,1):
                all_rows.append(dict(query_id=q['query_id'],question=q['question'],passage_id=d['passage_id'],text=d['text'],
                    rank=rank,similarity_score=d['similarity_score'],is_known_support=d['passage_id'] in set(q['gold_ids'])))
        return all_rows

    def retrieve(self,embed_api,retrieve_api=None):
        def work():
            questions=self.questions();batch_size=self.retrieval['query_batch_size'];rows=[]
            matrix=np.lib.format.open_memmap(self.folder/'query_embeddings.npy',mode='w+',dtype='float32',shape=(len(questions),self.retrieval['dimensions']))
            for start in range(0,len(questions),batch_size):
                batch=questions[start:start+batch_size]
                with request_scope('query_embedding',str(start),max_attempts=3):raw=as_dict(embed_api([q['question'] for q in batch]))
                vectors=[r['embedding'] for r in sorted(raw['data'],key=lambda r:r['index'])]
                matrix[start:start+len(batch)]=normalized_vectors(vectors,len(batch),self.retrieval['dimensions'])
                rows.extend(self._search(vectors,batch,retrieve_api=retrieve_api))
            matrix.flush();pd.DataFrame(rows).to_parquet(self.folder/'retrieved_top30.parquet',index=False)
            return dict(questions=len(questions),candidate_rows=len(rows),query_embedding_matrix='query_embeddings.npy',
                        candidates='retrieved_top30.parquet',passage_embedding_calls=0,run_id=self.run_id)
        return self._stage('retrieve',work)

    def rerank(self,rerank_api):
        def work():
            questions=self.questions()
            source=self.upstream('targeted')/'input/frozen_ranked_top30.parquet' if self.study=='targeted' else self.folder/'retrieved_top30.parquet'
            candidates=pd.read_parquet(source).copy();candidates=candidates.drop(columns=['rerank_rank','rerank_score'],errors='ignore')
            rows=[]
            for q in questions:
                docs=candidates[candidates.query_id.astype(str).eq(q['query_id'])].sort_values('rank').to_dict('records')
                if len(docs)!=30:raise ValueError('Each frozen candidate set must contain exactly 30 passages')
                with request_scope('reranking',q['query_id'],max_attempts=3):raw=as_dict(rerank_api(q['question'],docs))
                for rank,item in enumerate(rerank_scores(raw,30),1):rows.append(dict(docs[item['index']],rerank_rank=rank,rerank_score=item['logit']))
            pd.DataFrame(rows).to_parquet(self.folder/'reranked_top30.parquet',index=False)
            return dict(questions=len(questions),candidate_rows=len(rows),run_id=self.run_id)
        return self._stage('rerank',work)

    def diagnose(self):
        def work():
            questions=self.questions();path=self.folder/('reranked_top30.parquet' if (self.folder/'reranked_top30.parquet').exists() else 'retrieved_top30.parquet')
            data=pd.read_parquet(path);rows=[]
            for q in questions:
                docs=data[data.query_id.astype(str).eq(q['query_id'])];gold=set(q['gold_ids'])
                if not gold:continue
                top5=set(docs.loc[docs['rank']<=5,'passage_id']);top30=set(docs.passage_id)
                row=dict(query_id=q['query_id'],question=q['question'],subgroup=q['subgroup'],
                    embedding_recall_at_5=len(gold&top5)/len(gold),embedding_recall_at_30=len(gold&top30)/len(gold),run_id=self.run_id)
                if 'rerank_rank' in docs:row['reranker_recall_at_5']=len(gold&set(docs.loc[docs.rerank_rank<=5,'passage_id']))/len(gold)
                rows.append(row)
            frame=pd.DataFrame(rows, columns=list(rows[0]) if rows else ['query_id','embedding_recall_at_5','embedding_recall_at_30']);frame.to_csv(self.folder/'retrieval_metrics.csv',index=False)
            return dict(annotated_questions=len(frame),opportunities=int((frame.embedding_recall_at_30>frame.embedding_recall_at_5+1e-12).sum()),run_id=self.run_id)
        return self._stage('diagnose',work)

    def _answer_specs(self,extension=False,arm=None):
        questions=self.questions()
        if self.study=='targeted':
            ids=self.declarations['targeted_answer_examples']
            questions=[q for q in questions if q['query_id'] in ids]
        if extension:
            plan=read_json(self.project/'configs/generation_extension.json');ids=sum(plan['query_ids_by_subgroup'].values(),[])
            questions=[q for q in questions if q['query_id'] in ids]
            if len(questions)!=12:raise ValueError('Frozen extension needs its 12 preselected questions')
        source=self.upstream('targeted')/'input/frozen_ranked_top30.parquet' if self.study=='answers' else self.folder/('retrieved_top30.parquet' if arm=='baseline' else 'reranked_top30.parquet')
        data=pd.read_parquet(source)
        for position,q in enumerate(questions):
            docs=data[data.query_id.astype(str).eq(q['query_id'])]
            arms={'original':'rerank_rank','evidence_check':'rerank_rank','concise':'rerank_rank'} if extension else {'baseline':'rank','reranked':'rerank_rank'}
            if self.study=='answers' and not extension and position%2:arms=dict(reversed(list(arms.items())))
            if arm is not None:arms={arm:arms[arm]}
            for arm_name,column in arms.items():
                selected=docs[docs[column]<=5].sort_values(column).to_dict('records')
                if len(selected)!=5:raise ValueError('Each arm requires its own five frozen passages')
                settings=deepcopy(self.generation);routing=None
                if extension:
                    instruction=plan['variants'][arm_name]
                    if instruction:settings['system_prompt']+='\n\n'+instruction
                    routing=plan['provider_routing']
                yield q,('extension-' if extension else '')+arm_name,selected,settings,routing

    def generate(self,generate_api,extension=False,arm=None):
        if arm is not None and (extension or self.study!='demo' or arm not in {'baseline','reranked'}):
            raise ValueError('Separate arm generation is available only for the warm-up.')
        stage='extension' if extension else ('generate-'+arm if arm else 'generate')
        requested_arm=arm
        def work():
            inputs=[]
            for q,arm,selected,settings,routing in self._answer_specs(extension,requested_arm):
                blocks=[f"[{d['passage_id']}]\n{d['text']}" for d in selected];context='\n\n'.join(blocks)
                with request_scope('generation',q['query_id'],arm,max_attempts=1):
                    raw=as_dict(generate_api(generation_messages(q['question'],context,settings['system_prompt']),settings,routing))
                if routing and raw.get('provider')!='Nebius':raise ValueError('Extension provider pin was not honored')
                choice=raw['choices'][0];answer=choice['message']['content']
                if not isinstance(answer,str):raise ValueError('An answer must be a string')
                inputs.append(dict(query_id=q['query_id'],question=q['question'],arm=arm,path=arm,answer=answer,
                    reference_answers=q['reference_answers'],context=context,retrieved_contexts=blocks,
                    context_passage_ids=[d['passage_id'] for d in selected],finish_reason=choice.get('finish_reason'),
                    truncated=choice.get('finish_reason')=='length',subgroup=q['subgroup'],diagnostic=q['diagnostic'],
                    annotated_support_recall_at_5=len(set(q['gold_ids'])&{d['passage_id'] for d in selected})/len(q['gold_ids']) if q['gold_ids'] else 0.,
                    provider=raw.get('provider'),answer_words=len(answer.split()),run_id=self.run_id))
                write_json(self.folder/'answers'/stage/f"{q['query_id']}-{arm}.json",inputs[-1])
            write_json(self.folder/f'{stage}_inputs.json',inputs)
            if requested_arm:
                combined=[]
                for path in ['generate-baseline_inputs.json','generate-reranked_inputs.json']:
                    if (self.folder/path).exists():combined.extend(read_json(self.folder/path))
                write_json(self.folder/'generate_inputs.json',combined)
                if len(combined)==2*len(self.questions()):
                    write_json(self.folder/'generate.json',dict(answers=len(combined),questions=len(self.questions()),run_id=self.run_id))
            return dict(answers=len(inputs),questions=len({r['query_id'] for r in inputs}),run_id=self.run_id)
        return self._stage(stage,work)

    async def score(self,extension=False):
        name='extension_scores' if extension else 'scores'
        if self.mode=='plan':return self._stage(name,lambda:None)
        path=self.folder/f'{name}.json'
        if path.exists():return self._stage(name,lambda:None)
        if self.mode=='replay':raise FileNotFoundError('Scores are absent for this exact run; replay does not judge')
        from .answers import load_squad_metric,score_squad_answers,score_custom_answer
        from .judge import run_ragas_evaluation
        from .opportunity_study import load_opportunity_configuration
        parent=load_opportunity_configuration(self.project/'experiments/v3-answer-quality/config.json')
        source=self.folder/('extension_inputs.json' if extension else 'generate_inputs.json')
        inputs=read_json(source);out=self.folder/name
        for row in inputs:
            old=score_custom_answer(row['answer'],row['reference_answers'],parent['legacy_citation_pattern'])
            row.update(existing_token_f1_0_1=old['token_f1'],existing_normalized_em_0_1=old['normalized_em'])
        metric=load_squad_metric(parent['squad']['metric_source'],out/'metric-cache')
        frame=score_squad_answers(inputs,metric,parent['citation_pattern'],parent['legacy_citation_pattern'])
        out.mkdir(parents=True,exist_ok=True)
        if self.study!='demo':
            settings=deepcopy(parent['judge']);settings['embedding_question_policy']='evaluate'
            if extension:settings['judge_provider_routing']=read_json(self.project/'configs/generation_extension.json')['provider_routing']
            # The mode authorizes the declared evaluation; legacy helpers retain
            # their independent hard guard and persistent request accounting.
            old=os.environ.get('ALLOW_MODEL_CALLS');os.environ['ALLOW_MODEL_CALLS']='1'
            try:judge=await run_ragas_evaluation(inputs,self.folder,self.run_id,out,settings,allow_model_calls=True)
            finally:
                if old is None:os.environ.pop('ALLOW_MODEL_CALLS',None)
                else:os.environ['ALLOW_MODEL_CALLS']=old
            for i,row in frame.iterrows():
                for metric_name in settings['metrics']:
                    selected=judge[(judge.query_id.eq(str(row.query_id)))&(judge.arm.eq(row.arm))&(judge.metric.eq(metric_name))]
                    value=selected.iloc[0] if len(selected) else None
                    frame.loc[i,metric_name]=value.score if value is not None and value.status=='ok' else float('nan')
                    frame.loc[i,metric_name+'_status']=value.status if value is not None else 'missing'
        meta={(r['query_id'],r['arm']):r for r in inputs}
        for i,row in frame.iterrows():
            record=meta[(str(row.query_id),row.arm)]
            for key in ['subgroup','diagnostic','answer_words','provider','run_id']:frame.loc[i,key]=record[key]
        frame.to_csv(out/'per_answer_scores.csv',index=False)
        write_json(path,dict(answers=len(frame),run_id=self.run_id,table=str((out/'per_answer_scores.csv').relative_to(self.folder))))
        return read_json(path)

    def benchmark(self,embed_api,rerank_api,generate_api):
        def work():
            if self.resume and (any((self.folder/'requests').glob('*/attempt-*.json')) or any((self.folder/'benchmark_paths').glob('*.json'))):
                raise RuntimeError('Partial benchmark cannot reuse cached HTTP responses as new timings. Choose a fresh Live RUN_ID; completed benchmarks can be replayed unchanged.')
            source=pd.read_parquet(self.upstream('latency')/'input/questions.parquet').sort_values('latency_sample_order')
            questions={q['query_id']:q for q in self.questions()};self._load_assets();results=[]
            for assignment in source.to_dict('records'):
                q=questions[str(assignment['query_id'])]
                for arm in list(assignment['arm_order']):
                    start=time.perf_counter();timings={}
                    stage_start=time.perf_counter()
                    with request_scope('query_embedding',q['query_id'],arm,max_attempts=2):raw=as_dict(embed_api([q['question']]))
                    timings['query_embedding_seconds']=time.perf_counter()-stage_start
                    stage_start=time.perf_counter();docs=self._search([raw['data'][0]['embedding']],[q]);timings['retrieval_seconds']=time.perf_counter()-stage_start
                    if arm=='reranked':
                        stage_start=time.perf_counter()
                        with request_scope('reranking',q['query_id'],arm,max_attempts=2):ranked=as_dict(rerank_api(q['question'],docs))
                        docs=[docs[r['index']] for r in rerank_scores(ranked,30)];timings['rerank_seconds']=time.perf_counter()-stage_start
                    stage_start=time.perf_counter()
                    selected=docs[:5];blocks=[f"[{d['passage_id']}]\n{d['text']}" for d in selected];context='\n\n'.join(blocks)
                    messages=generation_messages(q['question'],context,self.generation['system_prompt'])
                    timings['selection_prompt_seconds']=time.perf_counter()-stage_start
                    stage_start=time.perf_counter()
                    with request_scope('generation',q['query_id'],arm,max_attempts=2):raw=as_dict(generate_api(messages,self.generation,None))
                    timings['generation_seconds']=time.perf_counter()-stage_start
                    choice=raw['choices'][0]
                    attempts=[read_json(p) for p in (self.folder/'requests').glob('*/attempt-*.json')]
                    attempts=[r for r in attempts if r['query_id']==q['query_id'] and r['arm']==arm]
                    timings['successful_request_seconds']=sum(r['elapsed_seconds'] for r in attempts if r['status']=='ok')
                    timings['failed_attempt_seconds']=sum(r['elapsed_seconds'] for r in attempts if r['status']!='ok')
                    timings['backoff_seconds']=sum(r.get('backoff_seconds',0.) for r in attempts)
                    timings['successful_path_seconds']=timings['successful_request_seconds']+timings['retrieval_seconds']+timings['selection_prompt_seconds']
                    results.append(dict(query_id=q['query_id'],question=q['question'],arm=arm,path=arm,answer=choice['message']['content'],
                        reference_answers=q['reference_answers'],context=context,retrieved_contexts=blocks,context_passage_ids=[d['passage_id'] for d in selected],
                        finish_reason=choice.get('finish_reason'),truncated=choice.get('finish_reason')=='length',subgroup=q['subgroup'],diagnostic=q['diagnostic'],
                        annotated_support_recall_at_5=len(set(q['gold_ids'])&{d['passage_id'] for d in selected})/len(q['gold_ids']) if q['gold_ids'] else 0.,
                        answer_words=len(choice['message']['content'].split()),provider=raw.get('provider'),run_id=self.run_id,
                        total_end_to_end_seconds=time.perf_counter()-start,**timings))
                    write_json(self.folder/'benchmark_paths'/f"{q['query_id']}-{arm}.json",results[-1])
            write_json(self.folder/'generate_inputs.json',results)
            pd.DataFrame([{k:v for k,v in r.items() if k.endswith('_seconds') or k in ['query_id','arm','run_id']} for r in results]).to_csv(self.folder/'latency.csv',index=False)
            return dict(questions=len(source),paths=len(results),run_id=self.run_id,preparation_excluded_from_timing=True)
        return self._stage('benchmark',work)

    def finalize(self):
        """Check the completed current run without printing or making model calls."""
        if self.mode=='plan':return self.plan
        expected=[self.folder/(('scores' if stage=='score' else stage)+'.json') for stage in STAGES[self.study]]
        if self.study=='answers':expected.append(self.folder/'extension_scores.json')
        if not all(path.is_file() for path in expected):
            raise RuntimeError('Declared stages are not all complete; do not present a partial run as complete.')
        from .tutorial_views import TutorialViews
        TutorialViews(self).audit()
        failed_metrics=0
        for stage,input_file in [('scores','generate_inputs.json'),('extension_scores','extension_inputs.json')]:
            if not (self.folder/(stage+'.json')).is_file():continue
            inputs=read_json(self.folder/input_file)
            frame=pd.read_csv(self.folder/stage/'per_answer_scores.csv',dtype={'query_id':str})
            pairs={(str(row['query_id']),row['arm']) for row in inputs}
            if len(frame)!=len(inputs) or set(zip(frame.query_id,frame.arm))!=pairs:
                raise ValueError('Scoring rows do not match this run’s generated answers.')
            for metric in ['squad_token_f1_pct','squad_exact_match_pct']:
                if metric not in frame or pd.to_numeric(frame[metric],errors='coerce').isna().any():
                    raise ValueError('Deterministic answer scoring is incomplete.')
            if self.study!='demo':
                for metric in ['faithfulness','answer_relevancy']:
                    if metric not in frame or metric+'_status' not in frame:
                        failed_metrics+=len(frame)
                    else:
                        failed_metrics+=int((frame[metric+'_status'].ne('ok')|pd.to_numeric(frame[metric],errors='coerce').isna()).sum())
        state='complete_with_metric_failures' if failed_metrics else 'complete'
        if self.mode=='live':
            manifest=read_json(self.folder/'manifest.json')
            manifest.update(state=state,failed_or_missing_metrics=failed_metrics)
            write_json(self.folder/'manifest.json',manifest)
        return dict(run_id=self.run_id,state=state,failed_or_missing_metrics=failed_metrics)

    def show(self,summary=True):
        from IPython.display import display,Markdown
        if self.mode=='plan':display(self.plan);return
        if not summary:
            from .tutorial_views import TutorialViews
            TutorialViews(self).audit()
        display(Markdown(f'**{self.mode.upper()} run `{self.run_id}` — only this run’s answers, scores and timings.**'))
        failed_metrics=0
        if (self.folder/'retrieval_metrics.csv').exists():
            frame=pd.read_csv(self.folder/'retrieval_metrics.csv');
            if summary:display(frame.groupby('subgroup').mean(numeric_only=True))
            if summary:display(frame[frame.query_id.astype(str).eq('1053665')])
        for name in ['scores','extension_scores']:
            path=self.folder/name/'per_answer_scores.csv'
            if path.exists():
                frame=pd.read_csv(path,dtype={'query_id':str})
                diagnostic=frame.diagnostic.map(lambda value:str(value).lower()=='true')
                main=frame[~diagnostic] if self.study=='baseline' else frame
                columns=[c for c in ['squad_token_f1_pct','faithfulness','answer_relevancy','answer_words'] if c in main]
                if summary:display(main.groupby('arm')[columns].mean())
                if summary and 'subgroup' in main:display(main.groupby(['subgroup','arm'])[columns].mean())
                for metric in ['faithfulness','answer_relevancy']:
                    if metric+'_status' in frame:
                        if summary:display(frame.groupby(['arm',metric+'_status']).size().rename('metric_outcomes'))
                        failed_metrics+=int(frame[metric+'_status'].ne('ok').sum())
                if summary:display(frame[frame.query_id.eq('1053665')][['arm','answer',*columns]])
        if summary and (self.folder/'latency.csv').exists():display(pd.read_csv(self.folder/'latency.csv').groupby('arm').mean(numeric_only=True))
        for file in ['generate_inputs.json','extension_inputs.json']:
            if summary and (self.folder/file).exists():
                for row in read_json(self.folder/file):
                    if row['query_id']=='1053665':
                        display(Markdown('#### IMEI — '+row['arm']));display(Markdown(row['answer']))
                        display(Markdown('**Exact context for this answer:**\n\n'+row['context']))
        self.finalize()

    def close(self):
        if self._client:self._client.close()
