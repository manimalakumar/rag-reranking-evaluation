"""RAGAS judge evaluation of immutable saved answers. Resume recorded metric outcomes.
Only judge requests and synthetic-question evaluation embeddings may reach OpenRouter.
"""
import asyncio, contextvars, hashlib, importlib.metadata, inspect, json, math, os, time, uuid
from datetime import datetime,timezone
from pathlib import Path
from .budget import reserve_request, ModelBudgetExceeded
from .provider import embedding_payload, QUERY_INSTRUCTION, require_model_calls
from .io import read_json as read, write_json as write, sha256_file as sha
from .records import load_saved_judge_scores
from .logs import append_evaluation_event, save_evaluation_attempt
from .provenance import evaluation_fingerprint

CONTEXT=contextvars.ContextVar('evaluation_task',default={})


def now():
    """Return an explicit UTC evaluation timestamp."""
    return datetime.now(timezone.utc).isoformat()
def digest(value):
    """Fingerprint JSON request bodies and evaluation-cache inputs."""
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def create_ragas_metrics(client, embeddings, settings):
    """Construct the real RAGAS metrics only for explicit judge evaluation."""
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness, AnswerRelevancy
    llm = llm_factory(settings['judge_model'], provider='openai', adapter='instructor', client=client,
        temperature=settings['judge_temperature'], top_p=settings['judge_top_p'], max_tokens=settings['judge_max_tokens'],
        seed=settings['judge_seed'], max_retries=settings.get('structured_output_max_attempts', 1),
        system_prompt=settings['judge_system_prompt'])
    available = {'faithfulness': Faithfulness(llm=llm),
                 'answer_relevancy': AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=settings['relevancy_strictness'])}
    unknown = set(settings['metrics']) - available.keys()
    if unknown:
        raise ValueError(f'Unsupported configured RAGAS metrics: {sorted(unknown)}')
    return {name: available[name] for name in settings['metrics']}


def _make_components(source_directory, run_id, output_directory, settings):
    """Configure explicit OpenRouter clients, reuse, pacing, retries, and raw traces."""
    require_model_calls()
    SOURCE, OUT, CFG = Path(source_directory), Path(output_directory), settings
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ['RAGAS_DO_NOT_TRACK'] = 'true' 
    import httpx
    from openai import AsyncOpenAI
    from ragas.embeddings.base import BaseRagasEmbedding
    key=os.environ.get(CFG['credential_env'])
    if not key:raise RuntimeError(f"Missing environment variable {CFG['credential_env']}")
    class AuditedTransport(httpx.AsyncBaseTransport):
        def __init__(self):
            self.inner=httpx.AsyncHTTPTransport();self.request_lock=asyncio.Lock();self.next_request=0.0
            self.request_sequences={}
        async def throttle(self):
            async with self.request_lock:
                wait=max(0.,self.next_request-time.monotonic())
                if wait:await asyncio.sleep(wait)
                self.next_request=time.monotonic()+CFG['request_min_interval_seconds']
        async def handle_async_request(self,request):
            # Explicit endpoint allowlist prevents retrieval/reranker/document work.
            if str(request.url) not in [CFG['api_base']+'/chat/completions',CFG['api_base']+'/embeddings']:
                raise ValueError('Judge evaluation attempted an unapproved endpoint.')
            payload=json.loads((await request.aread()).decode())
            if request.url.path.endswith('/embeddings'):
                if payload['model']!=CFG['embedding_model'] or payload.get('dimensions')!=CFG['embedding_dimensions']:
                    raise ValueError('Evaluation embeddings must use the configured model and query mode.')
                if any(x in embedding_payload(CFG['embedding_model'], original_questions, CFG['embedding_dimensions'],
                        instruction=CFG.get('query_instruction', QUERY_INSTRUCTION))['input'] for x in payload['input']):
                    raise ValueError('Original query embeddings must be reused, never recomputed.')
            elif payload['model']!=CFG['judge_model']:
                raise ValueError('Unexpected judge model in request.')
            # RAGAS intentionally repeats some prompts (relevancy strictness).
            # Each invocation gets its own HTTP retry allowance; retries within
            # this transport loop keep the same persisted identity.
            sequence_key=digest({'task':CONTEXT.get(),'url':str(request.url),'payload':payload})
            sequence=self.request_sequences.get(sequence_key,0)
            self.request_sequences[sequence_key]=sequence+1
            group=uuid.uuid4().hex
            for attempt in range(1,CFG['request_max_attempts']+1):
                path=OUT/'judge_requests'/f'{group}-{attempt}.json';path.parent.mkdir(exist_ok=True)
                record={'request_group':group,'attempt':attempt,'evaluation_task':CONTEXT.get(),
                    'url':str(request.url),'request_body':payload,'request_body_sha256':digest(payload),
                    'started_utc':now(),'status':'started'}
                save_evaluation_attempt(path,record);start=time.perf_counter()
                try:
                    await self.throttle()
                    reserve_request(str(request.url), payload, [run_id, CONTEXT.get(), sequence], CFG['request_max_attempts'])
                    response=await self.inner.handle_async_request(request)
                    data=await response.aread()
                    record.update(http_status=response.status_code,raw_response_text=data.decode(errors='replace'),
                        elapsed_seconds=time.perf_counter()-start,ended_utc=now(),status='ok' if response.is_success else 'http_error')
                    try:record['response_json']=json.loads(data)
                    except ValueError:pass
                    save_evaluation_attempt(path,record)
                    if response.status_code in [408,429,500,502,503,504] and attempt<CFG['request_max_attempts']:
                        delay=CFG['rate_limit_backoff_seconds'] if response.status_code==429 else CFG['retry_backoff_seconds'][attempt-1]
                        record['retry_delay_seconds']=delay;save_evaluation_attempt(path,record)
                        await response.aclose();await asyncio.sleep(delay);continue
                    return response
                except (httpx.TimeoutException,httpx.TransportError) as e:
                    record.update(status='transport_error',error=f'{type(e).__name__}: {e}',elapsed_seconds=time.perf_counter()-start,ended_utc=now())
                    save_evaluation_attempt(path,record)
                    if attempt==CFG['request_max_attempts']:raise
                    await asyncio.sleep(CFG['retry_backoff_seconds'][attempt-1])
        async def aclose(self):await self.inner.aclose()
    # Seed original-question embeddings from verified generation run, never recompute them.
    run=SOURCE/'runs'/run_id
    original_questions={}
    seed_provenance=[]
    question_policy = CFG.get('embedding_question_policy', 'reuse_saved')
    if question_policy not in {'reuse_saved', 'evaluate'}:
        raise ValueError('embedding_question_policy must be reuse_saved or evaluate.')
    # A frozen retrieval study may lack saved query vectors. Its explicit judge
    # action can embed evaluation questions without changing retrieval rankings.
    seed_records = read(run/'results.json') if question_policy == 'reuse_saved' else []
    for record in seed_records:
        trace_path=run/'requests'/f"{record['query_id']}-query_embedding.json"
        trace=read(trace_path);body=trace['request_body']
        expected_body=embedding_payload(CFG['embedding_model'], [record['query']], CFG['embedding_dimensions'],
            instruction=CFG.get('query_instruction', QUERY_INSTRUCTION))
        if body != expected_body or trace['raw_response_json']['data'][0]['embedding']!=record['query_embedding']:
            raise ValueError(f"Saved query vector/settings changed: {record['query_id']}")
        original_questions[record['query']]=record['query_embedding']
        seed_provenance.append({'query_id':record['query_id'],'request':str(trace_path.relative_to(SOURCE)),'sha256':sha(trace_path)})
    write(OUT/'saved_query_embedding_provenance.json',seed_provenance)
    transport=AuditedTransport()
    client=AsyncOpenAI(api_key=key,base_url=CFG['api_base'],max_retries=0,
        timeout=CFG['request_timeout_seconds'],http_client=httpx.AsyncClient(transport=transport,timeout=CFG['request_timeout_seconds']))
    class QwenEvaluationEmbeddings(BaseRagasEmbedding):
        def __init__(self):
            super().__init__();self.memory=dict(original_questions);self.locks={}
        def embed_text(self,text,**kwargs):
            raise RuntimeError('Use the asynchronous evaluation API')
        async def aembed_text(self,text,**kwargs):
            if text in self.memory:
                append_evaluation_event(OUT/'embedding_reuse.jsonl', {'evaluation_task':CONTEXT.get(),'text':text,'origin':'original_saved_query' if text in original_questions else 'synthetic_question_cache','utc':now()})
                return self.memory[text]
            lock=self.locks.setdefault(text,asyncio.Lock())
            async with lock:
                if text in self.memory:return self.memory[text]
                cache_path=OUT/'evaluation_embeddings'/f'{digest(text)}.json';cache_path.parent.mkdir(exist_ok=True)
                if cache_path.exists():
                    data=read(cache_path)
                    if data['text']!=text or data['settings']!=embedding_settings:
                        raise ValueError('Evaluation embedding cache has different inputs/settings.')
                    self.memory[text]=data['vector'];return data['vector']
                response=await client.embeddings.create(**embedding_payload(CFG['embedding_model'], [text],
                    CFG['embedding_dimensions'], instruction=CFG.get('query_instruction', QUERY_INSTRUCTION)))
                vector=response.data[0].embedding
                if len(vector)!=CFG['embedding_dimensions'] or not all(math.isfinite(x) for x in vector):
                    raise ValueError('Evaluation embedding response has invalid dimensions or values.')
                write(cache_path,{'text':text,'vector':vector,'settings':embedding_settings,'created_utc':now()})
                self.memory[text]=vector;return vector
    embedding_settings={k:CFG[k] for k in ['embedding_model','embedding_dimensions','query_instruction','api_base']}
    embeddings=QwenEvaluationEmbeddings()
    metrics=create_ragas_metrics(client, embeddings, CFG)
    # Save the exact tested prompt source, including schemas and examples.
    prompts=OUT/'ragas_prompt_sources';prompts.mkdir(exist_ok=True)
    sources=[]
    for name,metric in metrics.items():
        for attr in ['statement_generator_prompt','nli_statement_prompt','prompt']:
            prompt=getattr(metric,attr,None)
            if prompt is None:continue
            source_path=Path(inspect.getfile(type(prompt)))
            target=prompts/f'{name}-{attr}.py';target.write_text(source_path.read_text())
            sources.append({'metric':name,'attribute':attr,'class':type(prompt).__name__,'file':str(target.relative_to(OUT)),'sha256':sha(target)})
    manifest_value={'started_utc':now(),'settings':CFG,'settings_sha256':digest(CFG),
        'packages':{n:importlib.metadata.version(n) for n in ['ragas','evaluate','openai','instructor','httpx','numpy','pydantic']},
        'prompt_sources':sources,'api':'ragas.metrics.collections + llm_factory(adapter="instructor")',
        'original_question_embeddings':('verified saved vectors reused; only synthetic judge questions embedded'
            if question_policy == 'reuse_saved' else 'questions and synthetic questions embedded for evaluation only; saved retrieval rankings unchanged'),
        'request_retries':'up to 3 transport attempts; Instructor up to 3 schema/JSON attempts. One explicit technical-failure metric retry; keep first successful outcome. Failed and interrupted outcomes archived.',
        'context_policy':'Faithfulness receives only the current arm’s five exact saved passage blocks, including IDs. AnswerRelevancy API accepts question and saved answer only.',
        'human_correctness':'Judge estimates, not human correctness. Judge is the saved generator model; possible shared-model bias.'}
    sessions=OUT/'judge_sessions';sessions.mkdir(exist_ok=True)
    write(sessions/f'{uuid.uuid4().hex}.json',manifest_value)
    if not (OUT/'judge_manifest.json').exists():
        write(OUT/'judge_manifest.json',manifest_value)
    return client,metrics

async def run_ragas_evaluation(inputs, source_directory, run_id, output_directory, settings,
                               allow_model_calls=False, retry_failures=False):
    """Opt in to judge calls; keep successes fixed and archive technical retries."""
    if not allow_model_calls:
        raise PermissionError('Judge calls are disabled. Use load_saved_judge_scores() for replay, or explicitly set allow_model_calls=True.')
    require_model_calls()
    SOURCE, OUT, CFG = Path(source_directory), Path(output_directory), settings
    rows = list(inputs)
    # Reject changed settings before replaying or extending an existing evaluation.
    manifest_path = OUT / 'judge_manifest.json'
    if manifest_path.exists():
        original = read(manifest_path)['settings']
        relevant = lambda values: {key: value for key, value in values.items()
                                  if key.startswith('judge_') or key.startswith('embedding_') or key in
                                  ['api_base', 'enable_thinking', 'relevancy_strictness', 'ragas_response_policy', 'query_instruction']}
        current = relevant(CFG)
        recorded = relevant(original)
        recorded.pop('judge_system_prompt', None)
        current.pop('judge_system_prompt', None)
        if recorded != current:
            raise ValueError('Judge/model settings changed. Use a separate explicit output directory; existing scores stay fixed.')
        for request_path in (OUT / 'judge_requests').glob('*.json'):
            request = read(request_path)
            if request['url'].endswith('/chat/completions'):
                if CFG['judge_system_prompt'] not in request['request_body']['messages'][0]['content']:
                    raise ValueError('Judge system prompt differs from saved requests. Use a separate output directory.')
                break
    existing = { (record['query_id'], record['arm'], record['metric']): record
                for record in [read(path) for path in (OUT / 'ragas_results').glob('*.json')] }
    if all((row['query_id'], row['arm'], metric) in existing and
           (existing[(row['query_id'], row['arm'], metric)]['status'] == 'ok' or
            (not retry_failures and existing[(row['query_id'], row['arm'], metric)]['status'] != 'started'))
           for row in rows for metric in CFG['metrics']):
        return load_saved_judge_scores(OUT, rows, CFG['metrics'], CFG)
    client, metrics = _make_components(SOURCE, run_id, OUT, CFG)
    semaphore=asyncio.Semaphore(CFG['concurrency'])
    async def evaluate(row,name,metric):
        path=OUT/'ragas_results'/f"{row['query_id']}-{row['arm']}-{name}.json";path.parent.mkdir(exist_ok=True)
        fingerprint=evaluation_fingerprint(row, name, CFG, importlib.metadata.version('ragas'))
        if path.exists():
            previous=read(path)
            if previous['fingerprint']!=fingerprint:
                raise ValueError(f'Changed evaluation input/settings: {path.name}. Use a separate output directory.')
            if previous['status']=='ok':return
            if not retry_failures:
                if previous['status']=='started':raise RuntimeError(f'Unfinished metric requires inspection: {path}')
                return
            archive=OUT/'ragas_attempt_history';archive.mkdir(exist_ok=True)
            prior=list(archive.glob(path.stem+'-*.json'))
            if len(prior)>=CFG['metric_max_attempts']-1:return
            if previous['status']=='started':
                previous.update(status='interrupted',score=None,error='Judge pass interrupted to apply rate-limit throttling',ended_utc=now())
            write(archive/f'{path.stem}-{len(prior)+1}.json',previous)
        async with semaphore:
            token=CONTEXT.set({'query_id':row['query_id'],'arm':row['arm'],'metric':name})
            result={'query_id':row['query_id'],'arm':row['arm'],'path':row['path'],'metric':name,'truncated':row['truncated'],
                'fingerprint':fingerprint,'judge_system_prompt':CFG['judge_system_prompt'],'status':'started','started_utc':now(),'context_passage_ids':row['context_passage_ids']}
            save_evaluation_attempt(path,result);start=time.perf_counter()
            try:
                kwargs={'user_input':row['question'],'response':row['answer']}
                if name=='faithfulness':kwargs['retrieved_contexts']=row['retrieved_contexts']
                score=await metric.ascore(**kwargs)
                value=float(score.value)
                if not math.isfinite(value):raise ValueError('RAGAS returned a non-finite score (e.g. no assessable statements)')
                result.update(status='ok',score=value)
            except asyncio.CancelledError:
                result.update(status='interrupted',score=None,error='Explicit judge evaluation interrupted before completion')
                raise
            except ModelBudgetExceeded:
                raise
            except Exception as e:
                result.update(status='error',score=None,error=f'{type(e).__name__}: {e}')
            finally:
                result.update(elapsed_seconds=time.perf_counter()-start,ended_utc=now());save_evaluation_attempt(path,result);CONTEXT.reset(token)
                print(f"{row['query_id']} {row['arm']} {name}: {result['status']} {result.get('score')} ({result['elapsed_seconds']:.1f}s)",flush=True)
    try:await asyncio.gather(*(evaluate(row,name,metric) for row in rows for name,metric in metrics.items()))
    finally:await client.close()
    return load_saved_judge_scores(OUT, rows, CFG['metrics'], CFG)
