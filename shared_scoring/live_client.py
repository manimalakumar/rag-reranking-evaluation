"""Audited OpenAI SDK transport: fresh Live calls, explicit same-run resume only."""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib,json,time
from datetime import datetime,timezone
from pathlib import Path

import httpx
from .budget import reserve_request
from .io import read_json,write_json

_CONTEXT=ContextVar('tutorial_request',default=None)


@contextmanager
def request_scope(stage,query_id='',arm='',max_attempts=3):
    token=_CONTEXT.set(dict(stage=stage,query_id=str(query_id),arm=arm,max_attempts=max_attempts))
    try:yield
    finally:_CONTEXT.reset(token)


class TutorialTransport(httpx.BaseTransport):
    def __init__(self,directory,run_id,*,resume=False,retry_in_flight=False,retry_authorizations=None,inner=None,backoff=(2,5)):
        self.directory=Path(directory);self.run_id=run_id;self.resume=resume
        self.retry_in_flight=retry_in_flight
        self.retry_authorizations=retry_authorizations or {}
        self.inner=inner or httpx.HTTPTransport(retries=0);self.backoff=backoff

    @staticmethod
    def in_flight_delay(record):
        """Only OpenRouter's temporary in-flight 402, never insufficient funds."""
        if record.get('http_status')!=402:return None
        try:
            metadata=json.loads(record['raw_response_text'])['error']['metadata']
            if metadata.get('reason')!='in_flight_budget_exhausted':return None
            delay=float(metadata.get('headers',{}).get('Retry-After',120))
            return delay if 0<=delay<=300 else None
        except (KeyError,ValueError,TypeError):return None

    def wait_for_retry(self,record,path,delay):
        before=time.perf_counter();time.sleep(delay)
        record['backoff_seconds']=record.get('backoff_seconds',0.)+time.perf_counter()-before
        write_json(path,record)

    def handle_request(self,request):
        context=_CONTEXT.get()
        if context is None:raise RuntimeError('Every tutorial SDK call needs an explicit request_scope.')
        if request.url.path not in ['/api/v1/embeddings','/api/v1/rerank','/api/v1/chat/completions']:
            raise ValueError('Unexpected tutorial API endpoint.')
        payload=json.loads(request.read())
        if request.url.path.endswith('/embeddings') and context['stage']!='query_embedding':
            raise ValueError('Tutorials embed questions only; prepared passage vectors stay frozen.')
        contract=dict(run_id=self.run_id,context=context,url=str(request.url),payload=payload)
        identity=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
        # A separately approved exception keeps the original request identity and
        # ledger counter. It can add exactly one attempt to this request only.
        grant=self.retry_authorizations.get(identity,{}) if self.resume else {}
        attempt_cap=context['max_attempts']
        if grant:
            if grant.get('max_total_attempts')!=attempt_cap+1 or not grant.get('reason'):
                raise ValueError('A retry authorization must permit exactly one additional attempt with a reason.')
            attempt_cap=grant['max_total_attempts']
        folder=self.directory/identity
        previous=sorted(folder.glob('attempt-*.json'))
        first_attempt=1
        if previous:
            latest=read_json(previous[-1])
            if not self.resume:raise RuntimeError('This request already ran; start a new Live run or explicitly select RESUME=True.')
            if latest['status']=='ok':
                return httpx.Response(latest['http_status'],headers={'content-type':'application/json'},
                    content=latest['raw_response_text'].encode(),request=request)
            delay=self.in_flight_delay(latest)
            first_attempt=latest['attempt']+1
            if not grant and (not self.retry_in_flight or delay is None):
                raise RuntimeError('A failed/interrupted request is retained; inspect it before any explicit retry. Caps are not reset.')
            if first_attempt>attempt_cap:
                raise RuntimeError('This request exhausted its existing attempt cap. No request sent.')
            if delay is not None:
                elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(latest['started_utc'])).total_seconds()-latest.get('elapsed_seconds',0.)
                self.wait_for_retry(latest,previous[-1],max(0.,delay-elapsed))
        for attempt in range(first_attempt,attempt_cap+1):
            reserve_request(str(request.url),payload,[self.run_id,identity],attempt_cap)
            record=dict(**context,attempt=attempt,url=str(request.url),request_body=payload,status='started',run_id=self.run_id,
                        started_utc=datetime.now(timezone.utc).isoformat())
            if grant:record['explicit_retry_authorization']=grant
            path=folder/f'attempt-{attempt:02d}.json';write_json(path,record)
            started=time.perf_counter()
            try:
                response=self.inner.handle_request(request)
                body=response.read()
                manifest=self.directory.parent/'manifest.json'
                if manifest.exists():
                    state=read_json(manifest);state['model_calls_started']=True;state['state']='running';write_json(manifest,state)
                record.update(http_status=response.status_code,raw_response_text=body.decode(errors='replace'),
                    status='ok' if response.is_success else 'error',elapsed_seconds=time.perf_counter()-started)
                response.close();write_json(path,record)
                in_flight=self.in_flight_delay(record)
                if in_flight is not None and attempt<attempt_cap:
                    self.wait_for_retry(record,path,in_flight)
                    continue
                if record['http_status'] in [408,429,500,502,503,504] and attempt<attempt_cap:
                    before_backoff=time.perf_counter()
                    time.sleep(self.backoff[min(attempt-1,len(self.backoff)-1)])
                    record['backoff_seconds']=time.perf_counter()-before_backoff;write_json(path,record)
                    continue
                return httpx.Response(record['http_status'],headers={'content-type':'application/json'},content=body,request=request)
            except Exception as error:
                record.update(status='error',error=f'{type(error).__name__}: {error}',elapsed_seconds=time.perf_counter()-started)
                write_json(path,record)
                if attempt==attempt_cap:raise
                before_backoff=time.perf_counter()
                time.sleep(self.backoff[min(attempt-1,len(self.backoff)-1)])
                record['backoff_seconds']=time.perf_counter()-before_backoff;write_json(path,record)
        raise RuntimeError('Configured request attempts exhausted.')

    def close(self):self.inner.close()
