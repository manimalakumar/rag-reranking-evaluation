"""Persistence and execution helpers for the reranking tutorial.

Model payloads and the two RAG paths are defined in the notebook.
This module handles transport, recording, resumability, and display only.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import re
import time
from html import escape

import numpy as np
import pandas as pd


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def initialize_runtime(*, root, settings, mode):
    """Resolve one run and credentials; recorded replay never creates files."""
    import os
    import requests
    root = Path(root)
    if mode not in {'new_batch', 'resume_batch', 'recorded_batch'}:
        raise ValueError('Mode must be new_batch, resume_batch, or recorded_batch.')
    read_only = mode == 'recorded_batch'
    identity_path = root / 'batch_identity_v1.json'
    if mode == 'new_batch':
        identity = {'run_id': 'experiment1-v1-fresh-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}
        save_json(identity_path, identity)
    elif identity_path.exists():
        identity = json.loads(identity_path.read_text())
    else:
        raise FileNotFoundError('No saved batch identity. Restore the recorded run or explicitly select new_batch.')
    run = root / 'runs' / identity['run_id']
    if not read_only:
        run.mkdir(parents=True, exist_ok=True)
    complete = (run / 'results.json').exists()
    if read_only and not complete:
        raise FileNotFoundError('Recorded run is incomplete. Select resume_batch explicitly.')
    credential_env = settings.get('credential_env', 'OPENROUTER_API_KEY')
    if not read_only:
        credential_file = root / settings['credential_file']
        if credential_file.exists():
            for line in credential_file.read_text().splitlines():
                if line.strip() and not line.lstrip().startswith('#') and '=' in line:
                    name, value = line.split('=', 1)
                    os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))
        if not complete and not os.environ.get(credential_env):
            raise ValueError(f'Set {credential_env} in the kernel environment before hosted execution.')
    return dict(identity=identity, run=run, complete=complete, read_only=read_only,
                session=requests.Session(),
                headers={'Authorization': 'Bearer ' + os.environ.get(credential_env, '')})


def details(title, body):
    from IPython.display import HTML, display
    display(HTML(f'<details><summary>{escape(title)}</summary>{body}</details>'))


def show_table(title, frame):
    details(title, frame.to_html(index=False, escape=True))


def show_recorded_outputs(outputs_by_cell):
    """Show the complete saved outputs in an expandable archive."""
    sections = []
    for position, outputs in enumerate(outputs_by_cell.values(), 1):
        parts = []
        for output in outputs:
            if output['output_type'] == 'stream':
                text = output['text']
                parts.append('<pre>' + escape(''.join(text)) + '</pre>')
            else:
                data = output.get('data', {})
                if 'text/html' in data:
                    parts.append(''.join(data['text/html']))
                elif 'text/plain' in data:
                    parts.append('<pre>' + escape(''.join(data['text/plain'])) + '</pre>')
        sections.append(f'<details><summary>Saved output {position}</summary>' + ''.join(parts) + '</details>')
    details('Recorded v1 outputs', ''.join(sections))


def show_comparison(question, arms, names):
    from IPython.display import HTML, display
    columns = []
    for arm in ('baseline', 'reranked'):
        value = arms[arm]
        passages = value.get('top_five', [])
        frame = pd.DataFrame([
            {'Rank': i + 1, 'Passage ID': d['id'], 'Text excerpt': d['text'][:200]}
            for i, d in enumerate(passages)])
        text = value.get('answer') or value.get('error') or 'Unavailable'
        columns.append('<td style="vertical-align:top;width:50%;padding:12px">'
            f'<h4>{escape(names[arm])}</h4>{frame.to_html(index=False, escape=True)}'
            f'<p style="white-space:pre-wrap">{escape(text)}</p>'
            f'<p>Finish reason: {escape(str(value.get("finish_reason", "unavailable")))}</p></td>')
    display(HTML(f'<h3>{escape(question)}</h3><table><tr>{"".join(columns)}</tr></table>'))


def citation_ids(text):
    """Read citation IDs for evidence inspection; scoring remains in scoring.py."""
    groups = re.findall(r'\[\s*p_[a-f0-9]+(?:\s*,\s*p_[a-f0-9]+)*\s*\]', text, re.I)
    return {pid.lower() for group in groups for pid in re.findall(r'p_[a-f0-9]+', group, re.I)}


def evidence_trace(record, annotated_ids):
    """Join candidate ranks, context membership, and answer citations by ID.

    This describes selection and citation membership, not human correctness.
    """
    gold = set(annotated_ids)
    candidates = {d['id']: d for d in record.get('top30', [])}
    arms = record['arms']
    selected = {arm: {d['id'] for d in value.get('top_five', [])}
                for arm, value in arms.items()}
    cited = {arm: citation_ids(value.get('answer', '')) for arm, value in arms.items()}
    reranked = arms['reranked'].get('reranked_top30', [])
    ranks = {d['id']: i + 1 for i, d in enumerate(reranked)}
    relevant_ids = selected['baseline'] | selected['reranked'] | gold
    ordered = sorted(relevant_ids, key=lambda pid: (candidates.get(pid, {}).get('embedding_rank', 999), pid))
    return pd.DataFrame([{
        'Passage ID': pid,
        'Embedding rank / 30': candidates.get(pid, {}).get('embedding_rank'),
        'Reranked rank / 30': ranks.get(pid),
        'Embedding context': pid in selected['baseline'],
        'Reranked context': pid in selected['reranked'],
        'Annotated support': pid in gold,
        'Cited: embedding answer': pid in cited['baseline'],
        'Cited: reranked answer': pid in cited['reranked'],
        'Availability': 'Retrieved candidate' if pid in candidates else 'Not in retrieved 30',
    } for pid in ordered])


def show_evidence_focus(record, *, focus_id, source_phrase, answer_phrase, names):
    """Highlight an explicitly selected phrase after checking it exists.

    A matching phrase is an inspectable clue, not an automated support verdict.
    Only question/context/answer text from the recorded run is shown here.
    """
    from IPython.display import HTML, display
    passages = {d['id']: d for d in record.get('top30', [])}
    if focus_id not in passages:
        display(HTML('<p>The selected evidence-focus passage is not in this run\'s candidates.</p>'))
        return
    text = passages[focus_id]['text']
    start = text.casefold().find(source_phrase.casefold())
    if start < 0:
        display(HTML('<p>The selected phrase is absent from this run; inspect the full passage below.</p>'))
        return
    end = start + len(source_phrase)
    excerpt_start, excerpt_end = max(0, start - 50), min(len(text), end + 180)
    excerpt = escape(text[excerpt_start:start]) + '<mark>' + escape(text[start:end]) + '</mark>' + escape(text[end:excerpt_end])
    body = '<h4>Inspect the wording behind the answer</h4>'
    body += f'<p>Retrieved passage <code>{escape(focus_id)}</code>:</p><blockquote>{excerpt}</blockquote>'
    for arm in ('baseline', 'reranked'):
        answer = record['arms'][arm].get('answer', '')
        index = answer.casefold().find(answer_phrase.casefold())
        if index >= 0:
            marked = escape(answer[:index]) + '<mark>' + escape(answer[index:index+len(answer_phrase)]) + '</mark>' + escape(answer[index+len(answer_phrase):])
        else:
            marked = escape(answer) or 'Unavailable'
        body += f'<p><b>{escape(names[arm])}</b>: {marked}</p>'
    body += '<p>The highlight identifies matching wording. It does not establish that the passage answers the intended question or that the answer is correct.</p>'
    display(HTML(body))


def recorded_request(*, call_model, url, payload, run, request_id,
                     stage, query_id, arm=None, validate=None):
    """Attempt once; reuse an existing trace without replacing its response.

Transport success and usable model output are validated separately. A saved
request marked 'started' is never silently resubmitted.
"""
    trace_path = Path(run) / 'requests' / (request_id + '.json')
    if trace_path.exists():
        trace = json.loads(trace_path.read_text())
        if trace['request_body'] != payload or trace['url'] != url:
            raise ValueError('Recorded request does not match current inputs; start a new batch.')
        reused = True
    else:
        reused = False
        trace = dict(request_id=request_id, query_id=query_id, stage=stage,
                     url=url, request_body=payload, status='started',
                     started_utc=datetime.now(timezone.utc).isoformat())
        trace['request_body_sha256'] = digest(payload)
        if stage == 'generation':
            trace['prompt_sha256'] = digest(payload['messages'])
        if arm is not None:
            trace['arm'] = arm
        save_json(trace_path, trace)
        started = time.perf_counter()
        try:
            response = call_model(url, payload)
            trace.update(http_status=response.status_code, raw_response_text=response.text)
            try:
                trace['raw_response_json'] = response.json()
            except ValueError:
                pass
            response.raise_for_status()
            if 'raw_response_json' not in trace:
                raise ValueError('Model response is not JSON.')
            if validate is not None:
                validate(trace['raw_response_json'])
            trace['status'] = 'ok'
        except Exception as exc:
            trace.update(status='error', error_type=type(exc).__name__, error=str(exc))
        finally:
            trace.update(elapsed_seconds=time.perf_counter() - started,
                         ended_utc=datetime.now(timezone.utc).isoformat())
            save_json(trace_path, trace)
    if trace['status'] != 'ok':
        raise RuntimeError(trace.get('error', 'Recorded request is unfinished; no retry performed.'))
    output = trace['raw_response_json']
    if validate is not None:
        validate(output)
    return output, trace, reused


def execute_question(q, position, *, embed_query, retrieve_candidates,
                     rerank_candidates, build_messages, generate_answer, arms):
    """Record both independent answer branches, including failures."""
    qid = q['query_id']
    record = dict(query_id=qid, query=q['query'], scope=q['scope'], arms={},
                  request_ids=[], started_utc=datetime.now(timezone.utc).isoformat())
    candidate_q = None
    stage = 'query_embedding'
    started = time.perf_counter()
    try:
        record['request_ids'].append(qid + '-query_embedding')
        vector, trace, embedding_reused = embed_query(q)
        record['query_embedding'] = vector
        record['query_embedding_seconds'] = trace['elapsed_seconds']
        stage = 'top30_retrieval'
        search_started = time.perf_counter()
        candidates = retrieve_candidates(vector)
        record['local_retrieval_seconds'] = time.perf_counter() - search_started
        retrieval_seconds = None if embedding_reused else time.perf_counter() - started
        candidate_q = dict(q, candidates=candidates, retrieval_seconds=retrieval_seconds)
        record.update(top30=candidates, retrieval_seconds=retrieval_seconds, retrieval_status='ok')
    except Exception as exc:
        record.update(retrieval_status='error', retrieval_error=str(exc), failed_stage=stage)
        record['arms'] = {arm: dict(status='blocked', failed_stage=stage,
            error=str(exc), generation_attempted=False) for arm in arms}
    if candidate_q is not None:
        order = list(arms) if position % 2 == 0 else list(reversed(arms))
        record['generation_order'] = order
        for arm in order:
            branch_started = time.perf_counter()
            value = dict(status='started', generation_attempted=False)
            record['arms'][arm] = value
            stage = 'select_embedding_top5' if arm == 'baseline' else 'rerank_top30'
            try:
                rank_reused = False
                if arm == 'baseline':
                    ranked, rank_seconds = candidates, 0.0
                else:
                    record['request_ids'].append(qid + '-rerank')
                    ranked, rank_trace, rank_reused = rerank_candidates(q, candidates)
                    rank_seconds = rank_trace['elapsed_seconds']
                    value['reranked_top30'] = ranked
                top = ranked[:5]
                context, messages = build_messages(q['query'], top)
                value.update(top_five=top, context=context, messages=messages,
                             prompt_sha256=digest(messages),
                             ranking_order=[d['id'] for d in ranked], rank_seconds=rank_seconds)
                stage = 'generation'
                value['generation_attempted'] = True
                record['request_ids'].append(qid + '-' + arm + '-generation')
                output, generation_trace, generation_reused = generate_answer(q, arm, messages)
                choice = output['choices'][0]
                comparable = not (embedding_reused or rank_reused or generation_reused)
                branch_seconds = time.perf_counter() - branch_started if comparable else None
                value.update(status='ok', answer=choice['message']['content'],
                    finish_reason=choice.get('finish_reason'),
                    output_truncated=choice.get('finish_reason') == 'length',
                    usage=output.get('usage'), served_model=output.get('model'),
                    system_fingerprint=output.get('system_fingerprint'),
                    generation_seconds=generation_trace['elapsed_seconds'],
                    branch_seconds=branch_seconds, retrieval_seconds=retrieval_seconds,
                    end_to_end_seconds=(retrieval_seconds + branch_seconds) if comparable else None,
                    timing_comparable=comparable,
                    timing_note='Fresh path' if comparable else 'Resumed requests; path timing omitted')
            except Exception as exc:
                value.update(status='error', failed_stage=stage, error_type=type(exc).__name__,
                             error=str(exc), failed_elapsed_seconds=time.perf_counter() - branch_started)
    record['ended_utc'] = datetime.now(timezone.utc).isoformat()
    return record, candidate_q


def run_question_batch(questions, *, run, contract_hash, complete, execute_one):
    run = Path(run)
    if complete:
        return (json.loads((run / 'results.json').read_text()),
                json.loads((run / 'candidates.json').read_text()))
    results, candidates = [], []
    for position, q in enumerate(questions):
        checkpoint = run / 'question_checkpoints' / (q['query_id'] + '.json')
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if saved['contract_sha256'] != contract_hash:
                raise ValueError('Checkpoint inputs changed; start a new batch.')
            record, candidate_q = saved['result'], saved.get('candidate_question')
        else:
            record, candidate_q = execute_one(q, position)
            save_json(checkpoint, dict(contract_sha256=contract_hash,
                                      candidate_question=candidate_q, result=record))
        results.append(record)
        if candidate_q is not None:
            candidates.append(candidate_q)
    save_json(run / 'candidates.json', candidates)
    save_json(run / 'results.json', results)
    for record in results:
        if record.get('retrieval_status') == 'ok':
            save_json(run / 'retrieval' / (record['query_id'] + '.json'),
                      next(q for q in candidates if q['query_id'] == record['query_id']))
        for arm, value in record['arms'].items():
            if 'messages' in value:
                save_json(run / 'contexts_prompts' / (record['query_id'] + '-' + arm + '.json'), value)
    return results, candidates


def snapshot_run(*, root, run, notebook_path, destination, document_cache):
    """Create an immutable snapshot and a checksum inventory; never overwrite."""
    import shutil
    root, run, destination = Path(root), Path(run), Path(destination)
    if destination.exists():
        raise FileExistsError(f'Snapshot already exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + '.building')
    if staging.exists():
        raise FileExistsError(f'Incomplete snapshot exists: {staging}')
    staging.mkdir()
    try:
        shutil.copy2(notebook_path, staging / 'Experiment-1-Baseline.ipynb')
        shutil.copy2(root / 'experiment_helpers.py', staging / 'experiment_helpers.py')
        shutil.copy2(root / 'scoring.py', staging / 'scoring.py')
        shutil.copy2(root / 'config.json', staging / 'config.json')
        shutil.copytree(root / 'input', staging / 'input')
        shutil.copytree(run, staging / 'run')
        shutil.copytree(document_cache, staging / 'document_cache')
        inventory = {str(p.relative_to(staging)): file_hash(p)
                     for p in sorted(staging.rglob('*')) if p.is_file()}
        save_json(staging / 'sha256_inventory.json', inventory)
        for relative, expected in inventory.items():
            if file_hash(staging / relative) != expected:
                raise ValueError(f'Snapshot checksum mismatch: {relative}')
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def setup_document_cache(*, docs, cfg, ROOT, identity, read_only):
    document_embedding_contract = {
        'ordered_documents': docs, 'model': cfg['embedding_model'],
        'url': cfg['api_base'] + '/embeddings',
        'dimensions': cfg['embedding_dimensions'], 'encoding_format': 'float',
        'passage_char_budget': cfg['passage_char_budget'], 'dtype': 'float32'}
    document_cache_id = digest(document_embedding_contract)
    DOCUMENT_CACHE = ROOT / 'cache' / ('document_embeddings-' + document_cache_id)
    if not read_only:
        DOCUMENT_CACHE.mkdir(parents=True, exist_ok=True)
    cache_manifest_path = DOCUMENT_CACHE / 'manifest.json'
    matrix_path = DOCUMENT_CACHE / 'document_embeddings.npy'
    cache_on_entry = cache_manifest_path.exists() and matrix_path.exists()
    if cache_manifest_path.exists():
        cache_manifest = json.loads(cache_manifest_path.read_text())
        assert cache_manifest['document_contract_sha256'] == document_cache_id
        assert cache_manifest['origin'] == 'Generated by Experiment 1 notebook hosted document-embedding calls'
    else:
        if read_only:
            raise FileNotFoundError('Saved embedding cache is missing. recorded_batch never calls models; restore the cache or explicitly run a new batch.')
        cache_manifest = {'document_contract_sha256': document_cache_id,
            'origin': 'Generated by Experiment 1 notebook hosted document-embedding calls',
            'created_utc': datetime.now(timezone.utc).isoformat(), 'creator_run_id': identity['run_id'],
            'model': cfg['embedding_model'], 'endpoint': cfg['api_base'] + '/embeddings',
            'dimensions': cfg['embedding_dimensions'], 'document_count': len(docs),
            'batch_size': cfg['embedding_batch_size'], 'status': 'building'}
        save_json(cache_manifest_path, cache_manifest)
    cache_complete = cache_on_entry and cache_manifest.get('status') == 'complete'
    if not cache_complete and read_only:
        raise FileNotFoundError('Saved embedding cache is incomplete; use resume_batch explicitly.')
    return dict(document_cache_id=document_cache_id, directory=DOCUMENT_CACHE, manifest=cache_manifest, manifest_path=cache_manifest_path, matrix_path=matrix_path, complete=cache_complete)


def prepare_document_embeddings(*, docs, cfg, ROOT, RUN, identity, cache, make_request, call_model, read_only=False):
    document_cache_id = cache['document_cache_id']
    DOCUMENT_CACHE = cache['directory']
    cache_manifest = cache['manifest']
    cache_manifest_path = cache['manifest_path']
    matrix_path = cache['matrix_path']
    cache_complete = cache['complete']
    if read_only and not cache_complete:
        raise FileNotFoundError('Read-only mode requires a complete document embedding cache.')
    document_vectors = []
    embedding_total_batches = math.ceil(len(docs) / cfg['embedding_batch_size'])
    embedding_batches_generated = 0
    embedding_batches_reused = 0
    embedding_requests_this_execution = 0
    embedding_failed_attempts_this_execution = 0
    if cache_complete:
        # Load the saved document embedding matrix.
        assert file_hash(matrix_path) == cache_manifest['matrix_sha256'], 'Cached matrix checksum mismatch'
        matrix = np.load(matrix_path, allow_pickle=False)
        embedding_batches_reused = embedding_total_batches
    else:
        for batch_number, start in enumerate(range(0, len(docs), cfg['embedding_batch_size'])):
            stop = min(start + cfg['embedding_batch_size'], len(docs))
            batch = docs[start:stop]
            batch_id = f'documents-{batch_number:04d}'
            batch_path = DOCUMENT_CACHE / 'batches' / (batch_id + '.npy')
            batch_meta_path = DOCUMENT_CACHE / 'batches' / (batch_id + '.json')
            payload = make_request(batch)
            url = cfg['api_base'] + '/embeddings'
            batch_contract = digest({'request_body': payload, 'url': url,
                                     'ordered_document_ids': [d['id'] for d in batch]})
            if batch_path.exists() and batch_meta_path.exists():
                batch_meta = json.loads(batch_meta_path.read_text())
                assert batch_meta['contract_sha256'] == batch_contract
                assert batch_meta['vectors_sha256'] == file_hash(batch_path)
                batch_vectors = np.load(batch_path, allow_pickle=False)
                embedding_batches_reused += 1
            else:
                # Recover vectors from a saved successful response after an interruption.
                successful_trace = None
                previous_paths = sorted((DOCUMENT_CACHE / 'requests').glob(batch_id + '-attempt-*.json'))
                for previous_path in previous_paths:
                    previous = json.loads(previous_path.read_text())
                    assert previous['request_body'] == payload and previous['url'] == url
                    if previous['status'] == 'ok':
                        successful_trace = previous
                if successful_trace is not None:
                    trace = successful_trace
                    embedding_batches_reused += 1
                    output = json.loads((DOCUMENT_CACHE / trace['raw_response_file']).read_text())
                else:
                    for attempt_index in range(cfg['document_embedding_max_attempts']):
                        attempt_number = len(previous_paths) + attempt_index + 1
                        backoff = 0 if attempt_index == 0 else cfg['document_embedding_backoff_seconds'][attempt_index-1]
                        if backoff:
                            time.sleep(backoff)
                        request_id = batch_id + f'-attempt-{attempt_number:02d}'
                        trace_path = DOCUMENT_CACHE / 'requests' / (request_id + '.json')
                        raw_path = DOCUMENT_CACHE / 'raw_responses' / (request_id + '.json')
                        trace = {'request_id': request_id, 'stage': 'document_embedding',
                            'creator_run_id': identity['run_id'], 'batch_number': batch_number,
                            'start': start, 'stop': stop, 'ordered_document_ids': [d['id'] for d in batch],
                            'url': url, 'request_body': payload, 'attempt': attempt_number,
                            'backoff_seconds': backoff, 'status': 'started',
                            'started_utc': datetime.now(timezone.utc).isoformat()}
                        save_json(trace_path, trace)
                        embedding_requests_this_execution += 1
                        request_started = time.perf_counter()
                        try:
                            # FRESH HOSTED DOCUMENT-EMBEDDING MODEL CALL
                            response = call_model(url, payload)
                            raw_path.parent.mkdir(parents=True, exist_ok=True)
                            raw_path.write_text(response.text)
                            trace.update(http_status=response.status_code,
                                         raw_response_file=str(raw_path.relative_to(DOCUMENT_CACHE)))
                            response.raise_for_status()
                            output = response.json()
                            ordered = sorted(output['data'], key=lambda item: item['index'])
                            assert [item['index'] for item in ordered] == list(range(len(batch)))
                            values = np.asarray([item['embedding'] for item in ordered], dtype=np.float32)
                            assert values.shape == (len(batch), cfg['embedding_dimensions'])
                            assert np.isfinite(values).all()
                            trace.update(status='ok', served_model=output.get('model'), usage=output.get('usage'))
                        except Exception as exc:
                            embedding_failed_attempts_this_execution += 1
                            trace.update(status='error', error_type=type(exc).__name__, error=str(exc))
                        finally:
                            trace.update(elapsed_seconds=time.perf_counter() - request_started,
                                         ended_utc=datetime.now(timezone.utc).isoformat())
                            save_json(trace_path, trace)
                        if trace['status'] == 'ok':
                            break
                    if trace['status'] != 'ok':
                        print(f'Document embedding stopped: {embedding_batches_generated + embedding_batches_reused}/{embedding_total_batches} batches completed; '
                              f'{embedding_requests_this_execution} API attempts, {embedding_failed_attempts_this_execution} failed attempts.')
                        raise RuntimeError('Document preparation stopped after recorded failures. '
                            'Rerun to resume from this experiment’s successful batches. Last error: ' + trace['error'])
                    embedding_batches_generated += 1
                ordered = sorted(output['data'], key=lambda item: item['index'])
                assert [item['index'] for item in ordered] == list(range(len(batch)))
                batch_vectors = np.asarray([item['embedding'] for item in ordered], dtype=np.float32)
                assert batch_vectors.shape == (len(batch), cfg['embedding_dimensions']) and np.isfinite(batch_vectors).all()
                batch_path.parent.mkdir(parents=True, exist_ok=True)
                temp = batch_path.with_suffix('.npy.tmp')
                with temp.open('wb') as handle:
                    np.save(handle, batch_vectors)
                temp.replace(batch_path)
                save_json(batch_meta_path, {'contract_sha256': batch_contract,
                    'document_ids': [d['id'] for d in batch], 'vectors_sha256': file_hash(batch_path),
                    'successful_request_id': trace['request_id'], 'creator_run_id': trace['creator_run_id']})
            assert batch_vectors.shape == (len(batch), cfg['embedding_dimensions']) and np.isfinite(batch_vectors).all()
            document_vectors.append(batch_vectors)
        matrix = np.concatenate(document_vectors, axis=0)
        temp = matrix_path.with_suffix('.npy.tmp')
        with temp.open('wb') as handle:
            np.save(handle, matrix)
        temp.replace(matrix_path)
        cache_manifest.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
                              matrix_shape=list(matrix.shape), matrix_sha256=file_hash(matrix_path))
        save_json(cache_manifest_path, cache_manifest)

    assert matrix.shape == (len(docs), cfg['embedding_dimensions']) and np.isfinite(matrix).all()
    document_traces = [json.loads(p.read_text()) for p in sorted((DOCUMENT_CACHE / 'requests').glob('*.json'))]
    document_attempts = pd.DataFrame([{'request_id': t['request_id'], 'stage': t['stage'],
        'batch_number': t['batch_number'], 'attempt': t['attempt'], 'status': t['status'],
        'http_status': t.get('http_status'), 'elapsed_seconds': t.get('elapsed_seconds'),
        'backoff_seconds': t['backoff_seconds'], 'error': t.get('error')}
        for t in document_traces])
    if not read_only:
        document_attempts.to_csv(RUN / 'document_embedding_attempts.csv', index=False)
    if not read_only:
        save_json(RUN / 'document_embedding_origin.json', {'cache_id': document_cache_id,
        'cache_relative_path': str(DOCUMENT_CACHE.relative_to(ROOT)),
        'origin': cache_manifest['origin'], 'creator_run_id': cache_manifest['creator_run_id'],
        'matrix_sha256': cache_manifest['matrix_sha256'], 'matrix_shape': list(matrix.shape)})
    print(
        'Document embedding summary\n'
        f"Passages: {len(docs):,} | Batches: {embedding_total_batches} | Dimensions: {cfg['embedding_dimensions']:,}\n"
        f"Fresh batches: {embedding_batches_generated} | Reused batches: {embedding_batches_reused}\n"
        f"API attempts this execution: {embedding_requests_this_execution} | Failed attempts: {embedding_failed_attempts_this_execution}"
    )
    return matrix, document_attempts, dict(passages=len(docs), batches=embedding_total_batches, dimensions=cfg['embedding_dimensions'], fresh_batches=embedding_batches_generated, reused_batches=embedding_batches_reused, api_attempts=embedding_requests_this_execution, failed_attempts=embedding_failed_attempts_this_execution)
