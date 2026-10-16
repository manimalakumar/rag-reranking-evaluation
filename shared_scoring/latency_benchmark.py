from .provider import embedding_payload, rerank_payload, rerank_scores, QUERY_INSTRUCTION, require_model_calls
"""Explicit sequential benchmark; imports and saved replay make no hosted calls."""
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid
from datetime import datetime, timezone

import numpy as np

from .budget import reserve_request
from .io import read_json, write_json, sha256_file


def _now():
    return datetime.now(timezone.utc).isoformat()


class HostedStageFailure(RuntimeError):
    """A hosted stage exhausted its attempts; its full trace stays attached."""
    def __init__(self, trace):
        self.trace = trace
        super().__init__(f"{trace['stage']} failed after {trace['attempt_count']} attempts")


def _request_stage(session, stage, url, payload, api_key, settings, save_attempt):
    require_model_calls()
    started = time.perf_counter()
    attempts = []
    for number in range(1, settings['max_attempts_per_hosted_stage'] + 1):
        reserve_request(url, payload, settings.get('_request_identity', stage), settings['max_attempts_per_hosted_stage'])
        record = {'stage': stage, 'attempt': number, 'started_utc': _now(), 'status': 'started',
                  'url': url, 'request_body': payload, 'raw_response': None, 'raw_response_text': None,
                  'status_code': None, 'elapsed_seconds': None, 'error': None, 'backoff_after_seconds': 0.0}
        save_attempt(record)
        attempt_started = time.perf_counter()
        try:
            response = session.post(url, json=payload, headers={'Authorization': f'Bearer {api_key}'},
                                    timeout=settings['timeout_seconds'])
            record['status_code'] = response.status_code
            record['raw_response_text'] = response.text
            record['raw_response'] = response.json()
            if not response.ok:
                raise RuntimeError(f'HTTP {response.status_code}')
            record.update(status='success', elapsed_seconds=time.perf_counter() - attempt_started)
        except Exception as error:
            record.update(status='error', elapsed_seconds=time.perf_counter() - attempt_started,
                          error=f'{type(error).__name__}: {error}')
        save_attempt(record)
        if record['status'] == 'error' and number < settings['max_attempts_per_hosted_stage']:
            backoff_started = time.perf_counter()
            time.sleep(settings['backoff_schedule_seconds'][number - 1])
            record['backoff_after_seconds'] = time.perf_counter() - backoff_started
            save_attempt(record)
        attempts.append(record)
        if record['status'] == 'success':
            break
    trace = {'stage': stage, 'url': url, 'request_body': payload, 'attempts': attempts,
             'status': attempts[-1]['status'], 'attempt_count': len(attempts),
             'successful_request_seconds': attempts[-1]['elapsed_seconds'] if attempts[-1]['status'] == 'success' else None,
             'all_attempt_seconds': sum(row['elapsed_seconds'] for row in attempts),
             'backoff_seconds': sum(row['backoff_after_seconds'] for row in attempts),
             'stage_total_seconds': time.perf_counter() - started}
    if trace['status'] != 'success':
        raise HostedStageFailure(trace)
    return attempts[-1]['raw_response'], trace


def retrieve_candidates(index, corpus, vector, dimensions, candidate_count):
    """Search the preloaded index with one normalized query vector."""
    started = time.perf_counter()
    array = np.asarray([vector], dtype='float32')
    if array.shape != (1, dimensions) or not np.isfinite(array).all() or np.linalg.norm(array) == 0:
        raise ValueError('The query embedding has invalid dimensions, values or norm.')
    array /= np.linalg.norm(array, axis=1, keepdims=True)
    scores, indices = index.search(array, candidate_count)
    if (indices < 0).any():
        raise ValueError('The configured index returned an invalid corpus row.')
    rows = corpus.take(indices[0].tolist()).to_pydict()
    candidates = [{'id': rows['passage_id'][i], 'text': rows['text'][i],
                   'source_url': rows['source_url'][i], 'truncated': bool(rows['truncated'][i]),
                   'embedding_rank': i + 1, 'similarity_score': float(scores[0, i])}
                  for i in range(candidate_count)]
    if len({row['id'] for row in candidates}) != candidate_count:
        raise ValueError('The configured index returned duplicate passage IDs.')
    return candidates, time.perf_counter() - started


def _execute_path(config, assignment, arm, session, api_key, index, corpus, save_attempt, traces):
    from .opportunity_study import generation_messages
    started_utc, started = _now(), time.perf_counter()
    settings = {**config['benchmark'], '_request_identity': [config['run_id'], str(assignment['query_id']), arm]}
    embedding = settings['embedding_settings']
    payload = embedding_payload(embedding['model'], [assignment['question']], embedding['dimensions'],
                                instruction=embedding.get('query_instruction', QUERY_INSTRUCTION))
    raw, trace = _request_stage(session, 'query_embedding', embedding['endpoint'], payload,
                                api_key, settings, save_attempt)
    traces.append(trace)
    vector = sorted(raw['data'], key=lambda item: item['index'])[0]['embedding']
    candidates, retrieval_seconds = retrieve_candidates(index, corpus, vector,
        embedding['dimensions'], config['candidate_count'])
    ranked = candidates
    if arm == list(config['arms'])[1]:
        reranker = settings['reranker_settings']
        payload = rerank_payload(reranker['model'], assignment['question'], candidates)
        raw, trace = _request_stage(session, 'rerank_30', reranker['endpoint'], payload,
                                    api_key, settings, save_attempt)
        traces.append(trace)
        ranking = rerank_scores(raw, config['candidate_count'])
        if sorted(int(row['index']) for row in ranking) != list(range(config['candidate_count'])) or not all(math.isfinite(float(row['logit'])) for row in ranking):
            raise ValueError('The reranker returned an invalid candidate permutation or score.')
        ordered = sorted(ranking, key=lambda row: (-float(row['logit']), int(row['index'])))
        ranked = [{**candidates[int(row['index'])], 'rerank_rank': i + 1,
                   'rerank_score': float(row['logit'])} for i, row in enumerate(ordered)]
    selection_started = time.perf_counter()
    selected = ranked[:config['context_passage_count']]
    context = '\n\n'.join(f"[{row['id']}]\n{row['text']}" for row in selected)
    generation = config['generation']
    messages = generation_messages(assignment['question'], context, generation['system_prompt'])
    local = {'retrieve_30_seconds': retrieval_seconds,
             'select_and_build_prompt_seconds': time.perf_counter() - selection_started}
    payload = {'model': generation['model'], 'messages': messages, 'temperature': generation['temperature'],
               'max_tokens': generation['max_tokens'], **generation['extras']}
    raw, trace = _request_stage(session, 'generation', generation['api_base'] + '/chat/completions',
                                payload, api_key, settings, save_attempt)
    traces.append(trace)
    choice = raw['choices'][0]
    if not isinstance(choice['message']['content'], str):
        raise ValueError('The generator response must retain an answer string, including when empty.')
    successful = sum(row['successful_request_seconds'] for row in traces)
    return {'query_id': str(assignment['query_id']), 'question': assignment['question'],
        'generation_stratum': assignment['generation_stratum'], 'latency_sample_order': int(assignment['latency_sample_order']),
        'arm': arm, 'arm_position': list(assignment['arm_order']).index(arm) + 1, 'status': 'success',
        'started_utc': started_utc, 'completed_utc': _now(),
        'successful_request_seconds': successful,
        'hosted_all_attempt_seconds': sum(row['all_attempt_seconds'] for row in traces),
        'backoff_seconds': sum(row['backoff_seconds'] for row in traces),
        'local_compute_seconds': sum(local.values()), 'successful_path_seconds': successful + sum(local.values()),
        'total_end_to_end_seconds': time.perf_counter() - started,
        'retry_count': sum(row['attempt_count'] - 1 for row in traces), 'local_timings': local,
        'stage_traces': traces, 'retrieved_top30': candidates, 'selected_top5': selected, 'messages': messages,
        'answer': choice['message']['content'], 'finish_reason': choice.get('finish_reason'),
        'generation_usage': raw.get('usage'), 'served_model': raw.get('model'),
        'system_fingerprint': raw.get('system_fingerprint')}


def _load_assets(config, corpus_file, index_file):
    try:
        import faiss
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError('A new benchmark needs the pinned optional requirements-latency.txt.') from error
    settings = config['benchmark']
    for name, path in [('corpus', corpus_file), ('index', index_file)]:
        if not path.is_file():
            raise FileNotFoundError(f'Missing precomputed {name} asset: {path}')
        if path.stat().st_size != settings['asset_sizes'][name]:
            raise ValueError(f'The {name} asset size differs from the frozen contract; use a new configuration.')
    index = faiss.read_index(str(index_file))
    index.nprobe = settings['retrieval_settings']['nprobe']
    corpus = pq.read_table(corpus_file, columns=['passage_id', 'text', 'source_url', 'truncated'], memory_map=True)
    if index.d != settings['embedding_settings']['dimensions'] or index.ntotal != settings['corpus_rows'] or corpus.num_rows != index.ntotal:
        raise ValueError('Precomputed corpus/index count or embedding dimensions differ from configuration.')
    if index.nlist != settings['retrieval_settings']['actual_nlist'] or not 1 <= index.nprobe <= index.nlist:
        raise ValueError('FAISS nlist or nprobe differs from the recorded retrieval contract.')
    return index, corpus


def run_new_latency_benchmark(config, output_directory, corpus_file=None, index_file=None,
                              allow_model_calls=False, retry_failures=False):
    """Run/resume a separate sequential benchmark; preserve the original saved study."""
    if not allow_model_calls:
        raise PermissionError('New benchmark calls are disabled; explicitly set allow_model_calls=True.')
    from .latency_study import load_latency_population
    settings = config['benchmark']
    if settings['workers'] != 1 or settings['max_attempts_per_hosted_stage'] < 1:
        raise ValueError('This sequential benchmark requires one worker and positive attempt limits.')
    if len(settings['backoff_schedule_seconds']) < settings['max_attempts_per_hosted_stage'] - 1:
        raise ValueError('Provide a backoff delay for each possible hosted-stage retry.')
    output = Path(output_directory).resolve()
    frozen_paths = [config['study_directory'] / name for name in read_json(config['study_directory'] / 'input/file_checksums.json')]
    if any(path.resolve().is_relative_to(output) or output.is_relative_to(path.resolve()) for path in frozen_paths) or output == config['score_directory'] or output.is_relative_to(config['score_directory']):
        raise PermissionError('Choose a separate benchmark directory outside the frozen artifacts and scores.')
    asset_paths = {}
    for name, supplied in [('corpus', corpus_file), ('index', index_file)]:
        value = supplied or os.environ.get(settings['asset_env'][name])
        if value:
            asset_paths[name] = Path(value).expanduser().resolve()
    if len(asset_paths) != 2 or not all(path.is_file() for path in asset_paths.values()):
        from .execution import ensure_latency_assets
        project = next((parent for parent in [config['configuration_file'].parent, *config['configuration_file'].parents]
                        if (parent / 'configs/execution.json').is_file()), None)
        if project is None:
            raise FileNotFoundError('Missing corpus/index assets. Run inside the repository containing configs/execution.json.')
        config, corpus_path, index_path = ensure_latency_assets(config, project)
        settings = config['benchmark']
        asset_paths = {'corpus': corpus_path, 'index': index_path}
    key = os.environ.get(settings['credential_env'])
    if not key:
        raise RuntimeError(f'Missing environment variable {settings["credential_env"]}')
    questions = load_latency_population(config)
    identity = {'benchmark': settings, 'generation': config['generation'], 'arms': config['arms'],
                'candidate_count': config['candidate_count'], 'context_passage_count': config['context_passage_count'],
                'question_file_sha256': sha256_file(config['questions_file']),
                'asset_sha256': {name: sha256_file(path) for name, path in asset_paths.items()}}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest_path = output / 'benchmark_manifest.json'
    if manifest_path.exists() and read_json(manifest_path)['fingerprint'] != fingerprint:
        raise ValueError('New-benchmark inputs/settings changed; choose a fresh output directory.')
    if output.exists() and not manifest_path.exists() and any(output.iterdir()):
        raise FileExistsError('A nonempty output directory has no matching benchmark manifest.')
    for row in questions.itertuples():
        for arm in config['arms']:
            path = output / 'arm_checkpoints' / f'{row.query_id}-{arm}.json'
            if path.exists() and read_json(path)['status'] != 'success' and not retry_failures:
                raise PermissionError('A failed benchmark arm is retained; enable retry_failures explicitly to retry it.')
    setup_started = time.perf_counter()
    index, corpus = _load_assets(config, asset_paths['corpus'], asset_paths['index'])
    setup_seconds = time.perf_counter() - setup_started
    if not manifest_path.exists():
        write_json(manifest_path, {'fingerprint': fingerprint, 'settings': identity, 'created_utc': _now(),
            'asset_paths_saved': False, 'credentials_saved': False,
            'timing_scope': 'Precomputed document embeddings, asset hashing and index/corpus loading excluded from each path.'})
    import requests
    completed = 0
    with requests.Session() as session:
        for assignment in questions.sort_values('latency_sample_order').to_dict('records'):
            for arm in list(assignment['arm_order']):
                path = output / 'arm_checkpoints' / f'{assignment["query_id"]}-{arm}.json'
                if path.exists() and read_json(path)['status'] == 'success':
                    continue
                invocation = uuid.uuid4().hex
                traces = []
                def save_attempt(attempt):
                    write_json(output / 'request_attempts' / f'{assignment["query_id"]}-{arm}-{invocation}-{attempt["stage"]}-{attempt["attempt"]}.json',
                               {'query_id': str(assignment['query_id']), 'arm': arm, **attempt})
                try:
                    result = _execute_path(config, assignment, arm, session, key, index, corpus, save_attempt, traces)
                except Exception as error:
                    if isinstance(error, HostedStageFailure):
                        traces.append(error.trace)
                    result = {'query_id': str(assignment['query_id']), 'arm': arm, 'status': 'error',
                              'completed_utc': _now(), 'error': f'{type(error).__name__}: {error}',
                              'stage_traces': traces, 'excluded_from_latency_results': True}
                    write_json(output / 'path_attempts' / f'{assignment["query_id"]}-{arm}-{invocation}.json', result)
                    write_json(path, result)
                    raise
                write_json(output / 'path_attempts' / f'{assignment["query_id"]}-{arm}-{invocation}.json', result)
                write_json(path, result)
                completed += 1
    summary = {'completed_utc': _now(), 'questions': len(questions), 'expected_paths': len(questions) * len(config['arms']),
               'new_successful_paths': completed, 'workers': settings['workers'],
               'setup_seconds_excluded_from_paths': setup_seconds}
    write_json(output / 'run_summary.json', summary)
    from .latency_study import flatten_latency_checkpoints, derive_latency_summaries
    records = [read_json(output / 'arm_checkpoints' / f'{row.query_id}-{arm}.json')
               for row in questions.itertuples() for arm in config['arms']]
    tables = flatten_latency_checkpoints(records)
    for name in ['results', 'stages', 'attempts']:
        tables[name].to_parquet(output / f'{name}.parquet', index=False)
        tables[name].to_csv(output / f'{name}.csv', index=False)
    for name in ['request_bodies', 'raw_responses']:
        (output / f'{name}.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in tables[name]))
    arm_summary, stage_summary, paired = derive_latency_summaries(tables['results'], tables['stages'], questions, config)
    arm_summary.to_csv(output / 'latency_summary_by_arm.csv', index=False)
    stage_summary.to_csv(output / 'stage_summary.csv', index=False)
    paired.to_parquet(output / 'paired_latency_changes.parquet', index=False)
    paired.to_csv(output / 'paired_latency_changes.csv', index=False)
    return summary
