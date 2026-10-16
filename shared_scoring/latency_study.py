from .provider import embedding_payload, rerank_payload, rerank_scores, QUERY_INSTRUCTION
"""Frozen latency measurements and adapters for shared answer scoring."""
import json
import math
from pathlib import Path

import pandas as pd

from .answers import score_custom_answer
from .io import read_json, sha256_file
from .retrieval import score_retrieval


def read_jsonl(path):
    """Read a saved JSONL trace without modifying its records."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def select_latency_sample(parent, strata, questions_per_stratum, seed, arms):
    """Reproduce the historical draw: the same seed within every stratum."""
    parts = []
    for stratum in strata:
        group = parent[parent.generation_stratum.eq(stratum)]
        if group.empty:
            continue
        parts.append(group.sample(n=min(len(group), questions_per_stratum), random_state=seed))
    if not parts:
        raise ValueError('No eligible parent questions are available for the latency sample.')
    sample = pd.concat(parts).sort_values(['generation_stratum', 'query_id']).reset_index(drop=True)
    sample['latency_sample_order'] = range(len(sample))
    sample['arm_order'] = [list(arms) if order % 2 == 0 else list(arms)[::-1]
                           for order in sample.latency_sample_order]
    return sample


def load_latency_population(config):
    """Read the original latency sample and verify its draw from the parent study."""
    from .opportunity_study import load_opportunity_configuration, load_opportunity_population
    questions = pd.read_parquet(config['questions_file'])
    questions['query_id'] = questions.query_id.astype(str)
    if config.get('eligible_question_ids_file'):
        allowed = set(Path(config['eligible_question_ids_file']).read_text().splitlines())
        if not set(questions.query_id) <= allowed:
            raise ValueError('Latency study contains questions outside the fully covered eligible pool.')
    if questions.query_id.duplicated().any() or questions.cohort.value_counts().to_dict() != config['expected_cohort_counts']:
        raise ValueError('Frozen latency question IDs or cohort counts changed.')
    if set(questions.query_id) & set(config['diagnostic_ids']):
        raise ValueError('Configured excluded diagnostic entered the latency sample.')
    parent = load_opportunity_population(load_opportunity_configuration(config['parent_configuration']))
    settings = config['sampling']
    sample = select_latency_sample(parent, settings['strata'], settings['questions_per_stratum'],
                                   settings['seed'], config['arms'])
    for column in ['query_id', 'generation_stratum', 'latency_sample_order']:
        if sample[column].tolist() != questions[column].tolist():
            raise ValueError(f'Latency sample differs from its original seeded parent draw: {column}')
    parent_by_id = parent.set_index('query_id')
    for row in questions.itertuples():
        original = parent_by_id.loc[row.query_id]
        if list(row.arm_order) != sample.loc[row.latency_sample_order, 'arm_order']:
            raise ValueError(f'{row.query_id}: frozen latency arm order changed.')
        if row.question != original.question or row.cohort != original.cohort:
            raise ValueError(f'{row.query_id}: parent question or cohort changed.')
        for column in ['evaluation_references', 'known_supporting_passage_ids']:
            if list(getattr(row, column)) != list(original[column]):
                raise ValueError(f'{row.query_id}: original {column} changed.')
    return questions


def _checkpoint_context(record):
    passages = record['selected_top5']
    blocks = [f"[{passage['id']}]\n{passage['text']}" for passage in passages]
    return {'context': '\n\n'.join(blocks), 'retrieved_contexts': blocks,
            'context_passage_ids': [passage['id'] for passage in passages], 'passages': passages}


def _verify_checkpoint(record, question, config):
    from .opportunity_study import generation_messages
    arm, qid = record['arm'], record['query_id']
    if record['status'] != 'success':
        raise ValueError(f'{qid}/{arm}: inspect this failed historical complete-path checkpoint.')
    if record['question'] != question.question or record['generation_stratum'] != question.generation_stratum:
        raise ValueError(f'{qid}/{arm}: saved path question or stratum changed.')
    if record['arm_position'] != list(question.arm_order).index(arm) + 1:
        raise ValueError(f'{qid}/{arm}: saved path order changed.')
    candidates = record['retrieved_top30']
    ids = [passage['id'] for passage in candidates]
    if len(ids) != config['candidate_count'] or len(set(ids)) != len(ids):
        raise ValueError(f'{qid}/{arm}: incomplete or duplicate candidate passages.')
    if [passage['embedding_rank'] for passage in candidates] != list(range(1, len(ids) + 1)):
        raise ValueError(f'{qid}/{arm}: embedding ranks changed.')
    traces = record['stage_traces']
    if [trace['stage'] for trace in traces] != config['benchmark']['stages_by_arm'][arm]:
        raise ValueError(f'{qid}/{arm}: recorded hosted-stage contract changed.')
    for trace in traces:
        attempts = trace['attempts']
        if not attempts or len(attempts) != trace['attempt_count'] or attempts[-1]['status'] != 'success':
            raise ValueError(f'{qid}/{arm}/{trace["stage"]}: incomplete attempt history.')
    embedding = traces[0]
    settings = config['benchmark']['embedding_settings']
    expected_embedding = embedding_payload(settings['model'], [question.question], settings['dimensions'],
                                           instruction=settings.get('query_instruction', QUERY_INSTRUCTION))
    if embedding['request_body'] != expected_embedding or embedding['url'] != settings['endpoint']:
        raise ValueError(f'{qid}/{arm}: recorded query embedding request changed.')
    context = _checkpoint_context(record)
    if len(context['context_passage_ids']) != config['context_passage_count'] or len(set(context['context_passage_ids'])) != config['context_passage_count']:
        raise ValueError(f'{qid}/{arm}: invalid saved five-passage context.')
    selected = candidates[:config['context_passage_count']]
    if arm == list(config['arms'])[1]:
        rerank = traces[1]
        settings = config['benchmark']['reranker_settings']
        expected = rerank_payload(settings['model'], question.question, candidates)
        if rerank['request_body'] != expected or rerank['url'] != settings['endpoint']:
            raise ValueError(f'{qid}/{arm}: reranker did not receive this arm’s exact candidates.')
        ranking = rerank_scores(rerank['attempts'][-1]['raw_response'], config['candidate_count'])
        if sorted(int(item['index']) for item in ranking) != list(range(config['candidate_count'])):
            raise ValueError(f'{qid}/{arm}: incomplete reranker permutation.')
        if not all(math.isfinite(float(item['logit'])) for item in ranking):
            raise ValueError(f'{qid}/{arm}: non-finite reranker score.')
        ordered = sorted(ranking, key=lambda item: (-float(item['logit']), int(item['index'])))
        selected = [candidates[int(item['index'])] for item in ordered[:config['context_passage_count']]]
    if [(item['id'], item['text']) for item in selected] != [(item['id'], item['text']) for item in record['selected_top5']]:
        raise ValueError(f'{qid}/{arm}: selected passage IDs, text or order changed.')
    generation = traces[-1]
    settings = config['generation']
    messages = generation_messages(question.question, context['context'], settings['system_prompt'])
    payload = {'model': settings['model'], 'messages': messages, 'temperature': settings['temperature'],
               'max_tokens': settings['max_tokens'], **settings['extras']}
    if record['messages'] != messages or generation['request_body'] != payload or generation['url'] != settings['api_base'] + '/chat/completions':
        raise ValueError(f'{qid}/{arm}: frozen generation messages or settings changed.')
    choice = generation['attempts'][-1]['raw_response']['choices'][0]
    if record['answer'] != choice['message']['content'] or record['finish_reason'] != choice.get('finish_reason'):
        raise ValueError(f'{qid}/{arm}: saved answer differs from the original response.')


def load_latency_checkpoints(config):
    """Verify exact per-arm candidates, requests, answers and separate contexts."""
    questions = load_latency_population(config).set_index('query_id')
    expected = {(qid, arm) for qid in questions.index for arm in config['arms']}
    records = {}
    for path in sorted(config['checkpoint_file'].glob('*.json')):
        record = read_json(path)
        key = (str(record['query_id']), record['arm'])
        if key not in expected or key in records or path.stem != '-'.join(key):
            raise ValueError(f'Unexpected or duplicate latency checkpoint: {path.name}')
        _verify_checkpoint(record, questions.loc[key[0]], config)
        records[key] = record
    if set(records) != expected:
        raise ValueError('Missing frozen latency checkpoint; replay never generates a replacement.')
    return records


def load_latency_contexts(config, query_ids=None):
    """Load each measured arm’s own exact five passages without combining contexts."""
    selected = None if query_ids is None else {str(qid) for qid in query_ids}
    return {key: _checkpoint_context(record) for key, record in load_latency_checkpoints(config).items()
            if selected is None or key[0] in selected}


def load_latency_generation_records(config):
    """Adapt complete paths for shared replay; transport failures remain in traces."""
    from .opportunity_study import generation_fingerprint
    output = []
    for (qid, arm), record in load_latency_checkpoints(config).items():
        trace = record['stage_traces'][-1]
        successful = trace['attempts'][-1]
        raw = successful['raw_response']
        output.append({**record, 'source_status': record['status'], 'source_arm': arm,
            'status': 'ok', 'attempt_id': f'{qid}-{arm}-generation-{successful["attempt"]}',
            'truncated': record['finish_reason'] == 'length', 'raw_response_json': raw,
            'raw_response_text': json.dumps(raw, ensure_ascii=False),
            'raw_response_text_origin': 'Rendered from original saved response JSON; HTTP wire bytes unavailable.',
            'request_body': trace['request_body'], 'url': trace['url'],
            'request_body_origin': 'Exact recorded request body; original HTTP wire bytes unavailable.',
            'context_passage_ids': _checkpoint_context(record)['context_passage_ids'],
            'elapsed_seconds': trace['stage_total_seconds'], 'execution_workers': config['benchmark']['workers'],
            'fingerprint': generation_fingerprint(record['question'], arm, record['messages'], config['generation'])})
    return output


def build_latency_evaluation_inputs(config):
    """Score only the benchmark answers, using their own evidence and all references."""
    records = {(row['query_id'], row['arm']): row for row in load_latency_generation_records(config)}
    inputs, pairs = [], []
    for question in load_latency_population(config).itertuples():
        pairs.append(question.query_id)
        for arm, label in config['arms'].items():
            record = records[(question.query_id, arm)]
            context = _checkpoint_context(record)
            references = list(getattr(question, config['reference_answers_column']))
            custom = score_custom_answer(record['answer'], references, config['legacy_citation_pattern'])
            recall = score_retrieval(context['context_passage_ids'], question.known_supporting_passage_ids)['recall']
            inputs.append({'query_id': question.query_id, 'question': question.question,
                'arm': arm, 'path': label, 'answer': record['answer'], 'reference_answers': references,
                'context': context['context'], 'retrieved_contexts': context['retrieved_contexts'],
                'context_passage_ids': context['context_passage_ids'], 'finish_reason': record['finish_reason'],
                'truncated': record['truncated'], 'annotated_support_recall_at_5': recall,
                'existing_token_f1_0_1': custom['token_f1'], 'existing_normalized_em_0_1': custom['normalized_em'],
                'generation_fingerprint': record['fingerprint']})
    return inputs, pairs, []


def derive_latency_summaries(results, stages, questions, config):
    """Reproduce original medians, linearly interpolated p95 values and paired deltas."""
    percentile = config['latency_summary']['percentile']
    interpolation = config['latency_summary']['interpolation']
    metrics = config['latency_summary']['paired_metrics']
    rows = []
    for arm, group in results.groupby('arm', sort=True):
        row = {'arm': arm, 'questions': len(group), 'retries': int(group.retry_count.sum())}
        for metric in metrics:
            row['median_' + metric] = group[metric].median()
            row['p95_' + metric] = group[metric].quantile(percentile, interpolation=interpolation)
        row['median_retry_overhead_seconds'] = (group.total_end_to_end_seconds - group.successful_path_seconds).median()
        rows.append(row)
    summary = pd.DataFrame(rows)
    stage_summary = stages.groupby(['arm', 'stage'], as_index=False).agg(
        calls=('stage', 'size'), median_stage_seconds=('stage_total_seconds', 'median'),
        p95_stage_seconds=('stage_total_seconds', lambda values: values.quantile(percentile, interpolation=interpolation)),
        retries=('attempt_count', lambda values: int((values - 1).clip(lower=0).sum())))
    paired = results.pivot(index='query_id', columns='arm', values=metrics)
    paired.columns = [f'{metric}_{arm}' for metric, arm in paired.columns]
    paired = paired.reset_index()
    baseline, reranked = config['arms']
    for metric in metrics:
        paired[metric + '_added_by_reranking'] = paired[f'{metric}_{reranked}'] - paired[f'{metric}_{baseline}']
    paired = paired.merge(questions[['query_id', 'question', 'generation_stratum', 'latency_sample_order', 'arm_order']],
                          on='query_id', validate='one_to_one').sort_values('latency_sample_order')
    return summary, stage_summary, paired


def load_saved_latency_tables(config):
    """Read all original measurement tables, including failures and excluded preflight."""
    files = config['latency_files']
    return {**{name: pd.read_parquet(files[name]) for name in ['results', 'stages', 'attempts', 'paired']},
            **{name: pd.read_csv(files[name]) for name in ['summary', 'stage_summary']},
            **{name: read_jsonl(files[name]) for name in ['request_bodies', 'raw_responses', 'preflight_failures']},
            'run_metadata': read_json(files['run_metadata']), 'run_summary': read_json(files['run_summary'])}


def flatten_latency_checkpoints(records):
    """Convert complete path records to portable path, stage and attempt tables."""
    results, stages, attempts, bodies, responses = [], [], [], [], []
    for record in records:
        if record['status'] != 'success':
            raise ValueError('Failed paths remain in the ledger; complete-path summaries require successful pairs.')
        row = {key: value for key, value in record.items() if key not in
               ['stage_traces', 'retrieved_top30', 'selected_top5', 'messages', 'local_timings']}
        row['retrieved_top30_passage_ids'] = [p['id'] for p in record['retrieved_top30']]
        row['selected_top5_passage_ids'] = [p['id'] for p in record['selected_top5']]
        results.append(row)
        identity = {key: record[key] for key in ['query_id', 'arm', 'arm_position']}
        for stage, seconds in record['local_timings'].items():
            stages.append({**identity, 'stage': stage, 'stage_type': 'local', 'status': 'success',
                'attempt_count': 0, 'successful_request_seconds': 0.0, 'all_attempt_seconds': 0.0,
                'backoff_seconds': 0.0, 'stage_total_seconds': seconds})
        for trace in record['stage_traces']:
            item = {**identity, 'stage': trace['stage']}
            stages.append({**item, 'stage_type': 'hosted_request', **{key: trace[key] for key in
                ['status', 'attempt_count', 'successful_request_seconds', 'all_attempt_seconds', 'backoff_seconds', 'stage_total_seconds']}})
            bodies.append({**item, 'url': trace['url'], 'request_body': trace['request_body']})
            for attempt in trace['attempts']:
                attempts.append({**item, **{key: attempt[key] for key in
                    ['attempt', 'started_utc', 'status', 'status_code', 'elapsed_seconds', 'error', 'backoff_after_seconds']}})
                responses.append({**item, 'attempt': attempt['attempt'], 'raw_response': attempt['raw_response']})
    return {'results': pd.DataFrame(results).sort_values(['latency_sample_order', 'arm_position']),
            'stages': pd.DataFrame(stages).sort_values(['query_id', 'arm', 'stage']),
            'attempts': pd.DataFrame(attempts).sort_values(['query_id', 'arm', 'stage', 'attempt']),
            'request_bodies': bodies, 'raw_responses': responses}


def validate_saved_latency_study(config):
    """Check every frozen file and reproduce original path, stage and attempt accounting."""
    artifact_directory=config.get('artifact_directory',config['study_directory'])
    for relative, expected in read_json(artifact_directory / 'input/file_checksums.json').items():
        if sha256_file(artifact_directory / relative) != expected:
            raise ValueError(f'Frozen latency artifact bytes changed: {relative}')
    provenance = read_json(artifact_directory / 'input/provenance.json')
    for relative, expected in provenance['parent_files_sha256'].items():
        if sha256_file(artifact_directory / relative) != expected:
            raise ValueError(f'Parent study bytes changed: {relative}')
    questions = load_latency_population(config)
    records = load_latency_checkpoints(config)
    saved = load_saved_latency_tables(config)
    tolerance = config['latency_summary']['validation_tolerance_seconds']
    results = saved['results']
    if len(results) != len(records) or set(zip(results.query_id, results.arm)) != set(records):
        raise ValueError('Original result rows differ from the frozen path checkpoints.')
    stages, attempts, bodies, responses = [], [], [], []
    for row in results.itertuples():
        record = records[(str(row.query_id), row.arm)]
        for metric in config['latency_summary']['paired_metrics']:
            if abs(getattr(row, metric) - record[metric]) > tolerance:
                raise ValueError(f'{row.query_id}/{row.arm}: original path timing changed.')
        if row.answer != record['answer'] or row.finish_reason != record['finish_reason'] or list(row.selected_top5_passage_ids) != [p['id'] for p in record['selected_top5']]:
            raise ValueError(f'{row.query_id}/{row.arm}: answer or selected IDs differ from checkpoint.')
        trace = record['stage_traces']
        sums = {'successful_request_seconds': sum(t['successful_request_seconds'] for t in trace),
                'hosted_all_attempt_seconds': sum(t['all_attempt_seconds'] for t in trace),
                'backoff_seconds': sum(t['backoff_seconds'] for t in trace),
                'local_compute_seconds': sum(record['local_timings'].values())}
        sums['successful_path_seconds'] = sums['successful_request_seconds'] + sums['local_compute_seconds']
        if any(abs(record[key] - value) > tolerance for key, value in sums.items()):
            raise ValueError(f'{row.query_id}/{row.arm}: stage-to-path timing accounting changed.')
        if record['retry_count'] != sum(t['attempt_count'] - 1 for t in trace):
            raise ValueError('Original retry count differs from the attempt history.')
        if record['total_end_to_end_seconds'] + tolerance < sums['hosted_all_attempt_seconds'] + sums['backoff_seconds'] + sums['local_compute_seconds']:
            raise ValueError('Total wall time omits recorded work or backoff.')
        identity = {'query_id': row.query_id, 'arm': row.arm, 'arm_position': record['arm_position']}
        for stage, seconds in record['local_timings'].items():
            stages.append({**identity, 'stage': stage, 'stage_type': 'local', 'status': 'success',
                'attempt_count': 0, 'successful_request_seconds': 0.0, 'all_attempt_seconds': 0.0,
                'backoff_seconds': 0.0, 'stage_total_seconds': seconds})
        for trace in record['stage_traces']:
            item = {**identity, 'stage': trace['stage']}
            stages.append({**item, 'stage_type': 'hosted_request', **{key: trace[key] for key in
                ['status', 'attempt_count', 'successful_request_seconds', 'all_attempt_seconds', 'backoff_seconds', 'stage_total_seconds']}})
            bodies.append({**item, 'url': trace['url'], 'request_body': trace['request_body']})
            if abs(trace['all_attempt_seconds'] - sum(a['elapsed_seconds'] for a in trace['attempts'])) > tolerance:
                raise ValueError('Hosted stage lost an attempt duration.')
            for attempt in trace['attempts']:
                attempts.append({**item, **{key: attempt[key] for key in
                    ['attempt', 'started_utc', 'status', 'status_code', 'elapsed_seconds', 'error', 'backoff_after_seconds']}})
                responses.append({**item, 'attempt': attempt['attempt'], 'raw_response': attempt['raw_response']})
    for name, rows, keys in [('stages', stages, ['query_id', 'arm', 'stage']),
                             ('attempts', attempts, ['query_id', 'arm', 'stage', 'attempt'])]:
        actual = pd.DataFrame(rows)[saved[name].columns].sort_values(keys).reset_index(drop=True)
        expected = saved[name].sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=False, rtol=0, atol=tolerance)
    for name, rows in [('request_bodies', bodies), ('raw_responses', responses)]:
        order = lambda item: (item['query_id'], item['arm'], item['stage'], item.get('attempt', 0))
        if sorted(rows, key=order) != sorted(saved[name], key=order):
            raise ValueError(f'Original {name} differ from checkpoint traces.')
    summary, stage_summary, paired = derive_latency_summaries(results, saved['stages'], questions, config)
    for actual, expected in [(summary, saved['summary']), (stage_summary, saved['stage_summary']), (paired, saved['paired'])]:
        pd.testing.assert_frame_equal(actual[expected.columns].reset_index(drop=True), expected.reset_index(drop=True),
                                      check_dtype=False, check_exact=False, rtol=0, atol=1e-12)
    manifest_key = 'manifest_sha256' if config.get('fresh_run') else 'original_manifest_sha256'
    if saved['run_metadata']['input_manifest_sha256'] != provenance[manifest_key]:
        raise ValueError('Original run metadata no longer identifies its historical manifest.')
    return {'questions': len(questions), 'complete_pairs': len(records) // len(config['arms']),
            'saved_answers': len(records), 'hosted_stages': len(bodies), 'hosted_attempts': len(attempts),
            'retained_timed_failures': int(saved['attempts'].status.ne('success').sum()),
            'excluded_preflight_records': len(saved['preflight_failures']),
            'truncated_answers': sum(r['finish_reason'] == 'length' for r in records.values()),
            'sample_order_verified': True, 'exact_contexts_and_requests_verified': True,
            'timing_tables_and_paired_deltas_unchanged': True, 'model_calls': 0}
