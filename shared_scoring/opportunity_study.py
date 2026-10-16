"""Frozen large-cohort study: replay is read-only; inference is explicit."""
import asyncio
import hashlib
import json
import math
import os
import inspect
import shutil
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from .budget import reserve_request
from .answers import load_squad_metric, score_custom_answer, score_squad_answers
from .io import read_json, write_json, sha256_file
from .records import load_saved_judge_scores, combine_answer_scores
from .summaries import summarize_paired_scores


def load_opportunity_configuration(path):
    """Resolve the study's input, checkpoint and scoring paths portably."""
    path = Path(path).resolve()
    config = read_json(path)
    config['configuration_file'] = path
    config['study_directory'] = path.parent
    for key in ['questions_file', 'passages_file', 'checkpoint_file', 'score_directory', 'retrieval_summary_file']:
        config[key] = (path.parent / config[key]).resolve()
    for key in ['parent_configuration', 'frozen_generation_inputs_file', 'generation_manifest_file',
                'parent_provenance_file', 'manual_review_file', 'input_verification_file',
                'latency_source_directory', 'latency_plot_file', 'latency_provenance_file', 'artifact_directory',
                'eligible_question_ids_file']:
        if config.get(key):
            config[key] = (path.parent / config[key]).resolve()
    for key, value in config.get('latency_files', {}).items():
        config['latency_files'][key] = (path.parent / value).resolve()
    metric_source = config['squad']['metric_source']
    if metric_source.endswith('.py') or '/' in metric_source:
        config['squad']['metric_source'] = (path.parent / metric_source).resolve()
    config['squad']['cache_directory'] = (path.parent / config['squad']['cache_directory']).resolve()
    return config


def load_opportunity_population(config):
    """Read all fixed cohorts; reject changed counts, duplicate IDs or diagnostics."""
    if config.get('checkpoint_format') == 'frozen_latency':
        from .latency_study import load_latency_population
        return load_latency_population(config)
    frame = pd.read_parquet(config['questions_file'])
    frame['query_id'] = frame.query_id.astype(str)
    if config.get('eligible_question_ids_file'):
        allowed = set(Path(config['eligible_question_ids_file']).read_text().splitlines())
        if not set(frame.query_id) <= allowed:
            raise ValueError('Study contains questions outside the fully covered eligible pool.')
    counts = frame.cohort.value_counts().to_dict()
    if counts != config['expected_cohort_counts'] or frame.query_id.duplicated().any():
        raise ValueError(f'Opportunity cohort membership/counts changed: {counts}')
    if set(frame.query_id) & set(config['diagnostic_ids']):
        raise ValueError('Diagnostic question must not enter the opportunity/control aggregates.')
    gap = frame.embedding_recall_at_30 - frame.embedding_recall_at_5
    tolerance = config['summary']['tie_tolerance']
    if not gap[frame.cohort.eq('opportunity')].gt(tolerance).all() or not gap[frame.cohort.eq('control')].abs().le(tolerance).all():
        raise ValueError('Cohorts do not match the pre-reranking opportunity definition.')
    return frame


def load_frozen_contexts(config, query_ids=None):
    """Load each arm's five saved passages; never concatenate arm contexts."""
    if config.get('checkpoint_format') == 'frozen_latency':
        from .latency_study import load_latency_contexts
        return load_latency_contexts(config, query_ids)
    filters = [('query_id', 'in', list(query_ids))] if query_ids is not None else None
    data = pd.read_parquet(config['passages_file'], filters=filters)
    output = {}
    for arm, rank in config['rank_columns'].items():
        selected = data[data[rank].le(config['context_passage_count'])].sort_values(['query_id', rank])
        for qid, rows in selected.groupby('query_id', sort=False):
            ids = rows.passage_id.tolist()
            if len(ids) != config['context_passage_count'] or len(set(ids)) != len(ids):
                raise ValueError(f'{qid}/{arm}: context must contain five distinct saved passages.')
            blocks = [f"[{row.passage_id}]\n{row.text}" for row in rows.itertuples()]
            output[(str(qid), arm)] = {'context': '\n\n'.join(blocks), 'retrieved_contexts': blocks,
                'context_passage_ids': ids, 'passages': rows.to_dict('records')}
    return output


def generation_messages(question, context, system_prompt):
    """Use the baseline evidence-only prompt with the arm's exact context."""
    return [{'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f'Question: {question}\n\nPassages:\n{context}\n\nAnswer:'}]


def generation_fingerprint(question, arm, messages, settings):
    """Bind immutable answers to question, arm, prompt, endpoint and settings."""
    value = {'question': question, 'arm': arm, 'messages': messages, 'settings': settings}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_generation_records(config):
    """Read the existing attempt ledger; absence stays explicit and creates no file."""
    path = config['checkpoint_file']
    if not path.exists():
        return []
    if config.get('checkpoint_format') == 'frozen_latency':
        from .latency_study import load_latency_generation_records
        return load_latency_generation_records(config)
    if config.get('checkpoint_format') == 'frozen_paired_generation':
        from .paired_generation import load_frozen_paired_records
        return load_frozen_paired_records(config)
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as connection:
        records = [(row[0], json.loads(row[1])) for row in connection.execute('SELECT attempt_id, record FROM generation_attempts ORDER BY attempt_id')]
    if not config.get('fresh_run'):
        return [record for _, record in records]
    completed = {(record['query_id'], record['arm'], record.get('started_utc'))
                 for _, record in records if record['status'] != 'started'}
    records = [(number, record) for number, record in records
               if record['status'] != 'started' or
               (record['query_id'], record['arm'], record.get('started_utc')) not in completed]
    # Fresh native records use the same inspectable teaching views as imported answers.
    return [{**record, 'attempt_id': number, 'source_arm': record['arm'],
             'source_status': record['status'], 'error': record.get('error'),
             'execution_workers': config['generation']['concurrency'],
             'elapsed_seconds': record.get('elapsed_seconds'),
             'finish_reason': record.get('finish_reason'), 'truncated': record.get('truncated', False),
             'request_body_origin': 'Exact hosted request body',
             'raw_response_text_origin': 'Exact recorded HTTP response text'} for number, record in records]


def validate_frozen_opportunity_inputs(config):
    """Verify all frozen rank permutations and reproduce annotated recall."""
    if config.get('checkpoint_format') == 'frozen_latency':
        from .latency_study import validate_saved_latency_study
        return validate_saved_latency_study(config)
    from .retrieval import score_retrieval
    artifact_directory=config.get('artifact_directory',config['study_directory'])
    for relative, expected in read_json(artifact_directory / 'input/file_checksums.json').items():
        if sha256_file(artifact_directory / relative) != expected:
            raise ValueError(f'Frozen input bytes changed: {relative}')
    population = load_opportunity_population(config)
    candidates = pd.read_parquet(config['passages_file'], filters=[('query_id', 'in', population.query_id.tolist())])
    candidates['query_id'] = candidates.query_id.astype(str)
    if set(candidates.query_id) != set(population.query_id):
        raise ValueError('Candidate rows do not cover the fixed question population.')
    expected = set(range(1, config['candidate_count'] + 1))
    questions = population.set_index('query_id')
    tolerance = config['summary']['tie_tolerance']
    for qid, group in candidates.groupby('query_id', sort=False):
        if len(group) != config['candidate_count'] or group.passage_id.nunique() != len(group):
            raise ValueError(f'{qid}: duplicate or missing candidate passages.')
        for rank in config['rank_columns'].values():
            if set(group[rank]) != expected:
                raise ValueError(f'{qid}: incomplete saved rank permutation.')
        if not all(math.isfinite(value) for value in group.rerank_score) or not all(math.isfinite(value) for value in group.similarity_score):
            raise ValueError(f'{qid}: nonfinite saved ranking scores.')
        row = questions.loc[qid]
        gold = set(row.known_supporting_passage_ids)
        if set(group.loc[group.is_known_support, 'passage_id']) != gold & set(group.passage_id):
            raise ValueError(f'{qid}: saved support flags differ from annotated support IDs.')
        for arm, rank in config['rank_columns'].items():
            ids = group.sort_values(rank).passage_id.tolist()
            actual = score_retrieval(ids[:config['context_passage_count']], gold)['recall']
            if abs(actual - row[config['recall_columns'][arm]]) > tolerance:
                raise ValueError(f'{qid}/{arm}: changed annotated-support Recall@5.')
        if abs(score_retrieval(group.passage_id.tolist(), gold)['recall'] - row.embedding_recall_at_30) > tolerance:
            raise ValueError(f'{qid}: changed annotated-support Recall@30.')
    return {'questions': len(population), 'candidate_rows': len(candidates),
            'cohort_counts': population.cohort.value_counts().to_dict(),
            'rank_permutations_verified': True, 'recall_unchanged': True, 'model_calls': 0}


def current_generation_records(records):
    """Keep the first successful response, otherwise the latest failed attempt."""
    current = {}
    for record in records:
        key = (record['query_id'], record['arm'])
        if key not in current or current[key]['status'] != 'ok':
            current[key] = record
    return current


def refresh_opportunity_results(config):
    """Reload saved answers and scores without creating outputs or model clients."""
    records = load_generation_records(config)
    inputs, complete_ids, failures = build_saved_opportunity_inputs(config)
    return {'records': records, 'current': current_generation_records(records),
            'inputs': inputs, 'complete_ids': complete_ids, 'failures': failures,
            'standard': load_opportunity_scores(config)}


async def run_opportunity_generation(config, query_ids=None, allow_model_calls=False, retry_failures=False):
    """Explicitly generate missing answers; archive attempts and preserve successes."""
    if not allow_model_calls:
        raise PermissionError('Generation is disabled; explicitly set allow_model_calls=True.')
    if config.get('checkpoint_format') in ['frozen_paired_generation', 'frozen_latency']:
        raise PermissionError('Imported paired answers are immutable. Use a separate native checkpoint for any new generation run.')
    import httpx
    settings = config['generation']
    key = os.environ.get(settings['credential_env'])
    if not key:
        raise RuntimeError(f"Missing environment variable {settings['credential_env']}")
    population = load_opportunity_population(config)
    if query_ids is not None:
        if not set(query_ids) <= set(population.query_id):
            raise ValueError('Requested generation IDs are outside the fixed cohort.')
        population = population[population.query_id.isin(query_ids)]
    contexts = load_frozen_contexts(config, population.query_id.tolist())
    saved = current_generation_records(load_generation_records(config))
    config['checkpoint_file'].parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(config['checkpoint_file'])
    connection.execute('CREATE TABLE IF NOT EXISTS generation_attempts (attempt_id INTEGER PRIMARY KEY, record TEXT NOT NULL)')
    lock, semaphore = asyncio.Lock(), asyncio.Semaphore(settings['concurrency'])
    next_request = 0.0
    def store(record):
        connection.execute('INSERT INTO generation_attempts(record) VALUES (?)', (json.dumps(record, ensure_ascii=False, allow_nan=False),))
        connection.commit()
    async with httpx.AsyncClient(timeout=settings['timeout_seconds']) as client:
        async def generate(row, arm):
            nonlocal next_request
            context = contexts[(row.query_id, arm)]
            messages = generation_messages(row.question, context['context'], settings['system_prompt'])
            fingerprint = generation_fingerprint(row.question, arm, messages, settings)
            previous = saved.get((row.query_id, arm))
            if previous:
                if previous['fingerprint'] != fingerprint:
                    raise ValueError('Generation inputs/settings changed; use a separate output configuration.')
                if previous['status'] == 'ok' or not retry_failures:
                    return
            async with semaphore:
                async with lock:
                    wait = max(0., next_request - time.monotonic())
                    if wait:
                        await asyncio.sleep(wait)
                    next_request = time.monotonic() + settings['request_min_interval_seconds']
                payload = {'model': settings['model'], 'messages': messages,
                           'temperature': settings['temperature'], 'max_tokens': settings['max_tokens'], **settings['extras']}
                record = {'query_id': row.query_id, 'arm': arm, 'fingerprint': fingerprint,
                          'context_passage_ids': context['context_passage_ids'], 'request_body': payload,
                          'url': settings['api_base'] + '/chat/completions', 'status': 'started',
                          'started_utc': datetime.now(timezone.utc).isoformat()}
                reserve_request(record['url'], payload, [config['run_id'], row.query_id, arm], settings.get('max_total_attempts', 2))
                store(record.copy())
                start = time.perf_counter()
                try:
                    from .provider import require_model_calls
                    require_model_calls()
                    response = await client.post(record['url'], headers={'Authorization': 'Bearer ' + key}, json=payload)
                    record.update(http_status=response.status_code, raw_response_text=response.text)
                    try:
                        record['raw_response_json'] = response.json()
                    except ValueError:
                        pass
                    response.raise_for_status()
                    choice = record['raw_response_json']['choices'][0]
                    answer = choice['message']['content']
                    if not isinstance(answer, str):
                        raise ValueError('Generation response has no string answer.')
                    record.update(status='ok', answer=answer, finish_reason=choice.get('finish_reason'),
                                  truncated=choice.get('finish_reason') == 'length')
                except asyncio.CancelledError:
                    record.update(status='interrupted', error='Generation interrupted; inspect before explicit retry.')
                    raise
                except Exception as error:
                    record.update(status='error', error=f'{type(error).__name__}: {error}')
                finally:
                    record.update(elapsed_seconds=time.perf_counter() - start,
                                  ended_utc=datetime.now(timezone.utc).isoformat())
                    store(record)
        tasks = []
        for position, row in enumerate(population.itertuples()):
            arms = list(config['arms'])
            if position % 2:
                arms.reverse()
            tasks.extend(generate(row, arm) for arm in arms)
        try:
            await asyncio.gather(*tasks)
        finally:
            connection.close()
    return load_generation_records(config)


def build_saved_opportunity_inputs(config):
    """Verify saved responses; return only complete pairs plus all failure records."""
    if config.get('checkpoint_format') == 'frozen_latency':
        from .latency_study import build_latency_evaluation_inputs
        return build_latency_evaluation_inputs(config)
    population = load_opportunity_population(config)
    records = current_generation_records(load_generation_records(config))
    complete = [qid for qid in population.query_id if all(records.get((qid, arm), {}).get('status') == 'ok' for arm in config['arms'])]
    contexts = load_frozen_contexts(config, complete) if complete else {}
    inputs = []
    for row in population[population.query_id.isin(complete)].itertuples():
        for arm, label in config['arms'].items():
            context, record = contexts[(row.query_id, arm)], records[(row.query_id, arm)]
            messages = generation_messages(row.question, context['context'], config['generation']['system_prompt'])
            if record['fingerprint'] != generation_fingerprint(row.question, arm, messages, config['generation']):
                raise ValueError(f'{row.query_id}/{arm}: saved generation inputs/settings differ.')
            choice = record['raw_response_json']['choices'][0]
            expected_payload = {'model': config['generation']['model'], 'messages': messages,
                                'temperature': config['generation']['temperature'],
                                'max_tokens': config['generation']['max_tokens'], **config['generation']['extras']}
            if record['request_body'] != expected_payload or record['url'] != config['generation']['api_base'] + '/chat/completions' or record['context_passage_ids'] != context['context_passage_ids'] or record['answer'] != choice['message']['content'] or record['finish_reason'] != choice.get('finish_reason') or record['truncated'] != (choice.get('finish_reason') == 'length'):
                raise ValueError(f'{row.query_id}/{arm}: altered saved answer or context.')
            answer = record['answer']
            references = list(getattr(row, config.get('reference_answers_column', 'reference_answers')))
            old = score_custom_answer(answer, references, config['legacy_citation_pattern'])
            inputs.append({'query_id': row.query_id, 'question': row.question, 'arm': arm, 'path': label,
                           'answer': answer, 'reference_answers': references, 'context': context['context'],
                           'retrieved_contexts': context['retrieved_contexts'], 'context_passage_ids': context['context_passage_ids'],
                           'finish_reason': record['finish_reason'], 'truncated': record['truncated'],
                           'annotated_support_recall_at_5': getattr(row, config['recall_columns'][arm]),
                           'existing_token_f1_0_1': old['token_f1'], 'existing_normalized_em_0_1': old['normalized_em'],
                           'generation_fingerprint': record['fingerprint']})
    failures = [record for record in records.values() if record['status'] != 'ok']
    return inputs, complete, failures


def _squad_scoring_settings(config):
    """Describe effective scoring settings without embedding workstation paths."""
    source = config['squad']['metric_source']
    if isinstance(source, Path):
        source = os.path.relpath(source, config['configuration_file'].parent)
    return {'metric_source': source, 'citation_pattern': config['citation_pattern'],
            'legacy_citation_pattern': config['legacy_citation_pattern']}


def _verify_saved_squad(config, directory):
    """Verify current provenance, retaining compatibility with earlier saved scores."""
    prior = read_json(directory / 'squad_provenance.json')
    expected = _squad_scoring_settings(config)
    configured_source = config['squad']['metric_source']

    def same_metric_source(recorded_source):
        """Accept a moved local implementation only when its saved checksum matches."""
        return (recorded_source == expected['metric_source'] or
                (isinstance(configured_source, Path) and prior.get('metric_source_sha256') and
                 sha256_file(configured_source) == prior['metric_source_sha256']))

    if 'scoring_settings' in prior:
        recorded = dict(prior['scoring_settings'])
        recorded_source = recorded.pop('metric_source')
        current = {key: value for key, value in expected.items() if key != 'metric_source'}
        if recorded != current or not same_metric_source(recorded_source):
            raise ValueError('Saved SQuAD settings changed; use a fresh score directory.')
    else:
        # Earlier v3 provenance predates CSV hashes; preserve it rather than
        # rewriting its settings or scores during display.
        source = prior['settings']['metric_source']
        if Path(source).is_absolute():
            source = os.path.relpath(source, config['configuration_file'].parent)
        if not same_metric_source(source) or prior['citation_pattern'] != expected['citation_pattern']:
            raise ValueError('Saved SQuAD settings changed; use a fresh score directory.')
        if isinstance(config['squad']['metric_source'], Path) and prior['metric_source_sha256'] != sha256_file(config['squad']['metric_source']):
            raise ValueError('Saved SQuAD implementation bytes changed.')
    if prior.get('score_sha256') and prior['score_sha256'] != sha256_file(directory / 'squad_per_answer.csv'):
        raise ValueError('Saved SQuAD score bytes changed; inspect the recorded output.')
    if prior.get('configured_metric_source_sha256') and prior['configured_metric_source_sha256'] != sha256_file(config['squad']['metric_source']):
        raise ValueError('Configured SQuAD implementation bytes changed.')
    for name, expected_hash in prior.get('archived_metric_sources_sha256', {}).items():
        if sha256_file(directory / 'squad_metric_sources' / name) != expected_hash:
            raise ValueError('Archived SQuAD implementation bytes changed.')
        if isinstance(configured_source, Path) and sha256_file(configured_source.parent / name) != expected_hash:
            raise ValueError('Configured SQuAD implementation bytes changed.')
    if prior['input_sha256'] != sha256_file(directory / 'evaluation_inputs.json'):
        raise ValueError('Saved SQuAD input bytes changed.')


def run_opportunity_squad(config):
    """Explicit SQuAD action; exact prior inputs may replay, changed inputs fail."""
    inputs, ids, failures = build_saved_opportunity_inputs(config)
    if not inputs:
        raise ValueError('No complete saved answer pairs. Generate answers explicitly before scoring.')
    output = config['score_directory']
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'evaluation_inputs.json'
    if path.exists() and read_json(path) != inputs:
        raise ValueError('Evaluation inputs changed; use a fresh score directory.')
    score_path = output / 'squad_per_answer.csv'
    if score_path.exists():
        _verify_saved_squad(config, output)
        return pd.read_csv(score_path, dtype={'query_id': str})
    metric = load_squad_metric(config['squad']['metric_source'], config['squad']['cache_directory'])
    scores = score_squad_answers(inputs, metric, config['citation_pattern'], config['legacy_citation_pattern'])
    write_json(path, inputs)
    write_json(output / 'input_audit.json', {'complete_pair_ids': ids, 'failures': failures})
    from importlib.metadata import version
    loaded_source = Path(inspect.getfile(type(metric)))
    archived_sources = output / 'squad_metric_sources'
    archived_sources.mkdir(exist_ok=True)
    source_hashes = {}
    for source in loaded_source.parent.glob('*.py'):
        target = archived_sources / source.name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        source_hashes[source.name] = sha256_file(target)
    scores.to_csv(score_path, index=False)
    portable_settings = _squad_scoring_settings(config)
    write_json(output / 'squad_provenance.json', {'evaluate_version': version('evaluate'),
               'datasets_version': version('datasets'),
               'metric_source_sha256': sha256_file(loaded_source),
               'configured_metric_source_sha256': sha256_file(config['squad']['metric_source']) if isinstance(config['squad']['metric_source'], Path) else None,
               'archived_metric_sources_sha256': source_hashes,
               'input_sha256': sha256_file(path), 'settings': {
                   **read_json(config['configuration_file'])['squad'],
                   'metric_source': portable_settings['metric_source']},
               'scoring_settings': portable_settings, 'score_sha256': sha256_file(score_path),
               'settings_path_base': 'configuration file directory',
               'citation_pattern': config['citation_pattern'], 'multiple_references': 'all saved reference answers'})
    return scores


async def run_opportunity_judge(config, allow_model_calls=False, retry_failures=False):
    """Judge the exact scored pairs; reject changed inputs before making calls."""
    if not allow_model_calls:
        raise PermissionError('Judge calls are disabled; explicitly enable hosted evaluation.')
    inputs, _, _ = build_saved_opportunity_inputs(config)
    path = config['score_directory'] / 'evaluation_inputs.json'
    if not inputs or not path.is_file():
        raise ValueError('Run HF SQuAD scoring on complete saved answer pairs first.')
    if read_json(path) != inputs:
        raise ValueError('Evaluation pairs changed; use a fresh score directory and score it first.')
    _verify_saved_squad(config, config['score_directory'])
    from .judge import run_ragas_evaluation
    return await run_ragas_evaluation(inputs, config['study_directory'], config['run_id'],
        config['score_directory'], config['judge'], allow_model_calls, retry_failures)


def load_opportunity_scores(config):
    """Read existing standard scores without generating, embedding or judging."""
    directory = config['score_directory']
    if not (directory / 'squad_per_answer.csv').exists():
        return {'status': 'not_scored', 'scores': pd.DataFrame(), 'summary': pd.DataFrame(), 'pairs': {}}
    inputs, _, _ = build_saved_opportunity_inputs(config)
    if read_json(directory / 'evaluation_inputs.json') != inputs:
        raise ValueError('Saved scores do not match the current immutable answer pairs.')
    _verify_saved_squad(config, directory)
    squad = pd.read_csv(directory / 'squad_per_answer.csv', dtype={'query_id': str})
    expected = {(row['query_id'], row['arm']) for row in inputs}
    if set(zip(squad.query_id, squad.arm)) != expected or len(squad) != len(expected):
        raise ValueError('Saved SQuAD rows do not match the complete answer pairs.')
    if (directory / 'judge_manifest.json').exists():
        judge = load_saved_judge_scores(directory, inputs, config['judge']['metrics'], config['judge'])
    else:
        judge = pd.DataFrame(columns=['query_id', 'arm', 'metric', 'score', 'status', 'elapsed_seconds'])
    scores = combine_answer_scores(squad, judge, config['judge']['metrics'])
    population = load_opportunity_population(config)
    group_column = config.get('summary_group_column', 'cohort')
    columns = list(dict.fromkeys(['query_id', 'cohort', 'batch_id', group_column]))
    scores = scores.merge(population[columns], on='query_id', validate='many_to_one')
    scores['is_baseline_diagnostic'] = scores.query_id.isin(config.get('baseline_diagnostic_ids', []))
    summary, pairs = summarize_opportunity_scores(scores, config)
    return {'status': 'saved_scores', 'scores': scores, 'summary': summary, 'pairs': pairs}


def summarize_opportunity_scores(scores, config):
    """Compare configured groups, pairing successful answers independently per metric."""
    group_column = config.get('summary_group_column', 'cohort')
    summaries, pairs = [], {}
    settings = config['summary']
    groups = config.get('summary_groups', list(config['expected_cohort_counts']))
    if config.get('include_overall_summary', True):
        groups = ['all', *groups]
    for cohort in groups:
        subset = scores if cohort == 'all' else scores[scores[group_column].eq(cohort)]
        table, metric_pairs = summarize_paired_scores(subset, config['arms'], settings['metrics'],
            settings['bootstrap_samples'], settings['bootstrap_seed'], settings['tie_tolerance'],
            settings['delta_label'], settings['confidence_quantiles'], settings['bootstrap_chunk_size'])
        table.insert(0, group_column, cohort)
        summaries.append(table);pairs[cohort] = metric_pairs
    return pd.concat(summaries, ignore_index=True), pairs
