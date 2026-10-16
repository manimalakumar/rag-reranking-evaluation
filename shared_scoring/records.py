"""Load and verify saved inputs/results. No inference clients are created here."""
import math
from pathlib import Path
import pandas as pd
from .io import read_json, sha256_file
from .retrieval import score_retrieval
from .provenance import evaluation_fingerprint
from .answers import remove_passage_citations


def load_saved_inputs(config):
    """Verify exact saved answers, contexts, references, requests, and pair IDs."""
    source, directory = Path(config['source_directory']), Path(config['score_directory'])
    run = source / 'runs' / config['run_id']
    manifest = read_json(run / 'run_manifest.json')
    identity = read_json(source / 'batch_identity_v1.json')
    if not (identity['run_id'] == manifest['run_id'] == config['run_id']):
        raise ValueError('Configured run ID does not match the saved run identity and manifest.')
    questions = read_json(run / 'questions.json')
    results = read_json(run / 'results.json')
    qmap = {str(row['query_id']): row for row in questions}
    rmap = {str(row['query_id']): row for row in results}
    if len(qmap) != len(questions) or len(rmap) != len(results) or set(qmap) != set(rmap):
        raise ValueError('Saved questions and results must contain the same unique question IDs.')
    baseline_ids = [str(identifier) for identifier in read_json(run / config['baseline_ids_file'])]
    selected = {identifier for identifier, q in qmap.items() if q['scope'] == config['baseline_scope']}
    if set(baseline_ids) != selected or len(set(baseline_ids)) != len(baseline_ids):
        raise ValueError('Baseline IDs do not match the configured baseline question scope.')
    if set(config['diagnostic_ids']) & selected:
        raise ValueError('Diagnostic questions must be separate from baseline IDs.')
    eligible = [identifier for identifier in baseline_ids
                if all(rmap[identifier]['arms'].get(arm, {}).get('status') == 'ok' for arm in config['arms'])]
    baseline = read_json(directory / 'evaluation_inputs.json')
    diagnostic = read_json(directory / 'diagnostic_inputs.json')
    for rows, ids in [(baseline, eligible), (diagnostic, config['diagnostic_ids'])]:
        expected = {(identifier, arm) for identifier in ids for arm in config['arms']
                    if rmap.get(identifier, {}).get('arms', {}).get(arm, {}).get('status') == 'ok'}
        keys = {(str(row['query_id']), row['arm']) for row in rows}
        if keys != expected or len(rows) != len(keys):
            raise ValueError('Saved evaluation inputs have missing, extra, or duplicate question/arm pairs.')
        for row in rows:
            identifier, arm = str(row['query_id']), row['arm']
            question, answer = qmap[identifier], rmap[identifier]['arms'][arm]
            request_path = run / 'requests' / f'{identifier}-{arm}-generation.json'
            context_path = run / 'contexts_prompts' / f'{identifier}-{arm}.json'
            request, context = read_json(request_path), read_json(context_path)
            choice = request['raw_response_json']['choices'][0]
            passages = answer['top_five']
            blocks = [f"[{passage['id']}]\n{passage['text']}" for passage in passages]
            checks = [request['status'] == 'ok', row['answer'] == answer['answer'] == choice['message']['content'],
                      row['question'] == question['query'], row['reference_answers'] == question['answers'],
                      row['path'] == config['arms'][arm],
                      row['context'] == answer['context'] == context['context'] == '\n\n'.join(blocks),
                      row['retrieved_contexts'] == blocks, row['context_passage_ids'] == [p['id'] for p in passages],
                      answer['messages'] == context['messages'] == request['request_body']['messages'],
                      row['finish_reason'] == answer.get('finish_reason') == choice.get('finish_reason'),
                      row['truncated'] == (answer.get('finish_reason') == 'length'),
                      len(passages) == len({p['id'] for p in passages}) == config['context_passage_count'],
                      row['generation_request_sha256'] == sha256_file(request_path),
                      row['context_record_sha256'] == sha256_file(context_path),
                      row['squad_answer'] == remove_passage_citations(row['answer'], config['citation_pattern'])]
            recall = score_retrieval(row['context_passage_ids'], question['gold_ids'])['recall']
            if not all(checks) or abs(recall - row['annotated_support_recall_at_5']) > config['summary']['tie_tolerance']:
                raise ValueError(f'Changed saved evaluation input or provenance: {identifier} / {arm}')
    failures = read_json(directory / 'original_run_failures.json')
    actual = {(identifier, arm) for identifier, result in rmap.items() for arm, value in result['arms'].items()
              if value['status'] != 'ok'}
    if {(row['query_id'], row['arm']) for row in failures} != actual:
        raise ValueError('Saved original failure records do not match original result statuses.')
    audit = read_json(directory / 'input_audit.json')
    if audit['complete_baseline_ids'] != eligible or audit['complete_baseline_pairs'] != len(eligible):
        raise ValueError('Saved scoring eligibility differs from the original complete baseline pairs.')
    return baseline, diagnostic, failures, audit


def load_saved_judge_scores(directory, inputs, metric_names, settings=None):
    """Replay recorded RAGAS outcomes, preserving failures and partial coverage."""
    directory = Path(directory)
    keys = {(row['query_id'], row['arm']): row for row in inputs}
    records = [read_json(path) for path in sorted((directory / 'ragas_results').glob('*.json'))]
    if not records:
        raise FileNotFoundError('No saved RAGAS scores. Run judge evaluation explicitly before replaying it.')
    manifest = read_json(directory / 'judge_manifest.json')
    seen = set()
    for record in records:
        key = (record['query_id'], record['arm'])
        full_key = (*key, record['metric'])
        if full_key in seen or key not in keys or record['metric'] not in metric_names:
            raise ValueError(f'Unexpected or duplicate saved judge outcome: {full_key}')
        seen.add(full_key)
        source = keys[key]
        if settings is not None and record['fingerprint'] != evaluation_fingerprint(source, record['metric'], settings, manifest['packages']['ragas']):
            raise ValueError(f'Judge inputs/model settings differ from saved fingerprint: {full_key}')
        if record['context_passage_ids'] != source['context_passage_ids'] or record['truncated'] != source['truncated']:
            raise ValueError(f'Saved judge provenance changed for {full_key}.')
        if record['status'] == 'started':
            raise ValueError(f'Unfinished judge attempt: {full_key}. Inspect the logs; replay never retries it.')
        if record['status'] == 'ok' and not math.isfinite(record['score']):
            raise ValueError(f'Saved successful judge score is non-finite: {full_key}.')
        if record['status'] != 'ok' and record.get('score') is not None:
            raise ValueError(f'Failed judge score must remain null: {full_key}.')
    return pd.DataFrame(records)


def combine_answer_scores(squad_scores, judge_scores, metric_names):
    """Join per-answer SQuAD and judge outcomes without filling failures with zero."""
    result = squad_scores.copy()
    for name in metric_names:
        subset = judge_scores[judge_scores.metric.eq(name)][['query_id', 'arm', 'score', 'status', 'elapsed_seconds']]
        subset = subset.rename(columns={'score': name + '_0_1', 'status': name + '_status', 'elapsed_seconds': name + '_seconds'})
        result = result.merge(subset, on=['query_id', 'arm'], how='left', validate='one_to_one')
        result[name + '_0_1'] = pd.to_numeric(result[name + '_0_1'], errors='raise').astype(float)
        result[name + '_seconds'] = pd.to_numeric(result[name + '_seconds'], errors='raise').astype(float)
        result[name + '_status'] = result[name + '_status'].fillna('missing').astype('str')
    return result


def load_saved_scores(config):
    """Load and validate saved score tables only; never initialize frameworks."""
    inputs, diagnostic, failures, audit = load_saved_inputs(config)
    directory = Path(config['score_directory'])
    frame = pd.read_csv(directory / 'per_question_scores.csv', dtype={'query_id': str})
    expected = {(row['query_id'], row['arm']) for row in inputs}
    if set(zip(frame.query_id, frame.arm)) != expected or len(frame) != len(expected):
        raise ValueError('Saved score table no longer matches the complete baseline pairs.')
    if config.get('fresh_run') and not (directory / 'judge_manifest.json').is_file():
        judge = pd.DataFrame(columns=['query_id', 'arm', 'metric', 'score', 'status', 'elapsed_seconds'])
    else:
        judge = load_saved_judge_scores(directory, inputs, config['judge']['metrics'], config['judge'])
    squad = pd.read_csv(directory / 'squad_per_answer.csv', dtype={'query_id': str})
    reconstructed = combine_answer_scores(squad, judge, config['judge']['metrics'])
    pd.testing.assert_frame_equal(frame, reconstructed[frame.columns], check_exact=False, rtol=0, atol=config['summary']['tie_tolerance'])
    return {'scores': frame, 'summary': pd.read_csv(directory / 'paired_comparison.csv'), 'inputs': inputs,
            'diagnostic_inputs': diagnostic, 'diagnostic_scores': pd.read_csv(directory / 'diagnostic_squad.csv', dtype={'query_id': str}),
            'original_failures': failures, 'judge_scores': judge, 'audit': audit,
            'judge_failures': judge[judge.status.ne('ok')].copy(),
            'previous_attempts': pd.read_csv(directory / 'evaluation_attempt_history.csv')}
