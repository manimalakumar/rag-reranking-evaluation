"""Sampling and replay adapters for a frozen stratified generation study."""
import hashlib
import json
import sqlite3
from contextlib import closing

import numpy as np
import pandas as pd
from .io import read_json


def assign_generation_strata(population, strata, tolerance):
    """Split opportunities by recovery outcome and controls by support availability."""
    if len(strata) != 5:
        raise ValueError('Provide the five ordered generation strata.')
    result = pd.Series(index=population.index, dtype='object')
    for label, outcome in zip(strata[:3], ['gain', 'unchanged', 'loss']):
        result.loc[population.cohort.eq('opportunity') & population.outcome.eq(outcome)] = label
    controls = population.cohort.eq('control')
    result.loc[controls & population.embedding_recall_at_5.sub(1).abs().le(tolerance) & population.known_supports_absent_frozen_top30.eq(0)] = strata[3]
    result.loc[controls & population.known_supports_absent_frozen_top30.gt(0)] = strata[4]
    if result.isna().any():
        raise ValueError('Some questions do not fit the configured generation strata.')
    return result


def select_generation_sample(population, strata, questions_per_stratum, seed, tolerance):
    """Select up to the configured cap per stratum, retaining smaller groups whole."""
    if questions_per_stratum < 1:
        raise ValueError('The per-stratum sample cap must be positive.')
    frame = population.copy()
    frame['generation_stratum'] = assign_generation_strata(frame, strata, tolerance)
    parts = []
    for offset, stratum in enumerate(strata):
        pool = frame[frame.generation_stratum.eq(stratum)]
        parts.append(pool.sample(n=min(len(pool), questions_per_stratum), random_state=seed + offset))
    sample = pd.concat(parts).sort_values(['generation_stratum', 'query_id']).reset_index(drop=True)
    if sample.empty:
        raise ValueError('No questions are available in the configured answer-quality subgroups.')
    sample['sample_order'] = np.arange(len(sample))
    return sample


def load_frozen_paired_records(config):
    """Adapt saved attempts without modifying responses, prompts or the source DB."""
    from .opportunity_study import generation_fingerprint
    frozen = {record['query_id']: record for record in
              [json.loads(line) for line in config['frozen_generation_inputs_file'].read_text().splitlines() if line.strip()]}
    original = read_json(config['generation_manifest_file'])['generator_settings']
    settings = config['generation']
    extras = settings['extras']
    actual = {'model': settings['model'], 'endpoint': settings['api_base'] + '/chat/completions',
              'system_prompt': settings['system_prompt'], 'temperature': settings['temperature'],
              'top_p': extras['top_p'], 'seed': extras['seed'], 'max_tokens': settings['max_tokens'],
              'thinking_enabled': False, 'stream': extras['stream']}
    if actual != original:
        raise ValueError('Configured generation settings differ from the frozen historical study.')
    reverse = {source: arm for arm, source in config['source_arm_names'].items()}
    # This checkpoint is frozen and self-contained; immutable mode also prevents
    # SQLite from creating WAL/shared-memory companions during saved replay.
    with closing(sqlite3.connect(f"file:{config['checkpoint_file']}?mode=ro&immutable=1", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        records = []
        for saved in connection.execute('SELECT * FROM generation_attempts ORDER BY attempt_id'):
            saved = dict(saved)
            if saved['arm'] not in reverse or saved['query_id'] not in frozen:
                raise ValueError('Unexpected saved question or arm in the generation checkpoint.')
            row = frozen[saved['query_id']]
            source = row['arms'][saved['arm']]
            messages = source['messages']
            prompt_hash = hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if prompt_hash != source['prompt_sha256']:
                raise ValueError('Frozen prompt checksum changed.')
            arm = reverse[saved['arm']]
            payload = {'model': settings['model'], 'messages': messages, 'temperature': settings['temperature'],
                       'max_tokens': settings['max_tokens'], **extras}
            raw = json.loads(saved['raw_response_json']) if saved['raw_response_json'] else None
            record = {**saved, 'source_arm': saved['arm'], 'arm': arm,
                      'source_status': saved['status'], 'status': 'ok' if saved['status'] == 'success' else 'error',
                      'request_body': payload,
                      'url': original['endpoint'],
                      'request_body_origin': 'Reconstructed from exact frozen messages and saved generation settings; original HTTP request bytes unavailable.',
                      'raw_response_text': saved['raw_response_json'],
                      'raw_response_text_origin': 'Original saved JSON serialization; original HTTP response text unavailable.',
                      'raw_response_json': raw, 'context_passage_ids': source['passage_ids'],
                      'truncated': saved['finish_reason'] == 'length',
                      'fingerprint': generation_fingerprint(row['question'], arm, messages, settings)}
            if record['status'] == 'ok':
                choice = raw['choices'][0]
                if choice['message']['content'] != record['answer'] or choice.get('finish_reason') != record['finish_reason']:
                    raise ValueError('Saved answer differs from its original response.')
            records.append(record)
    return records
