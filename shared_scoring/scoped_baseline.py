"""Seed a new eligible baseline; keep the earlier reviewed baseline untouched."""
from copy import deepcopy
import json
import shutil

import pandas as pd

from .cached_scope import eligible_ids, source_prefix
from .fresh_retrieval import fingerprint, passage_id
from .io import read_json, write_json


def prepare_scoped_baseline(config, project, settings):
    source, _, _ = source_prefix(settings)
    allowed = set(eligible_ids(settings))
    queries = pd.read_parquet(source/'queries.parquet')
    pool = queries[queries.query_id.isin(allowed) & queries.has_known_support_labels & queries.original_answers.map(lambda answers: bool(len(answers)) and all(str(a).strip() for a in answers))]
    seed = settings['scope'].get('baseline_seed', 42)
    count = settings['scope'].get('baseline_questions', 31)
    selected = pool.sample(n=count + 1, random_state=seed)
    baseline_ids = selected.query_id.iloc[:count].tolist()
    diagnostic_id = str(selected.query_id.iloc[count])
    identity = fingerprint({'scope':settings['scope']['eligible_ids_sha256'], 'passages':settings['scope']['passage_count'],
                            'baseline_ids':baseline_ids,'diagnostic_id':diagnostic_id})[:20]
    folder = settings['runtime_directory']/'eligible-baseline-inputs'/identity
    original = config['source_directory']
    input_dir = folder/'input'
    input_dir.mkdir(parents=True,exist_ok=True)
    # Use the cached source Arrow rows; no downloads or embedding calls.
    import pyarrow as pa
    import pyarrow.ipc as ipc
    arrows = list((source/'downloads').rglob('ms_marco-validation.arrow'))
    if len(arrows) != 1:
        raise ValueError('Expected one pinned local validation dataset.')
    distractors = queries[queries.query_id.isin(allowed) & ~queries.query_id.isin(selected.query_id)].sample(n=min(1000,len(queries[queries.query_id.isin(allowed) & ~queries.query_id.isin(selected.query_id)])),random_state=seed)
    wanted = set(baseline_ids + [diagnostic_id] + distractors.query_id.tolist())
    rows = {}
    with pa.memory_map(str(arrows[0]),'r') as stream:
        for batch in ipc.open_stream(stream):
            for row in batch.to_pylist():
                if str(row['query_id']) in wanted:
                    rows[str(row['query_id'])] = row
    reviewed = []
    for qid in baseline_ids:
        row = rows[qid]
        passages = row['passages']
        supports = [(text.strip(),url) for text,url,flag in zip(passages['passage_text'],passages['url'],passages['is_selected']) if flag and text.strip()]
        reviewed.append({'query id':qid,'query':row['query'],'answer':json.dumps(row['answers']),
                         'decision':'keep','selection_method':'Seeded eligible sample; dataset support, human review pending',
                         'why':'Dataset annotated support; newly sampled baseline has not been human reviewed.',
                         'related passages':'\n\n'.join(f'Passage {i}\n{text}\nSource: {url}' for i,(text,url) in enumerate(supports,1))})
    chosen = [rows[qid] for qid in baseline_ids + distractors.query_id.tolist()]
    docs = {}
    for row in chosen:
        for text,url in zip(row['passages']['passage_text'],row['passages']['url']):
            text = text.strip()
            if text:
                key = passage_id(text)
                docs.setdefault(key,dict(id=key,text=text,urls=[]))
                if url not in docs[key]['urls']:
                    docs[key]['urls'].append(url)
    limit = settings['retrieval']['passage_char_budget']
    documents = [dict(d,text=d['text'][:limit],original_char_count=len(d['text']),truncated=len(d['text'])>limit)
                 for d in sorted(docs.values(),key=lambda d:d['id'])]
    pipeline = read_json(original/'config.json')
    pipeline.update(diagnostic_query_id=diagnostic_id,baseline_selection='31 labeled eligible questions sampled with seed 42; human review pending')
    pipeline['models_and_dataset']['corpus_mode'] = 'eligible_questions_plus_1000_eligible_distractors'
    files = {'baseline_ids.json':baseline_ids,'golden_reviewed.json':reviewed,'dataset_rows.json':chosen,
             f'diagnostic_{diagnostic_id}.json':rows[diagnostic_id], 'corpus.json':documents,
             'source_metadata.json':{'Source revision':settings['dataset']['revision'],'selection':'Frozen fully covered pool',
                                     'eligible_question_count':len(allowed),'human_review':'pending'}}
    for name,data in files.items():
        path = input_dir/name
        if path.exists() and read_json(path) != data:
            raise ValueError('Frozen eligible baseline input changed.')
        if not path.exists():
            write_json(path,data)
    for name in ['experiment_helpers.py','scoring.py']:
        if not (folder/name).exists():
            shutil.copy2(original/name,folder/name)
    write_json(folder/'config.json',pipeline)
    write_json(folder/'selection_manifest.json',dict(baseline_ids=baseline_ids,diagnostic_id=diagnostic_id,
               eligible_question_count=len(allowed),seed=42,human_review='pending',pipeline_source_directory=str(original)))
    result = deepcopy(config)
    result.update(source_directory=folder,diagnostic_ids=[diagnostic_id],inspect_query_id=baseline_ids[0],
                  eligible_baseline=True, selection_manifest=folder/'selection_manifest.json')
    return result
