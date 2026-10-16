"""Frozen eligible questions and retrieval assets from already embedded passages.

This module has no inference client and cannot create passage embeddings.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_json, write_json, sha256_file
from .matrix_storage import load_matrix, npy_sha256, write_reference


def eligible_ids(settings):
    scope = settings['scope']
    path = Path(scope['eligible_question_ids'])
    if sha256_file(path) != scope['eligible_ids_sha256']:
        raise ValueError('The frozen eligible question list changed.')
    ids = path.read_text().splitlines()
    if len(ids) != scope['eligible_question_count'] or len(set(ids)) != len(ids):
        raise ValueError('Eligible question count or uniqueness changed.')
    return ids


def assert_eligible(frame, settings):
    if settings.get('scope'):
        outside = set(frame.query_id.astype(str)) - set(eligible_ids(settings))
        if outside:
            raise ValueError(f'Questions outside the frozen eligible pool: {sorted(outside)[:10]}')


def source_prefix(settings):
    scope = settings['scope']
    source = Path(scope['source_directory'])
    count = scope['passage_count']
    batches = sorted((read_json(p) for p in (source/'embedding_batches').glob('*.json')), key=lambda b:b['start'])
    end = 0
    for batch in batches:
        if end >= count:
            break
        if batch['start'] != end or batch['stop'] <= end or batch['stop'] > count:
            raise ValueError('The frozen passage prefix is not fully checkpointed.')
        end = batch['stop']
    if end != count:
        raise ValueError('The frozen passage prefix is not fully checkpointed.')
    matrix = np.load(source/'embeddings.npy', mmap_mode='r')
    if matrix.shape[0] < count or matrix.shape[1] != settings['retrieval']['dimensions']:
        raise ValueError('Cached embedding matrix dimensions differ from the scope.')
    manifest = read_json(source/'assets_manifest.json')
    original = manifest['contract']['retrieval']
    for key in ['embedding_model', 'dimensions', 'passage_char_budget']:
        if original[key] != settings['retrieval'][key]:
            raise ValueError(f'Cached passage embedding settings differ: {key}')
    return source, count, matrix


def prepare_cached_dataset(settings, directory):
    directory = Path(directory)
    source, count, _ = source_prefix(settings)
    ids = eligible_ids(settings)
    scope = settings['scope']
    if sha256_file(scope['coverage_file']) != scope['coverage_sha256']:
        raise ValueError('Frozen question-to-passage coverage changed.')
    corpus = pd.read_parquet(source/'corpus.parquet').iloc[:count].copy()
    embedded_ids = set(corpus.passage_id)
    # Check ALL associated passages, not only annotated support passages.
    import json
    verified = set()
    wanted = set(ids)
    with Path(scope['coverage_file']).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['query_id'] in wanted:
                if not row['all_associated_passages_embedded'] or row['not_yet_embedded_passage_ids'] or not set(row['embedded_passage_ids']) <= embedded_ids:
                    raise ValueError('An eligible question has unembedded associated passages.')
                verified.add(row['query_id'])
    if verified != wanted:
        raise ValueError('Some eligible questions lack verified passage coverage.')
    queries = pd.read_parquet(source/'queries.parquet')
    queries = queries[queries.query_id.isin(wanted)].sort_values('query_row').reset_index(drop=True)
    if len(queries) != len(ids):
        raise ValueError('Some eligible questions are absent from the source dataset.')
    assert_eligible(queries, settings)
    contract = {'dataset': settings['dataset'], 'passage_char_budget': settings['retrieval']['passage_char_budget'],
                'scope': {**scope, **{k:str(scope[k]) for k in ['source_directory','eligible_question_ids','coverage_file']}}}
    manifest_path = directory/'dataset_manifest.json'
    if manifest_path.exists():
        saved = read_json(manifest_path)
        if saved['contract'] != contract or any(sha256_file(directory/name) != value for name,value in saved['sha256'].items()):
            raise ValueError('Frozen scoped dataset changed.')
        return saved
    directory.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(directory/'corpus.parquet', index=False)
    queries.to_parquet(directory/'queries.parquet', index=False)
    metadata = dict(contract=contract, validation_questions=len(queries), source_validation_questions=101093,
                    unique_passages=len(corpus), questions_with_known_support=int(queries.has_known_support_labels.sum()),
                    total_known_support_labels=sum(map(len, queries.known_supporting_passage_ids)),
                    sampling_limit='Dataset-order prefix coverage; not a random sample of the full validation split.',
                    sha256={name:sha256_file(directory/name) for name in ['corpus.parquet','queries.parquet']})
    write_json(manifest_path, metadata)
    return metadata


def ensure_cached_assets(settings, directory):
    """Reference the valid matrix prefix and build its index without inference."""
    import faiss
    directory = Path(directory)
    metadata = prepare_cached_dataset(settings, directory)
    source, count, original = source_prefix(settings)
    cfg = settings['retrieval']
    contract = {'dataset': metadata, 'retrieval': cfg}
    manifest_path = directory/'assets_manifest.json'
    matrix_path, index_path = directory/'embeddings.npy', directory/'index.faiss'
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest['contract'] != contract:
            raise ValueError('Scoped retrieval contract changed.')
        if manifest.get('status') == 'complete':
            if any((npy_sha256(load_matrix(directory/name)) if name=='embeddings.npy' and not (directory/name).exists() else sha256_file(directory/name)) != value for name,value in manifest['sha256'].items()):
                raise ValueError('Scoped retrieval asset checksum mismatch.')
            return {**manifest, 'corpus_file':directory/'corpus.parquet', 'index_file':index_path}
    write_json(manifest_path, {'contract':contract, 'status':'building_locally', 'new_passage_embedding_calls':0})
    matrix = original[:count]
    for start in range(0,count,cfg['index_add_batch_size']):
        stop = min(start+cfg['index_add_batch_size'],count)
        block = original[start:stop]
        if not np.isfinite(block).all() or not np.allclose(np.linalg.norm(block,axis=1),1,atol=1e-4):
            raise ValueError('Cached passage vectors are invalid or not normalized.')
    matrix_hash=write_reference(matrix_path,source/'embeddings.npy',matrix)
    faiss.omp_set_num_threads(cfg['threads'])
    nlist = min(cfg['nlist'], max(1,int(math.sqrt(count)*4)),count)
    index = faiss.IndexIVFFlat(faiss.IndexFlatIP(cfg['dimensions']),cfg['dimensions'],nlist,faiss.METRIC_INNER_PRODUCT)
    train = np.random.default_rng(cfg['seed']).choice(count,min(cfg['training_sample_size'],count),replace=False)
    print(f'Building local index from {count:,} saved passage vectors; no passage embedding calls.',flush=True)
    index.train(np.asarray(matrix[train]))
    for start in range(0,count,cfg['index_add_batch_size']):
        index.add(np.asarray(matrix[start:start+cfg['index_add_batch_size']]))
    index.nprobe = index.nlist if cfg['exhaustive_search'] else cfg['nprobe']
    faiss.write_index(index,str(index_path))
    manifest = dict(contract=contract,status='complete',corpus_rows=count,dimensions=cfg['dimensions'],
                    actual_nlist=index.nlist,nprobe=index.nprobe,new_passage_embedding_calls=0,
                    sha256={'embeddings.npy':matrix_hash,index_path.name:sha256_file(index_path)},
                    asset_sizes={'corpus':(directory/'corpus.parquet').stat().st_size,'index':index_path.stat().st_size})
    write_json(manifest_path,manifest)
    return {**manifest,'corpus_file':directory/'corpus.parquet','index_file':index_path}
