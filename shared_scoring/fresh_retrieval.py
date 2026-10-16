"""Prepare pinned raw data and resumable retrieval assets when saved data is absent."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .budget import reserve_request
from .io import read_json, write_json, sha256_file
from .provider import embedding_payload, rerank_payload, rerank_scores, QUERY_INSTRUCTION, require_model_calls


def fingerprint(value):
    """Hash portable input values; never include credentials or local asset paths."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def passage_id(text):
    """Use the original casefolded, whitespace-trimmed passage identity."""
    return 'p_' + fingerprint(text.strip().casefold())[:20]


def execution_settings(project, include_scope=False):
    """Resolve the separate fresh-run configuration without changing frozen configs."""
    path = Path(project) / 'configs/execution.json'
    settings = read_json(path)
    settings['runtime_directory'] = (path.parent / settings['runtime_directory']).resolve()
    scope_path = path.parent / 'retrieval_scope.json'
    if include_scope and scope_path.is_file():
        settings['scope'] = read_json(scope_path)
        for key in ['source_directory', 'eligible_question_ids', 'coverage_file']:
            settings['scope'][key] = (scope_path.parent / settings['scope'][key]).resolve()
    return settings


class RecordedOpenRouterClient:
    """Cache exact successful responses and retain every credential-free attempt."""

    def __init__(self, settings, directory, session=None):
        self.settings, self.directory = settings, Path(directory)
        self.session = session

    def request(self, stage, url, payload, validate=None, cache_key=None):
        """Load a matching response, otherwise call with the configured retry policy."""
        contract = {'stage': stage, 'url': url, 'payload': payload}
        if cache_key is not None:
            contract['cache_key'] = cache_key
        identity = fingerprint(contract)
        folder = self.directory / identity
        paths = sorted(folder.glob('attempt-*.json'))
        for path in paths:
            trace = read_json(path)
            if trace['request_body'] != payload or trace['url'] != url:
                raise ValueError('Request cache inputs changed; use a new runtime directory.')
            if trace['status'] == 'ok':
                raw = trace['raw_response_json']
                if validate:
                    validate(raw)
                return raw, {**trace, 'cache_hit': True}
        if paths:
            raise RuntimeError(f'{stage}: previous failed/interrupted attempts retained at {folder}. '
                               'Inspect them and choose a new runtime directory to retry explicitly.')
        require_model_calls()
        key = os.environ.get(self.settings['credential_env'])
        if not key:
            raise RuntimeError(f"Saved {stage} results are absent. Set {self.settings['credential_env']} "
                               'to your own OpenRouter credential for the fresh run.')
        if self.session is None:
            import requests
            self.session = requests.Session()
        for number in range(1, self.settings['max_attempts'] + 1):
            reserve_request(url, payload, identity, self.settings['max_attempts'])
            trace = {'stage': stage, 'request_id': identity, 'attempt': number, 'url': url,
                     'request_body': payload, 'status': 'started',
                     'started_utc': datetime.now(timezone.utc).isoformat(), 'backoff_seconds': 0}
            path = folder / f'attempt-{number:02d}.json'
            write_json(path, trace, deduplicate_embedding_response=True)
            started = time.perf_counter()
            try:
                response = self.session.post(url, json=payload,
                    headers={'Authorization': 'Bearer ' + key}, timeout=self.settings['timeout_seconds'])
                trace.update(http_status=response.status_code, raw_response_text=response.text)
                response.raise_for_status()
                raw = response.json()
                trace['raw_response_json'] = raw
                if validate:
                    validate(raw)
                trace['status'] = 'ok'
            except Exception as error:
                trace.update(status='error', error=f'{type(error).__name__}: {error}')
            finally:
                trace.update(elapsed_seconds=time.perf_counter() - started,
                             ended_utc=datetime.now(timezone.utc).isoformat())
                write_json(path, trace, deduplicate_embedding_response=True)
            if trace['status'] == 'ok':
                return raw, {**trace, 'cache_hit': False}
            if number < self.settings['max_attempts']:
                delay = self.settings['backoff_seconds'][number - 1]
                trace['backoff_seconds'] = delay
                write_json(path, trace, deduplicate_embedding_response=True)
                time.sleep(delay)
        raise RuntimeError(f'{stage}: attempts exhausted; inspect {folder}.')

    def embed(self, texts, input_type, dimensions):
        """Return ordered, finite nonzero vectors from OpenRouter's embedding endpoint."""
        payload = embedding_payload(self.settings['embedding_model'], texts, dimensions, input_type,
                                    self.settings.get('query_instruction', QUERY_INSTRUCTION))
        def validate(raw):
            ordered = sorted(raw['data'], key=lambda row: row['index'])
            if [row['index'] for row in ordered] != list(range(len(texts))):
                raise ValueError('Embedding response has missing or duplicate indices.')
            normalized_vectors([row['embedding'] for row in ordered], len(texts), dimensions)
        raw, trace = self.request(input_type + '_embedding', self.settings['api_base'] + '/embeddings', payload, validate)
        return np.asarray([row['embedding'] for row in sorted(raw['data'], key=lambda row: row['index'])], dtype='float32'), trace

    def rerank(self, question, candidates):
        """Rank the exact saved candidates; validate the full permutation and relevance scores."""
        payload = rerank_payload(self.settings['reranker_model'], question, candidates)
        def validate(raw):
            rerank_scores(raw, len(candidates))
        raw, trace = self.request('reranking', self.settings['reranker_url'], payload, validate)
        return rerank_scores(raw, len(candidates)), trace


def normalized_vectors(values, rows, dimensions):
    """Validate shape and normalize vectors for cosine search."""
    array = np.asarray(values, dtype='float32')
    if array.shape != (rows, dimensions) or not np.isfinite(array).all():
        raise ValueError(f'Expected finite embedding matrix {(rows, dimensions)}; received {array.shape}.')
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError('A zero embedding cannot be used for cosine retrieval.')
    return array / norms


def load_raw_dataset(settings, cache_directory):
    """Download the pinned MS MARCO split; Hugging Face manages download resumption."""
    from datasets import load_dataset
    data = settings['dataset']
    return load_dataset(data['name'], data['configuration'], split=data['split'],
                        revision=data['revision'], cache_dir=str(cache_directory))


def prepare_dataset(settings, directory, rows=None):
    """Save unique clipped passages, annotated supports and all reference answers."""
    if settings.get('scope', {}).get('mode') == 'checkpointed_passages':
        from .cached_scope import prepare_cached_dataset
        return prepare_cached_dataset(settings, directory)
    directory = Path(directory)
    contract = {'dataset': settings['dataset'], 'passage_char_budget': settings['retrieval']['passage_char_budget']}
    metadata_path = directory / 'dataset_manifest.json'
    paths = [directory / 'corpus.parquet', directory / 'queries.parquet']
    if metadata_path.is_file() and all(path.is_file() for path in paths):
        metadata = read_json(metadata_path)
        if metadata['contract'] != contract or any(sha256_file(path) != metadata['sha256'][path.name] for path in paths):
            raise ValueError('Prepared dataset changed; use a fresh runtime directory.')
        return metadata
    if rows is None:
        rows = load_raw_dataset(settings, directory / 'downloads')
    documents, queries = {}, []
    budget = contract['passage_char_budget']
    for position, row in enumerate(rows):
        passages = row['passages']
        supports = []
        for text, url, selected in zip(passages['passage_text'], passages['url'], passages['is_selected']):
            text = text.strip()
            if not text:
                continue
            pid = passage_id(text)
            if pid not in documents:
                documents[pid] = {'corpus_index': len(documents), 'passage_id': pid,
                    'text': text[:budget], 'source_url': url, 'original_char_count': len(text), 'truncated': len(text) > budget}
            if selected and pid not in supports:
                supports.append(pid)
        original = list(row.get('answers', []))
        well_formed = row.get('wellFormedAnswers', [])
        if isinstance(well_formed, str):
            well_formed = json.loads(well_formed)
        references = list(dict.fromkeys([*original, *well_formed]))
        queries.append({'query_row': position, 'query_id': str(row['query_id']), 'question': row['query'],
            'known_supporting_passage_ids': supports, 'has_known_support_labels': bool(supports),
            'reference_answers': references, 'original_answers': original, 'well_formed_answers': list(well_formed)})
    corpus, questions = pd.DataFrame(documents.values()), pd.DataFrame(queries)
    if corpus.empty or questions.empty or questions.query_id.duplicated().any():
        raise ValueError('Raw dataset must contain passages and unique question IDs.')
    directory.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(paths[0], index=False)
    questions.to_parquet(paths[1], index=False)
    metadata = {'contract': contract, 'validation_questions': len(questions), 'unique_passages': len(corpus),
        'questions_with_known_support': int(questions.has_known_support_labels.sum()),
        'total_known_support_labels': sum(map(len, questions.known_supporting_passage_ids)),
        'sha256': {path.name: sha256_file(path) for path in paths}}
    write_json(metadata_path, metadata)
    return metadata


def ensure_retrieval_assets(settings, directory, client=None, rows=None):
    """Download data, embed missing batches and build a validated FAISS IVF index."""
    if settings.get('scope', {}).get('mode') == 'checkpointed_passages':
        from .cached_scope import ensure_cached_assets
        return ensure_cached_assets(settings, directory)
    try:
        import faiss
    except ImportError as error:
        raise ImportError('Fresh retrieval needs: python -m pip install -r requirements-runtime.txt') from error
    directory = Path(directory)
    metadata = prepare_dataset(settings, directory, rows)
    cfg = settings['retrieval']
    contract = {'dataset': metadata, 'retrieval': cfg}
    manifest_path = directory / 'assets_manifest.json'
    if manifest_path.exists() and read_json(manifest_path)['contract'] != contract:
        raise ValueError('Embedding/index settings changed; choose a new runtime directory.')
    corpus = pd.read_parquet(directory / 'corpus.parquet')
    count, dimensions = len(corpus), cfg['dimensions']
    if count < cfg['candidate_count']:
        raise ValueError('The corpus is smaller than the configured candidate budget.')
    client = client or RecordedOpenRouterClient(cfg, directory / 'requests')
    vector_path, index_path = directory / 'embeddings.npy', directory / 'index.faiss'
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get('status') == 'complete' and vector_path.is_file() and index_path.is_file():
            if any(sha256_file(path) != manifest['sha256'][path.name] for path in [vector_path, index_path]):
                raise ValueError('Retrieval asset checksum mismatch; refusing to load altered vectors/index.')
            return {**manifest, 'corpus_file': directory / 'corpus.parquet', 'index_file': index_path}
    write_json(manifest_path, {'contract': contract, 'status': 'building'})
    matrix = np.lib.format.open_memmap(vector_path, mode='r+' if vector_path.exists() else 'w+', dtype='float32', shape=(count, dimensions))
    if matrix.shape != (count, dimensions):
        raise ValueError('Partial embedding matrix dimensions changed.')
    for start in range(0, count, cfg['passage_batch_size']):
        stop = min(count, start + cfg['passage_batch_size'])
        checkpoint = directory / 'embedding_batches' / f'{start}.json'
        if checkpoint.exists():
            record = read_json(checkpoint)
            if fingerprint(matrix[start:stop].tolist()) != record['vector_sha256']:
                raise ValueError('Saved embedding batch changed.')
            continue
        values, _ = client.embed(corpus.text.iloc[start:stop].tolist(), 'passage', dimensions)
        matrix[start:stop] = normalized_vectors(values, stop-start, dimensions)
        matrix.flush()
        write_json(checkpoint, {'vector_sha256': fingerprint(matrix[start:stop].tolist()), 'start': start, 'stop': stop})
        print(f'Document embeddings: {stop:,}/{count:,}', flush=True)
    faiss.omp_set_num_threads(cfg['threads'])
    nlist = min(cfg['nlist'], count, max(1, int(math.sqrt(count)*4)))
    index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dimensions), dimensions, nlist, faiss.METRIC_INNER_PRODUCT)
    train_ids = np.random.default_rng(cfg['seed']).choice(count, min(cfg['training_sample_size'], count), replace=False)
    index.train(np.asarray(matrix[train_ids]))
    for start in range(0, count, cfg['index_add_batch_size']):
        index.add(np.asarray(matrix[start:start+cfg['index_add_batch_size']]))
    index.nprobe = index.nlist if cfg['exhaustive_search'] else cfg['nprobe']
    faiss.write_index(index, str(index_path))
    manifest = {'contract': contract, 'status': 'complete', 'corpus_rows': count, 'dimensions': dimensions,
                'actual_nlist': index.nlist, 'nprobe': index.nprobe,
                'sha256': {path.name: sha256_file(path) for path in [vector_path, index_path]},
                'asset_sizes': {'corpus': (directory/'corpus.parquet').stat().st_size, 'index': index_path.stat().st_size}}
    write_json(manifest_path, manifest)
    return {**manifest, 'corpus_file': directory / 'corpus.parquet', 'index_file': index_path}


def retrieve_full_validation(settings, directory, client=None, rows=None):
    """Retrieve the entire validation split in resumable, independently verified shards."""
    directory = Path(directory)
    assets = ensure_retrieval_assets(settings, directory, client, rows)
    import faiss
    cfg = settings['retrieval']
    index = faiss.read_index(str(assets['index_file']))
    index.nprobe = assets['nprobe']
    corpus = pd.read_parquet(assets['corpus_file'])
    queries = pd.read_parquet(directory/'queries.parquet')
    client = client or RecordedOpenRouterClient(cfg, directory/'requests')
    metric_parts, ranking_paths = [], []
    for start in range(0, len(queries), cfg['query_batch_size']):
        folder = directory/'retrieval_batches'/str(start)
        ranking, metrics, manifest = folder/'top30.parquet', folder/'metrics.parquet', folder/'manifest.json'
        if manifest.exists() and ranking.exists() and metrics.exists():
            hashes = read_json(manifest)
            if any(sha256_file(path) != hashes[path.name] for path in [ranking, metrics]):
                raise ValueError('Saved retrieval shard changed.')
        else:
            batch = queries.iloc[start:start+cfg['query_batch_size']]
            vectors, _ = client.embed(batch.question.tolist(), 'query', cfg['dimensions'])
            scores, indices = index.search(normalized_vectors(vectors, len(batch), cfg['dimensions']), cfg['candidate_count'])
            if (indices < 0).any():
                raise ValueError('FAISS returned missing corpus rows.')
            ranked_rows, metric_rows = [], []
            for offset, query in enumerate(batch.to_dict('records')):
                support = set(query['known_supporting_passage_ids'])
                docs = corpus.iloc[indices[offset]].to_dict('records')
                for rank, doc in enumerate(docs, 1):
                    ranked_rows.append({**doc, **query, 'rank': rank, 'similarity_score': float(scores[offset, rank-1]),
                                        'is_known_support': doc['passage_id'] in support})
                recall5 = len(support & {doc['passage_id'] for doc in docs[:cfg['context_count']]})/len(support) if support else float('nan')
                recall30 = len(support & {doc['passage_id'] for doc in docs})/len(support) if support else float('nan')
                metric_rows.append({**query, 'embedding_recall_at_5': recall5,
                    'embedding_recall_at_30': recall30, 'recall_gap_30_minus_5': recall30-recall5})
            folder.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(ranked_rows).to_parquet(ranking, index=False)
            pd.DataFrame(metric_rows).to_parquet(metrics, index=False)
            write_json(manifest, {path.name: sha256_file(path) for path in [ranking, metrics]})
        metric_parts.append(pd.read_parquet(metrics))
        ranking_paths.append(ranking)
        print(f'Validation retrieval: {min(start+cfg["query_batch_size"],len(queries)):,}/{len(queries):,}', flush=True)
    frame = pd.concat(metric_parts, ignore_index=True)
    frame.to_parquet(directory/'per_query_metrics.parquet', index=False)
    return frame, ranking_paths, assets
