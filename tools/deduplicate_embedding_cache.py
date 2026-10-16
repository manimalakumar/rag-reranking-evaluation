"""Remove redundant embedding JSON and the copied passage matrix, without notebooks/API calls."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from shared_scoring.io import read_json,write_json,sha256_file
from shared_scoring.matrix_storage import write_reference,load_matrix,npy_sha256


def notebook_hashes():
    paths=[*PROJECT.glob('experiments/*/*.ipynb'),*PROJECT.glob('demos/*/*.ipynb'),
           *PROJECT.glob('diagnosis-retrieval-opportunity/*.ipynb')]
    return {str(p):sha256_file(p) for p in paths}


def main():
    before=notebook_hashes()
    cache=PROJECT/'.cache/qwen-runs'
    saved=0;changed=0
    # Do not rewrite notebook-facing run records or original execution outputs.
    paths=list(cache.glob('**/requests/*/attempt-*.json'))
    for path in paths:
        record=json.loads(path.read_text())
        if record.get('stage') not in {'passage_embedding','query_embedding'} or 'raw_response_json' not in record or 'raw_response_text' not in record:
            continue
        parsed=record['raw_response_json']
        raw_text=record['raw_response_text']
        if json.loads(raw_text)!=parsed:
            raise ValueError(f'Response representations differ; refusing to remove either: {path}')
        old_size=path.stat().st_blocks*512
        # Validate the exact original response before and after the atomic write.
        text_hash=hashlib.sha256(raw_text.encode()).hexdigest()
        write_json(path,record,deduplicate_embedding_response=True)
        del record
        restored=read_json(path)
        if restored['raw_response_json']!=parsed or hashlib.sha256(restored['raw_response_text'].encode()).hexdigest()!=text_hash:
            raise ValueError(f'Response verification failed: {path}')
        saved+=old_size-path.stat().st_blocks*512
        changed+=1
        del restored,parsed,raw_text
        if changed%10==0:
            print(f'Verified {changed} embedding records; removed {saved/1e9:.2f} GB of duplicate JSON.',flush=True)
    from shared_scoring.fresh_retrieval import execution_settings
    from shared_scoring.execution import _retrieval_directory
    settings=execution_settings(PROJECT,include_scope=True)
    folder=_retrieval_directory(settings)
    duplicate=folder/'embeddings.npy'
    manifest=read_json(folder/'assets_manifest.json')
    expected=manifest['sha256']['embeddings.npy']
    if duplicate.exists():
        source=settings['scope']['source_directory']/'embeddings.npy'
        count=settings['scope']['passage_count']
        original=np.load(source,mmap_mode='r')[:count]
        copy=np.load(duplicate,mmap_mode='r')
        if original.shape!=copy.shape or original.dtype!=copy.dtype:
            raise ValueError('Matrix shape or dtype differs; refusing duplicate removal.')
        for start in range(0,count,1024):
            if not np.array_equal(original[start:start+1024],copy[start:start+1024]):
                raise ValueError('Matrix values differ; refusing duplicate removal.')
        if sha256_file(duplicate)!=expected or npy_sha256(original)!=expected:
            raise ValueError('Original matrix file checksum cannot be preserved by a reference.')
        if write_reference(duplicate,source,original)!=expected:
            raise ValueError('Matrix reference checksum differs.')
        size=duplicate.stat().st_blocks*512
        del copy
        duplicate.unlink()
        saved+=size
    if npy_sha256(load_matrix(duplicate))!=expected:
        raise ValueError('Matrix reference verification failed.')
    if notebook_hashes()!=before:
        raise ValueError('A saved notebook changed during deduplication.')
    print(json.dumps(dict(embedding_records_deduplicated=changed,space_reclaimed_gb=round(saved/1e9,3),
                         saved_notebooks_byte_identical=True,vector_dimensions_unchanged=True,compression_used=False,model_calls=0)),flush=True)


if __name__=='__main__':
    main()
