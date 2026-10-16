"""Report cached passage coverage using local checkpoints only; no model calls.

Run from the project: .venv/bin/python tools/embedding_coverage.py
Only the contiguous checkpointed prefix of embeddings.npy is considered valid.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared_scoring.fresh_retrieval import passage_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, default=ROOT / '.cache/qwen-runs/full-validation/9d5b5fc55d9fea605a3e')
    parser.add_argument('--output-dir', type=Path, help='New report directory (must not exist).')
    args = parser.parse_args()
    cache = args.cache_dir.resolve()
    batches = sorted((json.loads(p.read_text()) for p in (cache / 'embedding_batches').glob('*.json')), key=lambda b: b['start'])
    end = 0
    for batch in batches:
        if batch['start'] != end or batch['stop'] <= end:
            raise ValueError('Embedding checkpoints contain a gap, overlap, or invalid range.')
        end = batch['stop']
    corpus = pq.read_table(cache / 'corpus.parquet')
    if end > len(corpus):
        raise ValueError('Checkpoints extend beyond the corpus.')
    embedded = corpus.slice(0, end)
    ids = set(embedded['passage_id'].to_pylist())
    if len(ids) != end:
        raise ValueError('Duplicate passage IDs in the embedded corpus.')
    arrows = list((cache / 'downloads').rglob('ms_marco-validation.arrow'))
    if len(arrows) != 1:
        raise ValueError('Expected exactly one local validation Arrow file.')
    now = datetime.now(timezone.utc)
    output = args.output_dir or ROOT / 'reports' / ('embedding-coverage-' + now.strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    groups = {name: [] for name in ('fully-covered', 'partially-covered', 'any-covered', 'all-support-embedded')}
    mapping = defaultdict(list)
    count = 0
    with (output / 'question-coverage.jsonl').open('w') as details, pa.memory_map(str(arrows[0]), 'r') as source:
        for batch in ipc.open_stream(source):
            for row in batch.to_pylist():
                query_id = str(row['query_id'])
                all_ids, support = set(), set()
                passages = row['passages']
                for text, selected in zip(passages['passage_text'], passages['is_selected']):
                    if text.strip():
                        key = passage_id(text)
                        all_ids.add(key)
                        if selected:
                            support.add(key)
                covered = all_ids & ids
                full = bool(all_ids) and covered == all_ids
                full_support = bool(support) and support <= ids
                if covered:
                    groups['any-covered'].append(query_id)
                    groups['fully-covered' if full else 'partially-covered'].append(query_id)
                    if full_support:
                        groups['all-support-embedded'].append(query_id)
                    for key in covered:
                        mapping[key].append(query_id)
                    details.write(json.dumps(dict(query_id=query_id, dataset_row=count, question=row['query'], associated_passage_count=len(all_ids), embedded_associated_passage_count=len(covered), all_associated_passages_embedded=full, all_annotated_support_embedded=full_support, embedded_passage_ids=sorted(covered), not_yet_embedded_passage_ids=sorted(all_ids - covered)), ensure_ascii=False) + '\n')
                count += 1
    if set(mapping) != ids:
        raise ValueError('Some embedded passage IDs were not found in the source questions.')
    for name, values in groups.items():
        (output / (name + '-question-ids.txt')).write_text(''.join(q + '\n' for q in values))
    fully_covered = set(groups['fully-covered'])
    with (output / 'embedded-passages.jsonl').open('w') as passages:
        for batch in embedded.to_batches(max_chunksize=1024):
            for row in batch.to_pylist():
                associated = mapping[row['passage_id']]
                row['query_ids'] = associated
                row['fully_covered_query_ids'] = [q for q in associated if q in fully_covered]
                passages.write(json.dumps(row, ensure_ascii=False) + '\n')
    summary = dict(snapshot_utc=now.isoformat(), cache_directory=os.path.relpath(cache, ROOT), report_directory=os.path.relpath(output.resolve(), ROOT), vector_file=os.path.relpath(cache / 'embeddings.npy', ROOT), coverage_definition='All nonempty associated passages in the source question have checkpointed embeddings; this does not indicate query embedding or downstream stage completion.', counts=dict(full_validation_document_embeddings=end, completed_batches=len(batches), total_corpus_passages=len(corpus), question_count=count, questions_with_any_embedded_passage=len(groups['any-covered']), questions_with_all_associated_passages_embedded=len(groups['fully-covered']), questions_with_partial_associated_passage_coverage=len(groups['partially-covered']), questions_with_all_annotated_support_embedded=len(groups['all-support-embedded'])))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
