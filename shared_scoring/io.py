"""Configuration and artifact I/O; importing this module performs no I/O."""
import hashlib
import json
from pathlib import Path


def read_json(path):
    """Read an existing JSON artifact with a useful path error."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing saved artifact: {path}')
    try:
        value=json.loads(path.read_text(encoding='utf-8'))
        if isinstance(value,dict) and value.get('response_json_storage')=='derived_from_raw_response_text':
            value['raw_response_json']=json.loads(value['raw_response_text'])
        return value
    except ValueError as error:
        raise ValueError(f'Invalid JSON in {path}: {error}') from error


def write_json(path, value, deduplicate_embedding_response=False):
    """Write JSON atomically; callers choose an explicit output path."""
    if deduplicate_embedding_response and isinstance(value,dict) and value.get('stage') in {'query_embedding','passage_embedding'} and 'raw_response_text' in value and 'raw_response_json' in value:
        # Retain the exact HTTP response once. Existing readers reconstruct JSON
        # in memory; no compression or numeric conversion is involved.
        if json.loads(value['raw_response_text'])==value['raw_response_json']:
            value={k:v for k,v in value.items() if k!='raw_response_json'}
            value['response_json_storage']='derived_from_raw_response_text'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha256_file(path):
    """Return a file checksum without reading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_configuration(path):
    """Resolve configured artifact paths relative to the JSON file."""
    path = Path(path).resolve()
    config = read_json(path)
    required = ['experiment_name', 'source_directory', 'score_directory', 'run_id',
                'arms', 'baseline_scope', 'diagnostic_ids', 'context_passage_count',
                'citation_pattern', 'summary', 'squad', 'judge']
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f'Configuration {path.name} is missing: {", ".join(missing)}')
    if len(config['arms']) != 2:
        raise ValueError('A paired comparison requires exactly two configured arms.')
    for key in ['source_directory', 'score_directory']:
        config[key] = (path.parent / config[key]).resolve()
    for key in ['metric_source', 'cache_directory']:
        if config['squad'].get(key):
            config['squad'][key] = (path.parent / config['squad'][key]).resolve()
    return config
