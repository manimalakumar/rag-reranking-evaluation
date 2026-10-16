"""Evaluation trace helpers preserve raw responses without storing credentials."""
import json
from pathlib import Path
from .io import read_json, write_json


def load_evaluation_logs(directory):
    """Read recorded judge requests, retries, sessions, and timing without calls."""
    directory = Path(directory)
    return {'requests': [read_json(path) for path in sorted((directory / 'judge_requests').glob('*.json'))],
            'attempt_history': [read_json(path) for path in sorted((directory / 'ragas_attempt_history').glob('*.json'))],
            'sessions': [read_json(path) for path in sorted((directory / 'judge_sessions').glob('*.json'))],
            'timing': read_json(directory / 'evaluation_audit.json')}


def append_evaluation_event(path, event):
    """Append a credential-free cache/retry event to an explicit JSONL file."""
    if any(key.lower() in {'headers', 'authorization', 'api_key'} for key in event):
        raise ValueError('Evaluation events must not contain credentials or HTTP headers.')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')


def save_evaluation_attempt(path, record):
    """Write one evolving attempt; successful recorded outcomes stay immutable."""
    path = Path(path)
    if path.exists() and read_json(path).get('status') == 'ok':
        if read_json(path) != record:
            raise ValueError(f'Refusing to replace a successful saved evaluation: {path.name}')
        return
    write_json(path, record)
