"""Reserve hosted attempts atomically before sending; counters survive restarts."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3

PROJECT = Path(__file__).resolve().parents[1]


class ModelBudgetExceeded(RuntimeError):
    pass


def reserve_request(url, payload, identity, max_attempts):
    config_path = PROJECT / 'configs/model_limits.json'
    limits = json.loads(config_path.read_text())
    ledger = Path(os.environ.get('MODEL_BUDGET_PATH') or (config_path.parent / limits['ledger_path'])).resolve()
    ledger.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(json.dumps({'url': url, 'payload': payload, 'identity': identity},
        sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    with sqlite3.connect(ledger, timeout=30) as connection:
        connection.execute('CREATE TABLE IF NOT EXISTS attempts (fingerprint TEXT PRIMARY KEY, count INTEGER NOT NULL)')
        connection.execute('BEGIN IMMEDIATE')
        total = connection.execute('SELECT COALESCE(SUM(count), 0) FROM attempts').fetchone()[0]
        if total >= limits['max_total_requests']:
            raise ModelBudgetExceeded(f'Hosted request cap reached ({total}/{limits["max_total_requests"]}). '
                                      'No request sent. Inspect the budget before explicitly raising its limit.')
        row = connection.execute('SELECT count FROM attempts WHERE fingerprint=?', (fingerprint,)).fetchone()
        used = row[0] if row else 0
        if used >= max_attempts:
            raise ModelBudgetExceeded(f'This logical request reached its lifetime cap of {max_attempts} attempts. No request sent.')
        connection.execute('INSERT INTO attempts VALUES (?, 1) ON CONFLICT(fingerprint) DO UPDATE SET count=count+1',
                           (fingerprint,))
    return total + 1
