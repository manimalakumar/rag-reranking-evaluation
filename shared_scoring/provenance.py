"""Stable fingerprints for the preserved evaluation schema."""
import hashlib
import json


def evaluation_fingerprint(row, metric_name, settings, ragas_version):
    """Match the tested saved fingerprints; prompts are verified separately."""
    model_settings = {key: value for key, value in settings.items()
                      if (key.startswith('judge_') and key != 'judge_system_prompt') or key.startswith('embedding_') or
                      key in ['api_base', 'enable_thinking', 'relevancy_strictness', 'ragas_response_policy', 'query_instruction']}
    value = {'question': row['question'], 'answer': row['answer'],
             'retrieved_contexts': row['retrieved_contexts'] if metric_name == 'faithfulness' else None,
             'metric': metric_name, 'model_settings': model_settings, 'ragas_version': ragas_version}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
