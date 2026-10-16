"""Deterministic answer scoring; importing helpers never loads a model."""
import json
import os
from pathlib import Path
import re
import string
from collections import Counter
import pandas as pd


def remove_passage_citations(answer, citation_pattern):
    """Remove only configured passage-ID brackets, retaining all other text."""
    if not isinstance(answer, str):
        raise TypeError('Saved answer must be a string, including when empty.')
    return re.sub(citation_pattern, ' ', answer)


def normalize_custom_answer(answer, citation_pattern):
    """Reproduce the preserved custom normalization for cross-checking."""
    text = remove_passage_citations(answer.lower(), citation_pattern)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def score_custom_answer(answer, reference_answers, citation_pattern):
    """Return the existing custom exact match and Token F1 on the 0–1 scale."""
    if not reference_answers:
        raise ValueError('Supply all reference answers; an empty reference list is invalid.')
    normalized = normalize_custom_answer(answer, citation_pattern)
    tokens = normalized.split()
    f1 = []
    for reference in reference_answers:
        other = normalize_custom_answer(reference, citation_pattern).split()
        overlap = sum((Counter(tokens) & Counter(other)).values())
        f1.append(2 * overlap / (len(tokens) + len(other)) if tokens or other else 1.0)
    return {'normalized_em': float(any(normalized == normalize_custom_answer(r, citation_pattern) for r in reference_answers)),
            'token_f1': max(f1)}


def load_squad_metric(metric_source='squad', cache_directory=None):
    """Load Hugging Face SQuAD explicitly; this is never called by replay."""
    if cache_directory:
        directory = Path(cache_directory).resolve()
        os.environ.setdefault('HF_HOME', str(directory / 'huggingface'))
        os.environ.setdefault('HF_MODULES_CACHE', str(directory / 'modules'))
        os.environ.setdefault('HF_EVALUATE_CACHE', str(directory / 'evaluate'))
    import evaluate
    return evaluate.load(str(metric_source), cache_dir=str(cache_directory) if cache_directory else None)


def score_squad_answer(metric, answer, reference_answers, question_id, citation_pattern):
    """Use SQuAD's multiple-reference scoring; returned scores are percentages."""
    if not reference_answers or not all(isinstance(ref, str) for ref in reference_answers):
        raise ValueError(f'Question {question_id}: provide a nonempty list of reference strings.')
    identifier = str(question_id)
    return metric.compute(
        predictions=[{'id': identifier, 'prediction_text': remove_passage_citations(answer, citation_pattern)}],
        references=[{'id': identifier, 'answers': {'text': list(reference_answers),
                                                   'answer_start': [0] * len(reference_answers)}}])


def score_squad_answers(inputs, metric, citation_pattern, legacy_citation_pattern):
    """Score each saved answer, keeping truncation and the old scorer comparison."""
    records = []
    for row in inputs:
        result = score_squad_answer(metric, row['answer'], row['reference_answers'], row['query_id'], citation_pattern)
        custom = score_custom_answer(row['answer'], row['reference_answers'], legacy_citation_pattern)
        for key, saved in [('token_f1', 'existing_token_f1_0_1'), ('normalized_em', 'existing_normalized_em_0_1')]:
            if saved in row and abs(custom[key] - row[saved]) > 1e-12:
                raise ValueError(f"Existing custom score changed: {row['query_id']} / {row['arm']} / {key}")
        records.append({key: row[key] for key in ['query_id', 'question', 'arm', 'path', 'answer', 'finish_reason', 'truncated',
                                                 'annotated_support_recall_at_5', 'existing_token_f1_0_1', 'existing_normalized_em_0_1']} |
                       {'squad_answer': remove_passage_citations(row['answer'], citation_pattern),
                        'reference_answers': json.dumps(row['reference_answers'], ensure_ascii=False),
                        'squad_exact_match_pct': result['exact_match'], 'squad_token_f1_pct': result['f1'],
                        'token_f1_delta_0_1': result['f1'] / 100 - custom['token_f1'],
                        'normalized_em_delta_0_1': result['exact_match'] / 100 - custom['normalized_em']})
    return pd.DataFrame(records)
