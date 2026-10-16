"""Annotation-based retrieval metrics, independent of answer/judge scoring."""
import math


def score_retrieval(retrieved_ids, support_ids):
    """Calculate the existing binary Recall, Hit, MRR, and nDCG definitions."""
    gold = set(support_ids)
    if not gold:
        raise ValueError('Annotated-support retrieval scoring requires nonempty support IDs.')
    ids = list(retrieved_ids)
    hits = [int(identifier in gold) for identifier in ids]
    ideal = sum(1 / math.log2(index + 2) for index in range(min(len(ids), len(gold))))
    return {'recall': len(set(ids) & gold) / len(gold), 'hit': float(any(hits)),
            'mrr': next((1 / (index + 1) for index, hit in enumerate(hits) if hit), 0.0),
            'ndcg': sum(hit / math.log2(index + 2) for index, hit in enumerate(hits)) / ideal if ideal else 0.0}
