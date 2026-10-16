"""Paired summaries use matching successful questions independently per metric."""
import numpy as np
import pandas as pd


def summarize_paired_metric(scores, metric_column, label, scale, arms, bootstrap_samples, bootstrap_seed, tie_tolerance, delta_label, confidence_quantiles, bootstrap_chunk_size=256):
    """Compare the two configured arms, retaining only matching metric outcomes."""
    reference, comparison = list(arms)
    if scores.duplicated(['query_id', 'arm']).any():
        raise ValueError(f'{label}: duplicate question/arm scores cannot be paired.')
    pivot = scores.pivot(index='query_id', columns='arm', values=metric_column).reindex(columns=list(arms)).dropna()
    delta = pivot[comparison] - pivot[reference]
    count = len(pivot)
    row = {'Metric': label, 'Scale': scale, 'Successful matching pairs': count,
           arms[reference]: float(pivot[reference].mean()) if count else None,
           arms[comparison]: float(pivot[comparison].mean()) if count else None,
           delta_label: float(delta.mean()) if count else None,
           'Improved': int((delta > tie_tolerance).sum()), 'Tied': int((delta.abs() <= tie_tolerance).sum()),
           'Lower': int((delta < -tie_tolerance).sum())}
    if count and bootstrap_samples:
        random = np.random.default_rng(bootstrap_seed)
        if bootstrap_chunk_size < 1:
            raise ValueError('bootstrap_chunk_size must be positive.')
        # Bound memory for large cohorts while preserving the seeded draw order.
        draws = np.concatenate([random.choice(delta.to_numpy(), size=(min(bootstrap_chunk_size, bootstrap_samples - start), count), replace=True).mean(axis=1)
                                for start in range(0, bootstrap_samples, bootstrap_chunk_size)])
        row['Delta bootstrap 95% low'], row['Delta bootstrap 95% high'] = map(float, np.quantile(draws, confidence_quantiles))
    return row, list(pivot.index)


def summarize_paired_scores(scores, arms, metric_settings, bootstrap_samples, bootstrap_seed, tie_tolerance, delta_label, confidence_quantiles, bootstrap_chunk_size=256):
    """Build retrieval/answer/judge rows without mixing their eligibility sets."""
    rows, question_ids = [], {}
    for metric in metric_settings:
        frame = scores
        if metric.get('status_column'):
            frame = frame[frame[metric['status_column']].eq('ok')]
        row, identifiers = summarize_paired_metric(frame, metric['column'], metric['label'], metric['scale'], arms,
                                                  bootstrap_samples, bootstrap_seed, tie_tolerance, delta_label, confidence_quantiles, bootstrap_chunk_size)
        rows.append(row)
        question_ids[metric['label']] = identifiers
    return pd.DataFrame(rows), question_ids
