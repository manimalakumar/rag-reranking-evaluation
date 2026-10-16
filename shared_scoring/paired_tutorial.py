"""Read-only teaching views for paired answer studies; scoring is a separate action."""
import inspect
import json
from html import escape

import pandas as pd

from .io import read_json
from .opportunity_study import (
    load_opportunity_population, load_opportunity_configuration,
    refresh_opportunity_results, summarize_opportunity_scores,
)
from .paired_generation import select_generation_sample


def load_answer_quality_study(config):
    """Read frozen answers, verified scores and provenance without model clients."""
    required = ['generation_manifest_file', 'parent_provenance_file',
                'manual_review_file', 'stratum_labels', 'tutorial']
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f'Answer-quality tutorial configuration needs: {", ".join(missing)}')
    population = load_opportunity_population(config)
    snapshot = refresh_opportunity_results(config)
    return {'config': config, 'population': population, **snapshot,
            'provenance': read_json(config['generation_manifest_file']),
            'parent_provenance': read_json(config['parent_provenance_file']),
            'manual_review': pd.read_parquet(config['manual_review_file'])}


def verify_answer_quality_sample(study):
    """Reproduce the seeded sample and compare its IDs, strata and order."""
    config = study['config']
    settings = config['sampling']
    parent = load_opportunity_configuration(config['parent_configuration'])
    reproduced = select_generation_sample(load_opportunity_population(parent),
        settings['strata'], settings['questions_per_stratum'], settings['seed'],
        config['summary']['tie_tolerance'])
    for column in ['query_id', 'generation_stratum', 'sample_order']:
        if reproduced[column].tolist() != study['population'][column].tolist():
            raise ValueError(f'Frozen seeded sample differs from its parent: {column}')
    return {'seeded_sample_verified': True, 'questions': len(reproduced), 'model_calls': 0}


def _table(frame):
    from IPython.display import display, HTML
    display(HTML(frame.to_html(index=False, escape=True, na_rep='Pending',
                              float_format=lambda value: f'{value:.2f}')))


def _details(label, value):
    from IPython.display import display, HTML
    if isinstance(value, pd.DataFrame):
        body = value.to_html(index=False, escape=True, na_rep='Pending')
    else:
        text = value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False)
        body = '<pre style="white-space:pre-wrap">' + escape(text) + '</pre>'
    display(HTML('<details><summary>' + escape(label) + '</summary>' + body + '</details>'))


def _note(text):
    from IPython.display import display, Markdown
    display(Markdown(text))


def _label_groups(frame, config):
    return frame.rename(columns={config['summary_group_column']: 'Stratum'}).assign(
        Stratum=lambda data: data.Stratum.map(config['stratum_labels']))


def show_answer_quality_settings(study):
    """Show study settings with the retrieval and evaluation embedding jobs separate."""
    config, parent = study['config'], study['parent_provenance']
    generation, judge = config['generation'], config['judge']
    _table(pd.DataFrame([
        ('Study', config['experiment_name']),
        ('Historical run ID', config['run_id']),
        ('Saved generator', generation['model']),
        ('Passages per answer', config['context_passage_count']),
        ('Saved generation limit', f"{generation['max_tokens']} tokens"),
        ('Retrieval embeddings (already saved)',
         f"{parent['embedding_model']} · {parent['retrieval_settings']['embedding_dimensions']} dimensions"),
        ('RAGAS judge (explicit action)', judge['judge_model']),
        ('Answer relevancy embeddings (explicit action)',
         f"{judge['embedding_model']} · {judge['embedding_dimensions']} dimensions"),
    ], columns=['Setting', 'Value']))
    _details('Full portable configuration', read_json(config['configuration_file']))


def show_answer_quality_sample(study):
    """Show parent and sample counts, retaining all fixed question IDs."""
    config, provenance = study['config'], study['provenance']
    counts = pd.DataFrame({'Parent questions': provenance['population_counts'],
                          'Saved sample': provenance['sample_counts']}).reindex(config['summary_groups']).fillna(0).astype(int)
    counts.index.name = config['summary_group_column']
    _table(_label_groups(counts.reset_index(), config))
    _details('Control definitions and sampling settings',
             {'controls': provenance['control_definitions'], 'sampling': config['sampling']})
    _details('All sample IDs and counterbalanced submission order', study['population'][
        ['sample_order', 'query_id', 'question', config['summary_group_column'], 'generation_order']])


def show_generation_contract(study):
    """Show the common saved generation contract and exact provenance."""
    provenance = study['provenance']
    settings = provenance['generator_settings']
    _table(pd.DataFrame([(key, settings[key]) for key in
        ['temperature', 'top_p', 'seed', 'max_tokens', 'thinking_enabled', 'stream']],
        columns=['Same setting in both arms', 'Saved value']))
    _details('Exact system prompt, generator settings and context policy',
             {'generator': settings, 'contexts': provenance['context_policy'],
              'reference_policy': provenance['reference_policy']})
    _details('Original source and parent checksums',
             {'source': provenance['source_study'],
              'source_file_sha256': provenance['source_file_sha256'],
              'parent_file_sha256': provenance['parent_file_sha256'],
              'parent_notebook_revision': study['config'].get('provenance_file_revisions', {})})


def show_generation_coverage(study):
    """Count retained answers, failures and truncations without filtering responses."""
    records, current = study['records'], study['current']
    _table(pd.DataFrame([
        ('Saved questions', len(study['population'])),
        ('Complete answer pairs', len(study['complete_ids'])),
        ('Current successful answers', sum(row['status'] == 'ok' for row in current.values())),
        ('Historical attempts', len(records)),
        ('Historical failed attempts', sum(row['status'] != 'ok' for row in records)),
        ('Current failed arms', len(study['failures'])),
        ('Retained truncated answers', sum(row['truncated'] for row in current.values())),
    ], columns=['Coverage', 'Count']))
    columns = ['attempt_id', 'query_id', 'source_arm', 'source_status', 'error',
               'elapsed_seconds', 'execution_workers', 'finish_reason', 'truncated']
    _details('All historical attempts, failures and generation timings',
             pd.DataFrame(records)[columns])
    _details('All retained truncated answers', pd.DataFrame([
        {'query_id': row['query_id'], 'Path': study['config']['arms'][row['arm']],
         'finish_reason': row['finish_reason'], 'answer': row['answer']}
        for row in current.values() if row['truncated']]))


def show_answer_metrics(study):
    """Explain configured answer metrics and show explicit judge settings."""
    config = study['config']
    _table(pd.DataFrame([{'Metric': metric['label'], 'Scale': metric['scale'],
                         'What it measures': metric['description']}
        for metric in config['summary']['metrics'] if metric['kind'] != 'retrieval']))
    _details('Judge prompts, embeddings, retries, pacing and failure policy', config['judge'])


def compact_answer_comparison(study, metric_column):
    """Select a saved metric's paired summary; never recalculate answer scores."""
    config = study['config']
    metrics = {metric['column']: metric for metric in config['summary']['metrics']}
    if metric_column not in metrics or metrics[metric_column]['kind'] == 'retrieval':
        raise ValueError(f'Choose a configured answer metric: {metric_column}')
    summary = study['standard']['summary']
    if summary.empty:
        return pd.DataFrame()
    metric = metrics[metric_column]
    columns = [config['summary_group_column'], 'Successful matching pairs',
               *config['arms'].values(), config['summary']['delta_label']]
    table = _label_groups(summary[summary.Metric.eq(metric['label'])][columns].copy(), config)
    suffix = ' (%)' if metric['scale'] == '0–100 percent' else f" ({metric['scale']})"
    labels = {label: label + suffix for label in config['arms'].values()}
    labels[config['summary']['delta_label']] = (
        'Paired delta (percentage points)' if suffix == ' (%)' else config['summary']['delta_label'])
    return table.rename(columns=labels)


def judge_coverage_table(study):
    """Count success, failure and pending outcomes separately for each metric."""
    config, scores = study['config'], study['standard']['scores']
    rows = []
    for metric in config['summary']['metrics']:
        if metric['kind'] != 'judge':
            continue
        for group in config['summary_groups']:
            subset = scores[scores[config['summary_group_column']].eq(group)] if not scores.empty else scores
            statuses = subset[metric['status_column']] if not scores.empty else pd.Series(dtype=str)
            rows.append({'Stratum': config['stratum_labels'][group], 'Metric': metric['label'],
                         'Successful answers': int(statuses.eq('ok').sum()),
                         'Failed answers': int((~statuses.isin(['ok', 'missing'])).sum()),
                         'Pending answers': int(statuses.eq('missing').sum()),
                         'Successful matching pairs': len(study['standard']['pairs'].get(group, {}).get(metric['label'], []))})
    return pd.DataFrame(rows)


def show_answer_quality_results(study):
    """Show a compact answer comparison, with full results and coverage expandable."""
    if study['standard']['scores'].empty:
        _note('Answer scores are pending. Run scoring explicitly after checking the saved inputs.')
        return
    primary = study['config']['tutorial']['primary_metric_column']
    _table(compact_answer_comparison(study, primary))
    coverage = judge_coverage_table(study)
    if coverage['Successful answers'].sum() == 0 and coverage['Pending answers'].sum() > 0:
        _note('**RAGAS is pending.** Missing outcomes are not zero scores. '
              f"Recorded evaluation failures: **{coverage['Failed answers'].sum()}**; "
              f"successful matching pairs: **{coverage['Successful matching pairs'].sum()}**.")
    _details('Faithfulness and Answer relevancy: coverage and metric-specific pair counts', coverage)
    answers = study['standard']['summary']
    labels = [m['label'] for m in study['config']['summary']['metrics'] if m['kind'] != 'retrieval']
    _details('All answer metrics, improved/tied/lower counts and paired bootstrap intervals',
             answers[answers.Metric.isin(labels)])
    _details('Every per-answer score, status and truncation flag', study['standard']['scores'])


def show_saved_answer_pair(study, query_id):
    """Display unchanged citation-bearing answers and each arm's separate evidence."""
    query_id = str(query_id)
    selected = {row['arm']: row for row in study['inputs'] if row['query_id'] == query_id}
    if set(selected) != set(study['config']['arms']):
        if study['config'].get('fresh_run'):
            _note(f'Question {query_id} has no complete pair. Its available responses and failures are retained below.')
            _details('Both saved arms, including failures', [record for key, record in study['current'].items() if key[0] == query_id])
            return
        raise ValueError(f'Question {query_id} has no complete saved pair; inspect the retained failures.')
    from IPython.display import display, HTML
    first = next(iter(selected.values()))
    _note(f"**Question {query_id}:** {escape(first['question'])}")
    _details('All evaluation reference answers (never sent to the generator)', first['reference_answers'])
    scores = study['standard']['scores']
    if not scores.empty:
        _table(scores[scores.query_id.eq(query_id)][['path', 'squad_token_f1_pct',
               'squad_exact_match_pct', 'finish_reason', 'truncated']].rename(columns={
                   'path': 'Path', 'squad_token_f1_pct': 'Token F1 (%)',
                   'squad_exact_match_pct': 'Normalized exact match (%)'}))
    for arm, label in study['config']['arms'].items():
        row, record = selected[arm], study['current'][(query_id, arm)]
        _note('**' + label + '**')
        display(HTML('<pre style="white-space:pre-wrap">' + escape(row['answer']) + '</pre>'))
        _details(label + ': exact five passages in saved order', row['context'])
        _details(label + ': exact messages and reconstructed request payload',
                 {'origin': record['request_body_origin'], 'url': record['url'],
                  'request_body': record['request_body'], 'fingerprint': record['fingerprint']})
        _details(label + ': original saved response JSON and attempt metadata',
                 {'origin': record['raw_response_text_origin'],
                  'saved_json_serialization': record['raw_response_text'],
                  'attempt_id': record['attempt_id'], 'elapsed_seconds': record['elapsed_seconds'],
                  'finish_reason': record['finish_reason'], 'truncated': record['truncated']})


def show_sample_retrieval_scores(study):
    """Display unchanged annotated retrieval summaries separately from answer metrics."""
    config = study['config']
    labels = [m['label'] for m in config['summary']['metrics'] if m['kind'] == 'retrieval']
    table = study['standard']['summary']
    if not table.empty:
        _table(_label_groups(table[table.Metric.isin(labels)][[
            config['summary_group_column'], 'Metric', 'Scale', 'Successful matching pairs',
            *config['arms'].values(), config['summary']['delta_label']]], config))


def show_saved_manual_review(study):
    """Show preserved human-review counts, keeping all original notes inspectable."""
    group = study['config']['summary_group_column']
    review = study['manual_review']
    if review.empty:
        _note('Human review is pending for this fresh run. No old correctness labels were transferred.')
        return
    table = review.groupby(group).agg(
        reviewed_questions=('query_id', 'size'),
        embedding_correct=('embedding_correct', 'sum'),
        reranked_correct=('reranked_correct', 'sum')).reindex(study['config']['summary_groups'])
    table = _label_groups(table.reset_index(), study['config']).rename(columns={
        'reviewed_questions': 'Reviewed questions',
        'embedding_correct': study['config']['arms']['baseline'] + ': marked correct',
        'reranked_correct': study['config']['arms']['reranked'] + ': marked correct'})
    _table(table)
    _details('All original human reviews, both evidence sets and notes', review)


def show_diagnostic_sensitivity(study):
    """Flag configured diagnostic cases without changing original sample membership."""
    config, scores = study['config'], study['standard']['scores']
    if scores.empty:
        _note('Diagnostic answer scores are pending.')
        return
    _details('Separate flagged baseline diagnostic cases: larger-corpus answers',
             scores[scores.is_baseline_diagnostic])
    sensitivity, pairs = summarize_opportunity_scores(scores[~scores.is_baseline_diagnostic], config)
    alternate = {**study, 'standard': {**study['standard'], 'summary': sensitivity, 'pairs': pairs}}
    _table(compact_answer_comparison(alternate, config['tutorial']['primary_metric_column']))


def show_scorer_agreement(study):
    """Compare saved per-answer standard and custom scores on a common scale."""
    scores = study['standard']['scores']
    if scores.empty:
        _note('Custom versus SQuAD comparison is pending.')
        return
    tolerance = study['config']['summary']['tie_tolerance']
    rows = []
    for label, column in [('Token F1', 'token_f1_delta_0_1'),
                          ('Normalized exact match', 'normalized_em_delta_0_1')]:
        delta = scores[column].abs()
        rows.append({'Metric': label, 'Compared answers': len(scores),
                     'Differences above tolerance': int(delta.gt(tolerance).sum()),
                     'Maximum absolute difference (0–1)': float(delta.max())})
    _table(pd.DataFrame(rows))
    _details('Any scorer discrepancies (SQuAD / 100 versus custom 0–1)', scores[
        scores.token_f1_delta_0_1.abs().gt(tolerance) |
        scores.normalized_em_delta_0_1.abs().gt(tolerance)])


def show_evaluation_entry_points():
    """Show real framework code without importing metrics or constructing clients."""
    from .answers import load_squad_metric, score_squad_answer
    from .judge import create_ragas_metrics
    _details('Hugging Face: evaluate.load and multiple-reference metric.compute',
             inspect.getsource(load_squad_metric) + '\n' + inspect.getsource(score_squad_answer))
    _details('RAGAS: llm_factory, Faithfulness and AnswerRelevancy',
             inspect.getsource(create_ragas_metrics))


def show_saved_input_audit(study):
    """Read the existing validation report and document the parent notebook revision."""
    config = study['config']
    _details('Saved frozen-rank and annotated-support audit', read_json(config['input_verification_file']))
    _details('Original parent notebook checksum and preserved later revision',
             config.get('provenance_file_revisions', {}))
