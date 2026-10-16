"""Read-only teaching views for latency versus answer quality."""
from pathlib import Path

import pandas as pd

from .io import read_json
from .latency_study import load_saved_latency_tables, load_latency_population
from .opportunity_study import refresh_opportunity_results
from .paired_tutorial import (
    _details, _note, _table, show_answer_metrics, show_answer_quality_results,
    show_saved_answer_pair, show_scorer_agreement, judge_coverage_table,
)


def load_latency_tutorial(config):
    """Load saved benchmark measurements and scores; never initialize model clients."""
    return {'config': config, 'population': load_latency_population(config),
            'manifest': read_json(config['generation_manifest_file']),
            'provenance': read_json(config.get('latency_provenance_file', config['study_directory'] / 'input/provenance.json')),
            'latency': load_saved_latency_tables(config), **refresh_opportunity_results(config)}


def show_latency_settings(study):
    """Show the recorded benchmark contract and separate inference from evaluation."""
    config, manifest = study['config'], study['manifest']
    _table(pd.DataFrame([
        ('Study', config['experiment_name']),
        ('Historical benchmark', config['run_id']),
        ('Corpus passages', manifest['corpus_version']['unique_passages']),
        ('Query embedding', f"{manifest['embedding_settings']['model']} · {manifest['embedding_settings']['dimensions']} dimensions"),
        ('Reranker', manifest['reranker_settings']['model']),
        ('Generator', manifest['generator_settings']['model']),
        ('Candidate / context budget', f"{config['candidate_count']} / {config['context_passage_count']} passages"),
        ('Generation limit', f"{manifest['generator_settings']['max_tokens']} tokens"),
        ('Workers', manifest['execution_settings']['workers']),
        ('Maximum attempts per hosted stage', manifest['execution_settings']['max_attempts_per_hosted_stage']),
        ('Attempt timeout', f"{manifest['execution_settings']['timeout_seconds']} seconds"),
    ], columns=['Setting', 'Recorded value']))
    _details('Exact portable configuration', read_json(config['configuration_file']))
    _details('Original corpus, retrieval, generation and retry settings', manifest)


def show_latency_sample(study):
    """Show fixed strata and alternating arm order without selecting on results."""
    config, questions = study['config'], study['population']
    counts = questions.groupby('generation_stratum').size().reindex(config['summary_groups'])
    _table(pd.DataFrame({'Stratum': counts.index.map(config['stratum_labels']), 'Saved pairs': counts.values}))
    table = questions[['latency_sample_order', 'query_id', 'generation_stratum', 'question']].copy()
    table['First path'] = [config['arms'][list(value)[0]] for value in questions.arm_order]
    _details('All ten fixed questions and counterbalanced run order', table)
    _details('Exact sampling settings', config['sampling'])


def show_latency_paths(study):
    """Describe what each measured path includes and what setup excludes."""
    _table(pd.DataFrame([{'Path': study['config']['arms'][arm], 'Recorded work': steps}
                        for arm, steps in study['manifest']['paths_measured'].items()]))
    _details('Common exact generation prompt and settings', study['manifest']['generator_settings'])
    _details('Setup excluded from the timed paths', study['latency']['run_summary'])


def show_latency_scopes(study):
    """Explain successful-request, successful-path and total-wall-time scopes."""
    _table(pd.DataFrame(study['manifest']['timing_definitions'].items(), columns=['Timing field', 'Definition']))
    results = study['latency']['results']
    attempts = study['latency']['attempts']
    _table(pd.DataFrame([
        ('Complete path pairs', len(study['complete_ids'])),
        ('Saved answers', len(study['inputs'])),
        ('Timed hosted stages', len(study['latency']['request_bodies'])),
        ('Timed hosted attempts', len(attempts)),
        ('Timed failed attempts (retained)', int(attempts.status.ne('success').sum())),
        ('Recorded retries', int(results.retry_count.sum())),
        ('Excluded setup records (retained separately)', len(study['latency']['preflight_failures'])),
        ('Truncated answers (retained)', sum(row['truncated'] for row in study['inputs'])),
    ], columns=['Coverage', 'Count']))


def compact_latency_comparison(study):
    """Select original timing summaries, labeling each duration in seconds."""
    table = study['latency']['summary'][['arm', 'questions', 'retries', 'median_successful_path_seconds',
        'median_total_end_to_end_seconds', 'p95_total_end_to_end_seconds']].copy()
    table['arm'] = table.arm.map(study['config']['arms'])
    return table.rename(columns={'arm': 'Path', 'questions': 'Complete paths', 'retries': 'Retries',
        'median_successful_path_seconds': 'Median successful path (s)',
        'median_total_end_to_end_seconds': 'Median total wall time (s)',
        'p95_total_end_to_end_seconds': 'p95 total wall time (s)'})


def show_latency_results(study):
    """Show original arm and paired medians with full measurements expandable."""
    from IPython.display import display, HTML
    table = compact_latency_comparison(study)
    display(HTML(table.to_html(index=False, escape=True, float_format=lambda value: f'{value:.3f}')))
    paired = study['latency']['paired']
    _table(pd.DataFrame([
        ('Successful path', len(paired), paired.successful_path_seconds_added_by_reranking.median()),
        ('Total wall time', len(paired), paired.total_end_to_end_seconds_added_by_reranking.median()),
    ], columns=['Timing scope', 'Matching pairs', 'Median paired added time (s)']))
    _details('All original medians and p95 values (seconds)', study['latency']['summary'])
    _details('Every original within-question latency difference (seconds)', paired)
    _details('Every original complete-path measurement', study['latency']['results'])


def show_latency_stages(study):
    """Show stage timings in milliseconds, retaining all exact durations in traces."""
    config = study['config']
    table = study['latency']['stage_summary'].copy()
    table['Path'] = table.arm.map(config['arms'])
    table['Stage'] = table.stage.map(config['stage_labels'])
    table['Median (ms)'] = table.median_stage_seconds * 1000
    table['p95 (ms)'] = table.p95_stage_seconds * 1000
    _table(table[['Path', 'Stage', 'calls', 'Median (ms)', 'p95 (ms)', 'retries']])
    _details('All 90 original stage measurements, with exact seconds', study['latency']['stages'])


def show_latency_failures(study):
    """Keep the timed retry and excluded preflight failures separate and inspectable."""
    attempts = study['latency']['attempts']
    failed = attempts[attempts.status.ne('success')]
    _table(failed[['query_id', 'arm', 'stage', 'attempt', 'elapsed_seconds', 'backoff_after_seconds']])
    _details('All timed request attempts, errors, actual backoff and status', attempts)
    _details('The two excluded setup/implementation failure records', study['latency']['preflight_failures'])
    _details('Original recorded run metadata', study['latency']['run_metadata'])


def paired_quality_latency_table(study):
    """Join quality and latency only for the same benchmark question and arm pair."""
    config = study['config']
    scores = study['standard']['scores']
    if scores.empty:
        return pd.DataFrame()
    baseline, reranked = config['arms']
    metric = config['tutorial']['primary_metric_column']
    quality = scores.pivot(index='query_id', columns='arm', values=metric).reindex(columns=list(config['arms'])).dropna()
    result = study['latency']['paired'][['query_id', 'question', 'generation_stratum',
        'successful_path_seconds_added_by_reranking', 'total_end_to_end_seconds_added_by_reranking']].copy()
    result = result.merge((quality[reranked] - quality[baseline]).rename('Token F1 delta (percentage points)'),
                          left_on='query_id', right_index=True, validate='one_to_one')
    result['Stratum'] = result.generation_stratum.map(config['stratum_labels'])
    return result.rename(columns={'successful_path_seconds_added_by_reranking': 'Successful path delta (s)',
        'total_end_to_end_seconds_added_by_reranking': 'Total wall-time delta (s)'})


def show_latency_quality_tradeoff(study):
    """Display same-answer quality deltas alongside original measured latency deltas."""
    table = paired_quality_latency_table(study)
    if table.empty:
        _note('Quality scoring is pending for these benchmark answers.')
        return
    _table(table[['query_id', 'Stratum', 'Successful path delta (s)', 'Total wall-time delta (s)',
                  'Token F1 delta (percentage points)']])
    _details('Question text and complete paired tradeoff values', table)
    show_answer_quality_results(study)


def show_latency_question(study, query_id):
    """Inspect original path timing, both benchmark answers and separate saved contexts."""
    selected = study['latency']['results'][study['latency']['results'].query_id.eq(str(query_id))]
    if selected.empty:
        raise ValueError(f'Question {query_id} is not in the frozen latency sample.')
    _table(selected[['arm', 'arm_position', 'successful_path_seconds',
                     'total_end_to_end_seconds', 'retry_count', 'backoff_seconds']])
    show_saved_answer_pair(study, query_id)
    for arm, label in study['config']['arms'].items():
        record = study['current'][(str(query_id), arm)]
        _details(label + ': this path’s exact candidate IDs, text and ranks', pd.DataFrame(record['retrieved_top30']))
        _details(label + ': embedding, reranking and generation requests/responses/timing', record['stage_traces'])


def show_latency_diagnostic(study):
    """Show configured diagnostic membership without silently changing the sample."""
    ids = set(study['config'].get('baseline_diagnostic_ids', []))
    selected = study['population'][study['population'].query_id.isin(ids)]
    if selected.empty:
        _note('**Baseline diagnostic:** none of the configured diagnostic IDs is in this saved latency sample.')
    else:
        from .paired_tutorial import show_diagnostic_sensitivity
        show_diagnostic_sensitivity(study)


def show_latency_provenance(study):
    """Show source checksums, portable-path migration and original framework attribution."""
    _details('Import provenance, exact source hashes and portable manifest migration', study['provenance'])
    _details('Saved local integrity audit', read_json(study['config']['input_verification_file']))
    _details('Judge coverage and metric-specific successful pair counts', judge_coverage_table(study))


def show_saved_latency_plot(study):
    """Display an existing plot; displaying it never computes scores or runs models."""
    from IPython.display import display, Image
    path = study['config']['latency_plot_file']
    if not path.is_file():
        _note('Saved plot unavailable. The numeric tables remain available; export a plot explicitly if needed.')
        return
    display(Image(filename=str(path)))


def show_latency_framework_source():
    """Show actual benchmark and scoring implementations without running them."""
    import inspect
    from .latency_benchmark import run_new_latency_benchmark, _execute_path, _request_stage
    from .paired_tutorial import show_evaluation_entry_points
    show_evaluation_entry_points()
    _details('Actual benchmark entry point, path stages and request tracing',
             '\n\n'.join(inspect.getsource(function) for function in
                         [run_new_latency_benchmark, _execute_path, _request_stage]))


def export_latency_quality_plot(study, output_file):
    """Export a standard Matplotlib figure from saved values; no inference or scoring."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError('Plot export needs the pinned optional requirements-latency.txt.') from error
    config = study['config']
    summary = study['latency']['summary'].set_index('arm').reindex(config['arms'])
    tradeoff = paired_quality_latency_table(study)
    if tradeoff.empty:
        raise ValueError('Score the saved benchmark answers locally before exporting the tradeoff plot.')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), constrained_layout=True)
    positions = list(range(len(config['arms'])))
    axes[0].bar(positions, summary.median_successful_path_seconds,
                color=[config['plot']['arm_colors'][arm] for arm in config['arms']],
                label='Successful requests + local work', width=0.55)
    axes[0].scatter(positions, summary.median_total_end_to_end_seconds,
                   c='#17233b', marker='D', s=65, label='Total wall time', zorder=3)
    axes[0].set_xticks(positions, [label.replace(' + ', ' +\n') for label in config['arms'].values()])
    axes[0].set_ylabel('Median latency (seconds)')
    axes[0].set_title('Ten recorded paths per arm')
    axes[0].legend(fontsize=8, frameon=False)
    for group in config['summary_groups']:
        rows = tradeoff[tradeoff.generation_stratum.eq(group)]
        axes[1].scatter(rows['Total wall-time delta (s)'], rows['Token F1 delta (percentage points)'],
                        c=config['plot']['stratum_colors'][group], s=65,
                        label=config['stratum_labels'][group], edgecolors='white', linewidths=0.7)
    axes[1].axhline(0, color='#b8c1d1', linewidth=1)
    axes[1].axvline(0, color='#b8c1d1', linewidth=1)
    axes[1].set_xscale('symlog', linthresh=1)
    axes[1].margins(x=0.15, y=0.1)
    axes[1].set_xlabel('Added total wall time (seconds; symmetric log scale)')
    axes[1].set_ylabel('Token F1 delta (percentage points)')
    axes[1].set_title('Same saved answers: quality versus latency')
    axes[1].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.25), frameon=False)
    fig.suptitle(config['experiment_name'] + f" · {len(study['complete_ids'])} saved question pairs")
    target = Path(output_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return target
