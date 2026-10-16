"""Explicit scoring actions; replay calls neither of these functions."""
from pathlib import Path
from .answers import load_squad_metric, score_squad_answers
from .records import load_saved_inputs, load_saved_judge_scores, combine_answer_scores
from .summaries import summarize_paired_scores
from .io import write_json


def run_squad_scoring(config, output_directory):
    """Run deterministic SQuAD explicitly, writing to a fresh output directory."""
    output_directory = Path(output_directory).resolve()
    if output_directory == Path(config['score_directory']).resolve():
        raise ValueError('Use a separate output directory; preserved score files must not be replaced.')
    inputs, _, _, _ = load_saved_inputs(config)
    metric = load_squad_metric(config['squad'].get('metric_source', 'squad'), config['squad'].get('cache_directory'))
    scores = score_squad_answers(inputs, metric, config['citation_pattern'], config['legacy_citation_pattern'])
    output_directory.mkdir(parents=True, exist_ok=True)
    target = output_directory / 'squad_per_answer.csv'
    if target.exists():
        raise FileExistsError(f'Refusing to replace an existing scoring output: {target}')
    scores.to_csv(target, index=False)
    from importlib.metadata import version
    from .io import sha256_file
    write_json(output_directory / 'evaluation_inputs.json', inputs)
    write_json(output_directory / 'scoring_provenance.json', {
        'source_run_id': config['run_id'], 'framework': 'Hugging Face SQuAD', 'evaluate_version': version('evaluate'),
        'metric_source_sha256': sha256_file(config['squad']['metric_source']),
        'citation_pattern': config['citation_pattern'], 'legacy_citation_pattern': config['legacy_citation_pattern'],
        'scale': config['squad']['scale'], 'model_calls': 0})
    return scores


def rebuild_saved_comparison(config):
    """Recompute summaries from saved judge outcomes in memory, without calls."""
    import pandas as pd
    inputs, _, _, _ = load_saved_inputs(config)
    directory = Path(config['score_directory'])
    squad = pd.read_csv(directory / 'squad_per_answer.csv', dtype={'query_id': str})
    judge = load_saved_judge_scores(directory, inputs, config['judge']['metrics'], config['judge'])
    scores = combine_answer_scores(squad, judge, config['judge']['metrics'])
    settings = config['summary']
    summary, pairs = summarize_paired_scores(scores, config['arms'], settings['metrics'],
        settings['bootstrap_samples'], settings['bootstrap_seed'], settings['tie_tolerance'],
        settings['delta_label'], settings['confidence_quantiles'])
    return scores, summary, pairs
