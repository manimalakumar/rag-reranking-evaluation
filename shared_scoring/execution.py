"""Notebook startup: load published artifacts, otherwise prepare a separate fresh study."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .fresh_retrieval import (
    RecordedOpenRouterClient, execution_settings, fingerprint, retrieve_full_validation,
    ensure_retrieval_assets,
)
from .io import read_json, write_json, sha256_file
from .opportunity_study import load_opportunity_configuration, load_opportunity_population


def ensure_baseline_study(config, project, client=None, rows=None):
    """Use the saved baseline, otherwise run its missing pipeline or scoring stages."""
    import shutil
    from .fresh_baseline import build_baseline_run, prepare_baseline_scores, repair_baseline_artifacts
    scoped_settings = execution_settings(project, include_scope=True)
    if scoped_settings.get('scope'):
        from .scoped_baseline import prepare_scoped_baseline
        config = prepare_scoped_baseline(config, project, scoped_settings)
    run = config['source_directory']/'runs'/config['run_id']
    required = [run/name for name in ['results.json','questions.json','config.json','run_manifest.json',
        'candidates.json','document_embedding_origin.json','document_embedding_attempts.csv','manual_review.csv','haystack_crosscheck.csv','audit.json']]
    score_files = ['evaluation_inputs.json','diagnostic_inputs.json','input_audit.json','squad_per_answer.csv',
        'per_question_scores.csv','paired_comparison.csv','diagnostic_squad.csv','original_run_failures.json',
        'scorer_discrepancies.csv','scorer_edge_cases.json','evaluation_attempt_history.csv','evaluation_audit.json']
    if all(path.is_file() for path in required) and all((config['score_directory']/name).is_file() for name in score_files):
        return config
    settings = scoped_settings
    contract = {'execution': _portable(settings, settings['runtime_directory']),
                'baseline': read_json(config['source_directory']/'config.json'),
                'membership_sha256':sha256_file(config['source_directory']/'input/baseline_ids.json'),
                'reviewed_input_sha256':sha256_file(config['source_directory']/'input/golden_reviewed.json')}
    folder = settings['runtime_directory']/'baseline'/fingerprint(contract)[:20]
    current = deepcopy(config)
    fresh_run=folder/'runs'/('baseline-fresh-'+folder.name)
    if all((fresh_run/path.name).is_file() for path in required) and all((folder/'standard_scoring'/name).is_file() for name in score_files):
        manifest=read_json(fresh_run/'run_manifest.json')
        for name,expected in manifest['contract']['input_sha256'].items():
            if sha256_file(folder/'input'/name)!=expected:
                raise ValueError('Fresh baseline input checksum mismatch.')
        import os
        current.update(source_directory=folder,run_id='baseline-fresh-'+folder.name,
            score_directory=folder/'standard_scoring',fresh_run=True,
            historical_reference_directory=config['source_directory']/'reference',
            document_embedding_example=os.path.relpath(folder/'document_embedding_example.json',project))
        print(f'Loaded the prepared baseline: {current["run_id"]}; no model calls.',flush=True)
        return current
    saved_pipeline = all((run/name).is_file() for name in ['results.json','config.json','run_manifest.json'])
    if saved_pipeline:
        for result in read_json(run/'results.json'):
            for arm,value in result['arms'].items():
                if value['status']=='ok' and not(run/'requests'/f"{result['query_id']}-{arm}-generation.json").is_file():
                    saved_pipeline=False
    if not all(path.is_file() for path in required) and saved_pipeline:
        print('Saved answers exist; recovering missing local display artifacts without model calls.',flush=True)
        current=repair_baseline_artifacts(config,project,folder)
        current['recovered_run']=True
    elif not all(path.is_file() for path in required):
        print('Saved baseline pipeline artifacts are missing; preparing a separate fresh baseline.', flush=True)
        current = build_baseline_run(config, project, folder, client, rows)
    else:
        print('Saved answers are available; rebuilding missing deterministic scoring outputs only.', flush=True)
        current['fresh_run'] = True
    current['score_directory'] = folder/'standard_scoring'
    current['squad']['cache_directory'] = settings['runtime_directory']/'scoring'
    if isinstance(current['squad']['metric_source'],Path) and not current['squad']['metric_source'].is_file():
        current['squad']['metric_source'] = 'squad'
    if config['score_directory'].is_dir() and saved_pipeline:
        # Carry forward saved judge successes/failures when only local scoring files are missing.
        if current.get('recovered_run'):
            for name in ['judge_manifest.json','ragas_results','judge_requests','judge_sessions','ragas_attempt_history']:
                original=config['score_directory']/name
                target=current['score_directory']/name
                if original.is_dir():
                    shutil.copytree(original,target,dirs_exist_ok=True)
                elif original.is_file():
                    target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(original,target)
        else:
            shutil.copytree(config['score_directory'],current['score_directory'],dirs_exist_ok=True)
    if not all((current['score_directory']/name).is_file() for name in score_files):
        prepare_baseline_scores(current)
    return current


def _portable(value, base):
    if isinstance(value, Path):
        import os
        return os.path.relpath(value.resolve(), Path(base).resolve())
    if isinstance(value, dict):
        return {key: _portable(item, base) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_portable(item, base) for item in value]
    return value


def _runtime_config(config, settings, topic):
    identity = read_json(config['configuration_file'])
    folder = (settings['runtime_directory'] / topic / fingerprint({'config': identity, 'execution': _portable(settings, settings['runtime_directory'])})[:20]).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    result = deepcopy(config)
    scoring_only=topic.endswith('-scoring')
    result.update(study_directory=folder, configuration_file=folder/'config.json',
                  run_id=config['run_id'] if scoring_only else f'{topic}-fresh-{folder.name}',
                  fresh_run=not scoring_only)
    if scoring_only:
        result.update(fresh_scores=True,artifact_directory=config['study_directory'])
    if settings.get('scope'):
        result['eligible_question_ids_file'] = settings['scope']['eligible_question_ids']
        result['eligible_question_count'] = settings['scope']['eligible_question_count']
        result['retrieval_scope'] = '152,576 cached passages; questions restricted to frozen fully covered IDs'
    result['score_directory'] = folder/'output/standard_scoring'
    result['squad']['cache_directory'] = settings['runtime_directory']/'scoring'
    if isinstance(result['squad']['metric_source'], Path) and not result['squad']['metric_source'].is_file():
        result['squad']['metric_source'] = 'squad'
    return result


def _save_config(config):
    write_json(config['configuration_file'], _portable(
        {key: value for key, value in config.items() if key not in ['configuration_file', 'study_directory']},
        config['study_directory']))


def _seal_inputs(config, provenance):
    import os
    folder = config['study_directory']
    if config.get('eligible_question_ids_file'):
        allowed = set(Path(config['eligible_question_ids_file']).read_text().splitlines())
        actual = set(pd.read_parquet(config['questions_file']).query_id.astype(str))
        if not actual <= allowed:
            raise ValueError('Study contains questions outside the fully covered eligible pool.')
    write_json(folder/'input/provenance.json', provenance)
    checksums = {
        str(path.relative_to(folder)): sha256_file(path) for path in (folder/'input').iterdir()
        if path.is_file() and path.name != 'file_checksums.json'}
    for key in ['questions_file','passages_file','parent_configuration']:
        if isinstance(config.get(key),Path) and config[key].is_file():
            checksums[os.path.relpath(config[key],folder)] = sha256_file(config[key])
    write_json(folder/'input/file_checksums.json',checksums)
    write_json(folder/'output/input_verification.json', {'run_id': config['run_id'],
        'origin': 'Fresh data, generated by shared helpers; published artifacts were not replaced.',
        'questions': len(pd.read_parquet(config['questions_file'])), 'human_review': 'pending'})


def _retrieval_directory(settings):
    contract = {'dataset': settings['dataset'], 'retrieval': settings['retrieval']}
    topic = 'full-validation'
    if settings.get('scope'):
        contract['scope'] = _portable(settings['scope'], settings['runtime_directory'])
        topic = 'cached-subset'
    return settings['runtime_directory']/topic/fingerprint(contract)[:20]


def select_cohorts(metrics, settings):
    """Select the configured seeded opportunity and control samples."""
    from .cached_scope import assert_eligible
    assert_eligible(metrics, settings)
    cfg = settings['cohorts']
    labeled = metrics[metrics.has_known_support_labels]
    gap = labeled.embedding_recall_at_30 - labeled.embedding_recall_at_5
    opportunity_pool = labeled[gap.gt(cfg['tolerance'])]
    opportunity_count = cfg.get('opportunity_questions', len(opportunity_pool))
    if len(opportunity_pool) < opportunity_count:
        raise ValueError(f'Only {len(opportunity_pool)} eligible opportunities; configuration requires {opportunity_count}.')
    opportunities = opportunity_pool.sample(n=opportunity_count, random_state=cfg['seed'])
    pool = labeled[gap.abs().le(cfg['tolerance'])]
    if len(pool) < cfg['control_questions']:
        raise ValueError(f'Only {len(pool)} eligible controls; configuration requires {cfg["control_questions"]}.')
    controls = pool.sample(n=cfg['control_questions'], random_state=cfg['seed'])
    parts = []
    for cohort, group, pilot in [('opportunity', opportunities, cfg['pilot_opportunities']),
                                  ('control', controls, cfg['pilot_controls'])]:
        first = min(pilot, len(group))
        batches = [group.iloc[:first]]
        if first < len(group):
            if cfg['expansion_batches'] < 1:
                raise ValueError('Expansion is disabled but the selected cohort exceeds its initial batch.')
            batches.extend(group.iloc[indices] for indices in
                           np.array_split(np.arange(first, len(group)), cfg['expansion_batches']))
        for index, batch in enumerate(batches, 1):
            frame = batch.copy() if isinstance(batch, pd.DataFrame) else group.iloc[batch].copy()
            if frame.empty:
                continue
            frame['cohort'], frame['batch_id'] = cohort, index
            frame['batch_name'] = 'pilot' if index == 1 else f'expansion_{index-1}'
            parts.append(frame)
    if opportunities.empty:
        raise ValueError('No opportunities were found; a targeted reranking study cannot be sampled.')
    return pd.concat(parts).sort_values(['batch_id', 'cohort', 'query_row']).reset_index(drop=True)


def _summary_group(frame, batch_id, cohort):
    rank_change = frame.first_support_rank_improvement
    return {'batch_id': batch_id, 'cohort': cohort, 'questions': len(frame),
        'embedding_mean_recall_at_5': frame.embedding_recall_at_5.mean(),
        'reranker_mean_recall_at_5': frame.reranker_recall_at_5.mean(),
        'mean_recall_at_5_change': frame.recall_at_5_change.mean(),
        'gains': int(frame.outcome.eq('gain').sum()), 'unchanged': int(frame.outcome.eq('unchanged').sum()),
        'losses': int(frame.outcome.eq('loss').sum()),
        'first_support_rank_improved': int(rank_change.gt(0).sum()),
        'first_support_rank_unchanged': int(rank_change.eq(0).sum()),
        'first_support_rank_worse': int(rank_change.lt(0).sum()),
        'median_rerank_seconds': frame.rerank_elapsed_seconds.median(),
        'p95_rerank_seconds': frame.rerank_elapsed_seconds.quantile(.95)}


def _build_targeted(config, settings, client=None, rows=None):
    metrics, shards, assets = retrieve_full_validation(settings, _retrieval_directory(settings), client, rows)
    selected = select_cohorts(metrics, settings)
    client = client or RecordedOpenRouterClient(settings['retrieval'], config['study_directory']/'output/rerank_requests')
    cohort_by_id = selected.set_index('query_id')
    questions, passage_parts = [], []
    # Read only selected rows from each shard rather than materializing all 3 million candidates.
    for shard in shards:
        candidates = pd.read_parquet(shard, filters=[('query_id', 'in', selected.query_id.tolist())])
        for qid, frame in candidates.groupby('query_id', sort=False):
            question = {**cohort_by_id.loc[qid].to_dict(), 'query_id': str(qid)}
            frame = frame.sort_values('rank').copy()
            ranks, trace = client.rerank(question['question'], frame.to_dict('records'))
            rank_by_index = {item['index']: (rank, float(item['logit'])) for rank, item in enumerate(ranks, 1)}
            frame['rerank_rank'] = [rank_by_index[i][0] for i in range(len(frame))]
            frame['rerank_score'] = [rank_by_index[i][1] for i in range(len(frame))]
            frame['cohort'], frame['batch_id'] = question['cohort'], question['batch_id']
            frame['rerank_elapsed_seconds'] = trace['elapsed_seconds']
            top = settings['retrieval']['context_count']
            embedding_ids = frame.passage_id.iloc[:top].tolist()
            ranked = frame.sort_values('rerank_rank')
            reranked_ids = ranked.passage_id.iloc[:top].tolist()
            gold = set(question['known_supporting_passage_ids'])
            reranked_recall = len(gold & set(reranked_ids))/len(gold)
            delta = reranked_recall - question['embedding_recall_at_5']
            embed_first = frame.loc[frame.is_known_support, 'rank'].min()
            rerank_first = ranked.loc[ranked.is_known_support, 'rerank_rank'].min()
            question.update(embedding_top5_passage_ids=embedding_ids, reranker_top5_passage_ids=reranked_ids,
                reranker_recall_at_5=reranked_recall, recall_at_5_change=delta,
                outcome='gain' if delta > settings['cohorts']['tolerance'] else 'loss' if delta < -settings['cohorts']['tolerance'] else 'unchanged',
                embedding_first_support_rank=embed_first, reranker_first_support_rank=rerank_first,
                first_support_rank_improvement=embed_first-rerank_first,
                known_supports_in_frozen_top30=len(gold & set(frame.passage_id)),
                known_supports_absent_frozen_top30=len(gold - set(frame.passage_id)),
                rerank_elapsed_seconds=trace['elapsed_seconds'], rerank_attempts=trace['attempt'],
                dataset_question=question['question'])
            questions.append(question)
            passage_parts.append(frame)
    population = pd.DataFrame(questions).sort_values(['batch_id', 'cohort', 'query_row']).reset_index(drop=True)
    if set(population.query_id) != set(selected.query_id):
        raise ValueError('Fresh retrieval did not produce the entire selected cohort.')
    folder = config['study_directory']
    (folder/'input').mkdir(exist_ok=True)
    (folder/'output').mkdir(exist_ok=True)
    config.update(questions_file=folder/'input/questions.parquet', passages_file=folder/'input/frozen_ranked_top30.parquet',
                  checkpoint_file=folder/'output/generation_attempts.sqlite',
                  retrieval_summary_file=folder/'output/retrieval_summary.csv',
                  expected_cohort_counts=population.cohort.value_counts().to_dict())
    population.to_parquet(config['questions_file'], index=False)
    pd.concat(passage_parts, ignore_index=True).to_parquet(config['passages_file'], index=False)
    summaries = [_summary_group(population, 0, 'all_batches')]
    summaries += [_summary_group(group, 0, name) for name, group in population.groupby('cohort')]
    for number, group in population.groupby('batch_id'):
        summaries.append(_summary_group(group, int(number), 'all'))
        summaries += [_summary_group(part, int(number), name) for name, part in group.groupby('cohort')]
    pd.DataFrame(summaries).to_csv(config['retrieval_summary_file'], index=False)
    review = population.head(config['review_examples_per_outcome']).copy()
    review['manual_relevance_assessment'] = 'pending'
    review['manual_notes'] = 'Fresh run: no human review has been performed.'
    review.to_parquet(folder/'output/prior_retrieval_manual_review.parquet', index=False)
    cfg = settings['retrieval']
    provenance = {'source_study': 'Fresh full-validation retrieval and targeted reranking',
        'source_file_sha256': {'per_query_metrics.parquet':sha256_file(_retrieval_directory(settings)/'per_query_metrics.parquet')
                              if (_retrieval_directory(settings)/'per_query_metrics.parquet').exists() else fingerprint(metrics.to_dict('records'))},
        'retrieval_asset_sha256':assets.get('sha256',{}),
        'rerank_request_sha256':{str(path.relative_to(folder)):sha256_file(path)
                                for path in (folder/'output/rerank_requests').glob('*/attempt-*.json')},
        'embedding_model': cfg['embedding_model'],
        'corpus_version': {**settings['dataset'], **assets['contract']['dataset']},
        'retrieval_settings': {**cfg, 'embedding_dimensions': cfg['dimensions'],
                               'actual_nlist': assets['actual_nlist'], 'nprobe': assets['nprobe']},
        'batch_counts': population.groupby(['batch_id','batch_name','cohort']).size().rename('questions').reset_index().to_dict('records'),
        'opportunity_questions': int(population.cohort.eq('opportunity').sum()),
        'control_questions': int(population.cohort.eq('control').sum()), 'total_questions': len(population),
        'frozen_candidate_rows': sum(map(len, passage_parts)), 'selection_seed': settings['cohorts']['seed'],
        'reference_policy': 'All original and well-formed answers, deduplicated in recorded order.',
        'human_review': 'pending', 'generation_calls': 0}
    if config['inspect_query_id'] not in set(population.query_id):
        config['inspect_query_id'] = str(population.iloc[0].query_id)
    _seal_inputs(config, provenance)
    _save_config(config)
    return config


def ensure_targeted_study(configuration_file, project, client=None, rows=None):
    """Load a complete saved retrieval study, otherwise build it from pinned raw data."""
    config = load_opportunity_configuration(configuration_file)
    required = [config['questions_file'], config['passages_file'], config['retrieval_summary_file'],
                config['study_directory']/'input/provenance.json',
                config['study_directory']/'output/prior_retrieval_manual_review.parquet']
    if all(path.is_file() for path in required):
        return config
    settings = execution_settings(project, include_scope=True)
    fresh = _runtime_config(config, settings, 'targeted-reranking')
    if fresh['configuration_file'].exists():
        previous = load_opportunity_configuration(fresh['configuration_file'])
        if all(path.is_file() for path in [previous['questions_file'], previous['passages_file'], previous['retrieval_summary_file']]):
            print(f'Loaded the previously prepared fresh study: {previous["run_id"]}')
            return previous
    if all(path.is_file() for path in [config['questions_file'],config['passages_file'],config['study_directory']/'input/provenance.json']):
        print('Saved rankings exist; rebuilding missing local summary/review files without model calls.',flush=True)
        population=load_opportunity_population(config)
        folder=fresh['study_directory']
        (folder/'output').mkdir(exist_ok=True)
        fresh['retrieval_summary_file']=folder/'output/retrieval_summary.csv'
        summaries=[_summary_group(population,0,'all_batches')]
        summaries += [_summary_group(group,0,name) for name,group in population.groupby('cohort')]
        for number,group in population.groupby('batch_id'):
            summaries.append(_summary_group(group,int(number),'all'))
            summaries += [_summary_group(part,int(number),name) for name,part in group.groupby('cohort')]
        pd.DataFrame(summaries).to_csv(fresh['retrieval_summary_file'],index=False)
        review=population.head(config['review_examples_per_outcome']).copy()
        review['manual_relevance_assessment']='pending'
        review['manual_notes']='Human review file unavailable; no labels or notes were fabricated.'
        review.to_parquet(folder/'output/prior_retrieval_manual_review.parquet',index=False)
        # Existing answer/scoring records keep their original paths and identities.
        fresh.update(run_id=config['run_id'],checkpoint_file=config['checkpoint_file'],score_directory=config['score_directory'])
        _seal_inputs(fresh,read_json(config['study_directory']/'input/provenance.json'))
        _save_config(fresh)
        return fresh
    print('Saved targeted study is incomplete. Preparing full validation retrieval and reranking; hosted calls may run.', flush=True)
    return _build_targeted(fresh, settings, client, rows)


def _generation_provenance(config, parent):
    settings = config['generation']
    extras = settings['extras']
    return {'source_study': 'Fresh paired generation using separately frozen contexts',
        'generator_settings': {'model': settings['model'], 'system_prompt': settings['system_prompt'],
            'temperature': settings['temperature'], 'max_tokens': settings['max_tokens'],
            'top_p': extras.get('top_p'), 'seed': extras.get('seed'),
            'thinking_enabled': False, 'stream': extras.get('stream')},
        'context_policy': 'Each arm uses its own exact saved top five passages; the two contexts are never combined.',
        'reference_policy': 'All original and well-formed references, deduplicated in recorded order.',
        'source_file_sha256': {}, 'parent_file_sha256': {str(parent[key].name): sha256_file(parent[key])
            for key in ['questions_file','passages_file','configuration_file']},
        'human_review': 'pending'}


async def ensure_answer_quality_study(configuration_file, project):
    """Reuse saved answers, or sample the parent and generate only missing native arms."""
    from .paired_generation import select_generation_sample, assign_generation_strata
    from .opportunity_study import run_opportunity_generation, run_opportunity_squad, load_generation_records, current_generation_records
    config = load_opportunity_configuration(configuration_file)
    required = [config['questions_file'], config['passages_file'], config['checkpoint_file'],
                config['generation_manifest_file'], config['manual_review_file'], config['parent_provenance_file']]
    if all(path.exists() for path in required):
        if not (config['score_directory']/'squad_per_answer.csv').is_file():
            config = _runtime_config(config, execution_settings(project, include_scope=True), 'answer-quality-scoring')
            _save_config(config)
            run_opportunity_squad(config)
        return config
    parent = ensure_targeted_study(config['parent_configuration'], project)
    settings = execution_settings(project, include_scope=True)
    fresh = _runtime_config(config, settings, 'answer-quality')
    folder = fresh['study_directory']
    if fresh['configuration_file'].exists():
        fresh = load_opportunity_configuration(fresh['configuration_file'])
    else:
        population = load_opportunity_population(parent)
        sample = select_generation_sample(population, config['sampling']['strata'],
            config['sampling']['questions_per_stratum'], config['sampling']['seed'], config['summary']['tie_tolerance'])
        sample['evaluation_references'] = sample.reference_answers
        arms = list(config['arms'])
        sample['generation_order'] = [' → '.join(arms if i % 2 == 0 else arms[::-1]) for i in range(len(sample))]
        (folder/'input').mkdir(exist_ok=True)
        (folder/'output').mkdir(exist_ok=True)
        sample.to_parquet(folder/'input/questions.parquet', index=False)
        fresh.update(questions_file=folder/'input/questions.parquet', passages_file=parent['passages_file'],
            checkpoint_file=folder/'output/generation_attempts.sqlite', checkpoint_format='native',
            parent_configuration=parent['configuration_file'], parent_provenance_file=parent['study_directory']/'input/provenance.json',
            generation_manifest_file=folder/'input/provenance.json', manual_review_file=folder/'output/source_manual_review.parquet',
            input_verification_file=folder/'output/input_verification.json',
            expected_cohort_counts=sample.cohort.value_counts().to_dict(), retrieval_summary_file=parent['retrieval_summary_file'])
        review = sample.iloc[:0].copy()
        for key in ['embedding_correct','reranked_correct']:
            review[key] = pd.Series(dtype='float64')
        review.to_parquet(fresh['manual_review_file'], index=False)
        if fresh['inspect_query_id'] not in set(sample.query_id):
            fresh['inspect_query_id'] = str(sample.iloc[0].query_id)
        provenance = _generation_provenance(fresh, parent)
        strata = assign_generation_strata(population, config['sampling']['strata'], config['summary']['tie_tolerance'])
        provenance.update(population_counts=strata.value_counts().to_dict(),
            sample_counts=sample.generation_stratum.value_counts().to_dict(),
            control_definitions={'all_support_already_present': 'Control with Recall@5 = 1 and no support absent from the top 30.',
                                 'some_support_absent_top30': 'Control with annotated support absent from the top 30.'})
        _seal_inputs(fresh, provenance)
        _save_config(fresh)
    recorded = current_generation_records(load_generation_records(fresh))
    expected = {(qid,arm) for qid in load_opportunity_population(fresh).query_id for arm in fresh['arms']}
    if set(recorded) != expected:
        print(f'Preparing missing answer pairs for {fresh["run_id"]}; saved native successes will be reused.', flush=True)
        await run_opportunity_generation(fresh, allow_model_calls=True)
    else:
        print(f'Loaded the prepared answer study: {fresh["run_id"]}; no generation calls.', flush=True)
    if any(row.get('status') == 'ok' for row in load_generation_records(fresh)):
        try:
            if not (fresh['score_directory']/'squad_per_answer.csv').exists():
                run_opportunity_squad(fresh)
        except ValueError as error:
            if 'No complete saved answer pairs' not in str(error):
                raise
    return fresh


def ensure_latency_assets(config, project, client=None, rows=None):
    """Build missing corpus/index assets before timing; return their actual contract."""
    from copy import deepcopy
    import os
    settings = execution_settings(project, include_scope=True)
    supplied = {name: os.environ.get(env) for name, env in config['benchmark']['asset_env'].items()}
    if not settings.get('scope') and all(supplied.values()) and all(Path(path).is_file() for path in supplied.values()):
        return config, Path(supplied['corpus']).resolve(), Path(supplied['index']).resolve()
    embedding=config['benchmark']['embedding_settings']
    retrieval=config['benchmark']['retrieval_settings']
    settings['retrieval'].update(embedding_model=embedding['model'],dimensions=embedding['dimensions'],
        api_base=embedding['endpoint'].removesuffix('/embeddings'),
        query_instruction=embedding.get('query_instruction', settings['retrieval'].get('query_instruction')), 
        passage_char_budget=retrieval['passage_char_budget'],nlist=retrieval['nlist'],
        passage_batch_size=retrieval['passage_batch_size'],query_batch_size=retrieval['query_batch_size'])
    assets = ensure_retrieval_assets(settings, _retrieval_directory(settings), client, rows)
    current = deepcopy(config)
    benchmark = current['benchmark']
    benchmark.update(asset_sizes=assets['asset_sizes'], corpus_rows=assets['corpus_rows'])
    benchmark['embedding_settings']['dimensions'] = assets['dimensions']
    benchmark['retrieval_settings'].update(actual_nlist=assets['actual_nlist'], nprobe=assets['nprobe'])
    current['fresh_asset_contract'] = {key: assets[key] for key in ['sha256','corpus_rows','dimensions','actual_nlist','nprobe']}
    return current, assets['corpus_file'], assets['index_file']


async def ensure_latency_study(configuration_file, project):
    """Load saved measurements, otherwise run both paths using prepared raw-data assets."""
    from .latency_study import select_latency_sample
    from .latency_benchmark import run_new_latency_benchmark
    from .opportunity_study import run_opportunity_squad
    config = load_opportunity_configuration(configuration_file)
    required = [config['questions_file'], config['checkpoint_file'], config['generation_manifest_file'],
                config['study_directory']/'input/provenance.json', *config['latency_files'].values()]
    if all(path.exists() for path in required):
        if not (config['score_directory']/'squad_per_answer.csv').is_file():
            config = _runtime_config(config, execution_settings(project, include_scope=True), 'latency-scoring')
            # Loading measurements must continue to use the published study directory.
            config['latency_provenance_file'] = Path(configuration_file).parent/'input/provenance.json'
            _save_config(config)
            run_opportunity_squad(config)
        return config
    fresh = _runtime_config(config, execution_settings(project, include_scope=True), 'latency-tradeoff')
    folder = fresh['study_directory']
    if fresh['configuration_file'].exists():
        fresh = load_opportunity_configuration(fresh['configuration_file'])
        if all(path.exists() for path in [fresh['questions_file'],fresh['checkpoint_file'],fresh['generation_manifest_file'],
                                           *fresh['latency_files'].values()]):
            if not(fresh['score_directory']/'squad_per_answer.csv').exists():
                run_opportunity_squad(fresh)
            print(f'Loaded the prepared latency study: {fresh["run_id"]}; no benchmark calls.',flush=True)
            return fresh
    parent = await ensure_answer_quality_study(config['parent_configuration'], project)
    if not fresh['questions_file'].exists() or not fresh['configuration_file'].exists():
        sample = select_latency_sample(load_opportunity_population(parent), config['sampling']['strata'],
            config['sampling']['questions_per_stratum'], config['sampling']['seed'], config['arms'])
        (folder/'input').mkdir(exist_ok=True)
        (folder/'output').mkdir(exist_ok=True)
        sample.to_parquet(folder/'input/questions.parquet', index=False)
        fresh.update(questions_file=folder/'input/questions.parquet', checkpoint_file=folder/'output/source/arm_checkpoints',
            parent_configuration=parent['configuration_file'], passages_file=parent['passages_file'],
            expected_cohort_counts=sample.cohort.value_counts().to_dict(),
            generation_manifest_file=folder/'input/input_manifest.json',
            input_verification_file=folder/'output/input_verification.json', latency_source_directory=folder/'output/source')
        if fresh['inspect_query_id'] not in set(sample.query_id):
            fresh['inspect_query_id'] = str(sample.iloc[0].query_id)
        _seal_inputs(fresh, {'source_study': 'Fresh latency benchmark; preparation excluded from timing', 'human_review': 'pending'})
        _save_config(fresh)
    fresh, corpus, index = ensure_latency_assets(fresh, project)
    benchmark_dir = folder/'output/source'
    run_new_latency_benchmark(fresh, benchmark_dir, corpus, index, allow_model_calls=True)
    protocol_manifest = read_json(config['generation_manifest_file']) if config['generation_manifest_file'].exists() else {}
    # Copy definitions only; historical asset hashes and run identities do not describe a fresh run.
    source_manifest = {key:protocol_manifest[key] for key in ['paths_measured','timing_definitions'] if key in protocol_manifest}
    source_manifest.update(run_id=fresh['run_id'],sampling=fresh['sampling'],
        protocol_source_sha256=sha256_file(config['generation_manifest_file']) if config['generation_manifest_file'].exists() else None,
        corpus_version={'unique_passages': fresh['benchmark']['corpus_rows'], **execution_settings(project, include_scope=True)['dataset']},
        embedding_settings=fresh['benchmark']['embedding_settings'], reranker_settings=fresh['benchmark']['reranker_settings'],
        generator_settings=_generation_provenance(fresh, parent)['generator_settings'], execution_settings=fresh['benchmark'])
    source_manifest.setdefault('paths_measured', {arm: fresh['benchmark']['stages_by_arm'][arm] for arm in fresh['arms']})
    source_manifest.setdefault('timing_definitions', {'successful_path_seconds': 'Successful hosted requests plus local work',
        'total_end_to_end_seconds': 'All attempts, actual backoff and local work'})
    write_json(fresh['generation_manifest_file'], source_manifest)
    manifest_hash = sha256_file(fresh['generation_manifest_file'])
    write_json(benchmark_dir/'run_metadata.json', {'origin': 'Fresh benchmark', 'benchmark': fresh['benchmark'],
        'human_review': 'pending', 'input_manifest_sha256': manifest_hash})
    import os
    _seal_inputs(fresh, {'source_study':'Fresh latency benchmark; preparation excluded from timing', 'human_review':'pending',
        'manifest_sha256':manifest_hash, 'parent_files_sha256':{
            os.path.relpath(parent['questions_file'],folder):sha256_file(parent['questions_file']),
            os.path.relpath(parent['configuration_file'],folder):sha256_file(parent['configuration_file'])}})
    (benchmark_dir/'excluded_preflight_failures.jsonl').write_text('')
    names = {'results': 'results.parquet', 'stages': 'stages.parquet', 'attempts': 'attempts.parquet',
        'paired': 'paired_latency_changes.parquet', 'summary': 'latency_summary_by_arm.csv', 'stage_summary': 'stage_summary.csv',
        'request_bodies': 'request_bodies.jsonl', 'raw_responses': 'raw_responses.jsonl',
        'preflight_failures': 'excluded_preflight_failures.jsonl', 'run_metadata': 'run_metadata.json', 'run_summary': 'run_summary.json'}
    fresh['latency_files'] = {name: benchmark_dir/filename for name, filename in names.items()}
    fresh['latency_plot_file'] = folder/'output/latency_quality.png'
    _save_config(fresh)
    run_opportunity_squad(fresh)
    return fresh


def ensure_diagnostic_data(root, project, client=None, rows=None):
    """Load the diagnostic bundle, or derive it from a fresh embedding-only search."""
    root, project = Path(root), Path(project)
    names = ['opportunity_records.csv','embedding_candidates.csv.gz','annotated_support.csv',
             'example_support_excerpt.csv','provenance.json','reported_full_validation_summary.json']
    if all((root/'data'/name).is_file() for name in names):
        return root/'data'
    settings = execution_settings(project, include_scope=True)
    folder = settings['runtime_directory']/'diagnosis'/_retrieval_directory(settings).name/'data'
    if all((folder/name).is_file() for name in names):
        return folder
    metrics, shards, assets = retrieve_full_validation(settings, _retrieval_directory(settings), client, rows)
    cfg = settings['retrieval']
    labeled = metrics[metrics.has_known_support_labels]
    opportunity = labeled[(labeled.embedding_recall_at_30-labeled.embedding_recall_at_5).gt(settings['cohorts']['tolerance'])].copy()
    if opportunity.empty:
        raise ValueError('The fresh retrieval produced no opportunity questions. Full metrics remain available in the runtime directory.')
    selected_ids = opportunity.query_id.tolist()
    ranked = pd.concat([pd.read_parquet(path,filters=[('query_id','in',selected_ids)]) for path in shards],ignore_index=True)
    records = opportunity.rename(columns={'embedding_recall_at_5':'recall_at_5','embedding_recall_at_30':'recall_at_30',
                                          'recall_gap_30_minus_5':'recall_30_minus_recall_5'})
    records['known_supporting_passage_ids'] = records.known_supporting_passage_ids.map(lambda ids:json.dumps(list(ids)))
    folder.mkdir(parents=True,exist_ok=True)
    records.to_csv(folder/'opportunity_records.csv',index=False)
    ranked[['query_id','passage_id','rank']].rename(columns={'rank':'embedding_rank'}).to_csv(folder/'embedding_candidates.csv.gz',index=False)
    pd.DataFrame([{'query_id':row.query_id,'passage_id':pid} for row in opportunity.itertuples()
                  for pid in row.known_supporting_passage_ids]).to_csv(folder/'annotated_support.csv',index=False)
    inspect_id=settings['diagnosis']['inspect_query_id']
    if inspect_id not in selected_ids:
        inspect_id=selected_ids[0]
    excerpt=ranked[ranked.query_id.eq(inspect_id)&ranked.is_known_support][['query_id','passage_id','rank','text']].rename(columns={'rank':'embedding_rank'})
    excerpt.to_csv(folder/'example_support_excerpt.csv',index=False)
    summary={'validation_questions':len(metrics),'annotated_questions':len(labeled),
        'mean_recall_at_5':labeled.embedding_recall_at_5.mean(),'mean_recall_at_30':labeled.embedding_recall_at_30.mean(),
        'mean_recall_gap':(labeled.embedding_recall_at_30-labeled.embedding_recall_at_5).mean(),
        'all_support_top5_questions':int(labeled.embedding_recall_at_5.ge(1-settings['cohorts']['tolerance']).sum()),
        'all_support_top30_questions':int(labeled.embedding_recall_at_30.ge(1-settings['cohorts']['tolerance']).sum()),
        'opportunity_questions':len(opportunity),
        'some_support_absent_top30_questions':int(labeled.embedding_recall_at_30.lt(1-settings['cohorts']['tolerance']).sum()),
        'source':'Computed from the complete fresh validation search. Published speaker-reported figures were preserved separately.'}
    write_json(folder/'reported_full_validation_summary.json',summary)
    write_json(folder/'provenance.json',{'fresh_run':True,'inspect_query_id':inspect_id,
        'retrieval_settings':{**cfg,'embedding_dimensions':cfg['dimensions'],'actual_nlist':assets['actual_nlist'],'nprobe':assets['nprobe']},
        'dataset':settings['dataset'],'source_file_sha256':{path.name:sha256_file(path) for path in folder.iterdir() if path.is_file()},
        'origin':'Fresh embedding-only retrieval; no reranker, generator or RAGAS calls.'})
    return folder
