"""Local, current-run notebook tables and walkthroughs. No hosted calls here."""
from html import escape
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_json, sha256_file
from .opportunity_study import generation_messages
from .notebook_display import display_table, details_html

METRICS = ['squad_token_f1_pct', 'faithfulness', 'answer_relevancy', 'answer_words']


class TutorialViews:
    def __init__(self, run):
        self.run = run

    def questions(self):
        if self.run.mode == 'plan':
            return pd.DataFrame(columns=['query_id', 'question', 'reference_answers', 'gold_ids', 'subgroup', 'diagnostic'])
        if self.run.mode == 'live':
            return pd.DataFrame(self.run.questions())
        return pd.DataFrame(read_json(self.run.folder / 'questions.json'))

    def candidates(self, query_id=None, frozen=False):
        if self.run.mode == 'plan':
            return pd.DataFrame(columns=['query_id', 'passage_id', 'text', 'rank', 'rerank_rank', 'is_known_support'])
        if frozen:
            path = self.run.upstream('targeted') / 'input/frozen_ranked_top30.parquet'
        else:
            path = self.run.folder / 'reranked_top30.parquet'
            if not path.exists():
                path = self.run.folder / 'retrieved_top30.parquet'
            if not path.exists() and self.run.study == 'answers':
                path = self.run.upstream('targeted') / 'input/frozen_ranked_top30.parquet'
        frame = pd.read_parquet(path)
        frame['query_id'] = frame.query_id.astype(str)
        frame = frame[frame.query_id.isin(set(self.questions().query_id))]
        if query_id is not None:
            frame = frame[frame.query_id.eq(str(query_id))]
        # Frozen reranker results are not new targeted-run rankings.
        if frozen and self.run.study == 'targeted':
            frame = frame.drop(columns=['rerank_rank', 'rerank_score'], errors='ignore')
        return frame.sort_values(['query_id', 'rank']).reset_index(drop=True)

    def answers(self, extension=False):
        path = self.run.folder / ('extension_inputs.json' if extension else 'generate_inputs.json')
        return pd.DataFrame(read_json(path)) if self.run.mode != 'plan' and path.exists() else pd.DataFrame()

    def scores(self, extension=False, main=False):
        path = self.run.folder / ('extension_scores' if extension else 'scores') / 'per_answer_scores.csv'
        frame = pd.read_csv(path, dtype={'query_id': str}) if self.run.mode != 'plan' and path.exists() else pd.DataFrame()
        if main and self.run.study == 'baseline' and not frame.empty:
            frame = frame[~frame.diagnostic.astype(str).str.lower().eq('true')]
        return frame

    def retrieval(self):
        if self.run.mode == 'plan':
            return pd.DataFrame()
        questions = self.questions().set_index('query_id')
        data = self.candidates()
        rows = []
        for query_id, docs in data.groupby('query_id', sort=False):
            q = questions.loc[query_id]; gold = set(q.gold_ids)
            if not gold:
                continue
            row = dict(query_id=query_id, question=q.question, subgroup=q.subgroup,
                       diagnostic=bool(q.diagnostic), gold_count=len(gold))
            for arm, column in [('embedding', 'rank'), ('reranker', 'rerank_rank')]:
                if column not in docs:
                    continue
                for k in [5, 30]:
                    selected = docs[docs[column].le(k)]
                    row[f'{arm}_recall_at_{k}'] = len(gold & set(selected.passage_id)) / len(gold)
                ranks = docs.loc[docs.passage_id.isin(gold), column]
                row[f'{arm}_first_support_rank'] = float(ranks.min()) if len(ranks) else np.nan
                top_ranks = ranks[ranks.le(5)].astype(int)
                row[f'{arm}_hit_at_5'] = float(len(top_ranks) > 0)
                row[f'{arm}_mrr_at_5'] = 1 / min(top_ranks) if len(top_ranks) else 0.
                ideal = sum(1 / np.log2(r + 1) for r in range(1, min(5, len(gold)) + 1))
                row[f'{arm}_ndcg_at_5'] = sum(1 / np.log2(r + 1) for r in top_ranks) / ideal
            row['opportunity'] = row['embedding_recall_at_30'] > row['embedding_recall_at_5'] + 1e-12
            if 'reranker_recall_at_5' in row:
                delta = row['reranker_recall_at_5'] - row['embedding_recall_at_5']
                row['recall_delta'] = delta
                row['outcome'] = 'gain' if delta > 1e-12 else 'loss' if delta < -1e-12 else 'unchanged'
            rows.append(row)
        return pd.DataFrame(rows)

    def completion(self):
        frame = self.answers()
        if frame.empty:
            return {'run_id': self.run.run_id, 'successful_answers': 0, 'complete_pairs': 0}
        pairs = frame.groupby('query_id').arm.agg(set)
        main_ids = set(self.questions().loc[~self.questions().diagnostic, 'query_id']) if self.run.study == 'baseline' else set(pairs.index)
        return dict(run_id=self.run.run_id, questions=len(pairs), successful_answers=len(frame),
                    complete_pairs=int(pairs.map(lambda arms: {'baseline', 'reranked'} <= arms).sum()),
                    main_complete_pairs=sum({'baseline', 'reranked'} <= arms for q, arms in pairs.items() if q in main_ids),
                    truncated_answers=int(frame.truncated.sum()),
                    request_outcomes=self.attempts().groupby('status').size().to_dict() if not self.attempts().empty else {})

    def attempts(self):
        if self.run.mode == 'plan':
            return pd.DataFrame()
        rows = [read_json(p) for p in sorted((self.run.folder / 'requests').glob('*/attempt-*.json'))]
        keys = ['stage', 'query_id', 'arm', 'attempt', 'status', 'http_status', 'elapsed_seconds', 'backoff_seconds']
        return pd.DataFrame([{key: r.get(key, 0. if key.endswith('_seconds') else '') for key in keys} for r in rows])

    def overlap(self):
        from .answers import score_custom_answer
        frame = self.answers()
        if frame.empty:
            return frame
        pattern = read_json(self.run.project / 'experiments/v3-answer-quality/config.json')['legacy_citation_pattern']
        return pd.DataFrame([dict(query_id=r['query_id'], question=r['question'], arm=r['arm'], diagnostic=r['diagnostic'],
                                  subgroup=r['subgroup'], **score_custom_answer(r['answer'], r['reference_answers'], pattern))
                             for r in frame.to_dict('records')])

    def paired(self, extension=False):
        frame = self.scores(extension, main=True)
        if frame.empty:
            return frame
        original = 'extension-original' if extension else 'baseline'
        arms = ['extension-evidence_check', 'extension-concise'] if extension else ['reranked']
        rows = []
        for subgroup, group in frame.groupby('subgroup', dropna=False):
            for arm in arms:
                for metric in [m for m in METRICS if m in frame]:
                    pivot = group.pivot(index='query_id', columns='arm', values=metric)
                    if original not in pivot or arm not in pivot:
                        continue
                    paired = pivot[[original, arm]].dropna()
                    if paired.empty:
                        continue
                    delta = paired[arm] - paired[original]
                    rng = np.random.default_rng(42)
                    means = np.array([rng.choice(delta.to_numpy(), len(delta), replace=True).mean() for _ in range(2000)])
                    lo, hi = np.quantile(means, [.025, .975])
                    rows.append(dict(subgroup=subgroup, comparison=arm, metric=metric, matched_pairs=len(paired),
                                     original_mean=paired[original].mean(), variant_mean=paired[arm].mean(),
                                     mean_delta=delta.mean(), bootstrap_low=lo, bootstrap_high=hi))
        return pd.DataFrame(rows)

    def context_rows(self, query_id='1053665'):
        data = self.candidates(query_id)
        columns = [c for c in ['passage_id', 'rank', 'rerank_rank', 'is_known_support', 'text', 'similarity_score', 'rerank_score'] if c in data]
        return data[columns]

    def walkthrough(self, query_id='1053665', extension=False, contexts=True):
        from IPython.display import display as _display, Markdown, HTML
        display = lambda value: display_table(value) if isinstance(value, pd.DataFrame) else _display(value)
        questions = self.questions(); q = questions[questions.query_id.eq(query_id)]
        if q.empty:
            display(Markdown(f'Question `{query_id}` is outside this notebook’s frozen sample.'))
            return
        display(Markdown(f'**{query_id}: {q.iloc[0].question}**'))
        answers = self.answers(extension)
        if answers.empty:
            display(self.context_rows(query_id))
            return
        rows = answers[answers.query_id.eq(query_id)]
        for row in rows.to_dict('records'):
            display(Markdown('#### ' + row['arm']))
            display(HTML('<pre style="white-space:pre-wrap;overflow-wrap:anywhere">' + escape(row['answer']) + '</pre>'))
            if contexts:
                display(HTML('<pre style="white-space:pre-wrap;overflow-wrap:anywhere">' + escape(row['context']) + '</pre>'))
        scores = self.scores(extension)
        if not scores.empty:
            display(scores[scores.query_id.eq(query_id)])

    def prompts(self, extension=False, query_id='1053665'):
        from IPython.display import display as _display, Markdown, HTML
        display = lambda value: display_table(value) if isinstance(value, pd.DataFrame) else _display(value)
        q = self.questions(); q = q[q.query_id.eq(query_id)]
        if q.empty:
            q = self.questions().head(1)
        if q.empty:
            return
        settings = self.run.generation
        if settings is None:
            settings = read_json(self.run.folder / 'manifest.json')['contract']['generation']
        variants = read_json(self.run.project / 'configs/generation_extension.json')['variants'] if extension else {'original': ''}
        docs = self.candidates(str(q.iloc[0].query_id))
        arms = {'reranked': 'rerank_rank'} if extension else {'baseline': 'rank', 'reranked': 'rerank_rank'}
        for arm, column in arms.items():
            if column not in docs:
                continue
            context = '\n\n'.join(f"[{d['passage_id']}]\n{d['text']}" for d in docs.sort_values(column).head(5).to_dict('records'))
            for variant, instruction in variants.items():
                messages = generation_messages(q.iloc[0].question, context, settings['system_prompt'] + ('\n\n' + instruction if instruction else ''))
                display(Markdown(f'**{arm} / {variant}**'))
                display(HTML('<pre style="white-space:pre-wrap;overflow-wrap:anywhere">' + escape(json.dumps(messages, ensure_ascii=False, indent=2)) + '</pre>'))

    def extension_results(self):
        from IPython.display import display as _display, Markdown
        display = lambda value: display_table(value) if isinstance(value, pd.DataFrame) else _display(value)
        frame = self.scores(extension=True)
        if frame.empty:
            return
        # Same matched 12 questions for every reported three-way metric mean.
        table = []
        for metric in [m for m in METRICS if m in frame]:
            pivot = frame.pivot(index='query_id', columns='arm', values=metric).dropna()
            for arm in pivot.columns:
                table.append(dict(metric=metric, arm=arm, matched_questions=len(pivot), mean=pivot[arm].mean()))
        display(pd.DataFrame(table)); display(self.paired(extension=True))
        examples = ['1053665']
        for metric, ascending in [('faithfulness', True), ('squad_token_f1_pct', True), ('squad_token_f1_pct', False)]:
            if metric not in frame:
                continue
            pivot = frame.pivot(index='query_id', columns='arm', values=metric)
            for arm in ['extension-evidence_check', 'extension-concise']:
                if arm not in pivot or 'extension-original' not in pivot:
                    continue
                delta = (pivot[arm] - pivot['extension-original']).dropna().sort_values(ascending=ascending)
                if ascending:
                    delta = delta[delta.lt(0)]
                for query_id in delta.index:
                    if query_id not in examples:
                        examples.append(query_id); break
                if len(examples) >= 3:
                    break
            if len(examples) >= 3:
                break
        from IPython.display import HTML
        for query_id in examples:
            selected = self.answers(extension=True)
            selected = selected[selected.query_id.eq(query_id)]
            if selected.empty:
                continue
            if selected.context.nunique() != 1:
                raise ValueError('Prompt variants must share exactly the same evidence.')
            display(Markdown(f'**{query_id}: {selected.iloc[0].question}**'))
            display(HTML('<pre style="white-space:pre-wrap;overflow-wrap:anywhere">' + escape(selected.iloc[0].context) + '</pre>'))
            cards = []
            for row in selected.to_dict('records'):
                cards.append('<section style="flex:1;min-width:240px"><h4>' + escape(row['arm']) + '</h4><pre style="white-space:pre-wrap;overflow-wrap:anywhere">' + escape(row['answer']) + '</pre></section>')
            display(HTML('<div style="display:flex;gap:16px;flex-wrap:wrap">' + ''.join(cards) + '</div>'))
            display(frame[frame.query_id.eq(query_id)][['arm', *[m for m in METRICS if m in frame]]])
        display(Markdown('**Manual check:** inspect claim support, completeness of requested parts, and appropriate abstention for each example.'))
        display(pd.DataFrame([dict(query_id=q, support='', completeness='', appropriate_abstention='') for q in examples]))

    def audit(self):
        if self.run.mode == 'plan':
            return self.run.plan
        manifest = read_json(self.run.folder / 'manifest.json')
        for relative, expected in manifest['contract']['upstream'].items():
            if sha256_file(self.run.project / relative) != expected:
                raise ValueError('A frozen input changed: ' + relative)
        questions = self.questions()
        assert len(questions) == self.run.plan['questions'] and questions.query_id.is_unique
        for extension in [False, True] if self.run.study == 'answers' else [False]:
            answers = self.answers(extension)
            if answers.empty:
                continue
            assert answers.run_id.eq(self.run.run_id).all()
            assert not answers.duplicated(['query_id', 'arm']).any()
            for row in answers.to_dict('records'):
                assert len(row['context_passage_ids']) == len(set(row['context_passage_ids'])) == 5
                assert row['context'] == '\n\n'.join(row['retrieved_contexts'])
                assert all(block.startswith(f'[{pid}]\n') for block, pid in zip(row['retrieved_contexts'], row['context_passage_ids']))
                if self.run.study != 'latency':
                    docs = self.candidates(row['query_id'])
                    column = 'rank' if row['arm'] == 'baseline' else 'rerank_rank'
                    assert row['context_passage_ids'] == docs.sort_values(column).head(5).passage_id.tolist()
            if extension:
                ids = sum(read_json(self.run.project / 'configs/generation_extension.json')['query_ids_by_subgroup'].values(), [])
                assert set(answers.query_id) == set(ids) and len(answers) == 36
        return dict(run_id=self.run.run_id, frozen_inputs='verified', questions=len(questions), context_ids='verified')

    def show(self, section, **kwargs):
        from IPython.display import display as _display, Markdown
        display = lambda value: display_table(value) if isinstance(value, pd.DataFrame) else _display(value)
        run = self.run
        if run.mode == 'plan':
            display(Markdown(f'**Plan:** {section}; no data loading or model calls.'))
            return
        if section == 'configuration':
            generation = run.generation or {}
            settings = [('Embedding model', run.retrieval['embedding_model']),
                        ('Embedding dimensions', run.retrieval['dimensions']),
                        ('Reranker model', run.retrieval['reranker_model']),
                        ('Generator model', generation.get('model', 'Not used in this stage') if run.study not in {'diagnosis'} else 'Not used in this stage'),
                        ('Candidate / context budget', '30 / 5 passages'),
                        ('Eligible questions in this experiment', run.plan['questions']),
                        ('Generation limit', str(generation.get('max_tokens', 400)) + ' tokens' if run.study not in {'diagnosis'} else 'Not used in this stage')]
            display(pd.DataFrame(settings, columns=['Setting','Value']))
        elif section == 'questions':
            questions = self.questions(); display(questions.groupby('subgroup').size().rename('questions'))
            display(questions[['query_id', 'question', 'subgroup', 'diagnostic']])
        elif section == 'corpus':
            if run.study == 'baseline':
                docs = read_json(run.upstream('baseline') / 'input/corpus.json')
                display({'passages': len(docs), 'passage_embedding_calls': 0}); display(pd.DataFrame(docs).head(3))
            else:
                display(read_json(run.upstream('prepared_retrieval') / 'assets_manifest.json'))
        elif section == 'document_embeddings':
            matrix = np.load(run.upstream('baseline') / 'document_embeddings.npy', mmap_mode='r')
            display({'matrix_shape': matrix.shape, 'dtype': str(matrix.dtype), 'new_document_embedding_calls': 0})
        elif section == 'input_contract':
            data = self.candidates(frozen=True)
            counts = data.groupby('query_id').passage_id.nunique()
            if not counts.eq(30).all() or len(counts) != len(self.questions()):
                raise ValueError('Each frozen question must have 30 unique candidate IDs')
            display({'questions': len(counts), 'candidate_rows': len(data), 'candidates_per_question': 30})
        elif section == 'completion': display(pd.DataFrame(self.completion().items(),columns=['Measure','Value']))
        elif section == 'retrieval': display(self.retrieval())
        elif section == 'retrieval_summary':
            frame = self.retrieval()
            if run.study == 'baseline' and not frame.empty: frame = frame[~frame.diagnostic]
            if not frame.empty:
                display(frame.groupby('subgroup').mean(numeric_only=True))
                if 'outcome' in frame: display(pd.crosstab(frame.subgroup, frame.outcome))
        elif section == 'opportunities':
            frame = self.retrieval(); display(frame[frame.opportunity].head(25))
        elif section == 'controls':
            frame = self.retrieval(); display(frame[~frame.opportunity & frame.outcome.eq('loss')])
        elif section == 'overlap': display(self.overlap())
        elif section == 'overlap_summary':
            frame = self.overlap()
            if run.study == 'baseline' and not frame.empty: frame = frame[~frame.diagnostic]
            if not frame.empty: display(frame.groupby(['subgroup','arm'])[['token_f1','normalized_em']].mean())
        elif section == 'example_scores':
            frame=self.scores()
            display(frame[[c for c in ['query_id','arm','squad_token_f1_pct','faithfulness','faithfulness_status','answer_relevancy','answer_relevancy_status'] if c in frame]])
        elif section == 'scores': display(self.scores(main=True)); display(self.paired())
        elif section == 'judge_outcomes':
            frame = self.scores()
            for metric in ['faithfulness', 'answer_relevancy']:
                if metric + '_status' in frame: display(frame.groupby(['arm', metric + '_status']).size().rename('outcomes'))
        elif section == 'walkthrough': self.walkthrough(**kwargs)
        elif section == 'contexts': display(self.context_rows(**kwargs))
        elif section == 'prompts': self.prompts(**kwargs)
        elif section == 'extension': self.extension_results()
        elif section == 'extension_contract':
            plan = read_json(run.project / 'configs/generation_extension.json')
            display(plan['query_ids_by_subgroup']); display(plan['provider_routing']); self.prompts(extension=True)
        elif section == 'manual_review':
            display(pd.DataFrame({'query_id':self.questions().query_id, 'correctness':'', 'completeness':'', 'evidence_support':''}))
        elif section == 'findings':
            pairs = self.paired()
            if not pairs.empty:
                display(pairs[[c for c in ['subgroup','metric','matched_pairs','mean_delta','bootstrap_low','bootstrap_high'] if c in pairs]])
                display(Markdown('Compare each subgroup separately. A confidence interval crossing zero does not establish a clear average improvement; retrieval gains do not guarantee better answers. Judge scores estimate support and relevance, while human correctness review remains separate.'))
        elif section == 'diagnostic_membership':
            ids=set(self.questions().query_id)
            display({q: q in ids for q in ['1053665','1100870']})
        elif section == 'attempts': display(self.attempts())
        elif section == 'latency':
            data = pd.read_csv(run.folder / 'latency.csv', dtype={'query_id':str})
            display(data); display(data.groupby('arm').mean(numeric_only=True))
        elif section == 'latency_stages':
            data = pd.read_csv(run.folder / 'latency.csv', dtype={'query_id':str})
            columns=[c for c in data if c.endswith('_seconds')]
            display(data.groupby('arm')[columns].median().mul(1000).add_suffix('_ms'))
        elif section == 'failures':
            attempts=self.attempts(); display(attempts[attempts.status.ne('ok')])
        elif section == 'latency_quality':
            latency = pd.read_csv(run.folder / 'latency.csv', dtype={'query_id':str})
            display(latency.merge(self.scores(), on=['query_id','arm','run_id'] if 'run_id' in self.scores() else ['query_id','arm'], validate='one_to_one'))
        elif section == 'haystack_crosscheck':
            from haystack import Document
            from haystack.components.evaluators import DocumentRecallEvaluator, DocumentMRREvaluator
            questions = self.questions().set_index('query_id'); data = self.candidates(); rows=[]
            for query_id, docs in data.groupby('query_id'):
                if not questions.loc[query_id].gold_ids: continue
                gold=[[Document(id=p,content=p) for p in questions.loc[query_id].gold_ids]]
                for arm,column in [('embedding','rank'),('reranker','rerank_rank')]:
                    found=[[Document(id=p,content=p) for p in docs.sort_values(column).head(5).passage_id]]
                    rows.append(dict(query_id=query_id,arm=arm,
                        recall=DocumentRecallEvaluator(mode='multi_hit',document_comparison_field='id').run(ground_truth_documents=gold,retrieved_documents=found)['score'],
                        mrr=DocumentMRREvaluator(document_comparison_field='id').run(ground_truth_documents=gold,retrieved_documents=found)['score']))
            measured=pd.DataFrame(rows); expected=self.retrieval().set_index('query_id')
            for row in rows:
                assert abs(row['recall']-expected.loc[row['query_id'],row['arm']+'_recall_at_5'])<1e-12
                assert abs(row['mrr']-expected.loc[row['query_id'],row['arm']+'_mrr_at_5'])<1e-12
            display(measured)
        elif section == 'audit':
            from IPython.display import HTML
            report=self.audit()
            display(pd.DataFrame([{'Check':'Frozen inputs, question IDs and evidence contexts','Status':'Verified'}]))
            display(HTML(details_html('Integrity details (optional)',report)))
        else: raise ValueError('Unknown notebook section: ' + section)
