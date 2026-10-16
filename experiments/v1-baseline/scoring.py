import re, math, string
from collections import Counter
import pandas as pd

def retrieval_scores(ids, gold):
    gold = set(gold)
    assert gold
    hits = [int(x in gold) for x in ids]
    return {'recall': len(set(ids) & gold) / len(gold), 'hit': float(any(hits)),
            'mrr': next((1/(i+1) for i,h in enumerate(hits) if h), 0.0),
            'ndcg': sum(h/math.log2(i+2) for i,h in enumerate(hits)) /
                    sum(1/math.log2(i+2) for i in range(min(len(ids),len(gold))))}

def normalize(text):
    text = text.lower()
    text = re.sub(
        r'\[\s*p_[a-f0-9]+(?:\s*,\s*p_[a-f0-9]+)*\s*\]',
        ' ',
        text,
    )
    text = text.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())

def answer_scores(answer, references):
    a = normalize(answer).split()
    scores = []
    for reference in references:
        b = normalize(reference).split()
        overlap = sum((Counter(a) & Counter(b)).values())
        scores.append(2*overlap/(len(a)+len(b)) if a or b else 1.0)
    return {'normalized_em': float(any(normalize(answer)==normalize(r) for r in references)),
            'token_f1': max(scores)}

def score_results(questions, results):
    lookup = {q['query_id']: q for q in questions}
    rows = []
    for record in results:
        q = lookup[record['query_id']]
        pool = retrieval_scores([d['id'] for d in q['candidates']], q['gold_ids'])
        for arm, result in record['arms'].items():
            row = {'query_id': q['query_id'], 'arm': arm, 'status': result['status'],
                   'recall_at_30': pool['recall'], 'hit_at_30': pool['hit']}
            if result['status'] == 'ok':
                ids = [d['id'] for d in result['top_five']]
                row.update({k+'_at_5': v for k,v in retrieval_scores(ids, q['gold_ids']).items()})
                row.update(answer_scores(result['answer'], q['answers']))
                row.update({k: result[k] for k in ['rank_seconds','generation_seconds','branch_seconds','end_to_end_seconds']})
                citations = set(re.findall(r'\[(p_[a-f0-9]+)\]', result['answer']))
                row['cites_known_gold'] = bool(citations & set(q['gold_ids']))
                row['invalid_citation_count'] = len(citations - set(ids))
            rows.append(row)
    return pd.DataFrame(rows)

def aggregate(scores, reviews=None):
    # Paired complete cases prevent different denominators between arms.
    counts = scores[scores.status.eq('ok')].groupby('query_id').arm.nunique()
    eligible = set(counts[counts.eq(2)].index)
    paired = scores[scores.query_id.isin(eligible)]
    records = []
    for arm in ['baseline','reranked']:
        frame = paired[paired.arm.eq(arm)]
        row = {'arm': arm, 'paired_questions': len(frame),
               'failed_questions': int(((scores.arm == arm)&(scores.status != 'ok')).sum())}
        for metric in ['recall_at_30','hit_at_30','recall_at_5','hit_at_5','mrr_at_5','ndcg_at_5','normalized_em','token_f1']:
            row[metric] = frame[metric].mean() if metric in frame else float('nan')
        for metric in ['rank_seconds','generation_seconds','end_to_end_seconds']:
            row['median_'+metric] = frame[metric].median() if metric in frame else float('nan')
        row.update(reviewed_pairs=0, rubric_pass_rate=float('nan'))
        if reviews is not None:
            reviewed = reviews.copy()
            reviewed['correct'] = pd.to_numeric(reviewed['correct'], errors='coerce')
            reviewed = reviewed[reviewed.query_id.astype(str).isin(eligible) & reviewed.correct.isin([0,1])]
            pairs = reviewed.groupby('query_id').arm.nunique()
            common = set(pairs[pairs.eq(2)].index)
            reviewed = reviewed[reviewed.query_id.isin(common)&reviewed.arm.eq(arm)]
            row.update(reviewed_pairs=len(reviewed), rubric_pass_rate=reviewed.correct.mean())
        records.append(row)
    return pd.DataFrame(records).set_index('arm')

def haystack_crosscheck(questions, results):
    from haystack import Document
    from haystack.components.evaluators import DocumentRecallEvaluator, DocumentMRREvaluator, AnswerExactMatchEvaluator
    lookup = {q['query_id']: q for q in questions}
    output = []
    for record in results:
        q = lookup[record['query_id']]
        for arm, result in record['arms'].items():
            if result['status'] != 'ok': continue
            args = {'ground_truth_documents': [[Document(id=x, content=x) for x in q['gold_ids']]],
                    'retrieved_documents': [[Document(id=d['id'], content=d['id']) for d in result['top_five']]]}
            output.append({'query_id': q['query_id'], 'arm': arm,
                'haystack_recall_at_5': DocumentRecallEvaluator(mode='multi_hit').run(**args)['score'],
                'haystack_mrr_at_5': DocumentMRREvaluator().run(**args)['score'],
                'haystack_raw_em': max(AnswerExactMatchEvaluator().run(ground_truth_answers=[a],
                                        predicted_answers=[result['answer']])['score'] for a in q['answers'])})
    return pd.DataFrame(output)
