"""OpenRouter wire contracts shared by retrieval, judging and latency paths."""
import math
import os

QUERY_INSTRUCTION = 'Given a web search query, retrieve relevant passages that answer the query'


def require_model_calls():
    if os.environ.get('ALLOW_MODEL_CALLS') != '1':
        raise PermissionError('Model calls are disabled. Run tools/rebuild_results.py --allow-model-calls '
                              'to run offline tests first, then rebuild with OpenRouter.')


def embedding_payload(model, texts, dimensions, input_type='query', instruction=QUERY_INSTRUCTION):
    if input_type not in {'query', 'passage'}:
        raise ValueError('Embedding mode must be query or passage.')
    inputs = list(texts)
    if input_type == 'query' and instruction:
        inputs = [f'Instruct: {instruction}\nQuery: {text}' for text in inputs]
    return {'model': model, 'input': inputs, 'dimensions': dimensions, 'encoding_format': 'float'}


def rerank_payload(model, question, candidates):
    return {'model': model, 'query': question,
            'documents': [row['text'] for row in candidates], 'top_n': len(candidates)}


def rerank_scores(raw, count):
    """Validate full provider results; retain the study's internal score field name."""
    results = raw['results']
    indices = [row['index'] for row in results]
    if any(type(i) is not int for i in indices) or sorted(indices) != list(range(count)):
        raise ValueError('Reranker response must contain every candidate exactly once.')
    if not all(math.isfinite(float(row['relevance_score'])) for row in results):
        raise ValueError('Reranker response must contain finite relevance scores.')
    return sorted([{'index': row['index'], 'logit': float(row['relevance_score'])}
                   for row in results], key=lambda row: (-row['logit'], row['index']))
