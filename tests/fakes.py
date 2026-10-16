import numpy as np

class FakeIndex:
    def search(self, vector, count):
        return np.asarray([[1 - i / 100 for i in range(count)]], dtype='float32'), np.asarray([list(range(count))])


class FakeCorpus:
    def take(self, indices):
        import pyarrow as pa
        return pa.table({'passage_id': [f'p_{i:08x}' for i in indices], 'text': [f'passage {i}' for i in indices],
                         'source_url': ['https://example.test'] * len(indices), 'truncated': [False] * len(indices)})
