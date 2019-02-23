import unittest
from unittest.mock import MagicMock

import luigi
import numpy as np

import redshells.app.word_item_similarity


class _DummyTask(luigi.Task):
    pass


class CalculateWordEmbeddingTest(unittest.TestCase):
    def setUp(self):
        self.input_data = dict()
        self.dump_data = None
        redshells.app.word_item_similarity.CalculateWordEmbedding.clear_instance_cache()

    def test_run(self):
        embedding_size = 10
        self.input_data['word'] = ['w1', 'w2']
        self.input_data['word2item'] = {'w1': ['item0', 'item1'], 'w2': ['item0', 'item2']}
        self.input_data['item2embedding'] = {f'item{i}': np.random.uniform(0, 1, size=embedding_size) for i in range(3)}

        task = redshells.app.word_item_similarity.CalculateWordEmbedding(
            word_task=_DummyTask(), word2item_task=_DummyTask(), item2embedding_task=_DummyTask())
        task.load = MagicMock(side_effect=self._load)
        task.dump = MagicMock(side_effect=self._dump)

        task.run()
        self.assertEqual(list(self.dump_data.keys()), ['w1', 'w2'])
        self.assertEqual(self.dump_data['w1'].shape, (embedding_size, ))

    def _load(self, *args, **kwargs):
        if 'target' in kwargs and kwargs['target'] is not None:
            return self.input_data.get(kwargs['target'], None)
        if len(args) > 0:
            return self.input_data.get(args[0], None)
        return self.input_data

    def _dump(self, *args):
        self.dump_data = args[0]


if __name__ == '__main__':
    unittest.main()
