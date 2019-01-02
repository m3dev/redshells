import unittest
from unittest.mock import MagicMock

import luigi
import pandas as pd
import numpy as np

import redshells.app.word_item_similarity


class _DummyTask(luigi.Task):
    pass


class TrainPairwiseSimilarityModelTest(unittest.TestCase):
    def setUp(self):
        self.input_data = dict()
        self.dump_data = None
        redshells.app.word_item_similarity.MakeSimilarityData.clear_instance_cache()

    def test_run(self):
        np.random.seed(12)
        items = [f'item_{i}' for i in range(100)]
        words = [f'word_{i}' for i in range(100)]

        self.input_data['word2items'] = {w: np.random.choice(items, np.random.randint(1, 11)) for w in words}
        self.input_data['similarity'] = pd.DataFrame(
            dict(
                item_id_0=np.random.choice(items, size=1000, replace=True),
                item_id_1=np.random.choice(items, size=1000, replace=True),
                similarity=np.random.uniform(0, 1, size=1000)))

        task = redshells.app.word_item_similarity.MakeSimilarityData(
            word2items_task=_DummyTask(),
            similarity_task=_DummyTask(),
            item_id_0_column_name='item_id_0',
            item_id_1_column_name='item_id_1',
            similarity_column_name='similarity')
        task.load = MagicMock(side_effect=self._load)
        task.dump = MagicMock(side_effect=self._dump)

        task.run()
        self.assertIsInstance(self.dump_data, pd.DataFrame)

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
