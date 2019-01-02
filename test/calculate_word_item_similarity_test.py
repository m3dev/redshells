import unittest
from io import StringIO
from unittest.mock import MagicMock

import luigi
import pandas as pd
import numpy as np

import redshells.app.word_item_similarity


class _DummyTask(luigi.Task):
    pass


class CalculateWordItemSimilarityTest(unittest.TestCase):
    def setUp(self):
        self.input_data = dict()
        self.dump_data = None
        redshells.app.word_item_similarity.CalculateWordItemSimilarity.clear_instance_cache()

    def test_run(self):
        prequery_return_size = 10
        return_size = 2
        embedding_size = 5
        word_size = 3
        item_size = 100

        np.random.seed(12)
        self.input_data['word2embedding'] = {
            f'word_{i}': np.random.uniform(-1, 1, embedding_size)
            for i in range(word_size)
        }
        self.input_data['item2embedding'] = {
            f'item_{i}': np.random.uniform(-1, 1, embedding_size)
            for i in range(item_size)
        }
        model = MagicMock()
        model.predict = lambda x: np.sum(x, axis=1)
        self.input_data['model'] = model

        task = redshells.app.word_item_similarity.CalculateWordItemSimilarity(
            word2embedding_task=_DummyTask(),
            item2embedding_task=_DummyTask(),
            similarity_model_task=_DummyTask(),
            prequery_return_size=prequery_return_size,
            return_size=return_size)
        task.load = MagicMock(side_effect=self._load)
        task.dump = MagicMock(side_effect=self._dump)

        task.run()

        # regression test
        expected = """
word,item,similarity
word_0,item_93,2.00833317836325
word_0,item_54,1.6076099461970408
word_1,item_33,3.232108807828523
word_1,item_42,2.585814339034334
word_2,item_75,2.243042210722768
word_2,item_89,1.61490367610421"""
        expected = pd.read_csv(StringIO(expected))
        pd.testing.assert_frame_equal(self.dump_data, expected, check_names=False, check_index_type=False)

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
