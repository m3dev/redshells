import unittest
from unittest.mock import MagicMock

import luigi
import pandas as pd
import xgboost
from sklearn.ensemble import RandomForestClassifier

from redshells.train import TrainPairwiseSimilarityModel


class _DummyTask(luigi.Task):
    pass


class TrainPairwiseSimilarityModelTest(unittest.TestCase):
    def setUp(self):
        self.input_data = dict()
        self.dump_data = None
        TrainPairwiseSimilarityModel.clear_instance_cache()

    def test_run(self):
        self.input_data['item2embedding'] = dict(i0=[1, 2], i1=[3, 4])
        self.input_data['similarity_data'] = pd.DataFrame(
            dict(item1=['i0', 'i0', 'i1'], item2=['i0', 'i1', 'i1'], similarity=[1, 0, 1]))

        task = TrainPairwiseSimilarityModel(
            item2embedding_task=_DummyTask(),
            similarity_data_task=_DummyTask(),
            model_name='RandomForestClassifier',
            item0_column_name='item1',
            item1_column_name='item2',
            similarity_column_name='similarity')
        task.load = MagicMock(side_effect=self._load)
        task.dump = MagicMock(side_effect=self._dump)

        task.run()
        self.assertIsInstance(self.dump_data, RandomForestClassifier)

    def _load(self, *args, **kwargs):
        if 'target' in kwargs:
            return self.input_data.get(kwargs['target'], None)
        if len(args) > 0:
            return self.input_data.get(args[0], None)
        return self.input_data

    def _dump(self, *args):
        self.dump_data = args[0]


if __name__ == '__main__':
    unittest.main()
