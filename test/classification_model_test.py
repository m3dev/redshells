import unittest
from unittest.mock import MagicMock

import luigi
import pandas as pd
import numpy as np
import sklearn

import redshells


class _DummyTask(luigi.Task):
    pass


class PairwiseSimilarityModelTest(unittest.TestCase):
    def setUp(self):
        self.input_data = None
        self.dump_data = None
        redshells.train.TrainClassificationModel.clear_instance_cache()

    def test_train(self):
        data_size = 20
        np.random.seed(1)
        features = [np.random.rand(3) for _ in range(data_size)]
        category = [np.random.choice(['a', 'b', 'c']) for _ in range(data_size)]

        self.input_data = pd.DataFrame(dict(features=features, category=category))
        task = redshells.train.TrainClassificationModel(
            train_data_task=_DummyTask(), model_name='RandomForestClassifier')
        task.load = MagicMock(side_effect=self._load)
        task.dump = MagicMock(side_effect=self._dump)

        task.run()
        self.assertIsInstance(self.dump_data, sklearn.ensemble.RandomForestClassifier)

    def test_cv(self):
        data_size = 20
        np.random.seed(1)
        features = [np.random.rand(3) for _ in range(data_size)]
        category = [np.random.choice(['a', 'b', 'c']) for _ in range(data_size)]

        self.input_data = pd.DataFrame(dict(features=features, category=category))
        task = redshells.train.ValidateClassificationModel(
            train_data_task=_DummyTask(), model_name='RandomForestClassifier', cross_validation_size=3)
        task.load = MagicMock(side_effect=self._load)
        task.dump = MagicMock(side_effect=self._dump)

        task.run()
        self.assertIsInstance(self.dump_data, list)

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
