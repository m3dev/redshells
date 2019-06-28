import unittest
from logging import getLogger
from unittest.mock import MagicMock

import numpy as np

import gokart

from redshells.model import FeatureAggregationSimilarityModel
from redshells.model.feature_aggregation_similarity_model import FeatureAggregationSimilarityDataset
from redshells.train import TrainFeatureAggregationSimilarityModel

logger = getLogger(__name__)


class _DummyTask(gokart.TaskOnKart):
    task_namespace = __file__


class TestTrainFeatureAggregationSimilarityModel(unittest.TestCase):
    def setUp(self):
        self.input_data = None
        self.dump_data = None

    def test_run(self):
        task = TrainFeatureAggregationSimilarityModel(
            dataset_task=_DummyTask(),
            embedding_size=7,
            learning_rate=0.001,
            batch_size=1,
            epoch_size=5,
            early_stopping_patience=2,
            max_data_size=7,
            test_size_rate=0.4)
        self.input_data = FeatureAggregationSimilarityDataset(
            x_item_indices=np.array([0, 1, 2]),
            y_item_indices=np.array([3, 2, 1]),
            x_item_features=np.array([[0, 1], [1, 2], [2, 4]]),
            y_item_features=np.array([[5, 0], [0, 4], [3, 2]]),
            scores=np.array([0.1, 0.2, 0.4]))
        task.load = MagicMock(side_effect=self._load)
        task.dump = MagicMock(side_effect=self._dump)
        task.run()
        self.assertIsInstance(self.dump_data, FeatureAggregationSimilarityModel)

    def _load(self, *args, **kwargs):
        if 'target' in kwargs:
            return self.__dict__.get(kwargs['target'], None)
        if len(args) > 0:
            return self.__dict__.get(args[0], None)
        return self.input_data

    def _dump(self, *args):
        self.dump_data = args[0]


if __name__ == '__main__':
    unittest.main()
