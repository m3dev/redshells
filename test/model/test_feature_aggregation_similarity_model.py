import unittest
from logging import getLogger

import numpy as np

from redshells.model import FeatureAggregationSimilarityModel
from redshells.model.feature_aggregation_similarity_model import FeatureAggregationSimilarityDataset

logger = getLogger(__name__)


class TestFeatureAggregationSimilarityModel(unittest.TestCase):
    def test(self):
        model = FeatureAggregationSimilarityModel(embedding_size=7, learning_rate=0.001, feature_size=2, item_size=4, max_feature_index=5)
        dataset = FeatureAggregationSimilarityDataset(
            x_item_indices=np.array([0, 1, 2]),
            y_item_indices=np.array([3, 2, 1]),
            x_item_features=np.array([[0, 1], [1, 2], [2, 4]]),
            y_item_features=np.array([[5, 0], [0, 4], [3, 2]]),
            scores=np.array([0.1, 0.2, 0.4]),
            batch_size=1)

        model.fit(dataset=dataset, epoch_size=5, early_stopping_patience=2, test_size_rate=0.4)

        similarities = model.calculate_similarity(
            x_item_indices=dataset.x_item_indices,
            y_item_indices=dataset.y_item_indices,
            x_item_features=dataset.x_item_features,
            y_item_features=dataset.y_item_features)
        self.assertEqual(similarities.shape, (3, ))

        embeddings = model.calculate_embeddings(item_features=np.array([[1, 2], [0, 0]]))
        self.assertNotEqual(np.sum(embeddings[0]**2), 0)
        self.assertEqual(np.sum(embeddings[1]**2), 0)  # feature index 0 must be masked index.


if __name__ == '__main__':
    unittest.main()
