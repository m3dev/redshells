import unittest
from logging import getLogger

import numpy as np

from redshells.model.graph_convolutional_matrix_completion import GcmcDataset

logger = getLogger(__name__)


class TestGCMCDataset(unittest.TestCase):
    def test_without_information(self):
        user_ids = np.array([1, 1, 2, 2, 2])
        item_ids = np.array([1, 2, 1, 2, 3])
        ratings = np.array([1, 0, 1, 0, 1])
        test_size = 0.0
        dataset = GcmcDataset(user_ids, item_ids, ratings, test_size)
        data = dataset.train_data()
        self.assertEqual(user_ids.shape, data['user'].shape)
        self.assertEqual(item_ids.shape, data['item'].shape)
        self.assertEqual((ratings.shape[0], 2), data['label'].shape)
        self.assertEqual(ratings.shape, data['rating'].shape)
        self.assertEqual(user_ids.shape, data['user_information'].shape)
        self.assertEqual(item_ids.shape, data['item_information'].shape)

    def test_with_information(self):
        user_ids = np.array([1, 1, 2, 2, 2])
        item_ids = np.array([1, 2, 1, 2, 3])
        ratings = np.array([1, 0, 1, 0, 1])
        test_size = 0.0
        user_information = [{1: np.array([10, 11]), 2: np.array([20, 21])}]
        item_information = [{1: np.array([10, 11, 12]), 2: np.array([20, 21, 22]), 3: np.array([30, 31, 32])}]
        dataset = GcmcDataset(user_ids, item_ids, ratings, test_size, user_information, item_information)
        data = dataset.train_data()
        self.assertEqual(user_ids.shape, data['user'].shape)
        self.assertEqual(item_ids.shape, data['item'].shape)
        self.assertEqual((ratings.shape[0], 2), data['label'].shape)
        self.assertEqual(ratings.shape, data['rating'].shape)
        self.assertEqual(user_ids.shape, data['user_information'].shape)
        self.assertEqual(item_ids.shape, data['item_information'].shape)

    def test_with_click_threshold(self):
        user_ids = np.array([1, 1, 2, 2, 2, 3])
        item_ids = np.array([1, 2, 1, 2, 3, 1])
        ratings = np.array([1, 0, 1, 0, 1, 0])
        test_size = 0.0
        user_information = [{1: np.array([10, 11]), 2: np.array([20, 21]), 3: np.array([30, 31])}]
        item_information = [{1: np.array([10, 11, 12]), 2: np.array([20, 21, 22]), 3: np.array([30, 31, 32])}]
        dataset = GcmcDataset(
            user_ids, item_ids, ratings, test_size, user_information, item_information, min_user_click_count=3)
        data = dataset.train_data()
        np.testing.assert_almost_equal([0, 0, 1, 1, 1, 0], dataset.user_indices)
        np.testing.assert_almost_equal([1, 2, 1, 2, 3, 1], dataset.item_indices)
        self.assertEqual(item_ids.shape, dataset.item_indices.shape)
        self.assertEqual((ratings.shape[0], 2), data['label'].shape)
        self.assertEqual(ratings.shape, data['rating'].shape)
        self.assertEqual(user_ids.shape, data['user_information'].shape)
        self.assertEqual(item_ids.shape, data['item_information'].shape)


if __name__ == '__main__':
    unittest.main()
