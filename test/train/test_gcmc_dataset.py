import os
import unittest

import numpy as np
import pandas as pd
import tensorflow as tf
from logging import getLogger, config

from redshells.model.graph_convolutional_matrix_completion import GCMCDataset

logger = getLogger(__name__)


"""
class GCMCDataset(object):
    def __init__(self,
                 user_ids: np.ndarray,
                 item_ids: np.ndarray,
                 ratings: np.ndarray,
                 test_size: float,
                 user_features: Optional[Dict[Any, np.ndarray]] = None,
                 item_features: Optional[Dict[Any, np.ndarray]] = None) -> None:
"""


class TestGCMCDataset(unittest.TestCase):
    def test_without_information(self):
        user_ids = np.array([1, 1, 2, 2, 2])
        item_ids = np.array([1, 2, 1, 2, 3])
        ratings = np.array([1, 0, 1, 0, 1])
        test_size = 0.0
        dataset = GCMCDataset(user_ids, item_ids, ratings, test_size)
        _user_indices, _item_indices, _rating_one_hot, _rating, _user_information, _item_information = dataset.train_data()
        self.assertEqual(user_ids.shape, _user_indices.shape)
        self.assertEqual(item_ids.shape, _item_indices.shape)
        self.assertEqual((ratings.shape[0], 2), _rating_one_hot.shape)
        self.assertEqual(ratings.shape, _rating.shape)
        self.assertEqual((user_ids.shape[0], 0), _user_information.shape)
        self.assertEqual((item_ids.shape[0], 0), _item_information.shape)

    def test_with_information(self):
        user_ids = np.array([1, 1, 2, 2, 2])
        item_ids = np.array([1, 2, 1, 2, 3])
        ratings = np.array([1, 0, 1, 0, 1])
        test_size = 0.0
        user_information = {1: np.array([10, 11]), 2: np.array([20, 21])}
        item_information = {1: np.array([10, 11, 12]), 2: np.array([20, 21, 22]), 3: np.array([30, 31, 32])}
        dataset = GCMCDataset(user_ids, item_ids, ratings, test_size, user_information, item_information)
        _user_indices, _item_indices, _rating_one_hot, _rating, _user_information, _item_information = dataset.train_data()
        self.assertEqual(user_ids.shape, _user_indices.shape)
        self.assertEqual(item_ids.shape, _item_indices.shape)
        self.assertEqual((ratings.shape[0], 2), _rating_one_hot.shape)
        self.assertEqual(ratings.shape, _rating.shape)
        self.assertEqual((user_ids.shape[0], 2), _user_information.shape)
        self.assertEqual((item_ids.shape[0], 3), _item_information.shape)

if __name__ == '__main__':
    unittest.main()

