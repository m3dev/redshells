import unittest

import numpy as np
import scipy.sparse as sp

from redshells.model.gcmc_dataset import GcmcDataset, GcmcGraphDataset


def _make_sparse_matrix(n, m, n_values):
    x = np.zeros(shape=(n, m), dtype=np.float32)
    x[np.random.choice(range(n), n_values), np.random.choice(range(m), n_values)] = 1.0
    return sp.csr_matrix(x)


class TestGcmcGraphDataset(unittest.TestCase):
    def test(self):
        # This tests that GraphConvolutionalMatrixCompletion runs without error, and its loss and rmse are small enough.
        n_users = 101
        n_items = 233
        n_data = 3007
        am1 = _make_sparse_matrix(n_users, n_items, n_data)
        am2 = 2 * _make_sparse_matrix(n_users, n_items, n_data)
        adjacency_matrix = am1 + am2
        user_ids = adjacency_matrix.tocoo().row
        item_ids = adjacency_matrix.tocoo().col
        ratings = adjacency_matrix.tocoo().data
        item_features = [{i: np.array([i]) for i in range(n_items)}]
        rating_data = GcmcDataset(user_ids, item_ids, ratings, item_features=item_features)
        dataset = GcmcGraphDataset(dataset=rating_data, test_size=0.2)
        self.assertEqual((n_items + 1, 1), dataset.item_features[0].shape)  # because of default index.


if __name__ == '__main__':
    unittest.main()
