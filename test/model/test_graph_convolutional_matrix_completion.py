import unittest

import numpy as np
import scipy.sparse as sp

from redshells.model import GraphConvolutionalMatrixCompletion
from redshells.model.gcmc_dataset import GcmcDataset, GcmcGraphDataset


def _make_sparse_matrix(n, m, n_values):
    x = np.zeros(shape=(n, m), dtype=np.float32)
    x[np.random.choice(range(n), n_values), np.random.choice(range(m), n_values)] = 1.0
    return sp.csr_matrix(x)


class GraphConvolutionalMatrixCompletionTest(unittest.TestCase):
    def test_run(self):
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
        dataset = GcmcDataset(user_ids, item_ids, ratings)
        graph_dataset = GcmcGraphDataset(dataset, test_size=0.1)
        encoder_hidden_size = 100
        encoder_size = 100
        scope_name = 'GraphConvolutionalMatrixCompletionGraph'
        model = GraphConvolutionalMatrixCompletion(
            graph_dataset=graph_dataset,
            encoder_hidden_size=encoder_hidden_size,
            encoder_size=encoder_size,
            scope_name=scope_name,
            batch_size=1024,
            epoch_size=10,
            learning_rate=0.01,
            dropout_rate=0.7,
            normalization_type='symmetric')
        reports = model.fit()
        test_loss = float(reports[-1].split(',')[-2].split('=')[-1])
        test_rmse = float(reports[-1].split(',')[-1].split('=')[-1][:-1])
        self.assertLess(test_loss, 1.0)
        self.assertLess(test_rmse, 0.7)

    def test_item_cold_start(self):
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
        dataset = GcmcDataset(user_ids, item_ids, ratings, item_features=item_features)
        graph_dataset = GcmcGraphDataset(dataset, test_size=0.1)
        encoder_hidden_size = 100
        encoder_size = 100
        scope_name = 'GraphConvolutionalMatrixCompletionGraph'
        model = GraphConvolutionalMatrixCompletion(
            graph_dataset=graph_dataset,
            encoder_hidden_size=encoder_hidden_size,
            encoder_size=encoder_size,
            scope_name=scope_name,
            batch_size=1024,
            epoch_size=10,
            learning_rate=0.01,
            dropout_rate=0.7,
            normalization_type='symmetric')
        model.fit()

        user_ids = [90, 62]
        item_ids = [11, 236]  # 236 is new items
        additional_dataset = GcmcDataset(np.array(user_ids), np.array(item_ids), np.array([1, 2]), item_features=[{236: np.array([236])}])
        results = model.predict_with_new_items(user_ids, item_ids, additional_dataset=additional_dataset)
        self.assertEqual(2, len(results))
        self.assertIsNotNone(results[0])
        self.assertIsNotNone(results[1])


if __name__ == '__main__':
    unittest.main()
