import unittest

import numpy as np
import scipy.sparse as sp
from unittest.mock import patch

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

    @patch('redshells.model.GraphConvolutionalMatrixCompletion.get_user_feature')
    def test_get_user_feature_with_new_items(self, dummy_get_user_feature):
        n_users = 10
        n_items = 20
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
        n_user_embed_dimension = 50
        dummy_get_user_feature.return_value = np.zeros((len(user_ids) * len(item_ids), n_user_embed_dimension))
        user_features = model.get_user_feature_with_new_items(item_ids, additional_dataset=dataset, with_user_embedding=False)
        self.assertEqual(len(user_features[0]), n_users)
        self.assertEqual(user_features[1].shape, (n_users, n_user_embed_dimension))

    def test_get_item_feature_with_new_items(self):
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

        user_ids = [90, 62, 3, 3]
        item_ids = [11, 236, 240, 243]
        additional_item_features = {item_id: np.array([999]) for item_id in item_ids}
        additional_dataset = GcmcDataset(np.array(user_ids), np.array(item_ids), np.array([1, 2, 1, 1]), item_features=[additional_item_features])

        target_item_ids = item_ids + [12, 13, 17, 55]  # item_ids to get embeddings

        item_feature = model.get_item_feature_with_new_items(item_ids=target_item_ids, additional_dataset=additional_dataset)
        self.assertEqual(len(item_feature), 2)
        self.assertEqual(list(item_feature[0]), target_item_ids)
        self.assertEqual(item_feature[1].shape, (len(target_item_ids), encoder_size))
        output_embedding = {k: v for k, v in zip(*item_feature)}
        np.testing.assert_almost_equal(output_embedding[240], output_embedding[243])


if __name__ == '__main__':
    unittest.main()
