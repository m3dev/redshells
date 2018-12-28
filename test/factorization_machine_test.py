import unittest
import numpy as np
import pandas as pd
import redshells
import tensorflow as tf


class FactorizationMachineTest(unittest.TestCase):
    def test_graph_shape(self):
        input_size = 10
        feature_kind_size = 40
        embedding_size = 20
        graph = redshells.model.FactorizationMachineGraph(
            input_size=input_size,
            feature_kind_size=feature_kind_size,
            embedding_size=embedding_size,
            l2_weight=1e-5,
            learning_rate=1e-3,
            scope_name='test_graph_shape')
        self.assertEqual(graph.xw.shape.as_list(), [None, input_size])
        self.assertEqual(graph.first_order.shape.as_list(), [None])
        self.assertEqual(graph.xv2.shape.as_list(), [None, embedding_size])
        self.assertEqual(graph.x2v2.shape.as_list(), [None, embedding_size])
        self.assertEqual(graph.second_order.shape.as_list(), [None])
        self.assertEqual(graph.y.shape.as_list(), [None])
        self.assertEqual(len(graph.regularization), 3)
        self.assertEqual(graph.loss.shape.as_list(), [])

    def test_fit_run_without_error(self):
        real_value_size = 10
        categorical_value_size = 3
        embedding_size = 20
        data_size = 1024
        model = redshells.model.FactorizationMachine(
            embedding_size=embedding_size,
            l2_weight=1e-5,
            learning_rate=1e-3,
            batch_size=256,
            epoch_size=10,
            test_size=0.1,
            scope_name='test_fit_run_without_error')

        np.random.seed(21)
        tf.random.set_random_seed(32)
        data = pd.DataFrame(
            np.random.uniform(0, 1, size=[data_size, real_value_size]),
            columns=[f'real_{i}' for i in range(real_value_size)])
        for i in range(categorical_value_size):
            data[f'c_{i}'] = np.random.choice(['a', 'b', 'c'], size=data_size)
            data[f'c_{i}'] = data[f'c_{i}'].astype('category')
        data['y'] = np.random.uniform(0, 1, size=data_size)
        model.fit(data.drop('y', axis=1), data['y'])


if __name__ == '__main__':
    unittest.main()
