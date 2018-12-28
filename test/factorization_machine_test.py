import unittest

import redshells


class FactorizationMachineTest(unittest.TestCase):
    def test_graph_shape(self):
        feature_size = 10
        embedding_size = 20
        graph = redshells.model.FactorizationMachineGraph(
            feature_size=feature_size, embedding_size=embedding_size, l2_weight=1e-5, model='binary_classification')
        self.assertEqual(graph.xw.shape.as_list(), [None, 1])
        self.assertEqual(graph.first_order.shape.as_list(), [None])
        self.assertEqual(graph.xv2.shape.as_list(), [None, embedding_size])
        self.assertEqual(graph.x2v2.shape.as_list(), [None, embedding_size])
        self.assertEqual(graph.second_order.shape.as_list(), [None])
        self.assertEqual(graph.y.shape.as_list(), [None])
        self.assertEqual(len(graph.regularization), 3)
        self.assertEqual(graph.loss.shape.as_list(), [])

    def test_graph_with_regression(self):
        feature_size = 10
        embedding_size = 20
        graph = redshells.model.FactorizationMachineGraph(
            feature_size=feature_size, embedding_size=embedding_size, l2_weight=1e-5, model='regression')
        self.assertEqual(graph.loss.shape.as_list(), [])


if __name__ == '__main__':
    unittest.main()
