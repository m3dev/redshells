import unittest

import numpy as np
import tensorflow as tf

import redshells


class MatrixFactorizationTest(unittest.TestCase):
    def test_graph_shape(self):
        n_items = 30
        n_users = 10
        n_latent_factors = 5
        n_services = 3
        reg_item = 1e-5
        reg_user = 1e-5
        scope_name = 'test_graph'

        graph = redshells.model.MatrixFactorizationGraph(
            n_items=n_items,
            n_users=n_users,
            n_latent_factors=n_latent_factors,
            n_services=n_services,
            reg_item=reg_item,
            reg_user=reg_user,
            scope_name=scope_name,
            use_l2_upper_regularization=True,
            average=0.5,
            standard_deviation=0.5)

        self.assertEqual(graph.item_biases.shape.as_list(), [None, 1])
        self.assertEqual(graph.user_biases.shape.as_list(), [None, 1])
        self.assertEqual(graph.item_factors.shape.as_list(), [None, n_latent_factors])
        self.assertEqual(graph.user_factors.shape.as_list(), [None, n_latent_factors])
        self.assertEqual(graph.predictions.shape.as_list(), [None])

    def test_fit_and_predict_run_without_error(self):
        n_items = 30
        n_users = 10
        n_latent_factors = 5
        n_services = 3
        learning_rate = 1e-3
        reg_item = 1e-5
        reg_user = 1e-5
        min_value = 0.0
        max_value = 1.0
        batch_size = 256
        epoch_size = 10
        test_size = 0.1
        data_size = 1024

        model = redshells.model.MatrixFactorization(
            n_latent_factors=n_latent_factors,
            learning_rate=learning_rate,
            reg_item=reg_item,
            reg_user=reg_user,
            batch_size=batch_size,
            epoch_size=epoch_size,
            bias_epoch_size=epoch_size,
            test_size=test_size,
            scope_name='test_fit_run_without_error',
            use_l2_upper_regularization=True)

        np.random.seed(21)
        tf.random.set_random_seed(32)
        user_ids = list(np.random.randint(0, n_users, size=data_size))
        item_ids = list(np.random.randint(0, n_items, size=data_size))
        service_ids = list(np.random.randint(0, n_services, size=data_size))
        ratings = list(np.random.uniform(min_value, max_value, size=data_size))
        model.fit(user_ids=user_ids, item_ids=item_ids, service_ids=service_ids, ratings=ratings)

        prediction_data_size = 20
        prediction_user_ids = list(np.random.randint(0, n_users, size=prediction_data_size))
        prediction_item_ids = list(np.random.randint(0, n_items, size=prediction_data_size))
        prediction_service_ids = list(np.random.randint(0, n_services, size=prediction_data_size))

        # assign user_id which is not included in train data
        prediction_user_ids[0] = n_users + 1
        prediction_item_ids[2] = n_items + 1
        prediction_service_ids[4] = n_services + 1

        predictions = model.predict(
            user_ids=prediction_user_ids, item_ids=prediction_item_ids, service_ids=prediction_service_ids)

        self.assertTrue(np.isnan(predictions[0]))
        self.assertTrue(np.isnan(predictions[2]))
        self.assertTrue(np.isnan(predictions[4]))
        self.assertTrue(0 <= predictions[1] <= 1)

    def test_get_item_factors_run_without_error(self):
        n_items = 30
        n_users = 10
        n_latent_factors = 5
        n_services = 3
        learning_rate = 1e-3
        reg_item = 1e-5
        reg_user = 1e-5
        min_value = 0.0
        max_value = 1.0
        batch_size = 256
        epoch_size = 10
        test_size = 0.1
        data_size = 1024

        model = redshells.model.MatrixFactorization(
            n_latent_factors=n_latent_factors,
            learning_rate=learning_rate,
            reg_item=reg_item,
            reg_user=reg_user,
            batch_size=batch_size,
            epoch_size=epoch_size,
            bias_epoch_size=epoch_size,
            test_size=test_size,
            scope_name='test_fit_run_without_error',
            use_l2_upper_regularization=True)

        np.random.seed(21)
        tf.random.set_random_seed(32)
        user_ids = list(np.random.randint(0, n_users, size=data_size))
        item_ids = list(np.random.randint(0, n_items, size=data_size))
        service_ids = list(np.random.randint(0, n_services, size=data_size))
        ratings = list(np.random.uniform(min_value, max_value, size=data_size))
        model.fit(user_ids=user_ids, item_ids=item_ids, service_ids=service_ids, ratings=ratings)

        target_data_size = 20
        target_item_ids = list(np.random.randint(0, n_items, size=target_data_size))

        # assign user_id which is not included in train data
        target_item_ids[2] = n_items + 1
        factors = model.get_item_factors(item_ids=target_item_ids)
        np.testing.assert_almost_equal(factors[2], np.zeros(n_latent_factors))
        self.assertEqual(factors.shape, (target_data_size, n_latent_factors))


if __name__ == '__main__':
    unittest.main()
