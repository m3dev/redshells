from logging import getLogger
from typing import List, Any, Dict

import numpy as np
import sklearn
import tensorflow as tf

import redshells
from redshells.model.early_stopping import EarlyStopping

logger = getLogger(__name__)


class MatrixFactorizationGraph(object):
    def __init__(self, n_items: int, n_users: int, n_latent_factors: int, n_services: int, learning_rate: float,
                 reg_item: float, reg_user: float, scope_name: str) -> None:
        self.n_items = n_items
        self.n_users = n_users
        self.n_services = n_services
        self.n_latent_factors = n_latent_factors

        # placeholder
        self.input_items = tf.placeholder(dtype=np.int32, shape=[None], name='input_items')
        self.input_users = tf.placeholder(dtype=np.int32, shape=[None], name='input_users')
        self.input_services = tf.placeholder(dtype=np.int32, shape=[None], name='input_services')
        self.input_ratings = tf.placeholder(dtype=np.float32, shape=[None], name='input_ratings')

        # matrix factorization
        scale = np.sqrt(2.0 / self.n_latent_factors)
        initializer = tf.random_uniform_initializer(0, scale)

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            self.item_bias_embedding = tf.keras.layers.Embedding(
                input_dim=n_items, output_dim=1, embeddings_initializer=tf.constant_initializer(0.1), name='item_bias')
            self.item_biases = self.item_bias_embedding(self.input_items)

            self.user_bias_embedding = tf.keras.layers.Embedding(
                input_dim=n_users * n_services,
                output_dim=1,
                embeddings_initializer=tf.constant_initializer(0.1),
                name='user_bias')
            self.user_biases = self.user_bias_embedding(self.input_users + self.input_services * n_users)

            self.item_embedding = tf.keras.layers.Embedding(
                input_dim=n_items,
                output_dim=n_latent_factors,
                embeddings_initializer=initializer,
                embeddings_regularizer=tf.contrib.layers.l2_regularizer(reg_item),
                name='item_embedding')
            self.item_factors = self.item_embedding(self.input_items)

            self.user_embedding = tf.keras.layers.Embedding(
                input_dim=n_users,
                output_dim=n_latent_factors,
                embeddings_initializer=initializer,
                embeddings_regularizer=tf.contrib.layers.l2_regularizer(reg_user),
                name='user_embedding')
            self.user_factors = self.user_embedding(self.input_users)

        self.bias_values = tf.squeeze(self.item_biases + self.user_biases, 1)
        self.latent_values = tf.reduce_sum(tf.multiply(self.item_factors, self.user_factors), axis=1)
        self.predictions = self.bias_values + self.latent_values

        self.regularization = [
            self.item_embedding.embeddings_regularizer(self.item_factors),
            self.user_embedding.embeddings_regularizer(self.user_factors)
        ]

        squared_error = tf.losses.mean_squared_error(self.input_ratings, self.predictions)
        self.loss = tf.add_n([squared_error] + self.regularization, name='loss')
        self.error = tf.sqrt(squared_error, name='error')

        optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate=learning_rate)
        self.op = optimizer.apply_gradients(optimizer.compute_gradients(self.loss))


class MatrixFactorization(object):
    def __init__(self,
                 n_latent_factors: int,
                 learning_rate: float,
                 reg_item: float,
                 reg_user: float,
                 batch_size: int,
                 epoch_size: int,
                 test_size: float,
                 scope_name: str,
                 save_directory_path: str = None,
                 n_items=None,
                 n_users=None,
                 n_services=None,
                 max_value=None,
                 min_value=None,
                 user2index=None,
                 item2index=None,
                 service2index=None) -> None:
        self.n_latent_factors = n_latent_factors
        self.learning_rate = learning_rate
        self.reg_item = reg_item
        self.reg_user = reg_user
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.test_size = test_size
        self.scope_name = scope_name
        self.save_directory_path = save_directory_path
        self.n_items = n_items
        self.n_users = n_users
        self.n_services = n_services
        self.max_value = max_value
        self.min_value = min_value
        self.user2index = user2index
        self.item2index = item2index
        self.service2index = service2index
        self.session = tf.Session()
        self.graph = None

    def fit(self, user_ids: List[Any], item_ids: List[Any], service_ids: List[Any], ratings: List[float]) -> None:
        if self.graph is None:
            logger.info('making graph...')
            self.n_users = len(set(user_ids))
            self.n_items = len(set(item_ids))
            self.n_services = len(set(service_ids))
            self.min_value = np.min(ratings)
            self.max_value = np.max(ratings)
            self.user2index = dict(zip(np.unique(user_ids), range(self.n_users)))
            self.item2index = dict(zip(np.unique(item_ids), range(self.n_items)))
            self.service2index = dict(zip(np.unique(service_ids), range(self.n_services)))
            self.graph = self._make_graph()
            logger.info('done making graph')

        user_indices = self._convert(user_ids, self.user2index)
        item_indices = self._convert(item_ids, self.item2index)
        service_indices = self._convert(service_ids, self.service2index)

        user_train, user_test, item_train, item_test, service_train, service_test, rating_train, rating_test = sklearn.model_selection.train_test_split(
            user_indices, item_indices, service_indices, ratings, test_size=self.test_size)

        early_stopping = EarlyStopping(self.save_directory_path)

        with self.session.as_default():
            logger.info('initializing valiables...')
            self.session.run(tf.global_variables_initializer())
            dataset = tf.data.Dataset.from_tensor_slices((user_train, item_train, service_train, rating_train))
            dataset = dataset.batch(self.batch_size)
            iterator = dataset.make_initializable_iterator()
            next_batch = iterator.get_next()

            logger.info('start to optimize...')
            for i in range(self.epoch_size):
                self.session.run(iterator.initializer)
                train_loss = None
                train_error = None
                while True:
                    try:
                        user_, item_, service_, rating_ = self.session.run(next_batch)
                        feed_dict = {
                            self.graph.input_users: user_,
                            self.graph.input_items: item_,
                            self.graph.input_services: service_,
                            self.graph.input_ratings: rating_
                        }
                        _, train_loss, train_error = self.session.run(
                            [self.graph.op, self.graph.loss, self.graph.error], feed_dict=feed_dict)
                    except tf.errors.OutOfRangeError:
                        logger.info(f'train: epoch={i + 1}/{self.epoch_size}, loss={train_loss}, error={train_error}.')
                        feed_dict = {
                            self.graph.input_users: user_test,
                            self.graph.input_items: item_test,
                            self.graph.input_services: service_test,
                            self.graph.input_ratings: rating_test
                        }
                        test_loss, test_error = self.session.run([self.graph.loss, self.graph.error],
                                                                 feed_dict=feed_dict)
                        logger.info(f'epoch={i + 1}/{self.epoch_size}, loss={test_loss}, error={test_error}.')
                        break

                # check early stopping
                if early_stopping.does_stop(test_error, self.session):
                    break

    def predict(self, user_ids: List[Any], item_ids: List[Any], service_ids: List[Any]) -> np.ndarray:
        """
        If input data is invalid, return np.nan. For example, this returns [1.1, nan, 2.0] when at least one of `user_id[1]`, `item_id[1]` and `service_id[1]` is invalid. 
        :param user_ids: 
        :param item_ids: 
        :param service_ids: 
        :return: 
        """
        if self.graph is None:
            RuntimeError('Please call fit first.')

        user_indices = self._convert(user_ids, self.user2index)
        item_indices = self._convert(item_ids, self.item2index)
        service_indices = self._convert(service_ids, self.service2index)
        valid_inputs = np.where(
            np.logical_and(np.logical_and(user_indices != -1, item_indices != -1), service_indices != -1))[0]

        with self.session.as_default():
            feed_dict = {
                self.graph.input_users: user_indices[valid_inputs],
                self.graph.input_items: item_indices[valid_inputs],
                self.graph.input_services: service_indices[valid_inputs]
            }
            valid_predictions = self.session.run(self.graph.predictions, feed_dict=feed_dict)
        valid_predictions = np.clip(valid_predictions, self.min_value, self.max_value)
        predictions = np.array([np.nan] * len(user_ids))
        predictions[valid_inputs] = valid_predictions
        return predictions

    def _convert(self, ids: List[Any], id2index: Dict[Any, int]) -> np.ndarray:
        return np.array([id2index.get(i, -1) for i in ids])

    def _make_graph(self) -> MatrixFactorizationGraph:
        return MatrixFactorizationGraph(
            n_items=self.n_items,
            n_users=self.n_users,
            n_latent_factors=self.n_latent_factors,
            n_services=self.n_services,
            learning_rate=self.learning_rate,
            reg_item=self.reg_item,
            reg_user=self.reg_user,
            scope_name=self.scope_name)

    def save(self, file_path: str) -> None:
        redshells.model.utils.save_tf_session(self, self.session, file_path)

    @staticmethod
    def load(file_path: str) -> 'MatrixFactorization':
        session = tf.Session()
        model = redshells.model.utils.load_tf_session(MatrixFactorization, session, file_path,
                                                      MatrixFactorization._make_graph)  # type: MatrixFactorization
        return model
