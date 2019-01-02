from logging import getLogger
from typing import List, Any, Dict

import numpy as np
import sklearn
import tensorflow as tf

import redshells
from redshells.model.early_stopping import EarlyStopping

logger = getLogger(__name__)


class MatrixFactorizationGraph(object):
    def __init__(self, n_items: int, n_users: int, n_latent_factors: int, n_services: int, reg_item: float,
                 reg_user: float, scope_name: str, use_l2_upper_regularization: bool, average: float,
                 standard_deviation: float) -> None:
        # placeholder
        self.input_items = tf.placeholder(dtype=np.int32, shape=[None], name='input_items')
        self.input_users = tf.placeholder(dtype=np.int32, shape=[None], name='input_users')
        self.input_services = tf.placeholder(dtype=np.int32, shape=[None], name='input_services')
        self.input_ratings = tf.placeholder(dtype=np.float32, shape=[None], name='input_ratings')
        self.input_batch_size = tf.placeholder(dtype=np.float32, name='input_batch_size')
        self.input_learning_rate = tf.placeholder(dtype=np.float32, name='input_learning_rate')

        scale = standard_deviation / np.sqrt(n_latent_factors)
        initializer = tf.random_uniform_initializer(0, scale)

        with tf.variable_scope(f'{scope_name}_bias', reuse=tf.AUTO_REUSE):
            self.item_bias_embedding = tf.keras.layers.Embedding(
                input_dim=n_items,
                output_dim=1,
                embeddings_initializer=tf.constant_initializer(average / 2.0),
                name='item_bias')
            self.item_biases = self.item_bias_embedding(self.input_items)

            self.user_bias_embedding = tf.keras.layers.Embedding(
                input_dim=n_users * n_services,
                output_dim=1,
                embeddings_initializer=tf.constant_initializer(average / 2.0),
                name='user_bias')
            self.user_biases = self.user_bias_embedding(self.input_users + self.input_services * n_users)

        with tf.variable_scope(f'{scope_name}_latent_factor', reuse=tf.AUTO_REUSE):
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

        adjustment = tf.sqrt(n_latent_factors * self.input_batch_size)
        self.elements = [
            # to reduce the dependency on the batch size and latent factor size.
            self.item_embedding.embeddings_regularizer(self.item_factors) / adjustment,
            self.user_embedding.embeddings_regularizer(self.user_factors) / adjustment
        ]

        if use_l2_upper_regularization:
            item_norm = tf.norm(self.item_factors, axis=1)
            self.item_normalization_penalty = tf.reduce_mean(
                tf.maximum(item_norm - tf.ones_like(item_norm), tf.zeros_like(item_norm)))
            self.elements.append(self.item_normalization_penalty)

        squared_error = tf.losses.mean_squared_error(self.input_ratings, self.predictions)
        self.elements.append(squared_error)
        self.loss = tf.add_n(self.elements, name='loss')
        self.error = tf.sqrt(squared_error, name='error')

        bias_squared_error = tf.losses.mean_squared_error(self.input_ratings, self.bias_values)
        self.bias_loss = bias_squared_error
        self.bias_error = tf.sqrt(bias_squared_error, name='bias_error')

        bias_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=f'{scope_name}_bias')
        latent_factor_var_list = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=f'{scope_name}_latent_factor')

        optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate=self.input_learning_rate)
        self.bias_op = optimizer.apply_gradients(optimizer.compute_gradients(self.bias_loss, var_list=bias_var_list))
        self.op = optimizer.apply_gradients(optimizer.compute_gradients(self.loss, var_list=latent_factor_var_list))


class MatrixFactorization(object):
    def __init__(self,
                 n_latent_factors: int,
                 learning_rate: float,
                 reg_item: float,
                 reg_user: float,
                 use_l2_upper_regularization: bool,
                 batch_size: int,
                 epoch_size: int,
                 bias_epoch_size: int,
                 test_size: float,
                 scope_name: str,
                 try_count: int = 3,
                 decay_speed: float = 10.0,
                 save_directory_path: str = None,
                 n_items=None,
                 n_users=None,
                 n_services=None,
                 max_value=None,
                 min_value=None,
                 average=None,
                 standard_deviation=None,
                 user2index=None,
                 item2index=None,
                 service2index=None) -> None:
        self.n_latent_factors = n_latent_factors
        self.learning_rate = learning_rate
        self.reg_item = reg_item
        self.reg_user = reg_user
        self.use_l2_upper_regularization = use_l2_upper_regularization
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.bias_epoch_size = bias_epoch_size
        self.test_size = test_size
        self.scope_name = scope_name
        self.try_count = try_count
        self.decay_speed = decay_speed
        self.save_directory_path = save_directory_path
        self.n_items = n_items
        self.n_users = n_users
        self.n_services = n_services
        self.max_value = max_value
        self.min_value = min_value
        self.average = average
        self.standard_deviation = standard_deviation
        self.user2index = user2index
        self.item2index = item2index
        self.service2index = service2index
        self.session = tf.Session()
        self.graph = None

    def fit(self, user_ids: List[Any], item_ids: List[Any], service_ids: List[Any], ratings: List[float]) -> None:
        logger.info(f'data size={len(user_ids)}.')
        if self.graph is None:
            logger.info('making graph...')
            self.n_users = len(set(user_ids))
            self.n_items = len(set(item_ids))
            self.n_services = len(set(service_ids))
            self.min_value = np.min(ratings)
            self.max_value = np.max(ratings)
            self.average = np.average(ratings)
            self.standard_deviation = np.std(ratings)
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

        test_feed_dict = {
            self.graph.input_users: user_test,
            self.graph.input_items: item_test,
            self.graph.input_services: service_test,
            self.graph.input_ratings: rating_test,
            self.graph.input_batch_size: len(user_test)
        }

        with self.session.as_default():
            logger.info('initializing variables...')
            self.session.run(tf.global_variables_initializer())
            dataset = tf.data.Dataset.from_tensor_slices((user_train, item_train, service_train, rating_train))
            dataset = dataset.batch(self.batch_size)
            iterator = dataset.make_initializable_iterator()
            next_batch = iterator.get_next()

            logger.info('start to optimize bias...')
            # This training makes the model fit the input data bias.
            self._train(
                epoch_size=self.bias_epoch_size,
                op=self.graph.bias_op,
                loss=self.graph.bias_loss,
                error=self.graph.bias_error,
                iterator=iterator,
                next_batch=next_batch,
                test_feed_dict=test_feed_dict,
                early_stopping=EarlyStopping(save_directory=self.save_directory_path, learning_rate=self.learning_rate))

            logger.info('start to optimize latent factor...')
            self._train(
                epoch_size=self.epoch_size,
                op=self.graph.op,
                loss=self.graph.loss,
                error=self.graph.error,
                iterator=iterator,
                next_batch=next_batch,
                test_feed_dict=test_feed_dict,
                early_stopping=EarlyStopping(
                    save_directory=self.save_directory_path,
                    try_count=self.try_count,
                    learning_rate=self.learning_rate,
                    decay_speed=self.decay_speed))

    def _train(self, epoch_size, op, loss, error, iterator, next_batch, test_feed_dict, early_stopping):
        test_loss, test_error = self.session.run([loss, error], feed_dict=test_feed_dict)
        logger.info(f'test: epoch=0/{epoch_size}, loss={test_loss}, error={test_error}.')

        for i in range(epoch_size):
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
                        self.graph.input_ratings: rating_,
                        self.graph.input_batch_size: len(user_),
                        self.graph.input_learning_rate: early_stopping.learning_rate
                    }
                    _, train_loss, train_error = self.session.run([op, loss, error], feed_dict=feed_dict)
                except tf.errors.OutOfRangeError:
                    logger.info(f'train: epoch={i + 1}/{epoch_size}, loss={train_loss}, error={train_error}.')
                    test_loss, test_error = self.session.run([loss, error], feed_dict=test_feed_dict)
                    logger.info(f'test: epoch={i + 1}/{epoch_size}, loss={test_loss}, error={test_error}.')
                    break

            # check early stopping
            if early_stopping.does_stop(test_error, self.session):
                break

    def predict(self, user_ids: List[Any], item_ids: List[Any], service_ids: List[Any], default=np.nan) -> np.ndarray:
        """If input data is invalid, return `default`. For example, this returns [1.1, `default`, 2.0]
         when at least one of `user_id[1]`, `item_id[1]` and `service_id[1]` is invalid. 
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
        predictions = np.array([default] * len(user_ids))
        predictions[valid_inputs] = valid_predictions
        return predictions

    def get_item_factors(self, item_ids: List[Any], default=None, normalize: bool = False) -> np.ndarray:
        """Return latent factors for given items."""

        if self.graph is None:
            RuntimeError('Please call fit first.')

        item_indices = self._convert(item_ids, self.item2index)
        valid_inputs = np.where(item_indices != -1)[0]

        with self.session.as_default():
            feed_dict = {self.graph.input_items: item_indices[valid_inputs]}
            valid_item_factors = self.session.run(self.graph.item_factors, feed_dict=feed_dict)

        if normalize:
            valid_item_factors = sklearn.preprocessing.normalize(valid_item_factors, axis=1, norm='l2')

        default = default or np.zeros(valid_item_factors.shape[1])
        predictions = np.array([default] * len(item_ids))
        predictions[valid_inputs, :] = valid_item_factors
        return predictions

    def get_valid_user_ids(self, ids: List):
        return [i for i in ids if i in self.user2index]

    def get_valid_item_ids(self, ids: List):
        return [i for i in ids if i in self.item2index]

    def _convert(self, ids: List[Any], id2index: Dict[Any, int]) -> np.ndarray:
        return np.array([id2index.get(i, -1) for i in ids])

    def _make_graph(self) -> MatrixFactorizationGraph:
        return MatrixFactorizationGraph(
            n_items=self.n_items,
            n_users=self.n_users,
            n_latent_factors=self.n_latent_factors,
            n_services=self.n_services,
            reg_item=self.reg_item,
            reg_user=self.reg_user,
            scope_name=self.scope_name,
            use_l2_upper_regularization=self.use_l2_upper_regularization,
            average=self.average,
            standard_deviation=self.standard_deviation)

    def save(self, file_path: str) -> None:
        redshells.model.utils.save_tf_session(self, self.session, file_path)

    @staticmethod
    def load(file_path: str) -> 'MatrixFactorization':
        session = tf.Session()
        model = redshells.model.utils.load_tf_session(MatrixFactorization, session, file_path,
                                                      MatrixFactorization._make_graph)  # type: MatrixFactorization
        return model
