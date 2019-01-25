import itertools
from logging import getLogger
from typing import Tuple

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf

import redshells
from redshells.model.early_stopping import EarlyStopping

logger = getLogger(__name__)


class FactorizationMachineGraph(object):
    def __init__(self,
                 input_size: int,
                 feature_kind_size: int,
                 embedding_size: int,
                 l2_weight: float,
                 learning_rate: float,
                 scope_name: str = '') -> None:

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            self.input_x_indices = tf.placeholder(dtype=np.int32, shape=[None, input_size], name='input_x_indices')
            self.input_x_values = tf.placeholder(dtype=np.float32, shape=[None, input_size], name='input_x_values')
            self.input_y = tf.placeholder(dtype=np.float, shape=[None], name='input_y')
            self.input_batch_size = tf.placeholder(dtype=np.float32, name='input_batch_size')

            regularizer = tf.contrib.layers.l2_regularizer(l2_weight)
            self.bias = tf.get_variable(name='bias', shape=[1], trainable=True)
            self.w_embedding = tf.keras.layers.Embedding(
                input_dim=feature_kind_size, output_dim=1, embeddings_regularizer=regularizer, name='w')
            self.w = tf.squeeze(self.w_embedding(self.input_x_indices), [2])
            self.v_embedding = tf.keras.layers.Embedding(
                input_dim=feature_kind_size, output_dim=embedding_size, embeddings_regularizer=regularizer, name='v')
            self.v = self.v_embedding(self.input_x_indices)

            self.xw = tf.multiply(self.input_x_values, self.w, name='xw')
            self.first_order = tf.reduce_sum(self.xw, axis=1, name='1st_order')
            self.xv2 = tf.pow(tf.einsum('ij, ijk -> ik', self.input_x_values, self.v), 2, name='xv2')
            self.x2v2 = tf.einsum('ij, ijk -> ik', tf.pow(self.input_x_values, 2), tf.pow(self.v, 2), name='x2v2')
            self.second_order = 0.5 * tf.reduce_sum(tf.subtract(self.xv2, self.x2v2), axis=1, name='2nd_order')
            self.y = tf.sigmoid(tf.add(self.bias, tf.add(self.first_order, self.second_order)), name='y')

        self.regularization = [
            # to reduce the dependency on the batch size and latent factor size.
            self.w_embedding.embeddings_regularizer(self.w) / tf.sqrt(feature_kind_size * self.input_batch_size),
            self.v_embedding.embeddings_regularizer(self.v) / tf.sqrt(feature_kind_size * self.input_batch_size)
        ]

        self.loss = tf.add_n([tf.losses.mean_squared_error(self.input_y, self.y)] + self.regularization, name='loss')
        self.error = tf.sqrt(tf.losses.mean_squared_error(self.input_y, self.y), name='error')

        optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate=learning_rate)
        self.op = optimizer.apply_gradients(optimizer.compute_gradients(self.loss))


class FactorizationMachine(sklearn.base.BaseEstimator):
    """
    FactorizationMachine is designed to predict CTR, so please use [0, 1] values for prediction targets.
    
    For details of the algorithm, see "Factorization Machines" which is available at https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
    """

    def __init__(self,
                 embedding_size: int,
                 l2_weight: float,
                 learning_rate: float,
                 batch_size: int,
                 epoch_size: int,
                 test_size: float,
                 scope_name: str,
                 save_directory_path: str = None,
                 input_size=None,
                 feature_kind_size=None,
                 real_columns=None,
                 categorical_columns=None,
                 category2index=None) -> None:
        self.session = tf.Session()
        self.embedding_size = embedding_size
        self.l2_weight = l2_weight
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.test_size = test_size
        self.scope_name = scope_name
        self.save_directory_path = save_directory_path
        self.input_size = input_size
        self.feature_kind_size = feature_kind_size
        self.real_columns = real_columns
        self.categorical_columns = categorical_columns
        self.category2index = category2index
        self.graph = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        if self.graph is None:
            logger.info('making graph...')
            self.input_size = x.shape[1]
            self.real_columns = list(x.select_dtypes(exclude='category').columns)
            self.categorical_columns = list(x.select_dtypes(include='category').columns)
            self.category2index = self._make_category2index(x)
            self.feature_kind_size = len(self.real_columns) + len(self.category2index.keys())
            self.graph = self._make_graph()
            logger.info('done making graph')

        x_values, x_indices = self._convert_x(x)
        x_values_train, x_values_test, x_indices_train, x_indices_test, y_train, y_test = sklearn.model_selection.train_test_split(
            x_values, x_indices, y.values, test_size=self.test_size)

        early_stopping = EarlyStopping(self.save_directory_path)

        with self.session.as_default():
            self.session.run(tf.global_variables_initializer())
            dataset = tf.data.Dataset.from_tensor_slices((x_values_train, x_indices_train, y_train))
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
                        x_values, x_indices, y_ = self.session.run(next_batch)
                        feed_dict = {
                            self.graph.input_x_values: x_values,
                            self.graph.input_x_indices: x_indices,
                            self.graph.input_y: y_,
                            self.graph.input_batch_size: len(y_)
                        }
                        _, train_loss, train_error = self.session.run(
                            [self.graph.op, self.graph.loss, self.graph.error], feed_dict=feed_dict)
                    except tf.errors.OutOfRangeError:
                        logger.info(f'train: epoch={i + 1}/{self.epoch_size}, loss={train_loss}, error={train_error}.')
                        feed_dict = {
                            self.graph.input_x_values: x_values_test,
                            self.graph.input_x_indices: x_indices_test,
                            self.graph.input_y: y_test,
                            self.graph.input_batch_size: len(y_test)
                        }
                        test_loss, test_error, y = self.session.run([self.graph.loss, self.graph.error, self.graph.y],
                                                                    feed_dict=feed_dict)
                        auc = redshells.model.utils.calculate_auc(y_test, y)
                        logger.info(
                            f'epoch={i + 1}/{self.epoch_size}, loss={test_loss}, error={test_error}, auc={auc}.')
                        break

                # TODO
                if early_stopping.does_stop(test_error, self.session):
                    break

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.graph is None:
            RuntimeError('Please call fit first.')

        x_values, x_indices = self._convert_x(x)

        with self.session.as_default():
            feed_dict = {self.graph.input_x_values: x_values, self.graph.input_x_indices: x_indices}
            y = self.session.run(self.graph.y, feed_dict=feed_dict)

        return y

    def _make_category2index(self, data: pd.DataFrame):
        categories = list(
            itertools.chain.from_iterable(
                [[f'{c}_{str(x)}' for x in set(data[c].tolist())] for c in self.categorical_columns]))
        categories = categories + self.categorical_columns
        return dict(zip(categories, range(len(self.real_columns), len(self.real_columns) + len(categories))))

    def _convert_x(self, original: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        indices = original.copy()

        for c in self.categorical_columns:
            keys = [f'{c}_{str(x)}' for x in indices[c]]
            indices[c] = [self.category2index.get(x, self.category2index[c]) for x in keys]
        for i, r in enumerate(self.real_columns):
            indices[r] = i

        values = original.copy()
        for c in self.categorical_columns:
            values[c] = 1
        values = values.fillna(0)

        values = values.astype(np.float32).values
        indices = indices.astype(np.int32).values
        return values, indices

    def _make_graph(self) -> FactorizationMachineGraph:
        return FactorizationMachineGraph(
            embedding_size=self.embedding_size,
            input_size=self.input_size,
            feature_kind_size=self.feature_kind_size,
            l2_weight=self.l2_weight,
            learning_rate=self.learning_rate,
            scope_name=self.scope_name)

    def save(self, file_path: str) -> None:
        redshells.model.utils.save_tf_session(self, self.session, file_path)

    @staticmethod
    def load(file_path: str) -> 'FactorizationMachine':
        session = tf.Session()
        model = redshells.model.utils.load_tf_session(FactorizationMachine, session, file_path,
                                                      FactorizationMachine._make_graph)  # type: FactorizationMachine
        return model
