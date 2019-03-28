import itertools
from logging import getLogger
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf

import redshells
from redshells.model.early_stopping import EarlyStopping
from redshells.model.gcmc_dataset import GcmcDataset

logger = getLogger(__name__)


def _make_weight_variable(shape, name: str = None):
    init_range = np.sqrt(10.0 / sum(shape))
    initial = tf.random_uniform(shape=shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def _convert_sparse_matrix_to_sparse_tensor(x):
    coo = x.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensorValue(indices, coo.data, coo.shape)


class GraphConvolutionalMatrixCompletionGraph(object):
    def __init__(self,
                 n_rating: int,
                 n_user: int,
                 n_item: int,
                 rating: np.ndarray,
                 encoder_hidden_size: int,
                 encoder_size: int,
                 normalization_type: str,
                 user_side_information: Optional[List[np.ndarray]] = None,
                 item_side_information: Optional[List[np.ndarray]] = None,
                 scope_name: str = 'GraphConvolutionalMatrixCompletionGraph',
                 weight_sharing: bool = True,
                 ignore_item_embedding: bool = False) -> None:
        logger.info(f'n_rating={n_rating}; n_user={n_user}; n_item={n_item}')
        assert isinstance(user_side_information, list) or user_side_information is None
        assert isinstance(item_side_information, list) or item_side_information is None

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            # placeholder
            self.input_learning_rate = tf.placeholder(dtype=np.float32, name='learning_rate')
            self.input_dropout = tf.placeholder(dtype=np.float32, name='learning_rate')
            self.input_label = tf.placeholder(dtype=np.int32, name='label')
            self.input_user = tf.placeholder(dtype=np.int32, name='user')
            self.input_item = tf.placeholder(dtype=np.int32, name='item')
            self.input_user_information = tf.placeholder(dtype=np.int32, name='user_information')
            self.input_item_information = tf.placeholder(dtype=np.int32, name='item_information')
            self.input_edge_size = [tf.placeholder(dtype=np.int32, name=f'edge_size_{r}') for r in range(n_rating)]
            self.input_rating = tf.placeholder(dtype=np.int32, name='rating')
            # shape=(n_user, n_item)
            self.input_adjacency_matrix = [
                tf.sparse.placeholder(dtype=np.float32, name=f'adjacency_matrix_{r}') for r in range(n_rating)
            ]
            # adjustment
            self.user_adjustment = [
                tf.reshape(tf.div_no_nan(1., tf.sparse.reduce_sum(m, axis=1)), shape=(-1, 1))
                for m in self.input_adjacency_matrix
            ]
            self.item_adjustment = [
                tf.div_no_nan(1., tf.sparse.reduce_sum(m, axis=0)) for m in self.input_adjacency_matrix
            ]
            self.rating = tf.constant(rating.reshape((-1, 1)), dtype=np.float32)

            # side information
            self.user_side_information = [tf.constant(information) for information in user_side_information]
            self.item_side_information = [tf.constant(information) for information in item_side_information]

            # adjusted adjacency matrix
            if normalization_type == 'symmetric':
                self.adjusted_adjacency_matrix = [
                    tf.sqrt(item) * m * tf.sqrt(user)
                    for item, m, user in zip(self.item_adjustment, self.input_adjacency_matrix, self.user_adjustment)
                ]
                self.adjusted_adjacency_matrix_transpose = [
                    tf.sparse.transpose(m) for m in self.adjusted_adjacency_matrix
                ]
            elif normalization_type == 'left':
                self.adjusted_adjacency_matrix = [
                    m * user for m, user in zip(self.input_adjacency_matrix, self.user_adjustment)
                ]
                self.adjusted_adjacency_matrix_transpose = [
                    tf.sparse.transpose(item * m) for item, m in zip(self.item_adjustment, self.input_adjacency_matrix)
                ]
            elif normalization_type == 'right':
                self.adjusted_adjacency_matrix = [
                    item * m for item, m in zip(self.item_adjustment, self.input_adjacency_matrix)
                ]
                self.adjusted_adjacency_matrix_transpose = [
                    tf.sparse.transpose(m * user) for m, user in zip(self.input_adjacency_matrix, self.user_adjustment)
                ]
            else:
                raise ValueError(
                    f'normalization_type must be "left", "right" or "symmetric", but {normalization_type} is passed.')

            # C X
            # (n_user, item_feature_size)
            self.item_cx = self.adjusted_adjacency_matrix
            # (n_item, user_feature_size)
            self.user_cx = self.adjusted_adjacency_matrix_transpose

            # encoder
            self.common_encoder_layer = self._simple_layer(encoder_size)

            self.item_encoder_hidden = self._encoder(
                feature_size=n_user,
                encoder_hidden_size=encoder_hidden_size,
                n_rating=n_rating,
                cx=self.user_cx,
                dropout=self.input_dropout,
                weight_sharing=weight_sharing,
                edge_size=self.input_edge_size,
                prefix='item')

            self.user_encoder_hidden = self._encoder(
                feature_size=n_item,
                encoder_hidden_size=encoder_hidden_size,
                n_rating=n_rating,
                cx=self.item_cx,
                dropout=self.input_dropout,
                weight_sharing=weight_sharing,
                edge_size=self.input_edge_size,
                prefix='user')

            self.item_encoder = self.common_encoder_layer(self.item_encoder_hidden)
            self.user_encoder = self.common_encoder_layer(self.user_encoder_hidden)
            self.user_encoder = tf.gather(self.user_encoder, self.input_user)
            self.item_encoder = tf.gather(self.item_encoder, self.input_item)

            if len(self.user_side_information) != 0:
                layer = self._cross_side_information_layer(
                    hidden_size=encoder_hidden_size, size=encoder_size, input_data=self.user_side_information)
                layer = tf.gather(layer, self.input_user_information)
                self.user_encoder = self.user_encoder + layer

            if len(self.item_side_information) != 0:
                layer = self._cross_side_information_layer(
                    hidden_size=encoder_hidden_size, size=encoder_size, input_data=self.item_side_information)
                layer = tf.gather(layer, self.input_item_information)
                if ignore_item_embedding:
                    self.item_encoder = layer
                else:
                    self.item_encoder = self.item_encoder + layer

            # decoder
            self.output = self._decoder(
                encoder_size=encoder_size,
                n_rating=n_rating,
                user_encoder=self.user_encoder,
                item_encoder=self.item_encoder)

            # output
            self.probability = tf.nn.softmax(self.output)
            self.expectation = tf.matmul(self.probability, tf.reshape(self.rating, shape=(-1, 1)))
            self.rmse = tf.sqrt(
                tf.reduce_mean(
                    tf.math.square(self.expectation - tf.reshape(tf.to_float(self.input_rating), shape=(-1, 1)))))

            # loss
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.input_label))

            # optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate=self.input_learning_rate)
            self.op = optimizer.apply_gradients(optimizer.compute_gradients(self.loss))

    @staticmethod
    def _side_information_layer(hidden_size: int, input_data):
        x = tf.keras.layers.Dense(
            hidden_size, use_bias=False, activation=None, kernel_initializer='glorot_normal')(input_data)
        return x

    @classmethod
    def _cross_side_information_layer(cls, hidden_size: int, size: int, input_data: List[np.ndarray]):
        layers = [cls._side_information_layer(hidden_size, data) + 1.0 for data in input_data]
        x = tf.reduce_prod(layers, axis=0)
        y = tf.keras.layers.Dense(size, use_bias=False, activation=None, kernel_initializer='glorot_normal')(x)
        return y

    @staticmethod
    def _simple_layer(output_size: int, input_size: Optional[int] = None):
        layer = tf.keras.layers.Dense(output_size, use_bias=False, activation=None, kernel_initializer='glorot_normal')
        if input_size is not None:
            layer.build(input_shape=(None, input_size))
        return layer

    @classmethod
    def _decoder(cls, encoder_size, n_rating, user_encoder, item_encoder):
        user_encoder = tf.nn.l2_normalize(user_encoder, axis=1)
        item_encoder = tf.nn.l2_normalize(item_encoder, axis=1)
        weights = [cls._simple_layer(encoder_size, input_size=encoder_size).weights[0] for _ in range(n_rating)]
        output = [tf.reduce_sum(tf.multiply(tf.matmul(user_encoder, w), item_encoder), axis=1) for w in weights]
        output = tf.stack(output, axis=1)
        return output

    @classmethod
    def _encoder(cls, feature_size, encoder_hidden_size, n_rating, cx, dropout, weight_sharing, prefix, edge_size):
        cx = [cls._dropout_sparse(x, 1. - dropout, num_nonzero_elements=s) for x, s in zip(cx, edge_size)]
        weights = [cls._simple_layer(encoder_hidden_size, input_size=feature_size).weights[0] for _ in range(n_rating)]
        if weight_sharing:
            for r in range(n_rating - 1):
                weights[r + 1].assign_add(weights[r])

        encoder_hidden = [
            tf.sparse_tensor_dense_matmul(cx[r], weights[r], name=f'{prefix}_encoder_hidden_{r}')
            for r in range(n_rating)
        ]
        result = tf.nn.relu(tf.reduce_sum(encoder_hidden, axis=0))
        return result

    @staticmethod
    def _node_dropout(x, keep_prob, size):
        random_tensor = keep_prob + tf.random_uniform([size])
        dropout_mask = tf.floor(random_tensor)
        return dropout_mask * x / keep_prob

    @staticmethod
    def _dropout_sparse(x, keep_prob, num_nonzero_elements):
        random_tensor = keep_prob + tf.random_uniform([num_nonzero_elements])
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        return tf.sparse_retain(x, dropout_mask) / keep_prob

    @staticmethod
    def _dropout(x, keep_prob):
        return tf.nn.dropout(x, rate=1 - keep_prob)

    @staticmethod
    def _to_constant(x):
        return tf.constant(x) if x is not None else None


class GraphConvolutionalMatrixCompletion(object):
    def __init__(self,
                 user_ids: np.ndarray,
                 item_ids: np.ndarray,
                 ratings: np.ndarray,
                 encoder_hidden_size: int,
                 encoder_size: int,
                 scope_name: str,
                 test_size: float,
                 batch_size: int,
                 epoch_size: int,
                 dropout_rate: float,
                 learning_rate: float,
                 normalization_type: str,
                 weight_sharing: bool = True,
                 ignore_item_embedding: bool = False,
                 save_directory_path: str = None,
                 user_features: Optional[List[Dict[Any, np.ndarray]]] = None,
                 item_features: Optional[List[Dict[Any, np.ndarray]]] = None) -> None:
        self.session = tf.Session()
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = ratings
        self.item_features = item_features
        self.user_features = user_features
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_size = encoder_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.scope_name = scope_name
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.normalization_type = normalization_type
        self.weight_sharing = weight_sharing
        self.ignore_item_embedding = ignore_item_embedding
        self.save_directory_path = save_directory_path
        self.dataset = GcmcDataset(
            self.user_ids,
            self.item_ids,
            self.ratings,
            self.test_size,
            user_information=self.user_features,
            item_information=self.item_features,
            min_user_click_count=5)
        self.graph = None

    def fit(self, try_count=1, decay_speed=10.) -> List[str]:
        if self.graph is None:
            logger.info('making graph...')
            self.graph = self._make_graph()
            logger.info('done making graph')

        early_stopping = EarlyStopping(
            try_count=try_count,
            decay_speed=decay_speed,
            save_directory=self.save_directory_path,
            learning_rate=self.learning_rate,
            threshold=1e-4)

        test_data = self.dataset.test_data()
        report = []
        with self.session.as_default():
            self.session.run(tf.global_variables_initializer())
            dataset = tf.data.Dataset.from_tensor_slices(self.dataset.train_data())
            dataset = dataset.shuffle(buffer_size=self.batch_size)
            batch = dataset.batch(self.batch_size)
            iterator = batch.make_initializable_iterator()
            next_batch = iterator.get_next()
            rating_adjacency_matrix = self.dataset.train_rating_adjacency_matrix()

            logger.info('start to optimize...')
            for i in range(self.epoch_size):
                self.session.run(iterator.initializer)
                while True:
                    try:
                        train_data = self.session.run(next_batch)
                        _rating_adjacency_matrix = [
                            self._eliminate(matrix, train_data['user'], train_data['item'])
                            for matrix in rating_adjacency_matrix
                        ]
                        feed_dict = {
                            self.graph.input_learning_rate: early_stopping.learning_rate,
                            self.graph.input_dropout: self.dropout_rate,
                            self.graph.input_user: train_data['user'],
                            self.graph.input_item: train_data['item'],
                            self.graph.input_label: train_data['label'],
                            self.graph.input_rating: train_data['rating'],
                            self.graph.input_user_information: train_data['user_information'],
                            self.graph.input_item_information: train_data['item_information'],
                        }
                        feed_dict.update({
                            g: _convert_sparse_matrix_to_sparse_tensor(m)
                            for g, m in zip(self.graph.input_adjacency_matrix, _rating_adjacency_matrix)
                        })
                        feed_dict.update({
                            g: m.count_nonzero()
                            for g, m in zip(self.graph.input_edge_size, _rating_adjacency_matrix)
                        })
                        _, train_loss, train_rmse = self.session.run([self.graph.op, self.graph.loss, self.graph.rmse],
                                                                     feed_dict=feed_dict)
                        report.append(f'train: epoch={i + 1}/{self.epoch_size}, loss={train_loss}, rmse={train_rmse}.')
                    except tf.errors.OutOfRangeError:
                        logger.info(report[-1])
                        feed_dict = {
                            self.graph.input_dropout: 0.0,
                            self.graph.input_user: test_data['user'],
                            self.graph.input_item: test_data['item'],
                            self.graph.input_label: test_data['label'],
                            self.graph.input_rating: test_data['rating'],
                            self.graph.input_user_information: test_data['user_information'],
                            self.graph.input_item_information: test_data['item_information'],
                        }
                        feed_dict.update({
                            g: _convert_sparse_matrix_to_sparse_tensor(m)
                            for g, m in zip(self.graph.input_adjacency_matrix, rating_adjacency_matrix)
                        })
                        feed_dict.update(
                            {g: m.count_nonzero()
                             for g, m in zip(self.graph.input_edge_size, rating_adjacency_matrix)})
                        test_loss, test_rmse = self.session.run([self.graph.loss, self.graph.rmse], feed_dict=feed_dict)
                        report.append(f'test: epoch={i + 1}/{self.epoch_size}, loss={test_loss}, rmse={test_rmse}.')
                        logger.info(report[-1])
                        break

                if early_stopping.does_stop(test_rmse, self.session):
                    break
        return report

    def predict(self, user_ids: List, item_ids: List, with_user_embedding: bool = True) -> np.ndarray:
        if self.graph is None:
            RuntimeError('Please call fit first.')

        rating_adjacency_matrix = self.dataset.train_rating_adjacency_matrix()
        user_indices, item_indices = self.dataset.to_indices(user_ids, item_ids)
        if not with_user_embedding:
            user_indices = np.array([0] * len(user_indices))  # TODO use default user index.

        user_information_indices, item_information_indices = self.dataset.to_information_indices(user_ids, item_ids)
        feed_dict = {
            self.graph.input_dropout: 0.0,
            self.graph.input_user: user_indices,
            self.graph.input_item: item_indices,
            self.graph.input_user_information: user_information_indices,
            self.graph.input_item_information: item_information_indices,
        }
        feed_dict.update({
            g: _convert_sparse_matrix_to_sparse_tensor(m)
            for g, m in zip(self.graph.input_adjacency_matrix, rating_adjacency_matrix)
        })
        feed_dict.update({g: m.count_nonzero() for g, m in zip(self.graph.input_edge_size, rating_adjacency_matrix)})
        with self.session.as_default():
            predictions = self.session.run(self.graph.expectation, feed_dict=feed_dict)
        predictions = predictions.flatten()
        predictions = np.clip(predictions, self.dataset.rating()[0], self.dataset.rating()[-1])
        return predictions

    def predict_item_scores(self, item_ids: List, with_user_embedding: bool = True) -> pd.DataFrame:
        user_ids = list(self.dataset.user_id_map.id2index.keys())
        _test_users, _test_items = zip(*list(itertools.product(user_ids, item_ids)))
        predicts = self.predict(user_ids=_test_users, item_ids=_test_items, with_user_embedding=with_user_embedding)
        results = pd.DataFrame(dict(user=_test_users, item=_test_items, score=predicts))
        results.sort_values('score', ascending=False, inplace=True)
        return results

    def _make_graph(self) -> GraphConvolutionalMatrixCompletionGraph:
        return GraphConvolutionalMatrixCompletionGraph(
            n_rating=len(self.dataset.rating_id_map.id2index),
            n_user=len(self.dataset.user_id_map.id2index) + 1,  # TODO
            n_item=len(self.dataset.item_id_map.id2index) + 1,  # TODO
            rating=self.dataset.rating(),
            normalization_type=self.normalization_type,
            encoder_hidden_size=self.encoder_hidden_size,
            encoder_size=self.encoder_size,
            weight_sharing=self.weight_sharing,
            scope_name=self.scope_name,
            user_side_information=self.dataset.user_information,
            item_side_information=self.dataset.item_information,
            ignore_item_embedding=self.ignore_item_embedding)

    @staticmethod
    def _eliminate(matrix: sp.csr_matrix, user_indices, item_indices):
        matrix = matrix.copy()
        matrix[user_indices, item_indices] = 0
        matrix.eliminate_zeros()
        return matrix

    def save(self, file_path: str) -> None:
        redshells.model.utils.save_tf_session(self, self.session, file_path)

    @staticmethod
    def load(file_path: str) -> 'GraphConvolutionalMatrixCompletion':
        session = tf.Session()
        model = redshells.model.utils.load_tf_session(
            GraphConvolutionalMatrixCompletion, session, file_path,
            GraphConvolutionalMatrixCompletion._make_graph)  # type: GraphConvolutionalMatrixCompletion
        return model


def _make_sparse_matrix(n, m, n_values):
    x = np.zeros(shape=(n, m), dtype=np.float32)
    x[np.random.choice(range(n), n_values), np.random.choice(range(m), n_values)] = 1.0
    return sp.csr_matrix(x)


def main():
    n_users = 101
    n_items = 233
    n_data = 3007
    n_features = 21
    adjacency_matrix = _make_sparse_matrix(n_users, n_items, n_data) + 2 * _make_sparse_matrix(n_users, n_items, n_data)
    user_ids = adjacency_matrix.tocoo().row
    item_ids = adjacency_matrix.tocoo().col
    ratings = adjacency_matrix.tocoo().data
    item_features = [dict(zip(range(n_items), np.random.uniform(size=(n_items, n_features))))]
    encoder_hidden_size = 100
    encoder_size = 100
    scope_name = 'GraphConvolutionalMatrixCompletionGraph'
    model = GraphConvolutionalMatrixCompletion(
        user_ids=user_ids,
        item_ids=item_ids,
        ratings=ratings,
        encoder_hidden_size=encoder_hidden_size,
        encoder_size=encoder_size,
        scope_name=scope_name,
        test_size=0.1,
        batch_size=1024,
        epoch_size=10,
        learning_rate=0.01,
        dropout_rate=0.7,
        normalization_type='symmetric',
        item_features=item_features)
    for report in model.fit():
        print(report)


if __name__ == '__main__':
    main()
