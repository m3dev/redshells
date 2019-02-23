from logging import getLogger
from typing import List, Optional, Dict, Tuple

import numpy as np
import scipy.sparse as sp
import sklearn
import tensorflow as tf

import redshells
from redshells.model.early_stopping import EarlyStopping

logger = getLogger(__name__)


def _dot(x, y, name: str, x_is_sparse: bool):
    if x_is_sparse:
        return tf.sparse_tensor_dense_matmul(x, y, name=name)
    return tf.matmul(x, y, name=name)


def _make_weight_variable(shape, name: str):
    init_range = np.sqrt(6.0 / sum(shape))
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
                 user_feature: Optional[np.ndarray] = None,
                 item_feature: Optional[np.ndarray] = None,
                 scope_name: str = 'GraphConvolutionalMatrixCompletionGraph',
                 weight_sharing: bool = True) -> None:
        logger.info(f'n_rating={n_rating}; n_user={n_user}; n_item={n_item}')
        user_feature_size = user_feature.shape[1] if user_feature else n_user
        item_feature_size = item_feature.shape[1] if item_feature else n_item

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            # placeholder
            self.input_learning_rate = tf.placeholder(dtype=np.float32, name='learning_rate')
            self.input_dropout = tf.placeholder(dtype=np.float32, name='learning_rate')
            self.input_label = tf.placeholder(dtype=np.int32, name='label')
            self.input_user = tf.placeholder(dtype=np.int32, name='user')
            self.input_item = tf.placeholder(dtype=np.int32, name='item')
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
            # feature
            self.user_feature = tf.constant(user_feature) if user_feature else None
            self.item_feature = tf.constant(item_feature) if item_feature else None

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
            if self.item_feature:
                self.item_cx = [
                    tf.sparse_tensor_dense_matmul(m, self.item_feature) for m in self.adjusted_adjacency_matrix
                ]
            else:
                self.item_cx = self.adjusted_adjacency_matrix
            # (n_item, user_feature_size)
            if self.user_feature:
                self.user_cx = [
                    tf.sparse_matmul(m, self.user_feature) for m in self.adjusted_adjacency_matrix_transpose
                ]
            else:
                self.user_cx = self.adjusted_adjacency_matrix_transpose

            # encoder
            self.common_encoder_weight = _make_weight_variable(
                shape=(encoder_hidden_size, encoder_size), name=f'common_encoder_weight')

            self.item_encoder_hidden = self._encoder(
                feature_size=user_feature_size,
                encoder_hidden_size=encoder_hidden_size,
                n_rating=n_rating,
                cx=self.user_cx,
                dropout=self.input_dropout,
                weight_sharing=weight_sharing,
                is_sparse=self.user_feature is None,
                size=self.input_edge_size,
                prefix='item')

            self.user_encoder_hidden = self._encoder(
                feature_size=item_feature_size,
                encoder_hidden_size=encoder_hidden_size,
                n_rating=n_rating,
                cx=self.item_cx,
                dropout=self.input_dropout,
                weight_sharing=weight_sharing,
                is_sparse=self.item_feature is None,
                size=self.input_edge_size,
                prefix='user')

            self.item_encoder = tf.matmul(self.item_encoder_hidden, self.common_encoder_weight)
            self.user_encoder = tf.matmul(self.user_encoder_hidden, self.common_encoder_weight)

            # decoder
            self.decoder_weight = [
                _make_weight_variable(shape=(encoder_size, encoder_size), name=f'decoder_weight_{r}')
                for r in range(n_rating)
            ]

            user_encoder = tf.gather(self.user_encoder, self.input_user)
            item_encoder = tf.gather(self.item_encoder, self.input_item)

            output = [
                tf.reduce_sum(tf.multiply(tf.matmul(user_encoder, w), item_encoder), axis=1)
                for w in self.decoder_weight
            ]
            self.output = tf.stack(output, axis=1)
            self.probability = tf.nn.softmax(self.output)
            self.expectation = tf.matmul(self.probability, tf.reshape(self.rating, shape=(-1, 1)))
            self.rmse = tf.sqrt(
                tf.losses.mean_squared_error(self.expectation, tf.reshape(self.input_rating, shape=(-1, 1))))

            # loss
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.input_label))

            # optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate=self.input_learning_rate)
            self.op = optimizer.apply_gradients(optimizer.compute_gradients(self.loss))

    @classmethod
    def _encoder(cls, feature_size, encoder_hidden_size, n_rating, cx, is_sparse, dropout, weight_sharing, prefix,
                 size):
        cx = [cls._dropout_sparse(x, 1. - dropout, num_nonzero_elements=s) for x, s in zip(cx, size)]
        encoder_weight = [
            _make_weight_variable(shape=(feature_size, encoder_hidden_size), name=f'{prefix}_encoder_weight_{r}')
            for r in range(n_rating)
        ]
        if weight_sharing:
            for r in range(n_rating - 1):
                encoder_weight[r + 1].assign_add(encoder_weight[r])

        encoder_hidden = [
            _dot(cx[r], encoder_weight[r], name=f'{prefix}_encoder_hidden_{r}', x_is_sparse=is_sparse)
            for r in range(n_rating)
        ]
        return tf.nn.relu(tf.reduce_sum(encoder_hidden, axis=0))

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


class _Dataset(object):
    def __init__(self, user_ids: np.ndarray, item_ids: np.ndarray, ratings: np.ndarray, test_size: float) -> None:
        self.user2index = self._index_map(user_ids)
        self.item2index = self._index_map(item_ids)
        self.rating2index = self._index_map(ratings)
        self.ratings = ratings
        self.user_indices = self._to_index(self.user2index, user_ids)
        self.item_indices = self._to_index(self.item2index, item_ids)
        self.rating_indices = self._to_index(self.rating2index, ratings)
        self.train_indices = np.random.uniform(0., 1., size=len(user_ids)) > test_size

    def train_adjacency_matrix(self):
        m = sp.csr_matrix((len(self.user2index), len(self.item2index)), dtype=np.float32)
        idx = self.train_indices
        # add 1 to rating_indices, because rating_indices starts with 0 and 0 is ignored in scr_matrix
        m[self.user_indices[idx], self.item_indices[idx]] = self.rating_indices[idx] + 1.
        return m

    def train_rating_adjacency_matrix(self) -> List[sp.csr_matrix]:
        adjacency_matrix = self.train_adjacency_matrix()
        return [sp.csr_matrix(adjacency_matrix == r + 1., dtype=np.float32) for r in range(len(self.rating2index))]

    def train_data(self):
        idx = self.train_indices
        shuffle_idx = sklearn.utils.shuffle(list(range(int(np.sum(idx)))))
        one_hot = self._to_one_hot(self.rating_indices[idx][shuffle_idx])
        return self.user_indices[idx][shuffle_idx], self.item_indices[idx][shuffle_idx], one_hot, self.ratings[idx][
            shuffle_idx]

    def convert(self, user_ids: List, item_ids: List) -> Tuple[np.ndarray, np.ndarray]:
        def _to_indices(id2index, ids):
            return np.array(list(map(lambda x: id2index.get(x, -1), ids)))

        return _to_indices(self.user2index, user_ids), _to_indices(self.item2index, item_ids)

    def test_data(self):
        idx = ~self.train_indices
        one_hot = self._to_one_hot(self.rating_indices[idx])
        return self.user_indices[idx], self.item_indices[idx], one_hot, self.ratings[idx]

    def rating(self):
        return np.array(sorted(self.rating2index.keys()))

    def _to_one_hot(self, ratings):
        return np.eye(len(self.rating2index))[ratings]

    @staticmethod
    def _index_map(x: np.ndarray) -> Dict:
        u = sorted(np.unique(x))
        return dict(zip(u, range(len(u))))

    @staticmethod
    def _to_index(id2map, ids) -> np.ndarray:
        return np.array(list(map(id2map.get, ids)))


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
                 save_directory_path: str = None) -> None:
        self.session = tf.Session()
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = ratings
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
        self.save_directory_path = save_directory_path
        self.dataset = _Dataset(self.user_ids, self.item_ids, self.ratings, self.test_size)
        self.graph = None

    def fit(self) -> List[str]:
        if self.graph is None:
            logger.info('making graph...')
            self.graph = self._make_graph()
            logger.info('done making graph')

        early_stopping = EarlyStopping(
            try_count=3,
            decay_speed=10,
            save_directory=self.save_directory_path,
            learning_rate=self.learning_rate,
            threshold=1e-4)

        test_user_indices, test_item_indices, test_labels, test_ratings = self.dataset.test_data()
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
                        _user_indices, _item_indices, _labels, _ratings = self.session.run(next_batch)
                        _rating_adjacency_matrix = [
                            self._eliminate(matrix, _user_indices, _item_indices) for matrix in rating_adjacency_matrix
                        ]
                        feed_dict = {
                            self.graph.input_learning_rate: early_stopping.learning_rate,
                            self.graph.input_dropout: self.dropout_rate,
                            self.graph.input_user: _user_indices,
                            self.graph.input_item: _item_indices,
                            self.graph.input_label: _labels,
                            self.graph.input_rating: _ratings,
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
                        logger.info(report[-1])
                    except tf.errors.OutOfRangeError:
                        feed_dict = {
                            self.graph.input_dropout: 0.0,
                            self.graph.input_user: test_user_indices,
                            self.graph.input_item: test_item_indices,
                            self.graph.input_label: test_labels,
                            self.graph.input_rating: test_ratings
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

    def predict(self, user_ids: List, item_ids: List) -> np.ndarray:
        if self.graph is None:
            RuntimeError('Please call fit first.')

        rating_adjacency_matrix = self.dataset.train_rating_adjacency_matrix()
        user_indices, item_indices = self.dataset.convert(user_ids, item_ids)
        valid_indices = np.logical_and(user_indices != -1, item_indices != -1)
        feed_dict = {
            self.graph.input_dropout: 0.0,
            self.graph.input_user: user_indices[valid_indices],
            self.graph.input_item: item_indices[valid_indices],
        }
        feed_dict.update({
            g: _convert_sparse_matrix_to_sparse_tensor(m)
            for g, m in zip(self.graph.input_adjacency_matrix, rating_adjacency_matrix)
        })
        feed_dict.update({g: m.count_nonzero() for g, m in zip(self.graph.input_edge_size, rating_adjacency_matrix)})
        with self.session.as_default():
            valid_predictions = self.session.run(self.graph.expectation, feed_dict=feed_dict)
        valid_predictions = valid_predictions.flatten()
        valid_predictions = np.clip(valid_predictions, self.dataset.rating()[0], self.dataset.rating()[-1])
        predictions = np.ones(len(user_ids), dtype=np.float) * np.mean(valid_predictions)
        predictions[valid_indices] = valid_predictions
        return predictions

    def _make_graph(self) -> GraphConvolutionalMatrixCompletionGraph:
        return GraphConvolutionalMatrixCompletionGraph(
            n_rating=len(self.dataset.rating2index),
            n_user=len(self.dataset.user2index),
            n_item=len(self.dataset.item2index),
            rating=self.dataset.rating(),
            normalization_type=self.normalization_type,
            encoder_hidden_size=self.encoder_hidden_size,
            encoder_size=self.encoder_size,
            weight_sharing=self.weight_sharing,
            scope_name=self.scope_name)

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
    adjacency_matrix = _make_sparse_matrix(n_users, n_items, n_data) + 2 * _make_sparse_matrix(n_users, n_items, n_data)
    user_ids = adjacency_matrix.tocoo().row
    item_ids = adjacency_matrix.tocoo().col
    ratings = adjacency_matrix.tocoo().data
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
        normalization_type='symmetric')
    for report in model.fit():
        print(report)


if __name__ == '__main__':
    main()
