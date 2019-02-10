from logging import getLogger
from typing import List, Optional, Dict

import numpy as np
import scipy.sparse as sp
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
    return tf.SparseTensor(indices, coo.data, coo.shape)


class GraphConvolutionalMatrixCompletionGraph(object):
    def __init__(self,
                 adjacency_matrix: List[sp.csr_matrix],
                 rating: np.ndarray,
                 user_adjustment: List[np.ndarray],
                 item_adjustment: List[np.ndarray],
                 encoder_hidden_size: int,
                 encoder_size: int,
                 user_feature: Optional[np.ndarray] = None,
                 item_feature: Optional[np.ndarray] = None,
                 scope_name: str = 'GraphConvolutionalMatrixCompletionGraph') -> None:
        n_ratings = len(adjacency_matrix)
        user_feature_size = user_feature.shape[1] if user_feature else user_adjustment[0].shape[0]
        item_feature_size = item_feature.shape[1] if item_feature else item_adjustment[0].shape[0]
        # placeholder
        self.input_learning_rate = tf.placeholder(dtype=np.float32, name='learning_rate')
        self.input_label = tf.placeholder(dtype=np.int32, name='label')
        self.input_user = tf.placeholder(dtype=np.int32, name='user')
        self.input_item = tf.placeholder(dtype=np.int32, name='item')
        self.input_rating = tf.placeholder(dtype=np.int32, name='rating')
        # shape=(n_user, n_item)
        self.adjacency_matrix = [_convert_sparse_matrix_to_sparse_tensor(m) for m in adjacency_matrix]
        # adjustment
        self.user_adjustment = [tf.reshape(tf.constant(u), shape=(-1, 1)) for u in user_adjustment]
        self.item_adjustment = [tf.constant(i) for i in item_adjustment]
        self.rating = tf.constant(rating.reshape((-1, 1)), dtype=np.float32)
        # feature
        self.user_feature = tf.constant(user_feature) if user_feature else None
        self.item_feature = tf.constant(item_feature) if item_feature else None

        # adjusted adjacency matrix
        self.adjusted_adjacency_matrix = [
            item * m * user for item, m, user in zip(self.item_adjustment, self.adjacency_matrix, self.user_adjustment)
        ]
        self.adjusted_adjacency_matrix_transpose = [tf.sparse.transpose(m) for m in self.adjusted_adjacency_matrix]

        # C X
        # (n_user, item_feature_size)
        if self.item_feature:
            self.item_cx = [tf.sparse_tensor_dense_matmul(m, self.item_feature) for m in self.adjusted_adjacency_matrix]
        else:
            self.item_cx = self.adjusted_adjacency_matrix
        # (n_item, user_feature_size)
        if self.user_feature:
            self.user_cx = [tf.sparse_matmul(m, self.user_feature) for m in self.adjusted_adjacency_matrix_transpose]
        else:
            self.user_cx = self.adjusted_adjacency_matrix_transpose

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            # encoder
            self.item_encoder_weight = [
                _make_weight_variable(shape=(item_feature_size, encoder_hidden_size), name=f'item_encoder_weight_{r}')
                for r in range(n_ratings)
            ]
            self.user_encoder_weight = [
                _make_weight_variable(shape=(user_feature_size, encoder_hidden_size), name=f'user_encoder_weight_{r}')
                for r in range(n_ratings)
            ]
            self.common_encoder_weight = _make_weight_variable(
                shape=(encoder_hidden_size, encoder_size), name=f'common_encoder_weight')

            self.item_encoder_hidden = tf.nn.relu(
                tf.reduce_sum([
                    _dot(
                        self.user_cx[r],
                        self.user_encoder_weight[r],
                        name=f'item_encoder_hidden_{r}',
                        x_is_sparse=self.user_feature is None) for r in range(n_ratings)
                ],
                              axis=0))

            self.user_encoder_hidden = tf.nn.relu(
                tf.reduce_sum([
                    _dot(
                        self.item_cx[r],
                        self.item_encoder_weight[r],
                        name=f'user_encoder_hidden_{r}',
                        x_is_sparse=self.item_feature is None) for r in range(n_ratings)
                ],
                              axis=0))

            self.item_encoder = tf.matmul(self.item_encoder_hidden, self.common_encoder_weight)
            self.user_encoder = tf.matmul(self.user_encoder_hidden, self.common_encoder_weight)

            # decoder
            self.decoder_weight = [
                _make_weight_variable(shape=(encoder_size, encoder_size), name=f'decoder_weight_{r}')
                for r in range(n_ratings)
            ]

            user_encoder = tf.gather(self.user_encoder, self.input_user)
            item_encoder = tf.gather(self.item_encoder, self.input_item)

            self.output = tf.stack([
                tf.reduce_sum(tf.multiply(tf.matmul(user_encoder, w), item_encoder), axis=1)
                for w in self.decoder_weight
            ],
                                   axis=1)
            self.probability = tf.nn.softmax(self.output)
            self.expectation = tf.matmul(self.probability, tf.reshape(self.rating, shape=(-1, 1)))
            self.rmse = tf.sqrt(
                tf.losses.mean_squared_error(self.expectation, tf.reshape(self.input_rating, shape=(-1, 1))))

            # loss
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.input_label))

            # optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate=self.input_learning_rate)
            self.op = optimizer.apply_gradients(optimizer.compute_gradients(self.loss))


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

    def train_data(self):
        idx = self.train_indices
        one_hot = self._to_one_hot(self.rating_indices[idx])
        return self.user_indices[idx], self.item_indices[idx], one_hot, self.ratings[idx]

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
                 learning_rate: float,
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
        self.learning_rate = learning_rate
        self.save_directory_path = save_directory_path
        self.dataset = _Dataset(self.user_ids, self.item_ids, self.ratings, self.test_size)
        self.graph = None

    def fit(self) -> None:
        if self.graph is None:
            logger.info('making graph...')
            self.graph = self._make_graph()
            logger.info('done making graph')

        early_stopping = EarlyStopping(save_directory=self.save_directory_path, learning_rate=self.learning_rate)

        test_user_indices, test_item_indices, test_labels, test_ratings = self.dataset.test_data()
        with self.session.as_default():
            self.session.run(tf.global_variables_initializer())
            dataset = tf.data.Dataset.from_tensor_slices(self.dataset.train_data())
            dataset = dataset.batch(self.batch_size)
            iterator = dataset.make_initializable_iterator()
            next_batch = iterator.get_next()

            logger.info('start to optimize...')
            for i in range(self.epoch_size):
                self.session.run(iterator.initializer)

                train_loss = None
                train_rmse = None
                while True:
                    try:
                        _user_indices, _item_indices, _labels, _ratings = self.session.run(next_batch)
                        feed_dict = {
                            self.graph.input_learning_rate: early_stopping.learning_rate,
                            self.graph.input_user: _user_indices,
                            self.graph.input_item: _item_indices,
                            self.graph.input_label: _labels,
                            self.graph.input_rating: _ratings,
                        }
                        _, train_loss, train_rmse, expectation = self.session.run(
                            [self.graph.op, self.graph.loss, self.graph.rmse, self.graph.expectation],
                            feed_dict=feed_dict)
                    except tf.errors.OutOfRangeError:
                        logger.info(f'train: epoch={i + 1}/{self.epoch_size}, loss={train_loss}, rmse={train_rmse}.')
                        feed_dict = {
                            self.graph.input_user: test_user_indices,
                            self.graph.input_item: test_item_indices,
                            self.graph.input_label: test_labels,
                            self.graph.input_rating: test_ratings
                        }
                        test_loss, test_rmse = self.session.run([self.graph.loss, self.graph.rmse], feed_dict=feed_dict)
                        logger.info(f'test: epoch={i + 1}/{self.epoch_size}, loss={test_loss}, rmse={test_rmse}.')
                        break

                if early_stopping.does_stop(test_rmse, self.session):
                    break

    def _make_graph(self) -> GraphConvolutionalMatrixCompletionGraph:
        adjacency_matrix = self.dataset.train_adjacency_matrix()

        rating_classes = sorted(np.unique(adjacency_matrix.tocoo().data))
        rating_adjacency_matrix = [
            sp.csr_matrix(adjacency_matrix == rating_class, dtype=np.float32) for rating_class in rating_classes
        ]

        def adjustment(x):
            x = np.array(x).flatten()
            return np.divide(1, np.sqrt(x), where=x != 0).reshape(-1)

        user_adjustment = [adjustment(matrix.sum(axis=1)) for matrix in rating_adjacency_matrix]
        item_adjustment = [adjustment(matrix.sum(axis=0)) for matrix in rating_adjacency_matrix]

        return GraphConvolutionalMatrixCompletionGraph(
            adjacency_matrix=rating_adjacency_matrix,
            rating=self.dataset.rating(),
            user_adjustment=user_adjustment,
            item_adjustment=item_adjustment,
            encoder_hidden_size=self.encoder_hidden_size,
            encoder_size=self.encoder_size,
            scope_name=self.scope_name)

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
    n_users = 100
    n_items = 230
    n_data = 3000
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
        epoch_size=1000,
        learning_rate=0.01)
    model.fit()


if __name__ == '__main__':
    main()
