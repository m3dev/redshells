import itertools
from datetime import datetime
from logging import getLogger
from typing import List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf

import redshells
from redshells.model.early_stopping import EarlyStopping
from redshells.model.gcmc_dataset import GcmcGraphDataset, GcmcDataset

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
                 user_feature_sizes: List[int],
                 item_feature_sizes: List[int],
                 scope_name: str = 'GraphConvolutionalMatrixCompletionGraph',
                 weight_sharing: bool = True,
                 ignore_item_embedding: bool = False) -> None:
        logger.info(f'n_rating={n_rating}; n_user={n_user}; n_item={n_item}')

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            # placeholder
            self.input_learning_rate = tf.placeholder(dtype=np.float32, name='learning_rate')
            self.input_dropout = tf.placeholder(dtype=np.float32, name='learning_rate')
            self.input_label = tf.placeholder(dtype=np.int32, name='label')
            self.input_user = tf.placeholder(dtype=np.int32, name='user')
            self.input_item = tf.placeholder(dtype=np.int32, name='item')
            self.input_user_feature_indices = tf.placeholder(dtype=np.int32, name='input_user_feature_indices')
            self.input_item_feature_indices = tf.placeholder(dtype=np.int32, name='input_item_feature_indices')
            self.input_edge_size = [tf.placeholder(dtype=np.int32, name=f'edge_size_{r}') for r in range(n_rating)]
            self.input_rating = tf.placeholder(dtype=np.int32, name='rating')
            # shape=(n_user, n_item)
            self.input_adjacency_matrix = [tf.sparse.placeholder(dtype=np.float32, name=f'adjacency_matrix_{r}') for r in range(n_rating)]
            # adjustment
            self.user_adjustment = [tf.reshape(tf.div_no_nan(1., tf.sparse.reduce_sum(m, axis=1)), shape=(-1, 1)) for m in self.input_adjacency_matrix]
            self.item_adjustment = [tf.div_no_nan(1., tf.sparse.reduce_sum(m, axis=0)) for m in self.input_adjacency_matrix]
            self.rating = tf.constant(rating.reshape((-1, 1)), dtype=np.float32)
            # features
            self.input_user_features = [
                tf.placeholder(dtype=np.float32, shape=[None, size], name=f'user_features_{i}') for i, size in enumerate(user_feature_sizes)
            ]
            self.input_item_features = [
                tf.placeholder(dtype=np.float32, shape=[None, size], name=f'item_features_{i}') for i, size in enumerate(item_feature_sizes)
            ]
            # adjusted adjacency matrix
            self._adjust_adjacency_matrix(normalization_type)

            # C X
            # (n_user, item_feature_size)
            self.item_cx = [tf.sparse.slice(m, [0, 0], [n_user, n_item]) for m in self.adjusted_adjacency_matrix]
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

            self._add_user_feature(encoder_hidden_size, encoder_size)
            self._add_item_feature(encoder_hidden_size, encoder_size, ignore_item_embedding)

            # decoder
            self.output = self._decoder(encoder_size=encoder_size, n_rating=n_rating, user_encoder=self.user_encoder, item_encoder=self.item_encoder)

            # output
            self.probability = tf.nn.softmax(self.output)
            self.expectation = tf.matmul(self.probability, tf.reshape(self.rating, shape=(-1, 1)))
            self.rmse = tf.sqrt(tf.reduce_mean(tf.math.square(self.expectation - tf.reshape(tf.to_float(self.input_rating), shape=(-1, 1)))))

            # loss
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.input_label))

            # optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate=self.input_learning_rate)
            self.op = optimizer.apply_gradients(optimizer.compute_gradients(self.loss))

    def _add_item_feature(self, encoder_hidden_size, encoder_size, ignore_item_embedding):
        if len(self.input_item_features) != 0:
            layer = self._cross_feature_layer(hidden_size=encoder_hidden_size, size=encoder_size, input_data=self.input_item_features)
            layer = tf.gather(layer, self.input_item_feature_indices)
            if ignore_item_embedding:
                self.item_encoder = layer
            else:
                self.item_encoder = self.item_encoder + layer

    def _add_user_feature(self, encoder_hidden_size, encoder_size):
        if len(self.input_user_features) != 0:
            layer = self._cross_feature_layer(hidden_size=encoder_hidden_size, size=encoder_size, input_data=self.input_user_features)
            layer = tf.gather(layer, self.input_user_feature_indices)
            self.user_encoder = self.user_encoder + layer

    def _adjust_adjacency_matrix(self, normalization_type):
        if normalization_type == 'symmetric':
            self.adjusted_adjacency_matrix = [
                tf.sqrt(item) * m * tf.sqrt(user) for item, m, user in zip(self.item_adjustment, self.input_adjacency_matrix, self.user_adjustment)
            ]
            self.adjusted_adjacency_matrix_transpose = [tf.sparse.transpose(m) for m in self.adjusted_adjacency_matrix]
        elif normalization_type == 'left':
            self.adjusted_adjacency_matrix = [m * user for m, user in zip(self.input_adjacency_matrix, self.user_adjustment)]
            self.adjusted_adjacency_matrix_transpose = [tf.sparse.transpose(item * m) for item, m in zip(self.item_adjustment, self.input_adjacency_matrix)]
        elif normalization_type == 'right':
            self.adjusted_adjacency_matrix = [item * m for item, m in zip(self.item_adjustment, self.input_adjacency_matrix)]
            self.adjusted_adjacency_matrix_transpose = [tf.sparse.transpose(m * user) for m, user in zip(self.input_adjacency_matrix, self.user_adjustment)]
        else:
            raise ValueError(f'normalization_type must be "left", "right" or "symmetric", but {normalization_type} is passed.')

    @staticmethod
    def _feature_layer(hidden_size: int, input_data):
        x = tf.keras.layers.Dense(hidden_size, use_bias=False, activation=None, kernel_initializer='glorot_normal')(input_data)
        return x

    @classmethod
    def _cross_feature_layer(cls, hidden_size: int, size: int, input_data: List[np.ndarray]):
        layers = [cls._feature_layer(hidden_size, data) + 1.0 for data in input_data]
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
        cx = [tf.cond(tf.equal(dropout, 0.0), lambda: x, lambda: cls._dropout_sparse(x, 1. - dropout, num_nonzero_elements=s)) for x, s in zip(cx, edge_size)]
        weights = [cls._simple_layer(encoder_hidden_size, input_size=feature_size).weights[0] for _ in range(n_rating)]
        if weight_sharing:
            for r in range(n_rating - 1):
                weights[r + 1].assign_add(weights[r])

        encoder_hidden = [tf.sparse_tensor_dense_matmul(cx[r], weights[r], name=f'{prefix}_encoder_hidden_{r}') for r in range(n_rating)]
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
                 graph_dataset: GcmcGraphDataset,
                 encoder_hidden_size: int,
                 encoder_size: int,
                 scope_name: str,
                 batch_size: int,
                 epoch_size: int,
                 dropout_rate: float,
                 learning_rate: float,
                 normalization_type: str,
                 weight_sharing: bool = True,
                 ignore_item_embedding: bool = False,
                 save_directory_path: str = None) -> None:
        self.session = tf.Session()
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_size = encoder_size
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.scope_name = scope_name
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.normalization_type = normalization_type
        self.weight_sharing = weight_sharing
        self.ignore_item_embedding = ignore_item_embedding
        self.save_directory_path = save_directory_path
        self.graph_dataset = graph_dataset
        self.graph = None

    def fit(self, try_count=1, decay_speed=10.) -> List[str]:
        if self.graph is None:
            logger.info('making graph...')
            self.graph = self._make_graph()
            logger.info('done making graph')

        early_stopping = EarlyStopping(
            try_count=try_count, decay_speed=decay_speed, save_directory=self.save_directory_path, learning_rate=self.learning_rate, threshold=1e-4)

        test_data = self.graph_dataset.test_data()
        report = []
        with self.session.as_default():
            self.session.run(tf.global_variables_initializer())
            dataset = tf.data.Dataset.from_tensor_slices(self.graph_dataset.train_data())
            dataset = dataset.shuffle(buffer_size=self.batch_size)
            batch = dataset.batch(self.batch_size)
            iterator = batch.make_initializable_iterator()
            next_batch = iterator.get_next()
            rating_adjacency_matrix = self.graph_dataset.train_rating_adjacency_matrix()

            logger.info('start to optimize...')
            for i in range(self.epoch_size):
                self.session.run(iterator.initializer)
                while True:
                    try:
                        train_data = self.session.run(next_batch)
                        _rating_adjacency_matrix = [self._eliminate(matrix, train_data['user'], train_data['item']) for matrix in rating_adjacency_matrix]
                        feed_dict = self._feed_dict(train_data, self.graph, self.graph_dataset, _rating_adjacency_matrix, self.dropout_rate,
                                                    early_stopping.learning_rate)
                        _, train_loss, train_rmse = self.session.run([self.graph.op, self.graph.loss, self.graph.rmse], feed_dict=feed_dict)
                        report.append(f'train: epoch={i + 1}/{self.epoch_size}, loss={train_loss}, rmse={train_rmse}.')
                    except tf.errors.OutOfRangeError:
                        logger.info(report[-1])
                        feed_dict = self._feed_dict(test_data, self.graph, self.graph_dataset, rating_adjacency_matrix)
                        test_loss, test_rmse = self.session.run([self.graph.loss, self.graph.rmse], feed_dict=feed_dict)
                        report.append(f'test: epoch={i + 1}/{self.epoch_size}, loss={test_loss}, rmse={test_rmse}.')
                        logger.info(report[-1])
                        break

                if early_stopping.does_stop(test_rmse, self.session):
                    break
        return report

    def predict(self, user_ids: List, item_ids: List, with_user_embedding: bool = True) -> np.ndarray:
        return self._predict(
            user_ids=user_ids, item_ids=item_ids, with_user_embedding=with_user_embedding, graph=self.graph, dataset=self.graph_dataset, session=self.session)

    def predict_with_new_items(self, user_ids: List, item_ids: List, additional_dataset: GcmcDataset, with_user_embedding: bool = True) -> np.ndarray:
        dataset = self.graph_dataset.add_dataset(additional_dataset, add_item=True)
        return self._predict(
            user_ids=user_ids, item_ids=item_ids, with_user_embedding=with_user_embedding, graph=self.graph, dataset=dataset, session=self.session)

    def get_user_feature(self, user_ids: List, item_ids: List, additional_dataset: GcmcDataset, with_user_embedding: bool = True) -> np.ndarray:
        dataset = self.graph_dataset.add_dataset(additional_dataset, add_item=True)
        return self._get_feature(user_ids=user_ids, item_ids=item_ids, with_user_embedding=with_user_embedding, graph=self.graph, dataset=dataset, session=self.session, feature='user')

    def get_item_feature(self, user_ids: List, item_ids: List, additional_dataset: GcmcDataset, with_user_embedding: bool = True) -> np.ndarray:
        dataset = self.graph_dataset.add_dataset(additional_dataset, add_item=True)
        return self._get_feature(user_ids=user_ids, item_ids=item_ids, with_user_embedding=with_user_embedding, graph=self.graph, dataset=dataset, session=self.session, feature='item')

    @classmethod
    def _predict(cls, user_ids: List, item_ids: List, with_user_embedding, graph: GraphConvolutionalMatrixCompletionGraph, dataset: GcmcGraphDataset,
                 session: tf.Session) -> np.ndarray:
        if graph is None:
            RuntimeError('Please call fit first.')

        rating_adjacency_matrix = dataset.train_rating_adjacency_matrix()
        user_indices, item_indices = dataset.to_indices(user_ids, item_ids)
        if not with_user_embedding:
            user_indices = np.array([0] * len(user_indices))  # TODO use default user index.

        user_feature_indices, item_feature_indices = dataset.to_feature_indices(user_ids, item_ids)
        input_data = dict(user=user_indices, item=item_indices, user_feature_indices=user_feature_indices, item_feature_indices=item_feature_indices)
        feed_dict = cls._feed_dict(input_data, graph, dataset, rating_adjacency_matrix, is_train=False)
        with session.as_default():
            predictions = session.run(graph.expectation, feed_dict=feed_dict)
        predictions = predictions.flatten()
        predictions = np.clip(predictions, dataset.rating()[0], dataset.rating()[-1])
        return predictions

    @classmethod
    def _get_feature(cls, user_ids: List, item_ids: List, with_user_embedding,
                     graph: GraphConvolutionalMatrixCompletionGraph, dataset: GcmcGraphDataset,
                     session: tf.Session, feature: str) -> np.ndarray:
        if graph is None:
            RuntimeError('Please call fit first.')

        rating_adjacency_matrix = dataset.train_rating_adjacency_matrix()
        user_indices, item_indices = dataset.to_indices(user_ids, item_ids)
        if not with_user_embedding:
            user_indices = np.array([0] * len(user_indices))  # TODO use default user index.

        user_feature_indices, item_feature_indices = dataset.to_feature_indices(user_ids, item_ids)
        input_data = dict(user=user_indices, item=item_indices, user_feature_indices=user_feature_indices,
                          item_feature_indices=item_feature_indices)
        feed_dict = cls._feed_dict(input_data, graph, dataset, rating_adjacency_matrix, is_train=False)
        encoder_map = dict(user=graph.user_encoder, item=graph.item_encoder)
        with session.as_default():
            feature = session.run(encoder_map[feature], feed_dict=feed_dict)
        return feature

    @staticmethod
    def _feed_dict(input_data, graph, graph_dataset, rating_adjacency_matrix, dropout_rate: float = 0.0, learning_rate: float = 0.0, is_train: bool = True):
        feed_dict = {
            graph.input_learning_rate: learning_rate,
            graph.input_dropout: dropout_rate,
            graph.input_user: input_data['user'],
            graph.input_item: input_data['item'],
            graph.input_user_feature_indices: input_data['user_feature_indices'],
            graph.input_item_feature_indices: input_data['item_feature_indices'],
        }

        if is_train:
            feed_dict.update({graph.input_label: input_data['label'], graph.input_rating: input_data['rating']})

        feed_dict.update({g: _convert_sparse_matrix_to_sparse_tensor(m) for g, m in zip(graph.input_adjacency_matrix, rating_adjacency_matrix)})
        feed_dict.update({g: m.count_nonzero() for g, m in zip(graph.input_edge_size, rating_adjacency_matrix)})
        feed_dict.update(dict(zip(graph.input_user_features, graph_dataset.user_features)))
        feed_dict.update(dict(zip(graph.input_item_features, graph_dataset.item_features)))
        return feed_dict

    def predict_item_scores(self, item_ids: List, with_user_embedding: bool = True) -> pd.DataFrame:
        user_ids = self.graph_dataset.user_ids
        users, items = zip(*list(itertools.product(user_ids, item_ids)))
        predicts = self.predict(user_ids=users, item_ids=items, with_user_embedding=with_user_embedding)
        results = pd.DataFrame(dict(user=users, item=items, score=predicts))
        results.sort_values('score', ascending=False, inplace=True)
        return results

    def predict_item_scores_with_new_items(self, item_ids: List, additional_dataset: GcmcDataset, with_user_embedding: bool = True) -> pd.DataFrame:
        user_ids = self.graph_dataset.user_ids
        users, items = zip(*list(itertools.product(user_ids, item_ids)))
        predicts = self.predict_with_new_items(user_ids=users, item_ids=items, additional_dataset=additional_dataset, with_user_embedding=with_user_embedding)
        results = pd.DataFrame(dict(user=users, item=items, score=predicts))
        results.sort_values('score', ascending=False, inplace=True)
        return results

    def get_user_feature_with_new_items(self, item_ids: List, additional_dataset: GcmcDataset, with_user_embedding: bool = True) -> pd.DataFrame:
        user_ids = self.graph_dataset.user_ids
        users, items = zip(*list(itertools.product(user_ids, item_ids)))
        user_feature = self.get_user_feature(user_ids=users, item_ids=items, additional_dataset=additional_dataset, with_user_embedding=with_user_embedding)
        indices = [i for i in range(len(users)) if i % len(item_ids) == 0]
        return user_ids, user_feature[indices]

    def get_item_feature_with_new_items(self, item_ids: List, additional_dataset: GcmcDataset, with_user_embedding: bool = True) -> pd.DataFrame:
        user_id = self.graph_dataset.user_ids[0]
        users, items = zip(*[(user_id, item_id) for item_id in item_ids])
        item_feature = self.get_item_feature(user_ids=users, item_ids=items, additional_dataset=additional_dataset, with_user_embedding=with_user_embedding)
        return items, item_feature

    def _make_graph(self) -> GraphConvolutionalMatrixCompletionGraph:
        return GraphConvolutionalMatrixCompletionGraph(
            n_rating=self.graph_dataset.n_rating,
            n_user=self.graph_dataset.n_user,
            n_item=self.graph_dataset.n_item,
            rating=self.graph_dataset.rating(),
            normalization_type=self.normalization_type,
            encoder_hidden_size=self.encoder_hidden_size,
            encoder_size=self.encoder_size,
            weight_sharing=self.weight_sharing,
            scope_name=self.scope_name,
            user_feature_sizes=[x.shape[1] for x in self.graph_dataset.user_features],
            item_feature_sizes=[x.shape[1] for x in self.graph_dataset.item_features],
            ignore_item_embedding=self.ignore_item_embedding)

    @staticmethod
    def _eliminate(matrix: sp.csr_matrix, user_indices, item_indices):
        matrix = matrix.copy()
        # `lil_matrix` is too slow
        matrix[list(user_indices), list(item_indices)] = 0
        matrix.eliminate_zeros()
        return matrix

    def save(self, file_path: str) -> None:
        redshells.model.utils.save_tf_session(self, self.session, file_path)

    @staticmethod
    def load(file_path: str) -> 'GraphConvolutionalMatrixCompletion':
        session = tf.Session()
        model = redshells.model.utils.load_tf_session(GraphConvolutionalMatrixCompletion, session, file_path,
                                                      GraphConvolutionalMatrixCompletion._make_graph)  # type: GraphConvolutionalMatrixCompletion
        return model


def _make_sparse_matrix(n, m, n_values):
    x = np.zeros(shape=(n, m), dtype=np.float32)
    x[np.random.choice(range(n), n_values), np.random.choice(range(m), n_values)] = 1.0
    return sp.csr_matrix(x)
