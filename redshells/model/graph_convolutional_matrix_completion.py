import itertools
from datetime import datetime
from logging import getLogger
from typing import List, Optional, Union, Type
from abc import ABC, abstractmethod

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


class GraphABC(ABC):
    @abstractmethod
    def set_encoder(self):
        pass

    @abstractmethod
    def set_decoder(self):
        pass

    @abstractmethod
    def set_loss(self):
        pass


class GraphBuilder:
    def __init__(self, graph):
        self._graph = graph

    def build_graph(self):
        self._log_settings()
        self._set_placeholders()
        self._graph.set_encoder()
        self._graph.set_decoder()
        self._graph.set_loss()

    def get_graph(self):
        return self._graph

    def _set_placeholders(self):
        self._graph.input_learning_rate = tf.placeholder(dtype=np.float32, name='input_learning_rate')
        self._graph.input_dropout = tf.placeholder(dtype=np.float32, name='input_dropout')
        self._graph.input_label = tf.placeholder(dtype=np.int32, name='input_label')
        self._graph.input_user = tf.placeholder(dtype=np.int32, name='input_user')
        self._graph.input_item = tf.placeholder(dtype=np.int32, name='input_item')
        self._graph.input_user_feature_indices = tf.placeholder(dtype=np.int32, name='input_user_feature_indices')
        self._graph.input_item_feature_indices = tf.placeholder(dtype=np.int32, name='input_item_feature_indices')
        self._graph.input_edge_size = [tf.placeholder(dtype=np.int32, name=f'input_edge_size_{r}') for r in range(self._graph.n_rating)]
        self._graph.input_rating = tf.placeholder(dtype=np.int32, name='input_rating')
        self._graph.input_adjacency_matrix = [tf.sparse.placeholder(dtype=np.float32, name=f'input_adjacency_matrix_{r}') for r in range(self._graph.n_rating)]
        self._graph.rating = tf.constant(self._graph.rating_input.reshape((-1, 1)), dtype=np.float32),
        self._graph.input_user_features = [tf.placeholder(dtype=np.float32, shape=[None, size], name=f'user_features_{i}') for i, size in enumerate(self._graph.user_feature_sizes)]
        self._graph.input_item_features = [tf.placeholder(dtype=np.float32, shape=[None, size], name=f'item_features_{i}') for i, size in enumerate(self._graph.item_feature_sizes)]

    def _log_settings(self):
        logger.info(f'n_rating={self._graph.n_rating}; n_user={self._graph.n_user}; n_item={self._graph.n_item}')
        logger.info(f'graph scope_name={self._graph.scope_name}')

    # set params ----
    def set_n_rating(self, n_rating):
        self._graph.n_rating = n_rating

    def set_n_user(self, n_user):
        self._graph.n_user = n_user

    def set_n_item(self, n_item):
        self._graph.n_item = n_item

    def set_scope_name(self, scope_name):
        self._graph.scope_name = scope_name

    def set_rating(self, rating):
        self._graph.rating_input = rating  # reconsider name

    def set_user_feature_sizes(self, user_feature_sizes):
        self._graph.user_feature_sizes = user_feature_sizes

    def set_item_feature_sizes(self, item_feature_sizes):
        self._graph.item_feature_sizes = item_feature_sizes

    def set_normalization_type(self, normalization_type):
        self._graph.normalization_type = normalization_type

    def set_encoder_size(self, encoder_size):
        self._graph.encoder_size = encoder_size

    def set_encoder_hidden_size(self, encoder_hidden_size):
        self._graph.encoder_hidden_size = encoder_hidden_size

    def set_weight_sharing(self, weight_sharing):
        self._graph.weight_sharing = weight_sharing


class GraphMethods():
    @staticmethod
    def _set_adjustments(input_adjacency_matrix):
        user_adjustment = [tf.reshape(tf.div_no_nan(1., tf.sparse.reduce_sum(m, axis=1)), shape=(-1, 1)) for m in input_adjacency_matrix]
        item_adjustment = [tf.div_no_nan(1., tf.sparse.reduce_sum(m, axis=0)) for m in input_adjacency_matrix]
        return user_adjustment, item_adjustment

    @staticmethod
    def _adjust_adjacency_matrix(normalization_type, input_adjacency_matrix, item_adjustment, user_adjustment):
        if normalization_type == 'symmetric':
            adjusted_adjacency_matrix = [
                tf.sqrt(item) * m * tf.sqrt(user) for item, m, user in zip(item_adjustment, input_adjacency_matrix, user_adjustment)
            ]
            adjusted_adjacency_matrix_transpose = [tf.sparse.transpose(m) for m in adjusted_adjacency_matrix]
        elif normalization_type == 'left':
            adjusted_adjacency_matrix = [m * user for m, user in zip(input_adjacency_matrix, user_adjustment)]
            adjusted_adjacency_matrix_transpose = [tf.sparse.transpose(item * m) for item, m in zip(item_adjustment, input_adjacency_matrix)]
        elif normalization_type == 'right':
            adjusted_adjacency_matrix = [item * m for item, m in zip(item_adjustment, input_adjacency_matrix)]
            adjusted_adjacency_matrix_transpose = [tf.sparse.transpose(m * user) for m, user in zip(input_adjacency_matrix, user_adjustment)]
        else:
            raise ValueError(f'normalization_type must be "left", "right" or "symmetric", but {normalization_type} is passed.')
        return adjusted_adjacency_matrix, adjusted_adjacency_matrix_transpose

    @classmethod
    def get_cx(cls, input_adjacency_matrix, normalization_type, n_user, n_item):
        user_adjustment, item_adjustment = cls._set_adjustments(input_adjacency_matrix)
        adjusted_adjacency_matrix, adjusted_adjacency_matrix_transpose = cls._adjust_adjacency_matrix(normalization_type, input_adjacency_matrix, item_adjustment, user_adjustment)

        item_cx = [tf.sparse.slice(m, [0, 0], [n_user, n_item]) for m in adjusted_adjacency_matrix]
        user_cx = adjusted_adjacency_matrix_transpose
        return item_cx, user_cx

    @staticmethod
    def simple_layer(output_size: int, input_size: Optional[int] = None):
        layer = tf.keras.layers.Dense(output_size, use_bias=False, activation=None, kernel_initializer='glorot_normal')
        if input_size is not None:
            layer.build(input_shape=(None, input_size))
        return layer

    @classmethod
    def _encoder(cls, feature_size, encoder_hidden_size, n_rating, cx, dropout, weight_sharing, prefix, edge_size):
        cx = [tf.cond(tf.equal(dropout, 0.0), lambda: x, lambda: cls._dropout_sparse(x, 1. - dropout, num_nonzero_elements=s)) for x, s in zip(cx, edge_size)]
        weights = [cls.simple_layer(encoder_hidden_size, input_size=feature_size).weights[0] for _ in range(n_rating)]
        if weight_sharing:
            for r in range(n_rating - 1):
                weights[r + 1].assign_add(weights[r])

        encoder_hidden = [tf.sparse_tensor_dense_matmul(cx[r], weights[r], name=f'{prefix}_encoder_hidden_{r}') for r in range(n_rating)]
        result = tf.nn.relu(tf.reduce_sum(encoder_hidden, axis=0))
        return result

    @staticmethod
    def _dropout_sparse(x, keep_prob, num_nonzero_elements):
        random_tensor = keep_prob + tf.random_uniform([num_nonzero_elements])
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        return tf.sparse_retain(x, dropout_mask) / keep_prob

    @classmethod
    def get_encoder(cls, feature_size, encoder_hidden_size, n_rating, cx, dropout, weight_sharing, edge_size, prefix,
                    common_encoder_layer, input_idx):
        encoder_hidden = cls._encoder(
            feature_size=feature_size,
            encoder_hidden_size=encoder_hidden_size,
            n_rating=n_rating,
            cx=cx,
            dropout=dropout,
            weight_sharing=weight_sharing,
            edge_size=edge_size,
            prefix=prefix)

        encoder = common_encoder_layer(encoder_hidden)
        encoder = tf.gather(encoder, input_idx)
        return encoder

    @classmethod
    def get_feature_layers(cls, input_features, encoder_hidden_size, encoder_size, info_layer_activation=None):
        feature_layers = [cls._feature_convert_layer(encoder_hidden_size) for _ in input_features]
        side_info_layer = tf.keras.layers.Dense(encoder_size, use_bias=True, activation=info_layer_activation, kernel_initializer="glorot_normal")
        return feature_layers, side_info_layer

    @staticmethod
    def _feature_convert_layer(hidden_size: int, kernel_initializer='glorot_normal'):
        return tf.keras.layers.Dense(hidden_size, use_bias=False, activation=None, kernel_initializer=kernel_initializer)

    @staticmethod
    def add_feature_to_encoder(encoder, feature_layers, side_info_layers, input_features, input_feature_indices, ignore_hidden, ignore_embedding, feature_activation=tf.identity):
        if len(input_features) != 0 and not ignore_embedding:
            x = tf.reduce_prod([layer(feature)+1.0 for layer, feature in zip(feature_layers, input_features)], axis=0)
            x = tf.gather(side_info_layers(x), input_feature_indices)
            x = feature_activation(x)
            if encoder is None or ignore_hidden:
                return x
            else:
                return encoder + x
        else:
            assert encoder is not None, "Feature is required."
            return encoder


class GCMCGraph(GraphABC):
    """Graph class of `Graph Convolutional Matrix Completion`.
    """

    def set_encoder(self):
        with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE):
            self.item_cx, self.user_cx = GraphMethods.get_cx(self.input_adjacency_matrix, self.normalization_type, self.n_user, self.n_item)
            self.common_encoder_layer = GraphMethods.simple_layer(output_size=self.encoder_size)
            user_encoder_hidden = GraphMethods.get_encoder(self.n_item, self.encoder_hidden_size, self.n_rating, self.item_cx, self.input_dropout,
                                                           self.weight_sharing, self.input_edge_size, 'user', self.common_encoder_layer, self.input_user)
            item_encoder_hidden = GraphMethods.get_encoder(self.n_user, self.encoder_hidden_size, self.n_rating, self.user_cx, self.input_dropout,
                                                           self.weight_sharing, self.input_edge_size, 'item', self.common_encoder_layer, self.input_item)

            self.user_feature_layers, self.user_side_info_layer = GraphMethods.get_feature_layers(self.input_user_features, self.encoder_hidden_size, self.encoder_size, None)
            self.item_feature_layers, self.item_side_info_layer = GraphMethods.get_feature_layers(self.input_item_features, self.encoder_hidden_size, self.encoder_size, None)

            self.user_encoder = GraphMethods.add_feature_to_encoder(user_encoder_hidden, self.user_feature_layers, self.user_side_info_layer, self.input_user_features,
                                                                    self.input_user_feature_indices, False, False, feature_activation=tf.identity)
            ignore_item_hidden = False
            self.item_encoder = GraphMethods.add_feature_to_encoder(item_encoder_hidden, self.item_feature_layers, self.item_side_info_layer, self.input_item_features,
                                                                    self.input_item_feature_indices, ignore_item_hidden, False, feature_activation=tf.identity)

    def set_decoder(self):
        with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE):
            user_encoder = tf.nn.l2_normalize(self.user_encoder, axis=1)
            item_encoder = tf.nn.l2_normalize(self.item_encoder, axis=1)
            weights = [GraphMethods.simple_layer(self.encoder_size, input_size=self.encoder_size).weights[0] for _ in range(self.n_rating)]
            output = [tf.reduce_sum(tf.multiply(tf.matmul(user_encoder, w), item_encoder), axis=1) for w in weights]
            self.output = tf.stack(output, axis=1)

    def set_loss(self):
        with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE):
            self.probability = tf.nn.softmax(self.output)
            self.expectation = tf.matmul(self.probability, tf.reshape(self.rating, shape=(-1, 1)))
            self.rmse = tf.sqrt(tf.reduce_mean(tf.math.square(self.expectation - tf.reshape(tf.to_float(self.input_rating), shape=(-1, 1)))))

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.input_label))

            optimizer = tf.train.AdamOptimizer(learning_rate=self.input_learning_rate)
            self.op = optimizer.apply_gradients(optimizer.compute_gradients(self.loss))


class NoItemHiddenGCMCGraph(GraphABC):
    """Graph class of `No Item Hidden Layer Graph Convolutional Matrix Completion`,
    a variant of GCMC except using no item hidden layer for computing item embeddings.
    """

    def set_encoder(self):
        with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE):
            self.item_cx, self.user_cx = GraphMethods.get_cx(self.input_adjacency_matrix, self.normalization_type, self.n_user, self.n_item)
            self.common_encoder_layer = GraphMethods.simple_layer(output_size=self.encoder_size)
            user_encoder_hidden = GraphMethods.get_encoder(self.n_item, self.encoder_hidden_size, self.n_rating, self.item_cx, self.input_dropout,
                                                           self.weight_sharing, self.input_edge_size, 'user', self.common_encoder_layer, self.input_user)
            item_encoder_hidden = None

            self.user_feature_layers, self.user_side_info_layer = GraphMethods.get_feature_layers(self.input_user_features, self.encoder_hidden_size, self.encoder_size, None)
            self.item_feature_layers, self.item_side_info_layer = GraphMethods.get_feature_layers(self.input_item_features, self.encoder_hidden_size, self.encoder_size, None)

            self.user_encoder = GraphMethods.add_feature_to_encoder(user_encoder_hidden, self.user_feature_layers, self.user_side_info_layer, self.input_user_features,
                                                                    self.input_user_feature_indices, False, False)
            ignore_item_hidden = True
            self.item_encoder = GraphMethods.add_feature_to_encoder(item_encoder_hidden, self.item_feature_layers, self.item_side_info_layer, self.input_item_features,
                                                                    self.input_item_feature_indices, ignore_item_hidden, False)

    def set_decoder(self):
        with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE):
            user_encoder = tf.nn.l2_normalize(self.user_encoder, axis=1)
            item_encoder = tf.nn.l2_normalize(self.item_encoder, axis=1)
            weights = [GraphMethods.simple_layer(self.encoder_size, input_size=self.encoder_size).weights[0] for _ in range(self.n_rating)]
            output = [tf.reduce_sum(tf.multiply(tf.matmul(user_encoder, w), item_encoder), axis=1) for w in weights]
            self.output = tf.stack(output, axis=1)

    def set_loss(self):
        with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE):
            self.probability = tf.nn.softmax(self.output)
            self.expectation = tf.matmul(self.probability, tf.reshape(self.rating, shape=(-1, 1)))
            self.rmse = tf.sqrt(tf.reduce_mean(tf.math.square(self.expectation - tf.reshape(tf.to_float(self.input_rating), shape=(-1, 1)))))

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.input_label))

            optimizer = tf.train.AdamOptimizer(learning_rate=self.input_learning_rate)
            self.op = optimizer.apply_gradients(optimizer.compute_gradients(self.loss))


def graph_picker(graph_type):
    if graph_type == "gcmc":
        return GCMCGraph()
    elif graph_type == "nhmc":
        return NoItemHiddenGCMCGraph()
    else:
        raise ValueError("Invalid value set for graph_type.")


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
                 save_directory_path: str = None,
                 graph_type: str = 'gcmc') -> None:

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
        self.graph_type = graph_type
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

    def _predict(self, user_ids: List, item_ids: List, with_user_embedding, graph: GraphABC, dataset: GcmcGraphDataset,
                 session: tf.Session) -> np.ndarray:
        if graph is None:
            RuntimeError('Please call fit first.')

        rating_adjacency_matrix = dataset.train_rating_adjacency_matrix()
        user_indices, item_indices = dataset.to_indices(user_ids, item_ids)
        if not with_user_embedding:
            user_indices = np.array([0] * len(user_indices))

        user_feature_indices, item_feature_indices = dataset.to_feature_indices(user_ids, item_ids)
        input_data = dict(user=user_indices, item=item_indices, user_feature_indices=user_feature_indices, item_feature_indices=item_feature_indices)
        feed_dict = self._feed_dict(input_data, graph, dataset, rating_adjacency_matrix, is_train=False)
        with session.as_default():
            predictions = session.run(graph.expectation, feed_dict=feed_dict)
        predictions = predictions.flatten()
        predictions = np.clip(predictions, dataset.rating()[0], dataset.rating()[-1])
        return predictions

    def _get_feature(self, user_ids: List, item_ids: List, with_user_embedding,
                     graph: GraphABC, dataset: GcmcGraphDataset,
                     session: tf.Session, feature: str) -> np.ndarray:
        if graph is None:
            RuntimeError('Please call fit first.')

        rating_adjacency_matrix = dataset.train_rating_adjacency_matrix()
        user_indices, item_indices = dataset.to_indices(user_ids, item_ids)
        if not with_user_embedding:
            user_indices = np.array([0] * len(user_indices))

        user_feature_indices, item_feature_indices = dataset.to_feature_indices(user_ids, item_ids)
        input_data = dict(user=user_indices, item=item_indices, user_feature_indices=user_feature_indices,
                          item_feature_indices=item_feature_indices)
        feed_dict = self._feed_dict(input_data, graph, dataset, rating_adjacency_matrix, is_train=False)
        encoder_map = dict(user=graph.user_encoder, item=graph.item_encoder)
        with session.as_default():
            feature = session.run(encoder_map[feature], feed_dict=feed_dict)
        return feature

    def _feed_dict(self, input_data, graph, graph_dataset, rating_adjacency_matrix, dropout_rate: float = 0.0, learning_rate: float = 0.0, is_train: bool = True):
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

    def _make_graph(self):
        graph = graph_picker(self.graph_type)
        self.graph_builder = GraphBuilder(graph)
        self.graph_builder.set_n_rating(self.graph_dataset.n_rating)
        self.graph_builder.set_n_user(self.graph_dataset.n_user)
        self.graph_builder.set_n_item(self.graph_dataset.n_item)
        self.graph_builder.set_scope_name(self.scope_name)
        self.graph_builder.set_rating(self.graph_dataset.rating())
        self.graph_builder.set_user_feature_sizes([x.shape[1] for x in self.graph_dataset.user_features])
        self.graph_builder.set_item_feature_sizes([x.shape[1] for x in self.graph_dataset.item_features])
        self.graph_builder.set_normalization_type(self.normalization_type)
        self.graph_builder.set_encoder_size(self.encoder_size)
        self.graph_builder.set_encoder_hidden_size(self.encoder_hidden_size)
        self.graph_builder.set_weight_sharing(self.weight_sharing)
        self.graph_builder.build_graph()

        return self.graph_builder.get_graph()

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
