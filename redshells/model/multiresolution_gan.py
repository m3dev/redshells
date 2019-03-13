from builtins import sorted
from collections import Counter
from logging import getLogger
from typing import List, Optional, Dict, Tuple, Any

import numpy as np
import scipy.sparse as sp
import sklearn
import tensorflow as tf
from tqdm import tqdm

import redshells
from redshells.model.early_stopping import EarlyStopping

logger = getLogger(__name__)

def _cos_sim(a, b):
    a, b = map(lambda x: tf.nn.l2_normalize(x, axis=2), [a, b])
    loss = tf.losses.cosine_distance(a, b, axis=2, reduction=tf.losses.Reduction.NONE)
    return 1.-loss

def _split(x, axis=0):
    return tf.split(x, x.shape[axis], axis=axis)

def _matmul(a, b, a_is_sparse=False, b_is_sparse=False):
    if a_is_sparse: return tf.sparse_tensor_dense_matmul(a, b)
    elif b_is_sparse: return tf.transpose(tf.sparse_tensor_dense_matmul(tf.sparse.transpose(b), tf.transpose(a)))
    else: return tf.matmul(a,b)

def _convert_sparse_matrix_to_sparse_tensor(x):
    coo = x.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensorValue(indices, coo.data, coo.shape)

class MultiresolutionGraphAttentionNetworksGraph(object):
    def __init__(self,
                 n_rating: int,
                 n_user: int,
                 n_item: int,
                 batch_size: int,
                 rating: np.ndarray,
                 lmda: float,
                 user_feature: Optional[np.ndarray] = None,
                 item_feature: Optional[np.ndarray] = None,
                 scope_name: str = 'MultiresolutionGraphAttentionNetworksGraph',
                 use_bias: bool = False) -> None:
        logger.info(f'n_rating={n_rating}; n_user={n_user}; n_item={n_item}')
        user_feature_size = user_feature.shape[1] if user_feature is not None else n_user
        item_feature_size = item_feature.shape[1] if item_feature is not None else n_item

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            self.input_learning_rate = tf.placeholder(dtype=np.float32, name='learning_rate')
            self.input_dropout = tf.placeholder(dtype=np.float32, name='dropout')
            self.input_label = tf.placeholder(dtype=np.int32, name='label')
            self.input_label_weight = tf.placeholder(dtype=np.float32, name='label_weight')
            self.input_user = tf.placeholder(dtype=np.int32, name='user')
            self.input_item = tf.placeholder(dtype=np.int32, name='item')
            self.input_rating = tf.placeholder(dtype=np.int32, name='rating')
            self.input_edge_size = tf.placeholder(dtype=np.float32, name='edge_size')

            self.input_adjacency_matrix = tf.sparse.placeholder(dtype=np.float32, name=f'adjacency_matrix')
            self.left_adjustment = tf.div_no_nan(1., tf.sparse.reduce_sum(self.input_adjacency_matrix,  axis=1))
            self.right_adjustment= tf.div_no_nan(1., tf.sparse.reduce_sum(self.input_adjacency_matrix,  axis=0))
            
            tilde_degree_reverse_matrix = tf.diag(lmda + self.left_adjustment)
            sparse_matrix_a = tf.sparse.add(self.input_adjacency_matrix, tf.sparse.eye(n_item)*lmda)
            self.adjusted_adjacency_matrix = _matmul(tilde_degree_reverse_matrix, sparse_matrix_a, b_is_sparse=True)
            make_shape = lambda x: tf.reshape(x, [batch_size, -1])
            self.input_user, self.input_item, self.input_rating = map(make_shape, [self.input_user, self.input_item, self.input_rating])
            self.rating = tf.constant(rating.reshape((-1, 1)), dtype=np.float32)
            
            if user_feature is not None:
                self.user_feature = tf.constant(user_feature) 
            else:
                self.user_feature = tf.keras.layers.Embedding(n_user, item_feature_size, input_length=1)(self.input_user)

            #d_e = item_feature_size
            #d_g = edge_size
            #d_q = num_user_attributes

            def _gcn_conv(x, adj_matrix, name, activation=tf.nn.relu, use_bias=False):
                with tf.variable_scope(name):
                    w = tf.get_variable(name='w',
                                        shape=[item_feature_size, item_feature_size] ,
                                        initializer=tf.initializers.glorot_uniform())
                    if use_bias:
                        b = tf.get_variable(name='b',
                                            shape=item_feature_size,
                                            initializer=tf.constant(0.0,
                                            shape=[item_feature_size])) 
                    h = _matmul(adj_matrix, _matmul(x, w))
                    if use_bias: h += b
                    return activation(h)

            def _attention(x, query, name):
                values, _ = tf.nn.top_k(tf.transpose(x), k=5, sorted=True)
                softmax = tf.nn.softmax(tf.tensordot(query, values, axes=[2, 0]))
                query_r = tf.keras.backend.batch_dot(query, softmax, axes=[1, 1])
                query_r = tf.transpose(query_r, [0, 2, 1])
                values = tf.stack([tf.transpose(values)]*batch_size)
                return _cos_sim(query_r, values)
            
            self.gcn_1 = _gcn_conv(item_feature, self.adjusted_adjacency_matrix, name='gcn_layer_1')
            self.gcn_2 = _gcn_conv(self.gcn_1, self.adjusted_adjacency_matrix, name='gcn_layer_2')
            self.gcn_3 = _gcn_conv(self.gcn_2, self.adjusted_adjacency_matrix, name='gcn_layer_3')

            self.match_vector_1 = _attention(self.gcn_1, self.user_feature, name='attention_1')
            self.match_vector_2 = _attention(self.gcn_2, self.user_feature, name='attention_2')
            self.match_vector_3 = _attention(self.gcn_3, self.user_feature, name='attention_3')

            #Rank-and-Pooling Layer 
            selected = tf.stack([self.match_vector_1, self.match_vector_2, self.match_vector_3], axis=1)
            self.matching_score_vector = tf.layers.Flatten()(selected)
            self.output = tf.keras.layers.Dense(n_rating, use_bias=False)(self.matching_score_vector)

            # output
            self.probability = tf.nn.softmax(self.output)
            self.expectation = tf.matmul(self.probability, tf.reshape(self.rating, [-1, 1]))
            self.rmse = tf.sqrt(
                tf.reduce_sum(
                    tf.reshape(self.input_label_weight, shape=(-1, 1)) 
                    * tf.math.square(self.expectation 
                    - tf.to_float(self.input_rating))) 
                / tf.reduce_sum(self.input_label_weight))

            # loss
            self.loss = tf.reduce_sum(self.input_label_weight * tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.output, labels=self.input_label)) / tf.reduce_sum(self.input_label_weight)

            # optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate=self.input_learning_rate)
            self.op = optimizer.apply_gradients(optimizer.compute_gradients(self.loss))


class _Dataset(object):
    def __init__(self,
                 user_ids: np.ndarray,
                 item_ids: np.ndarray,
                 ratings: np.ndarray,
                 test_size: float,
                 item_features: Optional[Dict[Any, np.ndarray]] = None,
                 user_features: Optional[Dict[Any, np.ndarray]] = None) -> None:

        self.user2index = self._index_map(user_ids)
        self.item2index = self._index_map(item_ids)
        self.rating2index = self._index_map(ratings)
        self.user_features = self._sort_features(
            features=user_features, order_map=self.user2index) if user_features is not None else None
        self.item_features = self._sort_features(
            features=item_features, order_map=self.item2index) if item_features is not None else None
        self.ratings = ratings
        self.weights = self._calculate_weights(ratings)
        self.user_indices = self._to_index(self.user2index, user_ids)
        self.item_indices = self._to_index(self.item2index, item_ids)
        self.rating_indices = self._to_index(self.rating2index, ratings)
        self.train_indices = np.random.uniform(0., 1., size=len(user_ids)) > test_size
        self.th_value = 1.

    def train_adjacency_matrix(self):
        num_n = len(self.item2index)
        m = sp.lil_matrix((num_n, num_n), dtype=np.float32)
        for i in range(num_n):
            for j in range(num_n):
                if i==j: relevance = 0
                else: relevance = 1./np.linalg.norm(self.item_features[i]-self.item_features[j])
                if relevance > self.th_value:
                    m[i,j] = relevance
                else: m[i,j] = 0
        return m.tocsr()

    def train_data(self):
        idx = self.train_indices
        shuffle_idx = sklearn.utils.shuffle(list(range(int(np.sum(idx)))))
        one_hot = self._to_one_hot(self.rating_indices[idx][shuffle_idx])
        return self.user_indices[idx][shuffle_idx], self.item_indices[idx][shuffle_idx], one_hot, self.ratings[idx][shuffle_idx], self.weights[idx]

    def convert(self, user_ids: List, item_ids: List) -> Tuple[np.ndarray, np.ndarray]:
        def _to_indices(id2index, ids):
            return np.array(list(map(lambda x: id2index.get(x, -1), ids)))

        return _to_indices(self.user2index, user_ids), _to_indices(self.item2index, item_ids)

    def test_data(self):
        idx = ~self.train_indices
        one_hot = self._to_one_hot(self.rating_indices[idx])
        return self.user_indices[idx], self.item_indices[idx], one_hot, self.ratings[idx], self.weights[idx]

    def rating(self):
        return np.array(sorted(self.rating2index.keys()))

    def _to_one_hot(self, ratings):
        return np.eye(len(self.rating2index))[ratings]

    @staticmethod
    def _calculate_weights(ratings: np.ndarray) -> np.ndarray:
        rating_count = dict(Counter(ratings))
        rating2weight = {r: len(ratings) / len(rating_count) / c for r, c in rating_count.items()}
        # return np.array(list(map(rating2weight.get, ratings)))
        return np.ones_like(ratings)

    @staticmethod
    def _index_map(x: np.ndarray) -> Dict:
        u = sorted(np.unique(x))
        return dict(zip(u, range(len(u))))

    @staticmethod
    def _to_index(id2map, ids) -> np.ndarray:
        return np.array(list(map(id2map.get, ids)))

    @staticmethod
    def _sort_features(features: Dict[Any, np.ndarray], order_map: Dict) -> np.ndarray:
        new_order, _ = zip(*list(sorted(order_map.items(), key=lambda x: x[1])))
        sorted_features = np.array(list(map(features.get, new_order)))
        return sorted_features.astype(np.float32)

class MultiresolutionGraphAttentionNetworks(object):
    def __init__(self,
                 user_ids: np.ndarray,
                 item_ids: np.ndarray,
                 ratings: np.ndarray,
                 scope_name: str,
                 test_size: float,
                 batch_size: int,
                 epoch_size: int,
                 dropout_rate: float,
                 learning_rate: float,
                 lmda: float,
                 use_bias: bool = False,
                 save_directory_path: str = None,
                 user_features: Optional[Dict[Any, np.ndarray]] = None,
                 item_features: Optional[Dict[Any, np.ndarray]] = None) -> None:
        self.session = tf.Session()
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = ratings
        self.user_features = user_features
        self.item_features = item_features
        self.test_size = test_size
        self.lmda = lmda
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.scope_name = scope_name
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.use_bias = use_bias
        self.save_directory_path = save_directory_path
        self.dataset = _Dataset(
            self.user_ids, self.item_ids, self.ratings, self.test_size, user_features=self.user_features, item_features=self.item_features)
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

        test_user_indices, test_item_indices, test_labels, test_ratings, test_weights = self.dataset.test_data()
        report = []
        with self.session.as_default():
            self.session.run(tf.global_variables_initializer())
            dataset = tf.data.Dataset.from_tensor_slices(self.dataset.train_data())
            dataset = dataset.shuffle(buffer_size=self.batch_size)
            batch = dataset.batch(self.batch_size)
            iterator = batch.make_initializable_iterator()
            next_batch = iterator.get_next()
            adjacency_matrix = self.dataset.train_adjacency_matrix()
            print(adjacency_matrix)

            logger.info('start to optimize...')
            for i in range(self.epoch_size):
                self.session.run(iterator.initializer)
                while True:
                    try:
                        _user_indices, _item_indices, _labels, _ratings, _weights = self.session.run(next_batch)
                        _adjacency_matrix = self._eliminate(adjacency_matrix, _item_indices)

                        feed_dict = {
                            self.graph.input_learning_rate: early_stopping.learning_rate,
                            self.graph.input_dropout: self.dropout_rate,
                            self.graph.input_user: _user_indices.reshape((-1, 1)),
                            self.graph.input_item: _item_indices.reshape((-1, 1)),
                            self.graph.input_label: _labels,
                            self.graph.input_rating: _ratings.reshape((-1, 1)),
                            self.graph.input_label_weight: _weights
                        }
                        feed_dict.update({
                            self.graph.input_adjacency_matrix: _convert_sparse_matrix_to_sparse_tensor(_adjacency_matrix)
                        })
                        feed_dict.update({
                            self.graph.input_edge_size: _adjacency_matrix.count_nonzero()
                        })
                        _, train_loss, train_rmse = self.session.run([self.graph.op, self.graph.loss, self.graph.rmse],
                                                                     feed_dict=feed_dict)
                        report.append(f'train: epoch={i + 1}/{self.epoch_size}, loss={train_loss}, rmse={train_rmse}.')
                    except tf.errors.OutOfRangeError:
                        logger.info(report[-1])
                        feed_dict = {
                            self.graph.input_dropout: 0.0,
                            self.graph.input_user: test_user_indices.reshape((-1, 1)),
                            self.graph.input_item: test_item_indices.reshape((-1, 1)),
                            self.graph.input_label: test_labels,
                            self.graph.input_rating: test_ratings.reshape((-1, 1)),
                            self.graph.input_label_weight: test_weights
                        }
                        feed_dict.update({
                            self.graph.input_adjacency_matrix: _convert_sparse_matrix_to_sparse_tensor(adjacency_matrix)
                        })
                        feed_dict.update({
                            self.graph.input_edge_size: adjacency_matrix.count_nonzero()
                        })
                        test_loss, test_rmse = self.session.run([self.graph.loss, self.graph.rmse], feed_dict=feed_dict)
                        report.append(f'test: epoch={i + 1}/{self.epoch_size}, loss={test_loss}, rmse={test_rmse}.')
                        logger.info(report[-1])
                        break

                if early_stopping.does_stop(test_rmse, self.session):
                    break
        return report

    def predict(self):
        if self.graph is None:
            RuntimeError('Please call fit first.')

        adjacency_matrix = self.dataset.train_adjacency_matrix()
        user_indices, item_indices = self.dataset.convert(user_ids, item_ids)
        valid_indices = np.logical_and(user_indices != -1, item_indices != -1)
        feed_dict = {
            self.graph.input_dropout: 0.0,
            self.graph.input_user: user_indices[valid_indices],
            self.graph.input_item: item_indices[valid_indices],
        }
        feed_dict.update({
            self.graph.input_adjacency_matrix: _convert_sparse_matrix_to_sparse_tensor(adjacency_matrix)
        })
        feed_dict.update({
            self.graph.input_edge_size: adjacency_matrix.count_nonzero()
        })

        with self.session.as_default():
            valid_predictions = self.session.run(self.graph.expectation, feed_dict=feed_dict)
        valid_predictions = valid_predictions.flatten()
        valid_predictions = np.clip(valid_predictions, self.dataset.rating()[0], self.dataset.rating()[-1])
        predictions = np.ones(len(user_ids), dtype=np.float) * np.mean(valid_predictions)
        predictions[valid_indices] = valid_predictions
        return predictions
    
    @staticmethod
    def _eliminate(matrix: sp.csr_matrix, item_indices):
        matrix = matrix.copy().tolil()
        for ind in item_indices:
            matrix[[ind]*len(item_indices), item_indices] = 0
        matrix = matrix.tocsr()
        matrix.eliminate_zeros()
        return matrix

    def _make_graph(self) -> MultiresolutionGraphAttentionNetworksGraph:
        return MultiresolutionGraphAttentionNetworksGraph(
            n_rating=len(self.dataset.rating2index),
            n_user=len(self.dataset.user2index),
            n_item=len(self.dataset.item2index),
            batch_size=self.batch_size,
            rating=self.dataset.rating(),
            use_bias=self.use_bias,
            scope_name=self.scope_name,
            user_feature=self.dataset.user_features,
            item_feature=self.dataset.item_features,
            lmda=self.lmda)

    def save(self, file_path: str) -> None:
        redshells.model.utils.save_tf_session(self, self.session, file_path)

    @staticmethod
    def load(file_path: str) -> 'MultiresolutionGraphAttentionNetworks':
        session = tf.Session()
        model = redshells.model.utils.load_tf_session(
            MultiresolutionGraphAttentionNetworks, session, file_path,
            MultiresolutionGraphAttentionNetworks._make_graph)  
        return model

