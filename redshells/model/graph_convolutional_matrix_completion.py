from logging import getLogger

import numpy as np
import sklearn
import tensorflow as tf
from typing import List, Any, Dict

import redshells
from redshells.model.early_stopping import EarlyStopping

logger = getLogger(__name__)


class GraphConvolutionalMatrixCompletionGraph(object):
    def __init__(self,
                 n_ratings: int,
                 n_users: int,
                 n_items: int,
                 convolution_layer_size: int,
                 dense_layer_size: int,
                 scope_name: str = 'GraphConvolutionalMatrixCompletionGraph') -> None:
        # placeholder
        self.input_learning_rate = tf.placeholder(dtype=np.float32, name='learning_rate')

        # (rating, user) -> items
        self.input_user2items = [[
            tf.placeholder(dtype=np.int32, shape=[None], name=f'user2items_{r}_{u}') for u in range(n_users)
        ] for r in range(n_ratings)]
        logger.info('done input_user2items.')
        # (rating, item) -> users
        self.input_item2users = [[
            tf.placeholder(dtype=np.int32, shape=[None], name=f'item2users_{r}_{i}') for i in range(n_items)
        ] for r in range(n_ratings)]
        logger.info('done input_item2users.')

        # (rating, user) -> the number of items
        self.input_user2count = [[tf.placeholder(dtype=np.float32, name=f'user2count_{r}_{u}') for u in range(n_users)]
                                 for r in range(n_ratings)]
        logger.info('done input_user2count.')
        # (rating, item) -> the number of users
        self.input_item2count = [[tf.placeholder(dtype=np.float32, name=f'item2count_{r}_{i}') for i in range(n_items)]
                                 for r in range(n_ratings)]
        logger.info('done input_item2count.')

        # (rating, user, item)
        self.input_targets = tf.placeholder(dtype=np.int32, shape=[None, 3], name='targets')
        logger.info('done input_targets.')

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            # (rating, user) -> latent variables
            self.user_w_r = [
                tf.keras.layers.Embedding(
                    input_dim=n_users,
                    output_dim=convolution_layer_size,
                    embeddings_initializer=tf.random_normal_initializer(0, 1.0),
                    name='user_w_r') for _ in range(n_ratings)
            ]
            logger.info('done user_w_r.')

            # (rating, item) -> latent variables
            self.item_w_r = [
                tf.keras.layers.Embedding(
                    input_dim=n_items,
                    output_dim=convolution_layer_size,
                    embeddings_initializer=tf.random_normal_initializer(0, 1.0),
                    name='item_w_r') for _ in range(n_ratings)
            ]
            logger.info('done item_w_r.')

            def hidden_layer(input_user2count, input_user2items, item_w_r):
                return [[tf.reduce_sum(item_w(items), axis=0) / count for count, items in zip(user2count, user2items)]
                        for user2count, user2items, item_w in zip(input_user2count, input_user2items, item_w_r)]

            def replace_nan(w):
                return tf.reshape(
                    tf.where(tf.is_nan(w), tf.zeros_like(w), w), shape=(-1, convolution_layer_size * n_ratings))

            # user -> user hidden layer vectors
            self.user_h = tf.nn.relu(
                replace_nan(
                    tf.stack(
                        tf.concat(hidden_layer(self.input_user2count, self.input_user2items, self.item_w_r), axis=1))))
            logger.info('done user_h.')
            # item -> item hidden layer vectors
            self.item_h = tf.nn.relu(
                replace_nan(
                    tf.stack(
                        tf.concat(hidden_layer(self.input_item2count, self.input_item2users, self.user_w_r), axis=1))))
            logger.info('done item_h.')

            # h -> z
            self.encoder_dense = tf.keras.layers.Dense(units=dense_layer_size, use_bias=False)
            self.user_z = self.encoder_dense(self.user_h)
            logger.info('done user_z.')
            self.item_z = self.encoder_dense(self.item_h)
            logger.info('done item_z.')

            # decoder
            self.decoder_dense = [
                tf.keras.layers.Dense(units=dense_layer_size, use_bias=False) for _ in range(n_ratings)
            ]
            logger.info('done decoder_dense.')
            self.logits = tf.stack(
                [tf.matmul(self.user_z, dense(self.item_z), transpose_b=True) for dense in self.decoder_dense])
            logger.info('done logits.')

        # probability
        self.probability = tf.nn.softmax(self.logits, axis=0)
        logger.info('done probability.')

        # loss
        self.loss = -tf.reduce_mean(tf.log(tf.gather_nd(self.probability, self.input_targets)))
        logger.info('done loss.')

        # optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.input_learning_rate)
        self.op = optimizer.apply_gradients(optimizer.compute_gradients(self.loss))


class GraphConvolutionalMatrixCompletion(object):
    def __init__(self,
                 convolution_layer_size: int,
                 dense_layer_size: int,
                 learning_rate: float,
                 batch_size: int,
                 epoch_size: int,
                 test_size: float,
                 scope_name: str,
                 try_count: int = 3,
                 decay_speed: float = 10.0,
                 save_directory_path: str = None,
                 n_items: int = None,
                 n_users: int = None,
                 n_ratings: int = None,
                 user2index=None,
                 item2index=None,
                 rating2index=None) -> None:
        self.convolution_layer_size = convolution_layer_size
        self.dense_layer_size = dense_layer_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.test_size = test_size
        self.scope_name = scope_name
        self.try_count = try_count
        self.save_directory_path = save_directory_path
        self.decay_speed = decay_speed
        self.n_items = n_items
        self.n_users = n_users
        self.n_ratings = n_ratings
        self.user2index = user2index
        self.item2index = item2index
        self.rating2index = rating2index
        self.session = tf.Session()
        self.graph = None

    def fit(self, user_ids: List[Any], item_ids: List[Any], ratings: List[int]) -> None:
        logger.info(f'data size={len(user_ids)}.')
        if self.graph is None:
            logger.info('making graph...')
            self.n_users = len(set(user_ids))
            self.n_items = len(set(item_ids))
            self.n_ratings = len(set(ratings))
            logger.info('done size.')
            self.user2index = dict(zip(np.unique(user_ids), range(self.n_users)))
            self.item2index = dict(zip(np.unique(item_ids), range(self.n_items)))
            self.rating2index = dict(zip(np.unique(ratings), range(self.n_ratings)))
            logger.info('done index.')
            self.graph = self._make_graph()
            logger.info('done making graph')

        logger.info('making graph structure...')
        rating_indices = self._convert(ratings, self.rating2index)
        user_indices = self._convert(user_ids, self.user2index)
        item_indices = self._convert(item_ids, self.item2index)
        targets = np.array(list(zip(rating_indices, user_indices, item_indices)))
        train_targets, test_targets = sklearn.model_selection.train_test_split(targets, test_size=self.test_size)

        rating_user2items = [[[] for _ in range(self.n_users)] for _ in range(self.n_ratings)]
        rating_item2users = [[[] for _ in range(self.n_items)] for _ in range(self.n_ratings)]
        for rating, user, item in train_targets:
            rating_user2items[rating][user].append(item)
            rating_item2users[rating][item].append(user)
        rating_user2count = [[len(items) for items in user2items] for user2items in rating_user2items]
        rating_item2count = [[len(users) for users in item2users] for item2users in rating_item2users]

        feed_dict = dict()
        feed_dict.update({
            self.graph.input_user2items[r][u]: rating_user2items[r][u]
            for r in range(self.n_ratings) for u in range(self.n_users)
        })
        feed_dict.update({
            self.graph.input_item2users[r][i]: rating_item2users[r][i]
            for r in range(self.n_ratings) for i in range(self.n_items)
        })
        feed_dict.update({
            self.graph.input_user2count[r][u]: rating_user2count[r][u]
            for r in range(self.n_ratings) for u in range(self.n_users)
        })
        feed_dict.update({
            self.graph.input_item2count[r][i]: rating_item2count[r][i]
            for r in range(self.n_ratings) for i in range(self.n_items)
        })
        test_feed_dict = feed_dict.copy()
        test_feed_dict[self.graph.input_targets] = test_targets
        logger.info('done making graph structure')

        early_stopping = EarlyStopping(save_directory=self.save_directory_path, learning_rate=self.learning_rate)

        with self.session.as_default():
            logger.info('initializing variables...')
            self.session.run(tf.global_variables_initializer())
            dataset = tf.data.Dataset.from_tensor_slices(train_targets)
            dataset = dataset.batch(self.batch_size)
            iterator = dataset.make_initializable_iterator()
            next_batch = iterator.get_next()

            logger.info('start to optimize...')
            test_loss = self.session.run(self.graph.loss, feed_dict=test_feed_dict)
            logger.info(f'test: epoch=0/{self.epoch_size}, loss={test_loss}.')

            for i in range(self.epoch_size):
                self.session.run(iterator.initializer)
                train_loss = None
                while True:
                    try:
                        feed_dict[self.graph.input_targets] = self.session.run(next_batch)
                        feed_dict[self.graph.input_learning_rate] = early_stopping.learning_rate
                        _, train_loss = self.session.run([self.graph.op, self.graph.loss], feed_dict=feed_dict)
                    except tf.errors.OutOfRangeError:
                        logger.info(f'train: epoch={i + 1}/{self.epoch_size}, loss={train_loss}.')
                        test_loss = self.session.run(self.graph.loss, feed_dict=test_feed_dict)
                        logger.info(f'test: epoch={i + 1}/{self.epoch_size}, loss={test_loss}.')
                        break

                # check early stopping
                if early_stopping.does_stop(test_loss, self.session):
                    break

    def _convert(self, ids: List[Any], id2index: Dict[Any, int]) -> np.ndarray:
        return np.array([id2index.get(i, -1) for i in ids])

    def _make_graph(self) -> GraphConvolutionalMatrixCompletionGraph:
        return GraphConvolutionalMatrixCompletionGraph(
            n_items=self.n_items,
            n_users=self.n_users,
            n_ratings=self.n_ratings,
            convolution_layer_size=self.convolution_layer_size,
            dense_layer_size=self.dense_layer_size,
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


def main():
    np.random.seed(12)
    tf.random.set_random_seed(32)
    n_ratings = 5
    n_users = 10
    n_items = 13
    hidden_size = 3
    data_size = 11
    rating_edges = [
        list(
            zip(
                np.random.randint(low=0, high=n_users, size=data_size),
                np.random.randint(low=0, high=n_items, size=data_size))) for _ in range(n_ratings)
    ]
    rating_user2items = [[[] for _ in range(n_users)] for _ in range(n_ratings)]
    rating_item2users = [[[] for _ in range(n_items)] for _ in range(n_ratings)]

    for edges, user2items, item2users in zip(rating_edges, rating_user2items, rating_item2users):
        for user, item in edges:
            user2items[user].append(item)
            item2users[item].append(user)

    rating_user2count = [[len(items) for items in user2items] for user2items in rating_user2items]
    rating_item2count = [[len(users) for users in item2users] for item2users in rating_item2users]

    targets = [
        np.array([rating, user_item[0], user_item[1]]) for rating, user_item_pairs in enumerate(rating_edges)
        for user_item in user_item_pairs
    ]

    graph = GraphConvolutionalMatrixCompletionGraph(
        n_ratings=n_ratings,
        n_users=n_users,
        n_items=n_items,
        convolution_layer_size=hidden_size,
        dense_layer_size=hidden_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = {
            graph.input_user2items[r][u]: rating_user2items[r][u]
            for r in range(n_ratings) for u in range(n_users)
        }
        feed_dict.update(
            {graph.input_item2users[r][i]: rating_item2users[r][i]
             for r in range(n_ratings) for i in range(n_items)})
        feed_dict.update(
            {graph.input_user2count[r][u]: rating_user2count[r][u]
             for r in range(n_ratings) for u in range(n_users)})
        feed_dict.update(
            {graph.input_item2count[r][i]: rating_item2count[r][i]
             for r in range(n_ratings) for i in range(n_items)})
        feed_dict[graph.input_targets] = targets

        results = sess.run(graph.loss, feed_dict=feed_dict)
        print(results)
        # print(results.shape)


if __name__ == '__main__':
    main()
