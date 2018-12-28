import tensorflow as tf
import numpy as np


class FactorizationMachineGraph(object):
    def __init__(self, feature_size: int, embedding_size: int, l2_weight=1e-5, model: str = 'regression') -> None:
        self.x = tf.placeholder(dtype=np.float32, shape=[None, feature_size], name='input_features')
        self.target = tf.placeholder(dtype=np.float, shape=[None], name='target_values')

        regularizer = tf.contrib.layers.l2_regularizer(l2_weight)
        self.bias = tf.get_variable(name='bias', shape=[1], trainable=True, regularizer=regularizer)
        self.w = tf.get_variable(name='w', shape=[feature_size, 1], trainable=True, regularizer=regularizer)
        self.v = tf.get_variable(
            name='v', shape=[feature_size, embedding_size], trainable=True, regularizer=regularizer)

        self.xw = tf.matmul(self.x, self.w, name='xw')
        self.first_order = tf.reduce_sum(self.xw, axis=1, name='1st_order')
        self.xv2 = tf.pow(tf.matmul(self.x, self.v), 2, name='xv2')
        self.x2v2 = tf.matmul(tf.pow(self.x, 2), tf.pow(self.v, 2), name='x2v2')
        self.second_order = tf.reduce_sum(tf.subtract(self.xv2, self.x2v2), axis=1, name='2nd_order')
        self.y = tf.add(self.bias, tf.add(self.first_order, self.second_order), name='y')

        # variable_l2_loss = [tf.nn.l2_loss(self.bias), tf.nn.l2_loss(self.w), tf.nn.l2_loss(self.v)]
        self.regularization = tf.losses.get_regularization_losses()
        if model == 'regression':
            self.loss = tf.add_n([tf.losses.mean_squared_error(self.target, self.y)] + self.regularization, name='loss')
        elif model == 'binary_classification':
            self.loss = tf.add_n([tf.losses.hinge_loss(self.target, self.y)] + self.regularization, name='loss')
        else:
            raise ValueError(f'"{model}" is not supported. Please use "regression" or "binary_classification"')


class FactorizationMachine(object):
    def __init__(self,
                 embedding_size: int,
                 l2_weight,
                 model: str,
                 batch_size: int,
                 epoch_size: int,
                 session: tf.Session = None) -> None:
        self.session = session or tf.Session()
        self.embedding_size = embedding_size
        self.l2_weight = l2_weight
        self.model = model
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.graph = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.graph = self.graph or FactorizationMachineGraph(
            embedding_size=self.embedding_size, feature_size=x.shape[1], l2_weight=self.l2_weight, model=self.model)

        with self.session.as_default():
            self.session.run(tf.global_variables_initializer())

            for epoch in range(self.epoch_size):
                pass
