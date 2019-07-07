from logging import getLogger

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

logger = getLogger(__name__)


class FeatureAggregationSimilarityDataset(tf.keras.utils.Sequence):
    def __init__(
            self,
            x_item_indices: np.ndarray,
            y_item_indices: np.ndarray,
            x_item_features: np.ndarray,
            y_item_features: np.ndarray,
            scores: np.ndarray,
            batch_size: int = 2**10) -> None:
        self.x_item_indices = x_item_indices
        self.y_item_indices = y_item_indices
        self.x_item_features = x_item_features
        self.y_item_features = y_item_features
        self.scores = scores
        self.batch_size = batch_size

    def get(self, size: int, batch_size: int):
        idx = np.arange(self.scores.shape[0])
        np.random.shuffle(idx)
        idx = idx[:size]
        return FeatureAggregationSimilarityDataset(
            x_item_indices=self.x_item_indices.copy()[idx],
            y_item_indices=self.y_item_indices.copy()[idx],
            x_item_features=self.x_item_features.copy()[idx],
            y_item_features=self.y_item_features.copy()[idx],
            scores=self.scores.copy()[idx],
            batch_size=batch_size)

    def train_test_split(self, test_size_rate: float):
        data_size = self.scores.shape[0]
        test_size = int(data_size * test_size_rate)
        train = FeatureAggregationSimilarityDataset(
            x_item_indices=self.x_item_indices.copy()[test_size:],
            y_item_indices=self.y_item_indices.copy()[test_size:],
            x_item_features=self.x_item_features.copy()[test_size:],
            y_item_features=self.y_item_features.copy()[test_size:],
            scores=self.scores.copy()[test_size:],
            batch_size=self.batch_size)
        test = FeatureAggregationSimilarityDataset(
            x_item_indices=self.x_item_indices.copy()[:test_size],
            y_item_indices=self.y_item_indices.copy()[:test_size],
            x_item_features=self.x_item_features.copy()[:test_size],
            y_item_features=self.y_item_features.copy()[:test_size],
            scores=self.scores.copy()[:test_size],
            batch_size=self.batch_size)
        return train, test



    def __getitem__(self, index):
        start_idx = index * self.batch_size
        last_idx = start_idx + self.batch_size
        x_item_indices = self.x_item_indices[start_idx:last_idx]
        y_item_indices = self.y_item_indices[start_idx:last_idx]
        x_item_features = self.x_item_features[start_idx:last_idx]
        y_item_features = self.y_item_features[start_idx:last_idx]
        scores = self.scores[start_idx:last_idx]
        return (x_item_indices, y_item_indices, x_item_features, y_item_features), scores

    def __len__(self):
        return int(np.floor(len(self.x_item_indices) / self.batch_size))


class Average(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Average, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Average, self).build(input_shape)

    def call(self, inputs, mask, **kwargs):
        mask = tf.cast(mask, tf.float32)
        return tf.div_no_nan(tf.keras.backend.batch_dot(inputs, mask, axes=1), tf.reduce_sum(mask, axis=1, keepdims=True))

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        base_config = super(Average, self).get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Clip(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Clip, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Clip, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.keras.backend.clip(inputs, -1.0, 1.0)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        base_config = super(Clip, self).get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class FeatureAggregationSimilarityGraph(object):
    def __init__(self,
                 feature_size: int,
                 embedding_size: int,
                 item_size: int,
                 max_feature_index: int,
                 embeddings_initializer=None,
                 bias_embeddings_initializer=None,
                 embeddings_regularizer=None):
        embeddings_initializer = embeddings_initializer or tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.005)
        embeddings_regularizer = embeddings_regularizer or tf.keras.regularizers.l2(0.0001)
        bias_embeddings_initializer = bias_embeddings_initializer or tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.005)
        self.input_x_index = layers.Input(shape=(1, ), name='input_x_index')
        self.input_y_index = layers.Input(shape=(1, ), name='input_y_index')
        self.input_x_feature = layers.Input(shape=(feature_size, ), name='input_x_feature')
        self.input_y_feature = layers.Input(shape=(feature_size, ), name='input_y_feature')

        self.embedding = layers.Embedding(
            max_feature_index + 1,
            embedding_size,
            mask_zero=True,
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
        )
        self.bias_embedding = tf.keras.layers.Embedding(
            item_size + 1,
            1,
            mask_zero=True,
            embeddings_initializer=bias_embeddings_initializer,
        )

        self.embedding_x = self.average(self.embedding(self.input_x_feature))
        self.embedding_y = self.average(self.embedding(self.input_y_feature))
        self.bias_x = self.average(self.bias_embedding(self.input_x_index))
        self.bias_y = self.average(self.bias_embedding(self.input_y_index))

        self.inner_prod = tf.keras.layers.dot([self.embedding_x, self.embedding_y], axes=1, normalize=True)
        self.similarity = tf.keras.layers.add([self.inner_prod, self.bias_x, self.bias_y])
        self.similarity = self.clip(self.similarity)

    @staticmethod
    def average(x):
        return Average()(x)

    @staticmethod
    def clip(x):
        return Clip()(x)


class FeatureAggregationSimilarityModel(object):
    def __init__(
            self,
            embedding_size: int,
            learning_rate: float,
            feature_size: int,
            item_size: int,
            max_feature_index: int,
    ) -> None:
        self.feature_size = feature_size
        graph = FeatureAggregationSimilarityGraph(
            feature_size=feature_size, embedding_size=embedding_size, item_size=item_size, max_feature_index=max_feature_index)
        self.model = tf.keras.models.Model(
            inputs=[graph.input_x_index, graph.input_y_index, graph.input_x_feature, graph.input_y_feature], outputs=graph.similarity)
        self.embeddings = tf.keras.models.Model(inputs=[graph.input_x_feature], outputs=graph.embedding_x)
        self.model.compile(optimizer=tf.train.AdamOptimizer(learning_rate), loss=tf.keras.losses.mse, metrics=[tf.keras.metrics.mse])

    def __getstate__(self):
        return self.feature_size, self.model.to_json(), self.model.get_weights(), self.embeddings.to_json(), self.embeddings.get_weights()

    def __setstate__(self, state):
        feature_size, json_config, weights, embedding_json_config, embedding_weights = state
        self.feature_size = feature_size
        self.model = tf.keras.models.model_from_json(json_config, custom_objects={'Clip': Clip, 'Average': Average})
        self.model.set_weights(weights)
        self.embeddings = tf.keras.models.model_from_json(embedding_json_config, custom_objects={'Clip': Clip, 'Average': Average})
        self.embeddings.set_weights(embedding_weights)

    def fit(self,
            dataset: FeatureAggregationSimilarityDataset,
            epoch_size: int,
            test_size_rate: float = 0.05,
            early_stopping_patience: int = 2):
        logger.info('prepare data...')
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=early_stopping_patience)]
        logger.info('start to fit...')
        data, validation_data = self._make_dataset(dataset=dataset, test_size_rate=test_size_rate)
        self.model.fit_generator(
            data, epochs=epoch_size, callbacks=callbacks, validation_data=validation_data)

    def calculate_similarity(self, x_item_indices, y_item_indices, x_item_features, y_item_features, batch_size=2**14):
        return self.model.predict(x=(x_item_indices, y_item_indices, x_item_features, y_item_features), batch_size=batch_size).reshape(-1)

    def calculate_embeddings(self, item_features, batch_size=2**14):
        return self.embeddings.predict(x=(item_features, ), batch_size=batch_size)

    def _make_dataset(self, dataset: FeatureAggregationSimilarityDataset, test_size_rate: float):
        data, validation_data = dataset.train_test_split(test_size_rate=test_size_rate)
        return data, validation_data
