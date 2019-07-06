from logging import getLogger

import luigi
import numpy as np

import gokart

from redshells.model import FeatureAggregationSimilarityModel
from redshells.model.feature_aggregation_similarity_model import FeatureAggregationSimilarityDataset

logger = getLogger(__name__)


class TrainFeatureAggregationSimilarityModel(gokart.TaskOnKart):
    dataset_task = gokart.TaskInstanceParameter(description='An instance of task which outputs `FeatureAggregationSimilarityDataset`.')
    embedding_size = luigi.IntParameter()  # type: int
    learning_rate = luigi.FloatParameter()  # type: float
    batch_size = luigi.IntParameter()  # type: int
    epoch_size = luigi.IntParameter()  # type: int
    test_size_rate = luigi.FloatParameter()  # type: float
    early_stopping_patience = luigi.IntParameter()  # type: int
    max_data_size = luigi.IntParameter()  # type: int
    output_file_path = luigi.Parameter(default='model/feature_aggregation)similarity_model.pkl')  # type: str

    def requires(self):
        return self.dataset_task

    def output(self):
        return self.make_target(self.output_file_path)

    def run(self):
        dataset = self.load()  # type: FeatureAggregationSimilarityDataset
        feature_size = dataset.x_item_features.shape[1]
        item_size = max(np.max(dataset.x_item_indices), np.max(dataset.y_item_indices))
        max_feature_index = max(np.max(dataset.x_item_features), np.max(dataset.y_item_features))

        logger.debug(f'embedding_size={self.embedding_size},'
                     f'learning_rate={self.learning_rate},'
                     f'feature_size={feature_size},'
                     f'item_size={item_size},'
                     f'max_feature_index={max_feature_index}')

        model = FeatureAggregationSimilarityModel(
            embedding_size=self.embedding_size,
            learning_rate=self.learning_rate,
            feature_size=feature_size,
            item_size=item_size,
            max_feature_index=max_feature_index)

        model.fit(
            dataset=dataset.get(size=self.max_data_size, batch_size=self.batch_size),
            epoch_size=self.epoch_size,
            test_size_rate=self.test_size_rate,
            early_stopping_patience=self.early_stopping_patience)

        self.dump(model)
