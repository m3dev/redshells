from typing import Any
from typing import Dict

import luigi
import numpy as np
import sklearn

import gokart
import redshells
import redshells.train.utils
from logging import getLogger

logger = getLogger(__name__)


class _PairwiseSimilarityModelTask(gokart.TaskOnKart):
    item2embedding_task = gokart.TaskInstanceParameter(
        description='A task outputs a mapping from item to embedding. The output must have type=Dict[Any, np.ndarray].')
    similarity_data_task = gokart.TaskInstanceParameter(
        description=
        'A task outputs a pd.DataFrame with columns={`item0_column_name`, `item`_column_name`, `similarity_column_name`}. '
        '`similarity_column_name` must be binary data.')
    item0_column_name = luigi.Parameter()  # type: str
    item1_column_name = luigi.Parameter()  # type: str
    similarity_column_name = luigi.Parameter()  # type: str
    model_name = luigi.Parameter(
        default='XGBClassifier',
        description='A model name which has "fit" interface, and must be registered by "register_prediction_model".'
    )  # type: str
    model_kwargs = luigi.DictParameter(
        default=dict(), description='Arguments of the model which are created with model_name.')  # type: Dict[str, Any]
    output_file_path = luigi.Parameter(default='model/pairwise_similarity_model.pkl')  # type: str

    def requires(self):
        return dict(item2embedding=self.item2embedding_task, similarity_data=self.similarity_data_task)

    def output(self):
        return self.make_target(self.output_file_path)

    def create_model(self):
        return redshells.factory.create_prediction_model(self.model_name, **self.model_kwargs)

    def create_train_data(self):
        logger.info('loading input data...')
        item2embedding = self.load('item2embedding')  # type: Dict[Any, np.ndarray]
        similarity_data = self.load_data_frame(
            'similarity_data',
            required_columns={self.item0_column_name, self.item1_column_name, self.similarity_column_name})
        logger.info(f'similarity_data size={similarity_data.shape}')
        similarity_data = sklearn.utils.shuffle(similarity_data)
        logger.info('making features...')
        similarity_data[self.similarity_column_name] = similarity_data[self.similarity_column_name].astype(int)
        similarity_data = sklearn.utils.shuffle(similarity_data)
        similarity_data = similarity_data[similarity_data[self.item0_column_name].isin(item2embedding)]
        similarity_data = similarity_data[similarity_data[self.item1_column_name].isin(item2embedding)]
        x = np.array([
            np.multiply(item2embedding[i1], item2embedding[i2]) for i1, i2 in zip(
                similarity_data[self.item0_column_name].tolist(), similarity_data[self.item1_column_name].tolist())
        ])

        y = similarity_data[self.similarity_column_name].tolist()

        logger.info('done making train data.')
        logger.info(f'size of x={len(x)}, {len(x[0])}')
        return x, y


class TrainPairwiseSimilarityModel(_PairwiseSimilarityModelTask):
    """
    Train pairwise similarity models. Please see `_PairwiseSimilarityModelTask` for more details and required parameters.
    """
    task_namespace = 'redshells'
    output_file_path = luigi.Parameter(default='model/pairwise_similarity_model.pkl')  # type: str

    def run(self):
        redshells.train.utils.fit_model(self)


class ValidatePairwiseSimilarityModel(_PairwiseSimilarityModelTask):
    """
    Train pairwise similarity models. Please see `_PairwiseSimilarityModelTask` for more details and required parameters.
    """
    task_namespace = 'redshells'
    cross_validation_size = luigi.IntParameter()  # type: int
    output_file_path = luigi.Parameter(default='model/pairwise_similarity_model.pkl')  # type: str

    def run(self):
        redshells.train.utils.validate_model(self, cv=self.cross_validation_size)
