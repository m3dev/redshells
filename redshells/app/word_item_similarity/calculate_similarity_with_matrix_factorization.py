import itertools
from logging import getLogger
from typing import List

import luigi
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

import gokart
import redshells

logger = getLogger(__name__)


class CalculateSimilarityWithMatrixFactorization(gokart.TaskOnKart):
    """Calculate similarity between items using latent factors which are calculated by Matrix Factorization.
    
    """
    task_namespace = 'redshells.word_item_similarity'
    target_item_task = gokart.TaskInstanceParameter(description='A task outputs item ids as type List.')
    matrix_factorization_task = gokart.TaskInstanceParameter(
        description='A task instance of `TrainMatrixFactorization`.')
    normalize = luigi.BoolParameter(description='Normalize item factors with l2 norm.')  # type: bool
    batch_size = luigi.IntParameter(default=1000, significant=False)
    output_file_path = luigi.Parameter(
        default='app/word_item_similarity/calculate_similarity_with_matrix_factorization.zip')  # type: str

    def requires(self):
        assert type(self.matrix_factorization_task) == redshells.train.TrainMatrixFactorization,\
            f'but actually {type(self.matrix_factorization_task)} is passed.'
        return dict(data=self.target_item_task, model=self.matrix_factorization_task)

    def output(self):
        return self.make_large_data_frame_target(self.output_file_path)

    def run(self):
        tf.reset_default_graph()
        data = self.load('data')  # type: List
        model = self.load('model')  # type: redshells.model.MatrixFactorization

        data = list(set(data))
        item_ids = model.get_valid_item_ids(data)
        factors = model.get_item_factors(item_ids, normalize=self.normalize)
        # Usually, ths size of item_ids is too large to calculate similarities at once. So I split data.
        split_size = factors.shape[0] // self.batch_size + 1
        factors_sets = np.array_split(factors, split_size)
        item_ids_sets = np.array_split(item_ids, split_size)

        def _calculate(x, y, x_ids, y_ids):
            if np.array_equal(x_ids, y_ids):
                indices = np.triu_indices(x_ids.shape[0], k=1)
            else:
                indices_ = np.indices([x_ids.shape[0], y_ids.shape[0]])
                indices = (indices_[0].flatten(), indices_[1].flatten())

            df = pd.DataFrame({
                'item_id_0': list(x_ids[indices[0]]),
                'item_id_1': list(y_ids[indices[1]]),
                'similarity': list(np.dot(x, y.T)[indices])
            })
            return df

        results = pd.concat([
            _calculate(factors_sets[i], factors_sets[j], item_ids_sets[i], item_ids_sets[j])
            for i, j in tqdm(list(itertools.combinations_with_replacement(range(split_size), 2)))
        ])
        self.dump(results)
