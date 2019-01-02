import os
from io import StringIO
from logging import getLogger
from typing import List

import luigi
import numpy as np
import pandas as pd
import sklearn
from tqdm import tqdm

import gokart
import redshells.data
import redshells.train
import tensorflow as tf

logger = getLogger(__name__)


class PreprocessNetflixData(gokart.TaskOnKart):
    task_namespace = 'examples'
    text_data_file_path = luigi.Parameter()

    def requires(self):
        return redshells.data.LoadExistingFile(file_path=self.text_data_file_path)

    def output(self):
        return self.make_target('netflix/preprocess.pkl')

    def run(self):
        lines = self.load()  # type: List[str]

        item_id = ''
        data = []
        for line in lines:
            xs = line.split(':')
            if len(xs) == 2:
                item_id = xs[0]
                continue
            xs = line.split(',')
            xs.append(item_id)
            data.append(xs)

        df = pd.DataFrame(data, columns=['user_id', 'rating', 'date', 'item_id'])
        df['user_id'] = df['user_id'].astype(int)
        df['item_id'] = df['item_id'].astype(int)
        df['rating'] = df['rating'].astype(float)
        df = df.drop('date', axis=1)
        self.dump(df)


class FilterNetflixData(gokart.TaskOnKart):
    task_namespace = 'examples'
    text_data_file_paths = luigi.ListParameter()  # type: List[str]
    data_size_rate = luigi.FloatParameter()

    def requires(self):
        return [PreprocessNetflixData(text_data_file_path=path) for path in self.text_data_file_paths]

    def output(self):
        return self.make_target('netflix/merged_data.pkl')

    def run(self):
        df = self.load_data_frame()
        df = sklearn.utils.shuffle(df)
        df['service_id'] = 0
        self.dump(df.head(n=int(df.shape[0] * self.data_size_rate)))


class MatrixFactorizationExample(gokart.TaskOnKart):
    task_namespace = 'examples'
    text_data_file_paths = luigi.ListParameter(
        default=[f'./resources/netflix/combined_data_{i}.txt' for i in range(1, 5)])  # type: List[str]
    data_size_rate = luigi.FloatParameter()

    def requires(self):
        data_task = FilterNetflixData(
            text_data_file_paths=self.text_data_file_paths, data_size_rate=self.data_size_rate)
        data_task = redshells.data.data_frame_utils.SplitTrainTestData(
            data_task=data_task,
            test_size_rate=0.1,
            train_output_file_path='netflix/train_data.pkl',
            test_output_file_path='netflix/test_data.pkl')
        train_data_task = redshells.data.LoadDataOfTask(data_task=data_task, target_name='train')
        test_data_task = redshells.data.LoadDataOfTask(data_task=data_task, target_name='test')
        validation_task = redshells.train.TrainMatrixFactorization(
            train_data_task=train_data_task,
            user_column_name='user_id',
            item_column_name='item_id',
            service_column_name='service_id',
            rating_column_name='rating',
            model_kwargs=dict(
                n_latent_factors=20,
                learning_rate=1e-3,
                reg_item=1e-5,
                reg_user=1e-5,
                batch_size=2**16,
                epoch_size=100,
                test_size=0.1,
                scope_name='MatrixFactorizationExample',
                save_directory_path=os.path.join(self.local_temporary_directory, 'matrix_factorization'),
            ),
            output_file_path='netflix/model.zip')

        return dict(model=validation_task, test_data=test_data_task)

    def output(self):
        return self.make_target('netflix/example_results.txt')

    def run(self):
        tf.reset_default_graph()
        model = self.load('model')  # type: redshells.model.MatrixFactorization
        test_data = self.load_data_frame('test_data')

        predictions = model.predict(
            user_ids=test_data['user_id'], item_ids=test_data['item_id'], service_ids=test_data['service_id'])
        valid_indices = np.where(~np.isnan(predictions))[0]

        error = np.sqrt(
            sklearn.metrics.mean_squared_error(predictions[valid_indices], test_data['rating'].values[valid_indices]))

        logger.info(f'error={error}')
        self.dump(error)


if __name__ == '__main__':
    # Please download Netflix data from https://www.kaggle.com/netflix-inc/netflix-prize-data, and copy combined_data_*.txt to resources/netflix/.
    luigi.configuration.add_config_path('./config/example.ini')
    luigi.run([
        'examples.MatrixFactorizationExample',
        '--data-size-rate',
        '1.0',
        '--local-scheduler',
    ])
