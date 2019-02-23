import os
from logging import getLogger
from typing import List

import luigi
import pandas as pd
import sklearn
import tensorflow as tf

import gokart
import redshells.data
import redshells.train
import numpy as np
logger = getLogger(__name__)


class PreprocessMLData(gokart.TaskOnKart):
    task_namespace = 'examples'
    text_data_file_path = luigi.Parameter()  # type: str

    def requires(self):
        return redshells.data.LoadExistingFile(
            file_path=os.path.join(self.workspace_directory, self.text_data_file_path))

    def output(self):
        return self.make_target('ml/preprocess_for_gc.pkl')

    def run(self):
        lines = self.load()  # type: List[str]
        data = [line.split() for line in lines]
        df = pd.DataFrame(data, columns=['user_id', 'item_id', 'rating', 'date'])
        df['user_id'] = df['user_id'].astype(int)
        df['item_id'] = df['item_id'].astype(int)
        df['rating'] = df['rating'].astype(int)
        df['service_id'] = 0
        df = df.drop('date', axis=1)
        self.dump(df)


class GraphConvolutionalMatrixCompletionExample(gokart.TaskOnKart):
    task_namespace = 'examples'
    text_data_file_path = luigi.Parameter(default='ml_100k/u.data.txt')  # type: str

    def _make_train_task(self,
                         train_data_task,
                         batch_size: int,
                         dropout_rate: float,
                         encoder_hidden_size: int,
                         encoder_size: int = 75,
                         normalization_type: str = 'left',
                         learning_rate: float = 1e-3):
        task = redshells.train.TrainGraphConvolutionalMatrixCompletion(
            # rerun=True,
            train_data_task=train_data_task,
            user_column_name='user_id',
            item_column_name='item_id',
            rating_column_name='rating',
            max_data_size=100000,
            model_kwargs=dict(
                encoder_hidden_size=encoder_hidden_size,
                encoder_size=encoder_size,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate,
                batch_size=2**batch_size,
                normalization_type=normalization_type,
                weight_sharing=True,
                epoch_size=40,
                test_size=0.2,
                scope_name='GraphConvolutionalMatrixCompletionExample',
                save_directory_path=os.path.join(self.local_temporary_directory,
                                                 'graph_convolutional_matrix_completion'),
            ),
            output_file_path=f'ml/model/model.zip')
        return task

    def requires(self):
        data_task = PreprocessMLData(text_data_file_path=self.text_data_file_path)
        data_task = redshells.data.data_frame_utils.SplitTrainTestData(
            data_task=data_task,
            test_size_rate=0.2,
            train_output_file_path='ml/train_data.pkl',
            test_output_file_path='ml/test_data.pkl')
        train_data_task = redshells.data.LoadDataOfTask(data_task=data_task, target_name='train')
        test_data_task = redshells.data.LoadDataOfTask(data_task=data_task, target_name='test')
        train_tasks = [
            self._make_train_task(
                train_data_task,
                batch_size=batch_size,
                dropout_rate=dropout_rate,
                encoder_hidden_size=encoder_hidden_size,
                encoder_size=encoder_size,
                normalization_type=normalization_type) for batch_size in [10] for dropout_rate in [0.7]
            for encoder_hidden_size in [500] for encoder_size in [75, 150]
            for normalization_type in ['left', 'right', 'symmetric']
        ]

        model = self._make_train_task(
            train_data_task,
            batch_size=10,
            dropout_rate=0.7,
            encoder_hidden_size=500,
            encoder_size=75,
            normalization_type='left',
            learning_rate=1e-3)
        return dict(optimize=train_tasks, test_data=test_data_task, model=model, train_data=train_data_task)

    def output(self):
        return self.make_target(f'ml/example_results.txt')

    def run(self):
        reports = [target['report'].load() for target in self.input()['optimize']]
        for report in reports:
            print('====================================================')
            print(report[0])
            print(report[-1])
        tf.reset_default_graph()
        model = self.load('model')['model']  # type: redshells.model.GraphConvolutionalMatrixCompletion
        test_data = self.load_data_frame('test_data')

        predictions = model.predict(user_ids=test_data['user_id'], item_ids=test_data['item_id'])
        error = np.sqrt(sklearn.metrics.mean_squared_error(predictions, test_data['rating'].values))

        logger.info(f'error={error}')
        # self.dump(error)


if __name__ == '__main__':
    # Please download ml100k data from http://files.grouplens.org/datasets/movielens/ml-100k/, and copy combined_data_*.txt to resources/ml_100k/.
    luigi.configuration.add_config_path('./config/example.ini')
    gokart.run([
        'examples.GraphConvolutionalMatrixCompletionExample',
        '--local-scheduler',
        '--text-data-file-path=ml_data/100k.txt',
        # '--tree-info-mode=all',
        # '--tree-info-output-path=sample_task_log.txt',
    ])
