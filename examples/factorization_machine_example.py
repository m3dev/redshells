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


def _get_target_column() -> str:
    return 'label'


def _get_integer_columns() -> List[str]:
    return [f'int_feat_{i}' for i in range(13)]


def _get_categorical_columns() -> List[str]:
    return [f'cat_feat_{i}' for i in range(26)]


class SampleCriteo(gokart.TaskOnKart):
    task_namespace = 'examples'
    text_data_file_path = luigi.Parameter()  # type: str
    data_size_rate = luigi.FloatParameter()  # type: float

    def requires(self):
        return redshells.data.LoadExistingFile(file_path=self.text_data_file_path)

    def output(self):
        return self.make_target('criteo/data_samples.tsv')

    def run(self):
        logger.info('loading...')
        data = self.load()
        logger.info('sampling...')
        data = [data[i] for i in np.where(np.random.uniform(size=len(data)) < self.data_size_rate)[0]]
        columns = [_get_target_column()] + _get_integer_columns() + _get_categorical_columns()
        data.insert(0, '\t'.join(columns))
        logger.info('making data frame...')
        df = pd.read_csv(StringIO('\n'.join(data)), sep='\t')
        logger.info('dumping...')
        self.dump(df)


class PreprocessCriteo(gokart.TaskOnKart):
    data_task = gokart.TaskInstanceParameter()

    def requires(self):
        return self.data_task

    def output(self):
        return self.make_target('criteo/train_data.pkl')

    def run(self):
        logger.info('loading...')
        df = self.load_data_frame()

        logger.info('preprocess for integer columns...')
        for c in tqdm(_get_integer_columns()):
            values = df[c].copy()
            m = np.min([x for x in values[values.notnull()]])
            values[values.notnull()] += -m + 2
            values[values.isnull()] = 1
            df[c] = np.log(values)

        logger.info('preprocess for category columns...')
        for c in _get_categorical_columns():
            df[c] = df[c].astype('category')

        logger.info('dumping...')
        self.dump(df)


class SplitTrainTestData(gokart.TaskOnKart):
    task_namespace = 'examples'
    data_task = gokart.TaskInstanceParameter()
    test_size_rate = luigi.FloatParameter()

    def requires(self):
        return self.data_task

    def output(self):
        return dict(train=self.make_target('criteo/train_data.pkl'), test=self.make_target('criteo/test_data.pkl'))

    def run(self):
        data = self.load_data_frame()
        data = sklearn.utils.shuffle(data)
        train, test = sklearn.model_selection.train_test_split(data, test_size=self.test_size_rate)
        self.dump(train, 'train')
        self.dump(test, 'test')


class FactorizationMachineExample(gokart.TaskOnKart):
    task_namespace = 'examples'
    text_data_file_path = luigi.Parameter()
    data_size_rate = luigi.FloatParameter()

    def requires(self):
        redshells.factory.register_prediction_model('FactorizationMachine', redshells.model.FactorizationMachine)
        data_task = SampleCriteo(text_data_file_path=self.text_data_file_path, data_size_rate=self.data_size_rate)
        data_task = PreprocessCriteo(data_task=data_task)
        train_test_data = SplitTrainTestData(data_task=data_task, test_size_rate=0.1)
        train_data_task = redshells.data.LoadDataOfTask(data_task=train_test_data, target_name='train')
        test_data_task = redshells.data.LoadDataOfTask(data_task=train_test_data, target_name='test')
        validation_task = redshells.train.TrainFactorizationMachine(
            train_data_task=train_data_task,
            target_column_name='label',
            model_name='FactorizationMachine',
            model_kwargs=dict(
                embedding_size=10,
                l2_weight=1e-6,
                learning_rate=1e-4,
                batch_size=2**8,
                epoch_size=100,
                test_size=0.1,
                save_directory_path=os.path.join(self.local_temporary_directory, 'factorization_machine'),
                scope_name='FactorizationMachineExample'),
            output_file_path='criteo/validation.zip')
        return dict(model=validation_task, test_data=test_data_task)

    def output(self):
        return self.make_target('criteo/example_results.txt')

    def run(self):
        tf.reset_default_graph()
        model = self.load('model')  # type: redshells.model.FactorizationMachine
        test_data = self.load_data_frame('test_data')
        y = test_data['label'].copy()
        x = test_data.drop('label', axis=1)
        predict = model.predict(x)
        auc = redshells.model.utils.calculate_auc(y, predict)
        self.dump(f'auc={auc}')


if __name__ == '__main__':
    # Please download criteo data from https://www.kaggle.com/c/criteo-display-ad-challenge and put train.txt on ./resouces/criteo/train.txt.
    luigi.configuration.add_config_path('./config/example.ini')
    luigi.run([
        'examples.FactorizationMachineExample',
        '--text-data-file-path',
        './resources/criteo/train.txt',
        '--data-size-rate',
        '0.1',
        '--local-scheduler',
    ])
