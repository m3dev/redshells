from typing import Any
from typing import Dict

import luigi
import sklearn

import gokart
from sklearn.model_selection import train_test_split

import redshells
import redshells.train.utils


class _BinaryClassificationModelTask(gokart.TaskOnKart):
    train_data_task = gokart.TaskInstanceParameter(
        description='A task outputs a pd.DataFrame with columns={`target_column_name`}.')
    target_column_name = luigi.Parameter(default='category', description='Category column names.')  # type: str
    model_name = luigi.Parameter(
        description='A model name which has "fit" interface, and must be registered by "register_prediction_model".'
    )  # type: str
    model_kwargs = luigi.DictParameter(
        default=dict(), description='Arguments of the model which are created with model_name.')  # type: Dict[str, Any]

    def requires(self):
        return self.train_data_task

    def output(self):
        return self.make_target(self.output_file_path)

    def create_model(self):
        return redshells.factory.create_prediction_model(self.model_name, **self.model_kwargs)

    def create_train_data(self):
        data = self.load_data_frame(required_columns={self.target_column_name})

        data = sklearn.utils.shuffle(data)
        y = data[self.target_column_name].values

        data.drop(self.target_column_name, axis=1, inplace=True)
        x = redshells.train.utils.to_numpy(data)
        return x, y


class TrainBinaryClassificationModel(_BinaryClassificationModelTask):
    """
    Train classification model. Please see `_BinaryClassificationModelTask` for more detail and required parameters.
    """
    task_namespace = 'redshells'
    output_file_path = luigi.Parameter(default='model/binary_classification_model.pkl')  # type: str

    def run(self):
        redshells.train.utils.fit_model(self)


class OptimizeBinaryClassificationModel(_BinaryClassificationModelTask):
    """
    Optimize classification model. Please see `_BinaryClassificationModelTask` for more detail and required parameters.
    """
    task_namespace = 'redshells'
    test_size = luigi.FloatParameter()  # type: float
    optuna_param_name = luigi.Parameter(description='The key of "redshells.factory.get_optuna_param".')
    output_file_path = luigi.Parameter(default='model/binary_classification_model.pkl')  # type: str

    def run(self):
        redshells.train.utils.optimize_model(self, param_name=self.optuna_param_name, test_size=self.test_size, binary=True)


class ValidateBinaryClassificationModel(_BinaryClassificationModelTask):
    """
    Validate classification model. Please see `_BinaryClassificationModelTask` for more detail and required parameters.
    """
    task_namespace = 'redshells'
    cross_validation_size = luigi.IntParameter()  # type: int
    output_file_path = luigi.Parameter(default='model/binary_classification_model.txt')  # type: str

    def run(self):
        redshells.train.utils.validate_model(self, cv=self.cross_validation_size)
