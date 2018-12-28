from typing import Any
from typing import Dict

import luigi
import sklearn

import gokart
import redshells
import redshells.train.utils


class _FactorizationMachineTask(gokart.TaskOnKart):
    train_data_task = gokart.TaskInstanceParameter(
        description='A task outputs a pd.DataFrame with columns={`target_column_name`}.')
    target_column_name = luigi.Parameter(default='category', description='Category column names.')  # type: str
    model_name = luigi.Parameter(
        default='XGBClassifier',
        description='A model name which has "fit" interface, and must be registered by "register_prediction_model".'
    )  # type: str
    model_kwargs = luigi.DictParameter(
        default=dict(), description='Arguments of the model which are created with model_name.')  # type: Dict[str, Any]

    def requires(self):
        return self.train_data_task

    def create_model(self):
        return redshells.factory.create_prediction_model(self.model_name, **self.model_kwargs)

    def create_train_data(self):
        data = self.load_data_frame(required_columns={self.target_column_name})
        data = sklearn.utils.shuffle(data)
        y = data[self.target_column_name].astype(int)

        x = data.drop(self.target_column_name, axis=1)
        return x, y


class TrainFactorizationMachine(_FactorizationMachineTask):
    """
    Train factorization machine. Please see `_FactorizationMachineTask` for more detail and required parameters.
    """
    task_namespace = 'redshells'
    output_file_path = luigi.Parameter(default='model/factorization_machine.pkl')  # type: str

    def output(self):
        return self.make_model_target(
            self.output_file_path,
            save_function=redshells.factory.get_prediction_model_type(self.model_name).save,
            load_function=redshells.factory.get_prediction_model_type(self.model_name).load)

    def run(self):
        redshells.train.utils.fit_model(self)


class ValidateFactorizationMachine(_FactorizationMachineTask):
    """
    Validate factorization machine with cross validation. Please see `_FactorizationMachineTask` for more detail and required parameters.
    """
    task_namespace = 'redshells'
    cross_validation_size = luigi.IntParameter()  # type: int
    output_file_path = luigi.Parameter(default='model/factorization_machine.txt')  # type: str

    def output(self):
        return self.make_target(self.output_file_path)

    def run(self):
        redshells.train.utils.validate_model(self, cv=self.cross_validation_size)
