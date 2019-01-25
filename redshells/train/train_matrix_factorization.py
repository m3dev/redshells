from typing import Any
from typing import Dict

import luigi
import sklearn
import tensorflow as tf

import gokart
from redshells.model import MatrixFactorization


class TrainMatrixFactorization(gokart.TaskOnKart):
    task_namespace = 'redshells'
    train_data_task = gokart.TaskInstanceParameter(
        description=
        'A task outputs a pd.DataFrame with columns={`user_column_name`, `item_column_name`, `service_column_name`, `target_column_name`}.'
    )
    user_column_name = luigi.Parameter(default='user', description='The column name of user id.')  # type: str
    item_column_name = luigi.Parameter(default='item', description='The column name of item id')  # type: str
    service_column_name = luigi.Parameter(default='service', description='The column name of service id.')  # type: str
    rating_column_name = luigi.Parameter(
        default='rating', description='The target column name to predict.')  # type: str
    model_kwargs = luigi.DictParameter(default=dict(), description='Arguments of the model.')  # type: Dict[str, Any]
    max_data_size = luigi.IntParameter(default=50000000)
    output_file_path = luigi.Parameter(default='model/matrix_factorization.zip')  # type: str

    def requires(self):
        return self.train_data_task

    def output(self):
        return self.make_model_target(
            self.output_file_path, save_function=MatrixFactorization.save, load_function=MatrixFactorization.load)

    def run(self):
        tf.reset_default_graph()
        df = self.load_data_frame(required_columns={
            self.user_column_name, self.item_column_name, self.service_column_name, self.rating_column_name
        })

        df.drop_duplicates(subset=[self.user_column_name, self.item_column_name], inplace=True)
        df = sklearn.utils.shuffle(df)
        df = df.head(n=self.max_data_size)

        user_ids = df[self.user_column_name]
        item_ids = df[self.item_column_name]
        service_ids = df[self.service_column_name]
        ratings = df[self.rating_column_name]
        model = MatrixFactorization(**self.model_kwargs)
        model.fit(user_ids=user_ids, item_ids=item_ids, service_ids=service_ids, ratings=ratings)
        self.dump(model)
