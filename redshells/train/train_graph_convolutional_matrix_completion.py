from typing import Any
from typing import Dict

import luigi
import sklearn
import tensorflow as tf

import gokart
from redshells.model.graph_convolutional_matrix_completion import GraphConvolutionalMatrixCompletion


class TrainGraphConvolutionalMatrixCompletion(gokart.TaskOnKart):
    task_namespace = 'redshells'
    train_data_task = gokart.TaskInstanceParameter(
        description=
        'A task outputs a pd.DataFrame with columns={`user_column_name`, `item_column_name`, `target_column_name`}.')
    user_column_name = luigi.Parameter(default='user', description='The column name of user id.')  # type: str
    item_column_name = luigi.Parameter(default='item', description='The column name of item id')  # type: str
    rating_column_name = luigi.Parameter(
        default='rating', description='The target column name to predict.')  # type: str
    model_kwargs = luigi.DictParameter(default=dict(), description='Arguments of the model.')  # type: Dict[str, Any]
    max_data_size = luigi.IntParameter(default=50000000)
    output_file_path = luigi.Parameter(default='model/graph_convolutional_matrix_completion.zip')  # type: str

    def requires(self):
        return self.train_data_task

    def output(self):
        return dict(
            model=self.make_model_target(
                self.output_file_path,
                save_function=GraphConvolutionalMatrixCompletion.save,
                load_function=GraphConvolutionalMatrixCompletion.load),
            report=self.make_target('model_report/report.txt'))

    def run(self):
        tf.reset_default_graph()
        df = self.load_data_frame(
            required_columns={self.user_column_name, self.item_column_name, self.rating_column_name})

        df.drop_duplicates(subset=[self.user_column_name, self.item_column_name], inplace=True)
        df = sklearn.utils.shuffle(df)
        df = df.head(n=self.max_data_size)

        user_ids = df[self.user_column_name].values
        item_ids = df[self.item_column_name].values
        ratings = df[self.rating_column_name].values

        model = GraphConvolutionalMatrixCompletion(
            user_ids=user_ids, item_ids=item_ids, ratings=ratings, **self.model_kwargs)
        self.task_log['report'] = [str(self.model_kwargs)] + model.fit()
        self.dump(self.task_log['report'], 'report')
        self.dump(model, 'model')
