from typing import Any
from typing import Dict

import luigi
import sklearn
import tensorflow as tf

import gokart
from redshells.model.graph_convolutional_matrix_completion import GraphConvolutionalMatrixCompletion


class NoneTask(gokart.TaskOnKart):
    def output(self):
        return self.make_target('none.pkl')

    def run(self):
        self.dump(None)


class TrainGraphConvolutionalMatrixCompletion(gokart.TaskOnKart):
    task_namespace = 'redshells'
    train_data_task = gokart.TaskInstanceParameter(
        description=
        'A task outputs a pd.DataFrame with columns={`user_column_name`, `item_column_name`, `target_column_name`}.')
    user_column_name = luigi.Parameter(default='user', description='The column name of user id.')  # type: str
    item_column_name = luigi.Parameter(default='item', description='The column name of item id')  # type: str
    rating_column_name = luigi.Parameter(
        default='rating', description='The target column name to predict.')  # type: str
    user_feature_task = gokart.TaskInstanceParameter(default=NoneTask())
    item_feature_task = gokart.TaskInstanceParameter(default=NoneTask())
    model_kwargs = luigi.DictParameter(default=dict(), description='Arguments of the model.')  # type: Dict[str, Any]
    max_data_size = luigi.IntParameter(default=50000000)  # type: int
    output_file_path = luigi.Parameter(default='model/graph_convolutional_matrix_completion.zip')  # type: str
    try_count = luigi.IntParameter(default=10)  # type: int
    decay_speed = luigi.FloatParameter(default=2.0)  # type: float
    test_size = luigi.FloatParameter(default=0.2)  # type: float
    # data parameters
    min_user_click_count = luigi.IntParameter(default=5)  # type: int
    max_user_click_count = luigi.IntParameter(default=200)  # type: int
    use_default_user = luigi.BoolParameter(default=False)  # type: bool

    def requires(self):
        return dict(
            train_data=self.train_data_task, user_features=self.user_feature_task, item_features=self.item_feature_task)

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
            'train_data', required_columns={self.user_column_name, self.item_column_name, self.rating_column_name})
        user_features = self.load('user_features')
        item_features = self.load('item_features')

        df.drop_duplicates(subset=[self.user_column_name, self.item_column_name], inplace=True)
        df = sklearn.utils.shuffle(df)
        df = df.head(n=int(self.max_data_size))

        user_ids = df[self.user_column_name].values
        item_ids = df[self.item_column_name].values
        ratings = df[self.rating_column_name].values

        model = GraphConvolutionalMatrixCompletion(
            user_ids=user_ids,
            item_ids=item_ids,
            ratings=ratings,
            user_features=user_features,
            item_features=item_features,
            test_size=self.test_size,
            **self.model_kwargs)
        self.task_log['report'] = [str(self.model_kwargs)] + model.fit(
            try_count=self.try_count, decay_speed=self.decay_speed)
        self.dump(self.task_log['report'], 'report')
        self.dump(model, 'model')
