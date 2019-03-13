from typing import Any, Dict
import luigi
import sklearn
import tensorflow as tf
import gokart
from redshells.model.multiresolution_gan import MultiresolutionGraphAttentionNetworks

class NoneTask(gokart.TaskOnKart):
    def output(self):
        return self.make_target('none.pkl')

    def run(self):
        self.dump(None)

class TrainMultiresolutionGraphAttentionNetworks(gokart.TaskOnKart):
    task_namespace = 'redshells'
    train_data_task = gokart.TaskInstanceParameter(
        description = 'A task outputs a pd.Dataframe')
    
    user_column_name = luigi.Parameter(default='user', description='The column name of user id.')  # type: str
    item_column_name = luigi.Parameter(default='item', description='The column name of item id')  # type: str
    rating_column_name = luigi.Parameter(
        default='rating', description='The target column name to predict.')  # type: str
    user_feature_task = gokart.TaskInstanceParameter(default=NoneTask())
    item_feature_task = gokart.TaskInstanceParameter(default=NoneTask())
    model_kwargs = luigi.DictParameter(default=dict(), description='Arguments of the model.')  # type: Dict[str, Any]
    max_data_size = luigi.IntParameter(default=50000000)  # type: int
    output_file_path = luigi.Parameter(default='model/multiresolution_graph_attention_networks.zip')  # type: str
    try_count = luigi.IntParameter(default=5)  # type: int
    decay_speed = luigi.FloatParameter(default=2.0)  # type: float

    def requires(self):
        return dict(train_data=self.train_data_task,
                    item_features=self.item_feature_task,
                    user_features=self.user_feature_task)

    def output(self):
        return dict(
            model=self.make_model_target(
                self.output_file_path,
                save_function=MultiresolutionGraphAttentionNetworks.save,
                load_function=MultiresolutionGraphAttentionNetworks.load),
            report=self.make_target('model_report/report.txt'))

    def run(self):
        tf.reset_default_graph()
        df = self.load_data_frame(
            'train_data', required_columns={self.user_column_name, self.item_column_name, self.rating_column_name})
        item_features = self.load('item_features')
        user_features = self.load('user_features')

        df.drop_duplicates(subset=[self.user_column_name, self.item_column_name], inplace=True)
        df = sklearn.utils.shuffle(df)
        df = df.head(n=int(self.max_data_size))

        user_ids = df[self.user_column_name].values
        item_ids = df[self.item_column_name].values
        ratings = df[self.rating_column_name].values

        model = MultiresolutionGraphAttentionNetworks(
            user_ids=user_ids, item_ids=item_ids, ratings=ratings, user_features=user_features, item_features=item_features, **self.model_kwargs)
        self.task_log['report'] = [str(self.model_kwargs)] + model.fit(
            try_count=self.try_count, decay_speed=self.decay_speed)
        self.dump(self.task_log['report'], 'report')
        self.dump(model, 'model')

