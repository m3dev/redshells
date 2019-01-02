from logging import getLogger

import gokart
import luigi
import numpy as np
import pandas as pd
import sklearn

logger = getLogger(__name__)


class MakeClickTrainData(gokart.TaskOnKart):
    task_namespace = 'redshells.word_item_similarity'
    click_data_task = gokart.TaskInstanceParameter()
    min_user_count = luigi.IntParameter(default=100)  # type: int
    min_item_count = luigi.IntParameter(default=100)  # type: int
    max_item_frequency = luigi.FloatParameter(default=0.05)  # type: float
    user_column_name = luigi.Parameter()  # type: str
    item_column_name = luigi.Parameter()  # type: str
    service_column_name = luigi.Parameter()  # type: str
    output_file_path = luigi.Parameter(default='app/word_item_similarity/clicks_train_data.pkl')  # type: str

    def requires(self):
        return self.click_data_task

    def output(self):
        return self.make_target(self.output_file_path)

    def run(self):
        data = self.load_data_frame(
            required_columns={self.user_column_name, self.item_column_name, self.service_column_name})
        data = pd.concat([self._make_click_data(grouped) for name, grouped in data.groupby(self.service_column_name)])
        logger.info('dumping...')
        self.dump(data)

    def _make_click_data(self, data: pd.DataFrame):
        logger.info(f'filtering... size={data.shape}')
        data = self._filter_data(data)
        logger.info(f'size={data.shape}')

        data['click'] = 1
        logger.info(f'data size is {data.shape}.')
        logger.info('sampling...')
        negative = self._sample_negative_examples(data)
        logger.info(f'negative samples size is {negative.shape}.')
        logger.info('concatenating...')
        data = pd.concat([data, negative], sort=False)
        return data

    def _sample_negative_examples(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info('preprocessing...')
        user_ids = df[self.user_column_name].unique()
        item_ids = df[self.item_column_name].unique()
        item2service = dict(zip(df[self.item_column_name].tolist(), df[self.service_column_name].tolist()))
        user2index = dict(zip(user_ids, list(range(len(user_ids)))))
        item2index = dict(zip(item_ids, list(range(len(item_ids)))))
        n_users = len(user_ids)
        n_items = len(item_ids)
        positive_examples = set(
            list(df[self.user_column_name].apply(user2index.get).values +
                 df[self.item_column_name].apply(item2index.get).values * n_users))
        n_positive_examples = len(positive_examples)
        logger.info('negative sampling...')
        negative_examples = set(np.random.randint(low=0, high=n_users * n_items, size=n_positive_examples * 2))
        logger.info('making unique list...')
        negative_examples = np.array(list(negative_examples - positive_examples))
        logger.info('shuffling...')
        negative_examples = sklearn.utils.shuffle(negative_examples)
        negative_examples = negative_examples[:n_positive_examples]

        logger.info('making data frame...')
        examples = pd.DataFrame(
            dict(user_id=negative_examples % n_users, item_id=negative_examples // n_users, click=0))
        examples[self.user_column_name] = user_ids[examples[self.user_column_name].values]
        examples[self.item_column_name] = item_ids[examples[self.item_column_name].values]
        examples[self.service_column_name] = examples[self.item_column_name].apply(item2service.get)
        examples.drop_duplicates(inplace=True)
        return examples

    def _filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df.drop_duplicates(inplace=True)
        n_users = len(set(df[self.user_column_name]))
        max_item_count = n_users * self.max_item_frequency
        logger.info(f'max_item_count={max_item_count}')
        logger.info(f'min_item_count={self.min_item_count}')
        logger.info(f'min_user_count={self.min_user_count}')
        df = df.groupby(self.item_column_name).filter(lambda xs: self.min_item_count <= len(xs) <= max_item_count)
        df = df.groupby(self.user_column_name).filter(lambda xs: self.min_user_count <= len(xs))
        return df
