from logging import getLogger

import luigi
import numpy as np
import pandas as pd

import gokart
import redshells.app.word_item_similarity
import redshells.data
import redshells.train

logger = getLogger(__name__)


class MakeDummyWordData(gokart.TaskOnKart):
    task_namespace = 'examples'

    data_size = luigi.IntParameter(default=500)  # type: int

    def output(self):
        return self.make_target('word_item_similarity/dummy_word.pkl')

    def run(self):
        self.dump([f'word_{i}' for i in range(self.data_size)])


class MakeDummyItemData(gokart.TaskOnKart):
    task_namespace = 'examples'
    data_size = luigi.IntParameter(default=100)  # type: int

    def requires(self):
        return MakeDummyWordData()

    def output(self):
        return self.make_target('word_item_similarity/dummy_word.pkl')

    def run(self):
        words = self.load()
        results = pd.DataFrame(
            dict(
                item_id=[f'item_{i}' for i in range(self.data_size)],
                text=[list(np.random.choice(words, size=100, replace=True)) for _ in range(self.data_size)]))
        self.dump(results)


class MakeDummyClickData(gokart.TaskOnKart):
    task_namespace = 'examples'
    user_size = luigi.IntParameter(default=100)  # type: int
    data_size = luigi.IntParameter(default=1000)  # type: int

    def requires(self):
        return MakeDummyItemData()

    def output(self):
        return self.make_target('word_item_similarity/dummy_word.pkl')

    def run(self):
        item_data = self.load_data_frame()
        items = item_data['item_id'].unique()
        users = [f'user_{u}' for u in range(self.user_size)]
        results = pd.DataFrame(
            dict(
                user_id=np.random.choice(users, size=self.data_size, replace=True),
                item_id=np.random.choice(items, size=self.data_size, replace=True),
                service_id=0))
        self.dump(results)


class WordItemSimilarityExample(gokart.TaskOnKart):
    task_namespace = 'examples'

    def requires(self):
        word_data = MakeDummyWordData()
        item_train_data = MakeDummyItemData()
        click_data = MakeDummyClickData()
        item_predict_data = MakeDummyItemData(data_size=1000)
        return redshells.app.word_item_similarity.BuildWordItemSimilarity(
            word_data_task=word_data,
            item_train_data_task=item_train_data,
            click_data_task=click_data,
            item_predict_data_task=item_predict_data)

    def output(self):
        return self.make_target('word_item_similarity/example.pkl')

    def run(self):
        data = self.load()
        print(data)


if __name__ == '__main__':
    luigi.configuration.add_config_path('./config/example.ini')
    luigi.run([
        'examples.WordItemSimilarityExample',
        '--local-scheduler',
    ])
