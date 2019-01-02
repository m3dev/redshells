import itertools
from logging import getLogger
from typing import Dict, Any

import luigi
import numpy as np
import pandas as pd

import gokart

logger = getLogger(__name__)


class MakeSimilarityData(gokart.TaskOnKart):
    task_namespace = 'redshells.word_item_similarity'
    word2items_task = gokart.TaskInstanceParameter(
        description='A task which outputs a mapping from word to items which includes the word.')
    similarity_task = gokart.TaskInstanceParameter(description='A task which outputs a similarity data.')
    item_id_0_column_name = luigi.Parameter(description='A column name of item id.')  # type: str
    item_id_1_column_name = luigi.Parameter(description='A column name of item id.')  # type: str
    similarity_column_name = luigi.Parameter(description='A column name of similarity.')  # type: str
    negative_sample_proportion = luigi.FloatParameter(
        default=4.0,
        description='A proportion of negative samples which are generate from positive sample data.')  # type: float
    positive_similarity_rate = luigi.FloatParameter(default=0.8)  # type: float
    negative_similarity_rate = luigi.FloatParameter(default=0.0)  # type: float
    use_mf = luigi.BoolParameter(default=True)
    output_file_path = luigi.Parameter(default='app/word_item_similarity/make_similarity_data.pkl')  # type: str

    def requires(self):
        return dict(word2items=self.word2items_task, similarity=self.similarity_task)

    def output(self):
        return self.make_target(self.output_file_path)

    def run(self):
        word2items = self.load('word2items')
        similarity = self.load_data_frame(
            'similarity',
            required_columns={self.item_id_0_column_name, self.item_id_1_column_name, self.similarity_column_name})
        positive_examples = list()  # type: pd.DataFrame
        negative_examples = list()  # type: pd.DataFrame

        positive_examples.append(self._word_positive_similarity(word2items))
        negative_examples.append(self._word_negative_similarity(word2items))

        if self.use_mf:
            positive_examples.append(self._positive_similarity(similarity, size=positive_examples[0].shape[0]))
            negative_examples.append(self._negative_similarity(similarity, size=negative_examples[0].shape[0]))

        data = pd.concat(positive_examples + negative_examples)
        data.drop_duplicates(subset=['item_id_0', 'item_id_1'], inplace=True)
        self.dump(data)

    def _word_positive_similarity(self, word2items: Dict[Any, Any]):
        logger.info(f'task_id={self.make_unique_id()}')
        data = pd.concat([
            pd.DataFrame(dict(item_id_0=v, item_id_1=np.random.choice(v, size=len(v)), similarity=1))
            for v in word2items.values() if len(v) > 1
        ])
        logger.info(f'word positive similarity size = {data.shape}.')
        return data

    def _word_negative_similarity(self, word2items: Dict[Any, Any]):
        all_items = set(list(itertools.chain.from_iterable(word2items.values())))
        data = pd.concat([
            pd.DataFrame(
                dict(
                    item_id_0=np.random.choice(v, size=int(len(v) * self.negative_sample_proportion)),
                    item_id_1=np.random.choice(
                        list(all_items - set(v)), size=int(len(v) * self.negative_sample_proportion)),
                    similarity=0)) for v in word2items.values() if len(v) > 1
        ])
        logger.info(f'word negative similarity size = {data.shape}.')
        return data

    def _positive_similarity(self, similarity: pd.DataFrame, size: int):
        data = similarity[
            similarity[self.similarity_column_name] > self.positive_similarity_rate].copy()  # type: pd.DataFrame
        data.sort_values(by=self.similarity_column_name, ascending=False, inplace=True)
        data = data.head(n=size)
        data = pd.DataFrame(
            dict(
                item_id_0=data[self.item_id_0_column_name].tolist(),
                item_id_1=data[self.item_id_1_column_name].tolist(),
                similarity=1))
        logger.info(f'mf positive similarity size = {data.shape}.')
        return data

    def _negative_similarity(self, similarity: pd.DataFrame, size: int):
        data = similarity[
            similarity[self.similarity_column_name] < self.negative_similarity_rate].copy()  # type: pd.DataFrame
        data.sort_values(by=self.similarity_column_name, ascending=False, inplace=True)
        data = data.head(n=size)
        data = pd.DataFrame(
            dict(
                item_id_0=data[self.item_id_0_column_name].tolist(),
                item_id_1=data[self.item_id_1_column_name].tolist(),
                similarity=0))
        logger.info(f'mf negative similarity size = {data.shape}.')
        return data
