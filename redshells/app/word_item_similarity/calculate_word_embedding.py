from typing import List

import luigi
import sklearn

import gokart
import numpy as np


class CalculateWordEmbedding(gokart.TaskOnKart):
    task_namespace = 'redshells.word_item_similarity'
    word_task = gokart.TaskInstanceParameter()
    word2item_task = gokart.TaskInstanceParameter()
    item2embedding_task = gokart.TaskInstanceParameter()
    output_file_path = luigi.Parameter(default='app/word_item_similarity/calculate_word_embedding.pkl')  # type: str

    def requires(self):
        return dict(word=self.word_task, word2item=self.word2item_task, item2embedding=self.item2embedding_task)

    def output(self):
        return self.make_target(self.output_file_path)

    def run(self):
        word_data = self.load('word')
        word2item = self.load('word2item')
        item2embedding = self.load('item2embedding')

        results = {word: self._calculate(word2item[word], item2embedding) for word in word_data if word in word2item}
        self.dump(results)

    def _calculate(self, items, item2embedding):
        embeddings = [item2embedding[item] for item in items if item in item2embedding]
        if not embeddings:
            return None
        return sklearn.preprocessing.normalize([np.sum(embeddings, axis=0)], norm='l2', axis=1)[0]


class CalculateWordEmbeddingWithSCDV(gokart.TaskOnKart):
    """
    Calculate word embeddings with scdv 
    """
    task_namespace = 'redshells.word_item_similarity'
    word_task = gokart.TaskInstanceParameter()
    scdv_task = gokart.TaskInstanceParameter()
    l2_normalize = luigi.BoolParameter()  # type: bool
    output_file_path = luigi.Parameter(default='app/word_item_similarity/calculate_word_embedding.pkl')  # type: str

    def requires(self):
        return dict(word=self.word_task, scdv=self.scdv_task)

    def output(self):
        return self.make_target(self.output_file_path)

    def run(self):
        scdv = self.load('scdv')
        words = self.load('word')  # type: List[str]

        embeddings = scdv.infer_vector([[word] for word in words], l2_normalize=self.l2_normalize)
        self.dump(dict(zip(list(words), list(embeddings))))
