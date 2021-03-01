from collections import defaultdict
from logging import getLogger

import luigi
import numpy as np

import gokart

logger = getLogger(__name__)


class FilterItemByWordSimilarity(gokart.TaskOnKart):
    word2items_task = gokart.TaskInstanceParameter()
    word2embedding_task = gokart.TaskInstanceParameter()
    item2title_embedding_task = gokart.TaskInstanceParameter()
    no_below = luigi.FloatParameter()
    output_file_path = luigi.Parameter(default='app/word_item_similarity/filter_item_by_word_similarity.pkl')  # type: str

    def requires(self):
        return dict(word2items=self.word2items_task, word2embedding=self.word2embedding_task, item2title_embedding=self.item2title_embedding_task)

    def output(self):
        return self.make_target(self.output_file_path)

    def run(self):
        word2items = self.load('word2items')
        word2embedding = self.load('word2embedding')
        item2title_embedding = self.load('item2title_embedding')

        filtered_word2items = defaultdict(list)
        for word, items in word2items.items():
            word_embedding = word2embedding[word]
            for item in items:
                title_embedding = item2title_embedding[item]
                if np.inner(word_embedding, title_embedding) > self.no_below:
                    filtered_word2items[word].append(item)

        self.dump(dict(filtered_word2items))
