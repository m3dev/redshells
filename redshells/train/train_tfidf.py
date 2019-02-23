from logging import getLogger
from typing import List

import luigi
from gensim.corpora import Dictionary

import gokart
import redshells

logger = getLogger(__name__)


class TrainTfidf(gokart.TaskOnKart):
    output_file_path = luigi.Parameter(default='model/tfidf.pkl')  # type: str
    tokenized_text_data_task = gokart.TaskInstanceParameter(
        description='A task outputs tokenized texts with type "List[List[str]]".')

    def requires(self):
        return self.tokenized_text_data_task

    def output(self):
        return self.make_target(self.output_file_path)

    def run(self):
        tokens = self.load()  # type: List[List[str]]
        dictionary = Dictionary(tokens)
        model = redshells.model.Tfidf(dictionary=dictionary, tokens=tokens)
        self.dump(model)
