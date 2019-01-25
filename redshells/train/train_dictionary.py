from typing import Any
from typing import Dict
from typing import List

import gensim
import gokart
import luigi

import redshells


class TrainDictionary(gokart.TaskOnKart):
    task_namespace = 'redshells'
    tokenized_text_data_task = gokart.TaskInstanceParameter(
        description='The task outputs tokenized texts with type "List[List[str]]".')
    output_file_path = luigi.Parameter(default='model/dictionary.pkl')  # type: str
    dictionary_filter_kwargs = luigi.DictParameter(
        default=dict(no_below=5, no_above=0.5, keep_n=100000, keep_tokens=None),
        description='Arguments for FastText except "sentences". Please see gensim.corpura.FastText for more details.'
    )  # type: Dict[str, Any]

    def requires(self):
        return self.tokenized_text_data_task

    def output(self):
        return self.make_target(self.output_file_path)

    def run(self):
        texts = self.load()  # type: List
        if isinstance(texts[0], str):
            texts = redshells.train.utils.TokenIterator(texts=texts)
        dictionary = gensim.corpora.Dictionary(texts)
        if len(self.dictionary_filter_kwargs):
            dictionary.filter_extremes(**self.dictionary_filter_kwargs)
        self.dump(dictionary)
