from logging import getLogger
from typing import List

import gensim
import gokart
import luigi
from typing import Dict

from typing import Any

import redshells

logger = getLogger(__name__)


class TrainLdaModel(gokart.TaskOnKart):
    output_file_path = luigi.Parameter(default='model/lda_model.pkl')  # type: str
    tokenized_text_data_task = gokart.TaskInstanceParameter(
        description='A task outputs tokenized texts with type "List[List[str]]".')
    dictionary_task = gokart.TaskInstanceParameter(description='A task outputs gensim.corpura.Dictionary.')
    lda_model_kwargs = luigi.DictParameter(
        default=dict(n_topics=100, chunksize=16, decay=0.5, offset=16, iterations=3, eta=1.e-16),
        description='Arguments for redshells.model.LdaModel.')  # type: Dict[str, Any]

    def requires(self):
        return dict(tokenized_texts=self.tokenized_text_data_task, dictionary=self.dictionary_task)

    def output(self):
        return self.make_target(self.output_file_path)

    def run(self):
        tokenized_texts = self.load('tokenized_texts')  # type: List[List[str]]
        dictionary = self.load('dictionary')  # type: gensim.corpora.Dictionary
        model = redshells.model.LdaModel(**self.lda_model_kwargs)
        model.fit(texts=tokenized_texts, dictionary=dictionary)
        self.dump(model)
