from typing import Any
from typing import Dict
from typing import List

import gensim
import gokart
import luigi

import redshells.model


class TrainSCDV(gokart.TaskOnKart):
    tokenized_text_data_task = gokart.TaskInstanceParameter(
        description='A task outputs tokenized texts with type "List[List[str]]".')
    dictionary_task = gokart.TaskInstanceParameter(description='A task outputs gensim.corpora.Dictionary.')
    word2vec_task = gokart.TaskInstanceParameter(
        description='A task outputs gensim.models.Word2Vec, gensim.models.FastText or models with the same interface.')
    cluster_size = luigi.IntParameter(
        default=60, description='A cluster size of Gaussian mixture model in SCDV.')  # type: int
    sparsity_percentage = luigi.FloatParameter(
        default=0.04, description='A percentage of sparsity in SCDV')  # type: float
    gaussian_mixture_kwargs = luigi.DictParameter(
        default=dict(),
        description='Arguments for Gaussian mixture model except for cluster size.')  # type: Dict[str, Any]
    output_file_path = luigi.Parameter(default='model/scdv.pkl')  # type: str

    def requires(self):
        return dict(text=self.tokenized_text_data_task, dictionary=self.dictionary_task, word2vec=self.word2vec_task)

    def output(self):
        return self.make_target(self.output_file_path)

    def run(self):
        texts = self.load('text')  # type: List[List[str]]
        dictionary = self.load('dictionary')  # type: gensim.corpora.Dictionary
        word2vec = self.load('word2vec')  # type: gensim.models.Word2Vec

        model = redshells.model.SCDV(
            documents=texts,
            cluster_size=self.cluster_size,
            sparsity_percentage=self.sparsity_percentage,
            gaussian_mixture_kwargs=self.gaussian_mixture_kwargs,
            dictionary=dictionary,
            w2v=word2vec)
        self.dump(model)
