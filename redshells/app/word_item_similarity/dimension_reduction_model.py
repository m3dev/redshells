from typing import Any
from typing import Dict

import luigi
import numpy as np
import sklearn

import gokart


class DimensionReductionModel(object):
    """ Reduce the dimension of vector values with respect to its importance.
    The importance is calculated by sum of squared values.
    """
    def __init__(self, dimension_size: int) -> None:
        self.dimension_size = dimension_size
        self.top_n_indices = None

    def fit(self, embeddings: np.ndarray) -> None:
        weights = np.sum(embeddings**2, axis=0)
        self.top_n_indices = weights.argsort()[-self.dimension_size:][::-1]

    def apply(self, embeddings: np.ndarray) -> np.ndarray:
        return embeddings[:, self.top_n_indices]


class TrainDimensionReductionModel(gokart.TaskOnKart):
    """ Train :py:class: `DimensionReductionModel` from item2embedding data with type `Dict[Any, np.ndarray]`."""
    task_namespace = 'redshells.word_item_similarity'
    item2embedding_task = gokart.TaskInstanceParameter(description='A task outputs item2embedding data with type = Dict[Any, np.ndarray].')
    dimension_size = luigi.IntParameter(description='the dimension of reduced vectors.')  # type: int
    output_file_path = luigi.Parameter(default='app/word_item_similarity/dimension_reduction_model.pkl')  # type: str

    def requires(self):
        return self.item2embedding_task

    def output(self):
        return self.make_target(self.output_file_path)

    def run(self):
        item2embedding = self.load()  # type: Dict[Any, np.ndarray]
        model = DimensionReductionModel(dimension_size=self.dimension_size)
        model.fit(np.array(list(item2embedding.values())))
        self.dump(model)


class ApplyDimensionReductionModel(gokart.TaskOnKart):
    task_namespace = 'redshells.word_item_similarity'
    item2embedding_task = gokart.TaskInstanceParameter(description='A task outputs item2embedding data with type = Dict[Any, np.ndarray].')
    dimension_reduction_model_task = gokart.TaskInstanceParameter(default='A task outputs a model instance of `DimensionReductionModel`.')
    l2_normalize = luigi.BoolParameter()  # type: bool
    output_file_path = luigi.Parameter(default='app/word_item_similarity/dimension_reduction_model.pkl')  # type: str

    def requires(self):
        return dict(item2embedding=self.item2embedding_task, model=self.dimension_reduction_model_task)

    def output(self):
        return self.make_target(self.output_file_path)

    def run(self):
        item2embedding = self.load('item2embedding')  # type: Dict[Any, np.ndarray]
        model = self.load('model')
        items = list(item2embedding.keys())
        embeddings = model.apply(np.array(list(item2embedding.values())))
        if self.l2_normalize:
            embeddings = sklearn.preprocessing.normalize(embeddings, axis=1, norm='l2')
        self.dump(dict(zip(items, list(embeddings))))
