from logging import getLogger
from typing import Any, Dict

import luigi
import numpy as np
import pandas as pd
from tqdm import tqdm

import gokart

logger = getLogger(__name__)


class CalculateWordItemSimilarity(gokart.TaskOnKart):
    """
    Calculate similarity between words and items. 
    """
    task_namespace = 'redshells.word_item_similarity'
    word2embedding_task = gokart.TaskInstanceParameter()
    item2embedding_task = gokart.TaskInstanceParameter()
    similarity_model_task = gokart.TaskInstanceParameter()
    prequery_return_size = luigi.IntParameter()  # type: int
    return_size = luigi.IntParameter()  # type: int
    output_file_path = luigi.Parameter(
        default='app/word_item_similarity/calculate_word_item_similarity.pkl')  # type: str

    def requires(self):
        return dict(
            word2embedding=self.word2embedding_task,
            item2embedding=self.item2embedding_task,
            model=self.similarity_model_task)

    def output(self):
        return self.make_target(self.output_file_path)

    def run(self):
        word2embedding = self.load('word2embedding')  # type: Dict[Any, np.ndarray]
        item2embedding = self.load('item2embedding')  # type: Dict[Any, np.ndarray]
        model = self.load('model')

        item_embeddings = np.array(list(item2embedding.values()))
        items = np.array(list(item2embedding.keys()))
        results = pd.concat([
            self._find_top_similarity(model, word, embedding, items, item_embeddings)
            for word, embedding in tqdm(word2embedding.items())
        ])
        self.dump(results.reset_index(drop=True))

    def _find_top_similarity(self, model, word, word_embedding: np.ndarray, items: np.ndarray,
                             item_embeddings: np.ndarray) -> pd.DataFrame:
        if word_embedding is None:
            logger.info(f'word {word} is not registered.')
            return pd.DataFrame(columns=['word', 'item', 'similarity'])
        filtered_indices = self._filter(word_embedding, item_embeddings)
        similarities = self._predict(model, word_embedding, item_embeddings[filtered_indices, :])
        top_indices = similarities.argsort()[-self.return_size:][::-1]
        return pd.DataFrame(
            dict(word=word, item=items[filtered_indices[top_indices]], similarity=similarities[top_indices]))

    def _predict(self, model, word_embedding: np.ndarray, item_embeddings: np.ndarray) -> np.ndarray:
        i = list(model.classes_).index(1)
        return model.predict_proba(item_embeddings * word_embedding)[:, i]

    def _filter(self, word_embedding: np.ndarray, item_embeddings: np.ndarray) -> np.ndarray:
        similarities = np.dot(item_embeddings, word_embedding.reshape([-1, 1])).flatten()
        top_indices = similarities.argsort()[-self.prequery_return_size:][::-1]
        return top_indices
