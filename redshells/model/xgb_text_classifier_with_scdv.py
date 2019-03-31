from logging import getLogger
from typing import Dict, List

import numpy as np
import redshells
import xgboost
from sklearn.utils import deprecated

logger = getLogger(__name__)


@deprecated('Please use redshells.')
class XGBTextClassifierWithSCDV(object):
    def __init__(self, scdv: redshells.model.SCDV, valid_dimension_size: int, **xgboost_kwargs) -> None:
        self._scdv = scdv
        self._valid_dimension_size = valid_dimension_size
        self._valid_indices = None  # type: List[int]
        self._label2index = None  # type: Dict[str, int]
        self._labels = None  # type: List[str]
        self._model = None  # type: xgboost.XGBClassifier
        self._model_kwargs = xgboost_kwargs

    def fit(self, tokens: List[List[str]], labels: List[str]) -> None:
        assert len(tokens) == len(labels), 'input sizes must be the same.'
        embeddings = self._to_embeddings(tokens)
        self._labels = list(set(labels))
        self._label2index = dict(zip(self._labels, range(len(self._labels))))
        label_indices = [self._label2index.get(label) for label in labels]
        self._model = xgboost.XGBClassifier(**self._model_kwargs)
        self._model.fit(embeddings, np.array(label_indices))
        return

    def predict(self, tokens: List[List]) -> List[str]:
        embeddings = self._to_embeddings(tokens)
        predicts = self._model.predict(embeddings)
        return [self._labels[predict] for predict in predicts]

    def predict_proba(self, tokens: List[List]) -> np.ndarray:
        embeddings = self._to_embeddings(tokens)
        return self._model.predict_proba(embeddings)

    @property
    def labels(self) -> List[str]:
        return self._labels

    def _to_embeddings(self, tokens: List[List[str]]):
        embeddings = self._scdv.infer_vector(tokens)
        importances = np.sum(embeddings**2, axis=0)
        if self._valid_indices is None:
            self._valid_indices, _ = zip(
                *sorted(zip(range(importances.shape[0]), importances), key=lambda x: x[1],
                        reverse=True)[:self._valid_dimension_size])
        return embeddings[:, self._valid_indices]
