from collections import Iterable
from logging import getLogger

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score

logger = getLogger(__name__)


def fit_model(task):
    x, y = task.create_train_data()
    model = task.create_model()
    model.fit(x, y)
    task.dump(model)


def validate_model(task, cv: int) -> None:
    x, y = task.create_train_data()
    model = task.create_model()

    scores = []

    def _scoring(y_true, y_pred):
        report = classification_report(y_true, y_pred, output_dict=True)
        logger.info(report)
        scores.append(report)
        return accuracy_score(y_true, y_pred)

    cross_val_score(model, x, y, cv=cv, scoring=make_scorer(_scoring))
    task.dump(scores)


def _flatten(items):
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in _flatten(x):
                yield sub_x
        else:
            yield x


def to_numpy(df: pd.DataFrame) -> np.ndarray:
    return np.array(list(_flatten(df.values))).reshape([df.shape[0], -1])
