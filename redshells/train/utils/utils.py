from collections import Iterable
from logging import getLogger

import numpy as np
import optuna
import pandas as pd
import sklearn
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score, train_test_split

import redshells

logger = getLogger(__name__)


def fit_model(task):
    x, y = task.create_train_data()
    model = task.create_model()
    model.fit(x, y)
    task.dump(model)
    logger.info(logger.info(classification_report(y, model.predict(x))))


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


def optimize_model(task, param_name, test_size: float, binary=False) -> None:
    x, y = task.create_train_data()

    def objective(trial):
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_size)
        param = redshells.factory.get_optuna_param(param_name, trial)
        model = task.create_model()
        model.set_params(**param)
        model.fit(train_x, train_y)
        predictions = model.predict(test_x)

        if binary:
            predictions = np.rint(predictions)

        return 1.0 - sklearn.metrics.accuracy_score(test_y, predictions)

    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    task.dump(dict(best_params=study.best_params, best_value=study.best_value))


def _flatten(items):
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in _flatten(x):
                yield sub_x
        else:
            yield x


def to_numpy(df: pd.DataFrame) -> np.ndarray:
    return np.array(list(_flatten(df.values))).reshape([df.shape[0], -1])
