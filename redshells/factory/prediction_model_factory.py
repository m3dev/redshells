from typing import Callable

from redshells.factory.singleton import Singleton


class _PredictionModelFactory(metaclass=Singleton):
    def __init__(self):
        self._models = dict()
        try:
            import sklearn.ensemble
            self._models['RandomForestClassifier'] = sklearn.ensemble.RandomForestClassifier
        except ImportError:
            pass

        try:
            import xgboost
            self._models['XGBClassifier'] = xgboost.XGBClassifier
        except ImportError:
            pass

    def get(self, key: str):
        if key in self._models:
            return self._models[key]
        raise RuntimeError(
            f'"{key}" is not registered. Please class "register_prediction_model" beforehand. The keys are {list(self._models.keys())}'
        )

    def register(self, key, class_name):
        self._models[key] = class_name


def get_prediction_model_type(key):
    return _PredictionModelFactory().get(key)


def create_prediction_model(key: str, **kwargs):
    return _PredictionModelFactory().get(key)(**kwargs)


def register_prediction_model(key: str, class_name: Callable) -> None:
    _PredictionModelFactory().register(key, class_name)
