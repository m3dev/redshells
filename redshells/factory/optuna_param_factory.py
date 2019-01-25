from typing import Dict

import optuna

from redshells.factory.singleton import Singleton


def _xgbclassifiler_default(trial: optuna.trial.Trial):
    param = {'silent': 1, 'objective': 'binary:logistic',
             'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
             'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
             'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0)
             }

    if param['booster'] == 'gbtree' or param['booster'] == 'dart':
        param['max_depth'] = trial.suggest_int('max_depth', 1, 9)
        param['eta'] = trial.suggest_loguniform('eta', 1e-8, 1.0)
        param['gamma'] = trial.suggest_loguniform('gamma', 1e-8, 1.0)
        param['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
    if param['booster'] == 'dart':
        param['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
        param['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
        param['rate_drop'] = trial.suggest_loguniform('rate_drop', 1e-8, 1.0)
        param['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)

    return param


class _OptunaParamFactory(metaclass=Singleton):
    def __init__(self):
        self._rules = dict()
        self._rules['XGBClassifier_default'] = _xgbclassifiler_default

    def get(self, key: str, trial: optuna.trial.Trial):
        if key not in self._rules:
            raise RuntimeError(
                f'"{key}" is not registered. Please class "register_optuna_param" beforehand. The keys are {list(self._rules.keys())}'
            )

        return self._rules[key](trial)

    def register(self, key, rule):
        self._rules[key] = rule


def get_optuna_param(key, trial):
    return _OptunaParamFactory().get(key, trial)


def register_optuna_param_rule(key: str, rule: Dict) -> None:
    _OptunaParamFactory().register(key, rule)
