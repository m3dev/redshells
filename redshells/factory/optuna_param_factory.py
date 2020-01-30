from typing import Dict

import optuna

from redshells.factory.singleton import Singleton


def _xgbclassifier_default(trial: optuna.trial.Trial):
    param = {
        'silent': 1,
        'objective': 'binary:logistic',
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


def _lgbmclassifier_default(trial: optuna.trial.Trial):
    # TODO: using LightGBMTuner
    params = {
        'boosting_type': trial.suggest_categorical('boosting', ['gbdt', 'dart', 'goss']),
        'objective': 'binary',
        'metric': ['binary', 'binary_error', 'auc'],
        'num_leaves': trial.suggest_int("num_leaves", 10, 500),
        'learning_rate': trial.suggest_loguniform("learning_rate", 1e-5, 1),
        'feature_fraction': trial.suggest_uniform("feature_fraction", 0.0, 1.0),
    }
    if params['boosting_type'] == 'dart':
        params['drop_rate'] = trial.suggest_loguniform('drop_rate', 1e-8, 1.0)
        params['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)
    if params['boosting_type'] == 'goss':
        params['top_rate'] = trial.suggest_uniform('top_rate', 0.0, 1.0)
        params['other_rate'] = trial.suggest_uniform('other_rate', 0.0, 1.0 - params['top_rate'])

    return params


def _catboostclassifier_default(trial: optuna.trial.Trial):
    params = {
        'iterations': trial.suggest_int('iterations', 50, 300),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'random_strength': trial.suggest_int('random_strength', 0, 100),
        'bagging_temperature': trial.suggest_loguniform('bagging_temperature', 0.01, 100.00),
        'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
        'od_wait': trial.suggest_int('od_wait', 10, 50)
    }

    return params

class _OptunaParamFactory(metaclass=Singleton):
    def __init__(self):
        self._rules = dict()
        self._rules['XGBClassifier_default'] = _xgbclassifier_default
        self._rules['LGBMClassifier_default'] = _lgbmclassifier_default
        self._rules['CatBoostClassifier_default'] = _catboostclassifier_default

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
