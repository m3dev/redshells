from logging import getLogger

import sklearn

logger = getLogger(__name__)


def calculate_auc(y_true, y_score, pos_label=1):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score, pos_label=pos_label)
    return sklearn.metrics.auc(fpr, tpr)
