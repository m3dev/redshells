import random
import string
import unittest
from logging import getLogger
from unittest.mock import MagicMock, patch

import pandas as pd

import redshells
from redshells.model import XGBTextClassifierWithSCDV
from redshells.train import TrainXGBTextClassifierWithSCDV

logger = getLogger(__name__)


class TestTrainXGBTextClassifierWithSCDV(unittest.TestCase):
    def test_run(self):
        scdv = MagicMock(spec=redshells.model.SCDV)
        n_data = 100
        valid_dimension_size = 10
        feature_column = 'feature'
        label_column = 'label'
        model_kwargs = dict()
        train_data = pd.DataFrame(
            dict(
                feature=[random.choices(string.ascii_letters, k=100) for _ in range(n_data)],
                label=random.choices(string.ascii_letters, k=n_data)))

        model_mock = MagicMock(spec=XGBTextClassifierWithSCDV)
        model_init_mock = MagicMock(return_value=model_mock)
        with patch('redshells.train.train_xgb_text_classifier_with_scdv.XGBTextClassifierWithSCDV', model_init_mock):
            result = TrainXGBTextClassifierWithSCDV._run(scdv, train_data, valid_dimension_size, feature_column,
                                                         label_column, model_kwargs)
        model_init_mock.assert_called_once_with(scdv=scdv, valid_dimension_size=valid_dimension_size, **model_kwargs)
        model_mock.fit.assert_called_once()
        self.assertEqual(model_mock, result)
