import random
import string
import unittest
from logging import getLogger
from unittest.mock import patch

from mock import MagicMock
import numpy as np
import redshells
from redshells.model.xgb_text_classifier_with_scdv import XGBTextClassifierWithSCDV

logger = getLogger(__name__)


class TestXGBTextClassifierWithSCDV(unittest.TestCase):
    def test_fit(self):
        scdv = MagicMock(spec=redshells.model.SCDV)
        scdv_dimension_size = 100
        valid_dimension_size = 10
        xgboost_kwargs = dict()
        n_data = 100
        n_tokens = 20
        tokens = [random.choices(string.ascii_letters, k=n_tokens) for _ in range(n_data)]
        labels = [random.choice(string.ascii_letters) for _ in range(n_data)]
        original_scdv_embeddings = np.random.uniform(size=(n_data, scdv_dimension_size))
        scdv.infer_vector.return_value = original_scdv_embeddings

        xgboost_mock = MagicMock()
        xgboost_mock.return_value = xgboost_mock
        with patch('xgboost.XGBClassifier', xgboost_mock):
            model = XGBTextClassifierWithSCDV(
                scdv=scdv, valid_dimension_size=valid_dimension_size, xgboost_kwargs=xgboost_kwargs)
            model.fit(tokens=tokens, labels=labels)
        xgboost_mock.assert_called_once_with(xgboost_kwargs=xgboost_kwargs)
        xgboost_mock.fit.assert_called_once()
        fit_args = xgboost_mock.fit.call_args[0]
        self.assertEqual((n_data, valid_dimension_size), fit_args[0].shape)


if __name__ == '__main__':
    unittest.main()
