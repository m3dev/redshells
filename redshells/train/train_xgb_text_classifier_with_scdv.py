from logging import getLogger
from typing import Dict

import gokart
import luigi
import sklearn

from redshells.model import XGBTextClassifierWithSCDV

logger = getLogger(__name__)


class TrainXGBTextClassifierWithSCDV(gokart.TaskOnKart):
    train_data_task = gokart.TaskInstanceParameter()
    scdv_task = gokart.TaskInstanceParameter()
    feature_column = luigi.Parameter()  # type: str
    label_column = luigi.Parameter()  # type: str
    valid_dimension_size = luigi.IntParameter(default=60)  # type: int
    model_kwargs = luigi.DictParameter(dict(n_estimators=10))  # type: Dict

    def requires(self):
        return dict(train_data=self.train_data_task, scdv=self.scdv_task)

    def output(self):
        return self.make_target('model/train_xgb_text_classifier_with_scdv.pkl')

    def run(self):
        train_data = self.load_data_frame('train_data', required_columns={self.feature_column, self.label_column})
        scdv = self.load('scdv')
        model = self._run(
            scdv=scdv,
            train_data=train_data,
            valid_dimension_size=self.valid_dimension_size,
            feature_column=self.feature_column,
            label_column=self.label_column,
            model_kwargs=self.model_kwargs)
        self.dump(model)

    @staticmethod  # to make it easy to test
    def _run(scdv, train_data, valid_dimension_size, feature_column, label_column,
             model_kwargs) -> XGBTextClassifierWithSCDV:
        model = XGBTextClassifierWithSCDV(scdv=scdv, valid_dimension_size=valid_dimension_size, **model_kwargs)
        train_data = sklearn.utils.shuffle(train_data)
        model.fit(train_data[feature_column].tolist(), train_data[label_column].tolist())
        return model
