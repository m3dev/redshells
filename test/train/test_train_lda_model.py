import unittest
from unittest.mock import MagicMock

import gensim
import luigi

import redshells


class _DummyTask(luigi.Task):
    pass


class TrainLdaModelTest(unittest.TestCase):
    def setUp(self):
        self.input_data = None
        self.dump_data = None
        redshells.train.TrainLdaModel.clear_instance_cache()

    def test_run(self):
        tokenized_texts = [['a', 'b'], ['a', 'c', 'd', 'e'], ['a']]
        dictionary = gensim.corpora.Dictionary(tokenized_texts)
        self.input_data = {'tokenized_texts': tokenized_texts, 'dictionary': dictionary}
        task = redshells.train.TrainLdaModel(tokenized_text_data_task=_DummyTask(), dictionary_task=_DummyTask())
        task.load = MagicMock(side_effect=self._load)
        task.dump = MagicMock(side_effect=self._dump)

        task.run()
        self.assertIsInstance(self.dump_data, redshells.model.LdaModel)

    def _load(self, *args, **kwargs):
        return self.input_data[args[0]]

    def _dump(self, *args):
        self.dump_data = args[0]


if __name__ == '__main__':
    unittest.main()
