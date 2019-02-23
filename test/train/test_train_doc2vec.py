import unittest
from unittest.mock import MagicMock

import gensim
import luigi

import redshells


class _DummyTask(luigi.Task):
    pass


class TrainDoc2VecTest(unittest.TestCase):
    def setUp(self):
        self.input_data = None
        self.dump_data = None
        redshells.train.TrainDoc2Vec.clear_instance_cache()

    def test_run(self):
        self.input_data = [['a', 'b'], ['a', 'c', 'd', 'e'], ['a']]
        task = redshells.train.TrainDoc2Vec(
            tokenized_text_data_task=_DummyTask(), doc2vec_kwargs=dict(vector_size=3, min_count=1))
        task.load = MagicMock(side_effect=self._load)
        task.dump = MagicMock(side_effect=self._dump)

        task.run()
        self.assertIsInstance(self.dump_data, gensim.models.Doc2Vec)

    def _load(self, *args, **kwargs):
        return self.input_data

    def _dump(self, *args):
        self.dump_data = args[0]


if __name__ == '__main__':
    unittest.main()
