import unittest
from unittest.mock import MagicMock

import gokart

import redshells


class _DummyTask(gokart.TaskOnKart):
    pass


class TrainFastTextTest(unittest.TestCase):
    def setUp(self):
        self.input_data = None
        self.dump_data = None

    def test_run(self):
        self.input_data = [
            'python is an interpreted high-level general-purpose programming language',
            'tt provides constructs that enable clear programming on both small and large scales'
        ]
        task = redshells.train.TrainFastText(tokenized_text_data_task=_DummyTask(), fasttext_kwargs=dict(min_count=1))
        task.load = MagicMock(side_effect=self._load)
        task.dump = MagicMock(side_effect=self._dump)
        task.run()
        self.assertTrue('python' in self.dump_data.wv.vocab.keys())
        self.assertTrue('programming' in self.dump_data.wv.vocab.keys())

    def _load(self, *args, **kwargs):
        if 'target' in kwargs:
            return self.__dict__.get(kwargs['target'], None)
        if len(args) > 0:
            return self.__dict__.get(args[0], None)
        return self.input_data

    def _dump(self, *args):
        self.dump_data = args[0]


if __name__ == '__main__':
    unittest.main()
