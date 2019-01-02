import unittest
from unittest.mock import MagicMock

import luigi
import pandas as pd

import redshells.app.word_item_similarity


class _DummyTask(luigi.Task):
    pass


class FindKeywordByMatchingTest(unittest.TestCase):
    def setUp(self):
        self.input_data = dict()
        self.dump_data = None
        redshells.app.word_item_similarity.FindItemKeywordByMatching.clear_instance_cache()

    def test_run(self):
        self.input_data['keyword'] = ['a', 'b']
        self.input_data['item'] = pd.DataFrame(
            dict(item_id=['i1', 'i2'], item_keyword=[['a', 'b', 'c'], ['b', 'c', 'd']]))

        task = redshells.app.word_item_similarity.FindItemKeywordByMatching(
            target_keyword_task=_DummyTask(),
            item_task=_DummyTask(),
            item_id_column_name='item_id',
            item_keyword_column_name='item_keyword')
        task.load = MagicMock(side_effect=self._load)
        task.dump = MagicMock(side_effect=self._dump)

        task.run()
        pd.testing.assert_frame_equal(
            self.dump_data, pd.DataFrame(dict(item_id=['i1', 'i1', 'i2'], keyword=['a', 'b', 'b'])),
            check_like=True)  # ignore order

    def _load(self, *args, **kwargs):
        if 'target' in kwargs and kwargs['target'] is not None:
            return self.input_data.get(kwargs['target'], None)
        if len(args) > 0:
            return self.input_data.get(args[0], None)
        return self.input_data

    def _dump(self, *args):
        self.dump_data = args[0]


if __name__ == '__main__':
    unittest.main()
