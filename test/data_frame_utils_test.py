import unittest
from unittest.mock import MagicMock

import luigi
import pandas as pd

import redshells


class _DummyTask(luigi.Task):
    pass


class TrainPairwiseSimilarityModelTest(unittest.TestCase):
    def setUp(self):
        self.input_data = None
        self.dump_data = None

    def test_extract_column_as_list(self):
        self.input_data = pd.DataFrame(dict(a=['a', 'b', None], b=['A', 'B', 'C']))
        task = redshells.data.data_frame_utils.ExtractColumnAsList(data_task=_DummyTask(), column_name='a')
        task.load = MagicMock(side_effect=self._load)
        task.dump = MagicMock(side_effect=self._dump)

        task.run()
        self.assertEqual(self.dump_data, ['a', 'b', None])

    def test_extract_column_as_list_with_dropna(self):
        self.input_data = pd.DataFrame(dict(a=['a', 'b', None], b=['A', 'B', 'C']))
        task = redshells.data.data_frame_utils.ExtractColumnAsList(
            data_task=_DummyTask(), column_name='a', drop_na=True)
        task.load = MagicMock(side_effect=self._load)
        task.dump = MagicMock(side_effect=self._dump)

        task.run()
        self.assertEqual(self.dump_data, ['a', 'b'])

    def test_extract_column_as_dict(self):
        self.input_data = pd.DataFrame(dict(a=['a', 'b', None], b=['A', 'B', 'C']))
        task = redshells.data.data_frame_utils.ExtractColumnAsDict(
            data_task=_DummyTask(), key_column_name='a', value_column_name='b')
        task.load = MagicMock(side_effect=self._load)
        task.dump = MagicMock(side_effect=self._dump)

        task.run()
        self.assertEqual(self.dump_data, {'a': 'A', 'b': 'B', None: 'C'})

    def test_extract_column_as_dict_with_duplicate(self):
        self.input_data = pd.DataFrame(dict(a=['a', 'b', 'b'], b=['A', 'B', 'C']))
        task = redshells.data.data_frame_utils.ExtractColumnAsDict(
            data_task=_DummyTask(), key_column_name='a', value_column_name='b')
        task.load = MagicMock(side_effect=self._load)
        task.dump = MagicMock(side_effect=self._dump)

        task.run()
        self.assertEqual(self.dump_data, {'a': 'A', 'b': 'B'})

    def test_extract_column_as_dict_with_dropna(self):
        self.input_data = pd.DataFrame(dict(a=['a', 'b', None], b=['A', 'B', 'C']))
        task = redshells.data.data_frame_utils.ExtractColumnAsDict(
            data_task=_DummyTask(), key_column_name='a', value_column_name='b', drop_na=True)
        task.load = MagicMock(side_effect=self._load)
        task.dump = MagicMock(side_effect=self._dump)

        task.run()
        self.assertEqual(self.dump_data, {'a': 'A', 'b': 'B'})

    def test_filter_by_column(self):
        self.input_data = pd.DataFrame(dict(a=['a', 'b', None], b=['A', 'B', 'C']))
        task = redshells.data.data_frame_utils.FilterByColumn(data_task=_DummyTask(), column_names=['a'])
        task.load = MagicMock(side_effect=self._load)
        task.dump = MagicMock(side_effect=self._dump)

        task.run()
        pd.testing.assert_frame_equal(self.dump_data, pd.DataFrame(dict(a=['a', 'b', None])))

    def test_filter_by_column_with_dropna(self):
        self.input_data = pd.DataFrame(dict(a=['a', 'b', None], b=['A', 'B', 'C']))
        task = redshells.data.data_frame_utils.FilterByColumn(data_task=_DummyTask(), column_names=['a'], drop_na=True)
        task.load = MagicMock(side_effect=self._load)
        task.dump = MagicMock(side_effect=self._dump)

        task.run()
        pd.testing.assert_frame_equal(self.dump_data, pd.DataFrame(dict(a=['a', 'b'])))

    def test_rename_column(self):
        self.input_data = pd.DataFrame(dict(a=['a', 'b', None], b=['A', 'B', 'C']))
        task = redshells.data.data_frame_utils.RenameColumn(data_task=_DummyTask(), rename_rule={'a': 'A'})
        task.load = MagicMock(side_effect=self._load)
        task.dump = MagicMock(side_effect=self._dump)

        task.run()
        pd.testing.assert_frame_equal(self.dump_data, pd.DataFrame(dict(A=['a', 'b', None], b=['A', 'B', 'C'])))

    def test_rename_column_with_dropna(self):
        self.input_data = pd.DataFrame(dict(a=['a', 'b', None], b=['A', 'B', 'C']))
        task = redshells.data.data_frame_utils.RenameColumn(
            data_task=_DummyTask(), rename_rule={'a': 'A'}, drop_na=True)
        task.load = MagicMock(side_effect=self._load)
        task.dump = MagicMock(side_effect=self._dump)

        task.run()
        pd.testing.assert_frame_equal(self.dump_data, pd.DataFrame(dict(A=['a', 'b'], b=['A', 'B'])))

    def test_group_by_column_as_dict(self):
        self.input_data = pd.DataFrame(dict(a=['a', 'b', 'b', 'a', 'b'], b=['A', 'B', None, 'D', 'B']))
        task = redshells.data.data_frame_utils.GroupByColumnAsDict(
            data_task=_DummyTask(), key_column_name='a', value_column_name='b')
        task.load = MagicMock(side_effect=self._load)
        task.dump = MagicMock(side_effect=self._dump)

        task.run()
        self.assertEqual(self.dump_data, {'a': ['A', 'D'], 'b': ['B', 'B']})

    def test_group_by_column_as_dict_with_drop_duplicate(self):
        self.input_data = pd.DataFrame(dict(a=['a', 'b', 'b', 'a', 'b'], b=['A', 'B', None, 'D', 'B']))
        task = redshells.data.data_frame_utils.GroupByColumnAsDict(
            data_task=_DummyTask(), key_column_name='a', value_column_name='b', drop_duplicate=True)
        task.load = MagicMock(side_effect=self._load)
        task.dump = MagicMock(side_effect=self._dump)

        task.run()
        self.assertEqual(self.dump_data, {'a': ['A', 'D'], 'b': ['B']})

    def _load(self, *args, **kwargs):
        return self.input_data.copy()

    def _dump(self, *args):
        self.dump_data = args[0]


if __name__ == '__main__':
    unittest.main()
