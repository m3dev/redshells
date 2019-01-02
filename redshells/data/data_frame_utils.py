import luigi
from typing import List
from typing import Dict

import pandas as pd
import sklearn

import gokart


class ExtractColumnAsList(gokart.TaskOnKart):
    """
    Extract column data of pd.DataFrame as list.
    """
    task_namespace = 'redshells.data_frame_utils'
    data_task = gokart.TaskInstanceParameter(description='A task outputs pd.DataFrame.')
    column_name = luigi.Parameter()  # type: str
    output_file_path = luigi.Parameter(default='data/extract_column_as_list.pkl')  # type: str

    def requires(self):
        return self.data_task

    def output(self):
        return self.make_target(self.output_file_path)

    def run(self):
        data = self.load_data_frame(required_columns={self.column_name})
        self.dump(data[self.column_name].tolist())


class ExtractColumnAsDict(gokart.TaskOnKart):
    """
    Extract column data of pd.DataFrame as dict, and keep the first value when values of `key_column_name` are duplicate.
    """
    task_namespace = 'redshells.data_frame_utils'
    data_task = gokart.TaskInstanceParameter(description='A task outputs pd.DataFrame.')
    key_column_name = luigi.Parameter()  # type: str
    value_column_name = luigi.Parameter()  # type: str
    output_file_path = luigi.Parameter(default='data/extract_column_as_dict.pkl')  # type: str

    def requires(self):
        return self.data_task

    def output(self):
        return self.make_target(self.output_file_path)

    def run(self):
        data = self.load_data_frame(required_columns={self.key_column_name, self.value_column_name})
        data.drop_duplicates(self.key_column_name, keep='first', inplace=True)
        self.dump(dict(zip(data[self.key_column_name].tolist(), data[self.value_column_name].tolist())))


class FilterByColumn(gokart.TaskOnKart):
    """
    Filter pd.DataFrame by column names.
    """
    task_namespace = 'redshells.data_frame_utils'
    data_task = gokart.TaskInstanceParameter(description='A task outputs pd.DataFrame.')
    column_names = luigi.ListParameter()  # type: List[str]
    output_file_path = luigi.Parameter(default='data/filter_by_column.pkl')  # type: str

    def requires(self):
        return self.data_task

    def output(self):
        return self.make_target(self.output_file_path)

    def run(self):
        data = self.load_data_frame(required_columns=set(self.column_names))
        self.dump(data[list(self.column_names)])


class RenameColumn(gokart.TaskOnKart):
    """
    Rename column names of pd.DataFrame.
    """
    task_namespace = 'redshells.data_frame_utils'
    data_task = gokart.TaskInstanceParameter(description='A task outputs pd.DataFrame.')
    rename_rule = luigi.DictParameter()  # type: Dict[str, str]
    output_file_path = luigi.Parameter(default='data/rename_column.pkl')  # type: str

    def requires(self):
        return self.data_task

    def output(self):
        return self.make_target(self.output_file_path)

    def run(self):
        column_names = set(list(self.rename_rule.keys()))
        data = self.load_data_frame(required_columns=column_names)
        self.dump(data.rename(columns=dict(self.rename_rule)))


class GroupByColumnAsDict(gokart.TaskOnKart):
    """
    Group by column names of pd.DataFrame and return map from `key_column_name` to a list of `value_column_name`.
    
    **This always drops na values.**
    """
    task_namespace = 'redshells.data_frame_utils'
    data_task = gokart.TaskInstanceParameter(description='A task outputs pd.DataFrame.')
    key_column_name = luigi.Parameter()  # type: str
    value_column_name = luigi.Parameter()  # type: str
    output_file_path = luigi.Parameter(default='data/group_by_column_as_dict.pkl')  # type: str

    def requires(self):
        return self.data_task

    def output(self):
        return self.make_target(self.output_file_path)

    def run(self):
        data = self.load_data_frame(required_columns={self.key_column_name, self.value_column_name})
        data.dropna(subset={self.key_column_name, self.value_column_name}, inplace=True)
        result = data.groupby(by=self.key_column_name)[self.value_column_name].apply(list).to_dict()
        self.dump(result)


class ConvertToOneHot(gokart.TaskOnKart):
    """
    Convert column values of `categorical_column_names` to one-hot.
    """
    task_namespace = 'redshells.data_frame_utils'
    data_task = gokart.TaskInstanceParameter(description='A task outputs pd.DataFrame.')
    categorical_column_names = luigi.ListParameter()  # type: List[str]
    output_file_path = luigi.Parameter(default='data/group_by_column_as_dict.pkl')  # type: str

    def requires(self):
        return self.data_task

    def output(self):
        return self.make_target(self.output_file_path)

    def run(self):
        categorical_column_names = list(self.categorical_column_names)
        data = self.load_data_frame(required_columns=set(categorical_column_names))
        result = pd.get_dummies(data[categorical_column_names])
        result = result.merge(data.drop(categorical_column_names, axis=1), left_index=True, right_index=True)
        self.dump(result)


class ConvertTypeToCategory(gokart.TaskOnKart):
    """
    Convert column types to 'category'.
    """
    task_namespace = 'redshells.data_frame_utils'
    data_task = gokart.TaskInstanceParameter(description='A task outputs pd.DataFrame.')
    categorical_column_names = luigi.ListParameter()  # type: List[str]
    output_file_path = luigi.Parameter(default='data/group_by_column_as_dict.pkl')  # type: str

    def requires(self):
        return self.data_task

    def output(self):
        return self.make_target(self.output_file_path)

    def run(self):
        categorical_column_names = list(self.categorical_column_names)
        data = self.load_data_frame(required_columns=set(categorical_column_names))
        for c in self.categorical_column_names:
            data[c] = data[c].astype('category')
        self.dump(data)


class SplitTrainTestData(gokart.TaskOnKart):
    task_namespace = 'redshells.data_frame_utils'
    data_task = gokart.TaskInstanceParameter()
    test_size_rate = luigi.FloatParameter()
    train_output_file_path = luigi.Parameter(default='data/train_data.pkl')  # type: str
    test_output_file_path = luigi.Parameter(default='data/test_data.pkl')  # type: str

    def requires(self):
        return self.data_task

    def output(self):
        return dict(
            train=self.make_target(self.train_output_file_path), test=self.make_target(self.test_output_file_path))

    def run(self):
        data = self.load_data_frame()
        data = sklearn.utils.shuffle(data)
        train, test = sklearn.model_selection.train_test_split(data, test_size=self.test_size_rate)
        self.dump(train, 'train')
        self.dump(test, 'test')


class SampleData(gokart.TaskOnKart):
    task_namespace = 'redshells.data_frame_utils'
    data_task = gokart.TaskInstanceParameter()
    sample_size = luigi.IntParameter()
    output_file_path = luigi.Parameter(default='data/sample_data.pkl')  # type: str

    def requires(self):
        return self.data_task

    def output(self):
        return self.make_target(self.output_file_path)

    def run(self):
        data = self.load_data_frame()
        data = sklearn.utils.shuffle(data)
        self.dump(data.head(n=self.sample_size))
