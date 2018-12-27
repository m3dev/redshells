import luigi
from typing import List

from typing import Dict

import gokart


class ExtractColumnAsList(gokart.TaskOnKart):
    """
    Extract column data of pd.DataFrame as list.
    """
    task_namespace = 'redshells.data_frame_utils'
    data_task = gokart.TaskInstanceParameter(description='A task outputs pd.DataFrame.')
    column_name = luigi.Parameter()  # type: str
    drop_na = luigi.BoolParameter(default=False)
    output_file_path = luigi.Parameter(default='data/extract_column_as_list.pkl')  # type: str

    def requires(self):
        return self.data_task

    def output(self):
        return self.make_target(self.output_file_path)

    def run(self):
        data = self.load_data_frame(required_columns={self.column_name})
        if self.drop_na:
            data.dropna(subset={self.column_name}, inplace=True)
        self.dump(data[self.column_name].tolist())


class ExtractColumnAsDict(gokart.TaskOnKart):
    """
    Extract column data of pd.DataFrame as dict, and keep the first value when values of `key_column_name` are duplicate.
    """
    task_namespace = 'redshells.data_frame_utils'
    data_task = gokart.TaskInstanceParameter(description='A task outputs pd.DataFrame.')
    key_column_name = luigi.Parameter()  # type: str
    value_column_name = luigi.Parameter()  # type: str
    drop_na = luigi.BoolParameter(default=False)
    output_file_path = luigi.Parameter(default='data/extract_column_as_dict.pkl')  # type: str

    def requires(self):
        return self.data_task

    def output(self):
        return self.make_target(self.output_file_path)

    def run(self):
        data = self.load_data_frame(required_columns={self.key_column_name, self.value_column_name})
        if self.drop_na:
            data.dropna(subset={self.key_column_name, self.value_column_name}, inplace=True)
        data.drop_duplicates(self.key_column_name, keep='first', inplace=True)
        self.dump(dict(zip(data[self.key_column_name].tolist(), data[self.value_column_name].tolist())))


class FilterByColumn(gokart.TaskOnKart):
    """
    Filter pd.DataFrame by column names.
    """
    task_namespace = 'redshells.data_frame_utils'
    data_task = gokart.TaskInstanceParameter(description='A task outputs pd.DataFrame.')
    column_names = luigi.ListParameter()  # type: List[str]
    drop_na = luigi.BoolParameter(default=False)
    output_file_path = luigi.Parameter(default='data/filter_by_column.pkl')  # type: str

    def requires(self):
        return self.data_task

    def output(self):
        return self.make_target(self.output_file_path)

    def run(self):
        data = self.load_data_frame(required_columns=set(self.column_names))
        if self.drop_na:
            data.dropna(subset=set(self.column_names), inplace=True)
        self.dump(data[list(self.column_names)])


class RenameColumn(gokart.TaskOnKart):
    """
    Rename column names of pd.DataFrame.
    """
    task_namespace = 'redshells.data_frame_utils'
    data_task = gokart.TaskInstanceParameter(description='A task outputs pd.DataFrame.')
    rename_rule = luigi.DictParameter()  # type: Dict[str, str]
    drop_na = luigi.BoolParameter(default=False)
    output_file_path = luigi.Parameter(default='data/rename_column.pkl')  # type: str

    def requires(self):
        return self.data_task

    def output(self):
        return self.make_target(self.output_file_path)

    def run(self):
        column_names = set(list(self.rename_rule.keys()))
        data = self.load_data_frame(required_columns=column_names)
        if self.drop_na:
            data.dropna(subset=column_names, inplace=True)
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
    drop_duplicate = luigi.BoolParameter(default=False)
    output_file_path = luigi.Parameter(default='data/group_by_column_as_dict.pkl')  # type: str

    def requires(self):
        return self.data_task

    def output(self):
        return self.make_target(self.output_file_path)

    def run(self):
        data = self.load_data_frame(required_columns={self.key_column_name, self.value_column_name})
        data.dropna(subset={self.key_column_name, self.value_column_name}, inplace=True)
        if self.drop_duplicate:
            data.drop_duplicates(subset={self.key_column_name, self.value_column_name}, inplace=True)
        result = data.groupby(by=self.key_column_name)[self.value_column_name].apply(list).to_dict()
        self.dump(result)
