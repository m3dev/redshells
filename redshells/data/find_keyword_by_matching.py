import itertools

import luigi
import pandas as pd
import gokart


class FindKeywordByMatching(gokart.TaskOnKart):
    """
    Find items which include keywords in its value of 'item_keyword_column_name'.
    """
    task_namespace = 'redshells'
    target_keyword_task = gokart.TaskInstanceParameter(
        description='A task outputs item data as type `pd.DataFrame` which has columns.')
    item_task = gokart.TaskInstanceParameter(description='A task outputs keywords as type `List[Any]` or `Set[Any]`.')
    item_id_column_name = luigi.Parameter()  # type: str
    item_keyword_column_name = luigi.Parameter()  # type: str
    output_file_path = luigi.Parameter(default='data/search_by_keyword_matching.pkl')  # type: str

    def requires(self):
        return dict(keyword=self.target_keyword_task, item=self.item_task)

    def output(self):
        return self.make_target(self.output_file_path)

    def run(self):
        keywords = set(self.load('keyword'))
        items = self.load_data_frame('item', required_columns={self.item_id_column_name, self.item_keyword_column_name})
        item_ids = items[self.item_id_column_name].tolist()
        match_keywords = items[self.item_keyword_column_name].apply(lambda x: set(x) & keywords).tolist()

        result = pd.DataFrame(
            dict(
                item_id=list(
                    itertools.chain.from_iterable(
                        [[item_id] * len(keywords) for item_id, keywords in zip(item_ids, match_keywords)])),
                keyword=list(itertools.chain.from_iterable(match_keywords))))
        self.dump(result)
