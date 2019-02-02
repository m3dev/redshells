import itertools

import luigi
import pandas as pd

import gokart
import redshells


class FindItemKeywordByMatching(gokart.TaskOnKart):
    """
    Find items which include keywords in its value of 'item_keyword_column_name'.
    Output pd.DataFrame with columns [item_id, keyword].
    """
    task_namespace = 'redshells.word_item_similarity'
    target_keyword_task = gokart.TaskInstanceParameter(
        description='A task outputs keywords as type `List[Any]` or `Set[Any]`.')
    item_task = gokart.TaskInstanceParameter(
        description='A task outputs item data as type `pd.DataFrame` which has `item_id_column_name`.')
    tfidf_task = gokart.TaskInstanceParameter(description='A task instance of TrainTfidf.')
    keep_top_rate = luigi.FloatParameter(description='A rate to filter words in texts.')  # type: float
    item_id_column_name = luigi.Parameter()  # type: str
    item_keyword_column_name = luigi.Parameter()  # type: str
    output_file_path = luigi.Parameter(
        default='app/word_item_similarity/find_item_by_keyword_matching.pkl')  # type: str

    def requires(self):
        return dict(keyword=self.target_keyword_task, item=self.item_task, tfidf=self.tfidf_task)

    def output(self):
        return self.make_target(self.output_file_path)

    def run(self):
        keywords = set(self.load('keyword'))
        items = self.load_data_frame('item', required_columns={self.item_id_column_name, self.item_keyword_column_name})
        tfidf = self.load('tfidf')  # type: redshells.model.Tfidf
        tokens = items[self.item_keyword_column_name].tolist()
        top_tokens = [list(zip(*values))[0] for values in tfidf.apply(tokens=tokens, keep_top_rate=self.keep_top_rate)]

        item_ids = items[self.item_id_column_name].tolist()
        match_keywords = [set(t) & keywords for t in top_tokens]
        result = pd.DataFrame(
            dict(
                item_id=list(
                    itertools.chain.from_iterable(
                        [[item_id] * len(keywords) for item_id, keywords in zip(item_ids, match_keywords)])),
                keyword=list(itertools.chain.from_iterable(match_keywords))))
        self.dump(result)
