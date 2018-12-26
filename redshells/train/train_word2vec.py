from random import shuffle
from typing import Any
from typing import Dict
from typing import List

import gensim
import gokart
import luigi


class TrainWord2Vec(gokart.TaskOnKart):
    task_namespace = 'redshells'
    tokenized_text_data_task = gokart.TaskInstanceParameter(
        description='The task outputs tokenized texts with type "List[List[str]]".')
    output_file_path = luigi.Parameter(default='model/word2vec.zip')  # type: str
    word2vec_kwargs = luigi.DictParameter(
        default=dict(),
        description='Arguments for Word2Vec except "sentences". Please see gensim.models.Word2Vec for more details.'
    )  # type: Dict[str, Any]

    def requires(self):
        return self.tokenized_text_data_task

    def output(self):
        return self.make_model_target(
            self.output_file_path, save_function=gensim.models.Word2Vec.save, load_function=gensim.models.Word2Vec.load)

    def run(self):
        texts = self.load()  # type: List[List[str]]
        shuffle(texts)
        model = gensim.models.Word2Vec(sentences=texts, **self.word2vec_kwargs)
        self.dump(model)
