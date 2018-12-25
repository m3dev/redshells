from random import shuffle
from typing import Any
from typing import Dict
from typing import List

import gensim
import gokart
import luigi


class TrainFastText(gokart.TaskOnKart):
    tokenized_text_data_task = gokart.TaskInstanceParameter(
        description='The task outputs tokenized texts with type "List[List[str]]".')
    output_file_path = luigi.Parameter(default='model/fasttext.zip')  # type: str
    fasttext_kwargs = luigi.DictParameter(
        default=dict(),
        description='Arguments for FastText except "sentences". Please see gensim.models.FastText for more details.'
    )  # type: Dict[str, Any]

    def requires(self):
        return self.tokenized_text_data_task

    def output(self):
        return self.make_model_target(
            self.output_file_path, save_function=gensim.models.FastText.save, load_function=gensim.models.FastText.load)

    def run(self):
        texts = self.load()  # type: List[List[str]]
        shuffle(texts)
        model = gensim.models.FastText(sentences=texts, **self.fasttext_kwargs)
        self.dump(model)
