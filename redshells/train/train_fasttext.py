from logging import getLogger
from random import shuffle
from typing import Any, Dict

import gensim
import luigi

import gokart
import redshells

logger = getLogger(__name__)


class TrainFastText(gokart.TaskOnKart):
    task_namespace = 'redshells'
    tokenized_text_data_task = gokart.TaskInstanceParameter(
        description='The task outputs tokenized texts with type `List[List[str]]` or `List[str]` separated with space.')
    fasttext_kwargs = luigi.DictParameter(
        default=dict(),
        description='Arguments for FastText except "sentences". Please see gensim.models.FastText for more details.'
    )  # type: Dict[str, Any]
    output_file_path = luigi.Parameter(default='model/fasttext.zip')  # type: str

    def requires(self):
        return self.tokenized_text_data_task

    def output(self):
        return self.make_model_target(
            self.output_file_path, save_function=gensim.models.FastText.save, load_function=gensim.models.FastText.load)

    def run(self):
        texts = self.load()
        assert len(texts) > 0
        shuffle(texts)

        if isinstance(texts[0], str):
            texts = redshells.train.utils.TokenIterator(texts=texts)

        logger.info(f'training FastText...')
        model = gensim.models.FastText(sentences=texts, **self.fasttext_kwargs)
        self.dump(model)
