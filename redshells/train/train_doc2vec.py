from random import shuffle
from typing import Any
from typing import Dict
from typing import List

import gensim
import gokart
import luigi


class TrainDoc2Vec(gokart.TaskOnKart):
    task_namespace = 'redshells'
    tokenized_text_data_task = gokart.TaskInstanceParameter(
        description='The task outputs tokenized texts with type "List[List[str]]".')
    output_file_path = luigi.Parameter(default='model/doc2vec.zip')  # type: str
    doc2vec_kwargs = luigi.DictParameter(
        default=dict(),
        description='Arguments for Doc2Vec except "documents". Please see gensim.models.Doc2Vec for more details.'
    )  # type: Dict[str, Any]

    def requires(self):
        return self.tokenized_text_data_task

    def output(self):
        return self.make_model_target(
            self.output_file_path, save_function=gensim.models.Doc2Vec.save, load_function=gensim.models.Doc2Vec.load)

    def run(self):
        texts = self.load()  # type: List[List[str]]
        shuffle(texts)
        documents = [gensim.models.doc2vec.TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]
        model = gensim.models.Doc2Vec(documents=documents, **self.doc2vec_kwargs)
        model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        self.dump(model)
