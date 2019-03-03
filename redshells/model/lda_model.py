from logging import getLogger
from typing import List, Optional

import gensim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

logger = getLogger(__name__)


class LdaModel(object):
    """TopicModel is a kind of wrapper of LdaModel in gensim module.
    """

    def __init__(self,
                 n_topics: int,
                 chunksize: int = 16,
                 decay: float = 0.5,
                 offset: int = 16,
                 iterations: int = 3,
                 eta: float = 1.0e-16) -> None:
        self.n_topics = n_topics
        self.chunksize = chunksize
        self.decay = decay
        self.offset = offset
        self.iterations = iterations
        self.eta = eta
        self._lda = None  # type: gensim.models.LdaModel
        self.log_perplexity = 0

    def fit(self,
            texts: List[List[str]],
            adjust_passes=True,
            test_size=0.1,
            random_state=123,
            dictionary: Optional[gensim.corpora.Dictionary] = None) -> None:
        texts = shuffle(texts)
        dictionary = dictionary or self._make_dictionary(texts)
        corpus = self._make_corpus(texts=texts, dictionary=dictionary)
        train, test = train_test_split(corpus, test_size=test_size, random_state=random_state)
        passes = np.clip(int(round(100000 / (len(corpus) + 1))), 1, 20) if adjust_passes else 1
        self._lda = gensim.models.LdaModel(
            alpha='auto',
            corpus=train,
            num_topics=self.n_topics,
            id2word=dictionary,
            iterations=self.iterations,
            passes=passes)
        self.log_perplexity = self._lda.log_perplexity(test)
        logger.info('log_perplexity=%s', self.log_perplexity)

    def get_document_topics(self, texts: List[List[str]]) -> List[np.ndarray]:
        corpus = self._make_corpus(texts=texts, dictionary=self._lda.id2word)
        topics = []
        for c in corpus:
            topic = np.zeros(self._lda.num_topics)
            for (t, p) in self._lda.get_document_topics(c):
                topic[t] = p
            topics.append(topic)
        return topics

    def show_topics(self) -> str:
        out = ''
        for i in range(self._lda.num_topics):
            out += 'topic_%d [%f]: %s\n' % (i, self._lda.alpha[i], self._lda.print_topic(i, topn=30))
        return out

    @staticmethod
    def _make_dictionary(texts: List[List[str]]):
        dictionary = gensim.corpora.Dictionary(texts)
        return dictionary

    @staticmethod
    def _make_corpus(texts, dictionary):
        return [dictionary.doc2bow(t) for t in texts]
