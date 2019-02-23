import math
from logging import getLogger
from operator import itemgetter
from typing import List, Tuple

import gensim

logger = getLogger(__name__)


class Tfidf(object):

    def __init__(self, dictionary: gensim.corpora.Dictionary, tokens: List[List[str]]) -> None:
        self.dictionary = dictionary
        self.tfidf = gensim.models.TfidfModel([dictionary.doc2bow(t) for t in tokens])

    def apply(self, tokens: List[List[str]], keep_top_rate: float = 1.0) -> List[List[Tuple[str, float]]]:
        bows = [self.dictionary.doc2bow(t) for t in tokens]
        index2token = dict(zip(self.dictionary.token2id.values(), self.dictionary.token2id.keys()))
        tfidf_values = [zip(*d) for d in self.tfidf[bows]]
        tfidf_values = [list(zip(map(index2token.get, indices), values)) for indices, values in tfidf_values]
        tfidf_values = [sorted(v, key=itemgetter(1), reverse=True) for v in tfidf_values]
        tfidf_values = [v[:int(math.ceil(len(v) * keep_top_rate) + 1)] for v in tfidf_values]
        return tfidf_values


