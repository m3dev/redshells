from typing import Dict, Any, List, Union

import itertools

import gensim
import numpy as np
import sklearn
from gensim.models import Word2Vec, TfidfModel, FastText
from sklearn.mixture import GaussianMixture
from logging import getLogger

logger = getLogger(__name__)


class SCDV(object):
    """ This is a model which is described in "SCDV : Sparse Composite Document Vectors using soft clustering over distributional representations"
    See https://arxiv.org/pdf/1612.06778.pdf for details

    """

    def __init__(self, documents: List[List[str]], cluster_size: int, sparsity_percentage: float,
                 gaussian_mixture_kwargs: Dict[Any, Any], dictionary: gensim.corpora.Dictionary,
                 w2v: Union[FastText, Word2Vec]) -> None:
        """

        :param documents: documents for training.
        :param cluster_size:  word cluster size.
        :param sparsity_percentage: sparsity percentage. This must be in [0, 1].
        :param gaussian_mixture_kwargs: Arguments to build `sklearn.mixture.GaussianMixture` except cluster_size. Please see `sklearn.mixture.GaussianMixture.__init__` for details.
        :param dictionary: `gensim.corpora.Dictionary`. 
        """
        logger.info('_build_dictionary...')
        self._dictionary = dictionary
        vocabulary_size = len(self._dictionary.token2id)
        embedding_size = w2v.wv.vector_size

        logger.info('_build_word_embeddings...')
        self._word_embeddings = self._build_word_embeddings(self._dictionary, w2v)
        assert self._word_embeddings.shape == (vocabulary_size, embedding_size)

        logger.info('_build_word_cluster_probabilities...')
        self._word_cluster_probabilities = self._build_word_cluster_probabilities(self._word_embeddings, cluster_size,
                                                                                  gaussian_mixture_kwargs)
        assert self._word_cluster_probabilities.shape == (vocabulary_size, cluster_size)

        logger.info('_build_idf...')
        self._idf = self._build_idf(self._dictionary)
        assert self._idf.shape == (vocabulary_size, )

        logger.info('_build_word_cluster_vectors...')
        word_cluster_vectors = self._build_word_cluster_vectors(self._word_embeddings, self._word_cluster_probabilities)
        assert word_cluster_vectors.shape == (vocabulary_size, cluster_size, embedding_size)

        logger.info('_build_word_topic_vectors...')
        word_topic_vectors = self._build_word_topic_vectors(self._idf, word_cluster_vectors)
        assert word_topic_vectors.shape == (vocabulary_size, (cluster_size * embedding_size))

        logger.info('_build_sparsity_threshold...')
        self._sparse_threshold = self._build_sparsity_threshold(word_topic_vectors, self._dictionary, documents,
                                                                sparsity_percentage)

    def infer_vector(self, new_documents: List[List[str]], l2_normalize: bool = True) -> np.ndarray:
        word_cluster_vectors = self._build_word_cluster_vectors(self._word_embeddings, self._word_cluster_probabilities)
        word_topic_vectors = self._build_word_topic_vectors(self._idf, word_cluster_vectors)
        document_vectors = self._build_document_vectors(word_topic_vectors, self._dictionary, new_documents)
        return self._build_scdv_vectors(document_vectors, self._sparse_threshold, l2_normalize)

    @staticmethod
    def _build_word_embeddings(dictionary: gensim.corpora.Dictionary, w2v: Union[FastText, Word2Vec]) -> np.ndarray:
        embeddings = np.zeros((len(dictionary.token2id), w2v.vector_size))
        for token, idx in dictionary.token2id.items():
            if token in w2v.wv:
                embeddings[idx] = w2v.wv[token]
        return sklearn.preprocessing.normalize(embeddings, axis=1, norm='l2')

    @staticmethod
    def _build_word_cluster_probabilities(word_embeddings: np.ndarray, cluster_size: int,
                                          gaussian_mixture_parameters: Dict[Any, Any]) -> np.ndarray:
        gm = GaussianMixture(n_components=cluster_size, **gaussian_mixture_parameters)
        gm.fit(word_embeddings)
        return gm.predict_proba(word_embeddings)

    @staticmethod
    def _build_idf(dictionary: gensim.corpora.Dictionary) -> np.ndarray:
        model = TfidfModel(dictionary=dictionary)
        idf = np.zeros(len(dictionary.token2id))
        for idx, value in model.idfs.items():
            idf[idx] = value
        return idf

    @staticmethod
    def _build_word_cluster_vectors(word_embeddings: np.ndarray, word_cluster_probabilities: np.ndarray) -> np.ndarray:
        vocabulary_size, embedding_size = word_embeddings.shape
        cluster_size = word_cluster_probabilities.shape[1]
        assert vocabulary_size == word_cluster_probabilities.shape[0]

        wcv = np.zeros((vocabulary_size, cluster_size, embedding_size))
        wcp = word_cluster_probabilities
        for v, c in itertools.product(range(vocabulary_size), range(cluster_size)):
            wcv[v][c] = wcp[v][c] * word_embeddings[v]
        return wcv

    @staticmethod
    def _build_word_topic_vectors(idf: np.ndarray, word_cluster_vectors: np.ndarray) -> np.ndarray:
        vocabulary_size, cluster_size, embedding_size = word_cluster_vectors.shape
        assert vocabulary_size == idf.shape[0]

        wtv = np.zeros((vocabulary_size, cluster_size * embedding_size))
        for v in range(vocabulary_size):
            wtv[v] = idf[v] * word_cluster_vectors[v].flatten()
        return wtv

    @staticmethod
    def _build_document_vectors(word_topic_vectors: np.ndarray, dictionary: gensim.corpora.Dictionary,
                                documents: List[List[str]]) -> np.ndarray:
        data = np.array([SCDV._calculate_document_vector(word_topic_vectors, dictionary, d) for d in documents])
        return data

    @staticmethod
    def _calculate_document_vector(word_topic_vectors: np.ndarray, dictionary: gensim.corpora.Dictionary,
                                   document: List[str]):
        doc2bow = dictionary.doc2bow(document)
        if len(doc2bow) > 0:
            return np.sum([word_topic_vectors[idx] * count for idx, count in doc2bow], axis=0)
        return np.zeros(word_topic_vectors.shape[1])

    @staticmethod
    def _build_sparsity_threshold(word_topic_vectors: np.ndarray, dictionary: gensim.corpora.Dictionary,
                                  documents: List[List[str]], sparsity_percentage) -> float:
        # To reduce memory usage, I use generator.
        document_vectors = (SCDV._calculate_document_vector(word_topic_vectors, dictionary, d) for d in documents)
        min_max_averages = (0.5 * (np.abs(np.min(d)) + np.abs(np.max(d))) for d in document_vectors)
        average = np.mean(list(min_max_averages))
        return sparsity_percentage * average

    @staticmethod
    def _build_scdv_vectors(document_vectors: np.ndarray, sparsity_threshold: float, l2_normalize: bool) -> np.ndarray:
        close_to_zero = np.abs(document_vectors) < sparsity_threshold
        document_vectors[close_to_zero] = 0.0
        if not l2_normalize:
            return document_vectors

        return sklearn.preprocessing.normalize(document_vectors, axis=1, norm='l2')
