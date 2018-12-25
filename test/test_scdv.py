import unittest

import gensim
import numpy as np

from redshells.model.scdv import SCDV


class TestSCDV(unittest.TestCase):
    def setUp(self):
        self._documents = [['a', 'b'], ['a', 'c', 'd', 'e'], ['a']]
        self._dictionary_filter_parameters = dict(no_below=0, no_above=0.5, keep_n=100000, keep_tokens=None)

    def test_build_word_embeddings(self):
        dictionary = gensim.corpora.Dictionary(self._documents)
        vocabulary_size = len(dictionary.token2id)
        embedding_size = 3
        word2vec = gensim.models.Word2Vec(self._documents, size=embedding_size, min_count=1)

        word_embeddings = SCDV._build_word_embeddings(dictionary=dictionary, w2v=word2vec)
        self.assertEqual(word_embeddings.shape, (vocabulary_size, embedding_size))

    def test_build_word_cluster_probabilities(self):
        vocabulary_size = 300
        embedding_size = 20
        cluster_size = 3

        np.random.seed(435)
        word_embeddings = np.random.uniform(size=(vocabulary_size, embedding_size))
        probabilities = SCDV._build_word_cluster_probabilities(
            word_embeddings=word_embeddings, cluster_size=cluster_size, gaussian_mixture_parameters=dict())
        self.assertEqual(probabilities.shape, (vocabulary_size, cluster_size))

    def test_build_idf(self):
        dictionary = gensim.corpora.Dictionary(self._documents)
        vocabulary_size = len(dictionary.token2id)
        idf = SCDV._build_idf(documents=self._documents, dictionary=dictionary)
        self.assertEqual(idf.shape, (vocabulary_size, ))

    def test_build_word_cluster_vectors(self):
        vocabulary_size = 30
        embedding_size = 5
        cluster_size = 3

        np.random.seed(678)
        word_embeddings = np.random.uniform(size=(vocabulary_size, embedding_size))
        cluster_probabilities = np.random.uniform(size=(vocabulary_size, cluster_size))

        wcv = SCDV._build_word_cluster_vectors(
            word_embeddings=word_embeddings, word_cluster_probabilities=cluster_probabilities)

        # Calculate expected values with a naive way.
        expected = np.zeros((vocabulary_size, cluster_size, embedding_size))
        for v in range(vocabulary_size):
            for c in range(cluster_size):
                expected[v][c] = cluster_probabilities[v][c] * word_embeddings[v]

        np.testing.assert_almost_equal(wcv, expected)

    def test_build_word_topic_vectors(self):
        vocabulary_size = 30
        embedding_size = 5
        cluster_size = 3

        np.random.seed(634)
        word_cluster_vectors = np.random.uniform(size=(vocabulary_size, cluster_size, embedding_size))
        idf = np.random.uniform(size=vocabulary_size)

        wtv = SCDV._build_word_topic_vectors(idf=idf, word_cluster_vectors=word_cluster_vectors)
        self.assertEqual(wtv.shape, (vocabulary_size, cluster_size * embedding_size))
        np.testing.assert_almost_equal(wtv[1], idf[1] * word_cluster_vectors[1].flatten())

    def test_build_document_vectors(self):
        dictionary = gensim.corpora.Dictionary(self._documents)
        vocabulary_size = len(dictionary.token2id)
        embedding_size = 5
        cluster_size = 3

        np.random.seed(342)
        word_topic_vectors = np.random.uniform(size=(vocabulary_size, cluster_size * embedding_size))

        dv = SCDV._build_document_vectors(
            word_topic_vectors=word_topic_vectors, dictionary=dictionary, documents=self._documents)
        self.assertEqual(dv.shape, (len(self._documents), cluster_size * embedding_size))

    def test_build_sparse_threshold(self):
        document_size = 7
        embedding_size = 5
        cluster_size = 3
        sparsity_percentage = 0.1

        np.random.seed(954)
        document_vectors = np.random.uniform(low=-1.0, size=(document_size, cluster_size * embedding_size))
        threshold = SCDV._build_sparsity_threshold(document_vectors, sparsity_percentage)
        # Calculate expected values with a naive way.
        average_max = np.average([np.max(document_vectors[d]) for d in range(document_size)])
        average_min = np.average([np.min(document_vectors[d]) for d in range(document_size)])
        expected = 0.5 * (np.abs(average_max) + np.abs(average_min)) * sparsity_percentage

        self.assertEqual(threshold, expected)

    def test_build_scdv_vectors(self):
        document_size = 5
        embedding_size = 3
        cluster_size = 2
        sparsity_threshold = 0.5

        np.random.seed(478)
        document_vectors = np.random.uniform(size=(document_size, cluster_size * embedding_size))
        scdv = SCDV._build_scdv_vectors(document_vectors, sparsity_threshold, l2_normalize=False)
        for original, sparse in zip(document_vectors.flatten(), scdv.flatten()):
            if np.abs(original) < sparsity_threshold:
                self.assertEqual(sparse, 0)
            else:
                self.assertEqual(sparse, original)

    def test_build_scdv_vectors_with_normalization(self):
        document_size = 5
        embedding_size = 3
        cluster_size = 2
        sparsity_threshold = 0.5

        np.random.seed(478)
        document_vectors = np.random.uniform(size=(document_size, cluster_size * embedding_size))
        scdv = SCDV._build_scdv_vectors(document_vectors, sparsity_threshold, l2_normalize=True)
        self.assertAlmostEqual(np.linalg.norm(scdv[0]), 1.0)


if __name__ == '__main__':
    unittest.main()
