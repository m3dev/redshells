import random
import string
import unittest
from logging import getLogger

import gensim

from redshells.model import Tfidf

logger = getLogger(__name__)


class TestTfidf(unittest.TestCase):
    def test_apply_with_empty(self):
        texts = [random.choices(string.ascii_letters, k=100) for _ in range(100)]
        dictionary = gensim.corpora.Dictionary(texts)
        model = Tfidf(dictionary, texts)
        results = model.apply(texts + [[]])
        self.assertEqual([], results[-1])


if __name__ == '__main__':
    unittest.main()
