from logging import getLogger
from typing import List

logger = getLogger(__name__)


class TokenIterator(object):
    def __init__(self, texts: List[str]) -> None:
        self.texts = texts
        self.i = 0

    def __iter__(self):
        return TokenIterator(texts=self.texts)

    def __next__(self):
        if self.i == len(self.texts):
            self.i = 0
            raise StopIteration()
        value = self.texts[self.i].split(' ')
        self.i += 1
        return value
