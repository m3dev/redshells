import os
import numpy as np
import pandas as pd
import tensorflow as tf
from logging import getLogger, config

logger = getLogger(__name__)


def main():
    t = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    x = tf.slice(t, [0, 0], [2, 2])  # [[[3, 3, 3]]]
    with tf.Session():
        print(x.eval())


if __name__ == '__main__':
    main()
