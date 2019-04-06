import os
import numpy as np
import pandas as pd
import tensorflow as tf
from logging import getLogger, config

logger = getLogger(__name__)


def main():
    x = np.array([0, 1, 2])
    y = np.array([3, 4])
    z = np.concatenate([x, y])
    print(z)


if __name__ == '__main__':
    main()
