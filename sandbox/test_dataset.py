import itertools
from builtins import sorted
from collections import Counter
from logging import getLogger
from typing import List, Optional, Dict, Tuple, Any

import numpy as np
import scipy.sparse as sp
import sklearn
import tensorflow as tf
import pandas as pd
import redshells
from redshells.model.early_stopping import EarlyStopping
from redshells.model.gcmc_dataset import GcmcDataset
from redshells.model.graph_convolutional_matrix_completion import GCMCDataset

logger = getLogger(__name__)


def _make_sparse_matrix(n, m, n_values):
    x = np.zeros(shape=(n, m), dtype=np.float32)
    x[np.random.choice(range(n), n_values), np.random.choice(range(m), n_values)] = 1.0
    return sp.csr_matrix(x)


def main():
    np.random.seed(12)
    n_users = 101
    n_items = 233
    n_data = 3007
    n_features = 21
    test_size = 0.2
    adjacency_matrix = _make_sparse_matrix(n_users, n_items, n_data) + 2 * _make_sparse_matrix(n_users, n_items, n_data)
    user_ids = adjacency_matrix.tocoo().row
    item_ids = adjacency_matrix.tocoo().col
    ratings = adjacency_matrix.tocoo().data
    item_features = dict(zip(range(n_items), np.random.uniform(size=(n_items, n_features))))

    np.random.seed(34)
    dataset0 = GCMCDataset(
        user_ids, item_ids, ratings, test_size, user_information=None, item_information=item_features)

    np.random.seed(34)
    dataset1 = GcmcDataset(
        user_ids, item_ids, ratings, test_size, user_information=None, item_information=item_features)

    import IPython
    IPython.embed()
    dataset0.user2index
    dataset1.user_id_map.id2index

    dataset0.item2index
    dataset1.item_id_map.id2index

    (dataset0.item_indices + 1 - dataset1.item_indices).max()
    dataset1.user_indices

    (dataset0.rating_indices - dataset1.rating_indices).max()

    (dataset0.item_indices + 1 - dataset1.item_information_indices).max()

    (dataset0.ratings - dataset1.ratings).max()

    (dataset0.train_indices.astype(int) - dataset1.train_indices.astype(int)).max()

    dataset0.item_information
    dataset1.item_information


if __name__ == '__main__':
    main()
