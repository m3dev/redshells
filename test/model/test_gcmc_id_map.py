import unittest
from logging import getLogger
import numpy as np
from redshells.model.gcmc_dataset import GcmcIdMap

logger = getLogger(__name__)


class TestGcmcIdMap(unittest.TestCase):
    def test_initialize(self):
        ids = np.array([0, 0, 1, 2, 3, 3])
        features = [{0: np.array([0]), 1: np.array([1])}]
        id_map = GcmcIdMap(ids=ids, features=features, min_count=2, use_default=True)
        # because min_count is 2 and the number of index 1 and 2 are less than 2.
        np.testing.assert_equal([1, 1, 0, 0, 2, 2], id_map.indices)
        np.testing.assert_equal([1, 1, 3, 4, 2, 2], id_map.feature_indices)

    def test_add(self):
        ids = np.array([0, 0, 1, 2, 3, 3])
        features = [{0: np.array([0]), 1: np.array([1])}]
        id_map = GcmcIdMap(ids=ids, features=features, min_count=2, use_default=True)
        additional_ids = np.array([2, 3, 4])
        additional_features = [{2: np.array([2]), 4: np.array([4])}]
        id_map.add(additional_ids, features=additional_features)
        # because min_count is 2 and the number of index 1 and 2 are less than 2, and in "add" min_count is ignored
        np.testing.assert_equal([0, 0, 1, 2, 3, 3, 2, 3, 4], id_map.ids)
        np.testing.assert_equal([1, 1, 0, 0, 2, 2, 0, 2, 3], id_map.indices)
        np.testing.assert_equal([1, 1, 3, 4, 2, 2, 4, 2, 5], id_map.feature_indices)
        self.assertEqual([{0: np.array([0]), 1: np.array([1]), 2: np.array([2]), 4: np.array([4])}], id_map.features)
        np.testing.assert_equal(id_map._sort_features(id_map.features, id_map._id2feature_index), id_map.feature_matrix)


if __name__ == '__main__':
    unittest.main()
