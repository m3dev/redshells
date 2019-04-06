import unittest
from logging import getLogger

import numpy as np

from redshells.model.gcmc_dataset import GcmcGraphDataset, GcmcDataset

logger = getLogger(__name__)


class TestGcmcGraphDataset(unittest.TestCase):
    def setUp(self) -> None:
        dataset = GcmcDataset(user_ids=np.array([0, 1, 2]), item_ids=np.array([10, 11, 12]), ratings=np.array([100, 101, 102]))
        self.graph_dataset = GcmcGraphDataset(dataset=dataset, test_size=0.1)
        self.additional_dataset = GcmcDataset(user_ids=np.array([1, 2, 3]), item_ids=np.array([13, 14, 15]), ratings=np.array([103, 101, 102]))

    def test(self):
        new_dataset = self.graph_dataset.add_dataset(self.additional_dataset, add_user=True, add_item=True, add_rating=True)
        self.assertEqual({0, 1, 2, 3}, set(new_dataset.user_ids))
        self.assertEqual({10, 11, 12, 13, 14, 15}, set(new_dataset.item_ids))
        self.assertEqual({100, 101, 102, 103}, set(new_dataset.rating()))

    def test_only_item(self):
        new_dataset = self.graph_dataset.add_dataset(self.additional_dataset, add_item=True)
        self.assertEqual({0, 1, 2}, set(new_dataset.user_ids))
        self.assertEqual({10, 11, 12, 14}, set(new_dataset.item_ids))
        self.assertEqual({100, 101, 102}, set(new_dataset.rating()))


if __name__ == '__main__':
    unittest.main()
