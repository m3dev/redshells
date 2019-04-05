from collections import Counter
from logging import getLogger
from typing import Optional, Dict, Any, Tuple, List, NamedTuple

import numpy as np
import scipy.sparse as sp
import sklearn
import sys

logger = getLogger(__name__)


class GcmcRatingData(NamedTuple):
    user_ids: np.ndarray
    item_ids: np.ndarray
    ratings: np.ndarray


class IdMap(object):
    def __init__(self, ids: np.ndarray, min_count=0, max_count=sys.maxsize, use_default: bool = True) -> None:
        id_count = dict(Counter(ids))
        in_ids = sorted([i for i, c in id_count.items() if min_count <= c <= max_count])
        out_ids = sorted(list(set(id_count.keys()) - set(in_ids)))

        if use_default:
            self._default_index = 0
            start = 1
        else:
            self._default_index = None
            start = 0

        self._id2index = self._make_map(in_ids, start=start)
        self._id2information_index = self._make_map(in_ids + out_ids, start=start)

    @staticmethod
    def _make_map(xs: List, start: int = 0) -> Dict:
        return dict(zip(xs, range(start, start + len(xs))))

    def to_indices(self, ids: Any) -> np.ndarray:
        return np.array([self._id2index.get(i, self._default_index) for i in ids])

    def to_information_indices(self, ids: Any) -> np.ndarray:
        return np.array([self._id2information_index.get(i, self._default_index) for i in ids])

    @property
    def id2index(self) -> Dict:
        return self._id2index

    @property
    def id2information_index(self) -> Dict:
        return self._id2information_index

    @property
    def index_count(self) -> int:
        return max(self._id2index.values()) + 1


class GcmcDataset(object):
    def __init__(self,
                 rating_data: GcmcRatingData,
                 test_size: float,
                 user_information: Optional[List[Dict[Any, np.ndarray]]] = None,
                 item_information: Optional[List[Dict[Any, np.ndarray]]] = None,
                 min_user_click_count: int = 0,
                 max_user_click_count: int = sys.maxsize) -> None:
        user_ids = rating_data.user_ids
        item_ids = rating_data.item_ids
        ratings = rating_data.ratings
        self.user_id_map = IdMap(user_ids, min_count=min_user_click_count, max_count=max_user_click_count)
        self.item_id_map = IdMap(item_ids)
        self.rating_id_map = IdMap(ratings, use_default=False)
        self._user_indices = self.user_id_map.to_indices(user_ids)
        self._item_indices = self.item_id_map.to_indices(item_ids)
        self._rating_indices = self.rating_id_map.to_indices(ratings)
        self.user_information = self._sort_features(
            features=user_information, order_map=self.user_id_map.id2information_index)
        self.item_information = self._sort_features(
            features=item_information, order_map=self.item_id_map.id2information_index)
        self.user_information_indices = self.user_id_map.to_information_indices(user_ids)
        self.item_information_indices = self.item_id_map.to_information_indices(item_ids)
        self.ratings = ratings
        self.train_indices = np.random.uniform(0., 1., size=len(user_ids)) > test_size

    def _train_adjacency_matrix(self):
        m = sp.csr_matrix((self.user_id_map.index_count, self.item_id_map.index_count), dtype=np.float32)
        idx = self.train_indices
        # add 1 to rating_indices, because rating_indices starts with 0 and 0 is ignored in scr_matrix
        m[self._user_indices[idx], self._item_indices[idx]] = self._rating_indices[idx] + 1.
        return m

    def train_rating_adjacency_matrix(self) -> List[sp.csr_matrix]:
        adjacency_matrix = self._train_adjacency_matrix()
        return [
            sp.csr_matrix(adjacency_matrix == r + 1., dtype=np.float32) for r in range(self.rating_id_map.index_count)
        ]

    def add_ratings(self, additional_rating_data: GcmcRatingData) -> 'GcmcDataset':

        pass

    def add_item_features(self, additional_item_features: List[Dict[Any, np.ndarray]]) -> 'GcmcDataset':
        pass

    def train_data(self):
        idx = self.train_indices
        shuffle_idx = sklearn.utils.shuffle(list(range(int(np.sum(idx)))))
        data = dict()
        data['user'] = self._user_indices[idx][shuffle_idx]
        data['item'] = self._item_indices[idx][shuffle_idx]
        data['label'] = self._to_one_hot(self._rating_indices[idx][shuffle_idx])
        data['rating'] = self.ratings[idx][shuffle_idx]
        data['user_information'] = self.user_information_indices[idx][shuffle_idx]
        data['item_information'] = self.item_information_indices[idx][shuffle_idx]
        return data

    def to_indices(self, user_ids: List, item_ids: List) -> Tuple[np.ndarray, np.ndarray]:
        return self.user_id_map.to_indices(user_ids), self.item_id_map.to_indices(item_ids)

    def to_information_indices(self, user_ids: List, item_ids: List) -> Tuple[np.ndarray, np.ndarray]:
        return self.user_id_map.to_information_indices(user_ids), self.item_id_map.to_information_indices(item_ids)

    def test_data(self):
        idx = ~self.train_indices
        data = dict()
        data['user'] = self._user_indices[idx]
        data['item'] = self._item_indices[idx]
        data['label'] = self._to_one_hot(self._rating_indices[idx])
        data['rating'] = self.ratings[idx]
        data['user_information'] = self.user_information_indices[idx]
        data['item_information'] = self.item_information_indices[idx]
        return data

    def rating(self):
        return np.array(sorted(self.rating_id_map.id2index.keys()))

    def _to_one_hot(self, ratings):
        return np.eye(self.rating_id_map.index_count)[ratings]

    @staticmethod
    def _sort_features_impl(features: Dict[Any, np.ndarray], order_map: Dict) -> np.ndarray:
        def _get_feature_size(values):
            for v in (v for v in values if v is not None):
                return len(v)
            return 0

        feature_size = _get_feature_size(features.values())
        new_order, _ = zip(*list(sorted(order_map.items(), key=lambda x: x[1])))
        sorted_features = np.array(list(map(lambda x: features.get(x, np.zeros(feature_size)), new_order)))
        sorted_features = np.vstack([np.zeros(feature_size), sorted_features])
        return sorted_features.astype(np.float32)

    @classmethod
    def _sort_features(cls, features: List[Dict[Any, np.ndarray]], order_map: Dict) -> List[np.ndarray]:
        if features is None:
            return []
        return [cls._sort_features_impl(feature, order_map) for feature in features]
