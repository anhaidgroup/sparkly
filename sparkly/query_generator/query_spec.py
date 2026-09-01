from copy import deepcopy
import pandas as pd
from sparkly.utils import type_check_call
from typing import Tuple, Iterable

class QuerySpec(dict):
    """
    A specification for generating queries
    """

    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if not isinstance(v, (set, list, tuple)):
                raise TypeError(f'value must be (set, list, tuple), not {type(v)}')
            for s in v:
                if not isinstance(s, str):
                    raise TypeError(f'all paths must be strings, not {type(s)}')
            else:
                self[k] = set(v)

        self._boost_map = {}
        # filter is only applied if the pair is also used in scoring
        # currently OR 
        self._filter = frozenset()
    
    def __hash__(self):
        return hash(frozenset(self.keys()))

    def __eq__(self, o):
        return super().__eq__(o) and self._boost_map == o._boost_map and self._filter == o._filter

    @property
    def size(self):
        return sum(map(len, self.values()))

    @property
    def filter(self):
        return self._filter

    @filter.setter
    def filter(self, fil: Iterable[Tuple[str, str]]):
        for k in fil:
            if not isinstance(k, tuple):
                raise TypeError(f'all keys must be tuple (got {type(k)})')
                if len(k) != 2 or isinstance(k[0], str) or isinstance(k[1], str):
                    raise TypeError(f'all keys must be pairs of strings (got {k})')

        fil = frozenset(fil)
        pairs = {(k,x) for k,v in self.items() for x in v}

        missing = fil - pairs
        if len(missing) != 0:
            raise RuntimeError(f'all pairs in the filter must be used for scoring in the query spec (missing {missing})')

        self._filter = fil

    @property
    def boost_map(self):
        """
        The boosting weights for each (search_field -> indexed_field) 
        pair. If a pair doesn't exist in the map, the boost weight is 1
        """
        return self._boost_map

    @boost_map.setter
    @type_check_call
    def boost_map(self, boost_map: dict | pd.Series):
        for k,v in boost_map.items():
            if not isinstance(k, tuple):
                raise TypeError(f'all keys must be tuples (got {type(k)})')
                if len(k) != 2 or isinstance(k[0], str) or isinstance(k[1], str):
                    raise TypeError(f'all keys must be pairs of strings (got {k})')

            if not isinstance(v, float):
                raise TypeError(f'all boosting weights must be floats (got {type(v)})')

        self._boost_map = {k : float(v) for k,v in boost_map.items()}

    @type_check_call
    def union(self, other):
        self = deepcopy(self)
        for k,v in other.items():
            self[k] = self.get(k, set()) | v

        return self

    @type_check_call
    def is_subset(self, other) -> bool:
        for k,v in other.items():
            if k not in self or self[k].issuperset(v):
                return False
        return True

    @classmethod
    def from_dict(cls, data: dict):
        spec = cls(data['spec'])
        spec.filter = [tuple(x) for x in data.get('filter', [])]

        boost_map = data.get('boost_map', {})
        if isinstance(boost_map, list):
            boost_map = {tuple(x['key']): x['value'] for x in boost_map}

        spec.boost_map = boost_map
        return spec

    @staticmethod
    def _serialize_tuple_key_map(data: dict) -> list:
        return [
            {'key': list(k), 'value': v}
            for k, v in sorted(data.items(), key=lambda x: x[0])
        ]

    def to_dict(self, json_safe: bool=False) -> dict:
        boost_map = deepcopy(self._boost_map)
        filter_pairs = list(self._filter)

        if json_safe:
            boost_map = self._serialize_tuple_key_map(boost_map)
            filter_pairs = [list(x) for x in sorted(filter_pairs)]

        return {
                'boost_map' : boost_map,
                'spec' : {
                    k : sorted(v) if json_safe else list(v)
                    for k,v in self.items()
                },
                'filter' : filter_pairs
        }
