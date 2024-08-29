from copy import deepcopy
import json
from sparkly.utils import type_check, type_check_iterable, type_check_call
from typing import Iterable

class IndexConfig:

    @type_check_call
    def __init__(self, *, store_vectors: bool=False, id_col: str='_id', weighted_queries: bool=False):
        self.field_to_analyzers = {}
        self.concat_fields = {}
        self._id_col = id_col
        self.default_analyzer = 'standard'
        self.sim = {'type' : 'BM25', 'k1' : 1.2, 'b' : .75}  
        type_check(store_vectors, 'store_vectors', bool)
        self._store_vectors = store_vectors
        self._frozen = False
        self._weighted_queries = weighted_queries
    

    def freeze(self):
        """
        Returns
        -------
        IndexConfig
            a frozen deepcopy of this index config
        """
        o = deepcopy(self)
        o._frozen = True
        return o

    @property
    def is_frozen(self):
        """
        Returns
        -------
        bool
            True if this index is frozen (not modifiable) else False
        """
        return self._frozen
    @property
    def weighted_queries(self):
        """
        True if the term vectors in the index should be stored, else False
        """
        return self._weighted_queries

    @weighted_queries.setter
    @type_check_call
    def weighted_queries(self, o: bool):
        self._raise_if_frozen()
        self._weighted_queries = o

    @property
    def store_vectors(self):
        """
        True if the term vectors in the index should be stored, else False
        """
        return self._store_vectors

    @store_vectors.setter
    @type_check_call
    def store_vectors(self, o: bool):
        self._raise_if_frozen()
        self._store_vectors = o
    
    @property
    def id_col(self):
        """
        The unique id column for the records in the index this must be a 32 or 64 bit integer
        """
        return self._id_col

    @id_col.setter
    @type_check_call
    def id_col(self, o: str):
        self._raise_if_frozen()
        self._id_col = o

    @classmethod
    def from_json(cls, data):
        """
        construct an index config from a dict or json string,
        see IndexConfig.to_dict for expected format

        Returns
        -------
        IndexConfig
        """
        if isinstance(data, str):
            data = json.loads(data)

        o = cls()
        o.field_to_analyzers = data['field_to_analyzers']
        o.concat_fields = data['concat_fields']
        o.default_analyzer = data['default_analyzer']
        o.sim = data['sim']
        o.id_col = data['id_col']
        o.weighted_queries = data['weighted_queries']
        return o

    def to_dict(self):
        """
        convert this IndexConfig to a dictionary which can easily 
        be stored as json

        Returns
        -------
        dict
            A dictionary representation of this IndexConfig
        """
        d = {
                'field_to_analyzers' : self.field_to_analyzers,
                'concat_fields' : self.concat_fields,
                'default_analyzer' : self.default_analyzer,
                'sim' : self.sim,
                'store_vectors' : self.store_vectors,
                'id_col' : self.id_col,
                'weighted_queries' : self.weighted_queries
        }
        return d

    def to_json(self):
        """
        Dump this IndexConfig to a valid json strings

        Returns
        -------
        str
        """
        return json.dumps(self.to_dict())

    @type_check_call
    def add_field(self, field : str, analyzers: Iterable[str]):
        """
        Add a new field to be indexed with this config

        Parameters
        ----------

        field : str
            The name of the field in the table to the index

        analyzers : set, list or tuple of str
            The names of the analyzers that will be used to index the field
        """
        self._raise_if_frozen()
        self.field_to_analyzers[field] = list(analyzers)

        return self

    @type_check_call
    def remove_field(self, field: str):
        """
        remove a field from the config

        Parameters
        ----------

        field : str 
            the field to be removed from the config

        Returns
        -------
        bool 
            True if the field existed else False
        """

        self._raise_if_frozen()
        if field in self.field_to_analyzers:
            self.field_to_analyzers.pop(field)
            if field in self.concat_fields:
                self.concat_fields.pop(field)
            return True
        else:
            return False

    @type_check_call
    def add_concat_field(self, field : str, concat_fields: Iterable[str], analyzers: Iterable[str]):
        """
        Add a new concat field to be indexed with this config

        Parameters
        ----------

        field : str
            The name of the field that will be added to the index

        concat_fields : set, list or tuple of strs
            the fields in the table that will be concatenated together to create `field`

        analyzers : set, list or tuple of str
            The names of the analyzers that will be used to index the field
        """
        self._raise_if_frozen()
        self.concat_fields[field] = list(concat_fields)
        self.field_to_analyzers[field] = list(analyzers)

        return self

    def get_analyzed_fields(self, query_spec=None):
        """
        Get the fields used by the index or query_spec. If `query_spec` is None, 
        the fields that are used by the index are returned.

        Parameters
        ----------

        query_spec : QuerySpec, optional
            if provided, the fields that are used by `query_spec` in creating a query

        Returns
        -------
        list of str
            the fields used
        """
        if query_spec is not None:
            fields = []
            for f in query_spec:
                if f in self.concat_fields:
                    fields += self.concat_fields[f]
                else:
                    fields.append(f)
        else:
            fields = sum(self.concat_fields.values(), [])
            fields += (x for x in self.field_to_analyzers if x not in self.concat_fields) 

        return list(set(fields))

    def _raise_if_frozen(self):
        if self.is_frozen:
            raise RuntimeError('Frozen IndexConfigs cannot be modified')
