from abc import abstractmethod, ABC, abstractproperty
import pandas as pd
from collections import namedtuple
from sparkly.query_generator import QuerySpec

query_result_fields = ['id1_list', 'scores', 'search_time']
QueryResult = namedtuple("QueryResult",
                            query_result_fields
                        )

EMPTY_QUERY_RESULT = QueryResult(None, None, None)
    
# TODO COMMENT
class Index(ABC):

    @abstractmethod
    def upsert_docs(self, df) -> None:
        pass

    @abstractmethod
    def delete_docs(self, ids) -> int:
        pass

    @abstractmethod
    def search(self, doc, query_spec, limit):
        pass

    @abstractmethod
    def search_many(self, docs, query_spec, limit):
        pass

    @abstractproperty
    def config(self):
        # the IndexConfig for the index
        pass
