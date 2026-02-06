import sys
sys.path.append('.')
import pyspark.sql.types as T
import pyspark
import pandas as pd
from sparkly.index import QueryResult, Index
from sparkly.query_generator import QuerySpec
from sparkly.utils import type_check_call
from pydantic import (
        PositiveInt,
)


CHUNK_SIZE=500
JSON_DATA = {}

class Searcher:
    """
    class for performing bulk search over a dataframe
    """
    @type_check_call
    def __init__(self, index: Index, search_chunk_size: PositiveInt=CHUNK_SIZE):
        """

        Parameters
        ----------
        index : Index
            The index that will be used for search
        search_chunk_size : int
            the number of records is each partition for searching
        """
        self._index = index
        self._search_chunk_size = search_chunk_size
    
    def get_full_query_spec(self):
        """
        get a query spec that searches on all indexed fields
        """
        return self._index.get_full_query_spec()
    
    @type_check_call
    def search(self, search_df: pyspark.sql.DataFrame, query_spec: QuerySpec, limit: PositiveInt, id_col: str='_id'):
        """
        perform search for all the records in search_df according to
        query_spec

        Parameters
        ----------

        search_df : pyspark.sql.DataFrame
            the records used for searching
        query_spec : QuerySpec
            the query spec for searching
        limit : int
            the topk that will be retrieved for each query
        id_col : str
            the id column from search_df that will be output with the query results

        Returns
        -------
        pyspark DataFrame
            a pyspark dataframe with the schema (id2, id1_list array<long> , scores array<float>, search_time float)
        """
        return self._search_spark(search_df, query_spec, limit, id_col)



    def _search_spark(self, search_df, query_spec, limit, id_col='_id'):
        # set data to spark workers
        self._index.to_spark()

        projection = self._index.config.get_analyzed_fields(query_spec)
        if id_col not in projection:
            projection.append(id_col)
        search_df = search_df.select(projection)\
                        .repartition(max(1, search_df.count() // self._search_chunk_size), id_col)

        f = lambda x : _search_spark(self._index, query_spec, limit, x, id_col)

        # Schema using QueryResult._fields with 'id2' prepended (instead of id_col)
        query_result_fields = ['id2'] + list(QueryResult._fields)
        query_result_types = [ T.LongType(), T.ArrayType(T.LongType()), T.ArrayType(T.FloatType()), T.FloatType()]
        query_result_schema = T.StructType(list(map(T.StructField, query_result_fields, query_result_types)))

        res = search_df.mapInPandas(f, query_result_schema)
        return res


def _search_spark(index, query_spec, limit, partition_itr, id_col):
    index.init()
    for part in partition_itr:
        part = part.set_index(id_col)
        result = _search_many(index, query_spec, limit, part)
        yield result

def _search_many(index, query_spec, limit, df):

    res = index.search_many(df, query_spec, limit)
    return res.reset_index(drop=False)


def search(index, query_spec, limit, search_recs):
    return  list(search_gen(index, query_spec, limit, search_recs))

def search_gen(index, query_spec, limit, search_recs):
    index.init()
    for rec in search_recs:
        yield index.search(rec, query_spec, limit)