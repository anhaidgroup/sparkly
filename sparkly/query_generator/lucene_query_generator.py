import pandas as pd
from sparkly.utils import is_null
import lucene
from org.apache.lucene.util import QueryBuilder
from org.apache.lucene.search import BooleanQuery, BooleanClause, BoostQuery


class LuceneQueryGenerator:
    """
    A class for generating queries for Lucene based indexes
    """

    def __init__(self, analyzer, config, index_reader):
        """
        Parameters
        ----------
        analyzer : 
            the luncene analyzer used to create queries
        config : IndexConfig
            the index config of the index that will be searched
        """
        self._analyzer = analyzer
        self._config = config
        # index reader not used 
        self._query_builder = QueryBuilder(analyzer)
        self._query_builder.setEnableGraphQueries(False)
    
    def generate_query(self, doc, query_spec):
        """
        Generate a query for doc given the query spec

        Parameters
        ----------
        doc : dict | pd.Series
            a record that will be used to generate the query
        query_spec : QuerySpec
            the template for the query being built

        Returns
        -------
        A lucene query which can be passed to an index searcher
        """
        query = BooleanQuery.Builder()
        filter_query = BooleanQuery.Builder()
        filters = query_spec.filter
        add_filter = False
        
        for field, indexed_fields in query_spec.items():
            if field not in doc:
                # create concat field on the fly
                if field in self._config.concat_fields:
                    val = ' '.join(str(doc[f]) for f in self._config.concat_fields[field])
                else:
                    raise RuntimeError(f'field {field} not in search document {doc}, (config : {self._config.to_dict()})')
            else:
                # otherwise just retrive from doc
                val = doc[field]

            # convert to lucene query if the val is valid
            if is_null(val):
                continue

            val = str(val)
            for f in indexed_fields:
                clause = self._query_builder.createBooleanQuery(f, val)
                # empty clause skip adding to query
                if clause is None:
                    continue

                if (field, f) in filters:
                    filter_query.add(clause, BooleanClause.Occur.SHOULD)
                    add_filter = True

                # add boosting weight if it exists
                weight = query_spec.boost_map.get((field, f))
                if weight is not None:
                    clause = BoostQuery(clause, weight)

                query.add(clause, BooleanClause.Occur.SHOULD)


        if len(filters) != 0 and add_filter:
            query.add(filter_query.build(), BooleanClause.Occur.FILTER)

        return query.build()

    def generate_query_clauses(self, doc, query_spec):
        """
        generate the clauses for each field -> analyzer pair, filters are ignored

        Parameters
        ----------
        doc : dict | pd.Series
            a record that will be used to generate the clauses
        query_spec : QuerySpec
            the template for the query being built

        Returns
        -------
        A dict of ((field, indexed_fields) -> BooleanQuery)
        """
        # this isn't great code writing considering that this is a 
        # duplicate of the code above but generate_query is a hot code path
        # and can use all the optimization that it can get
        clauses = {}
        for field, indexed_fields in query_spec.items():
            if field not in doc:
                # create concat field on the fly
                if field in self._config.concat_fields:
                    val = ' '.join(str(doc.get(f, '')) for f in self._config.concat_fields[field])
                else:
                    raise RuntimeError(f'field {field} not in search document {doc}')
            else:
                # otherwise just retrive from doc
                val = doc[field]
            # convert to lucene query if the val is valid
            if pd.isnull(val):
                continue

            for f in indexed_fields:
                clause = self._query_builder.createBooleanQuery(f, str(val))
                if clause is None:
                    continue
                # add boosting weight if it exists
                weight = query_spec.boost_map.get((field, f))
                if weight is not None:
                    clause = BoostQuery(clause, weight)

                clauses[(field, f)] = clause

        return clauses       
