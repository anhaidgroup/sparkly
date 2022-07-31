from copy import deepcopy
from typing import Union
import shutil 
from tqdm import tqdm
from sparkly.query_generator import QuerySpec, LuceneQueryGenerator
from sparkly.analysis import get_standard_analyzer_no_stop_words, Gram3Analyzer, StandardEdgeGram36Analyzer, UnfilteredGram5Analyzer, get_shingle_analyzer
from sparkly.utils import Timer, init_jvm, zip_dir, atomic_unzip, kill_loky_workers, spark_to_pandas_stream
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
from sparkly.utils import type_check

from pyspark import SparkFiles
from pyspark import SparkContext
from pyspark import sql

import lucene
from java.nio.file import Paths
from java.util import HashMap, HashSet
from org.apache.lucene.search import BooleanQuery, BooleanClause, IndexSearcher
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.miscellaneous import PerFieldAnalyzerWrapper
from org.apache.lucene.search.similarities import BM25Similarity
from org.apache.lucene.index import  DirectoryReader
from org.apache.lucene.document import Document, StoredField, Field, LongPoint
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.store import  FSDirectory
from org.apache.lucene.document import FieldType
from org.apache.lucene.index import IndexOptions

from .index_config import IndexConfig
from .index_base import Index, QueryResult, EMPTY_QUERY_RESULT


class _DocumentConverter:

    def __init__(self, config):
        type_check(config, 'config', IndexConfig)

        self._field_to_doc_fields = {}
        self._config = deepcopy(config)
        self._text_field_type = FieldType()
        self._text_field_type.setIndexOptions(IndexOptions.DOCS_AND_FREQS)
        self._text_field_type.setStoreTermVectors(self._config.store_vectors)

        for f, analyzers in config.field_to_analyzers.items():
            # each field goes to <FIELD>.<ANALYZER_NAME>
            fields = [f'{f}.{a}' for a in analyzers]
            self._field_to_doc_fields[f] = fields

        

    
    def _format_columns(self, df):
        for field, cols in self._config.concat_fields.items():
            df[field] = df[cols[0]].fillna('').astype(str).copy()
            df[field] = df[field].str.cat(df[cols[1:]].astype(str), sep=' ', na_rep='')

        for f, fields in self._field_to_doc_fields.items():
            for new_field in fields:
                if new_field != f:
                    df[new_field] = df[f]
        # get unique fields
        fields = list(set(sum(self._field_to_doc_fields.values(), [])))
        df.set_index(self._config.id_col, inplace=True)
        df = df[fields]
        return df
    
    def _row_to_lucene_doc(self, row):
        doc = Document()
        row.dropna(inplace=True)
        
        doc.add(StoredField(self._config.id_col, row.name))
        doc.add(LongPoint(self._config.id_col, row.name))
        for k,v in row.items():
            doc.add(Field(k, str(v), self._text_field_type))

        return doc

    def convert_docs(self, df):
        type_check(df, 'df', pd.DataFrame)
        # index of df is expected to be _id column
        df = self._format_columns(df)
        docs = df.apply(self._row_to_lucene_doc, axis=1)

        return docs



class LuceneIndex(Index):
    ANALYZERS = {
            'standard' : get_standard_analyzer_no_stop_words,
            'shingle' : get_shingle_analyzer,
            'standard_stopwords' : StandardAnalyzer,
            '3gram' : Gram3Analyzer,
            'standard36edgegram': StandardEdgeGram36Analyzer, 
            'unfiltered_5gram' : UnfilteredGram5Analyzer,
    }
    PY_META_FILE = 'PY_META.json'
    LUCENE_DIR = 'LUCENE_INDEX'

    def __init__(self, index_path):
        self._init_jvm()
        self._index_path = Path(index_path).absolute()
        self._spark = False
        self._query_gen = None
        self._searcher = None
        self._config = None
        self._index_reader = None
        self._spark_index_zip_file = None
        self._initialized = False
        self._index_build_chunk_size = 2500

    @property
    def config(self):
        """
        the index config used to build this index

        Returns
        -------
        IndexConfig
        """
        return self._config

    @property
    def query_gen(self):
        """
        the query generator for this index

        Returns
        -------
        LuceneQueryGenerator
        """
        return self._query_gen
    
    def _init_jvm(self):
        init_jvm(['-Xmx500m'])
        
    def init(self):
        """
        initialize the index for usage in a spark worker. This method 
        must be called before calling search or search_many.
        """
        self._init_jvm()
        if not self._initialized:
            p = self._get_index_dir(self._get_data_dir())
            config = self._read_meta_data()
            analyzer = self._get_analyzer(config)
            # default is 1024 and errors on some datasets
            BooleanQuery.setMaxClauseCount(50000)

            self._query_gen = LuceneQueryGenerator(analyzer, config)

            self._index_reader = DirectoryReader.open(p)
            self._searcher = IndexSearcher(self._index_reader)
            self._searcher.setSimilarity(self._get_sim(config))
            self._initialized = True

    def deinit(self):
        """
        release resources held by this Index
        """
        self._query_gen = None
        self._index_reader = None
        self._searcher = None
        self._initialized = False
    
    def _get_sim(self, config):
        sim_dict = config.sim
        if sim_dict['type'] != 'BM25':
            raise ValueError(sim_dict)
        else:
            s = BM25Similarity(float(sim_dict['k1']), float(sim_dict['b']))
            return s

    def _get_analyzer(self, config):
        mapping = HashMap()
        if config.default_analyzer not in self.ANALYZERS:
            raise ValueError(f'unknown analyzer {config.default_analyzer}, (current possible analyzers {list(self.ANALYZERS)}')

        for f, analyzers in config.field_to_analyzers.items():
            for a in analyzers:
                if a not in self.ANALYZERS:
                    raise ValueError(f'unknown analyzer {a}, (current possible analyzers {list(self.ANALYZERS)}')
                mapping.put(f'{f}.{a}', self.ANALYZERS[a]())
                

        analyzer = PerFieldAnalyzerWrapper(
                self.ANALYZERS[config.default_analyzer](),
                mapping
            )
        return analyzer
    
    def _get_data_dir(self):
        if self._spark:
            p = Path(SparkFiles.get(self._index_path.name))
            # if the file hasn't been unzipped yet,
            # atomically unzip the file and then use it
            if not p.exists():
                zipped = Path(SparkFiles.get(self._spark_index_zip_file.name))
                if not zipped.exists():
                    raise RuntimeError('unable to get zipped index file')
                atomic_unzip(zipped, p)
        else:
            self._index_path.mkdir(parents=True, exist_ok=True)
            p = self._index_path

        return p
    
    def _get_index_dir(self, index_path):
        p = index_path / self.LUCENE_DIR
        p.mkdir(parents=True, exist_ok=True)

        return FSDirectory.open(Paths.get(str(p)))
    
    def _get_index_writer(self, index_config, index_path):
        analyzer = self._get_analyzer(index_config)
        index_dir = self._get_index_dir(index_path)
        index_writer = IndexWriter(index_dir, IndexWriterConfig(analyzer))

        return index_writer
    
    def _write_meta_data(self, config):
        # write the index meta data 
        with open(self._index_path / self.PY_META_FILE, 'w') as ofs:
            ofs.write(config.to_json())

    def _read_meta_data(self):
        p = self._get_data_dir()
        with open(p / self.PY_META_FILE) as ofs:
            return IndexConfig.from_json(ofs.read()).freeze()
        

    @property
    def is_on_spark(self):
        """
        True if this index has been distributed to the spark workers else False

        Returns
        -------
        bool
        """
        return self._spark

    @property
    def is_built(self):
        """
        True if this index has been built else False

        Returns
        -------
        bool
        """
        return self.config is not None

    def to_spark(self):
        """
        send this index to the spark cluster. subsequent uses will read files from 
        SparkFiles, allowing spark workers to perform search with a local copy of 
        the index.
        """
        self.deinit()
        if not self.is_built:
            raise RuntimeError('LuceneIndex must be built before it can be distributed to spark workers')

        if not self._spark:
            sc = SparkContext.getOrCreate()
            self._spark_index_zip_file = zip_dir(self._index_path)
            sc.addFile(str(self._spark_index_zip_file))
            self._spark = True
    
    def _build_segment(self, df, config, tmp_dir_path):

        # use pid to decide which tmp index to write to
        path = tmp_dir_path/ str(multiprocessing.current_process().pid)
        return self._build(df, config, path, append=True)

    def _build(self, df, config, index_path, append=True):
        if len(df.columns) == 0:
            raise ValueError('dataframe with no columns passed to build')
        init_jvm()
        # clear the old index if we are not appending
        if not append and index_path.exists():
            shutil.rmtree(index_path)

        index_writer = self._get_index_writer(config, index_path)
        doc_conv = _DocumentConverter(config)
        docs = doc_conv.convert_docs(df)
        
        for d in docs.values:
            index_writer.addDocument(d)

        index_writer.commit()
        index_writer.close()

        return index_path
    
    def _merge_index_segments(self, config, dirs):

        # clear the old index
        if self._index_path.exists():
            shutil.rmtree(self._index_path)
        # create index writer for merged index
        index_writer = self._get_index_writer(config, self._index_path)
        # merge segments 
        index_writer.addIndexes(dirs)
        index_writer.forceMerge(1)
        index_writer.commit()
        index_writer.close()
    
    def _chunk_df(self, df):
        for i in range(0, len(df), self._index_build_chunk_size):
            end = min(len(df), i+self._index_build_chunk_size)
            yield df.iloc[i:end]


    def _arg_check_build(self, df : Union[pd.DataFrame, sql.DataFrame], config : IndexConfig):
        type_check(config, 'config', IndexConfig)
        type_check(df, 'df', (pd.DataFrame, sql.DataFrame))
        if self.config is not None:
            raise RuntimeError('This index has already been built')

        if len(config.field_to_analyzers) == 0:
            raise ValueError('config with no fields passed to build')
        
        if config.id_col not in df.columns:
            raise ValueError(f'id column {config.id_col} is not is dataframe columns {df.columns}')
        
        missing_cols = set(config.get_analyzed_fields()) - set(df.columns)
        if len(missing_cols) != 0:
            raise ValueError(f'dataframe is missing columns {list(missing_cols)} required by config (actual columns in df {df.columns})')

        if isinstance(df, pd.DataFrame):
            dtype = df[config.id_col].dtype
            if not pd.api.types.is_integer_dtype(dtype):
                raise TypeError(f'id_col must be integer type (got {dtype})')
        else:
            dtype = df.schema[config.id_col].dataType
            if dtype.typeName() not in {'integer', 'long'}:
                raise TypeError(f'id_col must be integer type (got {dtype})')
            
    
    def build(self, df, config):
        """
        build the index, indexing df according to config

        Parameters
        ----------

        df : pd.DataFrame or pyspark DataFrame
            the table that will be indexed, if a pyspark DataFrame is provided, the build will be done
            in parallel for suffciently large tables

        config : IndexConfig
            the config for the index being built
        """
        self._arg_check_build(df, config)

        if isinstance(df, sql.DataFrame):
            # project out unused columns
            df = df.select(config.id_col, *config.get_analyzed_fields())
            if df.count() > self._index_build_chunk_size * 10:
                # build large tables in parallel
                # put temp indexes in temp dir for easy deleting later
                with TemporaryDirectory() as tmp_dir_base:
                    tmp_dir_base = Path(tmp_dir_base)
                    # slice the dataframe into a local iterator of pandas dataframes
                    slices = spark_to_pandas_stream(df, self._index_build_chunk_size)
                    # use all available threads
                    pool = Parallel(n_jobs=-1)
                    # build in parallel in sub dirs of tmp dir
                    dirs = pool(delayed(self._build_segment)(s, config, tmp_dir_base) for s in tqdm(slices))
                    # dedupe the dirs
                    dirs = set(dirs)
                    # get the name of the index dir in each tmp sub dir
                    dirs = [self._get_index_dir(d) for d in dirs]
                    # merge the segments 
                    self._merge_index_segments(config, dirs)
                    # kill the threadpool to prevent them from sitting on resources
                    kill_loky_workers()
                # temp indexes deleted here
            else:
                # table is small, build it single threaded
                df = df.toPandas()

        if isinstance(df, pd.DataFrame):
            # if table is small just build directly
            self._build(df, config, self._index_path, append=False)

        # write the config
        self._write_meta_data(config)
        self._config = config.freeze()
    
    
    def get_full_query_spec(self, cross_fields=False):
        """
        get a query spec that uses all indexed columns

        Parameters
        ----------

        cross_fields : bool, default = False
            if True return <FIELD> -> <CONCAT FIELD> in the query spec if FIELD is used to create CONCAT_FIELD
            else just return <FIELD> -> <FIELD> and <CONCAT_FIELD> -> <CONCAT_FIELD> pairs

        Returns
        -------
        QuerySpec

        """
        type_check(cross_fields, 'cross_fields', bool)

        if self._config is None:
            self._config = self._read_meta_data()

        search_to_index_fields = {}
        for f, analyzers in self._config.field_to_analyzers.items():
            # each field goes to <FIELD>.<ANALYZER_NAME>
            fields = [f'{f}.{a}' for a in analyzers]
            search_to_index_fields[f] = fields

        if cross_fields:
            for f, search_fields in self._config.concat_fields.items():
                analyzers = self._config.field_to_analyzers[f]
                index_fields = [f'{f}.{a}' for a in analyzers]
                for sfield in search_fields:
                    search_to_index_fields[sfield] += index_fields

        return QuerySpec(search_to_index_fields)

    def search(self, doc, query_spec, limit):
        """
        perform search for `doc` according to `query_spec` return at most `limit` docs

        Parameters
        ----------

        doc : pd.Series or dict
            the record for searching

        query_spec : QuerySpec
            the query template that specifies how to search for `doc`

        limit : int
            the maximum number of documents returned

        Returns
        -------
        QueryResult
            the documents matching the `doc`
        """
        type_check(query_spec, 'query_spec', QuerySpec)
        type_check(limit, 'limit', int)
        type_check(doc, 'doc', (pd.Series, dict))

        if limit <= 0:
            raise ValueError('limit must be > 0 (limit passed was {limit})')

        
        load_fields = HashSet()
        load_fields.add(self.config.id_col)
        query = self._query_gen.generate_query(doc, query_spec)
        #query = query.rewrite(self._index_reader)

        if query is None:
            return EMPTY_QUERY_RESULT

        else:
            timer = Timer()
            res = self._searcher.search(query, limit)
            t = timer.get_interval()

            res = res.scoreDocs
            nhits = len(res)
            scores = np.fromiter((h.score for h in res), np.float32, nhits)
            # fetch docs and get our id
            ids = np.fromiter((int(self._searcher.doc(h.doc, load_fields).get(self.config.id_col)) for h in res), np.int64, nhits)
            return QueryResult(
                    ids = ids,
                    scores = scores, 
                    search_time = t,
                )
        
    def search_many(self, docs, query_spec, limit):
        """
        perform search for the documents in `docs` according to `query_spec` return at most `limit` docs
        per document `docs`.

        Parameters
        ----------

        doc : pd.DataFrame
            the records for searching

        query_spec : QuerySpec
            the query template that specifies how to search for `doc`

        limit : int
            the maximum number of documents returned

        Returns
        -------
        pd.DataFrame
            the search results for each document in `docs`, indexed by `docs`.index
            
        """
        type_check(query_spec, 'query_spec', QuerySpec)
        type_check(limit, 'limit', int)
        type_check(docs, 'docs', (pd.DataFrame))
        if limit <= 0:
            raise ValueError('limit must be > 0 (limit passed was {limit})')
        self.init()
        id_col = self.config.id_col
        load_fields = HashSet()
        load_fields.add(id_col)

        search_res = []
        for doc in docs.to_dict('records'):
            query = self._query_gen.generate_query(doc, query_spec)
            #query = query.rewrite(self._index_reader)
            if query is None:
                search_res.append(EMPTY_QUERY_RESULT)
            else:
                timer = Timer()
                res = self._searcher.search(query, limit)
                t = timer.get_interval()

                res = res.scoreDocs

                nhits = len(res)
                scores = np.fromiter((h.score for h in res), np.float32, nhits)
                # fetch docs and get our id
                ids = np.fromiter((int(self._searcher.doc(h.doc, load_fields).get(id_col)) for h in res), np.int64, nhits)
                search_res.append( QueryResult(
                        ids = ids,
                        scores = scores, 
                        search_time = t,
                    ) )

        return pd.DataFrame(search_res, index=docs.index)
        
    
    def id_to_lucene_id(self, i):
        q = LongPoint.newExactQuery(self.config.id_col, i)
        res = self._searcher.search(q, 2).scoreDocs
        if len(res) == 0:
            raise KeyError(f'no document with _id = {i} found')
        elif len(res) > 1:
            raise KeyError(f'multiple documents with _id = {i} found')

        return res[0].doc


    def _score_docs(self, ids_filter, query, limit):
        q = BooleanQuery.Builder()\
                .add(ids_filter, BooleanClause.Occur.FILTER)\
                .add(query, BooleanClause.Occur.SHOULD)\
                .build()
        
        res = self._searcher.search(q, limit)

        return res.scoreDocs


    def score_docs(self, ids, queries : dict):
        # queries = {(field, indexed_field) -> Query}
        # ids the _id fields in the documents
        if not isinstance(ids, list):
            raise TypeError()
        if len(ids) == 0:
            return pd.DataFrame()

        limit = len(ids)

        ids_filter = LongPoint.newSetQuery(self.config.id_col, ids)

        df_columns = [
                pd.Series(
                    data=ids,
                    index=[self.id_to_lucene_id(i) for i in ids],
                    name=self.config.id_col

                )
        ]
        for name, q in queries.items():
            res = self._score_docs(ids_filter, q, limit)
            nhits = len(res)
            df_columns.append(
                    pd.Series( 
                        data=np.fromiter((h.score for h in res), np.float32, nhits),
                        index=np.fromiter((h.doc for h in res), np.int64, nhits),
                        name=name
                    )
            )

        df = pd.concat(df_columns, axis=1).fillna(0.0)
        return df
