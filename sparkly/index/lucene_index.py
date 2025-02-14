from copy import deepcopy
from typing import Union
import shutil 
import tempfile
import os
from tqdm import tqdm
from sparkly.analysis import Gram4Analyzer, Gram2Analyzer, UnfilteredGram3Analyzer
from sparkly.query_generator import QuerySpec, LuceneQueryGenerator, LuceneWeightedQueryGenerator
from sparkly.analysis import get_standard_analyzer_no_stop_words, Gram3Analyzer, StandardEdgeGram36Analyzer, UnfilteredGram5Analyzer, get_shingle_analyzer
from sparkly.analysis import StrippedGram3Analyzer
from sparkly.utils import (
        Timer, init_jvm, zip_dir, 
        atomic_unzip, kill_loky_workers, 
        spark_to_pandas_stream, attach_current_thread_jvm,
        type_check_call,
)
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
import pyspark.sql.types as T
import pyspark.sql.functions as F
from sparkly.utils import type_check, type_check_iterable
import time
from itertools import islice
from pydantic import (
        FilePath,
        PositiveInt
)

from pyspark import SparkFiles
from pyspark import SparkContext
from pyspark import sql
import pickle

import lucene
from java.nio.file import Paths
from java.util import HashMap, HashSet
from java.lang import Long
from org.apache.lucene.search import BooleanQuery, BooleanClause, IndexSearcher, MatchAllDocsQuery
from org.apache.lucene.search import SortedNumericSortField, Sort, SortField
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.miscellaneous import PerFieldAnalyzerWrapper
from org.apache.lucene.search.similarities import BM25Similarity
from org.apache.lucene.index import  DirectoryReader
from org.apache.lucene.document import Document, StoredField, Field, LongPoint
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.store import  FSDirectory
from org.apache.lucene.document import FieldType, SortedNumericDocValuesField
from org.apache.lucene.index import IndexOptions

from sparkly.index_config import IndexConfig
from sparkly.thread_progress_bar import ThreadProgressBar
from sparkly.utils import get_logger
from .index_base import Index, QueryResult, EMPTY_QUERY_RESULT

logger = get_logger(__name__)


class _DocumentConverter:
    @type_check_call
    def __init__(self, config : IndexConfig):

        self._field_to_doc_fields = {}
        self._config = deepcopy(config)
        #self._id_field_type = FieldType()
        # self._id_field_type.setStored(True)
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
        # this has to be a string because 
        # pylucene improperly casts to an int, storing the incorrect 
        # value for 64 bit integers 
        f = StoredField(self._config.id_col, str(row.name))
        #f = StoredField(self._config.id_col, Long.MAX_VALUE)
        #f.setLongValue(int(row.name))
        doc.add(f)
        doc.add(LongPoint(self._config.id_col, int(row.name)))
        # needed for sorting by id_col
        #doc.add(SortedNumericDocValuesField(self._config.id_col, int(row.name)))
        for k,v in row.items():
            doc.add(Field(k, str(v), self._text_field_type))

        return doc

    @type_check_call
    def convert_docs(self, df : pd.DataFrame):
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
            '2gram' : Gram2Analyzer,
            '4gram' : Gram4Analyzer,
            'stripped_3gram' : StrippedGram3Analyzer,
            'unfiltered_3gram' : UnfilteredGram3Analyzer,
            'standard36edgegram': StandardEdgeGram36Analyzer, 
            'unfiltered_5gram' : UnfilteredGram5Analyzer,
    }
    PY_META_FILE = 'PY_META.json'
    LUCENE_DIR = 'LUCENE_INDEX'

    @type_check_call
    def __init__(self, index_path: Path | str, config: IndexConfig, delete_if_exists: bool=True):
        self._init_jvm()
        self._index_path = Path(index_path).absolute()
        self._config = config.freeze()
        self._spark = False
        self._query_gen = None
        self._searcher = None
        self._index_reader = None
        self._spark_index_zip_file = None
        self._spark_index_dir_name = None
        self._initialized = False
        self._is_built = False
        self._index_build_chunk_size = 2500

        self._arg_check_config(self.config)
        # clear old index if it exists
        if delete_if_exists and self._index_path.exists():
            shutil.rmtree(self._index_path)

        # write the config
        self._write_meta_data(config)
    
    @property
    def index_path(self):
        return self._index_path

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
        init_jvm(['-Xmx256m'])
        #attach_current_thread_jvm()
        
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
            self._index_reader = DirectoryReader.open(p)
            self._searcher = IndexSearcher(self._index_reader)
            self._searcher.setSimilarity(self._get_sim(config))
            # default is 1024 and errors on some datasets
            BooleanQuery.setMaxClauseCount(500000)
            if self.config.weighted_queries:
                self._query_gen = LuceneWeightedQueryGenerator(analyzer, config, self._index_reader)
            else:
                self._query_gen = LuceneQueryGenerator(analyzer, config, self._index_reader)

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
            p = Path(SparkFiles.get(self._spark_index_dir_name))
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
        self._index_path.mkdir(parents=True, exist_ok=True)
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
        return self._is_built

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
            self._spark_index_dir_name = self._spark_index_zip_file.stem
            sc.addFile(str(self._spark_index_zip_file))
            self._spark = True
    
    def __reduce__(self):
        self.deinit()
        return super().__reduce__()

    def _build_segment(self, df, config, tmp_dir_path, show_progress_bar):

        # use pid to decide which tmp index to write to
        path = tmp_dir_path/ str(multiprocessing.current_process().pid)
        self._init_jvm()
        index_writer = self._get_index_writer(config, path)
        self._add_docs(df, index_writer, show_progress_bar)
        index_writer.commit()
        index_writer.close()

        return path

    def _add_docs(self, df, index_writer, show_progress_bar):
        if len(df.columns) == 0:
            raise ValueError('dataframe with no columns passed to build')
        self._init_jvm()

        doc_conv = _DocumentConverter(self.config)
        docs = doc_conv.convert_docs(df)
        itr = docs.values if not show_progress_bar else tqdm(docs.values, desc="Adding documents")
        for d in itr:
            index_writer.addDocument(d)

    
    def _merge_index_segments(self, index_writer, dirs):
        # merge segments 
        index_writer.addIndexes(dirs)
        index_writer.forceMerge(1)
    
    def _chunk_df(self, df):
        for i in range(0, len(df), self._index_build_chunk_size):
            end = min(len(df), i+self._index_build_chunk_size)
            yield df.iloc[i:end]

    
    def _arg_check_config(self, config):
        if len(config.field_to_analyzers) == 0:
            raise ValueError('config with no fields passed to build')

    def _arg_check_upsert(self, df : Union[pd.DataFrame, sql.DataFrame]):

        config = self.config
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
    
    @classmethod
    def _build_spark_worker_local(cls, df_itr, config, acc):
        index = None
        index_path = None
        index_writer = None
        tmp_dir_path = Path(tempfile.gettempdir())
        tmp_dir_path.mkdir(parents=True, exist_ok=True)
        # 128 MB
        CHUNK_SIZE = 128 * (2**20)
        for df in df_itr:
            if len(df) == 0:
                continue

            if index is None:
                index_path = tmp_dir_path / f'{time.time()}_{df.iloc[0][config.id_col]}'
                index = cls(index_path, config)
                index_writer = index._get_index_writer(index.config, index_path)

            index._add_docs(df, index_writer, False)
            acc.add(df.count())

        index_writer.commit()
        index_writer.close()
        
        for p in cls._serialize_and_stream_files(index_path, CHUNK_SIZE):
            # change file path to be relvative so that it can be easily relocated 
            # elsewhere
            p['file'] = p['file'].apply(lambda x: str(Path(x).relative_to(tmp_dir_path)))
            yield p

        shutil.rmtree(index_path, ignore_errors=True)
    
    @staticmethod
    def _write_file_chunk(file, offset, data):
        path = Path(file)
        path.parent.mkdir(parents=True, exist_ok=True)
        # O_APPEND (i.e. open(file, 'ab') ) cannot be used because on linux it 
        # will just append and ignore the offset
        # 
        # create the file if it doesn't exist, else just open for write
        fd = os.open(file, os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            n = os.pwrite(fd, data, offset)
            if n != len(data):
                raise IOError(f'{n} != {len(data)}')
        finally:
            os.close(fd)


    @staticmethod
    def _stream_files(d, chunk_size):
        # go through all of the files and 
        for f in d.glob('**/*'):
            # skip non-files
            if not f.is_file():
                continue 
            f = str(f)
            # split file into chunks of size at most chunk_size
            with open(f, 'rb') as ifs:
                offset = 0
                while True:
                    data = ifs.read(chunk_size)
                    if len(data) == 0:
                        break
                    yield (f, offset, data)
                    offset += len(data)


    @staticmethod
    def _serialize_and_stream_files(d, part_size):
        # part_size in bytes
        # take all files, serialize with path and return 
        df_columns = ['file', 'offset', 'data']

        curr_size = 0
        rows = []
        for row in LuceneIndex._stream_files(d, part_size):
            rows.append(row)
            curr_size += len(row[-1])
            # buffer full
            if curr_size >= part_size:
                yield pd.DataFrame(rows, columns=df_columns)
                rows.clear()
                curr_size = 0
        # yield last rows if there are any
        if len(rows) > 0:
            yield pd.DataFrame(rows, columns=df_columns)
            rows.clear()
                
    def _build_spark(self, df, df_size, config, tmp_dir_path, show_progress_bar):
        nparts = df_size // self._index_build_chunk_size
        df = df.repartition(nparts, config.id_col)
        
        schema = T.StructType([
            T.StructField('file', T.StringType(), False),
            T.StructField('offset', T.LongType(), False),
            T.StructField('data', T.BinaryType(), False),
        ])
        
        # start progress tracking
        sc = SparkContext.getOrCreate()
        acc = sc.accumulator(0)
        total = df_size
        with ThreadProgressBar("Index building", total, acc, show_progress_bar):
            df = df.mapInPandas(lambda x : LuceneIndex._build_spark_worker_local(x, config, acc), schema=schema)\
                    .withColumn('file', F.concat( F.lit(str(tmp_dir_path) + '/'), F.col('file')) )\
                    .persist()
            # index stuff
            df.count()
            itr = df.toLocalIterator(True)

        # write with threads 
        Parallel(n_jobs=-1, backend='threading')(delayed(self._write_file_chunk)(*row) for row in itr)
        df.unpersist()
        return list(tmp_dir_path.iterdir())

    def _build_parallel_local(self, df, config, tmp_dir_base, show_progress_bar):
        # slice the dataframe into a local iterator of pandas dataframes
        slices = spark_to_pandas_stream(df, self._index_build_chunk_size)
        # use all available threads
        pool = Parallel(n_jobs=-1)
        # build in parallel in sub dirs of tmp dir
        slices = slices if not show_progress_bar else tqdm(slices, desc="Building Index")
        dirs = pool(delayed(self._build_segment)(s, config, tmp_dir_base) for s in slices)
        # dedupe the dirs
        dirs = set(dirs)
        # kill the threadpool to prevent them from sitting on resources

        return dirs
    
    def _count_docs(self, query):
        self.init()
        c = self._searcher.count(query)
        return c
    
    def _delete_docs(self, index_writer, ids):
        if isinstance(ids, (np.ndarray, pd.Series)):
            ids = ids.tolist()
        type_check_iterable(ids, 'ids', list, int)

        query = LongPoint.newSetQuery(self.config.id_col, ids)
        cnt = self._count_docs(query)
            
        index_writer.deleteDocuments(query)
        return cnt
    
    def _invalidate_spark(self):
        self._spark = False
        self.deinit()

    def delete_docs(self, ids):
        
        if not self.is_built:
            raise RuntimeError('index not built')

        index_writer = self._get_index_writer(self.config, self._index_path)
        try:
            cnt = self._delete_docs(index_writer, ids)
        except:
            index_writer.rollback()
            raise
        else:
            index_writer.commit()
            index_writer.close()

        self._invalidate_spark()
        return cnt

    
    @type_check_call
    def upsert_docs(self, df: Union[pd.DataFrame, sql.DataFrame], disable_distributed: bool=False, force_distributed: bool=False, show_progress_bar: bool=False):
        """
        build the index, indexing df according to self.config

        Parameters
        ----------

        df : pd.DataFrame or pyspark DataFrame
            the table that will be indexed, if a pyspark DataFrame is provided, the build will be done
            in parallel for suffciently large tables

        disable_distributed : bool, default=False
            disable using spark for building the index even for large tables

        force_distributed : bool, default=False
            force using spark for building the index even for smaller tables

        show_progress_bar: bool, default=False
            show the progress bar in addition to debug logs
        """
        if disable_distributed and force_distributed:
            raise ValueError('disable_distributed and force_distributed both set to True, only one can be set')
        self._arg_check_upsert(df)

        # verify the index is correct
        index_writer = self._get_index_writer(self.config, self._index_path)

        try:
            start_index_size = 0 
            num_docs_deleted = 0
            # upsert
            if self.is_built:
                tmp_df = df
                if isinstance(df, sql.DataFrame):
                    tmp_df = df.select(self.config.id_col).toPandas()
                ids = tmp_df[self.config.id_col].values.tolist()

                start_index_size = self.num_indexed_docs()
                num_docs_deleted = self._delete_docs(index_writer, ids)
            

            if isinstance(df, sql.DataFrame):
                # project out unused columns
                df_size = df.count()
                df = df.select(self.config.id_col, *self.config.get_analyzed_fields())
                if df_size > self._index_build_chunk_size * 10:
                    # build large tables in parallel
                    # put temp indexes in temp dir for easy deleting later
                    with TemporaryDirectory() as tmp_dir_base:
                        tmp_dir_base = Path(tmp_dir_base)
                        
                        if force_distributed or (df_size > self._index_build_chunk_size * 50 and not disable_distributed):
                            # build with spark if very large and disributed build is allowed
                            dirs = self._build_spark(df, df_size, self.config, tmp_dir_base, show_progress_bar)
                        else:
                            # else just use local threads
                            dirs = self._build_parallel_local(df, self.config, tmp_dir_base, show_progress_bar)
                        # get the name of the index dir in each tmp sub dir
                        dirs = [self._get_index_dir(d) for d in dirs]

                        # merge the segments 
                        self._merge_index_segments(index_writer, dirs)
                        #kill_loky_workers()
                    # temp indexes deleted here
                else:
                    # table is small, build it single threaded
                    df = df.toPandas()

            if isinstance(df, pd.DataFrame):
                df_size = len(df)
                # if table is small just build directly
                self._add_docs(df, index_writer, show_progress_bar)
        except:
            index_writer.rollback()
            raise 
        else:
            index_writer.commit()
        finally:
            index_writer.close()

        self._is_built = True

        self._invalidate_spark()
        # verify the index is correct
        end_index_size = self.num_indexed_docs()
        if df_size - num_docs_deleted + start_index_size != end_index_size:
            raise RuntimeError(f'index build failed, number of indexed docs ({end_index_size}) is different than number'
                                f'expected df_size={df_size}, n_delete={num_docs_deleted}, start_size={start_index_size}')
    
    def num_indexed_docs(self):
        """
        get the number of indexed documents
        """
        self.init()
        n = self._index_reader.numDocs()

        return n

    @type_check_call
    def get_full_query_spec(self, cross_fields: bool=False):
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

    @type_check_call
    def search(self, doc: pd.Series | dict, query_spec: QuerySpec, limit: PositiveInt):
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
        
    @type_check_call
    def search_many(self, docs: pd.DataFrame, query_spec: QuerySpec, limit: PositiveInt):
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
