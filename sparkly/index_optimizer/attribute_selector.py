import sys
from utils import init_spark, read_table_spark, repartition_df, Timer
from collections import Counter
from sparkly.utils import init_jvm
from sparkly.index import LuceneIndex
from sparkly.analysis import analyze
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import pyspark.sql.functions as F
import pyspark.sql.types as T
import mmh3
import numba as nb
from pprint import pformat 
from pathlib import Path
from pyspark import SparkFiles, SparkContext
from tempfile import mkstemp


class MmapArray:

    def __init__(self, file_name, shape, dtype):
        self._file = Path(file_name)
        self._shape = shape
        self._dtype = dtype
        self._arr = None
        self._on_spark = False
    

    @classmethod
    def from_array(cls, arr, file=None):
        if file is None:
            file = mkstemp(suffix='.mmap')[1]

        obj = cls(file, arr.shape, arr.dtype)
        obj.open(mode='w+')
        obj.arr[:] = arr[:]
        obj.close()
        return obj


    @property
    def arr(self):
        return self._arr
    
    def close(self):
        if self._arr is not None:
            self._arr.flush()
            self._arr = None

    def open(self, mode='r+'):
        fname = str(self._file.absolute()) if not self._on_spark else SparkFiles.get(self._file.name)
        self._arr = np.memmap(
                        filename=fname,
                        dtype=self._dtype,
                        mode=mode,
                        shape=self._shape,  
                    )

    def to_spark(self):
        if not self._on_spark:
            self.close()
            SparkContext.getOrCreate().addFile(str(self._file))
            self._on_spark = True
        else:
            raise RuntimeError('to_spark already called')





class BM25Scorer:

    def __init__(self, b=.75, k1=1.2):
        self._dim = None
        self._hash_func = None
        self._hashes = None
        self._idfs = None
        self._b = np.float32(b)
        self._k1 = np.float32(k1)
        self._avg_doc_len = None
        

    def init(self):
        self._idfs.open()
        self._hashes.open()
    
    def deinit(self):
        self._idfs.close()
        self._hashes.close()

    @classmethod
    def from_doc_freqs(cls, doc_freqs, avg_doc_len, corpus_size, b=.75, k1=1.2):
        v = cls(b, k1)
        doc_freqs.sort_index(inplace=True)
        v._avg_doc_len = np.float32(avg_doc_len)
        
        v._idfs = MmapArray.from_array(_bm25_idf(doc_freqs.values, corpus_size))
        v._idfs.to_spark()

        v._hashes = MmapArray.from_array(doc_freqs.index.values)
        v._hashes.to_spark()
        v._dim = len(doc_freqs)

        return v
        
    def score_bm25(self, hashes):
        if hashes is None or len(hashes) == 0:
            return 0.0

        hashes, tf = np.unique(hashes, return_counts=True)
        return _score_bm25(self._hashes.arr, self._idfs.arr, self._b, self._k1, self._avg_doc_len, hashes, tf)

@nb.njit('float32[:](int64[::1], int64)')
def _bm25_idf(doc_freqs, corpus_size):
    return np.log( ((corpus_size + .5 - doc_freqs) / (doc_freqs + .5)) + 1 ).astype(np.float32)

@nb.njit( 'float32(int64[::1], float32[::1], float32, float32, float32, int64[::1], int64[::1])')
def _score_bm25(hash_idx, idfs, b, k1, avg_doc_len, hashes, tf):
    # self._hashes is also sorted
    idxes = np.searchsorted(hash_idx, hashes)
    #if np.any(hash_idx[idxes] != hashes):
    #    raise ValueError('unknown hash')

    doc_len = tf.sum()

    idf = idfs[idxes]
    bottom = tf  + (k1 * ((1 - b) + (doc_len * (b / avg_doc_len))))
    tf = tf * (k1 + 1) 
    tf *= idf
    tf /= bottom 
    return tf.sum() / doc_len


class LuceneAnalyzer:

    def __init__(self, name, ctor):
        self._name = name
        self._ctor = ctor
        self._analyzer = None

    
    @property
    def name(self):
        return self._name

    def init(self):
        if self._analyzer is None:
            init_jvm(['-Xmx500m'])
            self._analyzer = self._ctor()

    def __call__(self, s):
        if isinstance(s, str):
            return analyze(self._analyzer, s)
        else:
            return None

    def column_name(self, base):
        return f'{self._name}({base})'




class AttributeSelector:
    
    def __init__(self, analyzer_names = ('standard', '3gram'), b=.75, k1=1.2, word_limit=50):
        self._analyzers = []
        for k in analyzer_names:
            self._analyzers.append(LuceneAnalyzer(k, LuceneIndex.ANALYZERS[k]))
        # the maximum average number of whitespace delimited tokens 
        # in a column that is scored
        self._word_limit = word_limit

        self._b = b
        self._k1 = k1
        self.time_ = {}
        self.column_stats_ = None


    def _init_column_stats(self, df):
        self.column_stats_ = pd.DataFrame(index=df.columns)
        self.column_stats_['is_null'] = df.select([F.when(df[c].isNull(), 1).otherwise(0).alias(c) for c in df.columns])\
                                            .agg(*[F.mean(c).alias(c) for c in df.columns])\
                                            .toPandas().iloc[0]

    def _count_average_words(self, df, exclude):
        column_names = [c for c in df.columns if c not in exclude]
        cols = [ F.when(F.col(c).isNotNull(), F.size(F.split(F.col(c).cast("string"), "\\s+"))).alias(c) for c in column_names]
        avg = df.select(cols)\
                .agg(*[F.mean(c).alias(c) for c in column_names])\
                .toPandas()\
                .iloc[0]

        return avg

    def _select_word_limit(self, df, word_limit, exclude):
        avg_word_cnts = self._count_average_words(df, exclude)
        self.column_stats_['avg_word_count'] = avg_word_cnts
        # take columns that are less than or equal to the word limit
        df = df.select(*avg_word_cnts.index[avg_word_cnts.values <= word_limit], *exclude)
        return df

    def _sample_dataframe(self, df, sample_fraction):
        if sample_fraction == 1.0:
            return df
        else:
            return df.sample(fraction=sample_fraction, withReplacement=False)
        
    def select_columns(self, df, k, id_col='_id', sample_fraction=1.0):
        exclude = {id_col}
        df = repartition_df(df, 1000, id_col)
        timer = Timer()
        self._init_column_stats(df)
        df = self._select_word_limit(df, self._word_limit, exclude)
        self.time_['select_word_limit'] = timer.get_interval()

        df = self._sample_dataframe(df, sample_fraction)

        df = self._select_bm25(df, k, exclude)

        self.time_['select_bm25'] = timer.get_interval()
        self.time_['total'] = timer.get_total()

        return df

    def _select_bm25(self, df, k, exclude):
        if len(df.columns)  - len(exclude) <= k:
            return df

        column_scores = self._score_bm25(df, exclude)
        self.column_stats_ = pd.concat([self.column_stats_, column_scores], axis=1)\
                                .sort_values('score', ascending=False)
        # take the topk largest scores
        df = df.select(*self.column_stats_.head(k).index)
        return df


    def _score_bm25(self, df, exclude):
        # cast to string
        df = df.select([df[c].cast('string').alias(c) for c in df.columns if c not in exclude])
        
        timer = Timer()
        # tokenizer columns
        out_cols = [a.column_name(c) for a in self._analyzers for c in df.columns]
        out_schema = T.StructType([ T.StructField(c, T.ArrayType(T.LongType()), True) for c in out_cols])
        tokens = df.mapInPandas(self.tokenize_and_hash_columns, schema=out_schema)\
                    .persist()

        tokens.count()
        self.time_['tokenize_for_bm25'] = timer.get_interval()

        vecs = self.build_vectorizers(tokens)
        
        self.time_['build_for_bm25'] = timer.get_interval()
        # take the mean bm25 score
        out_schema = T.StructType([ T.StructField(c, T.DoubleType(), True) for c in out_cols])
        scores = tokens.mapInPandas(lambda x : self.score_columns(x, vecs), schema=out_schema)\
                        .agg(*[F.mean(c).alias(c) for c in out_cols])\
                        .toPandas().iloc[0]
        
        self.time_['score_for_bm25'] = timer.get_interval()
        tokens.unpersist()
        
        rows = []
        for c in df.columns:
            row = {a.name : scores[a.column_name(c)] for a in self._analyzers}
            row['column'] = c
            rows.append(row)

        column_scores = pd.DataFrame(rows).set_index('column')
        column_scores['score'] = column_scores.sum(axis=1).values

        return column_scores

    def tokenize_and_hash_columns(self, df_itr):
        for a in self._analyzers:
            a.init()
        
        out = {}
        for df in df_itr:
            out.clear()
            for a in self._analyzers:
                for c in df.columns:
                    # analyze each string and hash the tokens
                    out[a.column_name(c)] = df[c].apply(lambda x : self.murmur64_list(a(x)) if x is not None else None)

            d = pd.DataFrame(out)
            yield d

    @staticmethod
    def count_tokens(df_itr):
        counters = None
        for df in df_itr:
            if counters is None:
                counters = [Counter() for _ in range(len(df.columns))]

            for ctr, c in zip(counters, df.columns):
                ctr.update( (x for l in df[c].values if l is not None for x in set(l)) )

        if counters is not None:
            for i, ctr in enumerate(counters):
                df = pd.DataFrame(list(ctr.items()), columns=['tok', 'count'])
                df['count'] = df['count'].astype(np.int32)
                df['col_idx'] = np.int32(i)
                yield df

    
    def build_vectorizers(self, toks):
        corpus_size = toks.count()
        columns = toks.columns

        doc_freqs = toks.mapInPandas(self.count_tokens, schema='tok long, count long, col_idx int')\
                        .groupby('col_idx', 'tok')\
                        .agg(F.sum('count').alias('count'))\
                        .toPandas()

        doc_freqs.set_index(['col_idx', 'tok'], inplace=True)
        doc_freqs.sort_index(inplace=True)
        doc_freqs = doc_freqs['count']

        avg_doc_lens = toks.select([F.when(F.col(c).isNotNull(), F.size(c)).alias(c) for c in toks.columns])\
                            .agg(*[F.mean(c).alias(c) for c in toks.columns])\
                            .toPandas()\
                            .iloc[0]
        vectorizers = {}
        for i, c in enumerate(columns):
            if i in doc_freqs.index:
                vectorizer = BM25Scorer.from_doc_freqs(doc_freqs.loc[i], avg_doc_lens.iat[i], corpus_size, self._b, self._k1)
            else:
                vectorizer = None
            vectorizers[c] = vectorizer


        return vectorizers

    @staticmethod
    def score_columns(df_itr, vecs):
        for v in vecs.values():
            if v is not None:
                v.init()

        for df in df_itr:
            for c in df.columns:
                vec = vecs[c]
                scores = np.nan if vec is None else df[c].apply(vec.score_bm25)
                df[c] = scores
            yield df

    @staticmethod
    def murmur64_list(x):
        if x is not None:
            return np.fromiter(map(AttributeSelector.murmur64, x), dtype=np.int64, count=len(x))

    @staticmethod
    def murmur64(x):
        return mmh3.hash64(x)[0]
