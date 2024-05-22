from contextlib import contextmanager
from tempfile import mkdtemp, mkstemp
import shutil
import os
import time
import sys
import logging
import warnings
import re
from pathlib import Path
from zipfile import ZipFile

import psutil
import pandas as pd
import numpy as np
from joblib.externals.loky import get_reusable_executor

from pyspark import SparkContext
from pyspark import StorageLevel
from pyspark.sql import SparkSession
import pyspark.sql.types as T
# needed to find the parquet module
import pyarrow.parquet
import pyarrow as pa
from pyarrow.parquet import ParquetFile
import lucene
import numba as nb


logging.basicConfig(
        stream=sys.stderr,
        format='[%(filename)s:%(lineno)s - %(funcName)s() ] %(asctime)-15s : %(message)s',
)
logger = logging.getLogger(__name__)



def get_index_name(n, *postfixes):
    """
    utility function for generating index names in a uniform way
    """
    s = n.lower().replace('-', '_')
    if len(postfixes) != 0:
        s += '_' + '_'.join(postfixes)
    return s



class Timer:
    """
    utility class for timing execution of code
    """

    def __init__(self):
        self.start_time = time.time()
        self._last_interval = time.time()

    def get_interval(self):
        """
        get the time that has elapsed since the object was created or the 
        last time get_interval() was called

        Returns
        -------
        float
        """
        t = time.time()
        interval = t - self._last_interval
        self._last_interval = t
        return interval

    def get_total(self):
        """
        get total time this Timer has been alive

        Returns
        -------
        float
        """
        return time.time() - self.start_time

    def set_start_time(self):
        """
        set the start time to the current time
        """
        self.start_time = time.time()


def get_logger(name, level=logging.DEBUG):
    """
    Get the logger for a module

    Returns
    -------
    Logger

    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    return logger

'''
AUC for an array sorted in decending order
slightly different results to np.trapz due to 
FP error
'''
@nb.njit('float32(float32[::1])')
def auc(x):
    return x[1:].sum() + (x[0] - x[-1]) / 2

'''
AUC for an array sorted in decending order
slightly different results to np.trapz due to 
FP error
'''
@nb.njit('float32(float32[::1])')
def norm_auc(x):
    return (x[1:].sum() + (x[0] - x[-1]) / 2) / len(x)


def atomic_unzip(zip_file_name, output_loc):
    """
    atomically unzip the file, that is this function is safe to call 
    from multiple threads at the same time

    Parameters
    ----------
    zip_file_name : str
        the name of the file to be unzipped

    output_loc : str
        the location that the file will be unzipped to
    """

    out = Path(output_loc).absolute()
    lock = Path(str(out) + '.lock')
    tmp_out = Path(str(out) + '.tmp_out')

    if out.exists():    
        return    
    
    try:    
        # try to acquire the lock
        # throws FileExistsError if someone else grabbed it
        lock_fd = os.open(str(lock.absolute()), os.O_EXCL | os.O_CREAT)  
        #os.fsync(lock_fd)
        #fd = os.open(str(out.parent.absolute()), os.O_RDONLY)
        #os.fsync(fd)
        #os.close(fd)
        try:    
            # unzip the dir if it doesn't exist
            if not out.exists():    
                with ZipFile(zip_file_name, 'r') as zf:
                    zf.extractall(str(tmp_out))
                # move to final output location
                tmp_out.rename(out)
        finally:    
            # release the lock
            os.close(lock_fd)
            lock.unlink()    
    
    except FileExistsError:    
        # failed to get lock 
        # wait for other thread to do the unzipping
        while lock.exists():    
            pass    
        # something is wrong if the lock was released but the dir
        # wasnt' created by someone else
        if not out.exists():    
            raise RuntimeError('atomic unzip failed for {f}')


def _add_file_recursive(zip_file, base, file):
    if file.is_dir():
        for f in file.iterdir():
            _add_file_recursive(zip_file, base, f)
    else:
        zip_file.write(file, arcname=file.relative_to(base))

def zip_dir(d, outfile=None):
    """
    Zip a directory `d` and output it to `outfile`. If 
    `outfile` is not provided, the zipped file is output in /tmp

    Parameters
    ----------
    d : str or Path
        the directory to be zipped

    outfile : str or Path, optional
        the output location of the zipped file

    Returns
    -------
    Path 
        the path to the new zip file

    """
    p = Path(d)

    tmp_zip_file = Path(outfile) if outfile is not None else Path(mkstemp(prefix=d.name, suffix='.zip')[1])

    with ZipFile(tmp_zip_file, 'w') as zf:
        _add_file_recursive(zf, p, p)

    return tmp_zip_file

def init_jvm(vmargs=[]):
    """
    initialize the jvm for PyLucene

    Parameters
    ----------
    vmargs : list[str]
        the jvm args to the passed to the vm
    """
    if not lucene.getVMEnv():
        lucene.initVM(vmargs=['-Djava.awt.headless=true'] + vmargs)

def attach_current_thread_jvm():
    """
    attach the current thread to the jvm for PyLucene
    """
    env = lucene.getVMEnv()
    env.attachCurrentThread()

def invoke_task(task):
    """
    invoke a task created by joblib.delayed
    """
    # task == (function, *args, **kwargs)
    return task[0](*task[1], **task[2])


@contextmanager
def persisted(df, storage_level=StorageLevel.MEMORY_AND_DISK):
    """
    context manager for presisting a dataframe in a with statement.
    This automatically unpersists the dataframe at the end of the context
    """
    if df is not None:
        df = df.persist(storage_level) 
    try:
        yield df
    finally:
        if df is not None:
            df.unpersist()

def is_persisted(df):
    """
    check if the pyspark dataframe is persist
    """
    sl = df.storageLevel
    return sl.useMemory or sl.useDisk


def repartition_df(df, part_size, by=None):
    """
    repartition the dataframe into chunk of size 'part_size'
    by column 'by'
    """
    cnt = df.count()
    n = max(cnt // part_size, SparkContext.getOrCreate().defaultParallelism * 4)
    n = min(n, cnt)
    if by is not None:
        return df.repartition(n, by)
    else:
        return df.repartition(n)



def is_null(o):
    """
    check if the object is null, note that this is here to 
    get rid of the weird behavior of np.isnan and pd.isnull
    """
    r = pd.isnull(o)
    return r if isinstance(r, bool) else False



_loky_re = re.compile('LokyProcess-\\d+')    
def _is_loky(c):    
    return any(map(_loky_re.match, c.cmdline()))    
    
def kill_loky_workers():
    """
    kill all the child loky processes of this process. 
    used to prevent joblib from sitting on resources after using 
    joblib.Parallel to do computation
    """
    '''
    get_reusable_executor().shutdown(wait=True, kill_workers=True)
    return 
    '''

    proc_killed = False
    parent_proc = psutil.Process()    
    for c in parent_proc.children(recursive=True):    
        if _is_loky(c):    
            c.terminate()    
            c.wait()
            proc_killed = True

    if not proc_killed:
        warnings.warn('kill_loky_workers invoked but no processes were killed', UserWarning)


 
def spark_to_pandas_stream(df, chunk_size, by='_id'):
    """
    repartition df into chunk_size and return as iterator of 
    pandas dataframes
    """
    df_size = df.count()
    batch_df = df.repartition(max(1, df_size // chunk_size), by)\
            .rdd\
            .mapPartitions(lambda x : iter([pd.DataFrame([e.asDict(True) for e in x]).convert_dtypes()]) )\
            .persist(StorageLevel.DISK_ONLY)
    # trigger read
    batch_df.count()
    for batch in batch_df.toLocalIterator(True):
        yield batch

    batch_df.unpersist()

def type_check(var, var_name, expected):
    """
    type checking utility, throw a type error if the var isn't the expected type
    """
    if not isinstance(var, expected):
        raise TypeError(f'{var_name} must be type {expected} (got {type(var)})')

def type_check_iterable(var, var_name, expected_var_type, expected_element_type):
    """
    type checking utility for iterables, throw a type error if the var isn't the expected type
    or any of the elements are not the expected type
    """
    type_check(var, var_name, expected_var_type)
    for e in var:
        if not isinstance(e, expected_element_type):
            raise TypeError(f'all elements of {var_name} must be type{expected_element_type} (got {type(var)})')




_PYARROW_TO_PYSPARK_TYPE = {
        pa.int32() : T.IntegerType(),
        pa.int64() : T.LongType(),
        pa.float32() : T.FloatType(),
        pa.float64() : T.DoubleType(),
        pa.string() : T.StringType(),
        pa.bool_() : T.BooleanType(),
}

_PYARROW_TO_PYSPARK_TYPE.update([(pa.list_(a), T.ArrayType(s)) for a,s in _PYARROW_TO_PYSPARK_TYPE.items()])

def _arrow_schema_to_pyspark_schema(schema):
    fields = []
    for i in range(len(schema)):
        af = schema.field(i)
        fields.append(
                T.StructField(af.name, _PYARROW_TO_PYSPARK_TYPE[af.type])
        )

    return T.StructType(fields)




def local_parquet_to_spark_df(file):
    file = Path(file).absolute()
    spark = SparkSession.builder.getOrCreate()
        
    pf = ParquetFile(file)
    # get the schema for the dataframe
    arrow_schema = pf.schema_arrow
    schema = _arrow_schema_to_pyspark_schema(arrow_schema)
    pdf = pd.read_parquet(file)
    df = spark.createDataFrame(pdf, schema=schema, verifySchema=False)\

    return df
            

