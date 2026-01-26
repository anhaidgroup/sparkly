from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from sparkly.index import  LuceneIndex
from sparkly.index_config import IndexConfig
from sparkly.search import Searcher
from pathlib import Path

# the number of candidates returned per record
limit = 50
# path to the test data
data_path = (Path(__file__).parent / 'data' / 'abt_buy').absolute()
# table to be indexed
table_a_path = data_path / 'table_a.parquet'
# table for searching
table_b_path = data_path / 'table_b.parquet'
# the ground truth
gold_path = data_path / 'gold.parquet'
# the analyzers used to convert the text into tokens for indexing
analyzers = ['3gram']

# initialize a local spark context
spark = SparkSession.builder\
                    .master('local[*]')\
                    .appName('Sparkly Example')\
                    .getOrCreate()
# read all the data as spark dataframes
table_a = spark.read.parquet(f'file://{str(table_a_path)}')
table_b = spark.read.parquet(f'file://{str(table_b_path)}')
gold = spark.read.parquet(f'file://{str(gold_path)}')
# the index config, '_id' column will be used as the unique 
# id column in the index. Note id_col must be an integer (32 or 64 bit)
config = IndexConfig(id_col='_id')
# add the 'name' column to be indexed with analyzer above
config.add_field('name', analyzers)
# create a new index stored at /tmp/example_index/
index = LuceneIndex('/tmp/example_index/', config)
# index the records from table A according to the config we created above
index.upsert_docs(table_a)

# get a query spec (template) which searches on 
# all indexed fields
query_spec = index.get_full_query_spec()
# create a searcher for doing bulk search using our index
searcher = Searcher(index)
# search the index with table b
candidates = searcher.search(table_b, query_spec, id_col='_id', limit=limit).cache()

candidates.show()
# output is rolled up 
# search record id -> (indexed ids + scores + search time)
#
# explode the results to compute recall
pairs = candidates.select(
                    F.explode('ids').alias('a_id'),
                    F.col('_id').alias('b_id')
                )
# number of matches found
true_positives = gold.intersect(pairs).count()
# precentage of matches found
recall = true_positives / gold.count()

print(f'true_positives : {true_positives}')
print(f'recall : {recall}')

# saves the results to file
candidates.toPandas().to_parquet('./candidates.parquet')

candidates.unpersist()
