from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from sparkly.index import  LuceneIndex
from sparkly.index_config import IndexConfig
from sparkly.query_generator import QuerySpec
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
# see LuceneIndex.ANALYZERS.keys() for currently implemented analyzers
analyzers = ['3gram', 'standard']

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
# add the 'name' column to be indexed with analyzers above
# note that this will create two sub indexes name.3gram and name.standard
# which can be searched independently
config.add_field('name', analyzers)
# do the same for the description
config.add_field('description', analyzers)
# create a new index stored at /tmp/example_index/
index = LuceneIndex('/tmp/example_index/', config)
# index the records from table A according to the config we created above
index.upsert_docs(table_a)
# this index now has 4 searchable subindexes each named '<FIELD_NAME>.<ANALYZER>', specifically
# 'name.3gram', 'name.standard', 'description.3gram', and 'description.standard'

# Pass a mapping of {<SEARCH FIELD> -> {<SUBINDEX NAME>, ...}}
# to create a QuerySpec which will specify how queries should be created for documents
query_spec = QuerySpec({
                # use name from table b to search name.3gram and description.standard in the index
                # notice that you can use any field to search any subindex, although 
                # typically you will just want to search the subindexes created by with 
                # the same column
                'name' : {'name.3gram', 'description.standard'},
                # use description from table_b to search description.standard in the index
                'description' : {'description.standard'}
            })

# kwargs can also be used like a python dict 
# this is equivalent to the spec above
query_spec = QuerySpec(
                name = {'name.3gram', 'description.standard'},
                description = {'description.standard'}
            )

# boost the contribution of BM25(b.name, a.name.3gram) by 2 times
# this changes the scores of documents from 
# score(b, a) = BM25(b.name, a.name.3gram) + BM25(b.name, a.description.standard) + BM25(b.description, a.description.standard) 
# to 
# score(b, a) = 2 * BM25(b.name, a.name.3gram) + BM25(b.name, a.description.standard) + BM25(b.description, a.description.standard) 
# 
# where b is the search record and a is the indexed record
# BM25 is a TF/IDF based similarity measure, for exact formula of BM25 : https://en.wikipedia.org/wiki/Okapi_BM25
query_spec.boost_map = {
        ('name', 'name.3gram') : 2.0
    }

# create a searcher for doing bulk search using our index
searcher = Searcher(index)
# search the index with table b
candidates = searcher.search(table_b, query_spec, id_col='_id', limit=limit).cache()

candidates.show()
# output is rolled up as 
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

ids = table_a.toPandas()['_id']
# delete a bunch of docs
index.delete_docs(ids[:500])
# update on spark cluster
index.to_spark()

# create a searcher for doing bulk search using our index
searcher = Searcher(index)
# search the index with table b
candidates = searcher.search(table_b, query_spec, id_col='_id', limit=limit).cache()

candidates.show()
# output is rolled up as 
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
# uncomment to save results to file
# candidates.toPandas().to_parquet('./candidates.parquet')

candidates.unpersist()
