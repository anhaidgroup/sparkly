from sparkly.index import  LuceneIndex
from sparkly.index_config import IndexConfig
from sparkly.search import Searcher
from pathlib import Path
import pandas as pd

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

# read all the data as spark dataframes
table_a = pd.read_parquet(table_a_path)
table_b = pd.read_parquet(table_b_path)
gold =  pd.read_parquet(gold_path)
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
# returns a pandas dataframe with the schema
# (id2,id1_list, scores, search_time) 
# where id2 is the id of the search record and id1_list is the list of indexed ids for the search record
candidates = index.search_many(table_b.set_index('_id'), query_spec, limit).reset_index()

print(candidates.head(10))
# output is rolled up 
# search record id -> (indexed ids + scores + search time)
#
# explode the results to compute recall
pairs = candidates.drop(columns=['scores', 'search_time'])\
                .explode('id1_list')\
                .rename(columns={'id1_list' : 'a_id', 'id2' : 'b_id'})
# convert to sets of tuples for computing recall
blocking_out = set(pairs[['a_id', 'b_id']].itertuples(name=None, index=False))
gold_set = set(gold.itertuples(name=None, index=False))
# number of matches found
true_positives = len(gold_set & blocking_out)
# precentage of matches found
recall = true_positives / len(gold)

print(f'true_positives : {true_positives}')
print(f'recall : {recall}')
