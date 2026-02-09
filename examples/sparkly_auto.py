import pandas as pd
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from sparkly.index import  LuceneIndex
from sparkly.index_config import IndexConfig
from sparkly.index_optimizer import IndexOptimizer
from sparkly.search import Searcher
from sparkly.utils import local_parquet_to_spark_df
from sparkly.utils import check_tables_auto
from argparse import ArgumentParser

argp = ArgumentParser()
# blocking config
argp.add_argument('--k', type=int, required=False, default=50)
# 'standard' and '3gram' and the only two options right now
argp.add_argument('--table_a', type=str, required=True, help='the table to be indexed')
argp.add_argument('--table_b', type=str, required=False, default=None, help='the table used for search')
argp.add_argument('--gold', type=str, required=False, help='the ground truth for the dataset')
argp.add_argument('--output_file', type=str, required=False, help='where the parquet file of the script output will be placed')

args = argp.parse_args()

def main(args):
    # the number of candidates returned per record
    limit = args.k

    # initialize a local spark context
    spark = SparkSession.builder\
                        .appName('Sparkly-Auto')\
                        .getOrCreate()
    # read all the data as spark dataframes
    table_a = local_parquet_to_spark_df(args.table_a)
    table_b = table_a if args.table_b is None else local_parquet_to_spark_df(args.table_b)

    # check that the tables fit the expected format
    check_tables_auto(table_a, '_id', table_b, '_id')



    index_optimizer = IndexOptimizer(args.table_b is None)
    config = index_optimizer.make_index_config(table_a)

    # create a new index stored at /tmp/example_index/
    index = LuceneIndex('/tmp/lucene_index/', config)
    # index the records from table A according to the config we created above
    index.upsert_docs(table_a)

    # get a query spec (template) which searches on 
    # all indexed fields
    query_spec = index_optimizer.optimize(index, table_b)
    # create a searcher for doing bulk search using our index
    searcher = Searcher(index)
    # search the index with table b
    candidates = searcher.search(table_b, query_spec, id_col='_id', limit=limit).persist()
    
    candidates.count()
    candidates.show()
    # output is rolled up 
    # search record id -> (indexed ids + scores + search time)
    #
    # explode the results to compute recall
    
    if args.gold:
        gold = local_parquet_to_spark_df(args.gold)

        pairs = candidates.select(
                            F.explode('id1_list').alias('a_id'),
                            F.col('id2').alias('b_id')
                        )
        # dedupe case
        if args.table_b is None:
            pairs = pairs.select(
                    F.least('a_id', 'b_id').alias('a_id'),
                    F.greatest('a_id', 'b_id').alias('b_id')
                    ).drop_duplicates()

        # number of matches found
        true_positives = gold.intersect(pairs).count()
        # precentage of matches found
        recall = true_positives / gold.count()

        print(f'true_positives : {true_positives}')
        print(f'recall : {recall}')
    else:
        print('gold not provided, skipping computing recall')

    # write output locally
    if args.output_file:
        candidates.toPandas().to_parquet(args.output_file, index=False)

    candidates.unpersist()

if __name__ == '__main__':
    main(argp.parse_args())

