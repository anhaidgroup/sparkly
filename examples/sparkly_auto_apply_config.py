from argparse import ArgumentParser
import json

import pyspark.sql.functions as F
from pyspark.sql import SparkSession

from sparkly.index import LuceneIndex
from sparkly.recommended_config import RecommendedConfig
from sparkly.search import Searcher
from sparkly.utils import local_parquet_to_spark_df


def compute_recall(candidates, gold, is_dedupe):
    pairs = candidates.select(
        F.explode("id1_list").alias("a_id"),
        F.col("id2").alias("b_id"),
    )

    if is_dedupe:
        pairs = pairs.select(
            F.least("a_id", "b_id").alias("a_id"),
            F.greatest("a_id", "b_id").alias("b_id"),
        ).drop_duplicates()

    true_positives = gold.intersect(pairs).count()
    recall = true_positives / gold.count()

    print(f"true_positives : {true_positives}")
    print(f"recall : {recall}")


def main(args):
    SparkSession.builder \
        .appName("Sparkly-Auto-Apply-Config") \
        .getOrCreate()

    table_a = local_parquet_to_spark_df(args.table_a)
    table_b = table_a if args.table_b is None else local_parquet_to_spark_df(args.table_b)

    with open(args.config) as f:
        recommended_config = RecommendedConfig.from_json(json.load(f))

    index_config, query_spec = recommended_config.to_components()[0]

    index = LuceneIndex(
        args.index_dir,
        index_config,
        delete_if_exists=True,
    )

    index.upsert_docs(table_a)

    searcher = Searcher(index)

    candidates = searcher.search(
        table_b,
        query_spec,
        id_col=args.id_col,
        limit=args.k,
    ).persist()

    candidates.count()
    candidates.show()

    if args.gold:
        gold = local_parquet_to_spark_df(args.gold)
        compute_recall(candidates, gold, is_dedupe=args.table_b is None)
    else:
        print("gold not provided, skipping computing recall")

    if args.output_file:
        candidates.toPandas().to_parquet(args.output_file, index=False)

    candidates.unpersist()


if __name__ == "__main__":
    argp = ArgumentParser()

    argp.add_argument("--k", type=int, required=False, default=50)
    argp.add_argument("--table_a", required=True)
    argp.add_argument("--table_b", required=False, default=None)
    argp.add_argument("--gold", required=False, default=None)
    argp.add_argument("--output_file", required=False, default=None)
    argp.add_argument("--config", required=True)
    argp.add_argument("--index_dir", required=False, default="/tmp/lucene_index/")
    argp.add_argument("--id_col", required=False, default="_id")

    main(argp.parse_args())