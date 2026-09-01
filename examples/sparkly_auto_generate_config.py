import json
from argparse import ArgumentParser

from pyspark.sql import SparkSession

from sparkly.index import LuceneIndex
from sparkly.index_optimizer import IndexOptimizer
from sparkly.recommended_config import RecommendedConfig
from sparkly.utils import local_parquet_to_spark_df


def main(args):
    SparkSession.builder \
        .appName("Sparkly-Auto-Generate-Config") \
        .getOrCreate()

    table_a = local_parquet_to_spark_df(args.table_a)
    table_b = table_a if args.table_b is None else local_parquet_to_spark_df(args.table_b)
    top_k = args.top_k

    index_optimizer = IndexOptimizer(is_dedupe=args.table_b is None)

    index_config = index_optimizer.make_index_config(table_a)

    index = LuceneIndex(args.index_dir, index_config, delete_if_exists=True)
    index.upsert_docs(table_a)

    query_spec_list = index_optimizer.optimize_topk(index, table_b, top_k)

    recommended_config = RecommendedConfig.from_components(index_config=index_config, query_specs=query_spec_list)

    with open(args.output_config, "w") as f:
        json.dump(recommended_config.to_dicts(), f, indent=4)

    print(f"Wrote recommended config to {args.output_config}")


if __name__ == "__main__":
    argp = ArgumentParser()

    argp.add_argument("--table_a", required=True)
    argp.add_argument("--table_b", required=False, default=None)
    argp.add_argument("--output_config", required=True)
    argp.add_argument("--index_dir", required=False, default="/tmp/lucene_index/")
    argp.add_argument("--top_k", required=False, type=int, default=1, help="Number of recommended configurations to generate",)

    main(argp.parse_args())