## Examples of Running Sparkly

This directory contains examples of how to run Sparkly. In what follows we will give a brief overview of 
what is contained in each file. Before proceeding, ensure that Sparkly and its dependencies have been
installed. 

### example.ipynb

We recommend starting with [example.ipynb](https://github.com/anhaidgroup/sparkly/blob/main/examples/example.ipynb) in this 
directory. This notebook gives an overview of common configuration options for Sparkly. 

### basic\_example.py

As the name suggests, this is a minimal example of how to use Sparkly. This example uses a 
single analyzer and a single column to do blocking on the example dataset. 

### example.py

A slightly more complex example, this code creates an index on two fields (`name` and `description`) 
using multiple analyzers (`3gram` and `standard`). The example then demonstrates how to search using 
multiple sub indexes and columns.

### concat\_example.py

This example demonstrates how to use concatenated fields for indexing and search. For example, if 
you have a dataset with `first_name` and `last_name` as columns you may just want to combine them into a 
single attribute `name` for indexing and search.

### pandas\_example.py

This example demonstrates how to use Sparkly using only Pandas on a single thread.

### Running Sparkly-Manual 

To run Sparkly-Manual, you need to specify both the analyzer and 
columns to block on, as well as the dataset (which contains the two tables). For example, to run with the data provided in the 
repo you can run the command:

```bash
dir=./examples/data/abt_buy
spark-submit --master 'local[*]' \
		./experiments/manual.py \
		--table_a $dir/table_a.parquet \
		--table_b $dir/table_b.parquet \
		--gold $dir/gold.parquet \
		--analyzer 3gram \
		--k 50 \
		--blocking_columns 'name' \
		--output_file /tmp/abt_buy_manual_cands.parquet

```

### Running Sparkly-Auto

To run Sparkly-Auto, you only need to specify the dataset.
For example, to run with the data provided in the 
repo you can run the command:


```bash
dir=./examples/data/abt_buy
spark-submit --master 'local[*]' \
		./experiments/auto.py \
		--table_a $dir/table_a.parquet \
		--table_b $dir/table_b.parquet \
		--gold $dir/gold.parquet \
		--k 50 \
		--output_file /tmp/abt_buy_auto_cands.parquet

```
