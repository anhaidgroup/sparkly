# Running Manual 

To run sparkly manual, you need to specify both the analyzer and 
columns to block on, as well as the dataset. For example, to run with the data provided in the 
repo you can run the command,

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


# Running Auto

To run sparkly manual, you only need to specify the dataset.
For example, to run with the data provided in the 
repo you can run the command,


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
