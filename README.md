![license](https://img.shields.io/github/license/anhaidgroup/sparkly)

# Sparkly

Welcome to Sparkly! Sparkly is a TF/IDF top-k blocking for entity matching system built on
top of Apache Spark and PyLucene. 

## Paper and Data

A link to our paper can be found [here](https://pages.cs.wisc.edu/~anhai/papers1/sparkly-tr22.pdf).
Data used in the paper can be found [here](https://pages.cs.wisc.edu/~dpaulsen/sparkly_datasets/).



## Quick Start: Sparkly in 30 Seconds


There are three main steps to running Sparkly, 

1. Reading Data


```python
spark = SparkSession.builder.getOrCreate()

table_a = spark.read.parquet('./examples/data/abt_buy/table_a.parquet')
table_b = spark.read.parquet('./examples/data/abt_buy/table_b.parquet')
```

2. Index Building

```python
config = IndexConfig(id_col='_id')
config.add_field('name', ['3gram'])

index = LuceneIndex('/tmp/example_index/', config)
index.upsert_docs(table_a)
```

3. Blocking 

```python
query_spec = index.get_full_query_spec()

candidates = Searcher(index).search(table_b, query_spec, id_col='_id', limit=50)
candidates.show()
```

## Installing Dependencies 

### Python

Sparkly has been tested for python 3.10 on Ubuntu 22.04.

### PyLucene 

Unfortunately PyLucene is not available in pypi, to install PyLucene see 
[PyLucene docs](https://lucene.apache.org/pylucene/install.html). Sparkly has been 
tested with PyLucene 9.4.1.

### Other Requirements

This repo has a requirements file which will install the python 
packages, to install these dependencies simply use pip

`$ python3 -m pip install -r ./requirements.txt`

The requirements file will install pyspark with pip but any installation can be used 
as long as version 3.1.2 or greater is used.

## Tutorials

Examples of how to use sparkly are provided under the `examples` directory
in this repo. Additionally, we have a provided scripts for running Sparkly-Manual 
and Sparkly-Auto from the paper under the `experiments` directory. 

## How It Works 

Sparkly is built to do blocking for [entity matching](https://en.wikipedia.org/wiki/Record_linkage).
There have been many solutions developed to address this problem, from basic SQL joins to deep learning based approaches. 
Sparkly takes a top-k approach to blocking, in particular, each search record is 
paired with the top-k records with the highest [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) scores.
In terms of SQL this might look something like executing this query for each record,
```SQL 
SELECT id, BM25(<QUERY>, name) AS score 
FROM table_a 
ORDER BY score DESC
LIMIT <K>;
```

where `QUERY` derived from the search record. 

This kind of search is very common in information retrieval and keyword search applications. In fact, this is 
exactly what Apache Lucene is designed to do. While this form of search produces high quality results, it can also be very 
compute intensive, hence to speed up search, we leverage PySpark to distribute the computation. By using PySpark
we can easily leverage a large number of machines to perform search without having to rely on approximation algorithms.


## Experiments

Scripts for running Sparkly-Manual and Sparkly-Auto can be found in the `experiments` directory.

## API Docs

API docs can be found [here](https://derekpaulsen.github.io/sparkly/html/)

## Tips for Installing PyLucene

For tips on installing pylucene take a look at this [readme](https://github.com/anhaidgroup/sparkly/blob/main/tips/pylucene.md).
