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

Sparkly has been tested for Python 3.10 on Ubuntu 22.04.

### PyLucene 

Unfortunately PyLucene is not available in PyPI, to install PyLucene see 
[PyLucene docs](https://lucene.apache.org/pylucene/install.html). Sparkly has been 
tested with PyLucene 9.4.1.

### Other Requirements

Once PyLucene has been installed, Sparkly can be installed with pip by running the following
command in the root directory of this repository.

`$ python3 -m pip install .`

## Tutorials

To get started with Sparkly we recommend starting with the IPython notebook included with 
the repository [examples/example.ipynb](https://github.com/anhaidgroup/sparkly/blob/main/examples/example.ipynb).

Additional examples of how to use Sparkly are provided under the
[examples/](https://github.com/anhaidgroup/sparkly/tree/main/examples)
directory in this repository. 

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


## API Docs

API docs can be found [here](https://derekpaulsen.github.io/sparkly/html/)

## Tips for Installing PyLucene

For tips on installing PyLucene take a look at this [readme](https://github.com/anhaidgroup/sparkly/blob/main/tips/pylucene.md).
