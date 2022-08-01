# Sparkly

Welcome to Sparkly! Sparkly is a TF/IDF top-k blocking for entity matching system built on
top of Apache Spark and PyLucene. 

## Installing Dependencies 

### Python

Sparkly requires python 3.6+

### PyLucene 

Unfortunately PyLucene is not available in pypi, to install PyLucene see 
[PyLucene docs](https://lucene.apache.org/pylucene/install.html)

### Other Requirements

This repo has a requirements file which will install the python 
packages, to install these dependencies simply use pip

`$ python3 -m pip install -r ./requirements.txt`

The requirements file will install pyspark with pip but any installation can be used 
as long as version 3.1.2 or greater is used.

## Running Sparkly

Examples of how to use sparkly are provided under the `examples` folder 
in this repo. See these examples for usage.

## How It Works 

Sparkly is built for blocking for [entity matching](https://en.wikipedia.org/wiki/Record_linkage).
There have been many solutions this this problem from basic SQL joins to deep learning based. 
Sparkly takes a top-k approach to blocking, in particular, each search record is 
paired with the top-k records with the highest [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) scores.
In a SQL query this might look something like,
```SQL 
SELECT id, BM25(<QUERY>, name) AS score 
FROM table_a 
ORDER BY score DESC
LIMIT <K>;
```

This kind of search is very common in information retrieval and keyword search applications. In fact, this is 
exactly what Apache Lucene is designed to do. While this form of search is very powerful, it can also be very 
compute intensive, hence to speed up search, we leverage PySpark to distribute the computation. By using PySpark
we can easily leverage a large number of machines to perform search without having to rely on approximation algorithms.


