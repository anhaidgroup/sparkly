![license](https://img.shields.io/github/license/anhaidgroup/sparkly)

# Sparkly

Welcome to Sparkly! Sparkly is a TF/IDF top-k blocking for entity matching system built on
top of Apache Spark and PyLucene. 

## Installing Dependencies 

### Python

Sparkly has been tested for python3.10 on Ubuntu 22.04.

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

Examples of how to use sparkly are provided under the `examples` directory
in this repo. See these examples for usage.

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


## Data and Experiments

Scripts for running Sparkly-Manual and Sparkly-Auto can be found in the `experiments` directory.

Data used in the paper can be found [here](https://pages.cs.wisc.edu/~dpaulsen/sparkly_datasets/)


## Tips for Installing PyLucene

In order to install PyLucene, you must first install JCC. JCC requires 
a C++ compiler to compile the extension before it is installed. During this
install look for *non-fatal* errors and warnings. In some cases the extensions are 
not compiled for the correct architecture, resulting in non-fatal linker errors, 
which will prevent PyLucene from installing correctly.
If when running the `python3 setup.py build` the extension is being compiled with the 
incorrect CPU architecture, you may need to manually specify the architecture by appending 
them to `cflags` and `lflags` in `setup.py`. For example, you may want to add the lines
```python

cflags += ['-arch', 'arm64']
lflags += ['-arch', 'arm64']
```

before the `Extension` constructor is called in the script.


### MacOS

In MacOS High Sierra and later sometimes MacOS will kill pyspark workers when the JVM is initialized for PyLucene. If 
`examples/local_example.py` runs but `examples/basic_example.py` crashes, try modifying the enviorment variables 
as described in this [stack overflow post](https://stackoverflow.com/questions/50168647/multiprocessing-causes-python-to-crash-and-gives-an-error-may-have-been-in-progr).
