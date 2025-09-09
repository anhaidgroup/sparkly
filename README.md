## Sparkly: TF/IDF Blocking for Entity Matching

Sparkly is an open-source tool for the blocking step of entity matching. Entity matching finds tuples from two tables A and B that match, that is, refer to the same real-world entity. It typically proceeds in two steps. *The blocking step* uses heuristics to quickly identify a relatively small set of tuple pairs that can be matches. *The matching step* applies a (rule- or learning-based) matcher to each surviving pair to predict match/no-match. (See [this page](https://anhaidgroup.github.io/magellan/about) for details.)

Sparkly focuses on the blocking step, and is distinguished in four aspects: 

* It can scale to large tables, for example, with tens of millions or hundreds of millions of tuples per table.
* It uses the TF/IDF similarity measure to perform blocking. 
* It outperforms many state-of-the-art blocking solutions. See [this paper](https://pages.cs.wisc.edu/~anhai/papers1/sparkly-vldb2023.pdf) for details. 
* Variations of Sparkly have been implemented in industry and used by hundreds of enterprises.

Under some conditions, you may want to use Delex, an even more powerful blocking solution, which allows you to combine multiple blocking strategies, including TF/IDF. See the [Delex homepage](https://github.com/anhaidgroup/delex/tree/main) for more details, including when to use Sparkly versus Delex. 

### How Sparkly Works

Let A be the smaller table (the one with fewer tuples). For each tuple b in Table B, Sparkly finds k tuples in Table A that have the highest TF/IDF similarity score with tuple b. Let these tuples be a<sub>1</sub>, a<sub>2</sub>, ..., a<sub>k</sub>. Then Sparkly returns the tuple pairs (a<sub>1</sub>,b), (a<sub>2</sub>,b), ..., (a<sub>k</sub>,b) as potential matches in its output. TF/IDF is a similarity score commonly used in text document retrieval and keyword search on the Web. Many TF/IDF variations exist. Sparkly uses the well-known BM25 variation. 

Implementation-wise, Sparkly builds an index over the tuples in Table A, then uses this index and a Spark cluster to perform the top-k tuple findings fast. Sparkly uses Lucene to build indexes and perform top-k search. See [the paper](https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf) for details. 

### Case Studies and Performance Statistics

Sparkly can block tables of tens of millions of tuples in hours on relatively small clusters of machines. It scales to hundreds of millions of tuples. See [this page](https://github.com/anhaidgroup/sparkly/blob/main/doc/case-studies-n-performance-stats.md) for details. 

### Installation

See instructions to install Sparkly on [a single machine](https://github.com/anhaidgroup/sparkly/blob/main/doc/install-single-machine.md) or on [a cluster of machines](https://github.com/anhaidgroup/sparkly/blob/main/doc/install-cluster-machines.md). 

### How to Use

See [this page](https://github.com/anhaidgroup/sparkly/blob/main/doc/using-sparkly.md), which points to a technical report, slides, a tutorial, and examples.

### Further Pointers

See [API documentation](https://anhaidgroup.github.io/sparkly/). 
For questions / comments, contact [AnHai Doan](mailto:anhai@cs.wisc.edu).

