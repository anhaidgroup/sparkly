## Case Studies and Performance Statistics

A variation of Sparkly has been implemented in industry, as a part of a data integration platform software, and has been used by hundreds of customers (mostly enterprises), since 2022. 

As of 2024, other variations of Sparkly are being used to perform blocking for semantic matching tasks such as schema matching and linking table columns with ontology concepts for domain sciences. More details forthcoming. 

[This paper](https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf) reports that Sparkly can block large datasets at reasonable time and cost, e.g., blocking tables of 10M tuples under 100 minutes on an AWS cluster of 10 commodity nodes, costing only $12.5, and blocking tables of 26M tuples under 130 minutes on an AWS cluster of 30 nodes, costing $67.5.

We are also creating a benchmark to measure Sparkly's runtime performance on blocking large tables. We will report details here as they become available. 
