# Examples

This directory contains examples of how to use Sparkly, here we will give a brief overview of 
what is contained in each file. Each example should be run out of the root directory to 
ensure that the test data is loaded correctly. For example, 

```
$ python3 ./examples/basic_example.py
```


## basic_example.py

As the name suggests this is a minimal example of how to use Sparkly. This example uses a 
single analyzer and a single column to do blocking on the example dataset. 


## example.py

A slightly more complex example, this code creates an index on two fields (`name` and `description`) 
using mutliple analyzers (`3gram` and `standard`). The example then demonstrates how to search using 
multiple sub indexes and columns.



## concat_example.py

This example demonstrates how to use concatenated fields for indexing and search. For example, if 
you have a dataset with `first_name` and `last_name` as columns you may just want to combine them into a 
single attribute `name` for indexing and search.


## local_example.py

This example demonstrates how to use Sparkly using only Pandas on a single thread.
