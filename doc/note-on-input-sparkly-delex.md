## Note on the Input for Sparkly and Delex Blockers

When running a Sparkly blocker on a cluster of machines (that is, nodes), a process running on a node typically needs to read data files (such as Tables A and B) into DataFrames. These DataFrames can be Pandas DataFrames. But more likely they are Spark DataFrames. 

To enable the reading of the data files, we should store the data files somewhere where the processes on any node can read them. There are several ways to do this: 
* We can store the data files on a cluster-aware file system such as HDFS.
* We can store the data files in a data store (such as PostgreSQL, MongoDB) that is cluster-aware, that is, that can be read from any node in the cluster.
* We can also replicate and store all data files in *each node* in the cluster. Then any process on a particular node will read the data files from the local disk. With this option, it is also important to note that each file must be stored in the same directory on each node. For example, if you store a file 'table_a.csv' in a folder with the path 'path/data' on one node, you must store the file 'table_a.csv' in a folder with the path 'path/data' on every other node as well.

The last option (storing the data files on all nodes in the cluster) is a bit cumbersome but is the simplest option. You can use this option to experiment with Sparkly. 

[Delex](https://github.com/anhaidgroup/delex) is a more complex blocking solution than Sparkly, and the above note also applies to running Delex on a cluster of machines. 
