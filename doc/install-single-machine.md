## How to Install Sparkly on a Single Machine

You can install and run Sparkly on a single machine. If you have relatively small tables, or just want to experiment with Sparkly, this is often sufficient. Otherwise you may consider installing and running Sparkly on [an on-premise cluster]() or [a cluster on cloud]() (such as AWS). 

### Installing Python 3

To install and run Sparkly on a single machine, first you need to have Python 3 installed (if not already). You can find instructions for downloading and installing Python 3 here:
[https://wiki.python.org/moin/BeginnersGuide/Download](https://wiki.python.org/moin/BeginnersGuide/Download)

We strongly recommend installing Python 3.10 if possible, as we have tested Sparkly extensively with Python 3.10. 

### Pip Install from PyPI

Next, you can install Sparkly from PyPI, using the following command: 

```
python3 -m pip install sparkly-em
```

This command will install Sparkly and all of its dependencies, such as Joblib, mmh3, Numba, Numpy, Numpydoc, Pandas, Psutil, Pyarrow, Pyspark, Scipy, and Tqdm, *except Java, JCC, and PyLucene*. 

Java, JCC and PyLucene cannot be pip installed with the above command, because they are not available on PyPI. See [instructions to install Java, JCC, and PyLucene](https://github.com/anhaidgroup/sparkly/blob/docs-update/doc/install-java-jcc-pylucene.md).

You have now completed the installation of Sparkly on a single machine. See [how to use Sparkly on a single machine](). 

### Pip Install from Github

((Explain why you want to do this)). You can install Sparkly directly from its Github repo using the following command: 

```
python3 -m pip install git+https://github.com/anhaidgroup/sparkly.git@clean_up
```

Similar to pip installing from PyPI, the above command will install Sparkly and all of its dependencies, except Java, JCC, and Pylucene. Refer back to the case of pip installing from PyPI on how to install these. 
