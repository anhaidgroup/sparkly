## How to Install Sparkly on a Single Machine

You can install and run Sparkly on a single machine. If you have relatively small tables, or just want to experiment with Sparkly, this is often sufficient. Otherwise you may consider installing and running Sparkly on [an on-premise cluster]() or [a cluster on cloud]() (such as AWS). 

### Overview

The installation process consists of three main steps: 
1. Install Python 3 (if not already)
2. Pip install Sparkly from either PyPI or GitHub. This installs Sparkly in its entirety, except PyLucene and its dependencies, as they generally cannot be pip installed.
3. Install PyLucene and its dependencies: C++ compiler, Java, and JCC.

See here for an explanation of why Sparkly needs PyLucene. 

### Options

We currently offer the following ways to install Sparkly: 
* A simple way to install Sparkly is to use a Docker image that bundles Sparkly with all of its dependencies. See here for instructions.
* [Installing on Linux](./install-single-machine-linux.md)
* Installing on MacOS
* Installing on other OS environments 

#### Step 1: Installing Python 3

You need to start by installing Python 3 (if not already). You can find instructions here:
[https://wiki.python.org/moin/BeginnersGuide/Download](https://wiki.python.org/moin/BeginnersGuide/Download)

We recommend installing Python 3.10 if possible, as we have tested Sparkly extensively with Python 3.10. 

#### Step 2 - Option 1: Pip Install from PyPI

Next, you can install Sparkly from PyPI, using the following command: 

```
python3 -m pip install sparkly-em
```

This command will install Sparkly and all of its dependencies, such as Joblib, mmh3, Numba, Numpy, Numpydoc, Pandas, Psutil, Pyarrow, Pyspark, Scipy, and Tqdm, *except Java, JCC, and PyLucene*. 

Java, JCC and PyLucene cannot be pip installed with the above command, because they are not available on PyPI. 

### Step 2 - Option 2: Pip Install from GitHub

Instead of pip installing from PyPI, you may want to pip install Sparkly from its GitHub repo. This happens if you want to install the latest Sparkly version compared to the version on PyPI. For example, the GitHub version may contain bug fixes that the PyPI version does not. To install Sparkly directly from its GitHub repo, use the following command:

```
python3 -m pip install git+https://github.com/anhaidgroup/sparkly.git@clean_up
```

Similar to pip installing from PyPI, the above command will install Sparkly and all of its dependencies, *except Java, JCC, and PyLucene*. 

### Step 3: Install PyLucene and Its Dependencies (Java and JCC)

See [instructions to install Java, JCC, and PyLucene](https://github.com/anhaidgroup/sparkly/blob/docs-update/doc/install-java-jcc-pylucene.md).

You have now completed the installation of Sparkly on a single machine. See [how to use Sparkly on a single machine](). 

