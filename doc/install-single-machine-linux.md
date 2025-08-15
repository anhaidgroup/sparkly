## How to Install Sparkly on a Linux Machine

The following step-by-step guide describes how to install Sparkly on a single machine running Linux Ubuntu 22.04 (with x86 architecture), Python 3.10, g++ compiler, Java Temurin JDK 17, and PyLucene 9.4.1. You can adapt this guide for other similar configurations using Linux on x86 architecture. 

### Step 1: Installing Python 3

Start by installing Python 3 (if not already). You can find instructions here:
[https://wiki.python.org/moin/BeginnersGuide/Download](https://wiki.python.org/moin/BeginnersGuide/Download)

We recommend installing Python 3.10 if possible, as we have tested Sparkly extensively with Python 3.10. 

### Step 2: Installing Sparkly 

In the future you can install Sparkly using one of the following two options. **As of now, since Sparkly is still in testing, we do not yet enable Option 1 (Pip installing from PyPI). Thus you should use Option 2 (Pip installing from GitHub).**

#### Option 1: Pip Installing from PyPI

**Note that this option is not yet enabled. Please use Option 2.**

Next, you can install Sparkly from PyPI, using the following command: 

```
python3 -m pip install sparkly-em
```

This command will install Sparkly and all of its dependencies, such as Joblib, mmh3, Numba, Numpy, Numpydoc, Pandas, Psutil, Pyarrow, Pyspark, Scipy, and Tqdm, *except Java, JCC, and PyLucene*. 

Java, JCC and PyLucene cannot be pip installed with the above command, because they are not available on PyPI. 

#### Option 2: Pip Installing from GitHub

Instead of pip installing from PyPI, you may want to pip install Sparkly from its GitHub repo. This happens if you want to install the latest Sparkly version compared to the version on PyPI. For example, the GitHub version may contain bug fixes that the PyPI version does not. To install Sparkly directly from its GitHub repo, use the following command:

```
python3 -m pip install git+https://github.com/anhaidgroup/sparkly.git@main
```

Similar to pip installing from PyPI, the above command will install Sparkly and all of its dependencies, *except Java, JCC, and PyLucene*. 

### Step 3: Installing PyLucene and Its Dependencies

#### Step 3.1: Installing Java

We strongly recommend installing Java Temurin JDK 17, which is a specific Java release that we have extensively experimented with. As that is not available from the Ubuntu package repository, to install Java, you will need to use the following commands, which were taken from the PyLucene installation guide on their website:

```
sudo -s
apt install wget apt-transport-https gnupg
wget -O - https://packages.adoptium.net/artifactory/api/gpg/key/public | apt-key add -
echo "deb https://packages.adoptium.net/artifactory/deb $(awk -F= '/^VERSION_CODENAME/{print$2}' /etc/os-release)     main" | tee /etc/apt/sources.list.d/adoptium.list
apt update
apt install temurin-17-jdk
exit
```

You can check that you have successfully installed Java by running this command. If Java is installed, it should display a version number.

```
Java --version
```

#### Step 3.2: Installing g++

JCC requires a C++ compiler to work. For Ubuntu, it is recommended that you use the g++ compiler. Most versions of Ubuntu come with it included, but you can check if it is installed with the following command:

```
g++ --version
```

If it is installed, a version number will be displayed. If it is not installed, you can do so with the following commands:

```
sudo apt update
sudo apt install g++
```

#### Step 3.3: Installing JCC and PyLucene

You will need to install Setuptools in order to install JCC and PyLucene. Setuptools is a Python package that simplifies building, distributing, and installing Python packages. You can install Setuptools with the following command:

```
python3 -m pip install setuptools
```

Next, you must download and unpack PyLucene 9.4.1. You can do so by running these commands:

```
wget https://dlcdn.apache.org/lucene/pylucene/pylucene-9.4.1-src.tar.gz
tar -xvf pylucene-9.4.1-src.tar.gz
```

This will produce a folder called 'pylucene-9.4.1'. This is the main PyLucene directory. You should switch to it by running the following command:

```
cd pylucene-9.4.1
```

The source code for JCC is distributed with the PyLucene source code and must be installed before you can install PyLucene. The following commands will switch to the 'jcc' subdirectory, build and install JCC, then return to the main PyLucene directory.

```
pushd jcc
sudo python3 setup.py build
sudo python3 setup.py install
popd
```

Installing PyLucene requires Make. The following command will install Make, if it is not already installed.

```
sudo apt install make
```

In order to install PyLucene, you first have to build it. You can do so with the following command. Note that this and following commands assume that your Python installation is in the default location for Ubuntu. If it is not, you will have to change the value of the PYTHON= argument (in the commands below) to reflect it.

```
sudo make PYTHON='/usr/bin/python3' JCC='$(PYTHON) -m jcc.__main__ --shared --arch x86_64' NUM_FILES=16
```

Once the command has finished running, you should check that PyLucene is built properly. You can do so with the following command.

```
sudo make test PYTHON='/usr/bin/python3' JCC='$(PYTHON) -m jcc.__main__ --shared --arch x86_64' NUM_FILES=16
```

If PyLucene is built properly, the output of this command will end with several blocks that look like this:

```
----------------------------------------------------------------------
Ran 10 tests in 1.980s

OK
/usr/bin/python3 test3/test_StopAnalyzer.py
...
----------------------------------------------------------------------
Ran 3 tests in 0.007s

OK
/usr/bin/python3 test3/test_Similarity.py
.
----------------------------------------------------------------------
Ran 1 test in 0.311s

OK
/usr/bin/python3 test3/test_Not.py
.
----------------------------------------------------------------------
Ran 1 test in 0.330s

OK
/usr/bin/python3 test3/test_ThaiAnalyzer.py
...
----------------------------------------------------------------------
Ran 3 tests in 0.010s

OK
/usr/bin/python3 test3/test_PythonException.py
.
----------------------------------------------------------------------
Ran 1 test in 0.012s

OK
/usr/bin/python3 test3/test_bug1564.py
.
----------------------------------------------------------------------
Ran 1 test in 0.307s

OK

```

Once you have verified that there are no errors, the following command will install PyLucene. 

```
sudo make install PYTHON='/usr/bin/python3' JCC='$(PYTHON) -m jcc.__main__ --shared --arch x86_64' NUM_FILES=16
```

A simple way to test that PyLucene is installed is to open Python using one of the following commands:

```
python
```
```
python3
```

You can then import the PyLucene package like so:

```
import lucene
```

If PyLucene *is not* installed properly, you will get an error message that looks like this:

```
ModuleNotFoundError: No module named 'lucene'
```

If an error message does not appear, that means PyLucene is installed properly. Congratulations. You have now completed the installation of Sparkly on a single Linux machine. 

#### Further Pointers

[This page](https://github.com/anhaidgroup/sparkly/blob/main/tips/pylucene.md) lists more tips for installing PyLucene.




