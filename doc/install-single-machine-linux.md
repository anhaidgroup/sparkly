## How to Install Sparkly on a Linux Machine

The following step-by-step guide describes how to install Sparkly on a single machine running Linux Ubuntu 22.04 (with x86 architecture), Python 3.12, g++ compiler, Java Temurin JDK 17, and PyLucene 9.4.1. You can adapt this guide for other similar configurations using Linux on x86 architecture.

### Step 1: Installing Python

We now install Python 3.12, create a virtual environment, and install two Python packages setuptools and build. Other versions of Python, other environments, or incorrect installations of the setuptools and build packages can cause issues with Sparkly installation.

If you suspect that you may have Python downloaded on your machine already, open up your terminal. Then run the command:

```
    which python3
```

If the output path says

“/usr/local/bin/python3”

run:

```
    python3 --version
```

If the output of this is

“Python 3.12.x”

where x is a number, you can go to Step 1B (you do not need to complete Step 1A).

If

```
which python3
```

or

```
python3 --version
```

do not have the outputs listed above, continue to step 1A.

#### Step 1A: Installing Python 3.12

Here we download Python 3.12, install it, and make it the default verison.
Run the following commands in the terminal to install Python 3.12:

```
    cd /usr/src
    sudo curl -O https://www.python.org/ftp/python/3.12.3/Python-3.12.3.tgz
    sudo tar xzf Python-3.12.3.tgz
```

```
    cd Python-3.12.3
    sudo make clean
    sudo ./configure --enable-optimizations --with-system-ffi
    sudo make -j$(nproc)
    sudo make altinstall
```

Now run the following commands to make Python 3.12 the default:

```
sudo update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.12 1
python3.12 -m ensurepip --default-pip
python3.12 -m pip install --upgrade pip setuptools
```

#### Step 1B: Setting Up the Python Environment

Now we will create a Python environment with Python 3.12. This step is necessary to make sure we use the correct version of Python with the correct dependencies. First, we install the venv module with the following command:

```
    sudo apt install python3-venv
```

Next, in your terminal, run:

```
    python3 -m venv ~/sparkly-venv
```

This will create a virtual environment named sparkly-venv. To activate this environment, run the following:

```
    source ~/sparkly-venv/bin/activate
```

To make sure everything is correct, run:

```
    python3 --version
```

If the output says

“Python 3.12.x”

where x ≥ 0, then the Python environment setup was successful.

#### Step 1C: Installing the Python Packages setuptools and build

Before installing these two packages, make sure you are in the virtual environment. If you have just finished Step 1B, you are in the virtual environment. Otherwise, to make sure your virtual environment is active, you can run:

```
    source ~/sparkly-venv/bin/activate
```

To install setuptools, run:

```
    pip install setuptools
```

To install build, run:

```
    pip install build
```

If at any point during the installation you close your terminal, you will need to reactivate your virtual environment by running:

```
    source ~/sparkly-venv/bin/activate
```

### Step 2: Installing Sparkly

Before installing Sparkly, we should return to the root directory by running the following command in the terminal:

```
    cd
```

Also, to make sure we are still in the same virtual environment, we should run:

```
    source ~/sparkly-venv/bin/activate
```

In the future you can install Sparkly using one of the following two options. **As of now, since Sparkly is still in testing, we do not yet enable Option 1 (Pip installing from PyPI). Thus you should use Option 2 (Pip installing from GitHub).**

#### Option 1: Pip Installing from PyPI

**Note that this option is not yet enabled. Please use Option 2.**

Next, you can install Sparkly from PyPI, using the following command:

```
pip install sparkly-em
```

This command will install Sparkly and all of its dependencies, such as Joblib, mmh3, Numba, Numpy, Numpydoc, Pandas, Psutil, Pyarrow, Pyspark, Scipy, and Tqdm, _except Java, JCC, and PyLucene_.

Java, JCC and PyLucene cannot be pip installed with the above command, because they are not available on PyPI.

#### Option 2: Pip Installing from GitHub

Instead of pip installing from PyPI, you may want to pip install Sparkly from its GitHub repo. This happens if you want to install the latest Sparkly version compared to the version on PyPI. For example, the GitHub version may contain bug fixes that the PyPI version does not. To install Sparkly directly from its GitHub repo, use the following command:

```
pip install git+https://github.com/anhaidgroup/sparkly.git@main
```

Similar to pip installing from PyPI, the above command will install Sparkly and all of its dependencies, _except Java, JCC, and PyLucene_.

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
java --version
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

You must download and unpack PyLucene 9.4.1. You can do so by running these commands:

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

A simple way to test that PyLucene is installed is to open Python using the following:

```
python3
```

You can then import the PyLucene package like so:

```
import lucene
```

If PyLucene _is not_ installed properly, you will get an error message that looks like this:

```
ModuleNotFoundError: No module named 'lucene'
```

If an error message does not appear, that means PyLucene is installed properly. Congratulations. You have now completed the installation of Sparkly on a single Linux machine.

#### Further Pointers

[This page](https://github.com/anhaidgroup/sparkly/blob/main/tips/pylucene.md) lists more tips for installing PyLucene.
