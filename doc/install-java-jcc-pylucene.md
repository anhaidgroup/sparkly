## Installing Java, JCC, and PyLucene

### Motivation

At the heart of Sparkly are Lucene and Spark. The Lucene library allows us to quickly perform information retrieval (IR) style search, that is, find tuple pairs with high BM25 similarity score. Spark allows us to quickly perform this search for large tables in a distributed and parallel fashion. 

However, Lucene is written in Java, while the rest of Sparkly is written in Python, to take advantage of many other Python packages. To bridge this Java-Python gap, we use the well-known package PyLucene. Briefly, PyLucene uses JCC to compile Lucene's Java code to C, then compile C code to Python. So we need to install Java, JCC, and PyLucene. 

Installing PyLucene is tricky as it is somewhat finicky. We have extensively tested installing PyLucene on Linux Ubuntu 22.04 on machines with x86 architecture, with Python 3.10, so if you go with this configuration, use the installation instruction below. If you use MacOS, you can probably still use the instruction below, but read the tips for installing PyLucene on MacOS at the end of this page. If you go with any other configuration, you should refer to the PyLucene website for installation instructions: [https://lucene.apache.org/pylucene/install.html](https://lucene.apache.org/pylucene/install.html)

If you find installing PyLucene cumbersome, consider using the Docker image of Sparkly, as mentioned earlier. 

In what follows we provide a step-by-step guide for installing PyLucene on Linux Ubuntu 22.04 on machines with x86 architecture, with Python 3.10.

### Installing Java

((What if Java has been installed? Can we overinstall it?))

PyLucene is rather finicky and we can only confirm that it works with Temurin JDK 17, which is a specific Java release. As that is not available from the Ubuntu package repository, to install Java, you will need to use the following commands, which were taken from the PyLucene installation guide on their website:

```
sudo -s
apt install wget apt-transport-https gnupg
wget -O - https://packages.adoptium.net/artifactory/api/gpg/key/public | apt-key add -
echo "deb https://packages.adoptium.net/artifactory/deb $(awk -F= '/^VERSION_CODENAME/{print$2}' /etc/os-release)     main" | tee /etc/apt/sources.list.d/adoptium.list
apt update
apt install temurin-17-jdk
exit
```

You can check that you have successfully installed java by running this command. If java is installed, it should display a version number.

```
java --version
```

### Installing JCC and PyLucene

Next, you must download and unpack PyLucene 9.4.1. You can do so by running these commands:

```
wget https://dlcdn.apache.org/lucene/pylucene/pylucene-9.4.1-src.tar.gz
tar -xvf pylucene-9.4.1-src.tar.gz
```

This will produce a folder called 'pylucene-9.4.1'. This is the main Pylucene directory. You should switch to it by running the following command:

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

Installing Pylucene requires Make. The following command will install Make, if it is not already installed.

```
sudo apt install make
```

In order to install Pylucene, you first have to build it. You can do so with the following command. Note that this and following commands assume that your Python installation is in the default location. If it is not, you will have to change the value of the PYTHON= argument (in the commands below) to reflect it.

```
sudo make PYTHON='/usr/bin/python3' JCC='$(PYTHON) -m jcc.__main__ --shared --arch x86_64' NUM_FILES=16
```

Once the command has finished running, you should check that Pylucene is built properly. You can do so with the following command.

```
sudo make test PYTHON='/usr/bin/python3' JCC='$(PYTHON) -m jcc.__main__ --shared --arch x86_64' NUM_FILES=16
```

Once you have verified that there are no errors, the following command will install Pylucene. 

```
sudo make install PYTHON='/usr/bin/python3' JCC='$(PYTHON) -m jcc.__main__ --shared --arch x86_64' NUM_FILES=16
```

### Further Pointers

[This page](https://github.com/anhaidgroup/sparkly/blob/main/tips/pylucene.md) lists more tips for installing PyLucene.
