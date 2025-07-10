## How to Install Sparkly on a Single Machine

You can install and run Sparkly on a single machine. If you have relatively small tables, or just want to experiment with Sparkly, this is often sufficient. Otherwise you may consider installing and running Sparkly on a [cluster of machines](https://github.com/anhaidgroup/sparkly/blob/main/doc/install-cluster-machines.md) (such as on Amazon Web Services). 

### Overview

The installation process consists of three main steps: 
1. Install Python 3 (if not already)
2. Pip install Sparkly from either PyPI or GitHub. This installs Sparkly in its entirety, except PyLucene and its dependencies, as they generally cannot be pip installed.
3. Install PyLucene and its dependencies: C++ compiler, Java, and JCC.

See here for [an explanation of why Sparkly needs PyLucene](./why-pylucene.md).

### Options

We currently offer the following ways to install Sparkly: 
* [Installing on Linux](./install-single-machine-linux.md)
* [Installing on MacOS](./install-single-machine-macOS.md)

Additional ways to install Sparkly: 
* [Installing on other OS environments](./install-single-machine-otherOS.md) (no detailed instructions yet)
* [Installing using Docker](./install-single-machine-docker.md) (preliminary instructions)

