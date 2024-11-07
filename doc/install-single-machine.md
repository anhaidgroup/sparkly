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

