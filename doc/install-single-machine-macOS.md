## How to Install Sparkly on a MacOS M1 Machine

This is a step-by-step guide to install Sparkly and its necessary dependencies on a single macOS machine with an M1 chip. If you are unsure if your Mac has an M1 chip, click on the Apple in the top left corner of your screen > About This Mac. If it says Chip Apple M1, then you have an M1 chip. If it does not say Chip Apple M1, you do not have an M1 chip. 

This guide also assumes your profile uses a .zshrc profile for environment variables. To check if this is the case, open up a terminal and run 
```
	cat ~/.zshrc
```
If the output does not have one line containing 
```
“No such file or directory”
``` 
then you can proceed with the guide. 

Otherwise, your profile may be a bash profile. To check this, run 
```
	cat ~/.bash-profile
```
in your terminal.  If the output does not have one line containing 
```
“No such file or directory”
```
then you have a bash profile. In that case, wherever you see .zshrc in this guide, replace it with .bash-profile. Note: The default on macOS with an M1 chip is .zshrc, so this should be the case for most users. 

This guide has been tested on a 2020 MacBook Pro with an Apple M1 Chip, 8GB Memory, macOS version Sequoia 15.0.1, and a .zshrc profile. The following software versions were installed on the test machine using the steps in this guide: Python 3.12, ICU4C 74.2, Java Temurin 17 JDK, and PyLucene 9.12.0. You can try to adapt this guide to other configurations. 

If your machine has an Intel chip, this installation guide will not work for you.
If your machine has an M2, M3, or M4 chip, this installation guide may work for you, but we have not tested it, and we can not guarantee that it will work.


### Step 1: Xcode Tools Installation
Our first step will be installing Xcode command line tools. This is a set of tools and utilities provided by Apple that allow developers to perform software development tasks from the command line. 

Xcode command line tools include g++  (a C++ compiler which we need for JCC installation), make (which we need for ICU and PyLucene installations), as well as other tools that we will not use today (such as git, gcc, etc.). It is necessary to download all of the Xcode command line tools, however, to ensure correct setup. Note: We only need to download Xcode command line tools. This is different from the Xcode IDE, which is used for code development. We DO NOT need the Xcode IDE.

To install Xcode command line tools, open a terminal window. Then, run the command:
```
	xcode-select --install
```
If a popup window appears with the text: “The xcode-select command requires the command line developer tools. Would you like to install the tools now?”. Click Install. After it finishes installing (the popup will go away), you can make sure it was correctly installed by running:
```
	xcode-select --version
```
If the output includes 
```
“xcode-select version xxxx” 
```
where xxxx are four digits, then the installation was successful, and you can proceed to Step 2. 

Otherwise, if the command does not produce a popup, check if the output says 
```
“Command line tools are already installed". Use "Software Update" in System Settings or the softwareupdate command line interface to install updates”
```
If this is the case, then the tools are already installed, and you can proceed to Step 2.

### Step 2: Python Installation
This section deals with installing Python 3.12, creating a virtual environment, and then installing two Python packages (setuptools and build). This step is necessary for the JCC installation. Other versions of Python, other environments, or incorrect installations of the setuptools and build packages can cause issues with JCC installation. 

If you suspect that you may have Python downloaded on your machine already, open up your terminal. Then, run the command:
```
	which python
```
If the output path says 
```
“/usr/local/bin/python”
```
run:
```
	python --version
```
If the output of this is 
```
“Python 3.12.x”
```
where x is a number, you can go to Step 2C after completing Step 2A (you do not need to complete Step 2B). 

If 
```
which python
```
or 
```
python --version
``` 
did not have the outputs listed above, do all substeps, A-D.
#### Step 2A: Homebrew Installation
Before installing Python, we need to ensure we have Homebrew installed. Homebrew is a popular open-source package manager for macOS and Linux, used to simplify installing, updating, and managing software and libraries. 

To check if Homebrew is installed, open up a terminal. Then, type 
```
brew info
```
If the output contains kegs, files, and GB, then Homebrew is installed and you can go to Step 2B. Otherwise, you need to install Homebrew. 

To install Homebrew, run the following command in your terminal:
```
	/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com
    Homebrew/install/HEAD/install.sh)"
```

The installation may prompt you to follow further instructions to add this to your PATH variables; if so, follow those onscreen instructions. This will make it easier to use brew later. 
If you see 
```
“Installation Successful!” 
```
in the output, the download was successful.

#### Step 2B: Python Installation
To download Python environments, we will use Homebrew. Run the following in the terminal to install Python 3.12:
```
	brew install python@3.12
```

##### Step 2C: Python Environment Setup
Now, we will create a Python environment with Python 3.12. This step is necessary to make sure we use the correct version of Python with the correct dependencies for the PyLucene and JCC installation.  In your terminal, run:
```
	python -m venv sparkly
```

This will create a virtual environment named sparkly. To activate this environment, run the following:
```
	source sparkly/bin/activate
```
To make sure everything is correct, run:
```
	python --version
```
If the output says 
```
“Python 3.12.x”
```
where x ≥ 4, then the Python environment setup was successful.

#### Step 2D: Python Package Installation
We will be downloading two packages: setuptools and build. These packages are used to build and install JCC. Before installing, make sure you are in the virtual environment. If you have just finished Step 2C, you are in the virtual environment. Otherwise, to make sure your virtual environment is active, you can run:
```
	source sparkly/bin/activate
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
	source sparkly/bin/activate
```

### Step 3: ICU Installation
This section deals with the International Components for Unicode (ICU) download and installation. ICU is necessary for building JCC. ICU is used for internationalization and localization of text (think different languages and character types in different parts of the world). This is necessary because PyLucene needs to work in international contexts for its matching, searching, etc.  Although your machine may have a version of ICU installed, it must have the exact version listed below, or else it will be missing packages required by JCC.

#### Step 3A: ICU Download
First, download **icu4c-74_2-src.tgz** by using this link:
```
    https://github.com/unicode-org/icu/releases/tag/release-74-2 
```
and clicking 
```
	icu4c-74_2-src.tgz
```

To check if the download was successful, navigate to your Downloads folder in Finder by running:
```
	cd ~/Downloads
```
and look for 
```
icu4c-74_2-src.tgz
```
If it is present, then the download was successful.

#### Step 3B: Building & Installing ICU
Now that ICU has been downloaded, we will install it so that it can be used on your machine. 
First, open up your terminal.
Next, go into your Downloads folder (where the ICU download is) using the command:
```
	cd ~/Downloads
```
Then, extract the files from the download using the command:
```
	tar -xf icu4c-74_2-src.tgz
```
If there is no output, then this step was successful.

Then, go into the icu/source folder by using the command:
```
	cd icu/source
```
Stay in this folder for the rest of Step 3B.

The next command we are going to run is to configure ICU specifically for MacOS:
```
	./runConfigureICU MacOSX
```
If there is output, and the output includes 
```
“ICU for C/C++ 74.2 is ready to be built.”
```
then this step was successful.

The following two steps will build and install ICU.
First, to build ICU so it is ready to be installed, run the following:
```
	make
```
If there is output, and no error messages, then this step was successful.

Finally, to install ICU, run the following:
```
	sudo make install
```
This should prompt you for your password, and the password should be the same as what you use to log-in to your machine. If there is output, and no error messages, this step was successful.

### Step 4: Sparkly Installation
Now that you have the correct versions of Python and ICU installed, we can download Sparkly.
To download Sparkly, use one of the following options: 

#### Option 1: Pip Installing from PyPI
You can install Sparkly from PyPI, using the following command:
```
	pip install sparkly-em
```
This command will install Sparkly and all of its dependencies, such as Joblib, mmh3, Numba, Numpy, Numpydoc, Pandas, Psutil, Pyarrow, Pyspark, Scipy, and Tqdm, except Java, JCC, and PyLucene.


Java, JCC and PyLucene cannot be pip installed with the above command, because they are not available on PyPI.
#### Option 2: Pip Installing from GitHub
Instead of pip installing from PyPI, you may want to pip install Sparkly from its GitHub repo. This happens if you want to install the latest Sparkly version compared to the version on PyPI. For example, the GitHub version may contain bug fixes that the PyPI version does not. 

To install Sparkly directly from its GitHub repo, use the following command:
```
	pip install git+https://github.com/anhaidgroup/sparkly.git@clean_up
```
Similar to pip installing from PyPI, the above command will install Sparkly and all of its dependencies, except Java, JCC, and PyLucene.

### Step 5: Java Installation
We will now install Java Temurin 17 JDK onto your machine. Again, it is absolutely necessary to use this version of Java, or you will face issues building JCC. Using a non-Temurin version, or any other version number, can cause problems with JCC and PyLucene installations. 

Also, not using Homebrew for the installation could lead to downloading a version of Java that is not supported by an arm64 architecture, which would likely also cause problems with JCC and PyLucene installations.

##### Step 5A: Java Installation
We are going to download the Temurin 17 JDK that is optimized for your machine running on an M1 chip. To install Java, open a terminal and run the following:
```
	brew install --cask temurin@17
```

To check if this was successful, run the following command:
```
	java -version
```
If the output contains 
```
“openjdk version 17.x.y”
```
where x and y are numbers, the installation was successful.

#### Step 5B: Setting JAVA_HOME/JCC_JDK Environment Variables
We need to set this version of Java as the default for our machine.  To do so, open up a terminal and make sure you are in the root directory (if you are unsure if you are in the root directory or not, run cd to get back to the root directory). Next, run:
```
	echo 'export JAVA_HOME=$(/usr/libexec/java_home -v17)' >> ~/.zshrc
```
To enact these changes, run:
```
	source ~/.zshrc
```
To check if these changes were successful, run:
```
	echo $JAVA_HOME
```
If the installation was successful, you will see a file path output.

Additionally, for JCC, we need to set the environment variable JCC_JDK. This environment variable is used to build JCC with our version of JDK. To do so,  run:
```
	echo 'export JCC_JDK=$(/usr/libexec/java_home -v17)' >> ~/.zshrc
```
To enact these changes, run:
```
	source ~/.zshrc
```
To check if these changes were successful, run:
```
	echo $JCC_JDK
```
If the installation was successful, you will see a file path output.

You may notice the environment variables for JAVA_HOME and JCC_JDK have the same values, and this is what we expect and will need for proper installation of JCC. 

### Step 6: PyLucene and JCC Installation
This step installs JCC and PyLucene, which are both necessary for Sparkly. Before starting this step, double-check that you have completed all of the previous steps.

#### Step 6A: Download PyLucene
We are going to use pylucene-9.12.0. This version is compatible with the M1 chip on Mac. To download this version, go to:
```
	https://dlcdn.apache.org/lucene/pylucene/
```
and download **pylucene-9.12.0-src.tar.gz**  by clicking on the hyperlink
```
	pylucene-9.12.0-src.tar.gz
```
This will download a compressed folder to your Downloads folder. To verify the download, first open up your terminal. Then, navigate to your Downloads folder by typing:
```
	cd ~/Downloads 
```
Next, run:
```
ls
```
This command shows you all of the items in your Downloads folder, sorted alphabetically. If you see **pylucene-9.12.0-src.tar.gz**, then this step was successful.

Next, we will extract the files from this compressed folder. To do so, run the following command:
```
	tar -xf pylucene-9.12.0-src.tar.gz
```
Then, navigate to this new directory by running 
```
	cd ~/Downloads/pylucene-9.12.0
```

#### Step 6B: Building & Installing JCC
Now that you are in the pylucene-9.12.0 folder, we are going to build JCC, which is necessary for PyLucene. JCC has been downloaded as a part of PyLucene.

First, run the command:
```
	pushd jcc
```
Next, to build JCC, run:
```
	python setup.py build
```
If there are several lines of output, and the last few lines do not include the word “**error**”, then this step was successful.

Then, to install JCC run:
```
	sudo -E python setup.py install
```
You will be prompted to enter a password, and it will be the password you use to login to your computer. 

If the last line of the output says 
```
“error: The 'JCC==3.14' distribution was not found and is required by the application”
```
run the command again. Then, if the last line of the output says 
```
“Finished processing dependencies for JCC==3.14”
```
this step was successful.
Now, we can go back the the pylucene-9.12.0 folder by running the following:
```
	popd
```

#### Step 6C: PyLucene Installation
Now, we just need to finish installing PyLucene. Make sure you are still in the pylucene-9.12.0 directory. If you just finished Step 6B, you should already be there. Otherwise, open up your terminal and run:
```
	cd ~/Downloads/pylucene-9.12.0
```

Next, to build PyLucene so it is ready to be installed, run the following command where username is the user you have been completing this installation under. If you are unsure what your username is, in your terminal, run the command whoami and this will output your username. 

Once you have replaced the two instances of username in the following command, run it:
```
	sudo make PYTHON='/Users/username/sparkly/bin/python' JCC='$(PYTHON) -m jcc --arch aarch64' NUM_FILES=16 MODERN_PACKAGING='true' ICUSBIN='/Users/username/Downloads/icu/source/bin'
```

If this was successful, the last line of output will say “**build of pylucene 9.12.0 complete**”. 


The last step to install will be running a very similar command to the last. Make sure to change two instances of username again before running the command:
```
	sudo make install PYTHON='/Users/username/sparkly/bin/python' JCC='$(PYTHON) -m jcc --arch aarch64' NUM_FILES=16 MODERN_PACKAGING='true' ICUSBIN='/Users/username/Downloads/icu/source/bin'
```

If this was successful, the output will say “**Successfully installed lucene-9.12.0**”.

To make sure everything was installed correctly, run this command:
```
	python -c "import lucene; print(lucene.VERSION)"
```
If the output is “**9.12.0**”, then you are ready to use Sparkly!

### Additional Tips/FAQ’s
If at any time you have to take a break from the installation after Step 2, when you reopen your terminal, make sure to run 
```
	source sparkly/bin/activate
``` 
to ensure the correct Python environment. Using a different environment could lead to missing packages, or using the wrong version of Python, which can cause issues during JCC installation.
If anything else arises, refer to the source documentation for the specific software. 

To continue testing Sparkly, head over to the repository at https://github.com/anhaidgroup/sparkly/tree/main/examples to have some fun, and enjoy!

