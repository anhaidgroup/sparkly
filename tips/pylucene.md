# Tips for Installing PyLucene

## General

In order for the install to work correctly, both the JCC and PyLucene install scripts must be pointed to the 
same `JAVA_HOME`. If you choose to just use the system default java, ensure that `java`, `javac`, and `jar` are all the 
*same versions*. For example,

```bash 
$ java --version
openjdk 17.0.5 2022-10-18
OpenJDK Runtime Environment Temurin-17.0.5+8 (build 17.0.5+8)
OpenJDK 64-Bit Server VM Temurin-17.0.5+8 (build 17.0.5+8, mixed mode, sharing)

$ javac --version
javac 17.0.5

$ jar --version
jar 17.0.5

$ realpath "$(which java)"
/usr/lib/jvm/temurin-17-jdk-amd64/bin/java

$ realpath "$(which javac)"
/usr/lib/jvm/temurin-17-jdk-amd64/bin/javac

$ realpath "$(which jar)"
/usr/lib/jvm/temurin-17-jdk-amd64/bin/jar

```

Note that if the versions don't agree, JCC will likely install with no errors or warnings but PyLucene will
give unhelpful compiler or linker errors.


## MacOS

In MacOS High Sierra and later sometimes MacOS will kill pyspark workers when the JVM is initialized for PyLucene. If 
`examples/local_example.py` runs but `examples/basic_example.py` crashes, try modifying the environment variables 
as described in this [stack overflow post](https://stackoverflow.com/questions/50168647/multiprocessing-causes-python-to-crash-and-gives-an-error-may-have-been-in-progr).


In order to install PyLucene, you must first install JCC. JCC requires 
a C++ compiler to compile the extension before it is installed. During this
install look for *non-fatal* errors and warnings. In some cases the extensions are 
not compiled for the correct architecture, resulting in non-fatal linker errors, 
which will prevent PyLucene from installing correctly.
If when running the `python3 setup.py build` the extension is being compiled with the 
incorrect CPU architecture, you may need to manually specify the architecture by appending 
them to `cflags` and `lflags` in `setup.py`. For example, you may want to add the lines
```python

cflags += ['-arch', 'arm64']
lflags += ['-arch', 'arm64']
```

before the `Extension` constructor is called in the script.
