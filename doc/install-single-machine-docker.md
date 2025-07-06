## How to Install Sparkly Using Docker

If you are unable to install Sparkly or its prerequisites, especially PyLucene or JCC, we have also provided a Docker image with Sparkly and its prerequisites installed. The Sparkly image is based on Ubuntu 22.04 and contains everything needed to run Sparkly: Python 3.10, Java Temurin 17, JCC, PyLucene, Joblib, mmh3, Numba, Numpy, Numpydoc, Pandas, Psutil, Pyarrow, Pyspark, Scipy, Tqdm, and Sparkly itself. 

The Sparkly image allows you to create a Docker container that acts as a virtual machine - a separate, contained environment that has been configured to be compatible with Sparkly and all of its requirements, especially PyLucene and JCC. This is much simpler to set up than a manual install of Sparkly and can be used on any combination of OS and architecture as long as it supports Docker.

### Step 1: Installing Docker

In order to use the Sparkly Docker image you will need to download and install Docker. For single-machine usage, Docker has two notable forms - Docker Engine and Docker Desktop. Docker Engine is the core component that runs and manages Docker containers, while Docker Desktop is a GUI application that makes working with Docker easier. 

Only Docker Engine is necessary, but on Windows or MacOS installing Docker Desktop is the only way to install Docker Engine. Instructions to install Docker Engine and/or Docker Desktop can be found here:

[https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)

### Step 2: Downloading the Sparkly Docker Image

You can download the Sparkly image from Docker Hub with the following command:

```
docker pull uwmagellan/sparkly
```

You can check that the image has been downloaded successfully with this command:

```
docker images
```

If you have downloaded the image properly, you will get an output that looks like this:

```
REPOSITORY           TAG       IMAGE ID       CREATED          SIZE
uwmagellan/sparkly   latest    e5b2a12f74eb   27 minutes ago   5.21GB
```

### Step 3: Starting a Docker Container

In order to use Sparkly, you will need to start a container based on the Sparkly image. You can do so with the following command:

```
docker run --name sparkly -d uwmagellan/sparkly:latest
```

You can verify that the container has started with this command:

```
docker ps
```

If the container is running, you will get an output that looks like this:

```
CONTAINER ID   IMAGE                       COMMAND                  CREATED         STATUS         PORTS     NAMES
7198ca184781   uwmagellan/sparkly:latest   "/__cacert_entrypoin…"   7 seconds ago   Up 6 seconds             sparkly
```

Note that the value of the container id column will vary.

### Step 4: Connecting to the Sparkly Container

You will need to connect to the Sparkly Container in a manner similar to connecting to a remote computer. You can do so with the following command:

```
docker exec -it sparkly /bin/bash
```

This will open a terminal within the Docker container, allowing you to run normal terminal commands in the container. While you are connected to the Sparkly container, you can disconnect and return to your local terminal with the following command:

```
exit
```

### Step 5: Using Sparkly with the Docker Container

Before you can use Sparkly with Docker, you will need to copy python programs and any other necessary files, such as database files, into the Sparkly container. Let's say that you want to run [example.py](https://github.com/anhaidgroup/sparkly/blob/main/examples/example.py) from our examples. First, you need to download it. You can do so by right-clicking this link and selecting ```save link as```:

[https://raw.githubusercontent.com/anhaidgroup/sparkly/refs/heads/main/examples/example.py](https://raw.githubusercontent.com/anhaidgroup/sparkly/refs/heads/main/examples/example.py)

Be sure to name it ```example.py```.

Then you need to copy it into the Sparkly container, like so:

```
docker cp example.py sparkly:example.py
```

You also need the three datasets that example.py uses. You can download them by clicking on these links:

[https://github.com/anhaidgroup/sparkly/raw/refs/heads/main/examples/data/abt_buy/gold.parquet](https://github.com/anhaidgroup/sparkly/raw/refs/heads/main/examples/data/abt_buy/gold.parquet)

[https://github.com/anhaidgroup/sparkly/raw/refs/heads/main/examples/data/abt_buy/table_a.parquet](https://github.com/anhaidgroup/sparkly/raw/refs/heads/main/examples/data/abt_buy/table_a.parquet)

[https://github.com/anhaidgroup/sparkly/raw/refs/heads/main/examples/data/abt_buy/table_b.parquet](https://github.com/anhaidgroup/sparkly/raw/refs/heads/main/examples/data/abt_buy/table_b.parquet)

Next, you can copy them into the Sparkly container:

```
docker cp gold.parquet sparkly:gold.parquet
docker cp table_a.parquet sparkly:table_a.parquet
docker cp table_b.parquet sparkly:table_b.parquet
```

Then you should connect to the Sparkly container:

```
docker exec -it sparkly /bin/bash
```

Once you are connected, you will be able to run terminal commands as normal, and they will be executed on the virtual environment in the Sparkly container. For instance, you can use the ```ls``` command, and get this output:

```
bin  boot  __cacert_entrypoint.sh  dev  etc  example.py  gold.parquet  home  lib  media  mnt  opt  proc  root  run  sbin  srv  sys  table_a.parquet  table_b.parquet  tmp  usr  var
```

You can now run example.py through the ```python``` or ```python3``` command like you normally would:

```
python3 example.py
```

If you just run it "as is" it won't work right now, though, since example.py won't be able to recognize gold.parquet, table_a.parquet, and table_b.parquet - it's looking for them in the data/abt_buy folder, which doesn't exist in the Sparkly container. You will need to set up a folder structure that example.py can recognize:

```
mkdir data
mkdir data/abt_buy
mv gold.parquet data/abt_buy/gold.parquet
mv table_a.parquet data/abt_buy/table_a.parquet
mv table_b.parquet data/abt_buy/table_b.parquet
```

If you're a more advanced user you can also edit example.py to fix this bug, but that's beyond the scope of this example. In any case, you can now run example.py properly:

```
python3 example.py
```

You should get an output that looks like this:

```
+---+--------------------+--------------------+------------+                    
|_id|                 ids|              scores| search_time|
+---+--------------------+--------------------+------------+
|  2|[435, 958, 106, 1...|[122.16595, 116.1...| 0.019509554|
|  4|[160, 958, 435, 1...|[94.37205, 85.936...| 0.004992008|
|  5|[827, 863, 864, 8...|[103.68204, 38.85...| 0.003326416|
|  8|[1025, 134, 1028,...|[116.100494, 102....|0.0028777122|
| 12|[218, 1072, 478, ...|[168.21767, 31.60...|0.0033874512|
| 14|[214, 237, 97, 10...|[72.564606, 68.88...|0.0016140938|
| 15|[823, 1013, 585, ...|[103.8445, 32.933...| 0.001979351|
| 16|[59, 218, 449, 10...|[115.46915, 45.16...|0.0018417835|
| 17|[1080, 462, 344, ...|[116.46618, 80.70...|0.0015566349|
| 18|[99, 89, 1019, 86...|[99.18823, 51.784...| 0.006599903|
| 19|[826, 462, 1022, ...|[71.740364, 29.78...| 0.001894474|
| 23|[165, 443, 453, 4...|[44.777008, 43.96...|0.0029911995|
| 24|[407, 116, 295, 2...|[131.34975, 96.91...|0.0015757084|
| 25|[116, 407, 295, 1...|[147.75256, 78.82...|0.0014548302|
| 26|[100, 288, 435, 1...|[156.47028, 108.6...|0.0018053055|
| 27|[127, 128, 192, 2...|[53.533775, 52.37...|0.0010175705|
| 28|[128, 127, 138, 6...|[65.60057, 42.289...| 0.003352642|
| 29|[126, 585, 217, 1...|[93.191734, 53.42...|0.0019032955|
| 30|[217, 138, 127, 1...|[40.22264, 36.866...| 9.799004E-4|
| 31|[430, 82, 1043, 1...|[124.60372, 46.37...|0.0074050426|
+---+--------------------+--------------------+------------+
only showing top 20 rows

true_positives : 1095
recall : 0.9981768459434822
```

You will also generate an output file, candidates.parquet. If you run ```ls``` again, it will show up:

```
bin  boot  __cacert_entrypoint.sh  candidates.parquet  data  dev  etc  example.py  home  lib  media  mnt  opt  proc  root  run  sbin  srv  sys  tmp  usr  var
```

If you leave it in the Sparkly container, it will be deleted the next time the Sparkly container is stopped. To prevent this, you should copy it back to your local machine. Exit the Sparkly container with the following command:

```
exit
```

This will send you back to your local machine, and you will once again be running commands locally. You can now use the following command to copy candidates.parquet to your local machine:

```
docker cp sparkly:candidates.parquet candidates.parquet
```

With your output safely retrieved, you are free to stop the Sparkly container and free up the computing resources it's using:

```
docker stop sparkly
```

Note that all files copied into the Sparkly container or generated by running programs inside it will be lost when you stop it.

When you need to use Sparkly later, you can start the Sparkly container back up again like so:

```
docker start sparkly
```

If you don't need the Sparkly container anymore, you can completely remove it with the following command:

```
docker rm sparkly
```
