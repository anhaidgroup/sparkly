# How to Install Sparkly Jupyter Notebook on a Single Machine Using Docker

If you prefer to use Sparkly through Jupyter Notebook rather than the command line, we’ve provided a Dockerfile that allows you to build and run a complete Sparkly environment preconfigured with **JupyterLab**.
This image includes **Sparkly** and all its dependencies — including **PyLucene, JCC, and Java Temurin 17** — and runs a **JupyterLab server** that you can access through your web browser.

The Sparkly Jupyter image is based on **Ubuntu 24.04** and contains everything needed to run Sparkly interactively in notebooks. This is ideal for users who want to experiment with Sparkly in a contained environment without the hassle of installing dependencies manually.

However, if you're using **macOS with Apple Silicon (Mx)**, you should follow the [Install Sparkly Using Docker](./install-single-machine-docker.md) guide instead, which provides an **ARM64-compatible** setup.

---

## Step 1: Installing Docker

To build and run the Sparkly Jupyter Docker image, you must have Docker installed on your system.

If you haven’t installed Docker yet, follow the official installation guide here:

[https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)

Docker Desktop is required for **Windows** and **macOS** users.
For **Linux**, Docker Engine alone is sufficient.

---

## Step 2: Building the Sparkly Jupyter Docker Image

Once Docker is installed, copy the below content and save it as `Dockerfile` in a new directory:

```Dockerfile
FROM ubuntu:24.04
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /root

RUN apt update && apt install -y build-essential python3 python3-pip
RUN apt install -y git
RUN apt install -y wget apt-transport-https gnupg

RUN wget -O - https://packages.adoptium.net/artifactory/api/gpg/key/public | apt-key add - \
 && echo "deb https://packages.adoptium.net/artifactory/deb $(awk -F= '/^VERSION_CODENAME/{print $2}' /etc/os-release) main" \
    > /etc/apt/sources.list.d/adoptium.list \
 && apt update && apt install -y temurin-17-jdk

RUN pip3 install --break-system-packages setuptools
RUN pip3 install --break-system-packages git+https://github.com/anhaidgroup/sparkly.git@main

RUN mkdir -p /root/workspace \
 && cd /root/workspace \
 && wget https://dlcdn.apache.org/lucene/pylucene/pylucene-9.4.1-src.tar.gz \
 && tar -xvf pylucene-9.4.1-src.tar.gz \
 && rm pylucene-9.4.1-src.tar.gz

WORKDIR /root/workspace/pylucene-9.4.1/jcc
RUN python3 setup.py build && python3 setup.py install

RUN apt install -y make
RUN python3 -m jcc --help >/dev/null

WORKDIR /root/workspace/pylucene-9.4.1
RUN make PYTHON='/usr/bin/python3' JCC='$(PYTHON) -m jcc.__main__ --shared --arch x86_64' NUM_FILES=16 \
 && make install PYTHON='/usr/bin/python3' JCC='$(PYTHON) -m jcc.__main__ --shared --arch x86_64' NUM_FILES=16

RUN pip3 uninstall -y pandas --break-system-packages || true \
 && pip3 install --no-cache-dir "pandas==2.1.4" --break-system-packages

RUN pip3 install --no-cache-dir jupyterlab==4.3.5 ipywidgets --break-system-packages
EXPOSE 8888
CMD ["python3", "-m", "jupyterlab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

WORKDIR /root/workspace
```

Then, in the same directory, run the following command to build the image:

```
docker build -t sparkly-noble .
```

This process will take several minutes, as it compiles **PyLucene** and installs **Sparkly** and **JupyterLab**.

You can confirm the image was built successfully by listing your local Docker images:

```
docker images
```

Expected output:

```
REPOSITORY        TAG       IMAGE ID       CREATED          SIZE
sparkly-noble     latest    9f8a1b2c3d4e   10 minutes ago   6.07GB
```

---

## Step 3: Running the Sparkly Jupyter Container

Once the image is built, you can create and start a container running JupyterLab using the following command:

```
docker run -p 8888:8888 -d --name sparkly-jupyter sparkly-noble
```

This command starts a new container named `sparkly-jupyter` and maps port **8888** in the container to port **8888** on your local machine — the default port for Jupyter.

Check that the container is running:

```
docker ps
```

Expected output:

```
CONTAINER ID   IMAGE            COMMAND                  CREATED         STATUS         PORTS                    NAMES
f817c3a9e012   sparkly-noble    "python3 -m jupyterl…"   8 seconds ago   Up 7 seconds   0.0.0.0:8888->8888/tcp   sparkly-jupyter
```

---

## Step 4: Accessing JupyterLab

To access JupyterLab, open your web browser and go to:

```
http://localhost:8888
```

The first time you access it, you’ll be prompted for a **token**.
To get the token, view the container logs:

```
docker logs sparkly-jupyter
```

You’ll see output similar to this:

```
To access the server, open this file in a browser:
    http://127.0.0.1:8888/lab?token=abcd1234efgh5678ijkl
```

Copy that full URL (including the token) into your browser to open JupyterLab.

---

## Step 5: Using Sparkly in JupyterLab

Once inside JupyterLab, you can create a new **Python 3 notebook** and import Sparkly as you would in any other Python environment:

```python
import sparkly
```

From there, you can run Sparkly scripts, experiments, and interactive analyses directly in the notebook interface.

You can also upload your own datasets using the JupyterLab file browser on the left-hand side, or mount directories from your host machine when running Docker (see below).

---

## Step 6: Sharing Files Between Host and Container (Optional)

To persist files or share notebooks between your local machine and the container, you can **mount a local directory** when running Docker:

```
docker run -p 8888:8888 -v /path/to/your/data:/root/workspace -d --name sparkly-jupyter sparkly-noble
```

This way, any files you create or modify in `/root/workspace` inside Jupyter will also appear in `/path/to/your/data` on your host system.

---

## Step 7: Managing the Container

You can stop, restart, and remove the container as needed:

- **Stop the container:**

  ```
  docker stop sparkly-jupyter
  ```

- **Restart the container:**

  ```
  docker start sparkly-jupyter
  ```

- **Remove the container completely:**

  ```
  docker rm sparkly-jupyter
  ```

Your notebooks and files will remain safe if you used a mounted volume (`-v` option).
Otherwise, any data stored inside the container will be deleted when it’s removed.

---

## Step 8: Troubleshooting

- If the container fails to start, ensure port **8888** is not already in use.
- If you can’t connect, check Docker logs:

  ```
  docker logs sparkly-jupyter
  ```

- If `jupyterlab` doesn’t open or shows a blank page, try restarting Docker and your container.

---

## Summary

You now have a fully functioning **Sparkly + JupyterLab** environment running inside Docker.
This setup isolates Sparkly and its dependencies in a consistent, reproducible environment that can be used across different operating systems.

**Next steps:**

- Open a notebook in JupyterLab.
- Load or import your data.
- Start building and testing your Sparkly-based data linkage and search workflows interactively.
