## How to Install Sparkly on a Cluster of Machines

Here we provide instructions for installing Sparkly on a cluster of machines on the cloud, specifically on Amazon Web Services (AWS). You can adapt this guide to install Sparkly on a cluster of local machines. We have tested these installation instructions only with Ubuntu 22.04, Java Temurin JDK 17, Python 3.10, and Spark 3.5.1. 

### Creating EC2 Nodes

You will need to create at least two EC2 instances to serve as nodes in your cluster. Each instance will host a single node, so you must create as many EC2 instances as there are desired nodes in your cluster. The steps for creating an instance are as follows:
1. Create an account or login to AWS here: [https://aws.amazon.com/](https://aws.amazon.com/)
2. Go to Compute - EC2 under ‘All Services’.
3. Select ‘Launch Instance’.
4. Give it a name. Any name is fine and you can change it later; something like ‘spark-cluster-master’ for your first instance and ‘spark-cluster-worker-x’ where x is an unique integer for your other instances is perfectly usable.
5. Under ‘Application and OS image’, ‘Amazon Machine Image’ should be set to Ubuntu 22.04 and ‘Architecture’ should be set to 64-bit (x86).
6. Select whichever instance type suits your purposes. If you do not know where to begin, m-type instances are general purpose instances well suited for a variety of tasks. Make sure that your instance has enough memory for whatever task you run it on. We recommend at least 5-10GB if you are running blocking tasks on lists involving a million tuples each; larger lists will require more memory.
7. You will have to create a key pair to securely access each instance. You can set your key pair to whatever you wish, but unless you have unusually high security concerns we recommend that all your instances share the same key pair. You will need this key pair to connect to instances and having one key pair for all instances is much easier to manage than having one key pair per instance.
8. Under ‘Network Settings’ if you are creating a new security group make sure that the ‘Allow SSH traffic from’ option is not set to ‘Anywhere’, as this will allow anyone who can get ahold of your key pair to connect to the instance and constitutes a security risk. We recommend setting it to ‘My IP’ instead. Keep note of what your security group name is as new instances should be set to the same security group instead of new ones. This is not a mandatory step but configuring your cluster will be much easier if all of your EC2 instances share the same security group, as that will allow them to share connection rules, so it is recommended.
9. You can set your harddisk space (that is, EBS volume) under ‘Configure Storage’ to whatever you feel is necessary. We recommend at least 20 gigabytes.
10. You can now click on ‘Launch Instance’ at the bottom. Do not touch any other settings.
11. You should navigate to ‘Instances’ from the navigation panel on the left; you can view and manage all your created instances from this page.
12. When you create a new instance, it should start automatically. If you want to start or stop it manually, there is a button labeled ‘Instance state’ at the top of the page. Clicking on that will show a dropdown menu with ‘start instance’ and ‘stop instance’ buttons.
13. You can select an instance from the list of instances by clicking on the checkbox to the left of its name. When you select an instance on the instance page, an informational panel will appear at the bottom of the page. Switch to the ‘details’ tab and record the private and public IPv4 addresses of any instance you create, as these will be necessary for connecting to instances through networks. Note that while the private address is fixed, the public address is different every time you restart an instance. You will need to re-record it each time. Instances that have been stopped do not have a public IPv4 address.
14. You will also need to configure the instance’s security group to accept connections from your local machine and other instances. It is possible that this was set automatically when creating the security group, but check to make sure. If you have assigned all of your instances to a single security group, you will only need to do this once.
    * Switch to the ‘security’ tab in the informational panel and click on the security group link in the ‘security details’ section. This will open up the instance’s security group page.
    * Click on ‘edit inbound rules’ in the ‘inbound rules’ section.
    * Click on ‘add rule’ in the bottom left corner. This will create a new blank rule. Set the ‘type’ column to SSH and put your IP address (of your local machine) in the box to the right of the ‘source’ column. Optionally, you may also give your rule a description.
    * Once you have added this rule, you can click on ‘save rules’ in the lower right to finalize and save your changes.
    * You can now connect to your instance from your local machine. This will allow you to execute terminal commands on it. The exact method of connection will depend on your personal machine’s OS, but regardless of which method you use you will need to tell it to connect to ‘ubuntu@{the instance’s public IP address}’. You should wait for an instance to pass all checks under the ‘Status check’ column before connecting to it. Note that you may need to reload the AWS page in order for this column to update.

### Installing Sparkly
Next you need to install Sparkly. You can find instructions for installing Sparkly and its prerequisites (Python, Java, JCC, PyLucene, Joblib, mmh3, Numba, Numpy, Numpydoc, Pandas, Psutil, Pyarrow, Pyspark, Scipy, and Tqdm) [here](https://github.com/anhaidgroup/sparkly/blob/docs-update/doc/install-single-machine.md).

You will have to install Sparkly and its prerequisites on every node in your cluster.

### Installing Spark
Next you need to install Spark. Note that Spark requires Java in order to run. We recommend following the instructions for installing Java in the previous section as we can only guarantee PyLucene compatibility with specific versions of Java.

Spark must be installed on every node in order to set up a cluster - you will have to repeat the following steps for every EC2 instance you intend to use as a node.

You can use these commands to download and unpack Spark:

    wget "https://dlcdn.apache.org/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz"
    tar -xvf spark-3.5.1-bin-hadoop3.tgz

Running the above commands will create a folder called ‘spark-3.5.1-bin-hadoop3’. This contains everything you need to create a Spark node. You should rename this as the default name is rather unwieldy:

    mv spark-3.5.1-bin-hadoop3 spark

This guide will assume that you renamed the spark folder to ‘spark’, but you can use a different name if you wish.

#### Setting Up a Spark Cluster
Before setting up a Spark cluster you must open ports between the EC2 instances. Without this your nodes will not be able to communicate with each other. Step 14 in the section 'Creating EC2 Nodes' contains instructions for editing the inbound rules of your security group. You should add one inbound rule for each EC2 instance you have. When adding these rules, you will need to do the following:
* Keep the type as ‘all TCP’ or ‘custom TCP’.
* The IP address will need to be the private IPv4 address of the instance.
* You should set the port range to a value of 0-65535.
Doing this will allow each of your instances to accept connections from each other instance in your cluster.

On the instance you have selected as your driver node, you will need to start a Spark master. You can do so by navigating to the ‘spark’ folder and executing the following command:

    ./sbin/start-master.sh

The Spark master UI is a useful tool for monitoring the status of your cluster. You can open up the Spark master UI by entering the following into the search bar of a web browser.

    {public IPv4 address of your driver node}:8080

Make sure that you have opened port 8080 on your driver node. This is the same as the process for opening ports, but you must set the IP address to your IP address and set the port range to 8080. Once you open up the master UI, record the master URL in the URL section of the UI. A typical master URL will look something like this (the zeroes will be replaced with the private IPv4 address of your driver node):

    spark://ip-000-00-0-000.ec2.internal:7077

On each of your worker machines, you must start a Spark worker. You can do so by navigating to the ‘spark’ folder and executing the following command.

    ./sbin/start-worker.sh {URL of your spark master}

After you have started a worker, it will appear in the Spark master UI, under the workers section. If it does not appear, make sure that your ports are opened.

Each Spark worker you create will also provide a worker UI to allow you to monitor the status of that worker. You can open up a worker node’s UI by entering the following into the search bar of a Web browser. 

    {public IPv4 address of the worker node}:8081

Make sure that you have opened port 8081 on the worker node.

You now have a Spark cluster running. You can run spark applications using the following command on your driver node:

    {path to the ‘spark’ folder}/bin/spark-submit --master {master URL} {path to your spark application}

When you run a Spark application, a job UI will be available which will allow you to monitor the status of your cluster as it performs the Spark job. You can open up the job UI by entering the following into the search bar of a Web browser:

    {public IPv4 address of your driver node}:4040

Make sure that you have opened port 4040 on your driver node. Note that the job UI is only available when you have a Spark job running on your cluster; once the job finishes, the UI will become unavailable.

#### Testing the Spark Cluster
If you want to test the functionality of your Spark cluster, a number of examples are provided with the Spark download. You can run one of the simpler ones with the following command:

    {path to the ‘spark’ folder}/bin/spark-submit --master {master URL} {path to the ‘spark’ folder}/examples/src/main/python/pi.py 10

If this runs successfully, it will appear as a finished job in the master and worker UIs.

