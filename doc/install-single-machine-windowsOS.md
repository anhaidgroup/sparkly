## How to Install Sparkly on a Windows Machine

This is a step-by-step guide to install Sparkly and its necessary dependencies on a single Windows machine.

This guide has been tested only on a Dell Latitude 3340 with x86-64 bit architecture, running Windows 11 Home Edition, and having 16 GB of RAM. The test setup used WSL 2 (Ubuntu) and installed Python 3.10, the g++ compiler, Eclipse Temurin JDK 17, and PyLucene 9.4.1.

If you’re running a 64-bit version of Windows (Windows 10, Windows 11, or Windows Server) but your system specifications differ from the test machine, these steps may still work. However, we have not validated them on other configurations and cannot guarantee compatibility.

### Step 1: WSL Installation

Sparkly requires PyLucene, which does not provide clear Windows installation documentation. So we will install WSL (Windows Subsystem for Linux) to create a virtual Linux environment on your Windows machine (and then install PyLucene in that Linux environment). 

To install WSL, execute the following steps: 

#### Step 1.1: Checking for WSL

First, open Windows PowerShell: Press the Windows key, type PowerShell, and select Windows PowerShell. (Do not use Windows PowerShell (x86) or Windows PowerShell ISE.)

In the PowerShell window, run:

`wsl --list --verbose`

You should see one of the following:

1. A list of installed WSL distributions (it will start with “Windows Subsystem for Linux Distributions:”).
   - If Ubuntu appears in the list, complete Step 1.2, skip Step 1.3, and complete Steps 1.4 and 1.5 before proceeding to Step 2.
   - If you see a list but Ubuntu is not included, complete Steps 1.2, 1.3, 1.4, and 1.5 before proceeding to Step 2.
2. A message indicating no distributions are installed (e.g., “Windows Subsystem for Linux has no installed distributions”).
   - In this case, complete Steps 1.2, 1.3, 1.4, and 1.5 before proceeding to Step 2.

#### Step 1.2: Setting the default WSL Version

WSL has two versions. WSL 1 translates Linux system calls to Windows, while WSL 2 runs Linux in a lightweight virtual machine. This guide requires WSL 2.

To set WSL 2 as the default for newly installed distributions, run:

`wsl --set-default-version 2`

After this, you should proceed to Step 1.3 if Ubuntu is not already installed as indicated in Step 1.1. If Ubuntu is already installed as indicated in Step 1.1, proceed to Step 1.4.

#### Step 1.3: Installing WSL and Ubuntu

To install WSL and the default Ubuntu distribution, open PowerShell as Administrator. To do so, locate PowerShell on your computer, right-click on the icon, and click "Run as administrator". Then, within the PowerShell session, run the command:

`wsl.exe --install -d Ubuntu`

You may be prompted to restart Windows to finish enabling WSL.

After the installation completes, you’ll be prompted to create a default Unix user account. Choose a username and password and keep them for later use. Note that your password won't be visible as you type it. Once this is complete, PowerShell will open an Ubuntu (WSL) terminal session. 

_Note: Aside from this first time, however, opening an Ubuntu (WSL) terminal session will need to be done manually. To do so, open PowerShell (you do not need to "Run as administrator" anymore), and execute:_

_`wsl`

This will open an Ubuntu (WSL) terminal shell.

#### Step 1.4: Verifying that Ubuntu Uses WSL 2

Whether Ubuntu was already installed or you just installed it, confirm that Ubuntu is running under WSL 2.
Run:

`wsl --list --verbose`

If Ubuntu shows VERSION 1, convert it to WSL 2:

`wsl --set-version Ubuntu 2`

#### Step 1.5: Setting the Default WSL Distribution

If you have multiple WSL distributions installed, you need to set Ubuntu as the default for this guide.

To do so, open a new PowerShell window (or a new tab in Windows Terminal), then run:

`wsl --set-default Ubuntu`

After this, running `wsl` from PowerShell will launch the Ubuntu distribution by default.

### Step 2: Installing Sparkly and its Dependencies

Now that WSL is installed, open an Ubuntu (WSL) terminal by running `wsl` from PowerShell. Now, follow the [Linux single machine installation instructions](https://github.com/anhaidgroup/sparkly/blob/main/doc/install-single-machine-linux.md). All of the commands in the Linux guide should be run inside the Ubuntu (WSL) terminal.
