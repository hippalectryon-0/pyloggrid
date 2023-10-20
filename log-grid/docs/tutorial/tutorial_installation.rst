************
Installation
************

Via pip
*******

The easiest way is to simply install the package via ``pip``:

.. code-block:: bash

   git install pyloggrid

.. important:: Requirements

    - Windows:
        You must have ``gcc`` and ``make`` installed. You can easily obtain them by following these steps:

        1. Install Chocolatey (a package manager for Windows) if you haven't already. Open a Command Prompt or PowerShell with administrative privileges and run the following command:

            .. code-block:: bash

                Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

        2. After Chocolatey is installed, install ``gcc`` and ``make`` using Chocolatey:

            .. code-block:: bash

                choco install mingw
                choco install make

    - Linux:
        Ensure you have ``clang`` and the necessary build tools installed. You can use the following commands to install them on Ubuntu:

            .. code-block:: bash

                sudo apt update && sudo apt install clang build-essential


.. warning:: The Windows version is significantly less optimized than the Linux version, and does not support multithreading.


Manual installation (Linux)
***************************

*This guide is written for Ubuntu 22.x. If you are using a different version of Linux, you may need to adapt the instructions.*

Install git:

.. code-block:: bash

   sudo apt install git

Clone the repository:

.. code-block:: bash

   git clone https://drf-gitlab.cea.fr/amaury.barral/log-grid.git
   cd log-grid

Run the install script:

.. code-block:: bash

   source ./install.sh

This script will install required packages and dependencies, create a Python virtual environment, and compile the C code.

Manual installation (Windows)
*****************************

*Note: the windows installation script may be less stable. Moreover, the windows version does not support multithreading.*

Install `git <https://git-scm.com/download/win>`_.

Clone the repository:

.. code-block:: bash

   git clone https://drf-gitlab.cea.fr/amaury.barral/log-grid.git
   cd log-grid

Run the install script:

.. code-block:: bash

   powershell.exe -ExecutionPolicy bypass
   ./install.ps1

This will download python locally and install mingw via choco, download the required dependencies and compile the C code. It might ask for admin rights to install choco, and open popups during the python extraction.
