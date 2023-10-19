************
Installation
************

Linux
*****

*This guide is written for Ubuntu 22.x. If you are using a different version of Linux, you may need to adapt the instructions.*

Install git:

.. code-block:: bash

   sudo apt install git

Clone the repository:

.. code-block:: bash

   git clone https://drf-gitlab.cea.fr/amaury.barral/log-grid.git
   cd log-grid

Run the install script (make sure to use source to enter the virtualenv):

.. code-block:: bash

   source ./install.sh

This script will install required packages and dependencies, create a Python virtual environment, and compile the C code.

Windows
*******

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
