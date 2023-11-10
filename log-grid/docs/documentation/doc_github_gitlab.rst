****************
Github vs Gitlab
****************

PyLogGrid is hosted both on `Github <https://github.com/hippalectryon-0/pyloggrid>`_ and `Gitlab <https://drf-gitlab.cea.fr/amaury.barral/log-grid>`_.

The main development happens on the Gitlab for practical reasons. However, the Gitlab is not open to the public.
To bypass this problem, the code is mirrored (with slight adaptations, e.g. the CI) to Gitlab.
This enables both the `pypi <https://pypi.org/project/pyloggrid/>`_ release from which one can ``pip install pyloggrid``, the `public documentation <https://pyloggrid.readthedocs.io/>`_, as well as issues and pull requests from the community.

Notable differences
###################

* The Gitlab source is mirrored inside the **subfolder** ``log-grid`` in Github.
* Github-specific files (README, .gitignore, public docs configurations, Github workflow) are put in the root folder.
* A few Gitlab folders, such as private archives, are excluded from the Github version.

Mirroring process
#################

Right now, a script on the Github repository ``update.[ps1/sh]`` has to be ran manually on the latest tag for each new Gitlab release, then we create a Github release.

We hope to automate this process in the future.



