# This script is used to update the Github source rom the Gitlab source. You need Gitlab access to run it.

Remove-Item -Path .\log-grid -Recurse -ErrorAction SilentlyContinue
git clone https://drf-gitlab.cea.fr/amaury.barral/log-grid.git -b 2.2.0  # Change the tag accordingly
cd log-grid
Remove-Item -Recurse -Force .git
Set-Location ../
