Remove-Item -Path .\log-grid -Recurse -ErrorAction SilentlyContinue
git clone https://drf-gitlab.cea.fr/amaury.barral/log-grid.git
cd log-grid
git checkout pypi-package
Remove-Item -Recurse -Force .git
Set-Location ../