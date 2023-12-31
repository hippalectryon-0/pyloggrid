# This script is used to update the Github source from the Gitlab source. You need Gitlab access to run it.

param ([string]$tag)

if ([string]::IsNullOrEmpty($tag)) {
    Write-Host "Error: Please provide a non-empty tag as a command-line argument."
    exit 1
}

Remove-Item -Path .\log-grid -Recurse -ErrorAction SilentlyContinue
git -c advice.detachedHead=false clone "https://drf-gitlab.cea.fr/amaury.barral/log-grid.git" -b "$tag"
cd log-grid
Remove-Item -Recurse -Force .git
Copy-Item -Path ".\.pre-commit-config.yaml" -Destination ".."
Copy-Item -Path ".\LICENSE" -Destination ".."
Set-Location ../
