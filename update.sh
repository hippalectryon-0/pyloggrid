#!/bin/bash
# This script is used to update the Github source from the Gitlab source. You need Gitlab access to run it.

if [ -z "$1" ]; then
    echo "Error: Please provide a non-empty tag as a command-line argument."
    exit 1
fi

tag="$1"

rm -rf log-grid || true
git config advice.detachedHead false
git clone "https://drf-gitlab.cea.fr/amaury.barral/log-grid.git" -b "$tag"
cd log-grid || exit 1
rm -rf .git || true
cd ..
