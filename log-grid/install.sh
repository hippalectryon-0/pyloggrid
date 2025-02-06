#!/bin/bash

# Check if the script is sourced
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "This script must be sourced. Use 'source $0' or '. $0' to run it."
    exit 1
fi

# Install uv if necessary
if ! which uv &> /dev/null; then
  wget -qO- https://astral.sh/uv/install.sh | sh
fi

# ensure clang & omp are available
sudo apt update && sudo apt install -y build-essential clang-15 libomp-15-dev
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 1

uv sync --group docs --group examples
uv build
source .venv/bin/activate
rm -rf dist
pre-commit install
