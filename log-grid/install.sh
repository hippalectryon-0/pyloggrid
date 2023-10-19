#!/bin/bash

sudo apt-get update && sudo apt-get install -y software-properties-common && sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt-get update

python_v="3.11"

# python
if [ "$1" != "-s" ]; then
  sudo apt-get -y install python$python_v python$python_v-dev python$python_v-venv python3-pip
  curl -sS https://bootstrap.pypa.io/get-pip.py | python$python_v
  python$python_v -m pip install poetry
fi

# requirements
python$python_v -m poetry config virtualenvs.in-project true
python$python_v -m poetry install --with=docs,examples
. .venv/bin/activate
pre-commit install

# convolver
sudo apt install -y build-essential clang-15 libomp-15-dev
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 1
cd pyloggrid/LogGrid || exit
make
cd ../../
