#!/bin/bash -x

# This script is called from $GIT_ROOT/.github/workflows/build-test.yml
# This script is called while in $GIT_ROOT/chia-blockchain of chialisp

. ./venv/bin/activate

python -m pip install --upgrade pip
python -m pip uninstall clvm clvm_rs clvm_tools chialisp

git clone https://github.com/Chia-Network/clvm.git --branch=main --single-branch
python -m pip install ./clvm

echo "installing clvm_rs via pip"
pip install clvm_rs

echo "installing clvm_tools for clvm tests"

# Ensure clvm_tools is installed from its own repo.
git clone https://github.com/Chia-Network/clvm_tools.git --branch=main --single-branch
python -m pip install ./clvm_tools

# Install chialisp from the directory above.
python -m pip install ..
