#!/bin/bash -x

if [ "x$1" = x ] ; then
    echo "usage: test-game-referee.sh [game-referee-version]"
    exit 1
fi

REF_SUBDIR="$1"

python -m pip install --upgrade pip
python -m pip install "chia_rs>=0.50.0,<0.51"
python -m pip install clvm_tools
python -m pip install pytest

export PYTHONPATH=$PWD/resources/tests:.:$PYTHONPATH
(cd "${REF_SUBDIR}" && pytest -s .)
