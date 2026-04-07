#!/bin/bash -x

if [[ $# -ne 1 ]]; then
    echo "Illegal number of parameters" >&2
    echo "Usage: $0 <version_string>" >&2
    exit 2
fi

pip_installed_version=$(pip list | grep '^chialisp\s' | awk '{print $2}')
python_import_version=$(python -c 'import chialisp; print(chialisp.get_version())')

expected_version="$1"

if [ "$expected_version" == "$pip_installed_version" ] && [ "$expected_version" == "$python_import_version" ]; then
    exit 0
else
    echo "chialisp VERSIONS does not match expected version"
    echo "PIP INSTALLED VERSION: $pip_installed_version"
    echo "PYTHON IMPORTED VERSION: $python_import_version"
    echo "EXPECTED VERSION: $expected_version"
    exit 1
fi
