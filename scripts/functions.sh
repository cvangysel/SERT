#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Setup cvangysel-common library.
git submodule init
git submodule update

source "${SCRIPT_DIR}/../cvangysel-common/bash/functions.sh"

check_directory "$(package_root)/cvangysel-common/py/cvangysel"
check_file "$(package_root)/cvangysel-common/py/cvangysel/__init__.py"

check_installed "curl"
check_installed "diff"
check_installed "find"
check_installed "git"
check_installed "md5sum"
check_installed "python"
check_installed "sed"
check_installed "sort"
check_installed "tr"
check_installed "trec_eval"
check_installed "uniq"

# Initialization.

# Check Python version.
if [[ ! "$(python --version 2>&1)" =~ "3.5" ]]; then
    echo >&2 "Required tool 'Python 3.5' is not installed. Aborting."

    exit -1
fi

PYTHONPATH="$(package_root):$PYTHONPATH"
export SERT_ROOT=$(python -c "import os; import sert; print(os.path.abspath(sert.__path__[0]))")

# Verify that the correct sert library is selected.
if [[ ! "${SERT_ROOT}" =~ ^$(package_root) ]]; then
    echo "ERROR: using incorrect version of sert (${SERT_ROOT})."

    exit -1
fi

PYTHONPATH="$(package_root)/cvangysel-common/py:$PYTHONPATH"
export CVANGYSEL_ROOT=$(python -c "import os; import cvangysel; print(os.path.abspath(cvangysel.__path__[0]))")

# Verify that the correct cvangysel-common library is selected.
if [[ ! "${CVANGYSEL_ROOT}" =~ ^$(package_root) ]]; then
    echo "WARNING: you are using an external version of cvangysel-common (${CVANGYSEL_ROOT})."
fi
