#!/bin/zsh

# Define source directory
export SRC_DIR=$(dirname "${0:A}")

# Find all Python files in the directory and its subdirectories, excluding those in models/transformers_modules
find "${SRC_DIR}" -type f -name "*.py" | grep -v "^${SRC_DIR}/amp/models/transformers_modules" | xargs -P 16 -I {} sh -c '
    echo "Processing {}"
    pipx run autoflake --in-place --remove-unused-variables --remove-all-unused-imports "{}"
    pipx run autopep8 --in-place --aggressive --aggressive "{}"
    pipx run isort "{}"
'
