#!/bin/zsh

# Check if file location is provided
if [ $# -ne 1 ]; then
  echo "Usage: $0 <file_location>"
  exit 1
fi

file_location=$1

# Check if the file exists
if [ ! -f "$file_location" ]; then
  echo "Error: The file $file_location does not exist."
  exit 1
fi

# Run autoflake to remove unused variables and imports
echo "Running autoflake..."
pipx run autoflake --in-place --remove-unused-variables --remove-all-unused-imports "$file_location"

# Run autopep8 to fix PEP 8 formatting aggressively
echo "Running autopep8..."
pipx run autopep8 --in-place --aggressive --aggressive "$file_location"

# Run isort to sort imports
echo "Running isort..."
pipx run isort "$file_location"

echo "Done!"