#!/bin/bash

# Get the directory path
dir=$1

# Check if the directory exists
if [ ! -d "$dir" ]; then
  echo "Error: directory does not exist"
  exit 1
fi

# Loop through all files in the directory
for file in "$dir"/test_*; do
  # Check if the file is executable
  if [ -x "$file" ]; then
    # Execute the file
    "$file"
  else
    echo "Error: $file is not executable"
  fi
done
