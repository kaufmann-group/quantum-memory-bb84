#!/bin/bash

# this is a useful file to print out all python files in the src folder to upload the code base to AI

set -e 

START_DIR=$(pwd)
OUTPUT_FILE="$START_DIR/.srcout.txt"

> "$OUTPUT_FILE"

REPO=$(git rev-parse --show-toplevel)
cd "$REPO"

if [ ! -d "src" ]; then
    echo "Error: 'src' directory not found in the repository root."
    exit 1
fi

find src -type f \( -name "*.py" \) | while read -r file; do
    echo "//  [$file] --------------------------------------------------------------" >> "$OUTPUT_FILE"
    
    cat "$file" >> "$OUTPUT_FILE"
    
    echo -e "\n\n" >> "$OUTPUT_FILE"
done

echo "All source files have been combined into $OUTPUT_FILE."