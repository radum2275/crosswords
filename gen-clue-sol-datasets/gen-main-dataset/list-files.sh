#!/bin/bash

# Check if 5 folders are provided
if [ $# -ne 5 ]; then
    echo "Usage: $0 <folder1> <folder2> <folder3> <folder4> <folder5>"
    exit 1
fi

# Assign folders to variables
FOLDER1="$1"
FOLDER2="$2"
FOLDER3="$3"
FOLDER4="$4"
FOLDER5="$5"

# Verify all folders exist
for folder in "$FOLDER1" "$FOLDER2" "$FOLDER3" "$FOLDER4" "$FOLDER5"; do
    if [ ! -d "$folder" ]; then
        echo "Error: Directory '$folder' does not exist"
        exit 1
    fi
done

# Output CSV file
OUTPUT="folder_comparison.csv"

echo "Creating CSV file: $OUTPUT"

# Get file lists from each folder
files1=($(find "$FOLDER1" -type f -printf "%f\n" | sort))
files2=($(find "$FOLDER2" -type f -printf "%f\n" | sort))
files3=($(find "$FOLDER3" -type f -printf "%f\n" | sort))
files4=($(find "$FOLDER4" -type f -printf "%f\n" | sort))
files5=($(find "$FOLDER5" -type f -printf "%f\n" | sort))

# Find the maximum number of files
max_files=${#files1[@]}
[ ${#files2[@]} -gt $max_files ] && max_files=${#files2[@]}
[ ${#files3[@]} -gt $max_files ] && max_files=${#files3[@]}
[ ${#files4[@]} -gt $max_files ] && max_files=${#files4[@]}
[ ${#files5[@]} -gt $max_files ] && max_files=${#files5[@]}

# Write CSV header
echo "\"$FOLDER1\",\"$FOLDER2\",\"$FOLDER3\",\"$FOLDER4\",\"$FOLDER5\"" > "$OUTPUT"

# Write file rows
for ((i=0; i<$max_files; i++)); do
    col1="${files1[$i]:-}"
    col2="${files2[$i]:-}"
    col3="${files3[$i]:-}"
    col4="${files4[$i]:-}"
    col5="${files5[$i]:-}"
    
    # Escape quotes in filenames
    col1="${col1//\"/\"\"}"
    col2="${col2//\"/\"\"}"
    col3="${col3//\"/\"\"}"
    col4="${col4//\"/\"\"}"
    col5="${col5//\"/\"\"}"
    
    echo "\"$col1\",\"$col2\",\"$col3\",\"$col4\",\"$col5\"" >> "$OUTPUT"
done

echo "CSV file created successfully!"
echo "Total rows: $max_files files"