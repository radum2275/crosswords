#!/bin/bash

# Check if 8 folders are provided
if [ $# -ne 8 ]; then
    echo "Usage: $0 <folder1> <folder2> <folder3> <folder4> <folder5> <folder6> <folder7> <folder8>"
    exit 1
fi

# Assign folders to variables
FOLDER1="$1"
FOLDER2="$2"
FOLDER3="$3"
FOLDER4="$4"
FOLDER5="$5"
FOLDER6="$6"
FOLDER7="$7"
FOLDER8="$8"

# Verify all folders exist
for folder in "$FOLDER1" "$FOLDER2" "$FOLDER3" "$FOLDER4" "$FOLDER5" "$FOLDER6" "$FOLDER7" "$FOLDER8"; do
    if [ ! -d "$folder" ]; then
        echo "Error: Directory '$folder' does not exist"
        exit 1
    fi
done

# Output CSV file
OUTPUT="files_with_paths.csv"

echo "Creating CSV file: $OUTPUT"

# Write CSV header
echo "\"Filename\",\"Full Path\"" > "$OUTPUT"

# Process each folder
for folder in "$FOLDER1" "$FOLDER2" "$FOLDER3" "$FOLDER4" "$FOLDER5" "$FOLDER6" "$FOLDER7" "$FOLDER8"; do
    while IFS= read -r -d '' filepath; do
        # Get just the filename
        filename=$(basename "$filepath")
        
        # Get absolute path
        fullpath=$(realpath "$filepath")
        
        # Escape quotes in filename and path
        filename="${filename//\"/\"\"}"
        fullpath="${fullpath//\"/\"\"}"
        
        # Write to CSV
        echo "\"$filename\",\"$fullpath\"" >> "$OUTPUT"
    done < <(find "$folder" -type f -print0)
done

# Sort the CSV by filename (keeping header)
(head -n 1 "$OUTPUT" && tail -n +2 "$OUTPUT" | sort -t',' -k1,1) > "${OUTPUT}.tmp" && mv "${OUTPUT}.tmp" "$OUTPUT"

echo "CSV file created successfully!"
echo "Total files: $(($(wc -l < "$OUTPUT") - 1))"