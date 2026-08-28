"""
This script reads all CSV files in the same directory, identifies rows where the 
'match' column is True across all files, and writes new CSV files containing only those matching rows.
"""


import os
import csv



# Directory containing the CSV files
CSV_DIR = os.path.dirname(os.path.abspath(__file__))

# Get all CSV files in the directory
csv_files = [f for f in os.listdir(CSV_DIR) if f.endswith('.csv')]

# Read all CSV files into lists of rows
all_rows = []
headers = None
for fname in csv_files:
    with open(os.path.join(CSV_DIR, fname), newline='') as f:
        reader = list(csv.reader(f))
        if headers is None:
            headers = reader[0]
        all_rows.append(reader[1:])  # skip header

# Find the index of the 'match' column
try:
    match_idx = headers.index('match')
except ValueError:
    raise Exception("No 'match' column found in CSV files.")

# Find row indices where 'match' is True in all files
num_rows = min(len(rows) for rows in all_rows)
matching_indices = []
for i in range(num_rows):
    if all(rows[i][match_idx].strip().lower() == 'true' for rows in all_rows):
        matching_indices.append(i)

# For each CSV file, write a new file with only the matching rows
for file_idx, fname in enumerate(csv_files):
    out_fname = fname.replace('.csv', '_matched.csv')
    with open(os.path.join(CSV_DIR, out_fname), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i in matching_indices:
            writer.writerow(all_rows[file_idx][i])

print(f"Wrote filtered files: {[f.replace('.csv', '_matched.csv') for f in csv_files]}")
