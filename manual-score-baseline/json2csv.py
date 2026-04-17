import json
import csv
import sys

def json_to_csv(json_file, csv_file):
    """
    Convert JSON file to CSV format.
    
    Args:
        json_file: Path to input JSON file
        csv_file: Path to output CSV file
    """
    # Read JSON data
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check if data is not empty
    if not data:
        print("No data to convert")
        return
    
    # Get all unique keys from all objects (in case some objects have different keys)
    all_keys = set()
    for item in data:
        all_keys.update(item.keys())
    
    fieldnames = ['rationale_score', 'match', 'clue', 'answer', 'prediction', 'rationale'] 

    # Write to CSV
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        filtered_data = []
        for item in data:
            row = {k: item.get(k, '') for k in fieldnames if k != 'match'}
            # Convert rationale to single line
            if 'rationale' in row and isinstance(row['rationale'], str):
                row['rationale'] = row['rationale'].replace('\n', ' ').replace('\r', ' ').replace('\u2028', ' ').replace('\u2029', ' ')
                row['rationale'] = ' '.join(row['rationale'].split())
            answer = item.get('answer', '').lower()
            prediction = item.get('prediction', '').lower()
            row['match'] = str(answer == prediction)
            filtered_data.append(row)
        writer.writerows(filtered_data)
    
    print(f"Successfully converted {json_file} to {csv_file}")

# Example usage
if __name__ == "__main__":
    # Method 1: From file
    output_file = sys.argv[1].replace('.json', '_output.csv')
    json_to_csv(sys.argv[1], output_file)