import json
import re

input_file = "filtered-data-pairs.txt"
output_file = "baseline-dataset.json"

records = []
current_path = None

with open(input_file, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        # Detect path line
        m = re.match(r"dos2unix: converting file (.+?) to Unix format", line)
        if m:
            current_path = m.group(1)
            continue
        # Detect record line
        if " -- " in line and current_path:
            solution, clue = line.split(" -- ", 1)
            records.append({
                "solution": solution.strip(),
                "clue": clue.strip(),
                "path": current_path
            })

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)