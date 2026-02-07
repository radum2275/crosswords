import csv
import sys

def count_matches_and_rational_score(csv_file):
    match_true_count = 0
    rational_score_5_count = 0
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Count match == True (accept both string and bool representations)
            if str(row.get('match', '')).strip().lower() == 'true':
                match_true_count += 1
            # Count rationale_score >= 5
            try:
                score = float(row.get('rationale_score', '-inf'))
                if score >= 5:
                    rational_score_5_count += 1
            except ValueError:
                pass
    print(f"Total records: {reader.line_num - 2}")  # Subtract header
    print(f"Records with match = True: {match_true_count}")
    print(f"Records with rationale_score >= 5: {rational_score_5_count}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python count_csv_stats.py <csv_file>")
    else:
        count_matches_and_rational_score(sys.argv[1])
