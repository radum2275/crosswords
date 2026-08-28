#!/usr/bin/env python3
"""Select N random records from a JSON dataset, skipping records whose answer length is two."""
import argparse
import json
import random
import sys
from typing import Any, Dict, List, Optional

COMMON_ANSWER_KEYS = ("answer", "solution", "solution_text", "answer_text", "sol")
SOLS_TO_SKIP = {"DETA", "COD", "SIE","TASTE", "NIMB", "ADI"}
PUZZLES_TO_SKIP = {"/home/adi/workspace/rebus/careuri-definitii/js/fantezie-geografica.js",
                   "/home/adi/workspace/rebus/careuri-definitii-4/js/fluturas-combus1.js",}


def load_json(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        # try newline-delimited JSON
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return items


def get_answer_field(record: Dict[str, Any], preferred: Optional[str]) -> Optional[str]:
    if preferred and preferred in record:
        val = record.get(preferred)
    else:
        val = None
        for k in COMMON_ANSWER_KEYS:
            if k in record:
                val = record.get(k)
                break
    if isinstance(val, list):
        # join lists of tokens if present
        try:
            return "".join(val)
        except Exception:
            return None
    if val is None:
        return None
    return str(val)


def filter_records(data: List[Dict[str, Any]], answer_field: Optional[str]) -> List[Dict[str, Any]]:
    out = []
    for rec in data:
        ans = get_answer_field(rec, answer_field)
        if isinstance(ans, str) and len(ans.strip()) == 2:
            continue
        if ans in SOLS_TO_SKIP:
            continue
        if isinstance(ans, str) and not ans.isalpha():
            continue
        clue = get_answer_field(rec, "clue")
        if isinstance(clue, str) and "!" in clue:
            continue
        puzzle = get_answer_field(rec, "filePath")
        if isinstance(puzzle, str) and puzzle in PUZZLES_TO_SKIP:
            continue
        out.append(rec)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Select N random records from JSON, skipping two-letter answers")
    p.add_argument("input", help="Input JSON file (array or newline-delimited JSON)")
    p.add_argument("output", help="Output JSON file to write selected records")
    p.add_argument("-n", "--num", type=int, required=True, help="Number of records to select")
    p.add_argument("--answer-field", default=None, help="Explicit answer field name if dataset uses a different key")
    p.add_argument("--replace", action="store_true", help="Sample with replacement if requested number > available")
    p.add_argument("--seed", type=int, default=30, help="Integer seed for reproducible sampling")
    args = p.parse_args()

    data = load_json(args.input)
    if not isinstance(data, list):
        print("Input JSON did not decode to a list. Exiting.", file=sys.stderr)
        sys.exit(1)

    eligible = filter_records(data, args.answer_field)
    if not eligible:
        print("No eligible records after filtering out two-letter answers.", file=sys.stderr)
        sys.exit(1)

    if args.seed is not None:
        random.seed(args.seed)

    if args.replace:
        chosen = [random.choice(eligible) for _ in range(args.num)]
    else:
        if args.num > len(eligible):
            print(f"Requested {args.num} records but only {len(eligible)} eligible available. Use --replace to allow sampling with replacement.", file=sys.stderr)
            sys.exit(2)
        chosen = random.sample(eligible, args.num)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(chosen, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(chosen)} records to {args.output}")


if __name__ == "__main__":
    main()
