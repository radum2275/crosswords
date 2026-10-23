# Pick the items for the ICL clue-generation experiment (gen_clues_ro.py
# --prompt icl) from the polycross corpus.
#
# Two disjoint sets are drawn with a fixed seed and saved together, so every
# model sees the same in-context examples and is asked about the same answers:
#   "icl"  -- real (answer, clue) pairs shown to the model as examples
#   "eval" -- answers the model must write new clues for; their original clue is
#             kept as the ground truth
# No answer appears in both sets, so the model is never shown the clue it is
# being asked to reproduce.

import argparse
import json
import os
import random
import unicodedata
from collections import Counter
from typing import Any

from crosswords.backends import load_json_utf8_relaxed

# Answers whose sampled clue was checked by hand and found to be a poor example.
# They are skipped after the shuffle, so excluding one does not reshuffle the
# rest of the sample (the next candidate just moves up).
#   ENTUZIASM -- "Un cuvânt scris de mână" only makes sense inside its themed grid.
#   NOROCOS   -- "Obisnuit a castiga" is written without diacritics, which the
#                prompt asks the model to use.
EXCLUDED_ANSWERS = {"ENTUZIASM", "NOROCOS"}


def clean_clue(clue: str) -> str:
    """Collapse internal whitespace; a few scraped clues contain line breaks."""
    return " ".join(clue.split())


def fold(text: str) -> str:
    """Uppercase and drop diacritics, for comparing an answer with a clue."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).upper()


def is_candidate(
        record: dict[str, Any],
        answer_counts: Counter,
        min_answer_len: int,
        max_clue_words: int,
        max_answer_freq: int,
) -> bool:
    """
    True if the record is a usable, non-trivial (answer, clue) pair.

    Non-trivial means a reasonably long all-letter answer that is not one of the
    short fillers reused across many grids. A usable clue is short, is not a
    thematic heading written in capitals, and does not give the answer away.
    """
    answer = record.get("answer", "")
    clue = clean_clue(record.get("clue", ""))
    if len(answer) < min_answer_len or not answer.isalpha():
        return False
    if answer_counts[answer.upper()] > max_answer_freq:
        return False
    if not 2 <= len(clue.split()) <= max_clue_words:
        return False
    # Headings such as "BAIA MARE" are grid themes, not definitions.
    if clue.upper() == clue:
        return False
    # Reject clues that contain the answer or its first five letters (a crude
    # stem match, so "ACCEPTA" also catches "acceptat").
    folded_clue = fold(clue)
    folded_answer = fold(answer)
    return folded_answer[:5] not in folded_clue


def select(
        data: list[dict[str, Any]],
        num_eval: int,
        num_icl: int,
        seed: int,
        min_answer_len: int,
        max_clue_words: int,
        max_answer_freq: int,
) -> dict[str, list[dict[str, Any]]]:
    """Draw disjoint ICL and eval sets, with each answer used at most once."""
    answer_counts = Counter(r.get("answer", "").upper() for r in data)
    candidates = [
        r for r in data
        if is_candidate(r, answer_counts, min_answer_len, max_clue_words, max_answer_freq)
    ]
    print(f"{len(candidates)} of {len(data)} records are candidates.")

    rng = random.Random(seed)
    rng.shuffle(candidates)

    seen: set[str] = set()
    picked: list[dict[str, Any]] = []
    for record in candidates:
        key = record["answer"].upper()
        if key in seen or key in EXCLUDED_ANSWERS:
            continue
        seen.add(key)
        picked.append({
            "answer": record["answer"],
            "clue": clean_clue(record["clue"]),
            "filePath": record.get("filePath", ""),
        })
        if len(picked) == num_icl + num_eval:
            break

    if len(picked) < num_icl + num_eval:
        raise SystemExit(
            f"Only {len(picked)} distinct candidates; need {num_icl + num_eval}."
        )
    return {"icl": picked[:num_icl], "eval": picked[num_icl:]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', type=str, default="data/polycross_data.json")
    parser.add_argument('--output_file', type=str,
                        default="data/gen_clues/polycross_icl_sample.json")
    parser.add_argument('--num_eval', type=int, default=20)
    parser.add_argument('--num_icl', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min_answer_len', type=int, default=5)
    parser.add_argument('--max_clue_words', type=int, default=6)
    parser.add_argument('--max_answer_freq', type=int, default=3,
                        help='Skip answers that occur more often than this in the '
                             'corpus (the reused short fillers)')
    args = parser.parse_args()

    sample = select(
        load_json_utf8_relaxed(args.dataset_file),
        num_eval=args.num_eval,
        num_icl=args.num_icl,
        seed=args.seed,
        min_answer_len=args.min_answer_len,
        max_clue_words=args.max_clue_words,
        max_answer_freq=args.max_answer_freq,
    )
    sample["params"] = {k: v for k, v in vars(args).items() if k != "output_file"}

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)

    for name in ("icl", "eval"):
        print(f"\n{name} ({len(sample[name])}):")
        for r in sample[name]:
            print(f"  {r['answer']:14s} | {r['clue']}")
    print(f"\nSaved to {args.output_file}")
