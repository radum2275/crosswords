# Rescore every solving-task result file on a common item set with a
# diacritic-insensitive metric.
#
# Two result formats coexist in data/results/:
#   - {clues,baseline}_{granite,llama,gpt-oss}_v{V}_{K}.json: a list of records
#     (clue/answer/prediction/rationale) followed by a summary dict. On polycross
#     these cover the full ~26k corpus, and failed items are silently missing.
#   - {polycross,themcross}_v{V}[_k{K}]_claude_opus48.json: a dict whose
#     "predictions" list is position-aligned with the dataset (3000 / 1148 items).
#
# Everything is aligned to the same items -- polycross_sample3000.json and
# themcross_data.json (answers >= 3 letters) -- so all models share one
# denominator; an item missing from a run counts as wrong.
#
# Usage:
#   PYTHONPATH=src python -m crosswords.analysis.rescore

import argparse
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from crosswords.backends import load_json_utf8_relaxed

RESULTS_DIR = "data/results"
POLYCROSS_SAMPLE = "data/polycross_sample3000.json"
THEMCROSS_DATA = "data/themcross_data.json"
POLYCROSS_FULL = "data/polycross_data.json"

MODELS = {
    "granite": "Granite-4.0-small",
    "llama": "Llama-3.3-70B",
    "gpt-oss": "gpt-oss-120B",
    "claude": "Claude Opus 4.8",
    # Base model, run with solve_base_ro.py (few-shot completion prompts; v1-v3 only).
    "llmic-v2": "LLMic-v2-3B",
}
MODEL_ORDER = list(MODELS.values())

# The experimental grid: (version, prefix length k). v3_0/v4_0 duplicate v2 and
# are not part of the design; v6 is the reverse task.
CONDITIONS = [("v1", 0), ("v2", 0)] + [("v3", k) for k in range(1, 5)] \
    + [("v4", k) for k in range(1, 5)] + [("v5", 0)]

# Old-model runs on llama/polycross v3_4 were never produced on the full corpus;
# it was re-run on the 3000 sample under this name.
EXTRA_RUNS = {
    ("polycross", "llama", "v3", 4): "clues_sample3000_llama-3.3-70b-instruct_v3_4.json",
}

OLD_RE = re.compile(r"^(clues|baseline)_(granite|llama|gpt-oss|llmic-v2)_(v\d)_(\d)\.json$")
CLAUDE_RE = re.compile(r"^(polycross|themcross)_(v\d)(?:_k(\d))?_claude_opus48\.json$")
DATASET_ALIAS = {"clues": "polycross", "baseline": "themcross"}


def normalize(s: Optional[str]) -> str:
    """Uppercase, strip diacritics (ă->A, ș/ş->S, ...) and keep letters only."""
    if not s:
        return ""
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^A-Z]", "", s.upper())


def strict_match(answer: str, prediction: Optional[str]) -> bool:
    """The original metric: case-insensitive exact match after strip."""
    return (prediction or "").strip().lower() == answer.strip().lower()


def item_key(clue: str, answer: str) -> Tuple[str, str]:
    return clue.strip(), answer.strip().upper()


def load_items() -> Dict[str, pd.DataFrame]:
    """The common item sets, one row per item with a stable item_id."""
    poly = load_json_utf8_relaxed(POLYCROSS_SAMPLE)
    them = [r for r in load_json_utf8_relaxed(THEMCROSS_DATA) if len(r["solution"]) >= 3]
    items = {
        "polycross": pd.DataFrame({
            "item_id": [r["idx"] for r in poly],
            "clue": [r["clue"] for r in poly],
            "answer": [r["answer"] for r in poly],
        }),
        "themcross": pd.DataFrame({
            "item_id": range(len(them)),
            "clue": [r["clue"] for r in them],
            "answer": [r["solution"] for r in them],
        }),
    }
    for df in items.values():
        df["answer_norm"] = df["answer"].map(normalize)
        df["length"] = df["answer_norm"].str.len()
    return items


def records_of(path: str) -> List[Dict[str, Any]]:
    data = load_json_utf8_relaxed(path)
    if isinstance(data, dict):
        return data["predictions"]
    return [r for r in data if isinstance(r, dict) and "clue" in r]


def discover_runs(results_dir: str = RESULTS_DIR) -> List[Dict[str, Any]]:
    runs = []
    wanted = set(CONDITIONS)
    for name in sorted(os.listdir(results_dir)):
        m = OLD_RE.match(name)
        if m:
            ds, model, v, k = DATASET_ALIAS[m[1]], m[2], m[3], int(m[4])
            fmt = "old"
        else:
            m = CLAUDE_RE.match(name)
            if not m:
                continue
            ds, model, v, k = m[1], "claude", m[2], int(m[3] or 0)
            fmt = "claude"
        if (v, k) in wanted:
            runs.append(dict(dataset=ds, model=model, version=v, k=k, fmt=fmt,
                             path=os.path.join(results_dir, name)))
    for (ds, model, v, k), name in EXTRA_RUNS.items():
        path = os.path.join(results_dir, name)
        if os.path.exists(path):
            runs = [r for r in runs if (r["dataset"], r["model"], r["version"], r["k"]) != (ds, model, v, k)]
            runs.append(dict(dataset=ds, model=model, version=v, k=k, fmt="old", path=path))
    return runs


def align(run: Dict[str, Any], items: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Predictions for every item of the common set (None where missing)."""
    recs = records_of(run["path"])
    if run["fmt"] == "claude":
        # Position-aligned with the dataset; verify rather than trust it.
        assert len(recs) == len(items), run["path"]
        assert all(r["clue"] == c for r, c in zip(recs, items["clue"])), run["path"]
        preds = [r.get("prediction") for r in recs]
        rats = [None] * len(recs)
        found = len(recs)
    else:
        by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for r in recs:
            by_key.setdefault(item_key(r["clue"], r["answer"]), r)
        hits = [by_key.get(item_key(c, a)) for c, a in zip(items["clue"], items["answer"])]
        preds = [h.get("prediction") if h else None for h in hits]
        rats = [h.get("rationale") if h else None for h in hits]
        found = sum(h is not None for h in hits)

    df = items[["item_id", "clue", "answer", "answer_norm", "length"]].copy()
    df["dataset"], df["model"] = run["dataset"], MODELS[run["model"]]
    df["version"], df["k"] = run["version"], run["k"]
    # A few models answered with a bare number or a list; keep them as text.
    preds = [None if p is None else str(p) for p in preds]
    df["prediction"] = pd.Series(preds, index=df.index, dtype=object)
    df["rationale"] = rats
    df["missing"] = [p is None for p in preds]
    df["pred_norm"] = df["prediction"].map(normalize)
    df["correct_strict"] = [strict_match(a, p) for a, p in zip(df["answer"], df["prediction"])]
    df["correct"] = df["pred_norm"] == df["answer_norm"]

    info = dict(dataset=run["dataset"], model=MODELS[run["model"]], version=run["version"],
                k=run["k"], file=os.path.basename(run["path"]), n_in_file=len(recs),
                n_items=len(items), n_found=found,
                acc_strict=df["correct_strict"].mean(), acc=df["correct"].mean())
    # For continuity with the paper: the old polycross runs scored on everything
    # they contain (the full ~26k corpus), with the file's own denominator.
    if run["fmt"] == "old" and run["dataset"] == "polycross" and len(recs) > len(items):
        info["full_n"] = len(recs)
        info["full_acc_strict"] = sum(strict_match(r["answer"], r.get("prediction")) for r in recs) / len(recs)
        info["full_acc"] = sum(normalize(r["answer"]) == normalize(r.get("prediction")) for r in recs) / len(recs)
    return df, info


def main(out_dir: str, results_dir: str = RESULTS_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    items = load_items()
    frames, infos = [], []
    for run in discover_runs(results_dir):
        df, info = align(run, items[run["dataset"]])
        frames.append(df)
        infos.append(info)
        print(f"{info['dataset']:9} {info['model']:18} {info['version']} k={info['k']} "
              f"found {info['n_found']}/{info['n_items']}  strict={info['acc_strict']:.3f} "
              f"nodiac={info['acc']:.3f}")

    preds = pd.concat(frames, ignore_index=True)
    preds.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    runs = pd.DataFrame(infos).sort_values(["dataset", "model", "version", "k"])
    runs.to_csv(os.path.join(out_dir, "runs.csv"), index=False)

    # Base models only have v1-v3 prompts; do not report their v4/v5 as missing.
    base_only = {MODELS["llmic-v2"]}
    expected = {(ds, m, v, k) for ds in items for m in MODEL_ORDER for v, k in CONDITIONS
                if not (m in base_only and v in ("v4", "v5"))}
    have = set(zip(runs["dataset"], runs["model"], runs["version"], runs["k"]))
    for cell in sorted(expected - have):
        print("MISSING run:", cell)
    print(f"Wrote {len(preds)} rows to {out_dir}/predictions.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/results/analysis")
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    args = parser.parse_args()
    main(args.out_dir, args.results_dir)
