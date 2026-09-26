# Error analysis on the rescored predictions (run rescore.py first).
#
# The question driving everything here: when a hint is present, how much of the
# accuracy comes from understanding the clue, and how much from the hint alone
# pinning down the word? Outputs are CSV tables plus LaTeX fragments that
# paper/error_analysis.tex \input{}s.
#
# Usage:
#   PYTHONPATH=src python -m crosswords.analysis.error_analysis

import argparse
import itertools
import os
from collections import Counter, defaultdict
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from crosswords.analysis.rescore import (
    CONDITIONS, MODEL_ORDER, POLYCROSS_FULL, THEMCROSS_DATA, normalize,
)
from crosswords.backends import load_json_utf8_relaxed

DATASETS = ["polycross", "themcross"]
DATASET_LABEL = {"polycross": "PolyCross", "themcross": "ThemCross"}
N_BOOT = 2000
SEED = 0

# Error categories, in the order they are tested (first match wins).
ERROR_TYPES = ["missing", "length", "prefix", "near_miss", "other_word"]
ERROR_LABEL = {
    "missing": "No answer",
    "length": "Wrong length",
    "prefix": "Wrong prefix",
    "near_miss": "One letter off",
    "other_word": "Other word, fits hint",
}
STRATA = ["revealed", "missing_1_2", "missing_3plus"]
STRATUM_LABEL = {
    "revealed": r"$|a| \le k$ (answer fully revealed)",
    "missing_1_2": r"$|a|-k \in \{1,2\}$",
    "missing_3plus": r"$|a|-k \ge 3$",
}


def cond_label(v: str, k: int) -> str:
    return {"v1": "No hint", "v2": "Length", "v5": "Translate"}.get(v, f"{v} k={k}")


# --------------------------------------------------------------------------
# Item-level features
# --------------------------------------------------------------------------

def build_lexicon() -> Set[str]:
    """Every answer from both full corpora -- a proxy for the crossword vocabulary."""
    words = {normalize(r["answer"]) for r in load_json_utf8_relaxed(POLYCROSS_FULL)}
    words |= {normalize(r["solution"]) for r in load_json_utf8_relaxed(THEMCROSS_DATA)}
    return {w for w in words if w}


def candidate_counts(lexicon: Set[str]) -> Dict[tuple, int]:
    """(length, prefix) -> number of lexicon words fitting that hint."""
    counts: Counter = Counter()
    for w in lexicon:
        for k in range(0, 5):
            if len(w) >= k:
                counts[(len(w), w[:k])] += 1
    return counts


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def classify_error(row) -> str:
    """Why a wrong prediction is wrong, testing hint compliance before closeness."""
    p, a, v, k = row.pred_norm, row.answer_norm, row.version, row.k
    if not p:
        return "missing"
    if v != "v1" and len(p) != len(a):
        return "length"
    if v in ("v3", "v4") and not p.startswith(a[:k]):
        return "prefix"
    # One letter off only counts as a near miss when the model had to find at
    # least two letters itself; with a single hidden letter, every wrong word
    # that fits the hint is trivially one edit away.
    if len(a) - (k if v in ("v3", "v4") else 0) >= 2 and edit_distance(p, a) <= 1:
        return "near_miss"
    return "other_word"


def stratum(length: int, k: int) -> str:
    gap = length - k
    if gap <= 0:
        return "revealed"
    return "missing_1_2" if gap <= 2 else "missing_3plus"


def add_features(df: pd.DataFrame, cand: Dict[tuple, int], lexicon: Set[str]) -> pd.DataFrame:
    df = df.copy()
    df["pred_norm"] = df["pred_norm"].fillna("")
    df["stratum"] = [stratum(n, k) for n, k in zip(df["length"], df["k"])]
    hint_k = np.where(df["version"].isin(["v3", "v4"]), df["k"], 0)
    df["n_candidates"] = [cand.get((n, a[:k]), 1) for n, a, k in
                          zip(df["length"], df["answer_norm"], hint_k)]
    df["len_ok"] = df["pred_norm"].str.len() == df["length"]
    df["prefix_ok"] = [p.startswith(a[:k]) for p, a, k in zip(df["pred_norm"], df["answer_norm"], hint_k)]
    wrong = ~df["correct"]
    df["error_type"] = None
    df.loc[wrong, "error_type"] = df[wrong].apply(classify_error, axis=1)
    df["diacritic_only"] = df["correct"] & ~df["correct_strict"]
    df["pred_in_lexicon"] = df["pred_norm"].isin(lexicon)
    return df


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def boot_ci(x: np.ndarray, rng) -> tuple:
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def mcnemar(a: np.ndarray, b: np.ndarray) -> tuple:
    """Exact McNemar test on paired correctness vectors; returns (b_only, a_only, p)."""
    a_only = int((a & ~b).sum())
    b_only = int((b & ~a).sum())
    n = a_only + b_only
    p = binomtest(a_only, n, 0.5).pvalue if n else 1.0
    return b_only, a_only, p


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for (ds, m, v, k), g in df.groupby(["dataset", "model", "version", "k"]):
        c = g["correct"].to_numpy()
        lo, hi = boot_ci(c.astype(float), rng)
        rows.append(dict(dataset=ds, model=m, version=v, k=k, n=len(g),
                         acc=c.mean(), ci_lo=lo, ci_hi=hi,
                         acc_strict=g["correct_strict"].mean(),
                         diacritic_only=int(g["diacritic_only"].sum()),
                         missing=int(g["missing"].sum())))
    return pd.DataFrame(rows)


def strata_table(df: pd.DataFrame) -> pd.DataFrame:
    h = df[df["version"].isin(["v3", "v4"])]
    rows = []
    for (ds, m, v, k, s), g in h.groupby(["dataset", "model", "version", "k", "stratum"]):
        rows.append(dict(dataset=ds, model=m, version=v, k=k, stratum=s, n=len(g), acc=g["correct"].mean()))
    out = pd.DataFrame(rows)
    # Informative-hint accuracy: only items where the hint leaves letters unknown.
    info = (h[h["stratum"] != "revealed"].groupby(["dataset", "model", "version", "k"])["correct"]
            .agg(["mean", "size"]).reset_index().rename(columns={"mean": "acc", "size": "n"}))
    info["stratum"] = "informative"
    return pd.concat([out, info], ignore_index=True)


def candidates_table(df: pd.DataFrame) -> pd.DataFrame:
    # Items whose answer the hint spells out in full would all sit in the
    # 1-candidate bin and swamp it; keep only hints that leave letters hidden.
    h = df[(df["version"] == "v3") & (df["stratum"] != "revealed")].copy()
    bins = [0, 1, 3, 10, 30, 100, np.inf]
    labels = ["1", "2-3", "4-10", "11-30", "31-100", ">100"]
    h["cand_bin"] = pd.cut(h["n_candidates"], bins=bins, labels=labels)
    return (h.groupby(["dataset", "model", "k", "cand_bin"], observed=True)["correct"]
            .agg(["mean", "size"]).reset_index().rename(columns={"mean": "acc", "size": "n"}))


def compliance_table(df: pd.DataFrame) -> pd.DataFrame:
    h = df[df["version"] != "v1"]
    rows = []
    for (ds, m, v, k), g in h.groupby(["dataset", "model", "version", "k"]):
        answered = g[g["pred_norm"] != ""]
        row = dict(dataset=ds, model=m, version=v, k=k, n=len(g), answered=len(answered),
                   len_ok=answered["len_ok"].mean() if len(answered) else np.nan)
        if v in ("v3", "v4"):
            row["prefix_ok"] = answered["prefix_ok"].mean()
            row["both_ok"] = (answered["len_ok"] & answered["prefix_ok"]).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def error_table(df: pd.DataFrame) -> pd.DataFrame:
    w = df[~df["correct"]]
    t = (w.groupby(["dataset", "model", "version", "k"])["error_type"].value_counts()
         .unstack(fill_value=0).reindex(columns=ERROR_TYPES, fill_value=0).reset_index())
    tot = df.groupby(["dataset", "model", "version", "k"]).size().rename("n").reset_index()
    return t.merge(tot, on=["dataset", "model", "version", "k"])


def progression_table(df: pd.DataFrame) -> pd.DataFrame:
    """Along the v2 -> v3 k=1..4 ladder: the first k at which each item is solved,
    and how often a solved item is lost again at a larger k."""
    ladder = df[(df["version"] == "v2") | (df["version"] == "v3")]
    rows = []
    for (ds, m), g in ladder.groupby(["dataset", "model"]):
        piv = g.pivot_table(index="item_id", columns="k", values="correct", aggfunc="first")
        if set(piv.columns) != {0, 1, 2, 3, 4}:
            continue
        piv = piv[[0, 1, 2, 3, 4]].astype(bool)
        arr = piv.to_numpy()
        first = np.where(arr.any(axis=1), arr.argmax(axis=1), -1)
        # Lost later: solved at some k, wrong at a larger k.
        lost = np.array([any(r[i] and not r[j] for i in range(5) for j in range(i + 1, 5)) for r in arr])
        row = dict(dataset=ds, model=m, n=len(piv), never=int((first == -1).sum()),
                   non_monotone=int(lost.sum()))
        for k in range(5):
            row[f"first_k{k}"] = int((first == k).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def agreement_tables(df: pd.DataFrame) -> tuple:
    """How many models solve each item, per condition, plus exclusive solves.

    Only conditions that every model has are counted (base models have no v4/v5).
    """
    hist_rows, excl_rows = [], []
    for (ds, v, k), g in df.groupby(["dataset", "version", "k"]):
        piv = g.pivot_table(index="item_id", columns="model", values="correct", aggfunc="first")
        if piv.shape[1] != len(MODEL_ORDER):
            continue
        piv = piv[MODEL_ORDER].astype(bool)
        counts = piv.sum(axis=1).value_counts().reindex(range(len(MODEL_ORDER) + 1), fill_value=0)
        hist_rows.append(dict(dataset=ds, version=v, k=k, n_models=len(MODEL_ORDER),
                              **{f"solved_by_{i}": int(counts[i]) for i in range(len(MODEL_ORDER) + 1)}))
        only = piv[piv.sum(axis=1) == 1]
        excl_rows.append(dict(dataset=ds, version=v, k=k,
                              **{m: int(only[m].sum()) for m in MODEL_ORDER}))
    return pd.DataFrame(hist_rows), pd.DataFrame(excl_rows)


def pairwise_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ds, v, k), g in df.groupby(["dataset", "version", "k"]):
        piv = g.pivot_table(index="item_id", columns="model", values="correct", aggfunc="first")
        for a, b in itertools.combinations([m for m in MODEL_ORDER if m in piv.columns], 2):
            ca, cb = piv[a].astype(bool).to_numpy(), piv[b].astype(bool).to_numpy()
            b_only, a_only, p = mcnemar(ca, cb)
            rows.append(dict(dataset=ds, version=v, k=k, model_a=a, model_b=b,
                             acc_a=ca.mean(), acc_b=cb.mean(), a_only=a_only, b_only=b_only, p=p))
    return pd.DataFrame(rows)


def cot_table(df: pd.DataFrame) -> pd.DataFrame:
    """v4 (CoT) vs v3 at each k, paired on items."""
    rows = []
    for (ds, m, k), g in df[df["version"].isin(["v3", "v4"])].groupby(["dataset", "model", "k"]):
        piv = g.pivot_table(index="item_id", columns="version", values="correct", aggfunc="first")
        if set(piv.columns) != {"v3", "v4"}:
            continue
        v3, v4 = piv["v3"].astype(bool).to_numpy(), piv["v4"].astype(bool).to_numpy()
        _, v4_only_rev, p = mcnemar(v4, v3)
        rows.append(dict(dataset=ds, model=m, k=k, acc_v3=v3.mean(), acc_v4=v4.mean(),
                         delta=v4.mean() - v3.mean(), v4_only=int((v4 & ~v3).sum()),
                         v3_only=int((v3 & ~v4).sum()), p=p))
    return pd.DataFrame(rows)


def translation_table(df: pd.DataFrame) -> pd.DataFrame:
    """v5 (reason in English, answer in Romanian) against v2 (same length hint).

    Out-of-vocabulary = the answer is not an answer anywhere in either full
    corpus. It over-counts (inflected forms), so only the v5-vs-v2 difference on
    the same items is meaningful: extra OOV answers under v5 are words that did
    not survive the translation back into Romanian.
    """
    rows = []
    for (ds, m), g in df[df["version"].isin(["v2", "v5"])].groupby(["dataset", "model"]):
        piv = g.pivot_table(index="item_id", columns="version", values="correct", aggfunc="first")
        if not {"v2", "v5"} <= set(piv.columns):
            continue
        g2, g5 = g[g["version"] == "v2"], g[g["version"] == "v5"]
        a2, a5 = g2[g2["pred_norm"] != ""], g5[g5["pred_norm"] != ""]
        rows.append(dict(dataset=ds, model=m, acc_v2=piv["v2"].mean(), acc_v5=piv["v5"].mean(),
                         empty_v2=(g2["pred_norm"] == "").mean(), empty_v5=(g5["pred_norm"] == "").mean(),
                         oov_v2=(~a2["pred_in_lexicon"]).mean(), oov_v5=(~a5["pred_in_lexicon"]).mean(),
                         len_ok_v2=a2["len_ok"].mean(), len_ok_v5=a5["len_ok"].mean(),
                         diacritic_only=int(g5["diacritic_only"].sum())))
    return pd.DataFrame(rows)


def length_table(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy by answer length, for the heatmaps."""
    d = df[df["version"].isin(["v1", "v2", "v3"])].copy()
    d["length_bin"] = d["length"].clip(upper=10)
    return (d.groupby(["dataset", "model", "version", "k", "length_bin"])["correct"]
            .agg(["mean", "size"]).reset_index().rename(columns={"mean": "acc", "size": "n"}))


def pick_examples(df: pd.DataFrame, per_type: int = 3) -> pd.DataFrame:
    """A few wrong answers per error type from the strongest open model and Claude at v3 k=2."""
    rng = np.random.default_rng(SEED)
    rows = []
    sub = df[(df["version"] == "v3") & (df["k"] == 2) & ~df["correct"]
             & df["model"].isin(["gpt-oss-120B", "Claude Opus 4.8"])]
    for (ds, m, t), g in sub.groupby(["dataset", "model", "error_type"]):
        if t == "missing":
            continue
        take = g.iloc[rng.choice(len(g), size=min(per_type, len(g)), replace=False)]
        rows.append(take[["dataset", "model", "error_type", "clue", "answer", "prediction"]])
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------
# LaTeX fragments
# --------------------------------------------------------------------------

def tex_escape(s: str) -> str:
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


def pct(x: float, d: int = 1) -> str:
    return "--" if pd.isna(x) else f"{100 * x:.{d}f}"


def write_accuracy_tex(acc: pd.DataFrame, path: str, value: str = "acc", bold: bool = True) -> None:
    """Models x conditions, one block per dataset, best per column in bold."""
    conds = CONDITIONS
    head = " & ".join([r"\multicolumn{1}{c}{No}", r"\multicolumn{1}{c}{Len}"]
                      + [f"$k{{=}}{k}$" for k in range(1, 5)] * 2 + [r"\multicolumn{1}{c}{Tr.}"])
    lines = [r"\begin{tabular}{l" + "r" * len(conds) + "}", r"\toprule",
             r" & \multicolumn{2}{c}{} & \multicolumn{4}{c}{v3: length + prefix} & "
             r"\multicolumn{4}{c}{v4: v3 + CoT} & v5 \\",
             r"\cmidrule(lr){4-7}\cmidrule(lr){8-11}",
             "Model & " + head + r" \\", r"\midrule"]
    for ds in DATASETS:
        lines.append(rf"\multicolumn{{{len(conds) + 1}}}{{l}}{{\textit{{{DATASET_LABEL[ds]}}}}} \\")
        a = acc[acc["dataset"] == ds]
        best = {c: a[(a["version"] == c[0]) & (a["k"] == c[1])][value].max() for c in conds}
        for m in MODEL_ORDER:
            cells = []
            for v, k in conds:
                r = a[(a["model"] == m) & (a["version"] == v) & (a["k"] == k)]
                if r.empty:
                    cells.append("--")
                    continue
                x = r[value].iloc[0]
                s = pct(x)
                cells.append(rf"\textbf{{{s}}}" if bold and np.isclose(x, best[(v, k)]) else s)
            lines.append(f"\\quad {m} & " + " & ".join(cells) + r" \\")
        if ds != DATASETS[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


def write_delta_tex(acc: pd.DataFrame, path: str) -> None:
    """Gain from the diacritic-insensitive metric, in points."""
    acc = acc.assign(delta=acc["acc"] - acc["acc_strict"])
    write_accuracy_tex(acc.assign(acc=acc["delta"]), path, value="acc", bold=False)


def write_continuity_tex(runs: pd.DataFrame, acc: pd.DataFrame, path: str) -> None:
    """Old models on PolyCross: full corpus vs the 3000-item sample."""
    r = runs[(runs["dataset"] == "polycross") & runs["full_n"].notna()]
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r" & & \multicolumn{2}{c}{Full corpus ($\sim$26k)} & \multicolumn{2}{c}{Sample (3000)} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"Model & Cond. & strict & no-diac. & strict & no-diac. \\", r"\midrule"]
    groups = []
    for m in MODEL_ORDER:
        rows = []
        for v, k in [("v1", 0), ("v2", 0), ("v3", 2), ("v3", 4), ("v4", 4), ("v5", 0)]:
            x = r[(r["model"] == m) & (r["version"] == v) & (r["k"] == k)]
            if x.empty:
                continue
            x = x.iloc[0]
            rows.append(f"{m} & {cond_label(v, k)} & {pct(x.full_acc_strict)} & {pct(x.full_acc)} & "
                        f"{pct(x.acc_strict)} & {pct(x.acc)} \\\\")
        if rows:
            groups.append(rows)
    for i, rows in enumerate(groups):
        lines += rows
        if i < len(groups) - 1:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


def write_strata_tex(st: pd.DataFrame, path: str) -> None:
    """v3 accuracy split by how many letters the hint leaves unknown, datasets side by side."""
    v3 = st[st["version"] == "v3"]
    n_items = {"polycross": 3000, "themcross": 1148}
    ks = range(1, 5)
    head = " & ".join(rf"$k{{=}}{k}$" for k in ks)
    lines = [r"\begin{tabular}{ll" + "rrrr" * len(DATASETS) + "}", r"\toprule",
             " & & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{{DATASET_LABEL[ds]}}}" for ds in DATASETS) + r" \\",
             "".join(rf"\cmidrule(lr){{{3 + 4 * i}-{6 + 4 * i}}}" for i in range(len(DATASETS))),
             "Items & Model & " + " & ".join([head] * len(DATASETS)) + r" \\", r"\midrule"]
    strata = STRATA + ["informative"]
    for s_ in strata:
        label = STRATUM_LABEL.get(s_, r"$|a|>k$ (informative)")
        shares = []
        for ds in DATASETS:
            for k in ks:
                n = v3[(v3["dataset"] == ds) & (v3["stratum"] == s_) & (v3["k"] == k)]["n"]
                shares.append(r"\textcolor{gray}{" + (pct(n.iloc[0] / n_items[ds], 0) if len(n) else "0") + r"\%}")
        lines.append(rf"\multicolumn{{2}}{{l}}{{\textit{{{label}}}}} & " + " & ".join(shares) + r" \\")
        for m in MODEL_ORDER:
            cells = []
            for ds in DATASETS:
                for k in ks:
                    x = v3[(v3["dataset"] == ds) & (v3["model"] == m) & (v3["stratum"] == s_) & (v3["k"] == k)]
                    cells.append(pct(x["acc"].iloc[0]) if len(x) else "--")
            lines.append(f" & {m} & " + " & ".join(cells) + r" \\")
        if s_ != strata[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


def write_cot_tex(cot: pd.DataFrame, path: str) -> None:
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Dataset & Model & $k{=}1$ & $k{=}2$ & $k{=}3$ & $k{=}4$ \\", r"\midrule"]
    for ds in DATASETS:
        for m in MODEL_ORDER:
            cells = []
            for k in range(1, 5):
                x = cot[(cot["dataset"] == ds) & (cot["model"] == m) & (cot["k"] == k)]
                if x.empty:
                    cells.append("--")
                    continue
                x = x.iloc[0]
                star = "$^{*}$" if x.p < 0.01 else ""
                cells.append(f"{100 * x.delta:+.1f}{star}")
            lines.append(f"{DATASET_LABEL[ds] if m == MODEL_ORDER[0] else ''} & {m} & " + " & ".join(cells) + r" \\")
        if ds != DATASETS[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


def write_translation_tex(tr: pd.DataFrame, path: str) -> None:
    lines = [r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
             r" & & \multicolumn{2}{c}{Acc.} & \multicolumn{2}{c}{No ans.} & "
             r"\multicolumn{2}{c}{OOV} & \multicolumn{2}{c}{Len.\ ok} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
             r"Dataset & Model & v2 & v5 & v2 & v5 & v2 & v5 & v2 & v5 \\", r"\midrule"]
    for ds in DATASETS:
        for m in MODEL_ORDER:
            x = tr[(tr["dataset"] == ds) & (tr["model"] == m)]
            if x.empty:
                continue
            x = x.iloc[0]
            cells = [pct(x[c]) for c in ["acc_v2", "acc_v5", "empty_v2", "empty_v5",
                                         "oov_v2", "oov_v5", "len_ok_v2", "len_ok_v5"]]
            lines.append(f"{DATASET_LABEL[ds] if m == MODEL_ORDER[0] else ''} & {m} & " + " & ".join(cells) + r" \\")
        if ds != DATASETS[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


def write_progression_tex(pr: pd.DataFrame, path: str) -> None:
    lines = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r" & & \multicolumn{5}{c}{First solved at} & & \\",
             r"\cmidrule(lr){3-7}",
             r"Dataset & Model & Len & $k{=}1$ & $k{=}2$ & $k{=}3$ & $k{=}4$ & Never & Lost later \\",
             r"\midrule"]
    for ds in DATASETS:
        for m in MODEL_ORDER:
            x = pr[(pr["dataset"] == ds) & (pr["model"] == m)]
            if x.empty:
                continue
            x = x.iloc[0]
            cells = [pct(x[f"first_k{k}"] / x.n) for k in range(5)] + [pct(x.never / x.n), pct(x.non_monotone / x.n)]
            lines.append(f"{DATASET_LABEL[ds] if m == MODEL_ORDER[0] else ''} & {m} & " + " & ".join(cells) + r" \\")
        if ds != DATASETS[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


def write_compliance_tex(co: pd.DataFrame, path: str) -> None:
    conds = [("v2", 0), ("v3", 1), ("v3", 2), ("v3", 3), ("v3", 4), ("v5", 0)]
    lines = [r"\begin{tabular}{ll" + "r" * len(conds) + "}", r"\toprule",
             "Dataset & Model & " + " & ".join(cond_label(v, k) for v, k in conds) + r" \\", r"\midrule"]
    for ds in DATASETS:
        for m in MODEL_ORDER:
            cells = []
            for v, k in conds:
                x = co[(co["dataset"] == ds) & (co["model"] == m) & (co["version"] == v) & (co["k"] == k)]
                cells.append(pct(x["both_ok" if v == "v3" else "len_ok"].iloc[0]) if len(x) else "--")
            lines.append(f"{DATASET_LABEL[ds] if m == MODEL_ORDER[0] else ''} & {m} & " + " & ".join(cells) + r" \\")
        if ds != DATASETS[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


def write_examples_tex(ex: pd.DataFrame, path: str) -> None:
    lines = [r"\begin{tabular}{p{0.34\linewidth}lll}", r"\toprule",
             r"Clue & Answer & Prediction & Model \\", r"\midrule"]
    for t in ["near_miss", "other_word", "length", "prefix"]:
        g = ex[(ex["error_type"] == t) & (ex["dataset"] == "polycross")]
        if g.empty:
            continue
        lines.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{ERROR_LABEL[t]}}}}} \\")
        for r in g.itertuples():
            lines.append(f"{tex_escape(r.clue)} & {tex_escape(r.answer.upper())} & "
                         f"{tex_escape(str(r.prediction).upper())} & {r.model} \\\\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------

def main(analysis_dir: str, tables_dir: str) -> None:
    os.makedirs(tables_dir, exist_ok=True)
    df = pd.read_csv(os.path.join(analysis_dir, "predictions.csv"),
                     keep_default_na=False, na_values=[""],
                     dtype={"prediction": object, "pred_norm": object, "rationale": object})
    runs = pd.read_csv(os.path.join(analysis_dir, "runs.csv"))
    # Every table iterates MODEL_ORDER; keep only models that have results, so a
    # model not yet run (e.g. a base model awaiting its GPU job) is left out
    # instead of showing up as a row of dashes.
    global MODEL_ORDER
    MODEL_ORDER = [m for m in MODEL_ORDER if m in set(df["model"])]

    lexicon = build_lexicon()
    df = add_features(df, candidate_counts(lexicon), lexicon)
    df.to_csv(os.path.join(analysis_dir, "predictions_features.csv"), index=False)

    out = {
        "accuracy": accuracy_table(df),
        "strata": strata_table(df),
        "candidates": candidates_table(df),
        "compliance": compliance_table(df),
        "errors": error_table(df),
        "progression": progression_table(df),
        "cot": cot_table(df),
        "translation": translation_table(df),
        "by_length": length_table(df),
        "pairwise": pairwise_tests(df),
        "examples": pick_examples(df),
    }
    out["agreement"], out["exclusive"] = agreement_tables(df)
    for name, t in out.items():
        t.to_csv(os.path.join(analysis_dir, f"{name}.csv"), index=False)

    write_accuracy_tex(out["accuracy"], os.path.join(tables_dir, "accuracy.tex"))
    write_accuracy_tex(out["accuracy"], os.path.join(tables_dir, "accuracy_strict.tex"), value="acc_strict")
    write_delta_tex(out["accuracy"], os.path.join(tables_dir, "accuracy_delta.tex"))
    write_continuity_tex(runs, out["accuracy"], os.path.join(tables_dir, "continuity.tex"))
    write_strata_tex(out["strata"], os.path.join(tables_dir, "strata.tex"))
    write_cot_tex(out["cot"], os.path.join(tables_dir, "cot.tex"))
    write_translation_tex(out["translation"], os.path.join(tables_dir, "translation.tex"))
    write_progression_tex(out["progression"], os.path.join(tables_dir, "progression.tex"))
    write_compliance_tex(out["compliance"], os.path.join(tables_dir, "compliance.tex"))
    write_examples_tex(out["examples"], os.path.join(tables_dir, "examples.tex"))
    print(f"Wrote {len(out)} tables to {analysis_dir} and LaTeX to {tables_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_dir", default="data/results/analysis")
    parser.add_argument("--tables_dir", default="paper/tables")
    args = parser.parse_args()
    main(args.analysis_dir, args.tables_dir)
