# Solve the Romanian crossword clues with a *base* (non-instruction-tuned) LLM
# run in-process through Hugging Face transformers -- e.g. faur-ai/LLMic_v2.
#
# clues_ro.py cannot be used for such models: every Mellea backend wraps the
# prompt in a chat template (base models have none) and the instruction prompts
# ask for JSON, which a base model does not produce. Here each condition is a
# few-shot completion prompt (prompts_base.py), decoded greedily until the first
# newline.
#
# Output is the same list-of-records-plus-summary format as clues_ro.py, with
# the same filename scheme, so rescore.py / error_analysis.py pick it up as-is.
#
# Example (one condition; run_llmic.sh sweeps all of them):
#   PYTHONPATH=src python src/crosswords/solve_base_ro.py --model_id llmic-v2 \
#     --dataset_file data/polycross_sample3000.json --dataset_type clues \
#     --version v3 --prefix_len 2 --output_name clues --output_dir data/results

import argparse
import json
import os
import time
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

from crosswords.analysis.rescore import normalize
from crosswords.backends import CONFIG_DIR, load_json_utf8_relaxed, load_model_config
from crosswords.prompts_base import BASE_VERSIONS, build_prompt, parse_completion
from crosswords.runner import filter_and_limit, resolve_keys

HF_CONFIG = CONFIG_DIR / "hf_models.json"


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def resolve_model(model_id: str) -> dict[str, Any]:
    """The config record for a --model_id, or a bare record for a raw HF id / local path."""
    models = load_model_config(HF_CONFIG)
    if model_id in models:
        return models[model_id]
    if "/" in model_id or os.path.isdir(model_id):
        return {"name": model_id, "model_id": model_id}
    raise SystemExit(f"Unknown --model_id '{model_id}'. Known: {', '.join(models) or '(none)'}; "
                     "or pass a Hugging Face id / local directory.")


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str, requested: str) -> torch.dtype:
    if requested != "auto":
        return getattr(torch, requested)
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float16 if device == "mps" else torch.float32


class BaseModelSolver:
    """Batched greedy completion with a causal LM."""

    def __init__(self, record: dict[str, Any], model_path: str | None, device: str,
                 dtype: torch.dtype, max_new_tokens: int, attn_implementation: str = "auto"):
        source = model_path or record["model_id"]
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_context = int(record.get("max_context", 2048))

        # transformers 5 rebuilds some tokenizers from their class defaults and
        # drops the pre-tokenizer in tokenizer.json. For LLMic_v2 that turns every
        # "\n" into the wrong token, and the model then emits garbage; the
        # config record opts into loading tokenizer.json verbatim.
        tokenizer_cls = PreTrainedTokenizerFast if record.get("tokenizer_from_file") else AutoTokenizer
        self.tokenizer = tokenizer_cls.from_pretrained(source)
        # Batched generation must pad on the left so every prompt ends at the
        # same position; truncate (if ever needed) from the left for the same reason.
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        # The Hub config of LLMic_v2 points eos at <|unk|>; the config record
        # carries the ids that actually work.
        self.eos_token_id = record.get("eos_token_id", self.tokenizer.eos_token_id)
        self.pad_token_id = record.get("pad_token_id", self.tokenizer.pad_token_id)
        if self.pad_token_id is None:
            self.pad_token_id = self.eos_token_id
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.pad_token_id

        # SDPA on MPS mishandles the left-padding mask: padded rows emit EOS at
        # once. Eager attention gives the same outputs as CPU, so use it there.
        if attn_implementation == "auto":
            attn_implementation = "eager" if device == "mps" else "sdpa"
        self.model = AutoModelForCausalLM.from_pretrained(
            source, dtype=dtype, attn_implementation=attn_implementation).to(device)
        self.model.eval()

    @torch.inference_mode()
    def _generate(self, prompts: list[str]) -> list[str]:
        # add_special_tokens=False: the LLMic tokenizer has add_eos_token=True,
        # which would otherwise append <|endoftext|> to the end of each prompt.
        enc = self.tokenizer(prompts, add_special_tokens=False, padding=True, truncation=True,
                             max_length=self.max_context - self.max_new_tokens,
                             return_tensors="pt").to(self.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            stop_strings=["\n"],
            tokenizer=self.tokenizer,
        )
        new = out[:, enc["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(new, skip_special_tokens=True)

    def generate(self, prompts: list[str]) -> list[str]:
        """Generate for a batch; on out-of-memory fall back to one prompt at a time."""
        try:
            return self._generate(prompts)
        except torch.OutOfMemoryError:
            if len(prompts) == 1:
                raise
            torch.cuda.empty_cache()
            return [self._generate([p])[0] for p in prompts]

    def check_batching(self, prompts: list[str]) -> None:
        """Fail fast if batched (padded) generation is broken.

        Guards against backend bugs in padded attention (e.g. SDPA on MPS), which
        otherwise show up only as a run full of empty answers. An occasional
        different answer is expected in half precision -- padding changes the
        matmul shapes, which can flip a close greedy choice -- so only empty
        batched answers or disagreement on most prompts stop the run.
        """
        batched = [parse_completion(c) for c in self._generate(prompts)]
        single = [parse_completion(self._generate([p])[0]) for p in prompts]
        diff = [(b, s) for b, s in zip(batched, single) if b != s]
        emptied = [(b, s) for b, s in diff if not b and s]
        if emptied or len(diff) > len(prompts) // 2:
            raise SystemExit(f"Batched generation looks broken: {len(diff)}/{len(prompts)} prompts differ "
                             f"from single-prompt generation, {len(emptied)} came back empty "
                             f"(e.g. {diff[:3]}). Try --attn_implementation eager, --dtype float32 "
                             "or --batch_size 1.")
        note = f"; {len(diff)} differ, as half precision allows: {diff}" if diff else ""
        print(f"Batching self-check passed on {len(prompts)} prompts{note}")

    def count_tokens(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt, add_special_tokens=False))


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

def item_key(clue: str, answer: str) -> tuple[str, str]:
    return clue.strip(), answer.strip().upper()


def load_records(path: str) -> list[dict[str, Any]]:
    """Records from an output file, without the trailing summary dict."""
    if not os.path.exists(path):
        return []
    return [r for r in load_json_utf8_relaxed(path) if isinstance(r, dict) and "clue" in r]


def write_results(records: list[dict[str, Any]], path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)
    os.replace(tmp, path)


def eval_results(path: str, bertscore: bool = True) -> dict[str, Any]:
    """Score an output file and append the summary as its final element.

    Same summary keys as clues_ro.py, plus the diacritic-insensitive accuracy
    used in the error analysis. Re-running replaces an existing summary.
    """
    records = load_records(path)
    assert records, f"No results to evaluate in {path}."
    refs = [r["answer"] for r in records]
    preds = [r.get("prediction") or "" for r in records]

    exact = [float(a.strip().lower() == p.strip().lower()) for a, p in zip(refs, preds)]
    nodiac = [float(normalize(a) == normalize(p)) for a, p in zip(refs, preds)]
    summary: dict[str, Any] = {
        "num_samples": len(records),
        "exact_matches": float(np.sum(exact)),
        "accuracy": float(np.mean(exact)),
        "accuracy_nodiac": float(np.mean(nodiac)),
        "empty_predictions": int(sum(not p.strip() for p in preds)),
        "avg_answer_length": float(np.mean([len(a) for a in refs])),
        "avg_prediction_length": float(np.mean([len(p) for p in preds])),
    }
    if bertscore:
        from bert_score import BERTScorer
        scorer = BERTScorer(model_type="bert-base-uncased", device="cpu")
        # bert-score encodes an empty prediction via build_inputs_with_special_tokens,
        # which transformers 5 removed; restore its BERT behaviour ([CLS] ids [SEP]).
        tok = scorer._tokenizer
        if not hasattr(tok, "build_inputs_with_special_tokens"):
            tok.build_inputs_with_special_tokens = (
                lambda ids: [tok.cls_token_id] + list(ids) + [tok.sep_token_id])
        _, _, f1 = scorer.score(preds, refs)
        summary["sbert_mean"] = float(f1.mean())
        summary["sbert_std"] = float(f1.std(unbiased=False))

    print(f"Evaluation results: {summary}")
    write_results(records + [summary], path)
    return summary


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def process_data(data: list[dict[str, Any]], solver: BaseModelSolver, version: str,
                 prefix_len: int, dataset_type: str, output_filename: str,
                 num_samples: int | None, batch_size: int, save_every: int,
                 resume: bool, min_answer_len: int) -> list[dict[str, Any]]:
    answer_key, clue_key = resolve_keys(data, dataset_type)
    print(f"Answer field: '{answer_key}'")
    data = filter_and_limit(data, answer_key, num_samples=num_samples, min_answer_len=min_answer_len)

    done: dict[tuple[str, str], dict[str, Any]] = {}
    if resume:
        for r in load_records(output_filename):
            done.setdefault(item_key(r["clue"], r["answer"]), r)
        print(f"Resuming: {len(done)} items already in {output_filename}")

    todo = [i for i, it in enumerate(data) if item_key(it[clue_key], it[answer_key]) not in done]
    prompts = {i: build_prompt(version, data[i][clue_key], data[i][answer_key], prefix_len) for i in todo}
    if todo:
        lengths = [solver.count_tokens(prompts[i]) for i in todo]
        print(f"Prompt tokens: mean {np.mean(lengths):.0f}, max {max(lengths)} "
              f"(context {solver.max_context})")
        print("Example prompt:\n" + prompts[todo[0]] + "\n" + "-" * 60)
    # Longest-first ordering keeps padding low and surfaces OOM early.
    todo.sort(key=lambda i: -len(prompts[i]))
    if len(todo) > 1 and batch_size > 1:
        # Mix the longest and shortest prompts so the check exercises heavy padding.
        probe = todo[:3] + todo[-3:] if len(todo) >= 6 else todo
        solver.check_batching([prompts[i] for i in probe])

    fresh: dict[int, dict[str, Any]] = {}

    def assemble() -> list[dict[str, Any]]:
        out = []
        for i, it in enumerate(data):
            key = item_key(it[clue_key], it[answer_key])
            rec = fresh.get(i) or done.get(key)
            if rec is not None:
                out.append(rec)
        return out

    t0 = time.time()
    n_batches = (len(todo) + batch_size - 1) // batch_size
    for b in range(n_batches):
        idx = todo[b * batch_size:(b + 1) * batch_size]
        completions = solver.generate([prompts[i] for i in idx])
        for i, raw in zip(idx, completions):
            answer = data[i][answer_key]
            fresh[i] = {
                "clue": data[i][clue_key],
                "answer": answer,
                "prediction": parse_completion(raw),
                "rationale": "",
                "raw_completion": raw,
            }
        n = min((b + 1) * batch_size, len(todo))
        acc = np.mean([normalize(r["answer"]) == normalize(r["prediction"]) for r in fresh.values()])
        print(f"  batch {b + 1}/{n_batches}: {n}/{len(todo)} items, "
              f"{n / (time.time() - t0):.1f} items/s, running acc (no diacritics) {acc:.3f}", flush=True)
        if (b + 1) % save_every == 0:
            write_results(assemble(), output_filename)

    results = assemble()
    write_results(results, output_filename)
    print(f"Wrote {len(results)} records to {output_filename}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", default="llmic-v2",
                        help=f"A name from {HF_CONFIG}, a Hugging Face id, or a local directory.")
    parser.add_argument("--model_path", default=None,
                        help="Load weights from this directory instead of the Hub (e.g. a pre-downloaded copy).")
    parser.add_argument("--dataset_file")
    parser.add_argument("--dataset_type", default="clues",
                        choices=["clues", "polycross", "roco", "baseline", "themcross", "base"])
    parser.add_argument("--version", default="v3", choices=BASE_VERSIONS)
    parser.add_argument("--prefix_len", type=int, default=0)
    parser.add_argument("--output_name", default=None,
                        help="Filename prefix. Defaults to 'clues' for polycross and 'baseline' for "
                             "themcross, matching the existing result files.")
    parser.add_argument("--output_dir", default="data/results")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--min_answer_len", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32, help="Prompts per generate() call.")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:1, mps or cpu")
    parser.add_argument("--dtype", default="auto", help="auto, bfloat16, float16 or float32")
    parser.add_argument("--attn_implementation", default="auto",
                        help="auto (eager on MPS, sdpa elsewhere), eager, sdpa or flash_attention_2")
    parser.add_argument("--save_every", type=int, default=10, help="Checkpoint every N batches.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip items already present in the output file.")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--no_bertscore", action="store_true", help="Skip BERTScore in the summary.")
    parser.add_argument("--list_models", action="store_true")
    args = parser.parse_args()

    if args.list_models:
        print(f"Base models via transformers ({HF_CONFIG}):")
        for name, rec in load_model_config(HF_CONFIG).items():
            print(f"  {name:24s} {rec['model_id']}")
        return
    if not args.dataset_file and not args.eval_only:
        parser.error("--dataset_file is required")
    if args.version != "v3" and args.prefix_len:
        parser.error(f"--prefix_len only applies to v3 (got {args.version})")

    if args.output_name is None:
        args.output_name = "baseline" if args.dataset_type in ("baseline", "themcross", "base") else "clues"
    safe_id = args.model_id.replace("/", "_")
    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = os.path.join(
        args.output_dir, f"{args.output_name}_{safe_id}_{args.version}_{args.prefix_len}.json")

    if not args.eval_only:
        record = resolve_model(args.model_id)
        device = pick_device(args.device)
        dtype = pick_dtype(device, args.dtype)
        print(f"Model: {record['model_id']} on {device} ({dtype}); version {args.version}, "
              f"prefix_len {args.prefix_len}; output {output_filename}")
        solver = BaseModelSolver(record, args.model_path, device, dtype, args.max_new_tokens,
                                 args.attn_implementation)
        process_data(load_json_utf8_relaxed(args.dataset_file), solver, args.version,
                     args.prefix_len, args.dataset_type, output_filename, args.num_samples,
                     args.batch_size, args.save_every, args.resume, args.min_answer_len)

    eval_results(output_filename, bertscore=not args.no_bertscore)
    print("Done.")


if __name__ == "__main__":
    main()
