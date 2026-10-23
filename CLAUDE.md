# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A research benchmark for evaluating LLMs on Romanian crossword puzzle clues. The premise: published crossword clues are deliberately polysemantic — the obvious reading is usually wrong, and the correct answer requires finding a subtler meaning. The project runs several LLMs over these clues, scores their predictions, and compares against a baseline of straightforward general-knowledge clues.

Two datasets, both lists of clue/answer dicts in `data/`:
- `data/extracted_data.json` — the **main benchmark** (the proposed novel contribution). Keys: `answer`, `clue`, `filePath`. Note: `answer` values mix case intentionally (e.g. `pRAF`, `IaTAC`) — lowercase letters mark intersection cells from the original grid. Comparison is always done case-insensitively.
- `data/baseline-dataset.json` — straightforward thematic clues for comparison. Keys: `solution`, `clue`, `path`. **The answer field is `solution` here, not `answer`.**

The two key fields differ between datasets; `process_data` in `clues_ro.py` selects `answer_key`/`clue_key` based on the `--dataset_type` argument (`clues` vs `baseline`).

## Running experiments

The main entry point is `src/crosswords/clues_ro.py` (the solving task). The reverse task — generating clues from answers — lives in its own script, `src/crosswords/gen_clues_ro.py` (see "Clue generation" below). Everything else orchestrates or post-processes these.

The bash runners (`run_clues.sh`, `run_baseline.sh`, `run_*_single.sh`) hardcode absolute paths under `/home/radu/storage/git/crosswords/...` and call a local `./timeout` binary — **they are tuned for the author's machine and will not run as-is here.** To run locally, invoke the Python script directly:

```bash
python src/crosswords/clues_ro.py \
  --model_id gpt-oss \
  --dataset_file data/extracted_data.json \
  --dataset_type clues \
  --version v3 \
  --prefix_len 2 \
  --output_name clues \
  --output_dir data/results
```

Output filename is constructed as `{output_name}_{model_id}_{version}_{prefix_len}.json` inside `--output_dir`.

Key arguments:
- `--model_id`: `llama`, `granite`, `mistral`, or `gpt-oss`. These map to specific RITS model enums in the `__main__` block.
- `--version`: prompt version `v1`–`v6` (see Prompt versions below).
- `--prefix_len`: how many leading letters of the answer to reveal as a hint (the `s` / "seed" in the runner scripts). Used by `v3`/`v4`.
- `--num_samples`: cap the number of clues processed.
- `--batch_size`: max concurrent async requests (semaphore size, default 50).
- `--rate_limit`: requests/minute throttle (default 1500).
- `--eval_only`: skip generation, just re-score an existing output file.

### Prompt versions (the core experimental variable)

Each version is a full prompt template (`INSTRUCTION_V*` constants) representing a different amount of hinting / reasoning strategy:
- `v1`: clue only.
- `v2`: clue + answer length.
- `v3`: clue + length + first `prefix_len` letters.
- `v4` (`_COT`): v3 hints with chain-of-thought; answer wrapped in `[brackets]`, reasoning in `<think>` tags.
- `v5` (`_COT`): length hint only, reasons in English then translates back to Romanian; bracketed answer.
- `v6`: **reverse task** — given an answer, generate a polysemantic clue (JSON output).

Output parsing depends on the version: `v1`–`v3` parse a JSON code block (`strip_code_fences` + `json.loads`); `v4`/`v5` extract `[...]` via `extract_last_square_brackets` and `<think>` via `get_think_tags`; `v6` takes the raw string. When adding a version, update the dispatch in both `acall_one` (prompt + variables + requirements) and the result-parsing loop in `process_data`.

## Backend

LLM calls go through the [`mellea`](https://pypi.org/project/mellea/) library and IBM's internal `mellea_ibm.rits` (RITS) backend — **`mellea_ibm` is an IBM-internal package, not on PyPI.** `pyproject.toml` installs it from `git+ssh://git@github.ibm.com/generative-computing/mellea-ibm.git`, pinned to a commit because the repo has no tags. That needs `[tool.hatch.metadata] allow-direct-references = true` and an SSH key registered on github.ibm.com, so `pip install -e .` fails without IBM GitHub access. Requests use `RejectionSamplingStrategy(loop_budget=5)` with per-version validation requirements. Generation is async via `mfuncs.ainstruct`, throttled by a semaphore + sleep interval. Expect RITS to be unavailable outside IBM infrastructure.

`load_dotenv()` is called at import, so credentials/config are expected via a `.env` file (not committed).

### Model configs (`configs/`)

All three generation scripts (`clues_ro.py`, `gen_clues_ro.py`, `gen_rationales_ro.py`) source their
models from JSON config files instead of hardcoded tables, so models can be added or retired by editing
config rather than code. The shared factory lives in **`src/crosswords/backends.py`**
(`build_backend`, `print_models`, `available_models`) — scripts import it rather than constructing
backends themselves. Both config files hold a list of `{"name", "model_id", "backend", ...}` records;
`name` is what you pass to `--model_id`, and every script supports `--list_models` and `--max_tokens`.

- `configs/rits_models.json` — RITS models. Each record needs an explicit `base_url` (the bare
  endpoint; `RITSBackend` appends `/v1` itself). Needs `RITS_API_KEY` in `.env`.
- `configs/local_models.json` — local models run **in-process** via Mellea's `LocalVLLMBackend`
  (`"backend": "vllm"`). **Mellea 0.8.0 (the pinned version) removed `LocalVLLMBackend`**, so this path
  currently errors out with a clear message; it only works with an older Mellea. When it did work it
  needed `pip install vllm outlines` **and a CUDA GPU** — the weights are loaded
  inside the script, so this cannot run on a machine without one. Per-model vLLM engine knobs go under
  `model_options` (e.g. `gpu_memory_utilization`, `max_model_len`). `VLLM_USE_V1=0` is set automatically
  because the Mellea backend does not support vLLM V1.
- `configs/frontier_models.json` — models on the IBM litellm gateway. `model_id` is prefixed
  `openai/` so litellm uses the OpenAI-compatible route the gateway exposes. Needs
  `ANTHROPIC_BASE_URL` + `ANTHROPIC_AUTH_TOKEN` in `.env`. Gateway ids are not always the obvious
  name (Claude Opus 5.5 is `aws/claude-opus-5-5`); list them with `GET {ANTHROPIC_BASE_URL}/v1/models`.

A `--model_id` containing a `/` that matches neither config is passed through verbatim as a raw
litellm model id. A missing config file is not fatal — that family is just unavailable.

`backends.py` is the only place that touches `mellea_ibm`, and the import is guarded, so the litellm
path works even where RITS is unavailable. Since mellea 0.8.0 LiteLLM is an optional extra:
`pyproject.toml` depends on `mellea[litellm]`, and a bare `pip install mellea` makes every script fail
at import with `requires extra dependencies ... mellea[litellm]`.

In 0.8.0 `Backend.generate_from_context` is a `@final` base-class wrapper around each backend's
`_generate_from_context`. `throttle_backend` still patches the public method on the instance, which is
what the sampler and `ainstruct` call, so retries are still rate-limited. The old hardcoded `RITS.*` enum aliases (`llama`, `granite`,
`mistral`, `gpt-oss`) are gone; note that `RITS.GPT_OSS_120B` and `RITS.MISTRAL_LARGE_3_675B_2512` now
return **HTTP 404** (retired deployments), which is why the config points at live endpoints instead —
`gpt-oss-120b-a100` replaces the dead gpt-oss. All five models in `configs/rits_models.json` were
verified working. The reasoning models (`deepseek-v3.2`, `granite-4-2-30b`, `qwen-3-6-35b`) emit a
separate `reasoning` field and need a generous `--max_tokens`.

Because `--model_id` values changed, output filenames change too (e.g. `clues_llama_v3_2.json` becomes
`clues_llama-3.3-70b-instruct_v3_2.json`); older committed result files keep their original names.

### Throttling and the shared async driver (`runner.py`)

`src/crosswords/runner.py` holds the one async driver all three scripts use: `RateLimiter`,
`throttle_backend`, `run_throttled`, plus `resolve_keys` / `filter_and_limit`.

**`--rate_limit` is enforced on every real LLM call, retries included.** `throttle_backend` wraps
`backend.generate_from_context`, which is the single method every request passes through — including the
retries `RejectionSamplingStrategy` issues inside its `loop_budget` loop. This matters: the previous
approach spaced *item submissions*, so one item with `loop_budget=5` counted as 1 call while making up to
5, letting real traffic reach ~5x the budget. Use `--loop_budget` to trade retry attempts against quota.

`run_throttled` wraps each item in try/except and returns `(result, item, error)` triples in input order,
so one bad item cannot abort a 30k-item run.

`resolve_keys` auto-detects whether records use `answer` (polycross/roco) or `solution`
(themcross/base), so `--dataset_type` only needs to be right for output naming; `clues`/`polycross`/`roco`
and `baseline`/`themcross`/`base` are accepted aliases. `filter_and_limit` drops short answers **before** truncating, so `--num_samples N`
returns exactly N items (the old order silently returned fewer — 82 instead of 100 on polycross).

## Evaluation

`eval_results()` runs automatically after generation (and standalone with `--eval_only`). It computes exact-match accuracy (case-insensitive) and BERTScore F1 (`bert-base-uncased`, CPU) between `answer` and `prediction`, then **appends a summary dict as the final element of the output JSON array**. Be aware: result files therefore end with a stats object, not a clue record — downstream readers must handle this.

## Clue generation (`gen_clues_ro.py`)

A separate, standalone script for the **reverse task**: given an answer, generate short polysemantic Romanian clues. This is what `clues_ro.py` v6 does, extracted into its own file so it can produce *multiple* candidates and score them — it does not share `clues_ro.py`'s code.

```bash
python src/crosswords/gen_clues_ro.py \
  --model_id gpt-oss \
  --dataset_file data/extracted_data.json \
  --dataset_type clues \
  --num_candidates 3 \
  --num_hints 2 \
  --num_samples 100 \
  --output_name gen_clues \
  --output_dir data/results
```

- Output filename: `{output_name}_{dataset_type}_{model_id}_h{num_hints}.json` (the hint level and dataset type are encoded so different sweeps don't collide).
- `--num_candidates` (default 3): clues generated per answer, in a single LLM call returning `{"answer", "clues": [...]}`.
- `--num_hints` (default 0): how many **content words** of the **ground-truth clue** to feed back as a hint. `0` uses `INSTRUCTION_GEN` (no hint); `≥1` uses `INSTRUCTION_GEN_HINTS` with the first N content words (via `get_hint_words`, which skips Romanian stop words in `RO_STOP_WORDS`). If a clue is all stop words the hint is empty and the script falls back to the plain prompt for that item.
- `--eval_only`: re-score an existing output file (reconstructs the same filename, so pass the same `--dataset_type`/`--model_id`/`--num_hints`).
- Throttling (`--batch_size`, `--rate_limit`), the async pattern, and the `<think>`-before-JSON parsing are the same as `clues_ro.py`. Note: parsing uses `parse_clues_response` / `validate_clues_response`, which extract the fenced block from *anywhere* in the response (`extract_first_code_block`) — unlike `clues_ro.py`, which only strips a leading fence and would mis-handle a `<think>` preamble before JSON.

Evaluation here is **different from the solving task**: `eval_results()` scores each generated clue against the ground-truth `source_clue` via BERTScore F1, records the best candidate per answer (`best_clue`/`best_sbert`), and appends a `{num_samples, num_scored, sbert_mean, sbert_std}` summary as the final array element. The full clue is scored as-is even when hint words were given (hint leakage is intentionally not removed).

### Few-shot prompt from real clues (`--prompt icl`)

`--prompt icl` uses `INSTRUCTION_GEN_ICL`, whose examples are **real published clues** rather than the hand-written OCUPAT/ROASA/SAC ones, so the model copies their brevity and kind of wordplay. Both the examples and the answers to generate for come from one frozen sample file:

```bash
PYTHONPATH=src python -m crosswords.select_icl_sample      # -> data/gen_clues/polycross_icl_sample.json
python src/crosswords/gen_clues_ro.py --model_id claude-opus-5-5 --prompt icl \
  --dataset_type clues --output_name gen_clues_polycross --output_dir data/results/gen_clues
```

- `select_icl_sample.py` draws, with a fixed seed, 8 `icl` examples and 20 `eval` items that share no answer, so a model is never shown the clue it is asked about. "Non-trivial": answer ≥5 letters, letters only, and occurring ≤3 times in the corpus (drops the reused fillers like `ERA`/`ATA`). Clues must be 2-6 words, not an all-caps theme heading, and must not contain the answer's first 5 letters. `EXCLUDED_ANSWERS` holds hand-rejected examples; they are skipped **after** the shuffle, so adding one does not reshuffle the rest of the sample.
- With `--prompt icl`, `--dataset_file` is ignored (the sample file is the dataset), `--num_hints` is rejected, and the output is `{output_name}_{dataset_type}_{model_id}_icl.json`. Each record also keeps `filePath`; `source_clue` is the ground truth.
- Clues are capped at `--max_clue_words` (default 6, the corpus 90th percentile; median is 4). The cap, the clue count and "no answer leakage" are **enforced**: an extra requirement fails responses that break them, and this path uses `RepairTemplateStrategy` instead of rejection sampling. That matters, because plain rejection resends the identical prompt, and Opus 5.5 returned the same 7-word clue for TITULARA on all 5 attempts. The repair strategy feeds the failure reason (`over 6 words: '...'`) back to the model, and it fixed the clue on the first retry.
- The summary also reports `clue_words_mean` and `within_word_limit`.
- Results for `claude-opus-5-5` and `gpt-6-astra` are in `data/results/gen_clues/`: 20/20 items each, all clues within 6 words, no leakage. BERTScore against a single reference clue says little about wordplay quality, so read the clues.
- mellea 0.8.0 gotcha when probing a model by hand: `ainstruct(..., strategy=None)` returns an **uncomputed** thunk, so `str(out)` is `''`. Pass `await_result=True`. The scripts are unaffected because a sampling strategy always awaits.

Runner: `run_gen_clues.sh <model_id> <dataset_type> [num_candidates] [num_samples]` sweeps `--num_hints` over `0 1 2`. Like the other runners it hardcodes `/home/radu/...` paths and uses `./timeout`, so it is author-machine-specific.

## User-study rationale experiment

`user-study/` holds two curated 100-record samples for human annotation:
`base-sample-100.json` (keys `solution`/`clue`/`path`) and `roco-sample-100.json`
(keys `answer`/`clue`/`filePath`), plus `instructiuni.md` (participant instructions) and
`select_random.py` (the sampler that produced them).

Rationales for both were generated with two models — `gpt-oss-120b-a100` (RITS) and `claude-opus-5`
(litellm gateway) — into a directory per dataset:

```
data/results/user-study/{base,roco}/rationales_{model}.json  (+ _summary.json)
```

Reproduce one cell with:

```bash
python src/crosswords/gen_rationales_ro.py --model_id claude-opus-5 \
  --dataset_file user-study/roco-sample-100.json --dataset_type roco \
  --min_answer_len 1 --resume --flat_names --output_name rationales \
  --output_dir data/results/user-study/roco
```

Two flags exist for this layout: `--flat_names` drops the dataset segment from the filename (the
directory already names the dataset), and `--min_answer_len 1` keeps a curated sample intact — the
default of 3 would silently drop 1-2 letter answers. Both output files for a dataset are row-aligned on
the same clues, so annotators can compare the two models item by item.

### Two prompts, selected by `--dataset_type`

The two samples need different prompts, and `select_prompt()` picks one from `--dataset_type` — there is
no separate flag, so the commands above already choose correctly:

- `clues`/`polycross`/`roco` → `INSTRUCTION_RATIONALE`. The roco clues are deliberately polysemantic, so
  the model must also return `surface_reading` (the misleading reading), `intended_sense` and
  `wordplay_type`.
- `baseline`/`themcross`/`base` → `INSTRUCTION_RATIONALE_BASE`. The base clues are **direct** thematic
  and encyclopedic definitions with no hidden sense, so this prompt returns only `{solution, rationale}`,
  caps the rationale at **2-4 short sentences**, and forbids both inventing a misleading reading and
  adding encyclopedic detail beyond the clue → solution link.

Records keep the same seven fields either way, so the two datasets stay row-comparable for annotators; a
base run simply leaves `surface_reading`/`intended_sense`/`wordplay_type` as `""`. Consequently the
`wordplay_types` histogram is **omitted** from a base run's `_summary.json` (it would always be empty)
and present for roco.

Why this split: run under the polysemantic prompt, the base dataset produced fabricated ambiguity
(ISAC — *"poate duce gândul spre numele biblic Isaac"*) and encyclopedic padding, with 25/100 items
given a real wordplay label. The direct prompt cut the average rationale from 461 to 221 chars for
`claude-opus-5` and from 294 to 189 for `gpt-oss-120b-a100`.

## Base models via transformers (`solve_base_ro.py`)

For **base** (non-instruction-tuned) models such as `faur-ai/LLMic_v2` (3B, Romanian-English, no chat
template). `clues_ro.py` cannot run these: every Mellea backend applies a chat template and the prompts
ask for JSON. `solve_base_ro.py` instead uses few-shot completion prompts (`prompts_base.py`: the same
OCUPAT/ROASA/SAC examples and hint wording as v1-v3, ending in `Răspuns:`), greedy decoding up to the
first newline, and writes the **same output format and filenames** as `clues_ro.py`
(`clues_llmic-v2_v3_2.json`, `baseline_llmic-v2_v1_0.json`), so `rescore.py` picks the results up.

```bash
./run_llmic.sh [model=llmic-v2] [batch=64]      # all 12 runs: {polycross 3000, themcross 1148} x {v1, v2, v3 k=1..4}
NUM_SAMPLES=50 OUT=/tmp/llmic_smoke ./run_llmic.sh   # smoke test
MODEL_PATH=/path/to/LLMic_v2 ./run_llmic.sh          # pre-downloaded weights
```

- Models live in `configs/hf_models.json` (separate from `local_models.json`, which is the Mellea/vLLM
  chat path). Only v1/v2/v3 exist for base models; v4 (CoT) and v5 (translate) are not defined.
- **LLMic tokenizer quirks**, handled via the config record: the Hub `config.json` says
  `eos_token_id=2`, which is `<|unk|>`; the real end-of-text is id 1. The tokenizer has
  `add_eos_token=True`, so prompts are encoded with `add_special_tokens=False`.
- **SDPA attention on Apple MPS breaks left-padded batches** (padded rows emit EOS at once); the solver
  uses eager attention on MPS and SDPA elsewhere. Every run starts with a batched-vs-single self-check
  that aborts if padded generation looks broken (an occasional differing answer is normal in fp16).
- `--resume` skips items already in the output file (the runner always passes it); results are
  checkpointed every `--save_every` batches.
- Ollama does **not** work for LLMic: `ollama create` from the safetensors succeeds but mis-tokenizes
  (unrelated English output for every prompt), because of the model's custom GPT-NeoX-style tokenizer.
- **transformers 5 mis-tokenizes LLMic via `AutoTokenizer`**: it rebuilds `GPTNeoXTokenizer` from class
  defaults and drops the custom `\n`-splitting pre-tokenizer in `tokenizer.json`, so every newline in the
  few-shot prompt becomes the wrong token and the model emits garbage (0% accuracy). The config record sets
  `"tokenizer_from_file": true`, which makes the solver load `tokenizer.json` verbatim via
  `PreTrainedTokenizerFast`. The batching self-check does **not** catch this (batched and single are
  equally wrong) — sanity-check the first predictions of any new base model.
- bert-score calls `build_inputs_with_special_tokens` for empty predictions, which transformers 5 removed;
  `eval_results` shims it back (`[CLS] ids [SEP]`), so scores match bert-score under transformers 4.
- **On the CCC cluster (LSF)**: env `/u/radu/envs/crosswords` (Python 3.12, torch 2.8.0+cu128, `pip
  install -e .` plus litellm, scipy, matplotlib, seaborn). Pre-download weights on the login node (the job
  runs with `HF_HUB_OFFLINE=1`), then submit `run_llmic_lsf.sh`, which wraps `run_llmic.sh`:
  `bsub -G grp_big_job_lease -q normal -gpu "num=1" -n 8 -R "rusage[mem=64GB]" -J llmic -o
  data/results/logs_llmic-v2/lsf_%J.out ./run_llmic_lsf.sh`. The full 12-run sweep takes ~45 min on one
  H100; re-submitting resumes. There is no system TeX; `/u/radu/envs/tex/bin/tectonic -X compile
  error_analysis.tex` builds the paper.
- The LLMic results are in and `paper/error_analysis.tex` has `\llmictrue`; the agreement analysis
  (figure and text) now covers five models over v1-v3 only.

## Error analysis (`src/crosswords/analysis/`, `paper/`)

Three steps, run in order from the repo root (each reads the previous step's output):

```bash
PYTHONPATH=src python -m crosswords.analysis.rescore          # -> data/results/analysis/{predictions,runs}.csv
PYTHONPATH=src python -m crosswords.analysis.error_analysis   # -> data/results/analysis/*.csv, paper/tables/*.tex
PYTHONPATH=src python -m crosswords.analysis.plots            # -> paper/figures/*.pdf
cd paper && latexmk -pdf error_analysis.tex                   # ICLR 2026 style files are vendored in paper/
```

- **Common item sets.** Every model is scored on `polycross_sample3000.json` and on `themcross_data.json`
  (answers >= 3 letters, 1148 items). The old granite/llama/gpt-oss polycross runs cover the full ~26k
  corpus and are aligned to the sample by (clue, uppercased answer); items missing from a run count as
  wrong. Claude files (`{polycross,themcross}_v*_[k*_]claude_opus48.json`) are **Claude Opus 4.8**, a
  dict whose `predictions` list is position-aligned with the dataset.
- **Metric.** `rescore.normalize` (NFD, drop combining marks, uppercase, letters only) is the main,
  diacritic-insensitive metric; `correct_strict` keeps the original case-insensitive exact match.
- Llama polycross v3 k=4 never existed on the full corpus; it was run on the sample as
  `clues_sample3000_llama-3.3-70b-instruct_v3_4.json` and wired in via `EXTRA_RUNS` in `rescore.py`.
- The "crossword vocabulary" used for candidate counts and OOV rates is the union of answers in both
  full corpora (~11.9k types), not a Romanian dictionary.

## Post-processing & manual annotation

- `json2csv.py <file.json>` → `<file>_output.csv`. Emits fixed columns `rationale_score, match, clue, answer, prediction, rationale` and computes `match` itself (case-insensitive answer==prediction). `rationale_score` is filled in **manually** (0–5 scale).
- `count_csv_stats.py <file.csv>` prints total / match-True / rationale_score≥5 counts.
- `*_annotated.csv` files are hand-annotated copies kept separate so re-running `json2csv.py` won't clobber them.

## Dataset generation (rarely needed)

`src/gen-clue-sol-datasets/` contains one-off scripts that produced the two committed datasets (JS scraping, `.docx`/`.rbs` parsing for the baseline). The outputs are already in `data/`, so you normally don't touch this. See `gen-baseline-dataset/README.md` for that pipeline.

## Conventions

- Python ≥ 3.11. Dev tooling declared in `pyproject.toml` (`pytest`, `ruff`, `mypy`) but there is no configured test suite or CI yet.
- All clue/answer text is Romanian with diacritics — use `load_json_utf8_relaxed()` (handles BOM + trailing commas) rather than bare `json.load` when reading possibly-hand-edited files.
- Result files are written with `ensure_ascii=False`; older committed results use escaped unicode.
