# Generate polysemantic Romanian crossword clues from known answers.
#
# Given a dataset of (answer, clue) pairs, this script ignores the existing
# clue and asks an LLM (via mellea/RITS) to generate several SHORT,
# polysemantic Romanian clues for each answer. This is the reverse of the
# solving task in clues_ro.py: here the answer is the input and we produce
# candidate clues as output.

import argparse
import json
import os
from typing import Any

import mellea.stdlib.functional as mfuncs
import numpy as np
from bert_score import BERTScorer
from dotenv import load_dotenv
from mellea.backends import Backend
from mellea.core import MelleaLogger
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.requirements import check, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy, RepairTemplateStrategy

# Disable Mellea logging
MelleaLogger.get_logger().setLevel(MelleaLogger.ERROR)

# Local imports
from crosswords.backends import build_backend, load_json_utf8_relaxed, print_models
from crosswords.runner import RateLimiter, run_throttled, throttle_backend
from crosswords.select_icl_sample import fold
from crosswords.utils import (
    extract_first_code_block,
    get_think_tags,
    strip_code_fences,
)

INSTRUCTION_GEN = """
You are an expert at composing crossword puzzle clues in Romanian.
Given the correct answer to a crossword (a word in Romanian), you will compose
{{num_candidates}} distinct candidate clues that could lead to that answer.

Important: each clue must be a POLYSEMANTIC clue. This means it should have
multiple possible meanings and could plausibly lead to several different
answers, but only one of them is the given answer. The more obvious reading of
the clue should ideally point elsewhere, so that finding the given answer
requires some lateral reasoning.

Rules:
- Every clue must be written in Romanian.
- Every clue must be SHORT: a brief definition or phrase, not a full sentence.
- A clue must NOT be a direct synonym of the answer. It should be an indirect
  definition that requires reasoning to reach the answer.
- The {{num_candidates}} clues must be distinct from one another.
- Do not translate the answer or the clues into English.
- Reason step by step: first consider the possible meanings of the answer and
  ways to allude to them indirectly, then craft and refine the candidate clues.
- Mark your reasoning with the following tags: <think> ** your thoughts ** </think>
- After your reasoning, output the final response as a JSON object inside a
  markdown code block, with exactly the following structure:

```json
{
  "answer": "<the input answer here>",
  "clues": ["<clue 1>", "<clue 2>", "<clue 3>"]
}
```

The "clues" array must contain exactly {{num_candidates}} clues.

Use the following examples to learn your task better.

Example 1:
ANSWER: OCUPAT
```json
{
  "answer": "OCUPAT",
  "clues": ["Prins asupra faptului", "Nu mai are loc liber", "Linie telefonica fara raspuns"]
}
```

Example 2:
ANSWER: ROASA
```json
{
  "answer": "ROASA",
  "clues": ["Marcată de o purtare abuzivă", "Macinata de griji", "Tocita la calcai"]
}
```

Example 3:
ANSWER: SAC
```json
{
  "answer": "SAC",
  "clues": ["Unitate de morărit", "Loc de dormit pentru drumeti", "Fundul lui nu se mai vede"]
}
```

ANSWER: {{answer_text}}
"""

INSTRUCTION_GEN_HINTS = """
You are an expert at composing crossword puzzle clues in Romanian.
Given the correct answer to a crossword (a word in Romanian), you will compose
{{num_candidates}} distinct candidate clues that could lead to that answer.

As a hint, you are given a few key content words taken from a real clue that a
human author wrote for this answer (common stop words have been removed). Use
these words as inspiration: your clues should incorporate or build around them
so that the full clue stays close in meaning and style to what the human author
intended.

Important: each clue must be a POLYSEMANTIC clue. This means it should have
multiple possible meanings and could plausibly lead to several different
answers, but only one of them is the given answer. The more obvious reading of
the clue should ideally point elsewhere, so that finding the given answer
requires some lateral reasoning.

Rules:
- Every clue must be written in Romanian.
- Every clue should make use of the given hint words (or close variants).
- Every clue must be SHORT: a brief definition or phrase, not a full sentence.
- A clue must NOT be a direct synonym of the answer. It should be an indirect
  definition that requires reasoning to reach the answer.
- The {{num_candidates}} clues must be distinct from one another.
- Do not translate the answer or the clues into English.
- Reason step by step: first consider the possible meanings of the answer and
  how the hint words constrain the clue, then craft and refine the candidates.
- Mark your reasoning with the following tags: <think> ** your thoughts ** </think>
- After your reasoning, output the final response as a JSON object inside a
  markdown code block, with exactly the following structure:

```json
{
  "answer": "<the input answer here>",
  "clues": ["<clue 1>", "<clue 2>", "<clue 3>"]
}
```

The "clues" array must contain exactly {{num_candidates}} clues.

Use the following example to learn your task better.

Example:
ANSWER: SAC
HINT: Unitate morărit
```json
{
  "answer": "SAC",
  "clues": ["Unitate de morărit", "Recipient de morărit pentru faina", "Masura veche la morărit"]
}
```

ANSWER: {{answer_text}}
HINT: {{hint_text}}
"""


load_dotenv()


# Few-shot variant (--prompt icl). Same task as INSTRUCTION_GEN, but the examples
# are real published (answer, clue) pairs from the corpus, filled in from the
# sample file by format_icl_examples, so the model imitates their length and
# style. The {{icl_examples}} block is plain text: one "ANSWER / CLUE" pair each.
INSTRUCTION_GEN_ICL = """
You are an expert at composing crossword puzzle clues in Romanian.
Given the correct answer to a crossword (a word in Romanian), you will compose
{{num_candidates}} distinct candidate clues that could lead to that answer.

Important: each clue must be a POLYSEMANTIC clue, like those published in
Romanian crossword magazines. The obvious, literal reading of the clue should
point somewhere else, and only a second, less obvious meaning of its words leads
to the given answer. Finding the answer should require lateral reasoning.

Rules:
- Every clue must be written in Romanian, with correct diacritics (ă, â, î, ș, ț).
- Every clue must be SHORT: at most {{max_clue_words}} words. Most published clues
  have 3 to 5 words. A clue is a phrase, not a full sentence or an explanation.
- A clue must NOT contain the answer, a word derived from it, or a direct synonym.
- If the answer is plural and the clue does not make that obvious, you may end
  the clue with "(pl.)", as the published clues do.
- The {{num_candidates}} clues must be distinct from one another, and should play
  on different meanings of the answer where possible.
- Do not translate the answer or the clues into English.
- Output the final response as a JSON object inside a markdown code block, with
  exactly the following structure:

```json
{
  "answer": "<the input answer here>",
  "clues": ["<clue 1>", "<clue 2>", "<clue 3>"]
}
```

The "clues" array must contain exactly {{num_candidates}} clues.

Here are real published clues. Match their brevity and their kind of wordplay:

{{icl_examples}}

ANSWER: {{answer_text}}
"""


def format_icl_examples(examples: list[dict[str, Any]]) -> str:
    """Render the sample file's ICL pairs as the prompt's examples block."""
    return "\n\n".join(f"ANSWER: {ex['answer']}\nCLUE: {ex['clue']}" for ex in examples)


def load_icl_sample(file_name: str) -> dict[str, list[dict[str, Any]]]:
    """
    Load a sample file written by select_icl_sample.py: {"icl": [...], "eval": [...]}.
    The "icl" pairs go into the prompt and the "eval" items are generated for.
    """
    sample = load_json_utf8_relaxed(file_name)
    if not isinstance(sample, dict) or not {"icl", "eval"} <= sample.keys():
        raise ValueError(
            f"{file_name} is not an ICL sample file: expected 'icl' and 'eval' keys. "
            "Create one with: python -m crosswords.select_icl_sample"
        )
    return sample


def clue_problems(clues: Any, answer: str, num_candidates: int, max_words: int) -> list[str]:
    """
    Reasons a list of generated clues breaks the ICL prompt's hard rules: the
    count, the word limit, and not giving the answer away. Empty means valid.
    Used for rejection sampling, so a failing response is regenerated instead
    of being silently kept.
    """
    if not isinstance(clues, list) or not all(isinstance(c, str) for c in clues):
        return ["'clues' must be a list of strings"]
    problems = []
    if len(clues) != num_candidates:
        problems.append(f"expected {num_candidates} clues, got {len(clues)}")
    # Compare without diacritics or case; the first five letters catch inflected
    # forms of the answer (e.g. ACCEPTA in "acceptat").
    stem = fold(answer)[:5]
    for clue in clues:
        if len(clue.split()) > max_words:
            problems.append(f"over {max_words} words: {clue!r}")
        if stem and stem in fold(clue):
            problems.append(f"contains the answer: {clue!r}")
    return problems


# Common Romanian stop words (and a few function-word forms) that carry little
# meaning on their own and so make poor content hints.
RO_STOP_WORDS = {
    "a", "ai", "al", "ale", "alor", "am", "ar", "as", "au", "ca", "ce", "cei",
    "cel", "cele", "celor", "cea", "cu", "cum", "da", "dar", "de", "din", "doar",
    "dupa", "după", "ei", "el", "ele", "este", "eu", "fi", "fie", "fiind", "iar",
    "in", "în", "isi", "își", "la", "le", "li", "lor", "lui", "mai", "mea",
    "mele", "meu", "mi", "mie", "mine", "ne", "ni", "noi", "nostru", "nu", "o",
    "or", "pe", "pentru", "peste", "prin", "sa", "să", "sale", "sau", "său",
    "se", "si", "și", "sub", "sunt", "ta", "te", "ti", "ție", "tu", "tot",
    "toti", "toți", "un", "una", "unei", "unele", "unor", "unu", "unui", "va",
    "vi", "voi", "vor",
}


def get_hint_words(clue: str, num_hints: int) -> str:
    """
    Return the first num_hints content words of the clue, skipping Romanian
    stop words (and pure-punctuation tokens) so the hints are informative.
    Comparison is case-insensitive; the original casing of the kept words is
    preserved.
    """
    if num_hints <= 0 or not clue:
        return ""
    hints = []
    for word in clue.split():
        # Strip surrounding punctuation only for the stop-word/empty check.
        stripped = word.strip(".,;:!?\"'„”«»()[]-")
        if not stripped:
            continue
        if stripped.lower() in RO_STOP_WORDS:
            continue
        hints.append(word)
        if len(hints) >= num_hints:
            break
    return " ".join(hints)


def get_sbert(reference: str, prediction: str, scorer) -> float:
    """BERTScore F1 between a reference and a prediction string."""
    _, _, F1 = scorer.score([prediction], [reference])
    return F1.cpu().detach().numpy().tolist()[0]


def parse_clues_response(text: str) -> dict[str, Any]:
    """
    Parse an LLM response into a JSON dict, tolerating a <think>...</think>
    preamble before the fenced JSON block. Raises on failure.
    """
    cleaned = extract_first_code_block(text, ignore_language=True)
    if not cleaned:
        cleaned = strip_code_fences(text)
    return json.loads(cleaned)


def validate_clues_response(text: str, required_keys: list[str]) -> bool:
    """
    Validation for the rejection sampler. Unlike validate_json_code_block, this
    extracts the fenced block from anywhere in the response, so a valid answer
    preceded by <think> reasoning is not wrongly rejected.
    """
    try:
        data = parse_clues_response(text)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(data, dict) and all(k in data for k in required_keys)


def load_data(file_name: str) -> list[dict[str, Any]]:
    """Load the dataset."""
    try:
        with open(file_name, "r") as f:
            data = json.load(f)
        return data
    except (OSError, ValueError) as e:
        print(e)
        return None


def process_data(
        data: list[dict[str, Any]],
        backend: Backend,
        dataset_type: str,
        num_candidates: int,
        num_hints: int = 0,
        num_samples: int | None = None,
        output_filename: str = "gen_clues.json",
        batch_size: int = 50,
        icl_examples: list[dict[str, Any]] | None = None,
        max_clue_words: int = 6,
) -> list[dict[str, Any]]:
    """
    Generate candidate clues for the answers in the dataset.

    :param data: list of (answer, clue) dicts
    :param backend: mellea backend used to call the LLM
    :param dataset_type: "clues" or "baseline" (selects the answer field)
    :param num_candidates: number of candidate clues to request per answer
    :param num_hints: number of leading words of the ground-truth clue to give
        as a hint (0 = no hint, use the plain generation prompt)
    :param icl_examples: real (answer, clue) pairs. When given, use the few-shot
        INSTRUCTION_GEN_ICL prompt and also enforce the clue count, the
        max_clue_words limit and no answer leakage during rejection sampling.
    :param max_clue_words: word limit per clue for the ICL prompt
    """

    print(f"Generating clues for {len(data)} answers...")
    print(f"Using LLM: {backend.model_id}")
    print(f"Dataset type: {dataset_type}")
    print(f"Candidates per answer: {num_candidates}")
    print(f"Hint words from ground-truth clue: {num_hints}")
    if icl_examples:
        print(f"ICL prompt: {len(icl_examples)} examples, <= {max_clue_words} words per clue")
    icl_text = format_icl_examples(icl_examples) if icl_examples else ""

    assert dataset_type in ["clues", "baseline"], f"Unknown dataset type: {dataset_type}"

    # extracted_data.json uses "answer"; baseline-dataset.json uses "solution".
    if dataset_type == "clues":
        answer_key = "answer"
    else:
        answer_key = "solution"
    clue_key = "clue"

    print(f"Initial number of answers: {len(data)}")

    # Filter out the very short (<= 2 letter) answers, matching clues_ro.py.
    data = [item for item in data if len(item[answer_key]) > 2]
    print(f"After filtering out short answers, {len(data)} answers remain.")

    if num_samples is not None:
        data = data[:num_samples]
        print(f"Limited to {len(data)} answers (num_samples={num_samples}).")

    async def acall_one(item):
        answer = item[answer_key]

        # With num_hints >= 1, feed the first num_hints words of the real clue
        # as a hint and use the hint-aware prompt; otherwise plain generation.
        hint_text = get_hint_words(item.get(clue_key, ""), num_hints)
        if icl_examples:
            instruction = INSTRUCTION_GEN_ICL
            user_variables = {
                "answer_text": answer,
                "num_candidates": num_candidates,
                "max_clue_words": max_clue_words,
                "icl_examples": icl_text,
            }
        elif num_hints >= 1 and hint_text:
            instruction = INSTRUCTION_GEN_HINTS
            user_variables = {
                "answer_text": answer,
                "num_candidates": num_candidates,
                "hint_text": hint_text,
            }
        else:
            instruction = INSTRUCTION_GEN
            user_variables = {"answer_text": answer, "num_candidates": num_candidates}

        requirements = [check(
            "The output must contain a valid JSON code block with 'answer' and 'clues' keys.",
            validation_fn=simple_validate(
                lambda s: validate_clues_response(s, required_keys=["answer", "clues"])
            ),
        )]
        strategy = RejectionSamplingStrategy(loop_budget=5)
        if icl_examples:
            def check_clue_rules(s: str) -> tuple[bool, str]:
                if not validate_clues_response(s, ["clues"]):
                    return False, "The output must contain a valid JSON code block."
                problems = clue_problems(
                    parse_clues_response(s).get("clues"), answer,
                    num_candidates, max_clue_words,
                )
                return not problems, "; ".join(problems)

            requirements.append(check(
                f"Exactly {num_candidates} clues, each at most {max_clue_words} words, "
                "none containing the answer.",
                validation_fn=simple_validate(check_clue_rules),
            ))
            # Plain rejection sampling resends the same prompt, and a model that
            # writes one clue a word too long tends to repeat it every time. The
            # repair strategy appends the failure reasons (e.g. "over 6 words:
            # ...") to the instruction, so each retry knows what to fix.
            strategy = RepairTemplateStrategy(loop_budget=5)

        return await mfuncs.ainstruct(
            instruction,
            context=SimpleContext(),
            backend=backend,
            requirements=requirements,
            user_variables=user_variables,
            icl_examples=[],
            strategy=strategy,
            return_sampling_results=True,
        )

    ordered = run_throttled(data, acall_one, batch_size=batch_size)

    results = []
    num_generated = 0
    for i, (output, item, error) in enumerate(ordered):
        answer = item[answer_key]
        source_clue = item.get(clue_key, "")
        if error is not None:
            print(f"  [{i + 1}/{len(data)}] [ERROR] answer: {answer} "
                  f"({type(error).__name__}: {error})")
        status = "OK" if (error is None and output.success) else "FAIL"

        clues = []
        rationale = get_think_tags(str(output)) if status == "OK" else ""
        if status == "OK":
            try:
                # The model emits <think>...</think> reasoning before the
                # fenced JSON block; parse_clues_response handles that.
                pred_dict = parse_clues_response(str(output))
                clues = pred_dict.get("clues", [])
                if not isinstance(clues, list):
                    clues = [str(clues)]
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"  [{i + 1}/{len(data)}] [PARSE-FAIL] answer: {answer} ({e})")

        if clues:
            num_generated += 1

        results.append({
            "answer": answer,
            "source_clue": source_clue,
            "filePath": item.get("filePath", item.get("path", "")),
            "hint": get_hint_words(source_clue, num_hints),
            "num_hints": num_hints,
            "clues": clues,
            "rationale": rationale,
        })

        print(f"  [{i + 1}/{len(data)}] [{status}] answer: {answer}")
        for c in clues:
            print(f"    clue: {c}")

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Finished {len(data)} answers")
    print(f"Generated clues for {num_generated}/{len(data)} answers.")

    return results


def eval_results(output_filename: str, max_clue_words: int | None = None) -> dict[str, Any]:
    """
    Score the generated clues against the ground-truth (source) clue using
    BERTScore F1. For each answer with at least one generated clue, the best
    candidate (highest F1 vs the source clue) is taken as the predicted clue.
    A summary dict is appended as the final element of the output file. With
    max_clue_words, it also reports the share of clues within that word limit.
    """

    print(f"Evaluating generated clues from {output_filename} ...")
    with open(output_filename, "r", encoding="utf-8") as f:
        results = json.load(f)
    # Re-scoring a file that was already evaluated: drop the old summary.
    if results and "num_scored" in results[-1]:
        results = results[:-1]

    assert len(results) > 0, "No results to evaluate."

    scorer = BERTScorer(model_type='bert-base-uncased', device='cpu')

    best_scores = []
    num_scored = 0
    for item in results:
        source_clue = item.get("source_clue", "")
        clues = item.get("clues", [])
        # Need a reference clue and at least one candidate to score.
        if not source_clue or not clues:
            item["best_clue"] = None
            item["best_sbert"] = None
            continue

        scored = [(c, get_sbert(source_clue, c, scorer)) for c in clues if c]
        if not scored:
            item["best_clue"] = None
            item["best_sbert"] = None
            continue

        best_clue, best_sbert = max(scored, key=lambda t: t[1])
        item["best_clue"] = best_clue
        item["best_sbert"] = float(best_sbert)
        best_scores.append(best_sbert)
        num_scored += 1

    if best_scores:
        sbert_mean = float(np.mean(best_scores))
        sbert_std = float(np.std(best_scores))
    else:
        sbert_mean = 0.0
        sbert_std = 0.0

    summary = {
        "num_samples": len(results),
        "num_scored": num_scored,
        "sbert_mean": sbert_mean,
        "sbert_std": sbert_std,
    }
    lengths = [len(c.split()) for item in results for c in item.get("clues", []) if c]
    if lengths:
        summary["clue_words_mean"] = float(np.mean(lengths))
        if max_clue_words is not None:
            summary["max_clue_words"] = max_clue_words
            summary["within_word_limit"] = float(np.mean([n <= max_clue_words for n in lengths]))

    print(f"Evaluation results: {summary}")
    print(f"Saving evaluation results to {output_filename} ...")
    results.append(summary)
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default="gpt-oss-120b-a100",
                        help='A model name from configs/rits_models.json or '
                             'configs/frontier_models.json, or a raw litellm model id '
                             'containing a "/". Use --list_models to see the options.')
    parser.add_argument('--dataset_file', type=str,
                        help='Input JSON file (required unless --list_models)')
    parser.add_argument('--dataset_type', type=str, default="clues")
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--num_candidates', type=int, default=3)
    parser.add_argument('--num_hints', type=int, default=0,
                        help='Number of leading words of the ground-truth clue to give as hint (0 = none)')
    parser.add_argument('--output_name', type=str, default="gen_clues")
    parser.add_argument('--output_dir', type=str, default=".")
    parser.add_argument('--batch_size', type=int, default=50, help='Max concurrent async requests')
    parser.add_argument('--rate_limit', type=int, default=1500, help='Max requests per minute')
    parser.add_argument('--eval_only', action='store_true',
                        help='Skip generation and only score an existing output file')
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--list_models', action='store_true',
                        help='List the models available from the config files and exit')
    parser.add_argument('--prompt', choices=["gen", "icl"], default="gen",
                        help='"gen": hand-written examples (with --num_hints for hint '
                             'words). "icl": real corpus clues as few-shot examples, '
                             'read from --icl_file')
    parser.add_argument('--icl_file', type=str,
                        default="data/gen_clues/polycross_icl_sample.json",
                        help='Sample file from select_icl_sample.py. With --prompt icl '
                             'it supplies both the examples and the answers to generate '
                             'for, so --dataset_file is not needed')
    parser.add_argument('--max_clue_words', type=int, default=6,
                        help='Word limit per clue for --prompt icl')

    args = parser.parse_args()

    if args.list_models:
        print_models()
        raise SystemExit(0)

    if args.prompt == "icl":
        if args.num_hints:
            parser.error("--num_hints cannot be combined with --prompt icl")
        # The model id may contain "/" (a raw litellm id); keep the filename flat.
        safe_model = args.model_id.replace("/", "_")
        output_filename = f"{args.output_name}_{args.dataset_type}_{safe_model}_icl.json"
        max_clue_words = args.max_clue_words
    else:
        if not args.dataset_file:
            parser.error("--dataset_file is required (unless --list_models or --prompt icl)")
        output_filename = f"{args.output_name}_{args.dataset_type}_{args.model_id}_h{args.num_hints}.json"
        max_clue_words = None
    output_filename = os.path.join(args.output_dir, output_filename)

    if args.eval_only:
        eval_results(output_filename, max_clue_words=max_clue_words)
        print("Done.")
        raise SystemExit(0)

    if args.output_dir and not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Wrap so rejection-sampling retries also count against the rate limit.
        backend = throttle_backend(
            build_backend(args.model_id, max_tokens=args.max_tokens),
            RateLimiter(args.rate_limit, per=60.0),
        )
    except (RuntimeError, ValueError) as e:
        # Configuration problems (missing credentials, unknown model, RITS
        # unavailable) are user errors, not bugs: no traceback.
        raise SystemExit(f"Configuration error: {e}")

    icl_examples = None
    if args.prompt == "icl":
        try:
            sample = load_icl_sample(args.icl_file)
        except (OSError, ValueError) as e:
            raise SystemExit(f"Configuration error: {e}")
        data, icl_examples = sample["eval"], sample["icl"]
    else:
        data = load_data(args.dataset_file)
    process_data(
        data,
        backend,
        args.dataset_type,
        args.num_candidates,
        num_hints=args.num_hints,
        num_samples=args.num_samples,
        output_filename=output_filename,
        batch_size=args.batch_size,
        icl_examples=icl_examples,
        max_clue_words=args.max_clue_words,
    )

    # Score the generated clues against the ground-truth clues.
    eval_results(output_filename, max_clue_words=max_clue_words)

    print("Done.")
