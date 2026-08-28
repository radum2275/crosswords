# Generate polysemantic Romanian crossword clues from known answers.
#
# Given a dataset of (answer, clue) pairs, this script ignores the existing
# clue and asks an LLM (via mellea/RITS) to generate several SHORT,
# polysemantic Romanian clues for each answer. This is the reverse of the
# solving task in clues_ro.py: here the answer is the input and we produce
# candidate clues as output.

import os
import asyncio
import argparse
import json

import numpy as np
import mellea.stdlib.functional as mfuncs

from bert_score import BERTScorer

from dotenv import load_dotenv
from typing import Any, Dict, List
from mellea.backends import Backend
from mellea.backends import ModelOption
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.requirements import check, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

from mellea_ibm.rits import RITSBackend, RITS
from mellea.core import FancyLogger

# Disable Mellea logging
FancyLogger.get_logger().setLevel(FancyLogger.ERROR)

# Local imports
from crosswords.utils import (
    strip_code_fences,
    extract_first_code_block,
    get_think_tags,
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


def parse_clues_response(text: str) -> Dict[str, Any]:
    """
    Parse an LLM response into a JSON dict, tolerating a <think>...</think>
    preamble before the fenced JSON block. Raises on failure.
    """
    cleaned = extract_first_code_block(text, ignore_language=True)
    if not cleaned:
        cleaned = strip_code_fences(text)
    return json.loads(cleaned)


def validate_clues_response(text: str, required_keys: List[str]) -> bool:
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


def load_data(file_name: str) -> List[Dict[str, Any]]:
    """Load the dataset."""
    try:
        with open(file_name, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(e)
        return None


def process_data(
        data: List[Dict[str, Any]],
        backend: Backend,
        dataset_type: str,
        num_candidates: int,
        num_hints: int = 0,
        num_samples: int = None,
        output_filename: str = "gen_clues.json",
        batch_size: int = 50,
        rate_limit: int = 1500,
) -> List[Dict[str, Any]]:
    """
    Generate candidate clues for the answers in the dataset.

    :param data: list of (answer, clue) dicts
    :param backend: mellea backend used to call the LLM
    :param dataset_type: "clues" or "baseline" (selects the answer field)
    :param num_candidates: number of candidate clues to request per answer
    :param num_hints: number of leading words of the ground-truth clue to give
        as a hint (0 = no hint, use the plain generation prompt)
    """

    print(f"Generating clues for {len(data)} answers...")
    print(f"Using LLM: {backend.model_id}")
    print(f"Dataset type: {dataset_type}")
    print(f"Candidates per answer: {num_candidates}")
    print(f"Hint words from ground-truth clue: {num_hints}")

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
        if num_hints >= 1 and hint_text:
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

        return await mfuncs.ainstruct(
            instruction,
            context=SimpleContext(),
            backend=backend,
            requirements=requirements,
            user_variables=user_variables,
            icl_examples=[],
            strategy=RejectionSamplingStrategy(loop_budget=5),
            return_sampling_results=True,
        ), item

    print(f"Submitting {len(data)} prompts async (batch_size={batch_size}, rate_limit={rate_limit}/min) ...")
    sem = asyncio.Semaphore(batch_size)
    interval = 60.0 / rate_limit
    ordered = [None] * len(data)

    async def run_one(i, item):
        async with sem:
            ordered[i] = await acall_one(item)

    async def run_all():
        tasks = []
        for i, item in enumerate(data):
            if i > 0:
                await asyncio.sleep(interval)
            tasks.append(asyncio.create_task(run_one(i, item)))
        await asyncio.gather(*tasks)

    asyncio.run(run_all())

    results = []
    num_generated = 0
    for i, (output, item) in enumerate(ordered):
        answer = item[answer_key]
        source_clue = item.get(clue_key, "")
        status = "OK" if output.success else "FAIL"

        clues = []
        rationale = get_think_tags(str(output)) if output.success else ""
        if output.success:
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
            "hint": get_hint_words(source_clue, num_hints),
            "num_hints": num_hints,
            "clues": clues,
            "rationale": rationale,
        })

        print(f"  [{i + 1}/{len(data)}] [{status}] answer: {answer}")
        for c in clues:
            print(f"    clue: {c}")

    with open(output_filename, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Finished {len(data)} answers")
    print(f"Generated clues for {num_generated}/{len(data)} answers.")

    return results


def eval_results(output_filename: str) -> Dict[str, Any]:
    """
    Score the generated clues against the ground-truth (source) clue using
    BERTScore F1. For each answer with at least one generated clue, the best
    candidate (highest F1 vs the source clue) is taken as the predicted clue.
    A summary dict is appended as the final element of the output file.
    """

    print(f"Evaluating generated clues from {output_filename} ...")
    with open(output_filename, "r", encoding="utf-8") as f:
        results = json.load(f)

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

    print(f"Evaluation results: {summary}")
    print(f"Saving evaluation results to {output_filename} ...")
    results.append(summary)
    with open(output_filename, "w") as f:
        json.dump(results, f, indent=4)

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default="gpt-oss")
    parser.add_argument('--dataset_file', type=str, required=True)
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

    args = parser.parse_args()

    output_filename = f"{args.output_name}_{args.dataset_type}_{args.model_id}_h{args.num_hints}.json"
    output_filename = os.path.join(args.output_dir, output_filename)

    if args.eval_only:
        eval_results(output_filename)
        print("Done.")
        raise SystemExit(0)

    # Create a Mellea RITS backend
    if args.model_id == "llama":
        backend = RITSBackend(
            RITS.LLAMA_3_3_70B_INSTRUCT,
            model_options={ModelOption.MAX_NEW_TOKENS: 4096}
        )
    elif args.model_id == "granite":
        backend = RITSBackend(
            RITS.GRANITE_4_H_SMALL,
            model_options={ModelOption.MAX_NEW_TOKENS: 4096}
        )
    elif args.model_id == "mistral":
        backend = RITSBackend(
            RITS.MISTRAL_LARGE_3_675B_2512,
            model_options={ModelOption.MAX_NEW_TOKENS: 4096}
        )
    elif args.model_id == "gpt-oss":
        backend = RITSBackend(
            RITS.GPT_OSS_120B,
            model_options={ModelOption.MAX_NEW_TOKENS: 4096}
        )
    else:
        raise ValueError(f"Unknown LLM backend.")

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
        rate_limit=args.rate_limit,
    )

    # Score the generated clues against the ground-truth clues.
    eval_results(output_filename)

    print("Done.")
