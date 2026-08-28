# Generate LLM rationales for Romanian crossword clue/solution pairs.
#
# Given a dataset of (solution, clue) pairs where the solution is the KNOWN
# ground truth, this script asks an LLM to explain WHY that solution is the
# right answer to that clue. Unlike clues_ro.py (which must find the answer)
# nothing is being predicted here: the answer is given and the output is the
# explanation.
#
# Published crossword clues are deliberately polysemantic -- the obvious
# reading is usually wrong -- so the prompt asks the model to separate the
# misleading surface reading from the sense the author actually intended.
#
# Works with both RITS models and the frontier models served by the IBM
# litellm gateway (ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN in .env).

import os
import csv
import json
import asyncio
import argparse

import mellea.stdlib.functional as mfuncs

from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Tuple
from mellea.backends import Backend
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.requirements import check, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

from mellea.core import FancyLogger

# Disable Mellea logging
FancyLogger.get_logger().setLevel(FancyLogger.ERROR)

# Local imports
from crosswords.backends import (
    build_backend,
    print_models,
    load_json_utf8_relaxed,
)
from crosswords.utils import (
    strip_code_fences,
    extract_first_code_block,
    get_think_tags,
)

load_dotenv()


INSTRUCTION_RATIONALE = """
Ești un expert în rezolvarea și explicarea definițiilor de la cuvinte
încrucișate în limba română. Primești o DEFINIȚIE și SOLUȚIA corectă a acesteia.
Sarcina ta este să explici de ce soluția dată este răspunsul corect.

Important: definițiile publicate în revistele de cuvinte încrucișate sunt
intenționat polisemantice. Sensul evident, cel care sare primul în ochi, este de
obicei greșit, iar soluția corectă cere găsirea unui sens mai subtil, figurat sau
contextual. Explicația ta trebuie să arate exact acest lucru: care este citirea
înșelătoare și care este sensul pe care autorul l-a avut în minte.

Reguli:
- SOLUȚIA primită este corectă prin definiție. Nu o contesta, nu propune alt
  răspuns și nu spune că definiția ar fi greșită: sarcina ta este să o explici.
- Toate câmpurile trebuie scrise în ROMÂNĂ, cu diacritice corecte.
- Nu traduce definiția sau soluția în engleză.
- "rationale": 2-4 propoziții care urmăresc lanțul de sens de la definiție la
  soluție. Explică mecanismul, nu doar rezultatul.
- "surface_reading": sensul evident și înșelător al definiției, cel care duce
  gândul în altă direcție decât soluția.
- "intended_sense": sensul real, folosit de autorul definiției pentru a ajunge
  la soluție.
- "wordplay_type": exact una dintre valorile: polisemie, metaforă, sens figurat,
  omonimie, sinonimie contextuală, joc de cuvinte, cunoștințe generale.
- Dacă definiția este directă și nu conține nicio ambiguitate, folosește
  "cunoștințe generale" și scrie la "surface_reading" faptul că definiția este
  directă.
- Gândește pas cu pas: analizează mai întâi toate sensurile posibile ale
  definiției, vezi care dintre ele conduce la soluția dată, apoi formulează
  explicația.
- Marchează raționamentul cu următoarele etichete: <think> ** gândurile tale ** </think>
- După raționament, scrie răspunsul final ca un obiect JSON într-un bloc de cod
  markdown, exact cu următoarea structură:

```json
{
  "solution": "<soluția primită>",
  "rationale": "<explicația în 2-4 propoziții>",
  "surface_reading": "<sensul evident, înșelător>",
  "intended_sense": "<sensul real, intenționat de autor>",
  "wordplay_type": "<tipul de joc de sens>"
}
```

Folosește exemplele următoare pentru a înțelege mai bine sarcina.

Exemplul 1:
DEFINIȚIE: Unitate de morărit
SOLUȚIE: SAC
```json
{
  "solution": "SAC",
  "rationale": "Definiția pare să ceară un utilaj sau o secție tehnică legată de procesul de măcinare. În realitate, cuvântul „unitate” trimite la o unitate de măsură și de ambalare, nu la un echipament. În morărit, cerealele și făina se măsoară și se transportă tradițional la sac, astfel încât sacul funcționează ca unitate de referință a meseriei.",
  "surface_reading": "Un utilaj, o instalație sau o secție care ține de procesul de măcinare a cerealelor.",
  "intended_sense": "Unitate de măsură și de ambalare folosită în morărit, adică sacul în care se pun grânele și făina.",
  "wordplay_type": "polisemie"
}
```

Exemplul 2:
DEFINIȚIE: Şters din datoria personalului de serviciu
SOLUȚIE: PRAF
```json
{
  "solution": "PRAF",
  "rationale": "Definiția sugerează la prima vedere ceva eliminat dintr-o obligație formală a angajaților, ca și cum o datorie ar fi fost ștearsă. Sensul real vizează însă obiectul concret al muncii de curățenie: personalul de serviciu are datoria de a șterge praful. Ceea ce este „șters” în cadrul acestei datorii este deci chiar praful, iar „șters” are aici sensul de îndepărtat prin ștergere.",
  "surface_reading": "Ceva anulat sau eliminat dintr-o obligație formală a angajaților.",
  "intended_sense": "Ceea ce personalul de curățenie îndepărtează efectiv de pe suprafețe ca parte a atribuțiilor sale.",
  "wordplay_type": "polisemie"
}
```

Exemplul 3:
DEFINIȚIE: Marcată de o purtare abuzivă
SOLUȚIE: ROASA
```json
{
  "solution": "ROASA",
  "rationale": "Definiția pare să descrie o persoană afectată de comportamentul agresiv al cuiva. Sensul intenționat mută însă „purtarea” de la comportament la purtatul unui obiect: o haină sau o pereche de pantofi purtate excesiv se roade. Astfel, cea marcată de o purtare abuzivă este roasă, adică uzată prin folosire îndelungată.",
  "surface_reading": "O persoană afectată de comportamentul agresiv sau abuziv al altcuiva.",
  "intended_sense": "Un obiect uzat din cauza folosirii excesive, unde „purtare” înseamnă a purta un lucru.",
  "wordplay_type": "sens figurat"
}
```

DEFINIȚIE: {{clue_text}}
SOLUȚIE: {{solution_text}}
"""


# The JSON keys the model must return.
REQUIRED_KEYS = [
    "solution",
    "rationale",
    "surface_reading",
    "intended_sense",
    "wordplay_type",
]


def parse_rationale_response(text: str) -> Dict[str, Any]:
    """
    Parse an LLM response into a JSON dict, tolerating a <think>...</think>
    preamble before the fenced JSON block. Raises on failure.
    """
    cleaned = extract_first_code_block(text, ignore_language=True)
    if not cleaned:
        cleaned = strip_code_fences(text)
    return json.loads(cleaned)


def validate_rationale_response(text: str, required_keys: List[str]) -> bool:
    """
    Validation for the rejection sampler. Extracts the fenced block from
    anywhere in the response, so a valid answer preceded by <think> reasoning
    is not wrongly rejected, and requires all keys to be non-empty strings.
    """
    try:
        data = parse_rationale_response(text)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(data, dict):
        return False
    for key in required_keys:
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            return False
    return True


def load_data(file_name: str) -> Optional[List[Dict[str, Any]]]:
    """Load the dataset."""
    try:
        return load_json_utf8_relaxed(file_name)
    except Exception as e:
        print(f"Failed to load {file_name}: {e}")
        return None


def resolve_keys(data: List[Dict[str, Any]], dataset_type: str) -> Tuple[str, str]:
    """
    Work out which field holds the solution and which holds the clue.

    extracted_data.json uses "answer"; baseline-dataset.json uses "solution".
    The requested --dataset_type decides, but if that key is absent from the
    data we fall back to whichever of the two is actually present, so a plain
    {solution, clue} file works regardless of the flag.
    """
    preferred = "answer" if dataset_type == "clues" else "solution"
    sample = data[0] if data else {}

    if preferred in sample:
        solution_key = preferred
    elif "solution" in sample:
        solution_key = "solution"
    elif "answer" in sample:
        solution_key = "answer"
    else:
        raise KeyError(
            "Dataset records have neither a 'solution' nor an 'answer' field "
            f"(found: {sorted(sample.keys())})."
        )

    if "clue" not in sample:
        raise KeyError(
            f"Dataset records have no 'clue' field (found: {sorted(sample.keys())})."
        )

    return solution_key, "clue"


def load_existing(output_filename: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Load already-generated records from a previous run, keyed by
    (clue, solution), keeping only those that actually have a rationale.
    Used by --resume so an interrupted sweep is not re-paid for.
    """
    if not os.path.exists(output_filename):
        return {}
    try:
        previous = load_json_utf8_relaxed(output_filename)
    except Exception as e:
        print(f"Could not read existing output {output_filename}: {e}")
        return {}
    if not isinstance(previous, list):
        return {}

    done = {}
    for item in previous:
        if not isinstance(item, dict):
            continue
        if not str(item.get("rationale", "")).strip():
            continue
        done[(item.get("clue", ""), item.get("solution", ""))] = item
    return done


def write_results(results: List[Dict[str, Any]], output_filename: str) -> None:
    """Write results as UTF-8 JSON, keeping Romanian diacritics literal."""
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


def write_csv(results: List[Dict[str, Any]], csv_filename: str) -> None:
    """
    Emit a companion CSV for manual annotation. The rationale_score column is
    left empty, to be filled in by hand on a 0-5 scale (matching json2csv.py
    and the manual-score-baseline/ convention). utf-8-sig so Excel renders the
    diacritics correctly.
    """
    columns = [
        "rationale_score",
        "clue",
        "solution",
        "rationale",
        "surface_reading",
        "intended_sense",
        "wordplay_type",
    ]
    with open(csv_filename, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for item in results:
            row = {col: item.get(col, "") for col in columns}
            row["rationale_score"] = ""
            writer.writerow(row)
    print(f"Wrote CSV for manual annotation to {csv_filename}")


def summarize(
    results: List[Dict[str, Any]],
    model_id: str,
    summary_filename: str,
) -> Dict[str, Any]:
    """
    Write a run summary to a SIDECAR file rather than appending it into the
    results array. The other scripts append their summary as the final element,
    which makes re-running them non-idempotent and forces downstream readers to
    special-case the last record; a sidecar avoids both problems.
    """
    rationales = [str(r.get("rationale", "")) for r in results]
    generated = [r for r in rationales if r.strip()]

    histogram: Dict[str, int] = {}
    for item in results:
        wordplay = str(item.get("wordplay_type", "")).strip().lower()
        if wordplay:
            histogram[wordplay] = histogram.get(wordplay, 0) + 1

    avg_len = sum(len(r) for r in generated) / len(generated) if generated else 0.0

    summary = {
        "model_id": model_id,
        "num_samples": len(results),
        "num_generated": len(generated),
        "num_failed": len(results) - len(generated),
        "avg_rationale_len": round(avg_len, 2),
        "wordplay_types": dict(sorted(histogram.items(), key=lambda kv: -kv[1])),
    }

    with open(summary_filename, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(f"Summary: {json.dumps(summary, ensure_ascii=False)}")
    print(f"Wrote summary to {summary_filename}")
    return summary


def process_data(
        data: List[Dict[str, Any]],
        backend: Backend,
        dataset_type: str,
        num_samples: int = None,
        output_filename: str = "rationales.json",
        batch_size: int = 50,
        rate_limit: int = 1500,
        resume: bool = False,
) -> List[Dict[str, Any]]:
    """
    Generate a rationale for every (clue, solution) pair in the dataset.

    :param data: list of dicts with a clue field and a solution/answer field
    :param backend: mellea backend used to call the LLM
    :param dataset_type: "clues" or "baseline" (selects the solution field)
    :param resume: skip pairs that already have a rationale in the output file
    """

    assert dataset_type in ["clues", "baseline"], f"Unknown dataset type: {dataset_type}"

    solution_key, clue_key = resolve_keys(data, dataset_type)

    print(f"Using LLM: {backend.model_id}")
    print(f"Dataset type: {dataset_type} (solution field: '{solution_key}')")
    print(f"Initial number of pairs: {len(data)}")

    # Filter out the very short (<= 2 letter) answers, matching clues_ro.py.
    data = [item for item in data if len(str(item.get(solution_key, ""))) > 2]
    print(f"After filtering out short solutions, {len(data)} pairs remain.")

    # Filter first, then truncate, so --num_samples yields exactly that many.
    if num_samples is not None:
        data = data[:num_samples]
        print(f"Limited to {len(data)} pairs (num_samples={num_samples}).")

    done = load_existing(output_filename) if resume else {}
    if resume:
        print(f"Resume: found {len(done)} already-generated rationales in {output_filename}")

    todo = [
        item for item in data
        if (item.get(clue_key, ""), item.get(solution_key, "")) not in done
    ]
    if resume:
        print(f"Resume: {len(data) - len(todo)} skipped, {len(todo)} left to generate.")

    async def acall_one(item):
        clue_text = item[clue_key]
        solution_text = item[solution_key]

        requirements = [check(
            "The output must contain a valid JSON code block with the required keys.",
            validation_fn=simple_validate(
                lambda s: validate_rationale_response(s, required_keys=REQUIRED_KEYS)
            ),
        )]

        return await mfuncs.ainstruct(
            INSTRUCTION_RATIONALE,
            context=SimpleContext(),
            backend=backend,
            requirements=requirements,
            user_variables={"clue_text": clue_text, "solution_text": solution_text},
            icl_examples=[],
            strategy=RejectionSamplingStrategy(loop_budget=5),
            return_sampling_results=True,
        )

    print(f"Submitting {len(todo)} prompts async (batch_size={batch_size}, rate_limit={rate_limit}/min) ...")
    sem = asyncio.Semaphore(batch_size)
    interval = 60.0 / rate_limit
    ordered = [None] * len(todo)

    async def run_one(i, item):
        async with sem:
            try:
                ordered[i] = (await acall_one(item), item, None)
            except Exception as e:
                # Keep one bad item from taking down the whole gather.
                ordered[i] = (None, item, e)

    async def run_all():
        tasks = []
        for i, item in enumerate(todo):
            if i > 0:
                await asyncio.sleep(interval)
            tasks.append(asyncio.create_task(run_one(i, item)))
        await asyncio.gather(*tasks)

    if todo:
        asyncio.run(run_all())

    fresh: Dict[Tuple[str, str], Dict[str, Any]] = {}
    num_generated = 0
    for i, (output, item, error) in enumerate(ordered):
        clue = item[clue_key]
        solution = item[solution_key]

        record = {
            "clue": clue,
            "solution": solution,
            "rationale": "",
            "surface_reading": "",
            "intended_sense": "",
            "wordplay_type": "",
            "think": "",
            "status": "FAIL",
        }

        if error is not None:
            record["status"] = "ERROR"
            print(f"  [{i + 1}/{len(todo)}] [ERROR] {solution}: {type(error).__name__}: {error}")
        elif output is not None and output.success:
            text = str(output)
            record["think"] = get_think_tags(text)
            try:
                parsed = parse_rationale_response(text)
                for key in ("rationale", "surface_reading", "intended_sense", "wordplay_type"):
                    value = parsed.get(key, "")
                    record[key] = value if isinstance(value, str) else str(value)
                record["status"] = "OK" if record["rationale"].strip() else "PARSE-FAIL"
            except (json.JSONDecodeError, TypeError, AttributeError) as e:
                record["status"] = "PARSE-FAIL"
                print(f"  [{i + 1}/{len(todo)}] [PARSE-FAIL] {solution}: {e}")

        if record["rationale"].strip():
            num_generated += 1
            print(f"  [{i + 1}/{len(todo)}] [{record['status']}] {solution} <- {clue}")
            print(f"    rationale: {record['rationale'][:160]}")
        elif record["status"] == "FAIL":
            print(f"  [{i + 1}/{len(todo)}] [FAIL] {solution} <- {clue}")

        fresh[(clue, solution)] = record

    # Rebuild in dataset order, preferring a fresh record over a resumed one.
    results = []
    for item in data:
        key = (item.get(clue_key, ""), item.get(solution_key, ""))
        if key in fresh:
            results.append(fresh[key])
        elif key in done:
            results.append(done[key])

    write_results(results, output_filename)

    print(f"Finished {len(todo)} pairs ({num_generated} rationales generated).")
    print(f"Wrote {len(results)} records to {output_filename}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate LLM rationales explaining why a known solution "
                    "fits its Romanian crossword clue."
    )
    parser.add_argument('--model_id', type=str, default="claude-opus-4-8",
                        help='A model name from configs/rits_models.json or '
                             'configs/frontier_models.json, or a raw litellm model id '
                             'containing a "/" (e.g. openai/gpt-5.5). '
                             'Use --list_models to see what is configured.')
    parser.add_argument('--dataset_file', type=str,
                        help='Input JSON file (required unless --list_models)')
    parser.add_argument('--dataset_type', type=str, default="baseline",
                        choices=["clues", "baseline"],
                        help="'clues' reads the 'answer' field, 'baseline' reads 'solution'")
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--output_name', type=str, default="rationales")
    parser.add_argument('--output_dir', type=str, default=".")
    parser.add_argument('--batch_size', type=int, default=50, help='Max concurrent async requests')
    parser.add_argument('--rate_limit', type=int, default=1500, help='Max requests per minute')
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--to_csv', action='store_true',
                        help='Also write a CSV with an empty rationale_score column')
    parser.add_argument('--resume', action='store_true',
                        help='Skip pairs that already have a rationale in the output file')
    parser.add_argument('--summary_only', action='store_true',
                        help='Skip generation and only re-summarize an existing output file')

    parser.add_argument('--list_models', action='store_true',
                        help='List the models available from the config files and exit')

    args = parser.parse_args()

    if args.list_models:
        print_models()
        raise SystemExit(0)

    if not args.dataset_file:
        parser.error("--dataset_file is required (unless --list_models)")

    model_slug = args.model_id.replace("/", "-")
    output_base = f"{args.output_name}_{args.dataset_type}_{model_slug}"
    output_filename = os.path.join(args.output_dir, f"{output_base}.json")
    summary_filename = os.path.join(args.output_dir, f"{output_base}_summary.json")
    csv_filename = os.path.join(args.output_dir, f"{output_base}.csv")

    if args.summary_only:
        results = load_data(output_filename)
        if not results:
            raise SystemExit(f"Nothing to summarize in {output_filename}")
        summarize(results, args.model_id, summary_filename)
        if args.to_csv:
            write_csv(results, csv_filename)
        print("Done.")
        raise SystemExit(0)

    if args.output_dir and not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    try:
        backend = build_backend(args.model_id, max_tokens=args.max_tokens)
    except (RuntimeError, ValueError) as e:
        # Configuration problems (missing credentials, unknown model, RITS not
        # available) are user errors, not bugs: report them without a traceback.
        raise SystemExit(f"Configuration error: {e}")

    data = load_data(args.dataset_file)
    if not data:
        raise SystemExit(f"Could not load any data from {args.dataset_file}")

    results = process_data(
        data,
        backend,
        args.dataset_type,
        num_samples=args.num_samples,
        output_filename=output_filename,
        batch_size=args.batch_size,
        rate_limit=args.rate_limit,
        resume=args.resume,
    )

    summarize(results, args.model_id, summary_filename)

    if args.to_csv:
        write_csv(results, csv_filename)

    print("Done.")
