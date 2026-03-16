# Read the clues file and use an LLM to predict the answers. Use sbert to
# score the similarity between ground truth answer and prediction.

# Clue and answer pairs are in Romanian

import os
import asyncio
import argparse
import json
import numpy as np
import mellea.stdlib.functional as mfuncs

from dotenv import load_dotenv
from typing import Any, Dict, List
from mellea.backends import Backend
from mellea.backends import ModelOption
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.requirements import req, check, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

from mellea_ibm.rits import RITSBackend, RITS

# Local imports
from src.crosswords.utils import strip_code_fences, validate_json_code_block

INSTRUCTION_V1 = """
You are an expert at solving crossword puzzles. Given a clue in Romanian (i.e., a short definition) you will provide the answer to that clue.

Rules:
- The answer to the given clue must be in Romanian.
- The clue definition may have multiple meanings and you must select the most likely answer.
- Do not translate verbatim the clue definition in English.
- You must analyze carefuly the clue to determine the most likely meaning for your answer.
- Provide the rationale behind your answer.
- Format the response as a JSON object with the following structure.

```json
{
  "answer": "<your answer here>",
  "rationale": "<your rationale here>"
}
```

Use the following examples to learn your task better.

Example 1:
CLUE: Prins asupra faptului
ANSWER:
```json
{
  "answer": "OCUPAT",
  "rationale": "Prins asupra faptului inseamna ca persoana respectiva era ocupata cu o anumita activitate la momentul respectiv. Deci, OCUPAT este cel mai probabil raspuns."
}
```

Example 2:
CLUE: Marcată de o purtare abuzivă
ANSWER:
```json
{
  "answer": "ROASA",
  "rationale": "De exemplu, purtarea abuziva a unei perechi de pantofi implica a uzura pronuntata a pantofilor. Roasa inseamna uzat sau uzura."
}
```

Example 3:
CLUE: Unitate de morărit
ANSWER:
```json
{
  "answer": "SAC",
  "rationale": "Activitatea de morărit implica macinarea graului in faina. De obicei, faina rezultata este pusa intr-un SAC."
}
```

CLUE: {{clue_text}}
ANSWER:
"""

INSTRUCTION_V2 = """
You are an expert at solving crossword puzzles. Given a clue in Romanian (i.e., a short definition) you will provide the answer to that clue.
As a hint, you are given the number of letters that the answer must have.

Rules:
- The answer to the given clue must be in Romanian.
- The length of your answer must match the number given as hint.
- Do not use abbreviations.
- The clue definition may have multiple meanings and you must select the most likely answer.
- Do not translate verbatim the clue definition in English.
- You must analyze carefuly the clue to determine the most likely meaning for your answer.
- Provide the rationale behind your answer.
- Format the response as a JSON object with the following structure.

```json
{
  "answer": "<your answer here>",
  "rationale": "<your rationale here>"
}
```

Use the following examples to learn your task better.

Example 1:
CLUE: Prins asupra faptului
HINT: 6
ANSWER:
```json
{
  "answer": "OCUPAT",
  "rationale": "Prins asupra faptului inseamna ca persoana respectiva era ocupata cu o anumita activitate la momentul respectiv. Deci, OCUPAT este raspunsul corect pentru ca are 6 litere."
}
```

Example 2:
CLUE: Marcată de o purtare abuzivă
HINT: 5
ANSWER:
```json
{
  "answer": "ROASA",
  "rationale": "De exemplu, purtarea abuziva a unei perechi de pantofi implica a uzura pronuntata a pantofilor. ROASA inseamna uzat sau uzura. Mai mult, ROASA este raspunsul corect pentru ca are 5 litere."
}
```

Example 3:
CLUE: Unitate de morărit
HINT: 3
ANSWER:
```json
{
  "answer": "SAC",
  "rationale": "Activitatea de morărit implica macinarea graului in faina. De obicei, faina rezultata este pusa intr-un SAC iar cuvantul SAC are 3 litere."
}
```

CLUE: {{clue_text}}
HINT: {{num_letters}}
ANSWER:
"""

INSTRUCTION_V3 = """
You are an expert at solving crossword puzzles. Given a clue in Romanian (i.e., a short definition) you will provide the answer to that clue.
As a hint, you are given the number of letters that the answer must have and the letters the answer must start with.

Rules:
- The answer to the given clue must be in Romanian.
- The answer must have the number letters given as hint and must start with the letters given as hint.
- Do not use abbreviations.
- The clue definition may have multiple meanings and you must select the most likely answer.
- Do not translate verbatim the clue definition in English.
- You must analyze carefuly the clue to determine the most likely meaning for your answer.
- Provide the rationale behind your answer.
- Format the response as a JSON object with the following structure.

```json
{
  "answer": "<your answer here>",
  "rationale": "<your rationale here>"
}
```

Use the following examples to learn your task better.

Example 1:
CLUE: Prins asupra faptului
HINT: Raspunsul are 6 litere si incepe cu OC
ANSWER:
```json
{
  "answer": "OCUPAT",
  "rationale": "Prins asupra faptului inseamna ca persoana respectiva era ocupata cu o anumita activitate la momentul respectiv. Deci, OCUPAT este raspunsul corect pentru ca are 6 litere si incepe cu OC."
}
```

Example 2:
CLUE: Marcată de o purtare abuzivă
HINT: Raspunsul are 5 litere si incepe cu RO
ANSWER:
```json
{
  "answer": "ROASA",
  "rationale": "De exemplu, purtarea abuziva a unei perechi de pantofi implica a uzura pronuntata a pantofilor. ROASA inseamna uzat sau uzura. Mai mult, ROASA este raspunsul corect pentru ca are 5 litere si incepe cu RO."
}
```

Example 3:
CLUE: Unitate de morărit
HINT: Raspunsul are 3 litere si incepe cu SA
ANSWER:
```json
{
  "answer": "SAC",
  "rationale": "Activitatea de morărit implica macinarea graului in faina. De obicei, faina rezultata este pusa intr-un SAC iar cuvantul SAC are 3 litere si incepe cu SA."
}
```

CLUE: {{clue_text}}
HINT: Raspunsul are {{num_letters}} litere si incepe cu {{prefix_text}}
ANSWER:
"""


load_dotenv()

def get_sbert(reference: str, prediction: str, scorer) -> float:
    _, _, F1 = scorer.score([prediction], [reference])
    sbert = F1.cpu().detach().numpy().tolist()[0]
    return sbert

def load_data(file_name: str) -> List[Dict[str, Any]]:
    """
    Load the dataset
    
    """

    try:
        with open(file_name, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(e)
        return None
    
async def process_data(
        data: List[Dict[str, Any]], 
        backend: Backend, 
        version: str, 
        prefix_len: int,
        dataset_type: str,
        num_samples: int = None):
    """
    Process the dataset
    
    :param data: Description
    :type data: List[Dict[str, Any]]
    :param backend: Description
    :type backend: Backend
    """

    print(f"Processing {len(data)} clues...")
    print(f"Using LLM: {backend.model_id}")
    print(f"Prompt version: {version}")
    print(f"Dataset type: {dataset_type}")

    assert dataset_type in ["clues", "baseline"], f"Unknown dataset type: {dataset_type}"

    results = []
    references = []
    predictions = []
    corutines = []
    if num_samples is not None:
        data = data[:num_samples]

    for item in data:
        if dataset_type == "clues":
            answer = item["answer"] # ground truth answer (Romanian)
            clue_text = item["clue"] # clue text (Romanian)
        elif dataset_type == "baseline":
            answer = item["solution"] # ground truth answer (Romanian)
            clue_text = item["clue"] # clue text (Romanian)

        num_letters = len(answer)
        if num_letters <= 2:
            continue # skip very short answers for which prefix hint is not relevant
        prefix_text = answer[:prefix_len] if len(answer) > prefix_len else answer

        if version == "v1":
            instruction = INSTRUCTION_V1
            user_variables = {"clue_text": clue_text}
        elif version == "v2":
            instruction = INSTRUCTION_V2
            user_variables = {"clue_text": clue_text, "num_letters": num_letters}
        elif version == "v3":
            instruction = INSTRUCTION_V3
            user_variables = {"clue_text": clue_text, "num_letters": num_letters, "prefix_text": prefix_text}

        # Perform the instruction with validation
        corutine = mfuncs.ainstruct(
            instruction,
            context=SimpleContext(),
            backend=backend,
            requirements=[
                check(
                    "The output must be a valid JSON dictionary with markdown code fences.",
                    validation_fn=simple_validate(
                        lambda s: validate_json_code_block(s, required_keys=["answer", "rationale"])
                    ),
                )
            ],
            user_variables=user_variables,
            icl_examples=[],
            strategy=RejectionSamplingStrategy(loop_budget=5),
            return_sampling_results=True,
        )
        corutines.append(corutine)
    print(f"[Atomizer] Awaiting for the async execution ...")
    outputs = await asyncio.gather(*(corutines[i] for i in range(len(corutines))))        
    for i, output in enumerate(outputs):
        if output.success:
            answer = data[i]["answer"]
            clue = data[i]["clue"]
            cleaned = strip_code_fences(str(output))
            pred_dict = json.loads(cleaned)
            prediction = pred_dict["answer"]
            rationale = pred_dict["rationale"]
            references.append(answer)
            predictions.append(prediction)
            results.append({
                "clue": clue,
                "answer": answer,
                "prediction": prediction,
                "rationale": rationale,
            })
    
    print(f"Finished {len(data)} data points")
    print(f"Generated {len(predictions)} answers to {len(references)} questions.")
    print(f"Computing scores: exact match and sbert.")

    from bert_score import BERTScorer
    scorer = BERTScorer(model_type='bert-base-uncased', device='cpu')

    exact_matches = [1.0 if ref.lower() == pred.lower() else 0.0 for ref, pred in zip(references, predictions)]
    sbert_scores = [get_sbert(ref, pred, scorer) for ref, pred in zip(references, predictions)]
    
    results.append({"exact_matches": float(np.sum(exact_matches)), "sbert_mean": float(np.mean(sbert_scores)), "sbert_std": float(np.std(sbert_scores))})
    print(f"Exact match: {np.sum(exact_matches)}")
    print(f"SBERT score: {np.mean(sbert_scores):.4f} / {np.std(sbert_scores):.4f}")

    return results

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default="llama")
    parser.add_argument('--dataset_file', type=str)
    parser.add_argument('--dataset_type', type=str, default="clues")
    parser.add_argument('--version', type=str, default="v3")
    parser.add_argument('--prefix_len', type=int, default=0)
    parser.add_argument('--output_name', type=str)
    parser.add_argument('--output_dir', type=str)

    args = parser.parse_args()

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

    prefix_len = args.prefix_len
    dataset_type = args.dataset_type
    data = load_data(args.dataset_file)
    results = asyncio.run(process_data(data, backend, args.version, prefix_len, dataset_type))

    output_filename = f"{args.output_name}_{args.model_id}_{args.version}_{prefix_len}.json"
    with open(os.path.join(args.output_dir, output_filename), "w") as fp:
        json.dump(results, fp, indent=4)

    print("Done.")
