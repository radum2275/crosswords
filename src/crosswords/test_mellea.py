import re
import json

import mellea.stdlib.functional as mfuncs

from dotenv import load_dotenv
from typing import Any, Dict
from mellea.backends import Backend
from mellea.backends.types import ModelOption
from mellea.stdlib.base import SimpleContext, CBlock
from mellea.stdlib.requirement import req, check, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

from mellea_ibm.rits import RITSBackend, RITS

# Local imports
from src.crosswords.utils import strip_code_fences, validate_json_code_block

load_dotenv()

INSTRUCTION = """
You are an expert at solving crossword puzzles. Given a definition you will provide the clue.
Format the response as a JSON object with the following structure:
```json
{
  "clue": <your clue here>,
  "rationale": <your rationale here>
}
```

DEFINITION: {{definition}}
Your answer:
"""

icl_examples = []

definition = "A domesticated carnivorous mammal (Canis familiaris syn. Canis lupus subsp. familiaris) typically kept as a pet or for work or field sports."

# Create a Mellea RITS backend
backend = RITSBackend(
    RITS.LLAMA_3_3_70B_INSTRUCT, model_options={ModelOption.MAX_NEW_TOKENS: 500}
)
# Perform the instruction with validation
output = mfuncs.instruct(
    INSTRUCTION,
    context=SimpleContext(),
    backend=backend,
    requirements=[
        check(
            "The output must be a valid JSON dictionary with markdown code fences.",
            validation_fn=simple_validate(
                lambda s: validate_json_code_block(s, required_keys=["clue", "rationale"])
            ),
        )
    ],
    user_variables={"definition": definition},
    icl_examples=icl_examples,
    strategy=RejectionSamplingStrategy(loop_budget=3),
    return_sampling_results=True,
)

# The output is a validated JSON string; parse it
if output.success:
    cleaned = strip_code_fences(str(output))
    print(cleaned)
else:
    print(f"Failure!!!") 

print("Done.")

