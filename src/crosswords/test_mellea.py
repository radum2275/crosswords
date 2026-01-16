import re
import json

import mellea.stdlib.functional as mfuncs

from typing import Any, Dict
from mellea.backends import Backend
from mellea.backends.types import ModelOption
from mellea.stdlib.base import SimpleContext, CBlock
from mellea.stdlib.requirement import req, check, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

from mellea_ibm.rits import RITSBackend, RITS

INSTRUCTION = """
You are an expert at solving crossword puzzles. Given a definition you will provide the clue.
Format the response as a JSON object with the following structure:
```json
{
  "clue": <your clue here>,
  "rationale": <your rationale here>
}
```
"""

# Create a Mellea RITS backend
backend = RITSBackend(
    RITS.LLAMA_3_3_70B_INSTRUCT, model_options={ModelOption.MAX_NEW_TOKENS: 500}
)

model_output = m.instruct(
    PROMPT,
    requirements=[
        check(
            "The output must be a valid JSON object containing an 'atomic_units' key.",
            validation_fn=simple_validate(lambda x: validate_json_code_block(x))
        ),
    ],
    user_variables={"response": response},
    strategy=RejectionSamplingStrategy(loop_budget=3),
    return_sampling_results=True
)

print("***" * 20)
print(model_output)
print("***" * 20)
print(str(model_output))

