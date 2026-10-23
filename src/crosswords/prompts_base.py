# Few-shot completion prompts for *base* (non-instruction-tuned) models.
#
# A base model cannot follow the JSON-producing instruction prompts in
# clues_ro.py, so each hint condition is expressed as a pattern to continue:
# a short task line, the same three worked examples the instruction prompts use
# (OCUPAT / ROASA / SAC), then the target clue ending in "Răspuns:". The model's
# completion up to the first newline is the answer.
#
# Only the hint conditions that do not rely on following instructions are
# defined: v1 (clue only), v2 (+ length) and v3 (+ length and first k letters).
# v4 (chain-of-thought) and v5 (reason in English) have no base-model analogue.


BASE_VERSIONS = ("v1", "v2", "v3")

HEADER = "Rebus: pentru fiecare definiție, scrie cuvântul pe care îl definește."

# (clue, answer) -- the same examples as INSTRUCTION_V1..V3 in clues_ro.py.
EXAMPLES: list[tuple[str, str]] = [
    ("Prins asupra faptului", "OCUPAT"),
    ("Marcată de o purtare abuzivă", "ROASA"),
    ("Unitate de morărit", "SAC"),
]


def prefix_hint(answer: str, prefix_len: int) -> str:
    """The first prefix_len letters; the whole answer if it is not longer.

    Mirrors clues_ro.py, so a short answer is revealed in full exactly as it was
    for the instruction-tuned models.
    """
    return answer[:prefix_len] if len(answer) > prefix_len else answer


def hint_line(version: str, answer: str, prefix_len: int) -> str:
    # Same wording as the HINT lines of INSTRUCTION_V2/V3.
    if version == "v1":
        return ""
    if version == "v2":
        return f"Indiciu: Răspunsul are {len(answer)} litere\n"
    if version == "v3":
        return (f"Indiciu: Răspunsul are {len(answer)} litere și începe cu "
                f"{prefix_hint(answer, prefix_len).upper()}\n")
    raise ValueError(f"Version {version} has no base-model prompt (use one of {BASE_VERSIONS}).")


def block(clue: str, answer: str, version: str, prefix_len: int, with_answer: bool) -> str:
    text = f"Definiție: {clue}\n{hint_line(version, answer, prefix_len)}Răspuns:"
    return f"{text} {answer.upper()}\n" if with_answer else text


def build_prompt(version: str, clue: str, answer: str, prefix_len: int) -> str:
    """The full few-shot prompt for one item. `answer` is only used for the hint."""
    shots = "\n".join(block(c, a, version, prefix_len, with_answer=True) for c, a in EXAMPLES)
    return f"{HEADER}\n\n{shots}\n{block(clue, answer, version, prefix_len, with_answer=False)}"


def parse_completion(text: str) -> str:
    """The answer is the completion up to the first newline, stripped of punctuation."""
    line = text.split("\n", 1)[0]
    return line.strip().strip(".,;:!?\"'()[]").strip()
