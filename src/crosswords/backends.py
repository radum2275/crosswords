# Shared model/backend construction for every script in this package.
#
# Models are defined in JSON config files rather than hardcoded here, so models
# can be added or retired by editing config instead of code. Both files hold a
# list of {"name", "model_id", "backend", ...} records:
#
#   configs/rits_models.json      -- RITS models, each with an explicit base_url
#   configs/frontier_models.json  -- models served by the IBM litellm gateway
#
# "name" is what a script's --model_id takes; a name containing a "/" that
# matches neither config is passed through verbatim as a raw litellm model id.

import os
import re
import json

from pathlib import Path
from typing import Any, Dict, Tuple, Union

from mellea.backends import Backend
from mellea.backends import ModelOption
from mellea.backends.litellm import LiteLLMBackend

# RITS lives in mellea_ibm, an IBM-internal package that is not on PyPI. Guard
# the import so the litellm path stays usable outside IBM infrastructure.
try:
    from mellea_ibm.rits import RITSBackend
    _HAS_RITS = True
except ImportError:
    _HAS_RITS = False

# Known litellm/aiohttp interaction; see
# https://github.com/BerriAI/litellm/issues/13251
os.environ.setdefault("DISABLE_AIOHTTP_TRANSPORT", "True")

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
RITS_CONFIG = CONFIG_DIR / "rits_models.json"
FRONTIER_CONFIG = CONFIG_DIR / "frontier_models.json"

_TRAILING_COMMAS_RE = re.compile(r",(\s*[}\]])")


def load_json_utf8_relaxed(path: Union[str, "os.PathLike"]) -> Any:
    """
    Load a JSON file containing Unicode (e.g., Romanian diacritics) reliably.
    Also tolerates a common non-standard JSON issue: trailing commas before
    } or ].
    """
    # utf-8-sig removes BOM if present, while still handling normal UTF-8
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        text = f.read()

    # Remove trailing commas before closing braces/brackets
    text = _TRAILING_COMMAS_RE.sub(r"\1", text)

    return json.loads(text)


def load_model_config(path: Union[str, "os.PathLike"]) -> Dict[str, Dict[str, Any]]:
    """
    Load a model config file into a {name: record} dict.

    A missing file is not fatal: it just means that family of models is
    unavailable, so the other backend still works. Returns {} in that case.
    """
    if not os.path.exists(path):
        return {}
    try:
        records = load_json_utf8_relaxed(path)
    except Exception as e:
        raise RuntimeError(f"Could not parse model config {path}: {e}")

    if not isinstance(records, list):
        raise RuntimeError(f"Model config {path} must contain a JSON list.")

    models: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise RuntimeError(f"Model config {path} has a non-object entry.")
        for field in ("name", "model_id"):
            if not str(record.get(field, "")).strip():
                raise RuntimeError(
                    f"Model config {path} has an entry missing '{field}': {record}"
                )
        models[record["name"]] = record
    return models


def available_models() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Return the (rits, frontier) model config dicts."""
    return load_model_config(RITS_CONFIG), load_model_config(FRONTIER_CONFIG)


def print_models() -> None:
    """Print the configured models. Used by every script's --list_models."""
    rits_models, frontier_models = available_models()

    print(f"RITS models ({RITS_CONFIG}):")
    if rits_models:
        for name, record in rits_models.items():
            flag = "" if _HAS_RITS else "   [mellea_ibm not installed]"
            print(f"  {name:24s} {record['model_id']}{flag}")
    else:
        print("  (none configured)")

    print(f"\nFrontier models via litellm gateway ({FRONTIER_CONFIG}):")
    if frontier_models:
        for name, record in frontier_models.items():
            print(f"  {name:24s} {record['model_id']:44s} {record.get('provider', '')}")
    else:
        print("  (none configured)")


def build_backend(model_id: str, max_tokens: int = 4096) -> Backend:
    """
    Build a mellea backend for the given model name.

    The name is looked up in configs/rits_models.json first, then in
    configs/frontier_models.json. A name containing a "/" that matches neither
    is passed through verbatim as a raw litellm model id, so any model on the
    gateway can be used without touching config.
    """

    rits_models, frontier_models = available_models()
    model_options: Dict[Any, Any] = {ModelOption.MAX_NEW_TOKENS: max_tokens}

    # --- RITS path -----------------------------------------------------------
    if model_id in rits_models:
        record = rits_models[model_id]
        if not _HAS_RITS:
            raise RuntimeError(
                f"--model_id '{model_id}' is a RITS model, but the 'mellea_ibm' "
                "package is not installed. It is IBM-internal and not on PyPI, "
                "so it is only importable on IBM infrastructure. Install it, or "
                "use a frontier model instead (e.g. --model_id claude-opus-4-8)."
            )
        base_url = str(record.get("base_url", "")).strip()
        if not base_url:
            raise RuntimeError(
                f"RITS model '{model_id}' has no 'base_url' in {RITS_CONFIG}."
            )
        if "RITS_API_KEY" not in os.environ:
            raise RuntimeError(
                "RITS_API_KEY is not set. It is the RITS inference key. "
                "Add it to your .env file."
            )
        # RITSBackend appends "/v1" to the endpoint itself, so pass the bare
        # base_url from config.
        return RITSBackend(
            record["model_id"],
            endpoint=base_url.rstrip("/"),
            model_options=model_options,
        )

    # --- litellm gateway path ------------------------------------------------
    if model_id in frontier_models:
        litellm_model = frontier_models[model_id]["model_id"]
    elif "/" in model_id:
        litellm_model = model_id
    else:
        known = sorted(rits_models) + sorted(frontier_models)
        raise ValueError(
            f"Unknown --model_id '{model_id}'. Known models: "
            f"{', '.join(known) if known else '(no config files found)'}. "
            "You can also pass a raw litellm model id containing a '/' "
            "(e.g. openai/gpt-5.5). Use --list_models to see the full list."
        )

    base_url = os.getenv("ANTHROPIC_BASE_URL")
    api_key = os.getenv("ANTHROPIC_AUTH_TOKEN")
    if not base_url:
        raise RuntimeError(
            "ANTHROPIC_BASE_URL is not set. It must point at the litellm "
            "gateway. Add it to your .env file."
        )
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_AUTH_TOKEN is not set. It is the litellm gateway API "
            "key. Add it to your .env file."
        )

    # NOTE: api_base/api_key must be passed via model_options, NOT via the
    # base_url constructor argument -- LiteLLMBackend stores base_url but never
    # forwards it to the completion call. Mellea logs a warning that it "may
    # drop" these two keys; that is a false positive, litellm does receive them.
    model_options["api_base"] = base_url.rstrip("/")
    model_options["api_key"] = api_key

    return LiteLLMBackend(model_id=litellm_model, model_options=model_options)
