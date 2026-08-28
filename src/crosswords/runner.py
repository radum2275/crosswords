# Shared async driver, rate limiting and dataset-key handling for the
# generation scripts in this package.
#
# Every script here does the same three things: work out which fields of a
# dataset hold the clue and the answer, fire many LLM calls concurrently, and
# keep the total call rate under a budget. That logic lives here once instead
# of being copy-pasted per script.

import time
import asyncio

from collections import deque
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from mellea.backends import Backend


class RateLimiter:
    """
    Sliding-window rate limiter: at most `rate` acquisitions per `per` seconds.

    A deque of acquisition timestamps is trimmed to the trailing window on each
    attempt. When the window is full the caller sleeps until the oldest entry
    ages out. Unlike spacing submissions by a fixed interval, this bounds the
    number of calls in *any* rolling window, which is what an API quota
    actually measures.
    """

    def __init__(self, rate: int, per: float = 60.0):
        assert rate > 0, "rate must be positive"
        self.rate = rate
        self.per = per
        self._times: deque = deque()
        self._lock = asyncio.Lock()
        self.total = 0  # total acquisitions granted, for reporting/tests

    async def acquire(self) -> None:
        """Block until a slot is free in the trailing window, then take it."""
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._times and now - self._times[0] >= self.per:
                    self._times.popleft()
                if len(self._times) < self.rate:
                    self._times.append(now)
                    self.total += 1
                    return
                # Window is full: wait for the oldest acquisition to age out.
                wait = self.per - (now - self._times[0])
            # Sleep outside the lock so other tasks can make progress. Cap the
            # nap so a task cannot oversleep past a freed slot.
            await asyncio.sleep(min(max(wait, 0.0), 0.05))


def throttle_backend(backend: Backend, limiter: RateLimiter) -> Backend:
    """
    Wrap a backend so every LLM call acquires from `limiter` first.

    This patches `generate_from_context`, which is the single point every real
    request passes through -- including the retries that
    RejectionSamplingStrategy issues inside its loop_budget loop. Throttling
    around the higher-level `ainstruct` call would count one item as one call
    and miss up to loop_budget-1 additional requests per item, letting actual
    traffic exceed the budget several times over.

    The backend is mutated in place and returned for convenience.
    """
    original = backend.generate_from_context

    async def limited(*args, **kwargs):
        await limiter.acquire()
        return await original(*args, **kwargs)

    backend.generate_from_context = limited  # type: ignore[method-assign]
    backend._rate_limiter = limiter  # type: ignore[attr-defined]
    return backend


async def _gather_throttled(
    items: Sequence[Any],
    call_fn: Callable[[Any], Awaitable[Any]],
    batch_size: int,
) -> List[Tuple[Any, Any, Optional[BaseException]]]:
    """Run call_fn over items with bounded concurrency, preserving order."""
    sem = asyncio.Semaphore(batch_size)
    ordered: List[Any] = [None] * len(items)

    async def run_one(i: int, item: Any) -> None:
        async with sem:
            try:
                ordered[i] = (await call_fn(item), item, None)
            except Exception as e:
                # Keep one bad item from taking down the whole run. Callers get
                # the exception and decide how to record it.
                ordered[i] = (None, item, e)

    await asyncio.gather(*(asyncio.create_task(run_one(i, it))
                           for i, it in enumerate(items)))
    return ordered


def run_throttled(
    items: Sequence[Any],
    call_fn: Callable[[Any], Awaitable[Any]],
    batch_size: int = 50,
    verbose: bool = True,
) -> List[Tuple[Any, Any, Optional[BaseException]]]:
    """
    Run `call_fn` over `items` concurrently and return a list of
    (result, item, error) triples in input order.

    Pacing is NOT done here: throttle the backend with `throttle_backend` so
    that retries are counted too. `batch_size` caps how many requests are in
    flight at once.
    """
    if not items:
        return []
    if verbose:
        print(f"Submitting {len(items)} prompts async (batch_size={batch_size}) ...")
    return asyncio.run(_gather_throttled(items, call_fn, batch_size))


# --- dataset field handling -------------------------------------------------

# The two benchmark datasets name the answer field differently:
#   polycross_data.json / extracted_data.json  -> "answer"
#   themcross_data.json / baseline-dataset.json -> "solution"
# The dataset_type flag records which corpus a run used (it shapes output file
# names), and these aliases let it match either the old or the new file names.
DATASET_TYPE_ALIASES = {
    "clues": "answer",
    "polycross": "answer",
    "roco": "answer",
    "baseline": "solution",
    "themcross": "solution",
    "base": "solution",
}


def resolve_keys(
    data: List[Dict[str, Any]],
    dataset_type: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Work out which field holds the answer and which holds the clue.

    `dataset_type` states the expected field, but if that field is absent from
    the data we fall back to whichever of "answer"/"solution" is present, so
    any of the benchmark files works regardless of the flag.

    Returns (answer_key, clue_key). Raises KeyError if neither is found.
    """
    sample = data[0] if data else {}
    preferred = DATASET_TYPE_ALIASES.get(dataset_type or "", "")

    if preferred and preferred in sample:
        answer_key = preferred
    elif "answer" in sample:
        answer_key = "answer"
    elif "solution" in sample:
        answer_key = "solution"
    else:
        raise KeyError(
            "Dataset records have neither an 'answer' nor a 'solution' field "
            f"(found: {sorted(sample.keys())})."
        )

    if "clue" not in sample:
        raise KeyError(
            f"Dataset records have no 'clue' field (found: {sorted(sample.keys())})."
        )

    return answer_key, "clue"


def filter_and_limit(
    data: List[Dict[str, Any]],
    answer_key: str,
    num_samples: Optional[int] = None,
    min_answer_len: int = 3,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Drop answers shorter than `min_answer_len`, then truncate to `num_samples`.

    The order matters: filtering first means --num_samples N yields exactly N
    items. Doing it the other way around silently returns fewer.
    """
    if verbose:
        print(f"Initial number of clues: {len(data)}")

    data = [item for item in data if len(str(item.get(answer_key, ""))) >= min_answer_len]
    if verbose:
        print(f"After filtering out short answers, {len(data)} clues remain.")

    if num_samples is not None:
        data = data[:num_samples]
        if verbose:
            print(f"Limited to {len(data)} clues (num_samples={num_samples}).")

    return data
