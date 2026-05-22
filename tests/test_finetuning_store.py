#!/usr/bin/env python
"""Test the finetuning sample store (vllm.deltaserve.finetuning_store).

CPU-only: loads the real alpaca_1000.txt corpus with the opt-125m tokenizer and
checks load + length-bucketed selection + epoch marking. No GPU.

    python tests/test_finetuning_store.py

Run as a file (not `python -c`) so `vllm` resolves to the installed package.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/mnt/storage/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

_REPO = Path(__file__).resolve().parent.parent
_CORPUS = _REPO / "alpaca_1000.txt"

_passed = 0
_failed = 0
_TTY = sys.stdout.isatty()
_GREEN = "\033[92m" if _TTY else ""
_RED = "\033[91m" if _TTY else ""
_RESET = "\033[0m" if _TTY else ""


def green(msg):
    print(f"{_GREEN}{msg}{_RESET}")


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  {_GREEN}[PASS]{_RESET} {name}")
    else:
        _failed += 1
        print(f"  {_RED}[FAIL]{_RESET} {name}")


def main():
    green("=== Finetuning sample store ===")
    from transformers import AutoTokenizer

    from vllm.deltaserve.finetuning_store import FinetuningStore

    tok = AutoTokenizer.from_pretrained("facebook/opt-125m")
    tokenize = lambda text: tok(text)["input_ids"]  # noqa: E731

    green("load:")
    store = FinetuningStore(str(_CORPUS), tokenize, adapter="ft",
                            total_epochs=2, max_saved_finetuning_tokens=256)
    n = store.load()
    check("loaded 1000 samples", n == 1000 and len(store) == 1000)
    check("total tokens > 0", store.total_tokens_in_memory > 0)
    check("samples carry the adapter tag", store.samples[0].adapter == "ft")
    check("input_len matches token count",
          store.samples[0].input_len == len(store.samples[0].prompt_ids))

    green("pop_best_under (largest untrained <= max_tokens):")
    budget = 64
    s = store.pop_best_under(budget)
    expected_len = max((L for L in store.sorted_lengths if L <= budget), default=None)
    check("returns a sample within budget",
          s is not None and s.input_len <= budget)
    check("returns the LARGEST length <= budget",
          s is not None and s.input_len == expected_len)
    check("peek does not mark trained (same sample again)",
          store.pop_best_under(budget).request_id == s.request_id)
    check("exclude skips the given sample",
          store.pop_best_under(budget, exclude=[s]).request_id != s.request_id)
    too_small = store.sorted_lengths[0] - 1
    check("returns None when budget below the shortest sample",
          store.pop_best_under(too_small) is None)

    green("confirmed_trained marks + removes from buckets:")
    before = sum(not t for t in store.trained)
    marked = store.confirmed_trained([s])
    after = sum(not t for t in store.trained)
    check("marked exactly one", marked == 1 and before - after == 1)
    check("trained sample no longer returned",
          store.pop_best_under(budget, exclude=[]).request_id != s.request_id)

    green("epochs (total_epochs=2 -> two advances succeed, third fails):")
    # exhaust by marking everything, then advance_epoch resets
    store.confirmed_trained(list(store.samples))
    check("store empty after marking all", not store.has_next())
    check("advance_epoch #1 -> True (epoch 0->1)", store.advance_epoch() is True)
    check("store refilled after advance_epoch", store.has_next())
    check("advance_epoch #2 -> True (epoch 1->2)", store.advance_epoch() is True)
    check("advance_epoch #3 -> False (past total_epochs)",
          store.advance_epoch() is False)

    green("max_prepare cap:")
    capped = FinetuningStore(str(_CORPUS), tokenize, max_prepare=10)
    check("max_prepare limits load", capped.load() == 10)

    color = _RED if _failed else _GREEN
    print(f"\n{color}{_passed} passed, {_failed} failed{_RESET}")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
