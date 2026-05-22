#!/usr/bin/env python
"""Milestone 2 verification: per-token FT activation capture, hash-verified.

Runs the FT engine (reusing test_phase1_m1's ft worker, which injects FT samples
and generates) and asserts, from the engine's logs, that:
  - the capture hooks + buffers were set up and shared with the backward process,
  - the captured FT activations (per-layer attn_out/ffn_out first+last, final
    hidden states, target ids) hash-match between the worker and the backward
    process (zero-copy proof): "[capture] activation hashes parent==child: True",
  - no hash mismatch (weights or activations) occurred.

Pairs with test_phase1_m1.py (which separately proves inference is unchanged).

    python tests/test_phase1_m2.py

Run on the 5090 (uses VLLM_USE_FLASHINFER_SAMPLER=0).
"""

import os
import subprocess
import sys
from pathlib import Path

_M1 = Path(__file__).resolve().parent / "test_phase1_m1.py"

_TTY = sys.stdout.isatty()
_GREEN = "\033[92m" if _TTY else ""
_RED = "\033[91m" if _TTY else ""
_RESET = "\033[0m" if _TTY else ""


def main():
    env = dict(os.environ)
    env.setdefault("HF_HOME", "/mnt/storage/huggingface")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    proc = subprocess.run(
        [sys.executable, str(_M1), "--worker", "ft"],
        capture_output=True, text=True, env=env, timeout=600,
    )
    out = proc.stdout + proc.stderr

    passed = failed = 0

    def check(name, cond):
        nonlocal passed, failed
        mark = f"{_GREEN}[PASS]{_RESET}" if cond else f"{_RED}[FAIL]{_RESET}"
        print(f"  {mark} {name}")
        if cond:
            passed += 1
        else:
            failed += 1

    print(f"{_GREEN}=== Milestone 2: FT activation accumulation (cross-process hash) ==={_RESET}")
    check("accumulation hooks registered + buffers allocated",
          "[accumulate] hooks on" in out)
    check("activation buffers shared with backward process",
          "activations_received" in out)
    check("FT accumulation ran (buffer filled at least once)",
          "[coord] buffer FULL" in out)
    check("activation hashes parent==child: True",
          "[accumulate] activation hashes parent==child: True" in out)
    check("no hash mismatch anywhere",
          "parent==child: False" not in out)

    if failed:
        # Surface a tail of the log to help debug.
        print("\n--- engine log tail ---")
        print("\n".join(out.splitlines()[-25:]))

    color = _RED if failed else _GREEN
    print(f"\n{color}{passed} passed, {failed} failed{_RESET}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
