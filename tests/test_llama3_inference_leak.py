#!/usr/bin/env python
"""Llama-3-8B inference-correctness gate: FT co-serving must not perturb real
requests on the second model family either.

Runs tests/test_phase1_m1.py's leak check (profile ``llama3``) on
meta-llama/Meta-Llama-3-8B, one GPU: an amplified FT adapter, a positive
control, staggered arrivals so FT provably co-batches with real prefills and
decodes, eager and CUDA-graph modes, admission phase ``prefill`` and ``both``.
It also checks that the mixed steps of the graph modes replayed the ``has_ft``
graphs (Llama-3 saves post-RoPE q/k/v; the same hook-issued save op).

    python tests/test_llama3_inference_leak.py      # 1 GPU, ~10 min

Needs the Meta-Llama-3-8B weights in HF_HOME.
"""

import os
import sys
from pathlib import Path

os.environ["DSERVE_M1_PROFILE"] = "llama3"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_phase1_m1  # noqa: E402

if __name__ == "__main__":
    test_phase1_m1.main()
