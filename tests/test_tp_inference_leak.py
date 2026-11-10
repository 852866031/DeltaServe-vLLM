#!/usr/bin/env python
"""TP=2 inference-correctness gate: FT co-serving must not perturb real requests.

Runs tests/test_phase1_m1.py's leak check on Qwen3-14B with tensor parallelism 2
and the production preset's FT config unchanged (configs/preset/serving_config_finetuning_
qwen3_14b_tp2.yaml: has_ft CUDA graphs, FT-only graphs, rank-symmetric tier-C
aborts, pause_on_arrival, every activation save, backward CUDA graphs). Same
method: an amplified FT adapter, a positive control, staggered arrivals so FT
provably co-batches with real prefills and decodes, drift + near-tie checks.

    python tests/test_tp_inference_leak.py      # 2 GPUs, ~10 min

Needs both GPUs free and the Qwen3-14B-Base weights in HF_HOME.
"""

import os
import sys
from pathlib import Path

os.environ["DSERVE_M1_PROFILE"] = "tp2"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_phase1_m1  # noqa: E402

if __name__ == "__main__":
    test_phase1_m1.main()
