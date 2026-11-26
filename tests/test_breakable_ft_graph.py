#!/usr/bin/env python
"""FT batches on vLLM's breakable CUDA graph (VLLM_USE_BREAKABLE_CUDAGRAPH=1).

The breakable graph records the forward as it runs instead of splitting a
torch.compile'd graph, so the activation save reaches the ``has_ft`` graphs
without Dynamo: the hooks' save op is executed during the capture pass. This
runs the inference gate of tests/test_phase1_m1.py (Qwen3-0.6B) in that mode
and checks, on top of it, that the mixed steps of the graph modes replayed a
graph.

    python tests/test_breakable_ft_graph.py      # 1 GPU, ~5 min

vLLM 0.34+ (the first release with the breakable graph on the v2 runner).
"""

import os
import sys
from pathlib import Path

os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"
os.environ["DSERVE_M1_EXPECT_GRAPH"] = "1"
os.environ.setdefault("DSERVE_M1_PROFILE", "tp1")
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _supported() -> bool:
    try:
        import vllm.compilation.breakable_cudagraph  # noqa: F401
        from vllm.v1.worker.gpu import cudagraph_utils
    except Exception:  # noqa: BLE001
        return False
    return "use_breakable_cg" in open(cudagraph_utils.__file__).read()


if __name__ == "__main__":
    if not _supported():
        print("skipped: this vLLM has no breakable CUDA graph on the v2 runner")
        print("0 passed, 0 failed")
        sys.exit(0)
    import test_phase1_m1  # noqa: E402

    test_phase1_m1.main()
