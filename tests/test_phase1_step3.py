#!/usr/bin/env python
"""Phase 1 / Step 3 test: share GPU weights with the backward process (CUDA IPC).

Requires a GPU. Proves the plan's #1 risk is solved: CUDA tensors sent over the
torch.multiprocessing pipe are shared zero-copy with the spawned child, not
copied. Uses self-allocated tensor dicts (independent of vLLM), so it isolates
the IPC mechanism.

Checks:
  - the child receives the right tensor counts / element counts,
  - the child's fp64 checksum of the FT tensors matches the parent's,
  - after the parent mutates a shared tensor IN PLACE, the child's recomputed
    checksum reflects the change (=> same memory, not a copy).

    python tests/test_phase1_step3.py
"""

import sys

import torch

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


def _checksum(state):
    return sum(float(t.double().sum().item()) for t in state.values())


def main():
    green("=== Phase 1 / Step 3: share GPU weights via CUDA IPC ===")
    if not torch.cuda.is_available():
        print(f"{_RED}CUDA not available — this test needs a GPU.{_RESET}")
        sys.exit(1)

    from vllm.deltaserve.backward_process import BackwardProcess

    dev = "cuda:0"
    torch.manual_seed(0)
    # Mimic base weights (a few tensors) and a small FT-adapter dict.
    base_state = {f"layer{i}.weight": torch.randn(256, 256, device=dev)
                  for i in range(4)}
    ft_state = {f"lora_{k}": torch.randn(16, 256, device=dev)
                for k in ("A", "B")}

    want_base_numel = sum(t.numel() for t in base_state.values())
    want_ft_numel = sum(t.numel() for t in ft_state.values())
    parent_ft_checksum = _checksum(ft_state)

    bp = BackwardProcess(mps_percentage=10, device_index=0)
    bp.start()

    green("share + summary:")
    summary = bp.share_weights(base_state, ft_state)
    check("event=weights_received", summary.get("event") == "weights_received")
    check("base tensor count matches", summary.get("base_num") == len(base_state))
    check("base numel matches", summary.get("base_numel") == want_base_numel)
    check("ft tensor count matches", summary.get("ft_num") == len(ft_state))
    check("ft numel matches", summary.get("ft_numel") == want_ft_numel)
    check("ft checksum matches parent",
          abs(summary.get("ft_checksum", 0.0) - parent_ft_checksum) < 1e-3)

    green("zero-copy proof (mutate in parent, re-read in child):")
    # Mutate one element of a shared FT tensor in place, in the PARENT.
    ft_state["lora_A"].view(-1)[0] += 5.0
    torch.cuda.synchronize()  # ensure the write is visible cross-process
    parent_ft_checksum_after = _checksum(ft_state)
    child_ft_checksum_after = bp.checksum("ft")
    check("parent checksum changed by +5.0",
          abs((parent_ft_checksum_after - parent_ft_checksum) - 5.0) < 1e-3)
    check("child sees the mutation (shared memory, not a copy)",
          abs(child_ft_checksum_after - parent_ft_checksum_after) < 1e-3)

    green("clean shutdown:")
    exitcode = bp.shutdown()
    check("child exited (exitcode == 0)", exitcode == 0)

    color = _RED if _failed else _GREEN
    print(f"\n{color}{_passed} passed, {_failed} failed{_RESET}")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
