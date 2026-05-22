#!/usr/bin/env python
"""Phase 1 / Step 2 test: spawn the backward stub process + MPS env wrapping.

CPU-only — the Phase-1 backward stub does not touch CUDA, so this runs without a
GPU. It exercises vllm.deltaserve.backward_process.BackwardProcess directly
(independent of the full vLLM worker), proving:

  - the child spawns and handshakes 'ready',
  - the MPS env vars are applied to the CHILD only (parent env restored),
  - ping/shutdown round-trip works, and the child exits cleanly (exitcode 0).

    python tests/test_phase1_step2.py

Run as a file (not `python -c`) so `vllm` resolves to the installed package.
"""

import os
import sys

_passed = 0
_failed = 0
_TTY = sys.stdout.isatty()
_GREEN = "\033[92m" if _TTY else ""
_RED = "\033[91m" if _TTY else ""
_RESET = "\033[0m" if _TTY else ""

_MPS_ENV = "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
_MAX_CONN_ENV = "CUDA_DEVICE_MAX_CONNECTIONS"
_TEST_PCT = 37  # distinctive value to confirm it propagated to the child


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
    green("=== Phase 1 / Step 2: backward stub spawn + MPS wrapping ===")
    from vllm.deltaserve.backward_process import BackwardProcess

    # Record parent env up front so we can assert it's restored after spawn.
    prev_mps = os.environ.get(_MPS_ENV)
    prev_max_conn = os.environ.get(_MAX_CONN_ENV)

    bp = BackwardProcess(mps_percentage=_TEST_PCT, device_index=0)
    ready = bp.start()

    green("spawn + handshake:")
    check("child reported event=ready", ready.get("event") == "ready")
    check("child has a pid", isinstance(ready.get("pid"), int))
    check("handle.pid matches ready pid", bp.pid == ready.get("pid"))
    check("child is alive after start", bp.is_alive())

    green("MPS env applied to CHILD only:")
    check(f"child inherited {_MPS_ENV}={_TEST_PCT}",
          ready.get("mps_percentage") == str(_TEST_PCT))
    check(f"child inherited {_MAX_CONN_ENV}=1",
          ready.get("max_connections") == "1")
    check(f"parent {_MPS_ENV} restored (not leaked)",
          os.environ.get(_MPS_ENV) == prev_mps)
    check(f"parent {_MAX_CONN_ENV} restored (not leaked)",
          os.environ.get(_MAX_CONN_ENV) == prev_max_conn)

    green("ping round-trip:")
    pong = bp.ping(data="hello-deltaserve")
    check("pong event", pong.get("event") == "pong")
    check("pong echoes data", pong.get("data") == "hello-deltaserve")
    check("pong from same pid", pong.get("pid") == ready.get("pid"))

    green("clean shutdown:")
    exitcode = bp.shutdown()
    check("child exited (exitcode == 0)", exitcode == 0)
    check("child no longer alive", not bp.is_alive())

    color = _RED if _failed else _GREEN
    print(f"\n{color}{_passed} passed, {_failed} failed{_RESET}")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
