#!/usr/bin/env python
"""finetune.pause_on_arrival — the backward child's arrival wait (CPU only).

Drives ``BackwardService._arrival_blocked`` / ``_maybe_pause`` with a real
shared-memory arrival counter and a threading.Event standing in for the GPU
grant:
  - no arrival → never blocked;
  - an arrival → blocked until the worker's pause for its prefill (grant
    cleared) has been released (grant set again);
  - an arrival with no prefill pause → released after the 100 ms cap;
  - arrivals before a cycle starts are not waited for;
  - off (meta flag false) → never attaches, never blocks.

    python tests/test_pause_on_arrival.py
"""

import os
import sys
import threading
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bwd_harness as H  # noqa: E402

sys.path[:] = [p for p in sys.path
               if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]

from deltaserve.bwd_services.base import BackwardService  # noqa: E402
from deltaserve.coordinator import FtArrivalSignal  # noqa: E402

C = H.Checker(tol=0.0)


def _service(sig, on: bool = True) -> BackwardService:
    os.environ[FtArrivalSignal.ENV] = sig.name
    svc = BackwardService(0)
    svc.shared["meta"] = {"pause_on_arrival": on, "backward_run_ahead_boundaries": 0}
    svc._gpu_grant = threading.Event()
    svc._gpu_grant.set()
    return svc


def main() -> None:
    sig = FtArrivalSignal.create()
    try:
        g_svc = _service(sig)
        g = g_svc._gpu_grant
        print("pause_on_arrival state machine:")
        C.ok("no arrival → not blocked", not g_svc._arrival_blocked(g))
        sig.bump()
        C.ok("arrival → blocked", g_svc._arrival_blocked(g))
        g.clear()
        C.ok("worker pauses for its prefill → still blocked", g_svc._arrival_blocked(g))
        g.set()
        C.ok("prefill done (grant set) → released", not g_svc._arrival_blocked(g))
        C.ok("stays released", not g_svc._arrival_blocked(g))

        sig.bump()
        C.ok("second arrival → blocked", g_svc._arrival_blocked(g))
        time.sleep(0.12)
        C.ok("no prefill pause within 100 ms → released", not g_svc._arrival_blocked(g))

        sig.bump()
        g_svc._arrivals_reset()
        C.ok("arrival before the cycle started → not waited for",
             not g_svc._arrival_blocked(g))

        # _maybe_pause end to end (tp=1 path): an arrival blocks the boundary
        # until a pause/resume cycle of the grant, driven by another thread.
        g_svc._maybe_pause()   # one-time setup (the run-ahead probe) off the clock
        sig.bump()

        def worker():
            time.sleep(0.02)
            g.clear()
            time.sleep(0.03)
            g.set()

        t = threading.Thread(target=worker)
        t0 = time.monotonic()
        t.start()
        g_svc._maybe_pause()
        dt = time.monotonic() - t0
        t.join()
        C.ok(f"_maybe_pause waits for the prefill pause to end ({dt * 1e3:.0f} ms)",
             0.045 <= dt < 0.095)
        t0 = time.monotonic()
        g_svc._maybe_pause()
        C.ok("next boundary passes at once", time.monotonic() - t0 < 0.01)

        off = _service(sig, on=False)
        sig.bump()
        C.ok("off → never blocked", not off._arrival_blocked(off._gpu_grant))
        C.ok("off → never attached", off._arrivals is None)
    finally:
        sig.close()
    C.finish()


if __name__ == "__main__":
    main()
