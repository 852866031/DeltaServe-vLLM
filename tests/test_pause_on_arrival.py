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

        # [adapter routing] a request has arrived when the router service
        # receives it; its later ADD in the engine is the same arrival.
        print("a request that is routed first:")
        r_svc = _service(sig)
        rg = r_svc._gpu_grant
        C.ok("nothing routed → not blocked, nothing pending",
             not r_svc._arrival_blocked(rg) and sig.routing_pending() == 0)
        before = sig.read()
        sig.route_started()
        C.ok("route started = an arrival (the counter every reader compares moves)",
             sig.read() == before + 1 and sig.routing_pending() == 1)
        C.ok("→ the backward child is blocked", r_svc._arrival_blocked(rg))
        sig.route_finished()
        C.ok("route finished: nothing pending, the child still waits for the prefill",
             sig.routing_pending() == 0 and r_svc._arrival_blocked(rg))
        sig.bump()                 # the request reaches the engine
        C.ok("the request reaches the engine → still blocked",
             r_svc._arrival_blocked(rg))
        rg.clear()
        C.ok("its prefill pause → still blocked", r_svc._arrival_blocked(rg))
        rg.set()
        C.ok("prefill done → released", not r_svc._arrival_blocked(rg))
        C.ok("stays released (the ADD was not a second arrival to wait for)",
             not r_svc._arrival_blocked(rg))
        sig.route_started()
        sig.route_finished()
        C.ok("a routed request that never reaches the engine → blocked",
             r_svc._arrival_blocked(rg))
        time.sleep(0.12)
        C.ok("... for 100 ms only", not r_svc._arrival_blocked(rg))

        # pause / resume through _maybe_pause, the whole life of a routed request
        r_svc._maybe_pause()

        def routed_request():
            time.sleep(0.01)
            sig.route_started()        # arrives at the router
            time.sleep(0.015)
            sig.route_finished()       # routed
            time.sleep(0.002)
            sig.bump()                 # ADD in the engine
            time.sleep(0.005)
            rg.clear()                 # the worker pauses the child for its prefill
            time.sleep(0.03)
            rg.set()                   # prefill done

        t = threading.Thread(target=routed_request)
        t.start()
        passed_before = 0
        t0 = time.monotonic()
        while time.monotonic() - t0 < 0.009:      # boundaries before the arrival
            r_svc._maybe_pause()
            passed_before += 1
        time.sleep(0.004)                         # the route has started
        t1 = time.monotonic()
        r_svc._maybe_pause()                      # blocks here
        held = time.monotonic() - t1
        t.join()
        C.ok(f"boundaries pass freely before the request ({passed_before})",
             passed_before > 10)
        C.ok(f"one pause from the route's start to the end of the prefill "
             f"({held * 1e3:.0f} ms, no resume in between)", 0.04 <= held < 0.09)
        t0 = time.monotonic()
        for _ in range(50):
            r_svc._maybe_pause()
        C.ok("then the child runs again: 50 boundaries pass at once",
             time.monotonic() - t0 < 0.02)

        from deltaserve.adapter_routing.service import RouteArrival
        arr = RouteArrival()
        s0 = sig.route_counts()

        def many():
            for _ in range(200):
                arr.started()
                arr.finished()

        ths = [threading.Thread(target=many) for _ in range(8)]
        [t.start() for t in ths]
        [t.join() for t in ths]
        s1 = sig.route_counts()
        C.ok("the router service's handler threads lose no count (8 x 200)",
             arr.active and s1[0] - s0[0] == 1600 and s1[1] - s0[1] == 1600
             and sig.routing_pending() == 0)
        arr.close()
        C.ok("switched off (arrival_on_route: false) → inert",
             not RouteArrival(False).active)
        name = os.environ.pop(FtArrivalSignal.ENV)
        C.ok("no co-serving arrival signal → inert", not RouteArrival().active)
        os.environ[FtArrivalSignal.ENV] = name

        # the scheduler's wait for a request that is being routed
        print("the scheduler waits for a request that is being routed:")
        import queue as _queue

        from deltaserve.engine_core import EngineInputs

        class Eng:
            def __init__(self):
                self.input_queue = _queue.Queue()
                self.handled = []
                self._ds = type("H", (), {"arrival_signal": sig})()

            def _handle_client_request(self, *req):
                self.handled.append(req)

        eng = Eng()
        inp = EngineInputs(eng)
        t0 = time.monotonic()
        C.ok("nothing being routed → no wait",
             not inp.wait_for_routed() and time.monotonic() - t0 < 0.005)

        def arrive():
            time.sleep(0.02)
            sig.route_finished()
            time.sleep(0.001)
            eng.input_queue.put(("ADD", "req"))

        sig.route_started()
        t = threading.Thread(target=arrive)
        t0 = time.monotonic()
        t.start()
        got = inp.wait_for_routed()
        dt = time.monotonic() - t0
        t.join()
        C.ok(f"waits until the routed request lands and handles it ({dt * 1e3:.0f} ms)",
             got and eng.handled == [("ADD", "req")] and 0.018 <= dt < 0.05)
        sig.route_started()
        inp.ROUTE_WAIT_CAP_S = 0.05
        t0 = time.monotonic()
        got = inp.wait_for_routed()
        dt = time.monotonic() - t0
        C.ok(f"a route that never finishes is waited for up to the cap only ({dt * 1e3:.0f} ms)",
             not got and 0.045 <= dt < 0.09)
        t0 = time.monotonic()
        C.ok("... and not waited for again",
             not inp.wait_for_routed() and time.monotonic() - t0 < 0.005)
        sig.route_finished()

        off = _service(sig, on=False)
        sig.bump()
        C.ok("off → never blocked", not off._arrival_blocked(off._gpu_grant))
        C.ok("off → never attached", off._arrivals is None)
    finally:
        sig.close()
    C.finish()


if __name__ == "__main__":
    main()
