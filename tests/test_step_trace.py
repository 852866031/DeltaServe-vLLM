#!/usr/bin/env python
"""step_trace — the per-step predicted-vs-actual trace (CPU, no GPU).

Drives the scheduler-side helpers (``_trace_stamp`` / ``_trace_step`` /
``_trace_rollback``) on a stub with the real ``FinetuneScheduler`` methods
bound, the runner's ``extra`` tuple as the timing ring produces it, and the
daemon writer; then reads the CSV back.

    python tests/test_step_trace.py
"""

import csv
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bwd_harness as H  # noqa: E402

sys.path[:] = [p for p in sys.path
               if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]

from vllm.deltaserve.coordinator import FinetuneCoordinator  # noqa: E402
from vllm.deltaserve.estimator import (  # noqa: E402
    MergedExecutionEstimator, StepExecutionTracker, StepFeatures)
from vllm.deltaserve.ft_scheduler import FinetuneScheduler  # noqa: E402
from vllm.deltaserve.step_trace import COLUMNS, StepTraceWriter  # noqa: E402

C = H.Checker(tol=1e-9)


class _Out:
    total_num_scheduled_tokens = 1


class _Stub:
    """Just the attributes the trace helpers touch."""
    _TRACE_CTX_MAX = FinetuneScheduler._TRACE_CTX_MAX
    _EMPTY_ADMIT = FinetuneScheduler._EMPTY_ADMIT
    _trace_stamp = FinetuneScheduler._trace_stamp
    _trace_step = FinetuneScheduler._trace_step
    _trace_rollback = FinetuneScheduler._trace_rollback

    def __init__(self, path):
        self._trace = StepTraceWriter(path)
        self._step_seq = 0
        self._trace_ctx = {}
        self._admit_trace = None
        self._coord = FinetuneCoordinator(256, 256)
        self._estimator = MergedExecutionEstimator()


def _fit(est):
    tr = StepExecutionTracker()
    for i in range(12):
        f = StepFeatures(t_in=64 + 8 * i, p=2, t_ft=0, b_d=0, k=0,
                         prefill_lens=[32 + 4 * i, 32 + 4 * i])
        tr.add(f, 1e-4 * f.t_in + 2e-3)
        g = StepFeatures(t_in=96 + 8 * i, p=3, t_ft=32, b_d=0, k=0,
                         prefill_lens=[32 + 4 * i, 32 + 4 * i, 32])
        tr.add(g, 1e-4 * g.t_in + 5e-4 * g.t_ft + 2e-3)
    est.data_fit(tr)


def test_round_trip():
    print("test_round_trip (stamp → runner extra → drain → CSV):")
    d = tempfile.mkdtemp()
    path = os.path.join(d, "sub", "trace.csv")
    st = _Stub(path)
    _fit(st._estimator)

    # Step 0: co-serving step with an admission record.
    o0 = _Out()
    o0._ft_step_features = StepFeatures(t_in=128, p=3, t_ft=32, b_d=2, k=300,
                                        prefill_lens=[48, 48, 32])
    o0._ft_step_predicted = 0.02
    o0._ft_running_inf, o0._ft_waiting = 2, 1
    st._admit_trace = [96.0, 2, 2, 300.0, 1, 0.015, 0.019, 0.3, 0.004]
    st._coord.pending_backward = True
    st._trace_stamp(o0)
    C.ok("seq stamped on the output", o0._ft_step_seq == 0)
    C.ok("ctx kept", 0 in st._trace_ctx)

    # Step 1: a step that will be rolled back (tier B/C).
    o1 = _Out()
    o1._ft_step_features = StepFeatures(t_in=64, p=1, t_ft=64, b_d=0, k=0,
                                        prefill_lens=[64])
    o1._ft_step_predicted = 0.01
    o1._ft_running_inf, o1._ft_waiting = 0, 0
    st._admit_trace = [0.0, 0, 0, 0.0, 1, None, 0.009, None, 0.0]
    st._coord.pending_backward = False
    st._trace_stamp(o1)
    C.ok("seq increments", o1._ft_step_seq == 1)
    st._trace_rollback(o1)
    C.ok("rollback pops its ctx", 1 not in st._trace_ctx)
    st._trace_rollback(o1)  # idempotent: no second row

    # Runner drains step 0's timing with the extra tuple.
    t_exec = time.time()
    extra = (0, t_exec, 0.0031, True)
    st._trace_step(o0._ft_step_features, 0.0234, False, 0.02, extra)
    C.ok("step ctx consumed", 0 not in st._trace_ctx)
    # A sample with no ctx (e.g. from before the trace existed) still logs.
    st._trace_step(StepFeatures(t_in=0, p=0, t_ft=0, b_d=4, k=1000), 0.005,
                   True, 0.0048, None)
    st._trace.close()

    with open(path) as f:
        rows = list(csv.DictReader(f))
    C.ok("header matches COLUMNS", list(rows[0].keys()) == list(COLUMNS))
    C.ok("3 rows (step, rollback, ctx-less step)", len(rows) == 3)
    by_kind = {(r["seq"], r["kind"]): r for r in rows}
    r0 = by_kind[("0", "step")]
    C.ok("regime eager", r0["regime"] == "eager")
    C.ok("regime_used recorded", r0["regime_used"] == "eager")
    C.ok("actual = event time", abs(float(r0["actual"]) - 0.0234) < 1e-9)
    C.ok("host time", abs(float(r0["host"]) - 0.0031) < 1e-9)
    C.ok("t_exec", abs(float(r0["t_exec"]) - t_exec) < 1e-3)
    C.ok("pred (margined) kept", abs(float(r0["pred"]) - 0.02) < 1e-9)
    raw = st._estimator.predict(o0._ft_step_features, apply_margin=False)
    C.ok("pred_raw = model without margin", abs(float(r0["pred_raw"]) - raw) < 1e-6)
    C.ok("pending_bwd from scheduler", r0["pending_bwd"] == "1")
    C.ok("paused_bwd from runner", r0["paused_bwd"] == "1")
    C.ok("admission estimate", r0["est_t_in"] == "96" and r0["est_b_d"] == "2")
    C.ok("admission n_ft / t_admit", r0["n_ft"] == "1" and r0["t_admit"] == "0.019")
    C.ok("ttft_slack", r0["ttft_slack"] == "0.3")
    C.ok("occupancy", r0["running_inf"] == "2" and r0["waiting"] == "1")
    C.ok("s = Σ n²", float(r0["s"]) == 48 * 48 * 2 + 32 * 32)
    r1 = by_kind[("1", "rollback")]
    C.ok("rollback has no actual", r1["actual"] == "" and r1["t_exec"] == "")
    C.ok("rollback keeps its prediction", r1["pred"] == "0.01")
    r2 = by_kind[("", "step")]
    C.ok("ctx-less row: decode_only, blank scheduler fields",
         r2["regime"] == "decode_only" and r2["t_sched"] == ""
         and r2["actual"] == "0.005")


def test_ctx_eviction():
    print("test_ctx_eviction:")
    d = tempfile.mkdtemp()
    st = _Stub(os.path.join(d, "t.csv"))
    for _ in range(st._TRACE_CTX_MAX + 10):
        o = _Out()
        o._ft_step_features = StepFeatures(t_in=1, p=1)
        o._ft_step_predicted = None
        o._ft_running_inf = o._ft_waiting = 0
        st._trace_stamp(o)
    C.ok("ctx bounded", len(st._trace_ctx) == st._TRACE_CTX_MAX)
    C.ok("oldest evicted", 0 not in st._trace_ctx and 9 not in st._trace_ctx)
    st._trace.close()


def test_writer_cost():
    print("test_writer_cost (hot-path cost of one row):")
    d = tempfile.mkdtemp()
    w = StepTraceWriter(os.path.join(d, "t.csv"))
    row = (1, "step", time.time(), time.time(), "eager", "eager", False,
           128.0, 3, 32.0, 2, 300.0, 5632.0, 96.0, 2, 2, 300.0, 1, 0.015, 0.019,
           0.3, 0.004, 0.0198, 0.02, 0.0234, 0.0031, True, True, 2, 1)
    n = 20000
    t0 = time.perf_counter()
    for _ in range(n):
        w.put_row(row)
    per_row_us = (time.perf_counter() - t0) / n * 1e6
    w.close()
    print(f"  {per_row_us:.1f} µs per row on the caller's thread")
    C.ok("per-row cost < 50 µs", per_row_us < 50)
    with open(w.path) as f:
        C.ok("all rows written", sum(1 for _ in f) == n + 1)


if __name__ == "__main__":
    test_round_trip()
    test_ctx_eviction()
    test_writer_cost()
    C.finish()
