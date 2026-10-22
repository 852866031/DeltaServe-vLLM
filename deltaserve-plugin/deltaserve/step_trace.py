# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-step predicted-vs-actual trace for SLO-estimator validation.

One CSV row per timed engine step (plus one per rolled-back FT-only step),
written by a daemon thread so the scheduler's hot path only pays for a
string format and a queue put (a few µs against ≥5 ms steps). Enabled by
``finetune.step_trace_path``; ``eval-tp/analyze_step_trace.py`` reads it.

Column reference (all times in seconds, wall clocks are ``time.time()``):

  seq            scheduler step sequence (stamped at schedule())
  kind           ``step`` (timed) | ``rollback`` (tier B/C pre-emption)
  t_sched        wall clock when schedule() stamped the step
  t_exec         wall clock when rank 0 recorded the start CUDA event
  regime         composition regime of the realized batch
  regime_used    regime whose coefficients produced ``pred`` (cold-start
                 fallback makes these differ)
  was_graph      CUDA-graph mode the runner actually used
  t_in p t_ft b_d k s
                 realized features (FT tokens are a subset of t_in)
  est_t_in est_p est_b_d est_k
                 the inference composition admission REASONED on (before
                 super().schedule()); blank when admission returned early
  n_ft           FT samples admitted this step
  t_baseline     admission's margined prediction for the step WITHOUT FT
  t_admit        admission's margined prediction for the accepted set
  ttft_slack     ttft_deadline - now - queue_wait - t_baseline at admission
  queue_wait     the async queue-wait term admission used
  pred_raw       raw model prediction for the realized features
  pred           margined prediction (what the tracker records)
  actual         CUDA-event elapsed time of the forward (rank 0)
  host           host wall time of the forward dispatch (start→end record)
  pending_bwd    backward child had work in flight at schedule() (scheduler)
  paused_bwd     the runner paused the child for this step
  running_inf waiting
                 scheduler occupancy when the batch was built
"""

from __future__ import annotations

import os
import queue
import threading
import time

COLUMNS = (
    "seq", "kind", "t_sched", "t_exec", "regime", "regime_used", "was_graph",
    "t_in", "p", "t_ft", "b_d", "k", "s",
    "est_t_in", "est_p", "est_b_d", "est_k", "n_ft",
    "t_baseline", "t_admit", "ttft_slack", "queue_wait",
    "pred_raw", "pred", "actual", "host",
    "pending_bwd", "paused_bwd", "running_inf", "waiting",
)

_FLUSH_EVERY_S = 1.0


def _f(v) -> str:
    """Compact float / None formatting for a CSV cell."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        # Wall-clock stamps (~1.7e9 s) need their sub-ms digits; everything
        # else (durations, tokens, predictions) is fine at 6 significant.
        return f"{v:.6f}" if v >= 1e6 else f"{v:.6g}"
    return str(v)


class StepTraceWriter:
    """Daemon-thread CSV writer. ``put_row`` formats on the caller's thread
    (cheap) and enqueues; the thread owns the file handle, flushes at most
    once per second, and closes on ``close()``. The file is truncated at
    open so a run's trace never concatenates with a previous one."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._run, name="dserve-step-trace", daemon=True)
        self._thread.start()

    def put_row(self, values) -> None:
        self._q.put(",".join(_f(v) for v in values) + "\n")

    def close(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=3.0)

    def _run(self) -> None:
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "w", buffering=1 << 16) as f:
            f.write(",".join(COLUMNS) + "\n")
            f.flush()
            last_flush = time.monotonic()
            while True:
                try:
                    item = self._q.get(timeout=_FLUSH_EVERY_S)
                except queue.Empty:
                    item = ""
                if item is None:
                    break
                if item:
                    f.write(item)
                now = time.monotonic()
                if now - last_flush >= _FLUSH_EVERY_S:
                    f.flush()
                    last_flush = now
            f.flush()
