"""Average time-between-tokens over a sliding window (``avg_tbt_window_s``).

The per-step TBT check bounds ONE step (``max_tbt_slo``): every step may use
that bound, so nothing keeps the average near ``avg_tbt_slo``. This window
does: it holds the decode steps that finished in the last ``window_s`` seconds
and answers "what would the average TBT be if the step being built took this
long".

One completed step that carried ``b`` decode requests gave each of them one
token, ``gap`` seconds after their previous one. The window's average is the
mean token gap over every token of the window — the window's whole decode
throughput turned into one number:

    avg TBT = Σ b_i · gap_i / Σ b_i

``gap`` is the time since the previous step finished, plus the time of any
step in between that carried no decode request while inference requests were
running (an FT-only step they waited through). Time during which no inference
request was running is not a token gap and is dropped.
"""
from __future__ import annotations

from collections import deque


class TbtWindow:
    def __init__(self, window_s: float, max_gap_s: float = 1.0):
        self.window_s = float(window_s)
        self.max_gap_s = float(max_gap_s)
        self._steps: deque = deque()      # (t_done, b, b * gap)
        self._sum_b = 0.0
        self._sum_bt = 0.0
        self._prev_done: float | None = None
        self._carry = 0.0

    # ------------------------------------------------------------- record
    def step_done(self, now: float, decode_requests: int,
                  inference_running: bool) -> None:
        """A step's output arrived at ``now``. ``decode_requests``: how many
        requests it gave a decode token; ``inference_running``: whether any
        inference request is (still) running."""
        gap = None if self._prev_done is None else now - self._prev_done
        self._prev_done = now
        if gap is None or gap > self.max_gap_s:
            # First step, or the engine was idle: not a token gap.
            self._carry = 0.0
            return
        if decode_requests > 0:
            bt = decode_requests * (gap + self._carry)
            self._steps.append((now, decode_requests, bt))
            self._sum_b += decode_requests
            self._sum_bt += bt
            self._carry = 0.0
        elif inference_running:
            self._carry += gap
        else:
            self._carry = 0.0

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_s
        steps = self._steps
        while steps and steps[0][0] < cutoff:
            _, b, bt = steps.popleft()
            self._sum_b -= b
            self._sum_bt -= bt
        if not steps:
            self._sum_b = self._sum_bt = 0.0      # no float drift

    # -------------------------------------------------------------- query
    def average(self, now: float) -> float | None:
        """Average TBT (seconds) over the window, None when it is empty."""
        self._trim(now)
        return self._sum_bt / self._sum_b if self._sum_b > 0 else None

    def average_with(self, now: float, decode_requests: int,
                     step_s: float) -> float:
        """The window's average if a step with ``decode_requests`` decode
        requests taking ``step_s`` seconds were added to it."""
        self._trim(now)
        b = self._sum_b + decode_requests
        if b <= 0:
            return 0.0
        return (self._sum_bt + decode_requests * (step_s + self._carry)) / b
