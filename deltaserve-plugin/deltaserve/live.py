# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Live feed behind the dashboard (``dserve-top``), part of telemetry.

Owned by ``metrics.MetricsRecorder`` and fed from its hooks (same lock), so it
exists only while telemetry records. Three things:

- **Step stream**: one entry per engine step, newest last, each with a
  sequence number so a client polls only what is new. Consecutive plain
  decode steps (inference only, no prefill, no FT) fold into ONE growing
  entry (``k == "d"``, ``n`` steps) — the ``d×N`` block of the dashboard;
  any other step closes it. Other kinds: ``"x"`` a step with prefill and/or FT
  (composition, GPU ms, graph, the FT admission code + tokens, backward
  running / paused), ``"p"`` an FT-only step pre-empted for an arriving
  request (tier B / C). Bounded ring (``steps_ring`` entries).
- **Per-second series**: one row per second (last ``series_s`` seconds):
  TTFT p50 / p99 of the requests whose first token came out that second,
  per-request mean TPOT p50 / p99 of the requests that finished, token
  rates, forward GPU seconds split inference / FT, backward GPU seconds,
  mean loss.
- **Step log on disk**: the same step entries (decode runs once closed) as
  gzip'd JSON lines in rolling segments ``steps.<n>.jsonl.gz``; the oldest
  segments are deleted to keep the total under ``steps_log_max_mb``.
"""

from __future__ import annotations

import gzip
import json
import os
import re

_SEG_PER_CAP = 8      # segments the size cap is split into


class RollingGz:
    """Gzip'd JSON lines in rolling segments under a total size cap. Used from
    the metrics writer thread only."""

    def __init__(self, root: str, prefix: str, max_mb: float) -> None:
        self.root = root
        self.prefix = prefix
        self.max_bytes = max(1, int(max_mb * (1 << 20)))
        self.seg_bytes = max(64 << 10, self.max_bytes // _SEG_PER_CAP)
        self._re = re.compile(re.escape(prefix) + r"\.(\d+)\.jsonl\.gz$")
        self._idx = 1 + max(self._segments() or [(0, "")])[0]
        self._f = None
        self._path = None
        self._n = 0

    def _segments(self) -> list[tuple[int, str]]:
        out = []
        for name in os.listdir(self.root):
            m = self._re.match(name)
            if m:
                out.append((int(m.group(1)), os.path.join(self.root, name)))
        return sorted(out)

    def write(self, obj: dict) -> None:
        if self._f is None:
            self._path = os.path.join(self.root, f"{self.prefix}.{self._idx:06d}.jsonl.gz")
            self._f = gzip.open(self._path, "at")
        self._f.write(json.dumps(obj, separators=(",", ":")) + "\n")
        self._n += 1
        if self._n % 256 == 0 and os.path.getsize(self._path) >= self.seg_bytes:
            self.rotate()

    def flush(self) -> None:
        if self._f is not None:
            self._f.flush()

    def rotate(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None
            self._idx += 1
        segs = self._segments()
        total = sum(os.path.getsize(p) for _, p in segs)
        for _, p in segs:                       # oldest first
            if total <= self.max_bytes:
                break
            total -= os.path.getsize(p)
            os.remove(p)

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None


def _pct(v: list[float], q: float):
    if not v:
        return None
    v = sorted(v)
    return v[min(len(v) - 1, max(0, -(-q * len(v) // 100) - 1))]


class _Sec:
    __slots__ = ("t", "ttft", "tpot", "out", "inp", "tr", "ftf", "fi", "ff", "bw",
                 "loss", "steps")

    def __init__(self, t: int) -> None:
        self.t = t
        self.ttft: list[float] = []
        self.tpot: list[float] = []
        self.out = self.inp = self.tr = self.ftf = self.steps = 0
        self.fi = self.ff = self.bw = 0.0
        self.loss: list[float] = []

    def row(self) -> dict:
        r = lambda x, k=4: None if x is None else round(x, k)  # noqa: E731
        return {"t": self.t,
                "ttft50": r(_pct(self.ttft, 50)), "ttft99": r(_pct(self.ttft, 99)),
                "nt": len(self.ttft),
                "tpot50": r(_pct(self.tpot, 50), 5), "tpot99": r(_pct(self.tpot, 99), 5),
                "nf": len(self.tpot),
                "out": self.out, "inp": self.inp, "tr": self.tr, "ftf": self.ftf,
                "fi": r(self.fi), "ff": r(self.ff), "bw": r(self.bw), "st": self.steps,
                "loss": r(sum(self.loss) / len(self.loss), 5) if self.loss else None}


class LiveFeed:
    """See the module docstring. All methods run under the recorder's lock;
    disk writes go through the recorder's writer thread (``write``)."""

    def __init__(self, write=None, steps_ring: int = 20000,
                 series_s: int = 3600) -> None:
        self._write = write                 # callable(dict) → step log, or None
        self._ring = int(steps_ring)
        self._steps: list[dict] = []
        self._base = 1                      # seq of self._steps[0]
        self._seq = 0
        self._run: dict | None = None       # the open decode run
        self._series: list[dict] = []
        self._series_s = int(series_s)
        self._cur: _Sec | None = None

    # -- step stream --------------------------------------------------------

    def _append(self, e: dict) -> None:
        self._seq += 1
        e["s"] = self._seq
        self._steps.append(e)
        if len(self._steps) > 2 * self._ring:          # amortized trim
            drop = len(self._steps) - self._ring
            del self._steps[:drop]
            self._base += drop

    def _close_run(self) -> None:
        if self._run is not None:
            if self._write:
                self._write(dict(self._run))
            self._run = None

    def step(self, t: float, feats, gpu_s: float, graph=None, paused=None) -> None:
        t_in, t_ft, b_d = int(feats.t_in), int(feats.t_ft), int(feats.b_d)
        prefill = t_in - t_ft
        if t_in == 0 and b_d == 0:
            return                                  # empty step
        ms = gpu_s * 1000.0
        if t_in == 0:                               # plain decode
            r = self._run
            if r is None:
                r = self._run = {"k": "d", "t": round(t, 4), "te": round(t, 4),
                                 "n": 1, "ms": round(ms, 3), "dc": b_d}
                self._append(r)
            else:
                r["n"] += 1
                r["te"] = round(t, 4)
                r["ms"] = round(r["ms"] + ms, 3)
                r["dc"] = b_d
            return
        self._close_run()
        code, at = getattr(feats, "admit", ("", 0)) or ("", 0)
        e = {"k": "x", "t": round(t, 4), "pf": prefill, "ft": t_ft, "dc": b_d,
             "ms": round(ms, 3), "g": None if graph is None else int(bool(graph)),
             "a": code, "at": int(at or 0), "bw": int(bool(getattr(feats, "bwd", False))),
             "pz": None if paused is None else int(bool(paused))}
        self._append(e)
        if self._write:
            self._write(e)

    def preempt(self, t: float, feats) -> None:
        """An FT-only step undone for an arriving request (tier B / C)."""
        self._close_run()
        e = {"k": "p", "t": round(t, 4),
             "ft": int(getattr(feats, "t_ft", 0) or 0) if feats is not None else 0}
        self._append(e)
        if self._write:
            self._write(e)

    def steps_since(self, since: int, limit: int = 4000) -> list[dict]:
        """Entries with seq ≥ ``since`` (the open decode run is re-sent with
        its new count; clients replace entries by seq), at most the newest
        ``limit``."""
        i = max(0, int(since) - self._base, len(self._steps) - limit)
        out = self._steps[i:]
        if out and self._run is not None and out[-1] is self._run:
            out = out[:-1] + [dict(self._run)]
        return out

    # -- per-second series ------------------------------------------------------

    def _sec(self, now: float) -> _Sec:
        sec = int(now)
        cur = self._cur
        if cur is None:
            cur = self._cur = _Sec(sec)
        elif sec > cur.t:
            self._series.append(cur.row())
            for gap in range(cur.t + 1, min(sec, cur.t + 61)):   # idle seconds
                self._series.append(_Sec(gap).row())
            if len(self._series) > 2 * self._series_s:
                del self._series[:len(self._series) - self._series_s]
            cur = self._cur = _Sec(sec)
        return cur

    def tick(self, now: float) -> None:
        self._sec(now)

    def first_token(self, now: float, ttft: float) -> None:
        self._sec(now).ttft.append(ttft)

    def tokens(self, now: float, out: int) -> None:
        self._sec(now).out += out

    def finished(self, now: float, prompt: int, tpot_mean: float | None) -> None:
        b = self._sec(now)
        b.inp += prompt
        if tpot_mean is not None:
            b.tpot.append(tpot_mean)

    def fwd(self, now: float, inf_s: float, ft_s: float, ft_tokens: int) -> None:
        b = self._sec(now)
        b.fi += inf_s
        b.ff += ft_s
        b.ftf += ft_tokens
        b.steps += 1

    def backward(self, now: float, tokens: int, gpu_s: float, loss) -> None:
        b = self._sec(now)
        b.tr += tokens
        b.bw += gpu_s
        if loss is not None:
            b.loss.append(float(loss))

    def series_since(self, t: float) -> list[dict]:
        s = self._series
        lo, hi = 0, len(s)
        while lo < hi:                               # first row with t > since
            mid = (lo + hi) // 2
            if s[mid]["t"] <= t:
                lo = mid + 1
            else:
                hi = mid
        return s[lo:]

    @property
    def last_seq(self) -> int:
        return self._seq

    def flush_open_run(self) -> None:
        """At close: write the open decode run to the log."""
        self._close_run()
