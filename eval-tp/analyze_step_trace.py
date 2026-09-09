#!/usr/bin/env python3
"""analyze_step_trace.py — where (and why) the SLO estimator's per-step
prediction misses the measured step time.

Reads the per-step trace the server writes when launched with
``finetune.step_trace_path`` (``auto_benchmark_tp.py --step-trace``) and joins
it with the same run's request results, backward log and server log:

    step_trace<suffix>.csv          one row per timed step (+ rollbacks)
    timeline_results<suffix>.csv    per-request TTFT / latency (t_rel_s)
    bench_meta<suffix>.json         wall-clock anchor of t_rel_s = 0
    bwd_log<suffix>.csv             one row per completed backward
    server<suffix>.log              [backward] cycle lines, [ft-abort] lines

and prints, for the recorded timeline window:

  1. accuracy per regime (raw model, no safety margin): RMSE, bias, ratio
     percentiles, share of steps > 1.5× / > 2× the prediction;
  2. the same split by "backward in flight" / "child paused" / FT-only /
     regime of the previous step — the interference and transition suspects;
  3. host-vs-GPU: steps whose CPU dispatch time is the step time (CPU-bound);
  4. admission fidelity: what the admission loop predicted for the composition
     it reasoned on vs. the composition that actually ran, and vs. the actual;
  5. the worst misses with their neighbours;
  6. every TTFT-SLO violation with the steps that ran during the wait.

Usage:
    python eval-tp/analyze_step_trace.py --family qwen3-14b --tp 2 --tight
    python eval-tp/analyze_step_trace.py --trace eval-tp/output/step_trace_….csv
    … [--top 15] [--plot]   (PNG next to the trace: step_trace_<…>.png)
"""

from __future__ import annotations

import argparse
import bisect
import csv
import datetime as dt
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
OUTPUT_DIR = _HERE / "output"
MODES = ("loose", "tight", "nutanix", "nutanix-600-800")
_SLO_DEFAULT = {"qwen3-14b": 0.4, "llama3": 0.25, "qwen3-0.6b": 0.4}


# ─── loading ────────────────────────────────────────────────────────────────

def _fl(v: str) -> float | None:
    return None if v == "" else float(v)


@dataclass
class Step:
    seq: int | None
    kind: str
    t_sched: float | None
    t_exec: float | None
    regime: str
    regime_used: str
    was_graph: bool | None
    t_in: float
    p: int
    t_ft: float
    b_d: int
    k: float
    s: float
    est_t_in: float | None
    est_p: int | None
    est_b_d: int | None
    est_k: float | None
    n_ft: int | None
    t_baseline: float | None
    t_admit: float | None
    ttft_slack: float | None
    queue_wait: float | None
    pred_raw: float | None
    pred: float | None
    actual: float | None
    host: float | None
    pending_bwd: bool | None
    paused_bwd: bool | None
    running_inf: int | None
    waiting: int | None
    prev_regime: str = ""      # filled after load
    prev_actual: float | None = None

    @property
    def ft_only(self) -> bool:
        return self.t_ft > 0 and self.t_ft == self.t_in and self.b_d == 0

    @property
    def t(self) -> float | None:
        return self.t_exec if self.t_exec is not None else self.t_sched

    @property
    def ratio(self) -> float | None:
        if self.actual is None or not self.pred_raw:
            return None
        return self.actual / self.pred_raw

    @property
    def resid(self) -> float | None:
        if self.actual is None or self.pred_raw is None:
            return None
        return self.actual - self.pred_raw


def _b(v: str) -> bool | None:
    return None if v == "" else v in ("1", "True", "true")


def load_trace(path: str) -> list[Step]:
    steps: list[Step] = []
    with open(path) as f:
        for r in csv.DictReader(f):
            steps.append(Step(
                seq=int(r["seq"]) if r["seq"] else None, kind=r["kind"],
                t_sched=_fl(r["t_sched"]), t_exec=_fl(r["t_exec"]),
                regime=r["regime"], regime_used=r["regime_used"],
                was_graph=_b(r["was_graph"]),
                t_in=float(r["t_in"]), p=int(float(r["p"])), t_ft=float(r["t_ft"]),
                b_d=int(float(r["b_d"])), k=float(r["k"]), s=float(r["s"]),
                est_t_in=_fl(r["est_t_in"]),
                est_p=int(float(r["est_p"])) if r["est_p"] else None,
                est_b_d=int(float(r["est_b_d"])) if r["est_b_d"] else None,
                est_k=_fl(r["est_k"]),
                n_ft=int(float(r["n_ft"])) if r["n_ft"] else None,
                t_baseline=_fl(r["t_baseline"]), t_admit=_fl(r["t_admit"]),
                ttft_slack=_fl(r["ttft_slack"]), queue_wait=_fl(r["queue_wait"]),
                pred_raw=_fl(r["pred_raw"]), pred=_fl(r["pred"]),
                actual=_fl(r["actual"]), host=_fl(r["host"]),
                pending_bwd=_b(r["pending_bwd"]), paused_bwd=_b(r["paused_bwd"]),
                running_inf=int(float(r["running_inf"])) if r["running_inf"] else None,
                waiting=int(float(r["waiting"])) if r["waiting"] else None,
            ))
    # Rows arrive in drain order (ring of 4 → nearly sorted); sort by seq.
    steps.sort(key=lambda s: (s.seq if s.seq is not None else -1))
    prev = None
    for s in steps:
        if s.kind == "step":
            if prev is not None:
                s.prev_regime, s.prev_actual = prev.regime, prev.actual
            prev = s
    return steps


def load_results(path: str):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(dict(idx=int(r["idx"]), t_rel=float(r["t_rel_s"]),
                             latency=_fl(r["latency_s"]), status=r["status"],
                             ttft=_fl(r["ttft_s"]),
                             avg_tbt=_fl(r["avg_tbt_s"]),
                             worst_tbt=_fl(r["worst_tbt_s"])))
    return rows


def load_bwd_log(path: str):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                ts = dt.datetime.fromisoformat(r["timestamp"]).timestamp()
            except Exception:
                continue
            rows.append(dict(t=ts, tokens=int(r["batch_tokens"]),
                             loss=float(r["batch_loss"]) if r["batch_loss"] else None))
    return rows


_BWD_RE = re.compile(r"\[backward\]\s+(\d+(?:\.\d+)?)\s*ms\s+\((graph|eager)\)")
_ABORT_RE = re.compile(r"\[ft-abort\]")
_TS_RE = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d{3})")


def load_server_log(path: str):
    """Backward cycle durations (ms, per rank line) and abort count."""
    cycles: list[float] = []
    aborts = 0
    if not os.path.exists(path):
        return cycles, aborts
    with open(path, errors="replace") as f:
        for line in f:
            m = _BWD_RE.search(line)
            if m:
                cycles.append(float(m.group(1)))
            if _ABORT_RE.search(line):
                aborts += 1
    return cycles, aborts


# ─── stats helpers ──────────────────────────────────────────────────────────

def pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    return ys[min(len(ys) - 1, int(q * len(ys)))]


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def rmse(steps: list[Step]) -> float:
    r = [s.resid for s in steps if s.resid is not None]
    return math.sqrt(mean([x * x for x in r])) if r else float("nan")


def ms(x: float | None) -> str:
    return "   -  " if x is None or (isinstance(x, float) and math.isnan(x)) \
        else f"{x * 1e3:6.1f}"


def acc_line(label: str, steps: list[Step]) -> str:
    steps = [s for s in steps if s.ratio is not None]
    if not steps:
        return f"  {label:<34} n=0"
    ratios = [s.ratio for s in steps]
    act = [s.actual for s in steps]
    prd = [s.pred_raw for s in steps]
    over15 = sum(1 for r in ratios if r > 1.5) / len(ratios)
    over2 = sum(1 for r in ratios if r > 2.0) / len(ratios)
    return (f"  {label:<34} n={len(steps):5d}  actual {ms(mean(act))}ms  "
            f"pred {ms(mean(prd))}ms  rmse {ms(rmse(steps))}ms  "
            f"ratio p50 {pct(ratios, .5):4.2f} p90 {pct(ratios, .9):4.2f} "
            f"p99 {pct(ratios, .99):5.2f}  >1.5x {over15 * 100:4.1f}%  "
            f">2x {over2 * 100:4.1f}%")


# ─── the report ─────────────────────────────────────────────────────────────

def report(steps: list[Step], results, t0: float | None, slo: float,
           bwd_rows, cycles, aborts, top: int, out=print) -> dict:
    timed = [s for s in steps if s.kind == "step" and s.actual is not None]
    rollbacks = [s for s in steps if s.kind == "rollback"]
    fitted = [s for s in timed if s.pred_raw]   # a 0.0 prediction = cold start

    # Timeline window = the recorded requests' span (+ their latency).
    if results and t0 is not None:
        w0 = t0 + min(r["t_rel"] for r in results)
        w1 = t0 + max(r["t_rel"] + (r["latency"] or 0) for r in results)
        win = [s for s in fitted if s.t is not None and w0 <= s.t <= w1]
        win_rb = [s for s in rollbacks if s.t is not None and w0 <= s.t <= w1]
    else:
        w0 = w1 = None
        win, win_rb = fitted, rollbacks

    def rel(t: float | None) -> str:
        if t is None:
            return "   -   "
        return f"{t - (w0 or t):7.2f}"

    out("=" * 100)
    out(f"step trace: {len(steps)} rows | timed {len(timed)} | with prediction "
        f"{len(fitted)} | rollbacks {len(rollbacks)}"
        + (f" | timeline window {w1 - w0:.1f}s → {len(win)} steps, "
           f"{len(win_rb)} rollbacks" if w0 else ""))
    if cycles:
        out(f"backward cycles (server log, per rank line): {len(cycles)}  "
            f"p50 {pct(cycles, .5):.0f}ms  p90 {pct(cycles, .9):.0f}ms  "
            f"max {max(cycles):.0f}ms | tier-C abort lines: {aborts}")
    if bwd_rows:
        out(f"bwd_log rows in window: {len(bwd_rows)} "
            f"({sum(r['tokens'] for r in bwd_rows)} FT tokens)")

    # 1. per regime
    out("\n[1] accuracy per regime (raw model prediction, no safety margin)")
    for reg in REGIMES:
        out(acc_line(reg, [s for s in win if s.regime == reg]))
    out(acc_line("ALL", win))
    fb = [s for s in win if s.regime_used and s.regime_used != s.regime]
    if fb:
        out(f"  ! {len(fb)} steps predicted with a FALLBACK regime "
            f"(regime_used != regime) — cold-start")

    # 2. splits
    out("\n[2] the suspects — same metric, split")
    for reg in REGIMES:
        rs = [s for s in win if s.regime == reg]
        if not rs:
            continue
        out(f"  -- {reg}")
        out(acc_line("   backward in flight", [s for s in rs if s.pending_bwd]))
        out(acc_line("   no backward", [s for s in rs if s.pending_bwd is False]))
        if reg in ("inf_prefill", "eager"):
            out(acc_line("   child paused for this step",
                         [s for s in rs if s.paused_bwd]))
            out(acc_line("   bwd in flight, NOT paused",
                         [s for s in rs if s.pending_bwd and not s.paused_bwd]))
        if reg == "eager":
            out(acc_line("   FT-only (no inference)", [s for s in rs if s.ft_only]))
            out(acc_line("   co-serving (FT + inference)",
                         [s for s in rs if not s.ft_only]))
        out(acc_line("   prev step eager", [s for s in rs if s.prev_regime == "eager"]))
        out(acc_line("   prev step not eager",
                     [s for s in rs if s.prev_regime and s.prev_regime != "eager"]))
        out(acc_line("   was_graph=1", [s for s in rs if s.was_graph]))
        out(acc_line("   was_graph=0", [s for s in rs if s.was_graph is False]))

    # 3. host vs gpu
    out("\n[3] host dispatch vs GPU time (host ≥ 0.9·actual ⇒ the host is blocked for the "
        "whole step: CPU-bound dispatch, an internal sync, or launch-queue back-pressure "
        "— if host grows with tokens it is the GPU holding the CPU, not the reverse)")
    for reg in REGIMES:
        rs = [s for s in win if s.regime == reg and s.host is not None]
        if not rs:
            continue
        cpu = [s for s in rs if s.host >= 0.9 * s.actual]
        out(f"  {reg:<12} host p50 {ms(pct([s.host for s in rs], .5))}ms "
            f"p90 {ms(pct([s.host for s in rs], .9))}ms | gpu p50 "
            f"{ms(pct([s.actual for s in rs], .5))}ms | host-blocked steps "
            f"{len(cpu)}/{len(rs)} ({len(cpu) / len(rs) * 100:.1f}%)"
            + (f" — their ratio p50 {pct([s.ratio for s in cpu], .5):.2f}" if cpu else ""))
        if len(cpu) > 10:
            small = sorted(cpu, key=lambda s: s.t_in)[: max(3, len(cpu) // 4)]
            large = sorted(cpu, key=lambda s: s.t_in)[-max(3, len(cpu) // 4):]
            out(f"               host by tokens: t_in≈{mean([s.t_in for s in small]):.0f} → "
                f"{ms(mean([s.host for s in small]))}ms, t_in≈{mean([s.t_in for s in large]):.0f} → "
                f"{ms(mean([s.host for s in large]))}ms")

    # 4. admission fidelity
    out("\n[4] admission fidelity (steps where FT was admitted)")
    adm = [s for s in win if s.n_ft and s.t_admit]
    if adm:
        comp_mis = [s for s in adm
                    if s.est_t_in is not None
                    and (abs((s.t_in - s.t_ft) - s.est_t_in) > 0.5
                         or s.est_b_d != s.b_d)]
        out(f"  admitted steps {len(adm)} | realized composition ≠ what admission "
            f"reasoned on: {len(comp_mis)} ({len(comp_mis) / len(adm) * 100:.1f}%)")
        for s in comp_mis[:5]:
            out(f"    seq {s.seq}: est inf_prefill={s.est_t_in:.0f} b_d={s.est_b_d} "
                f"k={s.est_k:.0f} → ran prefill={s.t_in - s.t_ft:.0f}(+ft {s.t_ft:.0f}) "
                f"b_d={s.b_d} k={s.k:.0f}")
        ra = [s.actual / s.t_admit for s in adm]
        out(f"  actual / t_admit (margined admission prediction): p50 {pct(ra, .5):.2f} "
            f"p90 {pct(ra, .9):.2f} p99 {pct(ra, .99):.2f} max {max(ra):.2f} | "
            f"actual > t_admit on {sum(1 for r in ra if r > 1)}/{len(ra)} steps")
        sl = [s for s in adm if s.ttft_slack is not None]
        if sl:
            burn = [s.actual - s.t_baseline for s in sl if s.t_baseline]
            out(f"  TTFT slack at admission p50 {ms(pct([s.ttft_slack for s in sl], .5))}ms "
                f"p10 {ms(pct([s.ttft_slack for s in sl], .1))}ms | "
                f"step cost above baseline p50 {ms(pct(burn, .5))}ms p90 {ms(pct(burn, .9))}ms")
        over = [s for s in adm if s.ttft_slack is not None and s.actual > s.t_admit
                and (s.actual - s.t_admit) > s.ttft_slack]
        out(f"  admitted steps whose miss alone exceeded the remaining TTFT slack: "
            f"{len(over)}")
    else:
        out("  no admitted steps in the window")
    denied = [s for s in win if s.n_ft == 0 and s.est_t_in is not None
              and s.est_t_in > 0]
    out(f"  prefill steps admission evaluated but admitted nothing: {len(denied)}")

    # 5. worst misses
    out(f"\n[5] worst {top} misses by (actual − pred), with the previous step")
    hdr = ("   seq   t(s)   regime      g  t_in  t_ft b_d     k  pred  actual  host  "
           "ratio bwd pau nft  run wait | prev regime/actual")
    out(hdr)
    worst = sorted(win, key=lambda s: -(s.resid or 0))[:top]
    for s in worst:
        out(f"  {s.seq:5d} {rel(s.t)} {s.regime:<11} {int(bool(s.was_graph))}  "
            f"{s.t_in:4.0f} {s.t_ft:5.0f} {s.b_d:3d} {s.k:5.0f} {ms(s.pred_raw)} "
            f"{ms(s.actual)} {ms(s.host)} {s.ratio:5.2f}   "
            f"{int(bool(s.pending_bwd))}   {int(bool(s.paused_bwd))} "
            f"{s.n_ft if s.n_ft is not None else '-':>3} "
            f"{s.running_inf if s.running_inf is not None else '-':>4} "
            f"{s.waiting if s.waiting is not None else '-':>4} | "
            f"{s.prev_regime or '-'} {ms(s.prev_actual)}")

    # 6. TTFT violations
    summary = {}
    if results and w0 is not None:
        viol = [r for r in results if r["ttft"] is not None and r["ttft"] > slo]
        ok = [r for r in results if r["ttft"] is not None]
        out(f"\n[6] TTFT-SLO ({slo * 1e3:.0f}ms) violations: {len(viol)}/{len(ok)} "
            f"({(1 - len(viol) / max(1, len(ok))) * 100:.1f}% satisfied) — the steps "
            f"that ran between send and first token")
        ts = [s.t for s in win]
        by_t = sorted(zip(ts, win), key=lambda x: x[0])
        keys = [x[0] for x in by_t]
        rb_t = sorted(s.t for s in win_rb if s.t is not None)
        shown = 0
        cause = defaultdict(int)
        for r in sorted(viol, key=lambda r: -r["ttft"]):
            send = t0 + r["t_rel"]
            lo = bisect.bisect_left(keys, send - 0.05)
            hi = bisect.bisect_right(keys, send + r["ttft"])
            seg = [x[1] for x in by_t[lo:hi]]
            n_rb = bisect.bisect_right(rb_t, send + r["ttft"]) - bisect.bisect_left(rb_t, send)
            act = sum(s.actual for s in seg)
            prd = sum(s.pred_raw for s in seg)
            n_ft_steps = sum(1 for s in seg if s.t_ft > 0)
            ft_tok = sum(s.t_ft for s in seg)
            bwd = sum(1 for s in seg if s.pending_bwd)
            gap = r["ttft"] - act
            # classify
            if not seg:
                c = "no steps timed in the window (frontend / queue / idle?)"
            elif act > 1.3 * prd and act > 0.5 * r["ttft"]:
                c = "steps ran SLOWER than predicted" + (
                    " (backward in flight)" if bwd > len(seg) / 2 else "")
            elif ft_tok > 0 and prd > 0.5 * r["ttft"]:
                c = "predicted correctly but FT admitted into the wait"
            elif gap > 0.5 * r["ttft"]:
                c = "most of the TTFT is NOT step time (queueing / frontend)"
            else:
                c = "many inference steps (load)"
            cause[c] += 1
            if shown < top:
                shown += 1
                out(f"  req {r['idx']:4d} t={r['t_rel']:7.2f}s ttft {r['ttft'] * 1e3:6.0f}ms | "
                    f"{len(seg)} steps: actual Σ {act * 1e3:6.0f}ms pred Σ {prd * 1e3:6.0f}ms "
                    f"unaccounted {gap * 1e3:6.0f}ms | ft steps {n_ft_steps} "
                    f"({ft_tok:.0f} tok) bwd-in-flight {bwd} rollbacks {n_rb} → {c}")
                for s in seg[:6]:
                    out(f"        seq {s.seq} +{(s.t - send) * 1e3:6.0f}ms {s.regime:<11} "
                        f"t_in {s.t_in:4.0f} ft {s.t_ft:3.0f} b_d {s.b_d:2d} "
                        f"pred {ms(s.pred_raw)} actual {ms(s.actual)} "
                        f"bwd {int(bool(s.pending_bwd))} paused {int(bool(s.paused_bwd))}")
                if len(seg) > 6:
                    out(f"        … {len(seg) - 6} more steps")
        if viol:
            out("  causes:")
            for c, n in sorted(cause.items(), key=lambda x: -x[1]):
                out(f"    {n:4d}  {c}")
        summary["violations"] = len(viol)
        summary["cause"] = dict(cause)

    # 7. 1-second bins (only printed when the trace is long enough to matter)
    if w0 is not None and w1 - w0 > 30:
        out("\n[7] per-5s bins: ratio p50 / p90 of inf_prefill steps, eager steps, "
            "FT tokens, requests, backward-in-flight share, TTFT violations")
        out("   t(s)   n_inf p50  p90 | n_eag p50  p90 | ft_tok  reqs bwd%  viol")
        nb = int((w1 - w0) // 5) + 1
        bins = [[] for _ in range(nb)]
        for s in win:
            bins[int((s.t - w0) // 5)].append(s)
        rq = [[] for _ in range(nb)]
        for r in results:
            i = int(r["t_rel"] - (w0 - t0)) // 5
            if 0 <= i < nb:
                rq[i].append(r)
        for i in range(nb):
            b = bins[i]
            if not b:
                continue
            inf = [s.ratio for s in b if s.regime == "inf_prefill"]
            eag = [s.ratio for s in b if s.regime == "eager"]
            ft_tok = sum(s.t_ft for s in b)
            bwd = sum(1 for s in b if s.pending_bwd) / len(b)
            v = sum(1 for r in rq[i] if r["ttft"] is not None and r["ttft"] > slo)
            out(f"  {i * 5:5d}  {len(inf):5d} {pct(inf, .5):4.2f} {pct(inf, .9):4.2f} | "
                f"{len(eag):5d} {pct(eag, .5):4.2f} {pct(eag, .9):4.2f} | "
                f"{ft_tok:6.0f} {len(rq[i]):5d} {bwd * 100:4.0f}  {v:4d}")
    out("=" * 100)
    summary["window"] = (w0, w1)
    return summary


# ─── plot (optional) ────────────────────────────────────────────────────────

_SERIES = {"inf_prefill": "#2a78d6", "eager": "#eb6834", "decode_only": "#1baf7a",
           "decode_bwd": "#eda100"}
REGIMES = tuple(_SERIES)


def plot(steps: list[Step], results, t0, slo, window, out_png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w0, w1 = window
    win = [s for s in steps if s.kind == "step" and s.pred_raw and s.actual
           and s.t is not None and (w0 is None or w0 <= s.t <= w1)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    ax = axes[0]
    for reg, col in _SERIES.items():
        rs = [s for s in win if s.regime == reg]
        if not rs:
            continue
        ax.scatter([s.pred_raw * 1e3 for s in rs], [s.actual * 1e3 for s in rs],
                   s=9, alpha=0.45, color=col, edgecolors="none", label=f"{reg} (n={len(rs)})")
    lim = max([s.actual for s in win] + [s.pred_raw for s in win] + [1e-3]) * 1e3
    ax.plot([0.5, lim], [0.5, lim], color="#9a9a94", lw=1, ls="--", label="actual = pred")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("predicted step time (ms, raw model)"); ax.set_ylabel("measured (ms)")
    ax.set_title("predicted vs measured"); ax.legend(frameon=False, fontsize=8)
    ax.grid(True, which="both", color="#e6e6e2", lw=0.6)

    ax = axes[1]
    base = w0 if w0 is not None else min(s.t for s in win)
    for reg, col in _SERIES.items():
        rs = [s for s in win if s.regime == reg]
        if rs:
            ax.scatter([s.t - base for s in rs], [s.ratio for s in rs], s=7,
                       alpha=0.5, color=col, edgecolors="none", label=reg)
    pend = [s for s in win if s.pending_bwd]
    if pend:
        ax.scatter([s.t - base for s in pend], [0.35] * len(pend), s=3, color="#52514e",
                   marker="|", label="backward in flight")
    if results and t0 is not None:
        viol = [r for r in results if r["ttft"] is not None and r["ttft"] > slo]
        for r in viol:
            ax.axvline(t0 + r["t_rel"] - base, color="#e34948", lw=0.8, alpha=0.5)
        if viol:
            ax.plot([], [], color="#e34948", lw=0.8, label=f"TTFT > SLO ({len(viol)})")
    ax.axhline(1.0, color="#9a9a94", lw=1, ls="--")
    ax.set_yscale("log"); ax.set_ylim(0.3, max(4, max(s.ratio for s in win) * 1.1))
    ax.set_xlabel("time since first request (s)"); ax.set_ylabel("measured / predicted")
    ax.set_title("ratio over the run"); ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(True, color="#e6e6e2", lw=0.6)

    ax = axes[2]
    for lab, col, sel in (("backward in flight", "#eb6834", lambda s: s.pending_bwd),
                          ("no backward", "#2a78d6", lambda s: s.pending_bwd is False)):
        rs = sorted(s.ratio for s in win if sel(s) and s.regime in ("inf_prefill", "eager"))
        if rs:
            ax.plot(rs, [i / len(rs) for i in range(len(rs))], color=col, lw=2,
                    label=f"{lab} (n={len(rs)})")
    ax.axvline(1.0, color="#9a9a94", lw=1, ls="--")
    ax.set_xlim(0.8, max(1.5, pct([s.ratio for s in win if s.regime != "decode_only"], .999) * 1.05))
    ax.set_xlabel("measured / predicted (prefill-carrying steps)")
    ax.set_ylabel("CDF"); ax.set_title("interference check")
    ax.legend(frameon=False, fontsize=8); ax.grid(True, which="both", color="#e6e6e2", lw=0.6)
    for a in axes:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"[analyze] plot → {out_png}")


# ─── main ───────────────────────────────────────────────────────────────────

def _find(prefix: str, family: str, tp: int, mode: str) -> str | None:
    cands = sorted(OUTPUT_DIR.glob(f"{prefix}_{family}_tp{tp}_co_*_{mode}.csv"))
    cands = [c for c in cands if c.name.endswith(f"_{mode}.csv")]
    return str(cands[-1]) if cands else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", help="step_trace CSV (else derived from family/tp/mode)")
    ap.add_argument("--family", default="qwen3-14b")
    ap.add_argument("--tp", type=int, default=2)
    for m in MODES:
        ap.add_argument(f"--{m}", dest=f"mode_{m.replace('-', '_')}", action="store_true")
    ap.add_argument("--ttft-slo", type=float, default=None)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--report", help="also write the text report to this path")
    args = ap.parse_args()

    chosen = [m for m in MODES if getattr(args, f"mode_{m.replace('-', '_')}")]
    mode = chosen[0] if chosen else "tight"
    trace = args.trace or _find("step_trace", args.family, args.tp, mode)
    if not trace or not os.path.exists(trace):
        print(f"[analyze] no step trace for {args.family} tp{args.tp} {mode} "
              f"(run auto_benchmark_tp.py --co --step-trace)", file=sys.stderr)
        return 1
    suffix = Path(trace).name[len("step_trace"):-len(".csv")]
    results_p = OUTPUT_DIR / f"timeline_results{suffix}.csv"
    meta_p = OUTPUT_DIR / f"bench_meta{suffix}.json"
    bwd_p = OUTPUT_DIR / f"bwd_log{suffix}.csv"
    log_p = OUTPUT_DIR / f"server{suffix}.log"
    slo = args.ttft_slo or _SLO_DEFAULT.get(args.family, 0.4)

    steps = load_trace(trace)
    results = load_results(str(results_p)) if results_p.exists() else []
    t0 = None
    if meta_p.exists():
        with open(meta_p) as f:
            t0 = dt.datetime.fromisoformat(json.load(f)["t_first_wall_iso"]).timestamp()
    bwd_rows = load_bwd_log(str(bwd_p))
    cycles, aborts = load_server_log(str(log_p))
    print(f"[analyze] trace {trace}\n[analyze] results {results_p if results else '-'} "
          f"| t0 {'set' if t0 else '-'} | slo {slo}s")

    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    summary = report(steps, results, t0, slo, bwd_rows, cycles, aborts, args.top, out)
    if args.report:
        with open(args.report, "w") as f:
            f.write("\n".join(lines) + "\n")
    if args.plot:
        plot(steps, results, t0, slo, summary["window"],
             str(Path(trace).with_suffix(".png")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
