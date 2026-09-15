#!/usr/bin/env python3
"""compare_runs.py — side-by-side metrics of two co-serving runs (e.g. the
eager-FT baseline vs the mixed-fwd-cuda-graph run) from their output files.

Each run is a directory holding the bench outputs with a common suffix:
timeline_results<suffix>.csv, step_trace<suffix>.csv, bwd_log<suffix>.csv,
bench_meta<suffix>.json, server<suffix>.log.

    python eval-tp/compare_runs.py --suffix _qwen3-14b_tp2_co_factor_off_phase_prefill_tight \
        --a /path/baseline --b eval-tp/output [--ttft-slo 0.4] [--tbt-slo 0.05]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_step_trace import load_trace, pct  # noqa: E402

_BWD_RE = re.compile(r"\[backward\]\s+(\d+(?:\.\d+)?)\s*ms")


def load(dirpath: str, suffix: str, ttft_slo: float, tbt_slo: float) -> dict:
    d = Path(dirpath)
    out: dict = {}
    res = list(csv.DictReader(open(d / f"timeline_results{suffix}.csv")))
    ttft = sorted(float(r["ttft_s"]) for r in res if r["ttft_s"])
    out["requests"] = len(res)
    out["ttft_ok_%"] = 100.0 * sum(1 for t in ttft if t <= ttft_slo) / max(1, len(ttft))
    out["ttft_p50_ms"] = pct(ttft, .5) * 1e3
    out["ttft_p95_ms"] = pct(ttft, .95) * 1e3
    out["ttft_p99_ms"] = pct(ttft, .99) * 1e3
    out["worst_tbt>slo"] = sum(1 for r in res if float(r["worst_tbt_s"]) > tbt_slo)
    out["avg_tbt_p50_ms"] = pct(sorted(float(r["avg_tbt_s"]) for r in res), .5) * 1e3
    t0 = dt.datetime.fromisoformat(json.load(open(d / f"bench_meta{suffix}.json"))["t_first_wall_iso"]).timestamp()
    w0 = t0 + min(float(r["t_rel_s"]) for r in res)
    w1 = t0 + max(float(r["t_rel_s"]) + float(r["latency_s"]) for r in res)
    out["window_s"] = w1 - w0
    bwd = list(csv.DictReader(open(d / f"bwd_log{suffix}.csv")))
    toks = sum(int(r["batch_tokens"]) for r in bwd)
    out["bwd_cycles"] = len(bwd)
    out["ft_tokens"] = toks
    out["ft_tok/s"] = toks / (w1 - w0)
    if bwd:
        out["loss_first"] = float(bwd[0]["batch_loss"])
        out["loss_last"] = float(bwd[-1]["batch_loss"])
    steps = load_trace(str(d / f"step_trace{suffix}.csv"))
    win = [s for s in steps if s.kind == "step" and s.actual and s.t and w0 <= s.t <= w1]
    ft_steps = [s for s in win if s.t_ft > 0]
    mixed = [s for s in ft_steps if not s.ft_only]
    ftonly = [s for s in ft_steps if s.ft_only]
    pref = [s for s in win if s.t_in > s.t_ft]        # steps with inference prefill
    out["steps"] = len(win)
    out["prefill_steps"] = len(pref)
    out["prefill_steps_with_ft"] = sum(1 for s in pref if s.t_ft > 0)
    out["mixed_ft_steps"] = len(mixed)
    out["mixed_graphed"] = sum(1 for s in mixed if s.was_graph)
    out["mixed_step_ms_p50"] = pct([s.actual for s in mixed], .5) * 1e3 if mixed else float("nan")
    out["mixed_ft_tok_p50"] = pct([s.t_ft for s in mixed], .5) if mixed else float("nan")
    out["mixed_host_ms_p50"] = pct([s.host for s in mixed if s.host is not None], .5) * 1e3 if mixed else float("nan")
    out["ftonly_steps"] = len(ftonly)
    out["ftonly_step_ms_p50"] = pct([s.actual for s in ftonly], .5) * 1e3 if ftonly else float("nan")
    dec = [s for s in win if s.regime == "decode_only"]
    out["decode_step_ms_p50"] = pct([s.actual for s in dec], .5) * 1e3 if dec else float("nan")
    ip = [s for s in win if s.regime == "inf_prefill"]
    out["inf_prefill_ms_p50"] = pct([s.actual for s in ip], .5) * 1e3 if ip else float("nan")
    for reg in ("eager", "ft_mixed"):
        rs = [s for s in win if s.regime == reg and s.ratio]
        out[f"{reg}_ratio_p50"] = pct([s.ratio for s in rs], .5) if rs else float("nan")
        out[f"{reg}_n"] = len(rs)
    logp = d / f"server{suffix}.log"
    if logp.exists():
        cyc = [float(m.group(1)) for line in open(logp, errors="replace")
               for m in [_BWD_RE.search(line)] if m]
        out["bwd_cycle_ms_p50"] = pct(cyc, .5) if cyc else float("nan")
        txt = open(logp, errors="replace").read()
        m = re.search(r"GPU KV cache size: ([\d,]+) tokens", txt)
        out["kv_tokens"] = int(m.group(1).replace(",", "")) if m else None
        m = re.search(r"Graph capturing finished in (\d+) secs, took ([\d.]+) GiB", txt)
        out["graph_gib"] = float(m.group(2)) if m else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suffix", required=True)
    ap.add_argument("--a", required=True, help="baseline dir")
    ap.add_argument("--b", required=True, help="new dir")
    ap.add_argument("--ttft-slo", type=float, default=0.4)
    ap.add_argument("--tbt-slo", type=float, default=0.05)
    ap.add_argument("--label-a", default="eager FT (baseline)")
    ap.add_argument("--label-b", default="graphed mixed FT")
    args = ap.parse_args()
    A = load(args.a, args.suffix, args.ttft_slo, args.tbt_slo)
    B = load(args.b, args.suffix, args.ttft_slo, args.tbt_slo)
    print(f"{'metric':<26}{args.label_a:>22}{args.label_b:>22}")
    for k in A:
        va, vb = A.get(k), B.get(k)
        f = (lambda v: "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v)))
        print(f"{k:<26}{f(va):>22}{f(vb):>22}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
