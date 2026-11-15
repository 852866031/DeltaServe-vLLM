#!/usr/bin/env python3
"""margin_ab.py — A/B of the SLO admission margin (``slo.admission_margin``).

Each arm is the family preset with ``slo.admission_margin`` / ``_k`` set; runs
are interleaved (arm order rotates every rep, so drift spreads over the arms)
and each one is a full ``auto_benchmark_tp.py --co --step-trace`` replay of the
stress timeline (``scripts/timelines/margin_stress_gen.py``). Then every run is
scored on what the margin is for:

  over-limit   FT-carrying steps with running decodes whose measured GPU time
               exceeded ``max_tbt_slo``: admission promised ≤ the limit and
               missed. (Inference-only steps with decodes are the control.)
  under-pred   the same steps' actual / admission-prediction ratio (p50, p95):
               how wrong the admitted prediction was.
  TTFT         SLO attainment and p50 / p99 over the recorded timeline.
  FT           trained tokens / s over the recorded window (backward log).

    python eval-tp/margin_ab.py --arms legacy relative:1.5 absolute:1.5 --reps 2
    python eval-tp/margin_ab.py --analyze-only --out eval-tp/output/margin_ab/<tag>
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
PY = sys.executable


def arm_name(spec: str) -> str:
    return spec.replace(":", "_k")


def write_cfg(base: Path, spec: str, out: Path) -> Path:
    """The base YAML with ``slo.admission_margin[_k]`` set, its relative paths
    made absolute (the loader resolves them against the YAML's own dir)."""
    from deltaserve.config_loader import load_yaml_config
    mode, _, k = spec.partition(":")
    raw = load_yaml_config(base, verbose=False)
    slo = raw.setdefault("slo", {})
    slo["admission_margin"] = mode
    if k:
        slo["admission_margin_k"] = float(k)
    cfg = out / f"cfg_{arm_name(spec)}.yaml"
    cfg.write_text(yaml.safe_dump(raw, sort_keys=False))
    return cfg


def run_one(args, cfg: Path, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [PY, str(_HERE / "auto_benchmark_tp.py"), "--family", args.family,
           "--tp", str(args.tp), "--co", "--step-trace", "--config", str(cfg),
           "--timeline_csv", str(Path(args.timeline).resolve()),
           "--out-dir", str(run_dir), "--port", str(args.port)]
    with open(run_dir / "driver.log", "w") as log:
        t0 = time.time()
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=_ROOT)
    print(f"  {run_dir.name}: exit {rc} in {time.time() - t0:.0f} s", flush=True)
    if rc != 0:
        raise SystemExit(f"run failed: see {run_dir / 'driver.log'}")


def _one(run_dir: Path, pattern: str) -> Path | None:
    hits = sorted(run_dir.glob(pattern))
    return hits[0] if hits else None


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))] if xs else float("nan")


def score(run_dir: Path, max_tbt: float, ttft_slo: float) -> dict:
    meta = json.loads(_one(run_dir, "bench_meta*.json").read_text())
    t0 = dt.datetime.fromisoformat(meta["t_first_wall_iso"]).timestamp()
    res = list(csv.DictReader(open(_one(run_dir, "timeline_results*.csv"))))
    ok = [r for r in res if r["status"] in ("ok", "200", "success") or r["ttft_s"]]
    ttft = [float(r["ttft_s"]) for r in ok if r["ttft_s"]]
    t_end = t0 + max(float(r["t_rel_s"]) + float(r["latency_s"] or 0) for r in ok)
    ft_dec, inf_dec, ratio = [], [], []
    for r in csv.DictReader(open(_one(run_dir, "step_trace*.csv"))):
        if r["kind"] != "step" or not r["actual"] or not r["t_exec"]:
            continue
        if not (t0 <= float(r["t_exec"]) <= t_end) or int(float(r["b_d"] or 0)) == 0:
            continue
        act = float(r["actual"])
        if float(r["t_ft"] or 0) > 0:
            ft_dec.append(act)
            if r["t_admit"]:
                ratio.append(act / max(1e-9, float(r["t_admit"])))
        elif float(r["t_in"] or 0) > 0:
            inf_dec.append(act)
    toks = 0
    bw = _one(run_dir, "bwd_log*.csv")
    for r in csv.DictReader(open(bw)):
        ts = dt.datetime.fromisoformat(r["timestamp"]).timestamp()
        if t0 <= ts <= t_end:
            toks += int(r["batch_tokens"])
    return {
        "ft_steps": len(ft_dec),
        "ft_over": sum(a > max_tbt for a in ft_dec),
        "inf_steps": len(inf_dec),
        "inf_over": sum(a > max_tbt for a in inf_dec),
        "ratio_p50": pct(ratio, 0.5), "ratio_p95": pct(ratio, 0.95),
        "ttft_met": sum(t <= ttft_slo for t in ttft) / max(1, len(ttft)),
        "ttft_p50": pct(ttft, 0.5), "ttft_p99": pct(ttft, 0.99),
        "ft_tok_s": toks / max(1e-9, t_end - t0),
        "n_req": len(ttft),
    }


def report(out: Path, arms: list[str], reps: int, max_tbt: float, ttft_slo: float) -> None:
    rows = {}
    for spec in arms:
        for i in range(reps):
            d = out / f"{arm_name(spec)}_r{i}"
            if d.exists() and _one(d, "step_trace*.csv"):
                rows.setdefault(spec, []).append(score(d, max_tbt, ttft_slo))
    json.dump(rows, open(out / "scores.json", "w"), indent=1)
    print(f"\n{'arm':<16}{'FT steps over limit':>22}{'inf-only over':>16}"
          f"{'actual/pred p50·p95':>22}{'TTFT met':>10}{'TTFT p99':>10}{'FT tok/s':>10}")
    for spec, rs in rows.items():
        for r in rs:
            print(f"{arm_name(spec):<16}"
                  f"{r['ft_over']:>6} / {r['ft_steps']:<5} ({r['ft_over'] / max(1, r['ft_steps']):5.1%})"
                  f"{r['inf_over'] / max(1, r['inf_steps']):>15.1%}"
                  f"{r['ratio_p50']:>12.3f} · {r['ratio_p95']:.3f}"
                  f"{r['ttft_met']:>10.1%}{r['ttft_p99'] * 1e3:>8.0f}ms{r['ft_tok_s']:>10.0f}")
        if len(rs) > 1:
            fo = sum(r["ft_over"] for r in rs) / max(1, sum(r["ft_steps"] for r in rs))
            print(f"{'  mean':<16}{fo:>20.1%}{'':>16}{'':>22}"
                  f"{statistics.mean(r['ttft_met'] for r in rs):>10.1%}"
                  f"{statistics.mean(r['ttft_p99'] for r in rs) * 1e3:>8.0f}ms"
                  f"{statistics.mean(r['ft_tok_s'] for r in rs):>10.0f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["legacy", "relative:1.5"],
                    help="mode[:k] per arm (legacy | relative | absolute)")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--family", default="qwen3-14b")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--config", default=None, help="base YAML (default: the family preset)")
    ap.add_argument("--timeline", default=str(_ROOT / "data/timelines/5090/timeline_margin_stress.csv"))
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, str(_HERE))
    from launch_deltaserve import PRESETS
    base = Path(args.config or PRESETS[args.family]).resolve()
    out = Path(args.out or _HERE / "output" / "margin_ab" / time.strftime("%Y%m%d-%H%M%S")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw = yaml.safe_load(base.read_text())
    max_tbt = float(raw["slo"]["max_tbt_slo"])
    ttft_slo = float(raw["slo"]["ttft_slo"])
    if not args.analyze_only:
        print(f"out {out}\nbase {base}\narms {args.arms} × {args.reps}", flush=True)
        for i in range(args.reps):
            order = args.arms[i % len(args.arms):] + args.arms[:i % len(args.arms)]
            for spec in order:
                cfg = write_cfg(base, spec, out)
                run_one(args, cfg, out / f"{arm_name(spec)}_r{i}")
    report(out, args.arms, args.reps, max_tbt, ttft_slo)


if __name__ == "__main__":
    main()
