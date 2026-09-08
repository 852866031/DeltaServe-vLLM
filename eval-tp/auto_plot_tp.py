#!/usr/bin/env python3
"""auto_plot_tp.py — the 4-panel per-mode plots for eval-tp benchmark runs.

Same figure as eval/auto_plot.py (request timeline / E2E latency vs time /
throughput with the finetune band / TTFT SLO satisfaction) — the figure
builder is imported from there — over the per-family, per-TP output names
that eval-tp/auto_benchmark_tp.py writes:

    eval-tp/output/timeline_results_<family>_tp<N>_co_factor_<f>_phase_<p>_<mode>.csv
    eval-tp/output/bwd_log_…  bench_meta_…  (same stem)
    eval-tp/output/timeline_results_<family>_tp<N>_<mode>.csv   (inference-only overlay, if run)

One PNG per mode under eval-tp/plots/:

    <mode>_<family>_tp<N>_co_factor_<f>_phase_<p>.png

Usage:
    python eval-tp/auto_plot_tp.py --family qwen3-14b --tp 2            # every mode with results
    python eval-tp/auto_plot_tp.py --family llama3 --tp 2 --mode loose
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # eval-tp/
_ROOT = _HERE.parent
_EVAL = _ROOT / "eval"
sys.path.insert(0, str(_EVAL))
sys.path.insert(0, str(_HERE))

from auto_plot import (  # noqa: E402
    DEFAULT_SLO_FALLBACK, MODE_COLORS, make_figure_for_mode, read_slo,
    read_ttft_slo,
)
from auto_benchmark_tp import MODES, OUTPUT_DIR, TIMELINES_DIR  # noqa: E402
from launch_deltaserve import PRESETS  # noqa: E402

PLOTS_DIR = _HERE / "plots"
# One colour for every mode so the three figures read as a set (the shared
# builder colours by mode; inference = blue, finetune = the builder's orange).
INFERENCE_COLOR = "tab:blue"
for _m in MODES:
    MODE_COLORS[_m] = INFERENCE_COLOR


def discover_runs(output_dir: Path, family: str, tp: int, mode: str) -> list[str]:
    """Base suffixes (without ``_<mode>``) of the co-serving results present
    for (family, tp, mode): ``_<family>_tp<N>_co[_factor_<f>_phase_<p>]``."""
    stem = f"_{family}_tp{tp}_co"
    pat = str(output_dir / f"timeline_results{stem}*_{mode}.csv")
    rx = re.compile(r"^timeline_results(" + re.escape(stem) + r"(?:_factor_[^_]+_phase_[^_]+)?)_"
                    + re.escape(mode) + r"\.csv$")
    bases = []
    for path in glob.glob(pat):
        m = rx.match(os.path.basename(path))
        if m:
            bases.append(m.group(1))
    return sorted(set(bases))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=sorted(PRESETS), default="qwen3-14b")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--mode", choices=list(MODES) + ["all"], default="all")
    ap.add_argument("--config", default=None,
                    help="Serving YAML for the SLO values (default: the family preset).")
    ap.add_argument("--slo", type=float, default=None, help="TTFT SLO (s); overrides the YAML.")
    ap.add_argument("--window", type=float, default=5.0,
                    help="TTFT-satisfaction window (s).")
    ap.add_argument("--throughput-window", type=float, default=3,
                    help="Throughput smoothing window (s).")
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--plots-dir", default=str(PLOTS_DIR))
    args = ap.parse_args()

    cfg_yaml = args.config or str(PRESETS[args.family])
    slo_s = args.slo if args.slo is not None else read_ttft_slo(cfg_yaml)
    if slo_s is None:
        slo_s = DEFAULT_SLO_FALLBACK
    avg_tbt_slo = read_slo(cfg_yaml, "avg_tbt_slo")
    print(f"[auto_plot_tp] family={args.family} tp={args.tp} | TTFT SLO={slo_s:.3f}s "
          f"(from {os.path.basename(cfg_yaml)})", flush=True)

    out_dir, plots_dir = Path(args.output_dir), Path(args.plots_dir)
    modes = list(MODES) if args.mode == "all" else [args.mode]
    wrote = 0
    for mode in modes:
        bases = discover_runs(out_dir, args.family, args.tp, mode)
        if not bases:
            print(f"[auto_plot_tp] {mode}: no co-serving results for "
                  f"{args.family} tp={args.tp} (run auto_benchmark_tp.py --co --{mode})")
            continue
        for base in bases:
            tag = base.lstrip("_")
            knobs = base[len(f"_{args.family}_tp{args.tp}_"):]   # co_factor_<f>_phase_<p>
            out_path = plots_dir / f"{mode}_{tag}.png"
            try:
                make_figure_for_mode(
                    mode=mode, base_suffix=base, output_dir=str(out_dir),
                    plots_dir=str(plots_dir), timeline_csv_dir=str(TIMELINES_DIR),
                    out_path=str(out_path), slo_s=slo_s, window_s=args.window,
                    throughput_window_s=args.throughput_window,
                    avg_tbt_slo=avg_tbt_slo, factor_tag=None,
                    timeline_csv=str(TIMELINES_DIR / MODES[mode]),
                    infonly_csv=str(out_dir / f"timeline_results_{args.family}_tp{args.tp}_{mode}.csv"),
                    title=f"{mode}  ({args.family}, tp={args.tp}, {knobs})")
                wrote += 1
            except Exception as e:  # noqa: BLE001
                print(f"[auto_plot_tp] {mode} ({tag}): {e}")
    print(f"[auto_plot_tp] wrote {wrote} figure(s) to {plots_dir}")
    return 0 if wrote else 1


if __name__ == "__main__":
    sys.exit(main())
