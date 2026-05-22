"""compare_kv_plot.py — unified vs packed_kv allocator comparison.

Compares two auto_benchmark runs that differ only in --packed_kv. Given a
suffix that identifies a graph configuration (e.g. '_decode_prefill_bwd'),
this looks up four CSVs under eval/llama3/output/:

  timeline_results<suffix>.csv         (unified)
  bwd_log<suffix>.csv                  (unified)
  timeline_results<suffix>_kv.csv      (packed_kv)
  bwd_log<suffix>_kv.csv               (packed_kv)

and produces a 4-panel comparison PNG (request timeline | TTFT percentile
| latency vs time | cumulative finetune tokens).

Run the upstream benchmarks first, e.g.:
  python auto_benchmark.py --co --decode_graph --prefill_graph --bwd_graph
  python auto_benchmark.py --co --decode_graph --prefill_graph --bwd_graph --packed_kv

Default suffix: '_decode_prefill_bwd' (the --co --decode_graph
--prefill_graph --bwd_graph configuration).
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt

from bwd_graph_plot import (
    ensure_exists,
    load_results,
    parse_bwd_log_csv,
    plot_ttft_percentile,
    plot_latency_vs_time,
    plot_bwd_cumulative,
)
from compare_graphs_plot import load_timeline, plot_request_timeline


_HERE = os.path.dirname(os.path.abspath(__file__))


def _detect_gpu_subdir() -> str:
    """Return the timelines/ subdirectory name matching the local GPU.
    Greps `nvidia-smi`; falls back to '5090' on failure."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True, timeout=2.0,
        )
        name = (out.strip().splitlines() or [""])[0].upper()
        if "A100" in name:
            return "A100"
        if "5090" in name:
            return "5090"
    except Exception:
        pass
    return "5090"


_DEFAULT_GPU_SUBDIR = _detect_gpu_subdir()
# Script lives under llama3/scripts/; output/, plots/, and timelines/ are
# one level up from _HERE.
OUTPUT_DIR = os.path.abspath(os.path.join(_HERE, "..", "output"))
PLOTS_DIR = os.path.abspath(os.path.join(_HERE, "..", "plots"))
DEFAULT_TIMELINE_CSV = os.path.abspath(os.path.join(
    _HERE, "..", "timelines", _DEFAULT_GPU_SUBDIR, "timeline_live.csv"))

UNIFIED_LABEL = "unified"
PACKED_LABEL = "packed_kv"
UNIFIED_COLOR = "tab:blue"
PACKED_COLOR = "tab:purple"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--suffix",
        default="_decode_prefill_bwd",
        help="Tag suffix from auto_benchmark identifying the configuration "
             "(e.g. '_decode_prefill_bwd'). The script looks for "
             "timeline_results<suffix>.csv (unified) and "
             "timeline_results<suffix>_kv.csv (packed_kv).",
    )
    ap.add_argument("--output-dir", default=OUTPUT_DIR,
                    help="Where auto_benchmark writes its CSVs.")
    ap.add_argument("--plots-dir", default=PLOTS_DIR,
                    help="Where to write the PNG.")
    ap.add_argument("--timeline-csv", default=DEFAULT_TIMELINE_CSV,
                    help="Scheduled-request timeline (input to auto_benchmark). "
                         "Drives the leftmost req/s + tokens/s subplot.")
    ap.add_argument("--out", default=None,
                    help="Output PNG path. Default: plots/compare_kv<suffix>.png")
    args = ap.parse_args()

    suffix = args.suffix
    unified_results = os.path.join(args.output_dir, f"timeline_results{suffix}.csv")
    unified_log     = os.path.join(args.output_dir, f"bwd_log{suffix}.csv")
    packed_results  = os.path.join(args.output_dir, f"timeline_results{suffix}_kv.csv")
    packed_log      = os.path.join(args.output_dir, f"bwd_log{suffix}_kv.csv")

    for p in (args.timeline_csv, unified_results, unified_log,
              packed_results, packed_log):
        try:
            ensure_exists(p)
        except FileNotFoundError as e:
            sys.exit(f"[compare_kv_plot] missing input: {e}")

    os.makedirs(args.plots_dir, exist_ok=True)
    out_path = args.out or os.path.join(args.plots_dir, f"compare_kv{suffix}.png")

    timeline_df = load_timeline(args.timeline_csv)
    u_results = load_results(unified_results)
    p_results = load_results(packed_results)
    u_log     = parse_bwd_log_csv(unified_log)
    p_log     = parse_bwd_log_csv(packed_log)

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    labels = (UNIFIED_LABEL, PACKED_LABEL)
    colors = (UNIFIED_COLOR, PACKED_COLOR)
    plot_request_timeline(axes[0], timeline_df)
    plot_ttft_percentile(axes[1], u_results, p_results, labels=labels, colors=colors)
    plot_latency_vs_time(axes[2], u_results, p_results, labels=labels, colors=colors)
    plot_bwd_cumulative(axes[3], u_log, p_log, labels=labels, colors=colors)

    config_tag = suffix.lstrip("_") or "baseline"
    fig.suptitle(f"{UNIFIED_LABEL}  vs  {PACKED_LABEL}  ({config_tag})")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[compare_kv_plot] Wrote {out_path}")


if __name__ == "__main__":
    main()
