"""compare_emotion_alpaca_plot.py — emotion vs Alpaca-1000 comparison.

Compares two auto_benchmark runs that share --decode_graph --prefill_graph
--bwd_graph --packed_kv (the default) and differ only in the dataset
(--alpaca on/off). Generates one PNG per schedule shape (tight, loose).

Expected CSV layout under eval/llama3/output/ (run auto_benchmark.py first):

  timeline_results_decode_prefill_bwd_kv_<shape>.csv         (emotion)
  bwd_log_decode_prefill_bwd_kv_<shape>.csv                  (emotion)
  timeline_results_decode_prefill_bwd_kv_alpaca_<shape>.csv  (alpaca)
  bwd_log_decode_prefill_bwd_kv_alpaca_<shape>.csv           (alpaca)

where <shape> is 'tight' or 'loose'.

Each output PNG is the standard 4-panel comparison layout:
  [request timeline] [TTFT percentile] [latency vs time] [bwd cumulative tokens]

Run prerequisites (both shapes):
  python auto_benchmark.py --co --decode_graph --prefill_graph --bwd_graph --tight
  python auto_benchmark.py --co --decode_graph --prefill_graph --bwd_graph --loose
  python auto_benchmark.py --co --alpaca --decode_graph --prefill_graph --bwd_graph --tight
  python auto_benchmark.py --co --alpaca --decode_graph --prefill_graph --bwd_graph --loose

Then:
  python compare_emotion_alpaca_plot.py
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
# Script lives under llama3/scripts/; output/, plots/, and timelines/<gpu>/
# are one level up from _HERE.
OUTPUT_DIR = os.path.abspath(os.path.join(_HERE, "..", "output"))
PLOTS_DIR = os.path.abspath(os.path.join(_HERE, "..", "plots"))
TIMELINES_DIR = os.path.abspath(os.path.join(
    _HERE, "..", "timelines", _DEFAULT_GPU_SUBDIR))

EMOTION_LABEL = "emotion"
ALPACA_LABEL = "alpaca"
EMOTION_COLOR = "tab:blue"
ALPACA_COLOR = "tab:orange"

# Shared graph + allocator config the comparison fixes.
BASE_TAG = "_decode_prefill_bwd_kv"


def _shape_files(output_dir: str, shape: str):
    """Return (emotion_results, emotion_log, alpaca_results, alpaca_log) paths
    for the given schedule shape ('tight' or 'loose')."""
    emo_suffix = f"{BASE_TAG}_{shape}"
    alp_suffix = f"{BASE_TAG}_alpaca_{shape}"
    return (
        os.path.join(output_dir, f"timeline_results{emo_suffix}.csv"),
        os.path.join(output_dir, f"bwd_log{emo_suffix}.csv"),
        os.path.join(output_dir, f"timeline_results{alp_suffix}.csv"),
        os.path.join(output_dir, f"bwd_log{alp_suffix}.csv"),
    )


def _timeline_for_shape(here: str, shape: str) -> str:
    """auto_benchmark uses timeline_<shape>.csv to schedule requests for the
    --tight / --loose modes; mirror that here for the request-timeline panel."""
    return os.path.join(here, f"timeline_{shape}.csv")


def render_one(shape: str, output_dir: str, plots_dir: str,
               timeline_csv: str, out_path: str) -> None:
    emo_results, emo_log, alp_results, alp_log = _shape_files(output_dir, shape)

    for p in (timeline_csv, emo_results, emo_log, alp_results, alp_log):
        try:
            ensure_exists(p)
        except FileNotFoundError as e:
            sys.exit(f"[compare_emotion_alpaca_plot] missing input: {e}")

    timeline_df = load_timeline(timeline_csv)
    e_results = load_results(emo_results)
    a_results = load_results(alp_results)
    e_log = parse_bwd_log_csv(emo_log)
    a_log = parse_bwd_log_csv(alp_log)

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    labels = (EMOTION_LABEL, ALPACA_LABEL)
    colors = (EMOTION_COLOR, ALPACA_COLOR)
    plot_request_timeline(axes[0], timeline_df)
    plot_ttft_percentile(axes[1], e_results, a_results, labels=labels, colors=colors)
    plot_latency_vs_time(axes[2], e_results, a_results, labels=labels, colors=colors)
    plot_bwd_cumulative(axes[3], e_log, a_log, labels=labels, colors=colors)

    fig.suptitle(
        f"{EMOTION_LABEL}  vs  {ALPACA_LABEL}  "
        f"(decode+prefill+bwd graph, packed_kv, {shape})"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[compare_emotion_alpaca_plot] Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--output-dir", default=OUTPUT_DIR,
                    help="Where auto_benchmark writes its CSVs.")
    ap.add_argument("--plots-dir", default=PLOTS_DIR,
                    help="Where to write the PNGs.")
    ap.add_argument("--shapes", nargs="+", default=("tight", "loose"),
                    choices=("tight", "loose"),
                    help="Which schedule shapes to render (default: both).")
    args = ap.parse_args()

    for shape in args.shapes:
        timeline_csv = _timeline_for_shape(TIMELINES_DIR, shape)
        out_path = os.path.join(
            args.plots_dir, f"compare_emotion_alpaca_{shape}.png"
        )
        render_one(
            shape=shape,
            output_dir=args.output_dir,
            plots_dir=args.plots_dir,
            timeline_csv=timeline_csv,
            out_path=out_path,
        )


if __name__ == "__main__":
    main()
