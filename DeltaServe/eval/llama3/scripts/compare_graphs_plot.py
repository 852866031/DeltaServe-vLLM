"""compare_graphs_plot.py — ablation plots for the CUDA graph features.

Produces three side-by-side comparison PNGs, each pitting the baseline
(no CUDA graph) against a progressively more-enabled variant:

  plots/compare_prefill.png              baseline vs prefill graph
  plots/compare_prefill_decode.png       baseline vs prefill+decode graph
  plots/compare_prefill_decode_bwd.png   baseline vs prefill+decode+bwd graph

Each figure is a 4-panel layout, with the scheduled-request timeline
(req/s + output tokens/s, bucketed per integer second) on the far left:
  [request timeline] [TTFT percentile] [E2E latency vs time] [cumulative finetune tokens].

All four CSVs must live under eval/llama3/output/ (produced by
auto_benchmark.py with the matching flags). Missing variants are skipped
with a warning so partial ablations still produce what they can.

Run the upstream benchmarks first, e.g.:
  python auto_benchmark.py --co                                           # baseline
  python auto_benchmark.py --co --prefill_graph                           # prefill
  python auto_benchmark.py --co --decode_graph --prefill_graph            # prefill+decode
  python auto_benchmark.py --co --decode_graph --prefill_graph --bwd_graph
"""
import argparse
import os
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bwd_graph_plot import (
    ensure_exists,
    load_results,
    parse_bwd_log_csv,
    plot_ttft_percentile,
    plot_latency_vs_time,
    plot_bwd_cumulative,
)

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
# This script now lives under llama3/scripts/, so output/, plots/, and
# the timelines/ dir are one level up from _HERE.
OUTPUT_DIR = os.path.abspath(os.path.join(_HERE, "..", "output"))
PLOTS_DIR = os.path.abspath(os.path.join(_HERE, "..", "plots"))
DEFAULT_TIMELINE_CSV = os.path.abspath(os.path.join(
    _HERE, "..", "timelines", _DEFAULT_GPU_SUBDIR, "timeline_live.csv"))

BASELINE_LABEL = "no graph"
BASELINE_COLOR = "tab:blue"
VARIANT_COLOR = "tab:orange"

# (png_stem, variant_label, tag_suffix). Tag order is decode < prefill < bwd
# (matches auto_benchmark's filename composer), so the progressive variants
# stamp as _prefill, _decode_prefill, _decode_prefill_bwd.
COMPARISONS = [
    ("compare_prefill",            "prefill",              "_prefill"),
    ("compare_prefill_decode",     "prefill+decode",       "_decode_prefill"),
    ("compare_prefill_decode_bwd", "prefill+decode+bwd",   "_decode_prefill_bwd"),
]


def load_timeline(csv_path: str) -> pd.DataFrame:
    """Load the scheduled-request timeline (input to auto_benchmark).

    Expected columns: timestamp_s, prompt_length, max_new_tokens.
    """
    df = pd.read_csv(csv_path)
    required = ["timestamp_s", "max_new_tokens"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")
    t0 = float(df["timestamp_s"].astype(float).min())
    df = df.copy()
    df["t_rel_s"] = df["timestamp_s"].astype(float) - t0
    return df


def plot_request_timeline(ax, timeline_df: pd.DataFrame):
    """Leftmost subplot: req/s (bars, left y-axis) + output tokens/s
    (line, right y-axis), bucketized per integer second.

    Uses np.bincount with minlength so seconds with 0 requests still show
    as a zero-height bar — the gap in load is visible in the chart."""
    t = timeline_df["t_rel_s"].astype(float).to_numpy()
    tok = timeline_df["max_new_tokens"].astype(float).to_numpy()

    if len(t) == 0:
        ax.set_title("Request Timeline (empty)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Requests per second")
        return

    # Integer-second buckets [k, k+1). Requests with t_rel_s in that range
    # contribute to bucket k. np.floor covers fractional timestamps.
    bucket_idx = np.floor(t).astype(int)
    n_buckets = int(bucket_idx.max()) + 1
    req_per_s = np.bincount(bucket_idx, minlength=n_buckets).astype(float)
    tok_per_s = np.bincount(bucket_idx, weights=tok, minlength=n_buckets).astype(float)
    centers = np.arange(n_buckets, dtype=float)   # left edge of each bucket

    color_req = "tab:gray"
    color_tok = "tab:green"

    ax.bar(centers, req_per_s, width=1.0, color=color_req, alpha=0.55,
           align="edge", label="req/s", edgecolor="none")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Requests / s", color=color_req)
    ax.tick_params(axis="y", labelcolor=color_req)
    ax.set_xlim(0, n_buckets)
    # Anchor the left axis at 0 so empty seconds visibly sit on the
    # baseline rather than getting padded below it by matplotlib's
    # default margin.
    ax.set_ylim(bottom=0)
    ax.set_title("Scheduled Request Timeline")

    ax2 = ax.twinx()
    ax2.plot(centers + 0.5, tok_per_s, color=color_tok, marker="o",
             markersize=3, linewidth=1.5, label="tokens/s")
    ax2.set_ylabel("Output tokens / s", color=color_tok)
    ax2.tick_params(axis="y", labelcolor=color_tok)
    # Right axis also starts at 0 so the two scales are visually
    # comparable (both span [0, peak]).
    ax2.set_ylim(bottom=0)

    # Combined legend so both axes' series show in one box.
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)


def _make_figure(base_results, base_log, var_results, var_log,
                 timeline_df, variant_label: str, out_path: str, title: str):
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    labels = (BASELINE_LABEL, variant_label)
    colors = (BASELINE_COLOR, VARIANT_COLOR)
    plot_request_timeline(axes[0], timeline_df)
    plot_ttft_percentile(axes[1], base_results, var_results, labels=labels, colors=colors)
    plot_latency_vs_time(axes[2], base_results, var_results, labels=labels, colors=colors)
    plot_bwd_cumulative(axes[3], base_log, var_log, labels=labels, colors=colors)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _prefer_kv(output_dir: str, stem: str, suffix: str) -> Optional[str]:
    """Return the packed-kv variant path if it exists, else the plain one,
    else None. `auto_benchmark.py` appends `_kv` to the suffix when run with
    `--packed_kv`, so the packed-kv file is `<stem><suffix>_kv.csv`.
    """
    kv_path = os.path.join(output_dir, f"{stem}{suffix}_kv.csv")
    plain_path = os.path.join(output_dir, f"{stem}{suffix}.csv")
    if os.path.exists(kv_path):
        return kv_path
    if os.path.exists(plain_path):
        return plain_path
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default=OUTPUT_DIR,
                    help="Where auto_benchmark writes its CSVs.")
    ap.add_argument("--plots-dir", default=PLOTS_DIR,
                    help="Where to write the comparison PNGs.")
    ap.add_argument("--timeline-csv", default=DEFAULT_TIMELINE_CSV,
                    help="Scheduled-request timeline (input to auto_benchmark). "
                         "Drives the leftmost req/s + tokens/s subplot.")
    args = ap.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)

    try:
        ensure_exists(args.timeline_csv)
    except FileNotFoundError as e:
        sys.exit(f"[compare_graphs_plot] timeline CSV missing: {e}")
    timeline_df = load_timeline(args.timeline_csv)

    base_results_csv = _prefer_kv(args.output_dir, "timeline_results", "")
    base_log_csv = _prefer_kv(args.output_dir, "bwd_log", "")
    if base_results_csv is None or base_log_csv is None:
        sys.exit("[compare_graphs_plot] baseline CSV missing "
                 "(looked for timeline_results[_kv].csv and bwd_log[_kv].csv)")
    print(f"[compare_graphs_plot] baseline: {os.path.basename(base_results_csv)}, "
          f"{os.path.basename(base_log_csv)}")

    base_results = load_results(base_results_csv)
    base_log = parse_bwd_log_csv(base_log_csv)

    wrote = 0
    for png_stem, variant_label, suffix in COMPARISONS:
        var_results_csv = _prefer_kv(args.output_dir, "timeline_results", suffix)
        var_log_csv = _prefer_kv(args.output_dir, "bwd_log", suffix)
        if var_results_csv is None or var_log_csv is None:
            print(f"[compare_graphs_plot] skip {png_stem}: no CSVs for "
                  f"suffix '{suffix}' (with or without _kv)", file=sys.stderr)
            continue

        print(f"[compare_graphs_plot] {png_stem}: {os.path.basename(var_results_csv)}, "
              f"{os.path.basename(var_log_csv)}")
        var_results = load_results(var_results_csv)
        var_log = parse_bwd_log_csv(var_log_csv)
        out_path = os.path.join(args.plots_dir, f"{png_stem}.png")
        title = f"{BASELINE_LABEL}  vs  {variant_label}"
        _make_figure(base_results, base_log, var_results, var_log,
                     timeline_df, variant_label, out_path, title)
        print(f"[compare_graphs_plot] Wrote {out_path}")
        wrote += 1

    if wrote == 0:
        sys.exit("[compare_graphs_plot] no variant CSVs found — nothing to compare")


if __name__ == "__main__":
    main()
