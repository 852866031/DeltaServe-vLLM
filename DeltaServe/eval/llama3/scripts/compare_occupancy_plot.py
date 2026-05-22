"""compare_occupancy_plot.py — page-occupancy comparison plot.

Reads two occupancy CSVs (one per allocator) produced by auto_benchmark
when --track_occupancy is set, and plots used_pages / total_pages over
time on a shared axis. Each CSV row is:

  timestamp,t_rel_s,allocator,used_pages,total_pages,occupancy_pct

Default suffix '_decode_prefill_bwd' matches the
--decode_graph --prefill_graph --bwd_graph configuration. Override via
--suffix or pass full paths via --unified-csv / --packed-csv.

Run the upstream benchmarks first, e.g.:
  python auto_benchmark.py --co --decode_graph --prefill_graph --bwd_graph \\
      --track_occupancy
  python auto_benchmark.py --co --decode_graph --prefill_graph --bwd_graph \\
      --packed_kv --track_occupancy
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_HERE = os.path.dirname(os.path.abspath(__file__))
# Script lives under llama3/scripts/; output/ and plots/ are one level up.
OUTPUT_DIR = os.path.abspath(os.path.join(_HERE, "..", "output"))
PLOTS_DIR = os.path.abspath(os.path.join(_HERE, "..", "plots"))

UNIFIED_LABEL = "unified"
PACKED_LABEL = "packed_kv"
UNIFIED_COLOR = "tab:blue"
PACKED_COLOR = "tab:purple"


def load_occupancy(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = ["t_rel_s", "used_pages", "total_pages", "occupancy_pct"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")
    return df


def _annotate(df: pd.DataFrame, label: str) -> str:
    occ = df["occupancy_pct"].astype(float).to_numpy()
    if len(occ) == 0:
        return label
    return f"{label} (avg {occ.mean():.1f}%, max {occ.max():.1f}%)"


def plot_occupancy_pct(ax, u: pd.DataFrame, p: pd.DataFrame):
    ax.plot(u["t_rel_s"], u["occupancy_pct"],
            color=UNIFIED_COLOR, linewidth=1.5,
            label=_annotate(u, UNIFIED_LABEL))
    ax.plot(p["t_rel_s"], p["occupancy_pct"],
            color=PACKED_COLOR, linewidth=1.5,
            label=_annotate(p, PACKED_LABEL))
    ax.set_xlabel("Time since allocator init (s)")
    ax.set_ylabel("Occupancy (%)")
    ax.set_ylim(bottom=0)
    ax.set_title("Page occupancy: used_pages / total_pages")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")


def plot_used_pages(ax, u: pd.DataFrame, p: pd.DataFrame):
    total = int(u["total_pages"].iloc[0]) if len(u) else 0
    ax.plot(u["t_rel_s"], u["used_pages"],
            color=UNIFIED_COLOR, linewidth=1.5,
            label=f"{UNIFIED_LABEL} (peak {int(u['used_pages'].max())} pages)"
                  if len(u) else UNIFIED_LABEL)
    ax.plot(p["t_rel_s"], p["used_pages"],
            color=PACKED_COLOR, linewidth=1.5,
            label=f"{PACKED_LABEL} (peak {int(p['used_pages'].max())} pages)"
                  if len(p) else PACKED_LABEL)
    if total > 0:
        ax.axhline(total, color="gray", linestyle="--", linewidth=1,
                   alpha=0.6, label=f"pool capacity ({total} pages)")
    ax.set_xlabel("Time since allocator init (s)")
    ax.set_ylabel("Used pages")
    ax.set_ylim(bottom=0)
    ax.set_title("Used pages over time")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--suffix", default="_decode_prefill_bwd",
                    help="Tag suffix for auto_benchmark outputs. Looks up "
                         "occupancy<suffix>.csv (unified) and "
                         "occupancy<suffix>_kv.csv (packed_kv).")
    ap.add_argument("--unified-csv", default=None,
                    help="Override unified CSV path; bypasses --suffix.")
    ap.add_argument("--packed-csv", default=None,
                    help="Override packed_kv CSV path; bypasses --suffix.")
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    ap.add_argument("--plots-dir", default=PLOTS_DIR)
    ap.add_argument("--out", default=None,
                    help="Output PNG path. Default: "
                         "plots/compare_occupancy<suffix>.png")
    args = ap.parse_args()

    suffix = args.suffix
    unified_csv = args.unified_csv or os.path.join(
        args.output_dir, f"occupancy{suffix}.csv")
    packed_csv = args.packed_csv or os.path.join(
        args.output_dir, f"occupancy{suffix}_kv.csv")

    for p in (unified_csv, packed_csv):
        if not os.path.exists(p):
            sys.exit(f"[compare_occupancy_plot] missing input: {p}")

    os.makedirs(args.plots_dir, exist_ok=True)
    out_path = args.out or os.path.join(
        args.plots_dir, f"compare_occupancy{suffix}.png")

    u = load_occupancy(unified_csv)
    p = load_occupancy(packed_csv)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_occupancy_pct(axes[0], u, p)
    plot_used_pages(axes[1], u, p)
    config_tag = suffix.lstrip("_") or "baseline"
    fig.suptitle(f"Allocator page occupancy  ({config_tag})")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[compare_occupancy_plot] Wrote {out_path}")


if __name__ == "__main__":
    main()
