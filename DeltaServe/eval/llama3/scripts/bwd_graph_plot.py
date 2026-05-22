"""Compare an eager backward run against an --enable-bwd-cuda-graph run.

Reads two timeline_results CSVs and two bwd_log CSVs produced by
auto_benchmark.py (which writes under eval/llama3/output/) and replays the
comparable plots from auto_plot.py side-by-side.

Default inputs (override via CLI):
  output/timeline_results.csv      (baseline / eager backward)
  output/timeline_results_bwd.csv  (--bwd_graph run)
  output/bwd_log.csv               (baseline)
  output/bwd_log_bwd.csv           (--bwd_graph run)

Output:
  plots/bwd_graph_comparison.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------- Paths ----------------
_HERE = os.path.dirname(os.path.abspath(__file__))
# Script lives under llama3/scripts/; output/ and plots/ are one level up.
OUTPUT_DIR = os.path.abspath(os.path.join(_HERE, "..", "output"))
PLOTS_DIR = os.path.abspath(os.path.join(_HERE, "..", "plots"))

RESULTS_BASELINE = os.path.join(OUTPUT_DIR, "timeline_results.csv")
RESULTS_BWD_GRAPH = os.path.join(OUTPUT_DIR, "timeline_results_bwd.csv")
BWD_LOG_BASELINE = os.path.join(OUTPUT_DIR, "bwd_log.csv")
BWD_LOG_BWD_GRAPH = os.path.join(OUTPUT_DIR, "bwd_log_bwd.csv")
OUT_PATH = os.path.join(PLOTS_DIR, "bwd_graph_comparison.png")

BASELINE_LABEL = "eager"
BWD_GRAPH_LABEL = "bwd graph"
BASELINE_COLOR = "tab:blue"
BWD_GRAPH_COLOR = "tab:orange"


# ---------------- Utilities ----------------
def ensure_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")


def load_results(csv_path: str) -> pd.DataFrame:
    """
    Expected format (from orchestrate_run_timeline.py):
      idx,t_rel_s,latency_s,status,ttft_s,avg_tbt_s,worst_tbt_s
    """
    df = pd.read_csv(csv_path)
    required = ["idx", "t_rel_s", "latency_s", "status", "ttft_s"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")
    df["ok"] = df["status"].astype(str).str.strip().eq("ok")
    return df


def parse_bwd_log_csv(csv_path: str):
    """
    Parse backward log CSV:
      timestamp,epoch,batch_idx,batch_tokens,batch_loss,total_processed_tokens

    Returns:
      rel_time_s: np.ndarray
      cum_tokens: np.ndarray
      avg_tok_s: float
    """
    df = pd.read_csv(csv_path)
    required = ["timestamp", "batch_tokens"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"{csv_path} has no valid timestamp rows.")

    # Collapse rows that share a timestamp (ts resolution is only 1s in the
    # logger) down to the last row, which holds the latest cumulative value
    # for that moment. Without this the line can double back mid-second.
    df = df.drop_duplicates(subset="timestamp", keep="last").reset_index(drop=True)

    t0 = df["timestamp"].iloc[0]
    df["rel_time_s"] = (df["timestamp"] - t0).dt.total_seconds()

    if "total_processed_tokens" in df.columns:
        cum_tokens = df["total_processed_tokens"].astype(float).to_numpy()
    else:
        cum_tokens = df["batch_tokens"].astype(float).cumsum().to_numpy()

    rel_time_s = df["rel_time_s"].to_numpy()
    elapsed = float(rel_time_s[-1]) if len(rel_time_s) > 0 else 0.0
    total = float(cum_tokens[-1]) if len(cum_tokens) > 0 else 0.0
    avg_tok_s = total / elapsed if elapsed > 0 else float("nan")
    return rel_time_s, cum_tokens, avg_tok_s


# ---------------- Plotting ----------------
def plot_ttft_percentile(ax, base_results: pd.DataFrame, graph_results: pd.DataFrame,
                         labels=(BASELINE_LABEL, BWD_GRAPH_LABEL),
                         colors=(BASELINE_COLOR, BWD_GRAPH_COLOR)):
    percentiles = np.array([0, 20, 40, 60, 80, 100])
    for df, label, color in (
        (base_results, labels[0], colors[0]),
        (graph_results, labels[1], colors[1]),
    ):
        ttft = df.loc[df["ok"], "ttft_s"].dropna().astype(float).to_numpy()
        if len(ttft) == 0:
            continue
        values = np.percentile(ttft, percentiles)
        ax.plot(percentiles, values, marker="o", color=color, label=label)
    ax.set_title("TTFT Percentile Curve")
    ax.set_xlabel("Percentile")
    ax.set_ylabel("TTFT (s)")
    ax.legend()


def plot_latency_vs_time(ax, base_results: pd.DataFrame, graph_results: pd.DataFrame,
                         labels=(BASELINE_LABEL, BWD_GRAPH_LABEL),
                         colors=(BASELINE_COLOR, BWD_GRAPH_COLOR)):
    for df, label, color in (
        (base_results, labels[0], colors[0]),
        (graph_results, labels[1], colors[1]),
    ):
        ok = df[df["ok"]]
        latencies = ok["latency_s"].astype(float).to_numpy()
        avg_latency = float(np.mean(latencies)) if len(latencies) > 0 else float("nan")
        series_label = (
            f"{label} (avg {avg_latency:.3f}s)" if np.isfinite(avg_latency) else label
        )
        ax.scatter(
            ok["t_rel_s"],
            ok["latency_s"],
            s=10,
            color=color,
            alpha=0.7,
            label=series_label,
        )
        if np.isfinite(avg_latency):
            ax.axhline(avg_latency, color=color, linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title("Request E2E Latency vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Latency (s)")
    ax.legend()


def plot_bwd_cumulative(ax, base_log, graph_log,
                        labels=(BASELINE_LABEL, BWD_GRAPH_LABEL),
                        colors=(BASELINE_COLOR, BWD_GRAPH_COLOR)):
    base_t, base_cum, base_avg = base_log
    graph_t, graph_cum, graph_avg = graph_log

    base_label = f"{labels[0]} ({base_avg:.1f} tok/s)" if np.isfinite(base_avg) else labels[0]
    graph_label = f"{labels[1]} ({graph_avg:.1f} tok/s)" if np.isfinite(graph_avg) else labels[1]

    ax.plot(base_t, base_cum, color=colors[0], label=base_label)
    ax.plot(graph_t, graph_cum, color=colors[1], label=graph_label)
    ax.set_title("Finetuning Cumulative Tokens")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative Tokens")
    ax.legend()


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(
        description="Compare an eager backward run vs a bwd-cuda-graph run. "
                    "Defaults assume the tag scheme used by auto_benchmark.py "
                    "(output/bwd_log.csv vs output/bwd_log_bwd.csv).",
    )
    ap.add_argument("--baseline-results", default=RESULTS_BASELINE)
    ap.add_argument("--bwd-graph-results", default=RESULTS_BWD_GRAPH)
    ap.add_argument("--baseline-bwd-log", default=BWD_LOG_BASELINE)
    ap.add_argument("--bwd-graph-bwd-log", default=BWD_LOG_BWD_GRAPH)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    for p in (args.baseline_results, args.bwd_graph_results,
              args.baseline_bwd_log, args.bwd_graph_bwd_log):
        ensure_exists(p)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    base_results = load_results(args.baseline_results)
    graph_results = load_results(args.bwd_graph_results)
    base_log = parse_bwd_log_csv(args.baseline_bwd_log)
    graph_log = parse_bwd_log_csv(args.bwd_graph_bwd_log)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_ttft_percentile(axes[0], base_results, graph_results)
    plot_latency_vs_time(axes[1], base_results, graph_results)
    plot_bwd_cumulative(axes[2], base_log, graph_log)

    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    print(f"[bwd_graph_plot] Saved comparison figure to {args.out}")


if __name__ == "__main__":
    main()
