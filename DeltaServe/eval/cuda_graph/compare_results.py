import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _find_first(df, candidates, required=False):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise ValueError(
            f"Missing required column. Tried: {candidates}. Found: {list(df.columns)}"
        )
    return None


def load_live_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    ts_col = _find_first(df, ["timestamp_s"], required=True)
    prompt_col = _find_first(df, ["prompt_length"], required=True)
    new_col = _find_first(df, ["max_new_tokens"], required=True)

    df = df.copy()
    df["timestamp_s"] = pd.to_numeric(df[ts_col], errors="coerce")
    df["prompt_length"] = pd.to_numeric(df[prompt_col], errors="coerce")
    df["max_new_tokens"] = pd.to_numeric(df[new_col], errors="coerce")
    df = df.dropna(subset=["timestamp_s", "prompt_length", "max_new_tokens"]).sort_values("timestamp_s")
    df["total_tokens"] = df["prompt_length"] + df["max_new_tokens"]
    return df


def load_result_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    t_rel_col = _find_first(df, ["t_rel_s"], required=True)
    latency_col = _find_first(df, ["latency_s"], required=True)
    ttft_col = _find_first(df, ["ttft_s"], required=True)
    avg_tbt_col = _find_first(df, ["avg_tbt_s"], required=True)
    worst_tbt_col = _find_first(df, ["worst_tbt_s"], required=True)
    status_col = _find_first(df, ["status"], required=False)

    df = df.copy()
    df["t_rel_s"] = pd.to_numeric(df[t_rel_col], errors="coerce")
    df["latency_s"] = pd.to_numeric(df[latency_col], errors="coerce")
    df["ttft_s"] = pd.to_numeric(df[ttft_col], errors="coerce")
    df["avg_tbt_s"] = pd.to_numeric(df[avg_tbt_col], errors="coerce")
    df["worst_tbt_s"] = pd.to_numeric(df[worst_tbt_col], errors="coerce")

    if status_col is not None:
        df["status"] = df[status_col].astype(str)
        df = df[df["status"].str.lower() == "ok"]

    df = df.dropna(subset=["t_rel_s", "latency_s", "ttft_s", "avg_tbt_s", "worst_tbt_s"])
    df = df.sort_values("t_rel_s").reset_index(drop=True)
    return df


def compute_tokens_per_sec_from_live(live_df: pd.DataFrame) -> pd.DataFrame:
    df = live_df.copy()
    df["second_bin"] = np.floor(df["timestamp_s"]).astype(int)

    start_sec = int(df["second_bin"].min())
    end_sec = int(df["second_bin"].max())

    full_index = pd.Index(range(start_sec, end_sec + 1), name="second_bin")

    per_sec = (
        df.groupby("second_bin")["total_tokens"]
        .sum()
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    per_sec = per_sec.rename(
        columns={
            "second_bin": "time_s",
            "total_tokens": "tokens_per_sec",
        }
    )

    return per_sec


def percentile_curve(values: np.ndarray):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.array([]), np.array([])

    sorted_vals = np.sort(values)
    pct = np.linspace(0, 100, len(sorted_vals))
    return pct, sorted_vals


def add_box(ax, text, xy=(0.5, 0.98), fontsize=11):
    ax.text(
        xy[0],
        xy[1],
        text,
        transform=ax.transAxes,
        va="top",
        ha="center",
        fontsize=fontsize,
        family="monospace",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="gray",
            alpha=0.85,
            linewidth=1.5,
        ),
    )


def pair_live_and_result(live_df: pd.DataFrame, result_df: pd.DataFrame) -> pd.DataFrame:
    n = min(len(live_df), len(result_df))
    if n == 0:
        return pd.DataFrame(columns=["arrival_time_s", "latency_s", "ttft_s", "avg_tbt_s"])

    out = pd.DataFrame(
        {
            "arrival_time_s": live_df["timestamp_s"].iloc[:n].to_numpy(),
            "latency_s": result_df["latency_s"].iloc[:n].to_numpy(),
            "ttft_s": result_df["ttft_s"].iloc[:n].to_numpy(),
            "avg_tbt_s": result_df["avg_tbt_s"].iloc[:n].to_numpy(),
            "worst_tbt_s": result_df["worst_tbt_s"].iloc[:n].to_numpy(),
        }
    )
    return out.sort_values("arrival_time_s").reset_index(drop=True)


def make_plot(
    result_a_csv: str,
    result_b_csv: str,
    live_csv: str,
    output: str = "timeline_comparison.png",
    label_a: str = "Run A",
    label_b: str = "Run B",
):
    live_df = load_live_csv(live_csv)
    result_a_df = load_result_csv(result_a_csv)
    result_b_df = load_result_csv(result_b_csv)

    top_df = compute_tokens_per_sec_from_live(live_df)
    paired_a = pair_live_and_result(live_df, result_a_df)
    paired_b = pair_live_and_result(live_df, result_b_df)

    mean_latency_a = paired_a["latency_s"].mean() if len(paired_a) else np.nan
    mean_latency_b = paired_b["latency_s"].mean() if len(paired_b) else np.nan
    delta_latency = mean_latency_b - mean_latency_a
    delta_latency_pct = (
        delta_latency / mean_latency_a * 100.0
        if np.isfinite(mean_latency_a) and mean_latency_a != 0
        else np.nan
    )

    pct_a, curve_a = percentile_curve(result_a_df["latency_s"].to_numpy())
    pct_b, curve_b = percentile_curve(result_b_df["latency_s"].to_numpy())

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 3.8), dpi=150)
    ax0, ax1, ax2 = axes

    # Plot 1: timeline load
    ax0.step(top_df["time_s"], top_df["tokens_per_sec"], where="post", linewidth=2.0)
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Total tokens/sec")
    ax0.grid(True, alpha=0.25)

    # Plot 2: percentile comparison
    ax1.plot(pct_a, curve_a, linewidth=2.2, label=label_a)
    ax1.plot(pct_b, curve_b, linewidth=2.2, label=label_b)
    ax1.set_xlabel("Percentile")
    ax1.set_ylabel("Latency (s)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(frameon=True)

    # Plot 3: e2e latency vs arrival time
    ax2.plot(
        paired_a["arrival_time_s"],
        paired_a["latency_s"],
        linewidth=2.2,
        marker="o",
        markersize=3,
        label=label_a,
    )
    ax2.plot(
        paired_b["arrival_time_s"],
        paired_b["latency_s"],
        linewidth=2.2,
        marker="o",
        markersize=3,
        label=label_b,
    )
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Latency (s)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(frameon=True)

    if np.isfinite(delta_latency):
        pct_text = f"{delta_latency_pct:+.1f}%" if np.isfinite(delta_latency_pct) else "n/a"
        add_box(
            ax1,
            f"Δ Avg Latency: {delta_latency:+.3f}s ({pct_text})",
            fontsize=11,
        )

    plt.tight_layout(w_pad=2.2)
    fig.savefig(output, bbox_inches="tight")
    print(f"Saved plot to: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="timeline_results.csv")
    parser.add_argument("--graph", default="timeline_results_graph.csv")
    parser.add_argument("--live", default="timeline_live.txt")
    parser.add_argument("--label-a", default="DServe", help="Legend label for timeline_result.csv")
    parser.add_argument("--label-b", default="DServe+Graph", help="Legend label for timeline_result_graph.csv")
    parser.add_argument("--output", default="timeline_comparison.png", help="Output image path")
    args = parser.parse_args()

    make_plot(
        result_a_csv=args.result,
        result_b_csv=args.graph,
        live_csv=args.live,
        output=args.output,
        label_a=args.label_a,
        label_b=args.label_b,
    )


if __name__ == "__main__":
    main()