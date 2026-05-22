import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

# ------------------------------------------------------
# Utility Functions
# ------------------------------------------------------
def _safe_series(df, name):
    return df[name].dropna() if name in df.columns else pd.Series([], dtype=float)

def _percentile_curve(series):
    vals = series.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.array([])
    pct = np.linspace(0, 100, 101)
    return np.percentile(vals, pct)

def _safe_label(label: str) -> str:
    return label.replace("_", " ")

# ------------------------------------------------------
# Figure 1–2: Latency Percentile Comparisons (2 datasets)
# ------------------------------------------------------
def plot_latency_panels(ax_ttft, files, labels, colors, ttft_slo=None):
    assert len(files) == 2 and len(labels) == 2 and len(colors) == 2

    dfs = [pd.read_csv(f) for f in files]
    dfs = [df[df.get("status", "ok") == "ok"] for df in dfs]

    pct = np.linspace(0, 100, 101)
    all_vals = []

    # --- Plot TTFT / Avg TBT curves ---
    for df, label, color in zip(dfs, labels, colors):
        y_ttft = _percentile_curve(_safe_series(df, "ttft_s"))
        y_avg  = _percentile_curve(_safe_series(df, "avg_tbt_s"))

        if y_ttft.size:
            ax_ttft.plot(pct, y_ttft, color=color, linestyle="-",
                         label=f"TTFT ({_safe_label(label)})")
            all_vals.append(y_ttft)
        if y_avg.size:
            ax_ttft.plot(pct, y_avg, color=color, linestyle="--",
                         label=f"Avg TBT ({_safe_label(label)})")
            all_vals.append(y_avg)

    # --- Draw TTFT SLO ---
    if ttft_slo is not None:
        ax_ttft.axhline(ttft_slo, color="black", linestyle=":",
                        linewidth=1.2, alpha=0.9,
                        label=f"TTFT SLO = {ttft_slo:.3f}s")

    # --- Dynamic Y limit ---
    all_vals = np.concatenate(all_vals) if all_vals else np.array([])
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size:
        ax_ttft.set_ylim(top=all_vals.max())

    ax_ttft.set_title("TTFT / Avg TBT", fontsize=11, fontweight="bold")
    ax_ttft.set_xlabel("Percentile")
    ax_ttft.set_ylabel("Seconds")
    ax_ttft.grid(True, alpha=0.3)
    ax_ttft.legend(fontsize=8)


# ------------------------------------------------------
# GPU Utilization Panel (multi-GPU; Co vs SLoRA+FT)
# ------------------------------------------------------
def plot_gpu_usage_panel_multi(
    ax,
    co_files,
    slora_files,
    threshold=50,
    smooth_window=1,
    colors=None,
):
    if colors is None:
        colors = ["tab:orange", "tab:green"]

    # -------------- Helpers --------------
    def load_group(files):
        dfs = []
        for path in files:
            df = pd.read_csv(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["time_rel_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
            df["gpu_util_smooth"] = df["gpu_util"].rolling(
                window=smooth_window, min_periods=1
            ).mean()
            dfs.append(df)
        return dfs

    def find_start(df):
        idx = np.argmax(df["gpu_util"] > threshold)
        if df["gpu_util"].iloc[idx] > threshold:
            return df["time_rel_s"].iloc[idx]
        return df["time_rel_s"].iloc[0]

    # Load raw groups
    co_dfs = load_group(co_files)
    sl_dfs = load_group(slora_files)

    # Compute activity starts
    co_starts = [find_start(df) for df in co_dfs]
    sl_starts = [find_start(df) for df in sl_dfs]

    # Align each df so its activity start is 0
    def align_dfs(dfs, starts):
        aligned = []
        for df, t0 in zip(dfs, starts):
            df2 = df[df["time_rel_s"] >= t0].copy()
            df2["time_aligned"] = df2["time_rel_s"] - t0
            aligned.append(df2)
        return aligned

    co_aligned = align_dfs(co_dfs, co_starts)
    sl_aligned = align_dfs(sl_dfs, sl_starts)

    # Determine common end
    def find_last_active(df, active_threshold):
        util = df["gpu_util"].to_numpy()
        idx = np.where(util > active_threshold)[0]
        if len(idx) == 0:
            return None
        return df["time_aligned"].iloc[idx[-1]]


    co_last = [find_last_active(df, threshold) for df in co_aligned]
    sl_last = [find_last_active(df, threshold) for df in sl_aligned]

    # Remove None values
    co_last = [x for x in co_last if x is not None]
    sl_last = [x for x in sl_last if x is not None]

    if len(co_last) == 0 or len(sl_last) == 0:
        global_end = 0
    else:
        # Strict min ensures that all GPUs have activity in plotted region.
        global_end = min(min(co_last), min(sl_last))

    # Determine dt
    deltas = []
    for df in co_aligned + sl_aligned:
        t = df["time_aligned"].to_numpy()
        if t.size >= 2:
            diff = np.diff(t)
            diff = diff[diff > 0]
            if diff.size:
                deltas.append(np.median(diff))
    dt = min(deltas) if deltas else 1.0

    time_grid = np.arange(0.0, global_end + 1e-9, dt)

    # Interpolate and average curves
    def avg_curve(aligned):
        curves = []
        for df in aligned:
            df2 = df[df["time_aligned"] <= global_end]
            if len(df2) > 5:
                df2 = df2.iloc[:-2]
            if len(df2) < 2:
                continue
            x = df2["time_aligned"].to_numpy()
            y = df2["gpu_util_smooth"].to_numpy()
            y_interp = np.interp(time_grid, x, y)
            curves.append(y_interp)
        if not curves:
            return np.zeros_like(time_grid)
        return np.vstack(curves).mean(axis=0)

    co_curve = avg_curve(co_aligned)
    sl_curve = avg_curve(sl_aligned)

    # Average GPU util for legend
    # Average GPU util AFTER smoothing, trimming, alignment, and interpolation
    avg_co = float(np.mean(co_curve)) if len(co_curve) > 0 else float("nan")
    avg_sl = float(np.mean(sl_curve)) if len(sl_curve) > 0 else float("nan")

    # Plot curves
    ax.plot(time_grid, co_curve, color=colors[0], alpha=0.7)
    ax.plot(time_grid, sl_curve, color=colors[1], alpha=0.7)

    ax.axhline(avg_co, color=colors[0], linestyle="--", linewidth=2.0,
               label=f"Co-serving (4 GPUs avg) avg={avg_co:.1f}%")
    ax.axhline(avg_sl, color=colors[1], linestyle="--", linewidth=2.0,
               label=f"SLoRA+FT (4 GPUs avg) avg={avg_sl:.1f}%")

    ax.set_title("GPU Compute Utilization", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (s, aligned)")
    ax.set_ylabel("GPU Utilization (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

# ------------------------------------------------------
# Panel 4: Request Latency Timeline
# ------------------------------------------------------
def plot_request_latency_timeline(ax, latency_files, labels, colors):
    dfs = [pd.read_csv(f) for f in latency_files]

    for df, label, color in zip(dfs, labels, colors):
        df = df[df["status"] == "ok"]
        ax.scatter(df["t_rel_s"], df["latency_s"],
                   s=5, alpha=0.4, color=color, label=label)

    ax.set_title("Request Latency Over Time", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Latency (s)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

def plot_ft_tokens_timeline(ax, bwd_files_co, bwd_file_slora, colors):
    """
    Co-serving FT:
        - Load 4 logs
        - Convert timestamp → datetime
        - Shift each so first timestamp = 0
        - Concatenate all logs into one dataframe
        - Sort by time_rel_s
        - Compute cumulative total_processed_tokens

    SLoRA+FT:
        - Same, but only one log

    Then extend the shorter curve by holding its final value flat.
    """

    # -----------------------------
    # Load and align co-serving logs
    # -----------------------------
    co_all = []
    for f in bwd_files_co:
        df = pd.read_csv(f)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        t0 = df["timestamp"].iloc[0]
        df["time_rel_s"] = (df["timestamp"] - t0).dt.total_seconds()
        co_all.append(df[["time_rel_s", "batch_tokens"]])  # we will accumulate ourselves

    # Merge 4 logs → one event list
    co_merged = pd.concat(co_all, ignore_index=True)
    co_merged = co_merged.sort_values("time_rel_s").reset_index(drop=True)

    # Compute cumulative tokens
    co_merged["total_tokens"] = co_merged["batch_tokens"].cumsum()

    co_time = co_merged["time_rel_s"]
    co_tokens = co_merged["total_tokens"]

    # -----------------------------
    # Load and align SLoRA+FT log
    # -----------------------------
    sl = pd.read_csv(bwd_file_slora)
    sl["timestamp"] = pd.to_datetime(sl["timestamp"])
    t0_sl = sl["timestamp"].iloc[0]
    sl["time_rel_s"] = (sl["timestamp"] - t0_sl).dt.total_seconds()

    sl = sl[["time_rel_s", "batch_tokens"]].sort_values("time_rel_s")
    sl["total_tokens"] = sl["batch_tokens"].cumsum()

    sl_time = sl["time_rel_s"]
    sl_tokens = sl["total_tokens"]

    # -----------------------------
    # Equalize lengths via flat extension
    # -----------------------------
    end_co = co_time.iloc[-1]
    end_sl = sl_time.iloc[-1]
    max_end = max(end_co, end_sl)

    # Extend Co-serving if shorter
    if end_co < max_end:
        co_time = pd.concat([co_time, pd.Series([max_end])], ignore_index=True)
        co_tokens = pd.concat([co_tokens, pd.Series([co_tokens.iloc[-1]])], ignore_index=True)

    # Extend SLoRA if shorter
    if end_sl < max_end:
        sl_time = pd.concat([sl_time, pd.Series([max_end])], ignore_index=True)
        sl_tokens = pd.concat([sl_tokens, pd.Series([sl_tokens.iloc[-1]])], ignore_index=True)

    # -----------------------------
    # Plot
    # -----------------------------
    ax.plot(
        co_time, co_tokens,
        color=colors[0], label="Co-serving FT Progress"
    )
    ax.plot(
        sl_time, sl_tokens,
        color=colors[1], label="SLoRA+FT FT Progress"
    )

    ax.set_title("Fine-tuning Progress Over Time", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fine-tuned Tokens")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
# ------------------------------------------------------
# Panel 1: Timeline Panel (per-second)
# ------------------------------------------------------
def plot_timeline_panel(ax, csv_path="timelines/timeline_live.csv"):
    df = pd.read_csv(csv_path)
    df["total_tokens"] = df["prompt_length"] + df["max_new_tokens"]
    df["second_bin"] = df["timestamp_s"].astype(int)

    tokens_raw = df.groupby("second_bin")["total_tokens"].sum()
    req_raw = df.groupby("second_bin").size()

    t_min = int(df["second_bin"].min())
    t_max = int(df["second_bin"].max())
    all_seconds = np.arange(t_min, t_max + 1)

    tokens_per_sec = tokens_raw.reindex(all_seconds, fill_value=0)
    req_per_sec = req_raw.reindex(all_seconds, fill_value=0)

    ax.plot(tokens_per_sec.index, tokens_per_sec.values,
            color="tab:blue", linewidth=1.8)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Total tokens/sec", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax.twinx()
    ax2.bar(req_per_sec.index, req_per_sec.values,
            color="tab:orange", alpha=0.3, width=0.9)
    ax2.set_ylabel("#Requests/sec", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax.set_title("Request Timeline", fontsize=11, fontweight="bold")

# ------------------------------------------------------
# Combine All Panels: 2 rows × 3 columns = 6 plots
# ------------------------------------------------------
def plot_all_combined(ttft_slo=None, result_dir="results"):
    latency_files = [f"{result_dir}/latency_co-serving.csv", f"{result_dir}/latency_slora.csv"]

    co_gpu_usage_files = [
        f"{result_dir}/gpu_usage_co-serving_0.csv",
        f"{result_dir}/gpu_usage_co-serving_1.csv",
        f"{result_dir}/gpu_usage_co-serving_2.csv",
        f"{result_dir}/gpu_usage_co-serving_3.csv",
    ]

    slora_gpu_usage_files = [
        f"{result_dir}/gpu_usage_ft.csv",
        f"{result_dir}/gpu_usage_slora_0.csv",
        f"{result_dir}/gpu_usage_slora_1.csv",
        f"{result_dir}/gpu_usage_slora_2.csv",
    ]

    bwd_logs_co = [
        f"{result_dir}/bwd_log_1.csv",
        f"{result_dir}/bwd_log_2.csv",
        f"{result_dir}/bwd_log_3.csv",
        f"{result_dir}/bwd_log_4.csv",
    ]

    bwd_log_slora = f"{result_dir}/bwd_log_0.csv"

    timeline_file = f"{result_dir}/timeline.csv"
    labels = ["Co-serving", "SLoRA"]
    colors = ["tab:orange", "tab:green"]

    fig, axes = plt.subplots(5, 1, figsize=(10, 30))
    ax_timeline, ax_ttft, ax_req_lat, ax_ft, ax_gpu = axes

    # Panel 1
    plot_timeline_panel(ax_timeline, csv_path=timeline_file)

    # Panel 2
    plot_latency_panels(ax_ttft, latency_files, labels, colors, ttft_slo=ttft_slo)

    # Panel 3
    plot_request_latency_timeline(ax_req_lat, latency_files, labels, colors)

    # Panel 4
    plot_ft_tokens_timeline(ax_ft, bwd_logs_co, bwd_log_slora, colors)

    # Panel 5
    plot_gpu_usage_panel_multi(
        ax_gpu,
        co_files=co_gpu_usage_files,
        slora_files=slora_gpu_usage_files,
        colors=colors,
    )

    fig.tight_layout()
    out_path = os.path.abspath(f"{result_dir}/combined_summary_full.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"✅ Saved combined 6-panel figure to {out_path}")

if __name__ == "__main__":
    ttft_slo = 0.3
    plot_all_combined(ttft_slo=ttft_slo, result_dir="/projects/I20240005/jchen/slora-plus/S-LoRA/test/eval/results_nutanix")