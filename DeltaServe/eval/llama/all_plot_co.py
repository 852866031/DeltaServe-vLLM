import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
def plot_latency_panels(ax_ttft, ax_latency, files, labels, colors, ttft_slo=None):
    assert len(files) == 2 and len(labels) == 2 and len(colors) == 2, \
        "This version only supports exactly 2 datasets."

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

    # --- Draw TTFT SLO horizontal line ---
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

    # --- Plot Total Latency ---
    for df, label, color in zip(dfs, labels, colors):
        y_lat = _percentile_curve(_safe_series(df, "latency_s"))
        if y_lat.size:
            ax_latency.plot(pct, y_lat, color=color, linewidth=1.8,
                            label=f"{_safe_label(label)}")

    ax_latency.set_title("Total Latency", fontsize=11, fontweight="bold")
    ax_latency.set_xlabel("Percentile")
    ax_latency.set_ylabel("Seconds")
    ax_latency.grid(True, alpha=0.3)
    ax_latency.legend(fontsize=8)

    # ===========================================================
    # Compute averages
    # ===========================================================
    avg_ttft = [np.nanmean(_safe_series(df, "ttft_s")) for df in dfs]
    avg_tbt  = [np.nanmean(_safe_series(df, "avg_tbt_s")) for df in dfs]
    avg_lat  = [np.nanmean(_safe_series(df, "latency_s")) for df in dfs]

    # baseline = second dataset
    base = 1  

    def delta_str(base_val, comp_val):
        if np.isnan(base_val) or np.isnan(comp_val): return "N/A"
        diff = comp_val - base_val
        pct = (diff / base_val * 100) if base_val > 0 else float("nan")
        sign = "+" if diff > 0 else ""
        return f"{sign}{diff:.3f}s ({sign}{pct:.1f}%)"

    def delta_str2(base_val, comp_val):
        if np.isnan(base_val) or np.isnan(comp_val): return "N/A"
        diff = comp_val - base_val
        sign = "+" if diff > 0 else ""
        return f"{sign}{diff:.3f}s"

    # ===========================================================
    # TTFT + Avg TBT comparison box
    # ===========================================================
    box_ttft_tbt = (
        f"Δ Avg TTFT / Avg TBT (vs {_safe_label(labels[base])})\n"
        f"{_safe_label(labels[0])}: TTFT {delta_str(avg_ttft[base], avg_ttft[0])},  "
        f"TBT {delta_str2(avg_tbt[base], avg_tbt[0])}"
    )

    ax_ttft.text(
        0.03, 0.5, box_ttft_tbt,
        transform=ax_ttft.transAxes,
        fontsize=8.5,
        ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8),
        family="monospace",
    )

    # ===========================================================
    # Total latency comparison box
    # ===========================================================
    box_lat = (
        f"Δ Avg Total Latency (vs {_safe_label(labels[base])})\n"
        f"{_safe_label(labels[0])}: {delta_str(avg_lat[base], avg_lat[0])}"
    )

    ax_latency.text(
        0.98, 0.05, box_lat,
        transform=ax_latency.transAxes,
        fontsize=8.5,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8),
        family="monospace",
    )

# ------------------------------------------------------
# GPU Utilization Panel (unchanged, supports 2 or 3)
# ------------------------------------------------------
def plot_gpu_usage_panel(
    ax,
    file1, label1,
    file2, label2,
    smooth_window=3,
    threshold=50,         # NEW — threshold for activity detection
    pre_seconds=0,        # NEW — how many seconds before the activity moment to include
    ft_tokens=None,
    colors=None,
):
    def load_log(path):
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["time_rel_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
        df["gpu_util_smooth"] = df["gpu_util"].rolling(window=smooth_window, min_periods=1).mean()
        return df

    df1 = load_log(file1)
    df2 = load_log(file2)

    # ------------------------------------------------------------
    # Find the "activity start" — first time GPU util > threshold
    # ------------------------------------------------------------
    def find_activity_start(df):
        idx = np.argmax(df["gpu_util"] > threshold)
        if df["gpu_util"].iloc[idx] > threshold:
            return df["time_rel_s"].iloc[idx]
        else:
            return df["time_rel_s"].iloc[0]  # fallback if no activity

    t1 = find_activity_start(df1)
    t2 = find_activity_start(df2)

    # Align both so their activity start = 0
    df1 = df1[df1["time_rel_s"] >= max(t1 - pre_seconds, 0)].copy()
    df2 = df2[df2["time_rel_s"] >= max(t2 - pre_seconds, 0)].copy()

    df1["time_rel_s"] -= (t1 - pre_seconds)
    df2["time_rel_s"] -= (t2 - pre_seconds)

    # ------------------------------------------------------------
    # Clip to common end for consistent x-axis
    # ------------------------------------------------------------
    min_end = min(df1["time_rel_s"].iloc[-1], df2["time_rel_s"].iloc[-1])
    df1 = df1[df1["time_rel_s"] <= min_end]
    df2 = df2[df2["time_rel_s"] <= min_end]

    # ------------------------------------------------------------
    # Compute averages
    # ------------------------------------------------------------
    avg1 = df1["gpu_util"].mean()
    avg2 = df2["gpu_util"].mean()

    color_map = colors if colors is not None else ["tab:orange", "tab:green"]

    # ------------------------------------------------------------
    # Plot curves
    # ------------------------------------------------------------
    ax.plot(df1["time_rel_s"], df1["gpu_util_smooth"], color=color_map[0], alpha=0.6)
    ax.plot(df2["time_rel_s"], df2["gpu_util_smooth"], color=color_map[1], alpha=0.6)

    ax.axhline(avg1, color=color_map[0], linestyle="--", linewidth=2.5,
               label=f"{label1} avg={avg1:.1f}%")
    ax.axhline(avg2, color=color_map[1], linestyle="--", linewidth=2.5,
               label=f"{label2} avg={avg2:.1f}%")

    if ft_tokens is not None:
        ax.plot([], [], " ", label=f"Total fine-tuning tokens: {ft_tokens:,}")

    ax.set_title("GPU Compute Utilization", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (s, aligned to first activity)")
    ax.set_ylabel("GPU Utilization (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

def plot_timeline_panel(ax, csv_path="timelines/timeline_live.csv"):
    df = pd.read_csv(csv_path)
    df["total_tokens"] = df["prompt_length"] + df["max_new_tokens"]
    df["second_bin"] = np.floor(df["timestamp_s"]).astype(int)

    tokens_per_sec = df.groupby("second_bin")["total_tokens"].sum()
    req_per_sec = df.groupby("second_bin").size()

    ax.plot(tokens_per_sec.index, tokens_per_sec, color="tab:blue", linewidth=1.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Total tokens/sec", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax.twinx()
    ax2.bar(req_per_sec.index, req_per_sec,
            color="tab:orange", alpha=0.3, width=0.9)
    ax2.set_ylabel("#Requests/sec", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax.set_title("Request Timeline", fontsize=11, fontweight="bold")

import json
from pathlib import Path

def extract_slo(json_path: str):
    """
    Extract ttft_slo, avg_tbt_slo, and max_tbt_slo from a JSON config file.

    Returns a dict:
      {
         "ttft_slo": float,
         "avg_tbt_slo": float,
         "max_tbt_slo": float,
      }

    Raises:
      FileNotFoundError if file does not exist.
      KeyError if any SLO field is missing.
      ValueError if JSON is malformed.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"SLO config file not found: {json_path}")

    try:
        with path.open("r") as f:
            data = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON at {json_path}: {e}")

    required_keys = ["ttft_slo", "avg_tbt_slo", "max_tbt_slo"]

    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(f"Missing SLO fields in JSON: {missing}")

    return {
        "ttft_slo": float(data["ttft_slo"]),
    }

# ------------------------------------------------------
# Combine All Panels
# ------------------------------------------------------
def plot_all_combined(ft_tokens=None, ttft_slo=None):
    latency_files = ["results/latency_co-serving.csv", "results/latency_slora.csv"]
    gpu_usage_files = ["results/gpu_usage_co-serving_0.csv", "results/gpu_usage_slora_0.csv"]
    timeline_file = "timelines/timeline_live.csv"
    labels = ["Co-serving", "SLoRA"]
    colors = ["tab:orange", "tab:green"]

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    ax_timeline, ax_ttft, ax_latency, ax_gpu = axes

    plot_timeline_panel(ax_timeline, csv_path=timeline_file)
    plot_latency_panels(ax_ttft, ax_latency, latency_files, labels, colors, ttft_slo=ttft_slo)
    plot_gpu_usage_panel(
        ax_gpu,
        gpu_usage_files[0], labels[0],
        gpu_usage_files[1], labels[1],
        ft_tokens=ft_tokens,
        colors=colors
    )

    fig.tight_layout()
    out_path = os.path.abspath("results/combined_summary.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"✅ Saved combined 4-panel figure to {out_path}")

if __name__ == "__main__":
    ttft_slo = extract_slo(
        "/projects/I20240005/jchen/slora-plus/S-LoRA/test/eval/config/finetuning_config_d.json"
    )["ttft_slo"]
    plot_all_combined(ft_tokens=267922, ttft_slo=ttft_slo)