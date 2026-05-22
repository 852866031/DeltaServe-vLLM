# multi_experiment_summary_clean.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# ------------------------------------------------------------
# Paper-style setup
# ------------------------------------------------------------
def set_paper_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "legend.fontsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "grid.color": "#dddddd",
        "lines.linewidth": 1.2,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })


# ------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------
def _safe_series(df, name):
    return df[name].dropna() if name in df.columns else pd.Series([], dtype=float)

def _percentile_curve(series):
    vals = series.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.array([]), np.array([])
    pct = np.linspace(0, 100, 101)
    return pct, np.percentile(vals, pct)


# ------------------------------------------------------------
# Panel functions
# ------------------------------------------------------------
def plot_timeline(ax, csv_path):
    df = pd.read_csv(csv_path)
    df["total_tokens"] = df["prompt_length"] + df["max_new_tokens"]
    df["second_bin"] = np.floor(df["timestamp_s"]).astype(int)
    tokens_per_sec = df.groupby("second_bin")["total_tokens"].sum()
    req_per_sec = df.groupby("second_bin").size()

    ax.plot(tokens_per_sec.index, tokens_per_sec, color="tab:blue", linewidth=1.0)
    ax.set_ylabel("Total tokens/sec", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.bar(req_per_sec.index, req_per_sec, color="tab:orange", alpha=0.25, width=0.9)
    ax2.set_title("Request timeline", fontsize=8, pad=2)
    ax2.set_ylabel("#Requests/sec", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_ylim(0, req_per_sec.max() * 1.2)
    
    ax.set_xlabel("Time (s)")

def plot_ttft_tbt(ax, files, labels, colors):
    dfs = [pd.read_csv(f) for f in files]
    dfs = [df[df.get("status", "ok") == "ok"] for df in dfs]
    pct = np.linspace(0, 100, 101)
    all_vals = []

    # --- Plot TTFT and Avg TBT ---
    for df, label, color in zip(dfs, labels, colors):
        _, y_ttft = _percentile_curve(_safe_series(df, "ttft_s"))
        _, y_avg  = _percentile_curve(_safe_series(df, "avg_tbt_s"))
        if y_ttft.size:
            ax.plot(pct, y_ttft, color=color, linestyle="-", label=f"TTFT ({label})")
            all_vals.append(y_ttft)
        if y_avg.size:
            ax.plot(pct, y_avg, color=color, linestyle="--", label=f"Avg TBT ({label})")
            all_vals.append(y_avg)

    # --- Dynamic y-limit ---
    all_vals = np.concatenate(all_vals) if all_vals else np.array([])
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size:
        ymax_cap = np.percentile(all_vals, 80)
        hard_max = all_vals.max()
        ymax_cap = (hard_max + ymax_cap) / 2
        ax.set_ylim(top=0.35)
        if hard_max > ymax_cap:
            ax.annotate(
                f"up to {hard_max:.3f}s",
                xy=(0.98, 0.35),
                xycoords=("axes fraction", "data"),
                ha="right", va="bottom",
                fontsize=4,
                color="gray",
                bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.9),
            )

    ax.set_title("TTFT / Avg TBT", fontsize=8, pad=2)
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Seconds")
    ax.grid(True, alpha=0.3)
    #ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper left")

    # --- Δ comparison box ---
    avg_ttft = [np.nanmean(_safe_series(df, "ttft_s")) for df in dfs]
    avg_tbt  = [np.nanmean(_safe_series(df, "avg_tbt_s")) for df in dfs]

    def delta_str(base, compare):
        if np.isnan(base) or np.isnan(compare): return "N/A"
        diff = compare - base
        pct  = (diff / base * 100) if base > 0 else np.nan
        sign = "+" if diff > 0 else ""
        return f"{sign}{diff:.3f}s ({sign}{pct:.1f}%)"

    def delta_str2(base, compare):
        if np.isnan(base) or np.isnan(compare): return "N/A"
        diff = compare - base
        sign = "+" if diff > 0 else ""
        return f"{sign}{diff:.3f}s"

    try:
        box_text = (
            f"{labels[0]}: Δ Avg TTFT {delta_str(avg_ttft[2], avg_ttft[0])}, \nΔ Avg TBT {delta_str2(avg_tbt[2], avg_tbt[0])}\n"
            f"{labels[1]}: Δ Avg TTFT {delta_str(avg_ttft[2], avg_ttft[1])}, \nΔ Avg TBT {delta_str2(avg_tbt[2], avg_tbt[1])}"
        )
        # how to make it at top left?
        ax.text(
            0.03, 0.95, box_text,
            transform=ax.transAxes,
            fontsize=5,
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray", alpha=0.85),
            family="monospace",
        )
    except Exception as e:
        print(f"[warn] TTFT/TBT box failed: {e}")


def plot_latency(ax, files, labels, colors):
    dfs = [pd.read_csv(f) for f in files]
    dfs = [df[df.get("status", "ok") == "ok"] for df in dfs]
    pct = np.linspace(0, 100, 101)
    all_vals = []

    # --- Plot total latency ---
    for df, label, color in zip(dfs, labels, colors):
        _, y_lat = _percentile_curve(_safe_series(df, "latency_s"))
        if y_lat.size:
            ax.plot(pct, y_lat, color=color, linewidth=1, label=label)
            all_vals.append(y_lat)

    # --- Dynamic y-limit ---
    all_vals = np.concatenate(all_vals) if all_vals else np.array([])
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size:
        ymax_cap = np.percentile(all_vals, 90)
        hard_max = all_vals.max()
        ymax_cap = (hard_max + ymax_cap) / 2
        ax.set_ylim(top=1.5, bottom=0.2)

    ax.set_title("Total Latency", fontsize=8, pad=2)
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Seconds")
    ax.grid(True, alpha=0.3)
    #ax.legend(fontsize=6.5, frameon=False, loc="upper left")

    # --- Δ latency comparison box ---
    avg_lat = [np.nanmean(_safe_series(df, "latency_s")) for df in dfs]

    def delta_str(base, compare):
        if np.isnan(base) or np.isnan(compare): return "N/A"
        diff = compare - base
        pct  = (diff / base * 100) if base > 0 else np.nan
        sign = "+" if diff > 0 else ""
        return f"{sign}{diff:.3f}s ({sign}{pct:.1f}%)"

    try:
        box_text = (
            "Δ Avg Total Latency (vs " + labels[2] + ")\n"
            f"{labels[0]}: {delta_str(avg_lat[2], avg_lat[0])}\n"
            f"{labels[1]}: {delta_str(avg_lat[2], avg_lat[1])}"
        )
        # how to make the following box at top left?
        ax.text(
            0.03, 0.95, box_text,
            transform=ax.transAxes,
            fontsize=5,
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray", alpha=0.85),
            family="monospace",
        )
    except Exception as e:
        print(f"[warn] Latency box failed: {e}")

def plot_gpu_usage(ax, files, colors, threshold=80, pre_seconds=3.0):
    def load_log(path):
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["time_rel_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
        df["gpu_util_smooth"] = df["gpu_util"].rolling(window=3, min_periods=1).mean()
        return df

    dfs = [load_log(f) for f in files]
    processed_dfs = []

    for d in dfs:
        # --- find first index where GPU util > threshold ---
        idx_active = np.argmax(d["gpu_util"] > threshold)
        if d["gpu_util"].iloc[idx_active] > threshold:
            t_start = d["time_rel_s"].iloc[idx_active] - pre_seconds
            if t_start < 0:
                t_start = 0
            d_trimmed = d[d["time_rel_s"] >= t_start].copy()
            # shift timeline so t_start = 0
            d_trimmed["time_rel_s"] -= t_start
        else:
            # fallback if never crosses threshold
            d_trimmed = d.copy()
            d_trimmed["time_rel_s"] -= d_trimmed["time_rel_s"].iloc[0]

        processed_dfs.append(d_trimmed)

    # --- truncate to common end time for visual consistency ---
    min_end = min(d["time_rel_s"].iloc[-1] for d in processed_dfs)
    processed_dfs = [d[d["time_rel_s"] <= min_end] for d in processed_dfs]

    # --- plot smoothed curves and avg lines ---
    avg_utils = []
    for df, color in zip(processed_dfs, colors):
        ax.plot(df["time_rel_s"], df["gpu_util_smooth"], color=color, linewidth=1.0)
        avg = df["gpu_util"].mean()
        avg_utils.append(avg)
        ax.axhline(avg, color=color, linestyle="--", linewidth=1.0, alpha=0.8)

    # --- add GPU utilization comparison box (file2 vs file3) ---
    if len(avg_utils) >= 3:
        base = avg_utils[2]  # SLoRA or reference
        compare = avg_utils[1]  # co-serving
        diff = compare - base
        sign = "+" if diff > 0 else ""
        pct = (diff / base * 100) if base > 0 else 0
        text_box = (
            f"ΔUtil Co vs SLoRA: {sign}{diff:.1f}%"
        )
        # how to make it at top left?
        ax.text(
            0.03, 0.95, text_box,
            transform=ax.transAxes,
            fontsize=4,
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8),
            family="monospace",
        )

    # --- style ---
    ax.set_title("GPU Utilization", fontsize=8, pad=2)
    ax.set_ylabel("GPU Utilization (%)")
    ax.set_xlabel(f"Time (s)")
    ax.set_ylim(0, 120)
    ax.set_yticks(np.arange(0, 101, 25))
    ax.grid(True, alpha=0.3)
# ------------------------------------------------------------
# Combined plot
# ------------------------------------------------------------
def plot_multi_experiment_summary(base_dirs):
    set_paper_style()
    n_rows, n_cols = 4, len(base_dirs)
    fig = plt.figure(figsize=(7.5, 8))
    gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.05, hspace=0.55)  # <- larger hspace


    colors = sns.color_palette("colorblind", n_colors=3)
    labels = ["Inf", "Co", "SLoRA"]

    shared_axes = [None] * n_rows
    axes_matrix = [[None]*n_cols for _ in range(n_rows)]

    for col, base in enumerate(base_dirs):
        lat_files = [f"{base}/latency_inference.csv", f"{base}/latency_co-serving.csv", f"{base}/latency_slora.csv"]
        gpu_files = [f"{base}/gpu_usage_inference.csv", f"{base}/gpu_usage_co-serving.csv", f"{base}/gpu_usage_slora.csv"]
        timeline_csv = f"{base}/timeline_live.csv"

        # --- Row 1: Request timeline
        ax1 = fig.add_subplot(gs[0, col], sharey=shared_axes[0])
        plot_timeline(ax1, timeline_csv)
        if shared_axes[0] is None:
            shared_axes[0] = ax1
            ax1.set_ylabel("Total tokens/sec")
        else:
            ax1.set_ylabel("")
            ax1.tick_params(labelleft=False)

        # --- Row 2: TTFT / TBT
        ax2 = fig.add_subplot(gs[1, col], sharey=shared_axes[1])
        plot_ttft_tbt(ax2, lat_files, labels, colors)
        if shared_axes[1] is None:
            shared_axes[1] = ax2
            ax2.set_ylabel("Seconds")
        else:
            ax2.set_ylabel("")
            ax2.tick_params(labelleft=False)

        # --- Row 3: Latency
        ax3 = fig.add_subplot(gs[2, col], sharey=shared_axes[2])
        plot_latency(ax3, lat_files, labels, colors)
        if shared_axes[2] is None:
            shared_axes[2] = ax3
            ax3.set_ylabel("Seconds")
        else:
            ax3.set_ylabel("")
            ax3.tick_params(labelleft=False)

        # --- Row 4: GPU usage
        ax4 = fig.add_subplot(gs[3, col], sharey=shared_axes[3])
        plot_gpu_usage(ax4, gpu_files, colors)
        if shared_axes[3] is None:
            shared_axes[3] = ax4
            ax4.set_ylabel("GPU Utilization (%)")
        else:
            ax4.set_ylabel("")
            ax4.tick_params(labelleft=False)

        # # --- Column title (experiment name)
        # ax1.set_title(os.path.basename(base).capitalize(), fontweight="bold", fontsize=10, pad=16)
        # axes_matrix[0][col] = ax1
        # axes_matrix[1][col] = ax2
        # axes_matrix[2][col] = ax3
        # axes_matrix[3][col] = ax4

    # --- Global legend (top)
    handles = [plt.Line2D([0], [0], color=c, lw=1.2) for c in colors]
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1),   #
        ncol=3,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.5,
        fontsize=7,
    )

    # --- Global labels
    fig.text(0.015, 0.5, "Metric Value", va="center", rotation="vertical", fontsize=8)

    # --- Layout fix to avoid overlap
    plt.subplots_adjust(top=0.93, bottom=0.06, left=0.08, right=0.98, wspace=0.08, hspace=0.35)

    out_path = os.path.abspath("comparison_summary.pdf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved clean multi-experiment figure → {out_path}")


# ------------------------------------------------------------
if __name__ == "__main__":
    dirs = ["loose", "tight", "nutanix"]
    plot_multi_experiment_summary(dirs)