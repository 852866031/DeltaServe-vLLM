# multi_experiment_summary_clean.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

# ------------------------------------------------------------
# Global Plot Configuration
# ------------------------------------------------------------
PLOT_CFG = {
    "slo": 0.1,

    # Column layout
    "col_width_ratio": [1, 1, 1.5],       # <--- CHANGE COLUMN WIDTHS HERE
    "last_col_shift": 0.02,              # <--- SHIFT LAST COLUMN TO THE RIGHT

    "timeline": {
        "left_ylim":  [0, 1900],
        "left_ytick": [0, 500, 1000, 1500],
        "right_ylim": [0, 20],
        "right_ytick": [0, 5, 10, 15, 20],
        "xticks": [0, 20, 40, 60],
        "xticks_last": [0, 5, 10, 15, 20],
    },
    "ttft_tbt": {
        "left_ylim":  [0, 0.35],
        "left_ytick": [0, 0.1, 0.2, 0.3],
    },
    "latency": {
        "left_ylim":  [0, 3],
        "left_ytick": [0, 1, 2],
    },
    "gpu": {
        "left_ylim":  [0, 120],
        "left_ytick": [0, 25, 50, 75, 100],
        "xticks": [0, 20, 40, 60],
        "xticks_last": [0, 5, 10, 15, 20],
    },
    "x_tick_pad": -5,
    "y_tick_pad": -5
}

def compute_title_x(cfg = PLOT_CFG):
    """
    Center the title over the middle column (col=1) when there are exactly 3 columns.
    """
    ratios = cfg["col_width_ratio"]      # e.g. [1, 1, 1.5]
    shift  = cfg["last_col_shift"]       # e.g. 0.02

    # Normalize column widths
    total = sum(ratios)
    norm = [r / total for r in ratios]

    # Middle column boundaries
    left  = norm[0]              # boundary after col 0
    right = norm[0] + norm[1]    # boundary after col 1

    col_center = (left + right) / 2.0  # center of column 1 in figure coords

    # Convert figure-coord center to axes-coord center for col=1.
    # Because axes use local [0–1], we approximate correction:
    #
    # The last column is shifted right, so visually ALL titles must shift right slightly.
    #
    correction = shift * 5.0   # empirical good factor

    return 0.5 + correction + 0.1

# ------------------------------------------------------------
# Style
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
# Helpers
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
# Timeline plot
# ------------------------------------------------------------
def plot_timeline(
    ax, csv_path,
    left_color="#5E4FA2",
    right_color="#000000",
    right_alpha=0.1,
    processed_ft_tokens=None,
    show_title=True,
    use_last_cfg=False,
):
    cfg_key = "timeline"
    cfg = PLOT_CFG[cfg_key]

    df = pd.read_csv(csv_path)
    df["total_tokens"] = df["prompt_length"] + df["max_new_tokens"]
    df["second_bin"] = np.floor(df["timestamp_s"]).astype(int)

    start = int(df["second_bin"].min())
    end   = int(df["second_bin"].max())
    full_index = pd.Index(range(start, end + 1), name="second_bin")

    tokens_raw = df.groupby("second_bin")["total_tokens"].sum()
    rps_raw    = df.groupby("second_bin").size()

    tokens_per_sec = tokens_raw.reindex(full_index, fill_value=0)
    req_per_sec    = rps_raw.reindex(full_index, fill_value=0)

    duration = max(full_index.max() - full_index.min(), 1)

    if show_title:
        ax.set_title("Inference Request Timeline", fontsize=8, pad=2, x=compute_title_x())

    use_minutes = duration > 1000
    if use_minutes:
        x_vals = (tokens_per_sec.index - tokens_per_sec.index.min()) / 60.0
    else:
        x_vals = tokens_per_sec.index

    ax.plot(x_vals, tokens_per_sec, color=left_color, linewidth=1.0)
    ax.set_ylabel("Total tokens/sec", color=left_color)
    ax.tick_params(axis="y", labelcolor=left_color)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(x_vals, req_per_sec, color=right_color, alpha=0.8, linewidth=0.7)
    ax2.fill_between(x_vals, req_per_sec, color=right_color, alpha=right_alpha)

    ax2.set_ylabel("#Requests/sec", color=right_color)
    ax2.tick_params(axis="y", labelcolor=right_color)
    ax.tick_params(axis="x", pad=PLOT_CFG["x_tick_pad"])
    ax2.tick_params(axis="x", pad=PLOT_CFG["x_tick_pad"])
    ax.tick_params(axis="y", direction="out", length=0)
    ax2.tick_params(axis="y", direction="out", length=0)

    cfg = PLOT_CFG["timeline"]
    ax.set_ylim(cfg["left_ylim"])
    ax.set_yticks(cfg["left_ytick"])
    ax2.set_ylim(cfg["right_ylim"])
    ax2.set_yticks(cfg["right_ytick"])

    if use_last_cfg:
        ax.set_xticks(cfg["xticks_last"])
        ax2.set_xticks(cfg["xticks_last"])

    ax.set_xlabel("Time (min)" if use_minutes else "Time (s)")

    return ax2, duration

# ------------------------------------------------------------
# TTFT / TBT
# ------------------------------------------------------------
def plot_ttft_tbt(ax, files, labels, colors, show_slo_label=False, show_title=True, use_last_cfg=False):
    cfg_key = "ttft_tbt"
    cfg = PLOT_CFG[cfg_key]

    dfs = [pd.read_csv(f) for f in files]
    dfs = [df[df.get("status", "ok") == "ok"] for df in dfs]

    pct = np.linspace(0, 100, 101)
    all_vals = []

    for df, label, color in zip(dfs, labels, colors):
        _, y_ttft = _percentile_curve(_safe_series(df, "ttft_s"))
        _, y_avg  = _percentile_curve(_safe_series(df, "avg_tbt_s"))

        if y_ttft.size:
            ax.plot(pct, y_ttft, color=color, linestyle="-", label=f"TTFT ({label})")
            all_vals.append(y_ttft)
        if y_avg.size:
            ax.plot(pct, y_avg, color=color, linestyle="--", label=f"Avg TBT ({label})")
            all_vals.append(y_avg)

    if show_title:
        ax.set_title("Inference Request TTFT / Avg TBT", fontsize=8, pad=2, x=compute_title_x())

    ax.set_xlabel("Percentile")
    ax.set_ylabel("Seconds")
    ax.grid(True, alpha=0.3)

    slo_val = PLOT_CFG.get("slo", None)
    y_bottom, y_top = cfg["left_ylim"]

    ax.set_ylim(y_bottom, y_top)
    ax.set_yticks(cfg["left_ytick"])
    ax.tick_params(axis="y", pad=PLOT_CFG["y_tick_pad"])
    ax.tick_params(axis="x", pad=PLOT_CFG["x_tick_pad"])

    if slo_val is not None:
        ax.axhline(slo_val, color="#666666", linestyle="--", linewidth=1.0, alpha=0.8)
        if show_slo_label:
            ax.text(
                1.0, slo_val,
                f" SLO {slo_val:.1f}s",
                va="bottom", ha="left",
                fontsize=5.5,
                color="#444444",
                transform=ax.get_yaxis_transform()
            )

    if all_vals:
        real_max = np.max(np.concatenate(all_vals))
        if real_max > y_top:
            ax.text(
                1.03, 0.98,
                f"up to {real_max:.3f}s",
                transform=ax.transAxes,
                fontsize=5,
                ha="right", va="top",
                color="gray",
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="white", ec="none", alpha=0.8)
            )

    avg_ttft = [np.nanmean(_safe_series(df, "ttft_s")) for df in dfs]
    avg_tbt  = [np.nanmean(_safe_series(df, "avg_tbt_s")) for df in dfs]

    try:
        base_ttft = avg_ttft[1]
        co_ttft   = avg_ttft[0]
        base_tbt  = avg_tbt[1]
        co_tbt    = avg_tbt[0]

        def delta_1(base, other):
            if np.isnan(base) or np.isnan(other): return "N/A"
            diff = other - base
            pct  = diff / base * 100 if base > 0 else np.nan
            sign = "+" if diff > 0 else ""
            return f"{sign}{diff:.3f}s ({sign}{pct:.1f}%)"

        def delta_2(base, other):
            if np.isnan(base) or np.isnan(other): return "N/A"
            diff = other - base
            sign = "+" if diff > 0 else ""
            return f"{sign}{diff:.3f}s"

        text = (
            "Δ Avg TTFT : " + delta_1(base_ttft, co_ttft) + "\n"
            "Δ Avg TBT  : " + delta_2(base_tbt, co_tbt)
        )

        ax.text(
            0.03, 0.95, text,
            transform=ax.transAxes,
            fontsize=5,
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.35",
                      fc="white", ec="gray", alpha=0.85),
            family="monospace",
        )
    except Exception as e:
        print(f"[warn] TTFT/TBT box failed: {e}")

# ------------------------------------------------------------
# Latency
# ------------------------------------------------------------
def plot_latency(ax, files, labels, colors, show_title=True, use_last_cfg=False, processed_ft_tokens=None, duration=None):
    cfg_key = "latency"
    cfg = PLOT_CFG[cfg_key]
    dfs = [pd.read_csv(f) for f in files]
    dfs = [df[df.get("status", "ok") == "ok"] for df in dfs]
    pct = np.linspace(0, 100, 101)
    all_vals = []

    for df, label, color in zip(dfs, labels, colors):
        _, y_lat = _percentile_curve(_safe_series(df, "latency_s"))
        if y_lat.size:
            ax.plot(pct, y_lat, color=color, linewidth=1)
            all_vals.append(y_lat)

    if show_title:
        ax.set_title("Inference Request End-to-end Latency", fontsize=8, pad=2, x=compute_title_x())
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Seconds")
    ax.grid(True, alpha=0.3)

    y_bottom, y_top = cfg["left_ylim"]
    ax.set_ylim(y_bottom, y_top)
    ax.set_yticks(cfg["left_ytick"])
    ax.tick_params(axis="x", pad=PLOT_CFG["x_tick_pad"])
    ax.tick_params(axis="y", pad=PLOT_CFG["y_tick_pad"])

    if all_vals:
        real_max = np.max(np.concatenate(all_vals))
        if real_max > y_top:
            ax.text(
                1, 0.98,
                f"up to {real_max:.3f}s",
                transform=ax.transAxes,
                fontsize=5,
                ha="right", va="top",
                color="gray",
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="white", ec="none", alpha=0.8),
            )

    avg_lat = [np.nanmean(_safe_series(df, "latency_s")) for df in dfs]

    try:
        base = avg_lat[1]
        co   = avg_lat[0]

        def delta(base, other):
            if np.isnan(base) or np.isnan(other): return "N/A"
            diff = other - base
            pct  = diff / base * 100 if base > 0 else np.nan
            sign = "+" if diff > 0 else ""
            return f"{sign}{diff:.3f}s ({sign}{pct:.1f}%)"

        text = "Δ Avg Total Latency: " + delta(base, co)

        ax.text(
            0.03, 0.95, text,
            transform=ax.transAxes,
            fontsize=5,
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.35",
                      fc="white", ec="gray", alpha=0.85),
            family="monospace",
        )
        if processed_ft_tokens is not None:
            token_str = f"{processed_ft_tokens:,}"
            bold_total = rf"$\mathbf{{{token_str}}}$"
            avg_tokens_per_sec = processed_ft_tokens / duration
            bold_avg = rf"$\mathbf{{{avg_tokens_per_sec:.1f}}}$"

            text = (
                f"Total FT tokens processed: {bold_total}\n"
                f"(avg. {bold_avg} tokens/sec)"
            )

            ax.text(
                0.03, 0.80,
                text,
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=5,
                bbox=dict(boxstyle="round,pad=0.30",
                        fc="white", ec="gray", alpha=0.85)
            )

    except Exception as e:
        print(f"[warn] Latency box failed: {e}")

# ------------------------------------------------------------
# GPU Usage
# ------------------------------------------------------------
def plot_gpu_usage(ax, files, colors, threshold=80, pre_seconds=3.0, show_title=True, use_last_cfg=False):
    cfg_key = "gpu"
    cfg = PLOT_CFG[cfg_key]
    def load_log(path):
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["time_rel_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
        df["gpu_util_smooth"] = df["gpu_util"].rolling(window=3, min_periods=1).mean()
        return df

    dfs = [load_log(f) for f in files]
    processed = []

    for d in dfs:
        idx = np.argmax(d["gpu_util"] > threshold)
        if d["gpu_util"].iloc[idx] > threshold:
            t0 = max(d["time_rel_s"].iloc[idx] - pre_seconds, 0)
            dd = d[d["time_rel_s"] >= t0].copy()
            dd["time_rel_s"] -= t0
        else:
            dd = d.copy()
            dd["time_rel_s"] -= dd["time_rel_s"].iloc[0]
        processed.append(dd)

    min_end = min(d["time_rel_s"].iloc[-1] for d in processed)
    processed = [d[d["time_rel_s"] <= min_end] for d in processed]

    use_minutes = min_end > 1000
    x_label = "Time (min)" if use_minutes else "Time (s)"

    if use_minutes:
        for d in processed:
            d["time_rel_min"] = d["time_rel_s"] / 60.0
        x_key = "time_rel_min"
    else:
        x_key = "time_rel_s"

    avg_utils = []
    for df, color in zip(processed, colors):
        ax.plot(df[x_key], df["gpu_util_smooth"], color=color, linewidth=1.0)
        avg = df["gpu_util"].mean()
        avg_utils.append(avg)
        ax.axhline(avg, color=color, linestyle="--", linewidth=1.0, alpha=0.8)

    ax.set_ylim(cfg["left_ylim"])
    ax.set_yticks(cfg["left_ytick"])
    ax.set_xticks(cfg["xticks"])
    ax.tick_params(axis="x", pad=PLOT_CFG["x_tick_pad"])
    ax.tick_params(axis="y", pad=PLOT_CFG["y_tick_pad"])

    if use_last_cfg:
        ax.set_xticks(cfg["xticks_last"])
        ax.set_xlim(cfg["xticks_last"][0], cfg["xticks_last"][-1])

    if len(avg_utils) >= 2:
        base = avg_utils[1]
        co   = avg_utils[0]
        diff = co - base
        txt = f"ΔUtil: {diff:+.1f}%"

        ax.text(
            0.03, 0.95, txt,
            transform=ax.transAxes,
            fontsize=5,
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.4",
                      fc="white", ec="gray", alpha=0.8),
            family="monospace",
        )

    if show_title:
        ax.set_title("GPU Utilization", fontsize=8, pad=2, x=compute_title_x())
    ax.set_ylabel("GPU Utilization (%)")
    ax.set_xlabel(x_label)
    ax.grid(True, alpha=0.3)

# ------------------------------------------------------------
# Combined Plot
# ------------------------------------------------------------
def plot_multi_experiment_summary(base_dirs):
    set_paper_style()
    n_rows, n_cols = 4, len(base_dirs)

    fig = plt.figure(figsize=(7, 5))
    gs = GridSpec(
        n_rows,
        n_cols,
        figure=fig,
        wspace=0.05,
        hspace=0.55,
        width_ratios=PLOT_CFG["col_width_ratio"]
    )

    labels = ["Co", "S-LoRA"]
    colors = ["#FF7F0E", "#1F77B4"]

    LEFT_TIMELINE_COLOR = "#5E4FA2"
    RIGHT_TIMELINE_COLOR = "#000000"
    RIGHT_TIMELINE_ALPHA = 0.2

    num_ft_processed = [16834, 10481, 267922]
    shared_axes = [None] * n_rows

    for col, base in enumerate(base_dirs):

        lat_files = [
            f"{base}/latency_co-serving.csv",
            f"{base}/latency_slora.csv"
        ]
        gpu_files = [
            f"{base}/gpu_usage_co-serving.csv",
            f"{base}/gpu_usage_slora.csv"
        ]
        timeline_csv = f"{base}/timeline_live.csv"
        show_title = (col == 1)

        # --------------------------
        # ROW 1: TIMELINE
        # --------------------------
        if col == n_cols - 1:
            ax1 = fig.add_subplot(gs[0, col])   # last column = no sharey
        else:
            ax1 = fig.add_subplot(gs[0, col], sharey=shared_axes[0])

        use_last = (col == n_cols - 1)

        ax2, duration = plot_timeline(
            ax1, timeline_csv,
            left_color=LEFT_TIMELINE_COLOR,
            right_color=RIGHT_TIMELINE_COLOR,
            right_alpha=RIGHT_TIMELINE_ALPHA,
            processed_ft_tokens=num_ft_processed[col],
            show_title=show_title,
            use_last_cfg=use_last
        )

        # -- Hide right-axis for first two columns only
        if col != n_cols - 1:
            ax2.set_ylabel("")
            ax2.set_yticklabels([])
            ax2.tick_params(axis="y", right=False, labelright=False)
            if "right" in ax2.spines:
                ax2.spines["right"].set_visible(False)

        # -- Shared y-axis logic for columns 0 and 1
        if shared_axes[0] is None:
            shared_axes[0] = ax1
            ax1.set_ylabel("Total tokens/sec")
        else:
            ax1.set_ylabel("")
            ax1.tick_params(labelleft=False)


        # if col == n_cols - 1:
        #     ax1.tick_params(labelleft=True)
        #     ax1.set_ylabel("Total tokens/sec")

        # ---- SHIFT LAST COLUMN ----
        if col == n_cols - 1:
            pos = ax1.get_position()
            ax1.set_position([
                pos.x0 + PLOT_CFG["last_col_shift"],
                pos.y0,
                pos.width,
                pos.height
            ])
            ax2_pos = ax2.get_position()
            ax2.set_position([
                ax2_pos.x0 + PLOT_CFG["last_col_shift"],
                ax2_pos.y0,
                ax2_pos.width,
                ax2_pos.height
            ])

        # --------------------------
        # ROW 2: TTFT/TBT
        # --------------------------
        ax2_row = fig.add_subplot(gs[1, col], sharey=shared_axes[1])
        show_slo_label = (col == n_cols - 1)

        plot_ttft_tbt(
            ax2_row, lat_files, labels, colors,
            show_slo_label, show_title,
            use_last_cfg=use_last
        )

        if shared_axes[1] is None:
            shared_axes[1] = ax2_row
            ax2_row.set_ylabel("Seconds")
        else:
            ax2_row.set_ylabel("")
            ax2_row.tick_params(labelleft=False)

        if col == n_cols - 1:
            pos = ax2_row.get_position()
            ax2_row.set_position([
                pos.x0 + PLOT_CFG["last_col_shift"],
                pos.y0,
                pos.width,
                pos.height
            ])

        # --------------------------
        # ROW 3: LATENCY
        # --------------------------
        ax3 = fig.add_subplot(gs[2, col], sharey=shared_axes[2])
        plot_latency(ax3, lat_files, labels, colors, show_title, use_last_cfg=use_last, processed_ft_tokens=num_ft_processed[col], duration=duration)

        if shared_axes[2] is None:
            shared_axes[2] = ax3
            ax3.set_ylabel("Seconds")
        else:
            ax3.set_ylabel("")
            ax3.tick_params(labelleft=False)

        if col == n_cols - 1:
            pos = ax3.get_position()
            ax3.set_position([
                pos.x0 + PLOT_CFG["last_col_shift"],
                pos.y0,
                pos.width,
                pos.height
            ])

        # --------------------------
        # ROW 4: GPU UTILIZATION
        # --------------------------
        ax4 = fig.add_subplot(gs[3, col], sharey=shared_axes[3])
        plot_gpu_usage(ax4, gpu_files, colors, show_title=show_title, use_last_cfg=use_last)

        if shared_axes[3] is None:
            shared_axes[3] = ax4
            ax4.set_ylabel("GPU Utilization (%)")
        else:
            ax4.set_ylabel("")
            ax4.tick_params(labelleft=False)

        if col == n_cols - 1:
            pos = ax4.get_position()
            ax4.set_position([
                pos.x0 + PLOT_CFG["last_col_shift"],
                pos.y0,
                pos.width,
                pos.height
            ])

    # ------------------------
    # GLOBAL LEGEND
    # ------------------------
    # ----------------------------------------------------
    # Master legends (unchanged)
    # ----------------------------------------------------
    row1_handles = [
        plt.Line2D([0],[0],color=LEFT_TIMELINE_COLOR,lw=1.2),
        Patch(facecolor=RIGHT_TIMELINE_COLOR,alpha=RIGHT_TIMELINE_ALPHA),
        plt.Line2D([0],[0],color="#666666",linestyle="--",lw=1.2),
    ]
    row1_labels = [
        "Total Inference Tokens/sec",
        "Incoming Requests/sec",
        "TTFT SLO",
    ]
    fig.legend(
        row1_handles, row1_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.025),
        ncol=3, frameon=False, fontsize=7,
    )

    row2_handles = [
        plt.Line2D([0],[0],color=colors[0],lw=1.2),
        plt.Line2D([0],[0],color=colors[1],lw=1.2),
        plt.Line2D([0],[0],color=colors[0],linestyle="--",lw=1.2),
        plt.Line2D([0],[0],color=colors[1],linestyle="--",lw=1.2),
    ]
    row2_labels = ["DeltaServe","S-LoRA","DeltaServe Avg GPU","S-LoRA Avg GPU"]

    fig.legend(
        row2_handles, row2_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        ncol=4, frameon=False, fontsize=7,
    )

    plt.subplots_adjust(
        top=0.93, bottom=0.06,
        left=0.08, right=0.98,
        wspace=0.08, hspace=0.35
    )

    out_path = os.path.abspath("comparison_summary.pdf")
    plt.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    print(f"✅ Saved clean multi-experiment figure → {out_path}")
# ------------------------------------------------------------
if __name__ == "__main__":
    dirs = ["loose", "tight", "nutanix"]
    plot_multi_experiment_summary(dirs)