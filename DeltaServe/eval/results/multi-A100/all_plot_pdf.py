import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import Patch

# ------------------------------------------------------------
# Global Plot Configuration
# ------------------------------------------------------------
PLOT_CFG = {
    "slo": 0.3,
    # Column layout
    "col_width_ratio": [1, 1, 1.4],
    "last_col_shift": 0.02,
    "timeline": {
        "left_ylim":  [0, 16_000],
        "left_ytick": [0, 4_000, 8_000, 12_000, 16_000],
        "right_ylim": [0, 100],
        "right_ytick": [0, 25, 50, 75, 100],
        "xticks": [0, 20, 40, 60],
        "xticks_last": [0, 5, 10, 15, 20],
        "x_lim": [0, 65],
        "x_lim_last": [0, 22]
    },
    "ttft_tbt": {
        "left_ylim":  [0, 1.2],
        "left_ytick": [0, 0.4, 0.8, 1.2],
    },
    "req_latency_timeline": {
        "left_ylim":  [1, 7],          # adjust if needed
        "left_ytick": [2, 4, 6],
        "xticks":     None,            # auto unless overridden
        "xticks_last": None,
    },
    "ft_tokens": {
        "left_ylim":  [0, 1_000_000],     # example — your Nutanix case
        "left_ytick": [300_000, 600_000, 900_000],
        "left_ylim_small":  [0,  80_000],  # for "loose" and "tight"
        "left_ytick_small": [20_000, 40_000, 60_000, 80_000],
        "x_lim": [0, 62],
        "x_lim_last": [0, 22] ,
        "xticks_last": [0, 5, 10, 15, 20],
        "xticks": [0, 20, 40, 60],
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

def compute_title_x(cfg=PLOT_CFG):
    """
    Center titles roughly over the middle column for a 3-col layout,
    with a small empirical correction for the last-column shift.
    """
    ratios = cfg["col_width_ratio"]
    shift = cfg["last_col_shift"]
    total = sum(ratios)
    norm = [r / total for r in ratios]
    left = norm[0]
    right = norm[0] + norm[1]
    col_center = (left + right) / 2.0
    correction = shift * 5.0
    return 0.5 + correction

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

# ------------------------------------------------------
# Utility Functions
# ------------------------------------------------------
def _safe_series(df, name):
    return df[name].dropna() if name in df.columns else pd.Series([], dtype=float)

def _percentile_curve(series):
    vals = series.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.array([]), np.array([])
    pct = np.linspace(0, 100, 101)
    return pct, np.percentile(vals, pct)

def _safe_label(label: str) -> str:
    return label.replace("_", " ")

# ------------------------------------------------------
# Timeline (Row 1)
# ------------------------------------------------------
def plot_timeline(
    ax, csv_path,
    left_color="#5E4FA2",
    right_color="#000000",
    right_alpha=0.2,
    show_title=True,
    use_last_cfg=False,
):
    cfg = PLOT_CFG["timeline"]
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

    use_minutes = duration > 1000
    if use_minutes:
        x_vals = (tokens_per_sec.index - tokens_per_sec.index.min()) / 60.0
        x_label = "Time (min)"
    else:
        x_vals = tokens_per_sec.index - tokens_per_sec.index.min()
        x_label = "Time (s)"

    if show_title:
        ax.set_title("Inference Request Timeline", fontsize=8, pad=2, x=compute_title_x())

    # left axis: tokens/sec
    ax.plot(x_vals, tokens_per_sec.values, color=left_color, linewidth=1.0)
    ax.set_ylabel("Total tokens/sec", color=left_color)
    ax.tick_params(axis="y", labelcolor=left_color)
    ax.grid(True, alpha=0.3)

    # right axis: req/sec (filled)
    ax2 = ax.twinx()
    ax2.plot(x_vals, req_per_sec.values, color=right_color, linewidth=0.7, alpha=0.8)
    ax2.fill_between(x_vals, req_per_sec.values, color=right_color, alpha=right_alpha)
    ax2.set_ylabel("#Requests/sec", color=right_color)
    ax2.tick_params(axis="y", labelcolor=right_color)
    ax.tick_params(axis="y", direction="out", length=0)
    ax2.tick_params(axis="y", direction="out", length=0)

    # y limits / ticks
    ax.set_ylim(cfg["left_ylim"])
    ax.set_yticks(cfg["left_ytick"])
    ax2.set_ylim(cfg["right_ylim"])
    ax2.set_yticks(cfg["right_ytick"])
    ax.set_xlim(cfg["x_lim"] if not use_last_cfg else cfg["x_lim_last"])

    # x ticks
    if use_last_cfg:
        xticks = cfg["xticks_last"]
    else:
        xticks = cfg["xticks"]
    ax.set_xticks(xticks)
    ax2.set_xticks(xticks)

    ax.set_xlabel(x_label, labelpad=2)
    ax.tick_params(axis="x", pad=PLOT_CFG["x_tick_pad"])
    ax2.tick_params(axis="x", pad=PLOT_CFG["x_tick_pad"])
    #ax.tick_params(axis="y", pad=PLOT_CFG["y_tick_pad"]+1)

    return ax, ax2, duration

# ------------------------------------------------------
# TTFT / Avg TBT (Row 2)
# ------------------------------------------------------
def plot_ttft_tbt(ax, files, labels, colors, show_slo_label=False, show_title=True, use_last_cfg=False):
    cfg = PLOT_CFG["ttft_tbt"]
    dfs = [pd.read_csv(f) for f in files]
    dfs = [df[df.get("status", "ok") == "ok"] for df in dfs]

    pct = np.linspace(0, 100, 101)
    all_vals = []

    for df, label, color in zip(dfs, labels, colors):
        _, y_ttft = _percentile_curve(_safe_series(df, "ttft_s"))
        _, y_avg  = _percentile_curve(_safe_series(df, "avg_tbt_s"))

        if y_ttft.size:
            ax.plot(pct, y_ttft, color=color, linestyle="-", label=f"TTFT ({_safe_label(label)})")
            all_vals.append(y_ttft)
        if y_avg.size:
            ax.plot(pct, y_avg, color=color, linestyle="--", label=f"Avg TBT ({_safe_label(label)})")
            all_vals.append(y_avg)

    if show_title:
        ax.set_title("Inference Request TTFT / Avg TBT", fontsize=8, pad=2, x=compute_title_x())

    ax.set_xlabel("Percentile", labelpad=2)
    ax.tick_params(axis="x", pad=PLOT_CFG["x_tick_pad"])
    ax.tick_params(axis="y", pad=PLOT_CFG["y_tick_pad"])
    ax.set_ylabel("Seconds")
    ax.grid(True, alpha=0.3)

    y_bottom, y_top = cfg["left_ylim"]
    ax.set_ylim(y_bottom, y_top)
    ax.set_yticks(cfg["left_ytick"])

    slo_val = PLOT_CFG.get("slo", None)
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
                1.0, 0.98,
                f"up to {real_max:.3f}s",
                transform=ax.transAxes,
                fontsize=5,
                ha="right", va="top",
                color="gray",
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="white", ec="none", alpha=0.8)
            )

    # average TTFT / TBT box
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

def plot_request_latency_timeline(ax, latency_files, labels, colors, show_title=True, use_last_cfg=False):
    cfg = PLOT_CFG["req_latency_timeline"]

    dfs = [pd.read_csv(f) for f in latency_files]

    for df, label, color in zip(dfs, labels, colors):
        df = df[df.get("status", "ok") == "ok"]

        # Use existing t_rel_s if present
        if "t_rel_s" in df.columns:
            t = df["t_rel_s"]
        else:
            # fallback to monotonic time axis
            t = np.arange(len(df))

        ax.scatter(
            t,
            df["latency_s"],
            s=3,                  # much smaller marker
            alpha=0.15,           # more transparent
            linewidths=0.5,         # remove stroke
            rasterized=True,      # rasterize scatter for clarity
            color=color,
        )

    if show_title:
        ax.set_title(
            "Inference Request End-to-end Latency Timeline",
            fontsize=8,
            pad=2,
            x=compute_title_x()
        )

    ax.set_xlabel("Time (s)", labelpad=2)
    ax.set_ylabel("Latency (s)")
    ax.grid(True, alpha=0.3)

    # Apply cfg limits if provided
    if cfg["left_ylim"] is not None:
        ax.set_ylim(cfg["left_ylim"])
    if cfg["left_ytick"] is not None:
        ax.set_yticks(cfg["left_ytick"])

    # X ticks
    if use_last_cfg and cfg["xticks_last"] is not None:
        ax.set_xticks(cfg["xticks_last"])
    elif cfg["xticks"] is not None:
        ax.set_xticks(cfg["xticks"])

    ax.tick_params(axis="x", pad=PLOT_CFG["x_tick_pad"])
    ax.tick_params(axis="y", pad=PLOT_CFG["y_tick_pad"])
        # ------------------------------------------------------
    # Average latency difference annotation (Co vs SLoRA)
    # ------------------------------------------------------
    try:
        df_co = dfs[0][dfs[0].get("status", "ok") == "ok"]
        df_sl = dfs[1][dfs[1].get("status", "ok") == "ok"]

        avg_co = np.nanmean(df_co["latency_s"])
        avg_sl = np.nanmean(df_sl["latency_s"])

        # Absolute difference
        diff = avg_co - avg_sl
        sign = "+" if diff > 0 else ""

        # Percentage difference relative to SLoRA
        if avg_sl > 0:
            pct = diff / avg_sl * 100
            pct_sign = "+" if pct > 0 else ""
            pct_str = f"{pct_sign}{pct:.1f}%"
        else:
            pct_str = "N/A"

        txt = f"Δ Avg Latency: {sign}{diff:.3f}s ({pct_str})"

        ax.text(
            0.5, 0.8,                   # upper center ABOVE the plot
            txt,
            ha="center",
            va="bottom",
            transform=ax.transAxes,
            fontsize=5,
            bbox=dict(
                boxstyle="round,pad=0.30",
                fc="white",
                ec="gray",
                alpha=0.85,
            ),
        )

    except Exception as e:
        print(f"[warn] latency-diff annotation failed: {e}")
    

# ------------------------------------------------------
# FT Tokens Timeline (Row 4)
# ------------------------------------------------------
def plot_ft_tokens_timeline(
    ax, bwd_files_co, bwd_file_slora, colors,
    show_title=True, use_last_cfg=False
):
    """
    Fine-tuning progress comparison with:
      - Co-serving: merge 4 logs, align to t=0, concatenate, cumsum()
      - SLoRA+FT: single log, cumsum()
      - Shorter curve is extended flat in X-axis
      - Supports automatic minutes/seconds scaling
    """
    cfg = PLOT_CFG["ft_tokens"]

    # ----------------------------------------------------
    # Load + align 4 Co-serving logs (each shifted to t=0)
    # ----------------------------------------------------
    co_all = []
    for f in bwd_files_co:
        df = pd.read_csv(f)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        t0 = df["timestamp"].iloc[0]
        df["time_rel_s"] = (df["timestamp"] - t0).dt.total_seconds()
        co_all.append(df[["time_rel_s", "batch_tokens"]])

    co_merged = pd.concat(co_all, ignore_index=True)
    co_merged = co_merged.sort_values("time_rel_s").reset_index(drop=True)
    co_merged["total_tokens"] = co_merged["batch_tokens"].cumsum()

    co_time = co_merged["time_rel_s"]
    co_tokens = co_merged["total_tokens"]

    # ----------------------------------------------------
    # Load + align SLoRA FT log
    # ----------------------------------------------------
    sl = pd.read_csv(bwd_file_slora)
    sl["timestamp"] = pd.to_datetime(sl["timestamp"])
    t0_sl = sl["timestamp"].iloc[0]
    sl["time_rel_s"] = (sl["timestamp"] - t0_sl).dt.total_seconds()
    sl = sl[["time_rel_s", "batch_tokens"]].sort_values("time_rel_s")
    sl["total_tokens"] = sl["batch_tokens"].cumsum()

    sl_time = sl["time_rel_s"]
    sl_tokens = sl["total_tokens"]

    # ----------------------------------------------------
    # Equalize lengths (extend shorter curve flat)
    # ----------------------------------------------------
    end_co = co_time.iloc[-1]
    end_sl = sl_time.iloc[-1]
    max_end_s = max(end_co, end_sl)

    if end_co < max_end_s:
        co_time = pd.concat([co_time, pd.Series([max_end_s])], ignore_index=True)
        co_tokens = pd.concat([co_tokens, pd.Series([co_tokens.iloc[-1]])], ignore_index=True)
    if end_sl < max_end_s:
        sl_time = pd.concat([sl_time, pd.Series([max_end_s])], ignore_index=True)
        sl_tokens = pd.concat([sl_tokens, pd.Series([sl_tokens.iloc[-1]])], ignore_index=True)

    # ----------------------------------------------------
    # Decide whether to display in minutes or seconds
    # ----------------------------------------------------
    use_minutes = max_end_s > 1000     # consistent threshold used in timeline plots

    if use_minutes:
        x_co = co_time / 60.0
        x_sl = sl_time / 60.0
        x_label = "Time (min)"
        x_lim = [lim for lim in (cfg["x_lim_last"] if use_last_cfg else cfg["x_lim"])]
        xticks = (
            [xt for xt in cfg["xticks_last"]] if use_last_cfg else
            [xt for xt in cfg["xticks"]]
        )
    else:
        x_co = co_time
        x_sl = sl_time
        x_label = "Time (s)"
        x_lim = cfg["x_lim_last"] if use_last_cfg else cfg["x_lim"]
        xticks = cfg["xticks_last"] if use_last_cfg else cfg["xticks"]

    # ----------------------------------------------------
    # Plotting
    # ----------------------------------------------------
    if show_title:
        ax.set_title("Fine-tuning Progress", fontsize=8, pad=3, x=compute_title_x())

    ax.plot(x_co, co_tokens, color=colors[0], label="Co-serving FT Progress")
    ax.plot(x_sl, sl_tokens, color=colors[1], label="Baseline FT Progress")

    ax.grid(True, alpha=0.3)
    ax.set_xlabel(x_label, labelpad=2)
    ax.set_ylabel("Fine-tuned Tokens")
    ax.tick_params(axis="x", pad=PLOT_CFG["x_tick_pad"])
    ax.tick_params(axis="y", pad=PLOT_CFG["y_tick_pad"])
    ax.set_xlim(x_lim)
    ax.set_xticks(xticks)

    # Y-range selection
    if use_last_cfg:
        ax.set_ylim(cfg["left_ylim"])
        ax.set_yticks(cfg["left_ytick"])
    else:
        ax.set_ylim(cfg["left_ylim_small"])
        ax.set_yticks(cfg["left_ytick_small"])

    # ----------------------------------------------------
    # Annotation: FT token increase
    # ----------------------------------------------------
    try:
        total_co = float(co_tokens.iloc[-1])
        total_sl = float(sl_tokens.iloc[-1])

        diff = total_co - total_sl
        pct = (diff / total_sl * 100) if total_sl > 0 else float("nan")

        txt = f"Δ FT tokens: {pct:+.1f}%"

        ax.text(
            0.03, 0.95, txt,
            transform=ax.transAxes,
            fontsize=5,
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.35",
                      fc="white", ec="gray", alpha=0.85),
            family="monospace",
        )
    except Exception as e:
        print("[warn] FT tokens annotation failed:", e)

# ------------------------------------------------------
# GPU Utilization Multi-GPU Average (Row 5)
# ------------------------------------------------------
def plot_gpu_usage_panel_multi(
    ax,
    co_files,
    slora_files,
    colors,
    threshold=50,
    smooth_window=1,
    show_title=True,
    use_last_cfg=False,
    index =0,
):
    cfg = PLOT_CFG["gpu"]

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

    co_dfs = load_group(co_files)
    sl_dfs = load_group(slora_files)

    co_starts = [find_start(df) for df in co_dfs]
    sl_starts = [find_start(df) for df in sl_dfs]

    def align_dfs(dfs, starts):
        aligned = []
        for df, t0 in zip(dfs, starts):
            df2 = df[df["time_rel_s"] >= t0].copy()
            df2["time_aligned"] = df2["time_rel_s"] - t0
            aligned.append(df2)
        return aligned

    co_aligned = align_dfs(co_dfs, co_starts)
    sl_aligned = align_dfs(sl_dfs, sl_starts)

    def find_last_active(df, active_threshold):
        util = df["gpu_util"].to_numpy()
        idx = np.where(util > active_threshold)[0]
        if len(idx) == 0:
            return None
        return df["time_aligned"].iloc[idx[-1]]

    co_last = [find_last_active(df, threshold) for df in co_aligned]
    sl_last = [find_last_active(df, threshold) for df in sl_aligned]
    co_last = [x for x in co_last if x is not None]
    sl_last = [x for x in sl_last if x is not None]

    if len(co_last) == 0 or len(sl_last) == 0:
        global_end = 0
    else:
        global_end = min(min(co_last), min(sl_last))

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

    use_minutes = global_end > 1000
    if use_minutes:
        x_vals = time_grid / 60.0
        x_label = "Time (min)"
    else:
        x_vals = time_grid
        x_label = "Time (s)"

    avg_co = float(np.mean(co_curve)) if len(co_curve) > 0 else float("nan")
    avg_sl = float(np.mean(sl_curve)) if len(sl_curve) > 0 else float("nan")

    ax.plot(x_vals, co_curve, color=colors[0], alpha=0.7)
    ax.plot(x_vals, sl_curve, color=colors[1], alpha=0.7)

    ax.set_ylim(cfg["left_ylim"])
    ax.set_yticks(cfg["left_ytick"])

    if use_last_cfg:
        ax.set_xticks(cfg["xticks_last"])
        ax.set_xlim(cfg["xticks_last"][0], cfg["xticks_last"][-1])
    else:
        ax.set_xticks(cfg["xticks"])

    x_min, x_max = ax.get_xlim()
    if use_last_cfg:
        pad = 0.03 * (x_max - x_min)
    elif index == 0:
        pad = 0.1 * (x_max - x_min)
    else: 
        pad = 0.07 * (x_max - x_min)
    ax.set_xlim(x_min, x_max + pad)

    ax.axhline(avg_co, color=colors[0], linestyle="--", linewidth=1.2,
               label=f"Co-serving (4 GPUs avg) avg={avg_co:.1f}%")
    ax.axhline(avg_sl, color=colors[1], linestyle="--", linewidth=1.2,
               label=f"SLoRA+FT (4 GPUs avg) avg={avg_sl:.1f}%")

    # annotate averages on the right
    sorted_pairs = sorted(
        [(avg_co, colors[0]), (avg_sl, colors[1])],
        key=lambda x: x[0],
        reverse=True
    )
    high_avg, high_color = sorted_pairs[0]
    low_avg, low_color = sorted_pairs[1]

    if use_last_cfg:
        x_pad = x_max 
    elif index == 0:
        x_pad = x_max - pad * 0.6
    else:
        x_pad = x_max - pad * 0.2

    ax.text(
        x_pad,
        high_avg + 0.2,
        f"{high_avg:.1f}%",
        fontsize=5.5,
        color=high_color,
        ha="left",
        va="bottom",
        clip_on=False,
    )
    ax.text(
        x_pad,
        low_avg - 1.0,
        f"{low_avg:.1f}%",
        fontsize=5.5,
        color=low_color,
        ha="left",
        va="top",
        clip_on=False,
    )

    if len([avg_co, avg_sl]) >= 2:
        diff = avg_co - avg_sl
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
        ax.set_title("GPU Utilization (4-GPU Average)", fontsize=8, pad=2, x=compute_title_x())
    ax.set_ylabel("GPU Utilization (%)")
    ax.set_xlabel(x_label, labelpad=2)
    ax.tick_params(axis="x", pad=PLOT_CFG["x_tick_pad"])
    ax.tick_params(axis="y", pad=PLOT_CFG["y_tick_pad"])
    ax.grid(True, alpha=0.3)

def plot_multi_experiment_summary_5x3(base_dirs):
    """
    Clean layout using:
      - Outer GridSpec with 4 columns (col2 = spacer)
      - 3 experiment columns: col0, col1, col3
      - Guaranteed gap between col1 and col3
      - Scientific y-axis for FT progress
    """
    set_paper_style()

    assert len(base_dirs) == 3, "Must provide exactly 3 experiment directories."

    # ----------------------------------------------------
    # CREATE OUTER GRID WITH A SPACER COLUMN
    # ----------------------------------------------------
    fig = plt.figure(figsize=(7, 5.85))

    outer = GridSpec(
        1, 4,              # 4 columns: exp1, exp2, spacer, exp3
        figure=fig,
        width_ratios=[1, 1, 0.02, 1.4],    # 0.35 spacer creates clear gap
        wspace=0.05,
        hspace=0.0,
    )

    # Inside each experiment column we create a 5×1 subgrid
    gs0 = GridSpecFromSubplotSpec(5, 1, subplot_spec=outer[0], hspace=0.55)
    gs1 = GridSpecFromSubplotSpec(5, 1, subplot_spec=outer[1], hspace=0.55)
    gs2 = GridSpecFromSubplotSpec(5, 1, subplot_spec=outer[3], hspace=0.55)
    # Note: outer[2] is a pure spacer — nothing goes there.

    subgrids = [gs0, gs1, gs2]

    shared_axes = [None] * 5

    labels = ["Co", "Baseline"]
    colors = ["#FF7F0E", "#1F77B4"]

    LEFT_TIMELINE_COLOR = "#5E4FA2"
    RIGHT_TIMELINE_COLOR = "#000000"
    RIGHT_TIMELINE_ALPHA = 0.2

    # ----------------------------------------------------
    # Iterate over each experiment
    # ----------------------------------------------------
    for col, base in enumerate(base_dirs):
        g = subgrids[col]      # pick subgrid for this experiment
        show_title = (col == 1)
        use_last = (col == 2)

        # Paths
        lat_files = [
            os.path.join(base, "latency_co-serving.csv"),
            os.path.join(base, "latency_slora.csv"),
        ]
        timeline_csv = os.path.join(base, "timeline.csv")

        co_gpu_files = [
            os.path.join(base, f"gpu_usage_co-serving_{i}.csv") for i in range(4)
        ]
        slora_gpu_files = [
            os.path.join(base, "gpu_usage_ft.csv"),
            os.path.join(base, "gpu_usage_slora_0.csv"),
            os.path.join(base, "gpu_usage_slora_1.csv"),
            os.path.join(base, "gpu_usage_slora_2.csv"),
        ]

        bwd_logs_co = [
            os.path.join(base, f"bwd_log_{i}.csv") for i in [1,2,3,4]
        ]
        bwd_log_slora = os.path.join(base, "bwd_log_0.csv")

        # ============================
        # ROW 1: TIMELINE
        # ============================
        ax1 = fig.add_subplot(g[0], sharey=shared_axes[0])
        ax1_left, ax1_right, _ = plot_timeline(
            ax1,
            timeline_csv,
            left_color=LEFT_TIMELINE_COLOR,
            right_color=RIGHT_TIMELINE_COLOR,
            right_alpha=RIGHT_TIMELINE_ALPHA,
            show_title=show_title,
            use_last_cfg=use_last,
        )
        ax1_left.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        if shared_axes[0] is None:
            shared_axes[0] = ax1
            ax1.set_ylabel("Total tokens/sec")
        else:
            ax1.set_ylabel("")
            ax1.tick_params(labelleft=False)

        # Hide right axis for first two columns
        if col != 2:
            ax1_right.set_ylabel("")
            ax1_right.tick_params(labelright=False, right=False)

        # ============================
        # ROW 2: TTFT / Avg TBT
        # ============================
        ax2 = fig.add_subplot(g[1], sharey=shared_axes[1])
        show_slo_label = (col == 2)
        plot_ttft_tbt(
            ax2,
            lat_files,
            labels,
            colors,
            show_slo_label=show_slo_label,
            show_title=show_title,
            use_last_cfg=use_last,
        )

        if shared_axes[1] is None:
            shared_axes[1] = ax2
            ax2.set_ylabel("Seconds")
        else:
            ax2.set_ylabel("")
            ax2.tick_params(labelleft=False)

        # ============================
        # ROW 3: Latency timeline
        # ============================
        ax3 = fig.add_subplot(g[2], sharey=shared_axes[2])
        plot_request_latency_timeline(
            ax3,
            lat_files,
            labels,
            colors,
            show_title=show_title,
            use_last_cfg=use_last,
        )

        if shared_axes[2] is None:
            shared_axes[2] = ax3
            ax3.set_ylabel("Latency (s)")
        else:
            ax3.set_ylabel("")
            ax3.tick_params(labelleft=False)

        # ============================
        # ROW 4: FT Progress
        # ============================
        if col == 2:
            ax4 = fig.add_subplot(g[3])   # no sharing
        else:
            ax4 = fig.add_subplot(g[3], sharey=shared_axes[3])

        plot_ft_tokens_timeline(
            ax4,
            bwd_logs_co,
            bwd_log_slora,
            colors,
            show_title=show_title,
            use_last_cfg=use_last,
        )

        # Scientific notation for ALL FT axes
        ax4.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        if col == 2:
            ax4.set_ylabel("")            # no label
            ax4.tick_params(labelleft=True)
        else:
            if shared_axes[3] is None:
                shared_axes[3] = ax4
                ax4.set_ylabel("Total Fine-tuned Tokens")
            else:
                ax4.set_ylabel("")
                ax4.tick_params(labelleft=False)

        # ============================
        # ROW 5: GPU Utilization
        # ============================
        ax5 = fig.add_subplot(g[4], sharey=shared_axes[4])
        plot_gpu_usage_panel_multi(
            ax5,
            co_files=co_gpu_files,
            slora_files=slora_gpu_files,
            colors=colors,
            show_title=show_title,
            use_last_cfg=use_last,
            index =col,
        )

        if shared_axes[4] is None:
            shared_axes[4] = ax5
            ax5.set_ylabel("GPU Utilization (%)")
        else:
            ax5.set_ylabel("")
            ax5.tick_params(labelleft=False)

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
        bbox_to_anchor=(0.5, 1.018),
        ncol=3, frameon=False, fontsize=7,
    )

    row2_handles = [
        plt.Line2D([0],[0],color=colors[0],lw=1.2),
        plt.Line2D([0],[0],color=colors[1],lw=1.2),
        plt.Line2D([0],[0],color=colors[0],linestyle="--",lw=1.2),
        plt.Line2D([0],[0],color=colors[1],linestyle="--",lw=1.2),
    ]
    row2_labels = ["DeltaServe","Baseline","DeltaServe Avg GPU","Baseline Avg GPU"]

    fig.legend(
        row2_handles, row2_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4, frameon=False, fontsize=7,
    )

    # ----------------------------------------------------
    # Final save
    # ----------------------------------------------------
    plt.subplots_adjust(
        top=0.93, bottom=0.06,
        left=0.08, right=0.98,
    )

    out_path = os.path.abspath("comparison_summary_multi.pdf")
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved multi-experiment figure → {out_path}")

if __name__ == "__main__":
    # Example: replace with your actual result directories
    dirs = ["loose", "tight", "nutanix"]
    plot_multi_experiment_summary_5x3(dirs)