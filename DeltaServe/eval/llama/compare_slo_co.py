import json
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------ Utilities ------------------ #
def _percentile_curve(series: pd.Series) -> np.ndarray:
    vals = series.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.array([])
    pct = np.linspace(0, 100, 101)
    return np.percentile(vals, pct)


def _safe_series(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name].dropna() if name in df.columns else pd.Series([], dtype=float)


def load_slos(json_path: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    if not json_path:
        return None, None
    try:
        with open(json_path, "r") as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"⚠️  Failed to read SLO json '{json_path}': {e}")
        return None, None

    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return None

    return _to_float(cfg.get("ttft_slo")), _to_float(cfg.get("avg_tbt_slo"))


# ------------------ Plot helper ------------------ #
def _plot_percentile_comparison(ax, df1, df2, col, label1, label2, color):
    pct = np.linspace(0, 100, 101)
    y1 = _percentile_curve(_safe_series(df1, col))
    y2 = _percentile_curve(_safe_series(df2, col))
    if y1.size:
        ax.plot(pct, y1, color=color, linestyle="-", label=f"{col} ({label1})")
    if y2.size:
        ax.plot(pct, y2, color=color, linestyle="--", label=f"{col} ({label2})")
    return y1, y2


def _add_stats_box(ax, df1, df2, label1, label2, col):
    """Add a summary statistics box and auto-place it to minimize overlap."""
    s1 = _safe_series(df1, col)
    s2 = _safe_series(df2, col)
    if s1.empty or s2.empty:
        return

    def stats(s):
        return dict(mean=s.mean(), p90=np.percentile(s, 90), p99=np.percentile(s, 99))

    st1, st2 = stats(s1), stats(s2)

    def rel(a, b):
        return 100.0 * (a - b) / b if b > 0 else np.nan

    box_text = (
        f"{label1} vs {label2} ({col})\n"
        f"Mean:  {st1['mean']:.3f}s / {st2['mean']:.3f}s  (Δ{rel(st2['mean'], st1['mean']):.1f}%)\n"
        f"P90:   {st1['p90']:.3f}s / {st2['p90']:.3f}s  (Δ{rel(st2['p90'], st1['p90']):.1f}%)\n"
        f"P99:   {st1['p99']:.3f}s / {st2['p99']:.3f}s  (Δ{rel(st2['p99'], st1['p99']):.1f}%)"
    )

    # Determine data bounds
    y_min, y_max = ax.get_ylim()
    all_y = []
    for line in ax.lines:
        all_y.extend(line.get_ydata())
    all_y = np.array(all_y)
    all_y = all_y[np.isfinite(all_y)]
    data_bottom = np.percentile(all_y, 5) if all_y.size else y_min
    data_top = np.percentile(all_y, 95) if all_y.size else y_max

    # Adaptive placement:
    # if most data is near bottom (flat latency), push the box upward
    # if data rises high, keep at bottom-right; otherwise, switch to bottom-left if right side crowded
    loc_y = 0.02
    loc_x = 0.98
    va, ha = "bottom", "right"

    if data_bottom < (y_min + 0.1 * (y_max - y_min)):  # data close to axis bottom
        loc_y = 0.13
    # detect if right side of x-axis has denser lines (we’ll roughly infer using percentiles)
    # if yes, flip to left
    if np.random.rand() < 0.0:  # placeholder logic, can be refined further
        loc_x, ha = 0.02, "left"

    ax.text(
        loc_x, loc_y, box_text,
        transform=ax.transAxes,
        fontsize=9,
        ha=ha, va=va,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8),
        family="monospace"
    )


# ------------------ Plot Function ------------------ #
def plot_two_tbt_files(
    file1: str,
    label1: str,
    file2: str,
    label2: str,
    slo_json: Optional[str] = None,
    out_png: str = "compare_ttft_tbt.png",
    title_suffix: str = "",
):
    load_slos(slo_json)

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df1_ok = df1[df1.get("status", "ok") == "ok"].copy()
    df2_ok = df2[df2.get("status", "ok") == "ok"].copy()

    # ----- Plot 1: TTFT + Avg TBT ----- #
    fig, ax = plt.subplots(figsize=(10, 6))

    pct = np.linspace(0, 100, 101)
    y1_ttft = _percentile_curve(_safe_series(df1_ok, "ttft_s"))
    y2_ttft = _percentile_curve(_safe_series(df2_ok, "ttft_s"))
    y1_avg = _percentile_curve(_safe_series(df1_ok, "avg_tbt_s"))
    y2_avg = _percentile_curve(_safe_series(df2_ok, "avg_tbt_s"))

    if y1_ttft.size:
        ax.plot(pct, y1_ttft, color="tab:blue", linestyle="-", label=f"TTFT ({label1})")
    if y2_ttft.size:
        ax.plot(pct, y2_ttft, color="tab:blue", linestyle="--", label=f"TTFT ({label2})")
    if y1_avg.size:
        ax.plot(pct, y1_avg, color="tab:orange", linestyle="-", label=f"Avg TBT ({label1})")
    if y2_avg.size:
        ax.plot(pct, y2_avg, color="tab:orange", linestyle="--", label=f"Avg TBT ({label2})")

    # Dynamic y-cap
    all_vals = np.concatenate([y1_ttft, y2_ttft, y1_avg, y2_avg])
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size:
        ymax_cap = np.percentile(all_vals, 99) * 1.1
        hard_max = all_vals.max()
        if hard_max > ymax_cap:
            ax.set_ylim(top=ymax_cap)
            ax.annotate(
                f"up to {hard_max:.3f}s",
                xy=(0.98, ymax_cap),
                xycoords=("axes fraction", "data"),
                ha="right", va="bottom",
                fontsize=9, color="gray",
                bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.7),
            )

    for p in (50, 90, 99):
        ax.axvline(p, linestyle=":", color="gray", linewidth=1, alpha=0.4)

    ax.set_xlabel("Percentile")
    ax.set_ylabel("Seconds")
    ax.set_title(f"TTFT / Avg TBT — {label1} vs {label2}" + title_suffix)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", ncol=2)

    _add_stats_box(ax, df1_ok, df2_ok, label1, label2, "ttft_s")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"✅ Saved TTFT/TBT comparison plot to {out_png}")

    # ----- Plot 2: Total latency (latency_s) ----- #
    fig, ax = plt.subplots(figsize=(10, 6))
    y1_lat, y2_lat = _plot_percentile_comparison(ax, df1_ok, df2_ok, "latency_s", label1, label2, "tab:green")

    all_vals = np.concatenate([y1_lat, y2_lat])
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size:
        ymax_cap = np.percentile(all_vals, 99) * 1.1
        hard_max = all_vals.max()
        if hard_max > ymax_cap:
            ax.set_ylim(top=ymax_cap)
            ax.annotate(
                f"up to {hard_max:.3f}s",
                xy=(0.98, ymax_cap),
                xycoords=("axes fraction", "data"),
                ha="right", va="bottom",
                fontsize=9, color="gray",
                bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.7),
            )

    for p in (50, 90, 99):
        ax.axvline(p, linestyle=":", color="gray", linewidth=1, alpha=0.4)

    ax.set_xlabel("Percentile")
    ax.set_ylabel("Seconds")
    ax.set_ylim(bottom=0, top=ymax_cap*1.2)            # ensures y-axis starts at 0
    ax.set_title(f"Total Latency (latency_s) — {label1} vs {label2}" + title_suffix)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    _add_stats_box(ax, df1_ok, df2_ok, label1, label2, "latency_s")

    fig.tight_layout()
    out2 = out_png.replace("slo", "total_latency")
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"✅ Saved total latency comparison to {out2}")


# ------------------ Extract Parameters ------------------ #
def extract_request_params(csv_path: str) -> Tuple[int, int, int, int]:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path} is empty")
    grouped = df.groupby("second")["index_in_second"].nunique()
    rps = int(grouped.mode()[0]) if not grouped.empty else 0
    duration = int(df["second"].max()) + 1
    prompt_len = int(df["prompt_length"].iloc[0])
    max_new_tokens = int(df["max_new_tokens"].iloc[0])
    return rps, duration, prompt_len, max_new_tokens


# ------------------ Main ------------------ #
if __name__ == "__main__":
    rps, duration, prompt_len, max_new_tokens = extract_request_params("timelines/timeline_live.csv")
    suffix = f"{rps}rps-{duration}s-{prompt_len}in-{max_new_tokens}gen"
    plot_two_tbt_files(
        file1="results/latency_inference.csv",
        label1="inf",
        file2="results/latency_co-serving.csv",
        label2="co_serving",
        slo_json=None,
        out_png=f"co-slo-{suffix}.png",
        title_suffix=f" @ {suffix}"
    )