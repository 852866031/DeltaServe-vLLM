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


def _safe_label(label: str) -> str:
    return label.replace("_", " ")


# ------------------ Plot helper ------------------ #
def _add_stats_box(ax, df_list, labels, col):
    """Add a statistics box comparing multiple datasets."""
    if len(df_list) < 2:
        return

    stats_str = f"{col.upper()} Comparison\n"
    for i, (df, label) in enumerate(zip(df_list, labels)):
        s = _safe_series(df, col)
        if s.empty:
            continue
        stats_str += f"{_safe_label(label)}: mean={s.mean():.3f}s  p90={np.percentile(s,90):.3f}s  p99={np.percentile(s,99):.3f}s\n"

    ax.text(
        0.98, 0.02, stats_str.strip(),
        transform=ax.transAxes,
        fontsize=9,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8),
        family="monospace"
    )


def _plot_percentile(ax, df, col, label, color, linestyle="-"):
    pct = np.linspace(0, 100, 101)
    y = _percentile_curve(_safe_series(df, col))
    if y.size:
        ax.plot(pct, y, color=color, linestyle=linestyle, label=f"{col} ({_safe_label(label)})")
    return y


# ------------------ Plot Function ------------------ #
def plot_three_tbt_files(
    file1: str,
    label1: str,
    file2: str,
    label2: str,
    file3: str,
    label3: str,
    slo_json: Optional[str] = None,
    out_png: str = "compare3_ttft_tbt.png",
    title_suffix: str = "",
):
    load_slos(slo_json)

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)
    df_list = [df1[df1.get("status", "ok") == "ok"], df2[df2.get("status", "ok") == "ok"], df3[df3.get("status", "ok") == "ok"]]
    labels = [label1, label2, label3]

    colors = ["tab:blue", "tab:orange", "tab:green"]

    # ----- Plot 1: TTFT + Avg TBT ----- #
    fig, ax = plt.subplots(figsize=(10, 6))
    pct = np.linspace(0, 100, 101)
    all_vals = []

    for df, label, color in zip(df_list, labels, colors):
        y_ttft = _percentile_curve(_safe_series(df, "ttft_s"))
        y_avg = _percentile_curve(_safe_series(df, "avg_tbt_s"))
        if y_ttft.size:
            ax.plot(pct, y_ttft, color=color, linestyle="-", label=f"TTFT ({_safe_label(label)})")
            all_vals.append(y_ttft)
        if y_avg.size:
            ax.plot(pct, y_avg, color=color, linestyle="--", label=f"Avg TBT ({_safe_label(label)})")
            all_vals.append(y_avg)

    all_vals = np.concatenate(all_vals) if all_vals else np.array([])
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size:
        ymax_cap = np.percentile(all_vals, 99) 
        
        hard_max = all_vals.max()
        ymax_cap = (hard_max + ymax_cap) / 2
        ax.set_ylim(top=ymax_cap)
        if hard_max > ymax_cap:
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
    ax.set_title(f"TTFT / Avg TBT — {_safe_label(label1)}, {_safe_label(label2)}, {_safe_label(label3)}" + title_suffix)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", ncol=2)

    _add_stats_box(ax, df_list, labels, "ttft_s")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"✅ Saved TTFT/TBT comparison plot to {out_png}")

    # ----- Plot 2: Total latency (latency_s) ----- #
    fig, ax = plt.subplots(figsize=(10, 6))
    all_vals = []
    for df, label, color in zip(df_list, labels, colors):
        y_lat = _percentile_curve(_safe_series(df, "latency_s"))
        if y_lat.size:
            ax.plot(pct, y_lat, color=color, linestyle="-", label=f"Latency ({_safe_label(label)})")
            all_vals.append(y_lat)

    all_vals = np.concatenate(all_vals) if all_vals else np.array([])
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size:
        ymax_cap = np.percentile(all_vals, 99) * 1.1
        hard_max = all_vals.max()
        ax.set_ylim(top=ymax_cap)
        if hard_max > ymax_cap:
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
    ax.set_title(f"Total Latency — {_safe_label(label1)}, {_safe_label(label2)}, {_safe_label(label3)}" + title_suffix)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    _add_stats_box(ax, df_list, labels, "latency_s")

    fig.tight_layout()
    out2 = out_png.replace(".png", "_total_latency.png")
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"✅ Saved total latency comparison to {out2}")


# ------------------ Main ------------------ #
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


if __name__ == "__main__":
    rps, duration, prompt_len, max_new_tokens = extract_request_params("timelines/timeline_live.csv")
    suffix = f"{rps}rps-{duration}s-{prompt_len}in-{max_new_tokens}gen"
    plot_three_tbt_files(
        file1="results/latency_inference.csv",
        label1="Inference",
        file2="results/latency_co-serving.csv",
        label2="Co-serving",
        file3="results/latency_slora.csv",
        label3="SLoRA",
        slo_json=None,
        out_png=f"graphs/cp3-{suffix}.png",
        title_suffix=f" @ {suffix}"
    )