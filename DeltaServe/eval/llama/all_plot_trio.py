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
# Figure: Latency Percentile Comparisons (N datasets)
# ------------------------------------------------------
def plot_latency_panels(
    ax_ttft,
    ax_latency,
    files,
    labels,
    colors,
    ttft_slo=None,
    baseline_index=None,
):
    """
    Plot TTFT / Avg TBT and total latency percentiles for N datasets.

    Parameters
    ----------
    ax_ttft : matplotlib.axes.Axes
        Axis for TTFT / Avg TBT curves.
    ax_latency : matplotlib.axes.Axes
        Axis for total latency curves.
    files : list[str]
        List of CSV paths with columns: ttft_s, avg_tbt_s, latency_s, status.
    labels : list[str]
        List of labels for each dataset.
    colors : list[str]
        List of colors for each dataset.
    ttft_slo : float, optional
        TTFT SLO to draw as horizontal line.
    baseline_index : int, optional
        Index in `labels` / `files` to treat as baseline for Δ comparisons.
        Defaults to the last dataset.
    """
    assert len(files) == len(labels) == len(colors), \
        "files, labels, and colors must have the same length."
    assert len(files) >= 2, "This function requires at least 2 datasets."

    if baseline_index is None:
        baseline_index = len(files) - 1  # default: last dataset as baseline

    if not (0 <= baseline_index < len(files)):
        raise ValueError(f"baseline_index {baseline_index} is out of range.")

    dfs = [pd.read_csv(f) for f in files]
    dfs = [df[df.get("status", "ok") == "ok"] for df in dfs]

    pct = np.linspace(0, 100, 101)
    all_vals = []

    # --- Plot TTFT / Avg TBT curves ---
    for df, label, color in zip(dfs, labels, colors):
        y_ttft = _percentile_curve(_safe_series(df, "ttft_s"))
        y_avg  = _percentile_curve(_safe_series(df, "avg_tbt_s"))

        if y_ttft.size:
            ax_ttft.plot(
                pct, y_ttft,
                color=color,
                linestyle="-",
                label=f"TTFT ({_safe_label(label)})",
            )
            all_vals.append(y_ttft)

        if y_avg.size:
            ax_ttft.plot(
                pct, y_avg,
                color=color,
                linestyle="--",
                label=f"Avg TBT ({_safe_label(label)})",
            )
            all_vals.append(y_avg)

    # --- Draw TTFT SLO horizontal line ---
    if ttft_slo is not None:
        ax_ttft.axhline(
            ttft_slo,
            color="black",
            linestyle=":",
            linewidth=1.2,
            alpha=0.9,
            label=f"TTFT SLO = {ttft_slo:.3f}s",
        )

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
            ax_latency.plot(
                pct, y_lat,
                color=color,
                linewidth=1.8,
                label=_safe_label(label),
            )

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

    base = baseline_index

    def delta_str(base_val, comp_val):
        if np.isnan(base_val) or np.isnan(comp_val):
            return "N/A"
        diff = comp_val - base_val
        pct = (diff / base_val * 100) if base_val > 0 else float("nan")
        sign = "+" if diff > 0 else ""
        return f"{sign}{diff:.3f}s ({sign}{pct:.1f}%)"

    def delta_str2(base_val, comp_val):
        if np.isnan(base_val) or np.isnan(comp_val):
            return "N/A"
        diff = comp_val - base_val
        sign = "+" if diff > 0 else ""
        return f"{sign}{diff:.3f}s"

    # ===========================================================
    # TTFT + Avg TBT comparison box (multi-line)
    # ===========================================================
    lines_ttft_tbt = [
        f"Δ Avg TTFT / Avg TBT (vs {_safe_label(labels[base])})"
    ]
    for i, lbl in enumerate(labels):
        if i == base:
            continue
        lines_ttft_tbt.append(
            f"{_safe_label(lbl)}: TTFT {delta_str(avg_ttft[base], avg_ttft[i])},  "
            f"TBT {delta_str2(avg_tbt[base], avg_tbt[i])}"
        )
    box_ttft_tbt = "\n".join(lines_ttft_tbt)

    ax_ttft.text(
        0.03, 0.5, box_ttft_tbt,
        transform=ax_ttft.transAxes,
        fontsize=8.5,
        ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8),
        family="monospace",
    )

    # ===========================================================
    # Total latency comparison box (multi-line)
    # ===========================================================
    lines_lat = [
        f"Δ Avg Total Latency (vs {_safe_label(labels[base])})"
    ]
    for i, lbl in enumerate(labels):
        if i == base:
            continue
        lines_lat.append(
            f"{_safe_label(lbl)}: {delta_str(avg_lat[base], avg_lat[i])}"
        )
    box_lat = "\n".join(lines_lat)

    ax_latency.text(
        0.98, 0.05, box_lat,
        transform=ax_latency.transAxes,
        fontsize=8.5,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8),
        family="monospace",
    )

# ------------------------------------------------------
# SLO extraction (unchanged, but only returns TTFT SLO)
# ------------------------------------------------------
import json
from pathlib import Path

def extract_slo(json_path: str):
    """
    Extract ttft_slo from a JSON config file.

    Returns a dict:
      {
         "ttft_slo": float,
      }
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
# Combine Panels: ONLY the middle two (TTFT/TBT + Total Latency)
# ------------------------------------------------------
def plot_all_combined(
    latency_files=None,
    labels=None,
    colors=None,
    ft_tokens=None,      # kept for API compatibility; unused now
    ttft_slo=None,
    baseline_index=None,
    out_path="results/latency_summary.png",
):
    """
    Create a 2-panel figure with:
      - Left: TTFT / Avg TBT percentiles
      - Right: Total latency percentiles

    Supports 2 or more datasets.
    """
    # Default: example 3-case setup; adjust paths to your actual files.
    rps = 2
    if latency_files is None:
        latency_files = [
            f"results/latency_co-serving_{rps}rps.csv",
            f"results/latency_inference_{rps}rps.csv",
            f"results/latency_slora_{rps}rps.csv",
        ]
    if labels is None:
        labels = ["Co-serving", "Inference", "SLoRA"]
    if colors is None:
        colors = ["tab:orange", "tab:green", "tab:blue"]

    assert len(latency_files) == len(labels) == len(colors), \
        "latency_files, labels, and colors must have the same length."

    fig, (ax_ttft, ax_latency) = plt.subplots(1, 2, figsize=(14, 5))

    plot_latency_panels(
        ax_ttft,
        ax_latency,
        latency_files,
        labels,
        colors,
        ttft_slo=ttft_slo,
        baseline_index=baseline_index,
    )

    fig.tight_layout()
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"✅ Saved 2-panel latency figure to {out_path}")

if __name__ == "__main__":
    ttft_slo =0.1

    # Adjust `latency_files`, `labels`, and `colors` here if needed
    plot_all_combined(ttft_slo=ttft_slo)