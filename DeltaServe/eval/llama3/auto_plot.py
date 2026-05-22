"""auto_plot.py — single-trace 4-panel plots for loose, tight, and nutanix runs.

For each workload-shape variant the auto_benchmark.py emits, this script
builds one 4-subplot PNG showing the trace for that run.

Subplots (left → right):

  1. Scheduled request timeline (from timeline_<mode>.csv)
  2. Per-request E2E latency vs time
  3. Throughput (tokens/s) over time — two curves: inference-only and
     total (inference + finetune); the gap between them is shaded and
     equals the FT contribution at each instant.
  4. TTFT SLO satisfaction rate over time — rolling-window % of requests
     whose TTFT met the SLO threshold, with a 95% reference line.

Defaults assume the all-graphs-on co-serving configuration:
  --co --decode_graph --prefill_graph --bwd_graph

which produces output suffix `_decode_prefill_bwd`. Override via `--suffix`
if you ran a different combination. (Alpaca + packed_kv are bundled into
the default serving_config_finetuning.yaml, so they no longer get their
own suffix tags.)

The TTFT SLO threshold (subplot 4) is read from
serving_config_finetuning.yaml. Override with `--slo`.

Inputs (under eval/llama3/output/, suffix scheme from auto_benchmark.py):
  timeline_results<base>_<mode>.csv       per-request results
  bwd_log<base>_<mode>.csv                finetune backward log
  timeline_<mode>.csv                     input timeline (under
                                          eval/llama3/timelines/<gpu>/)

Run upstream first, e.g.:
  python auto_benchmark.py --co --graphs --loose
  python auto_benchmark.py --co --graphs --tight
  python auto_benchmark.py --co --graphs --nutanix

Then:
  python auto_plot.py
"""
import argparse
import os
import subprocess
import sys
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

# The plot helper modules now live under scripts/ — extend sys.path so
# the original sibling-style imports keep working.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "scripts"))

from bwd_graph_plot import (
    ensure_exists,
    load_results,
    parse_bwd_log_csv,
)
from compare_graphs_plot import load_timeline, plot_request_timeline


OUTPUT_DIR = os.path.join(_HERE, "output")
PLOTS_DIR = os.path.join(_HERE, "plots")
CONFIG_DIR = os.path.join(_HERE, "config")
DEFAULT_SLO_FALLBACK = 0.35


def _detect_gpu_subdir() -> str:
    """Return the timelines/ subdirectory name matching the local GPU.
    Greps `nvidia-smi`; falls back to '5090' on failure."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True, timeout=2.0,
        )
        name = (out.strip().splitlines() or [""])[0].upper()
        if "A100" in name:
            return "A100"
        if "5090" in name:
            return "5090"
    except Exception:
        pass
    return "5090"


_DEFAULT_GPU_SUBDIR = _detect_gpu_subdir()
TIMELINES_DIR = os.path.join(_HERE, "timelines", _DEFAULT_GPU_SUBDIR)


def _config_yaml_for_suffix(suffix: str) -> str:
    """Return the server YAML for the SLO lookup. Only one finetuning
    YAML exists now (alpaca + packed_kv defaults are bundled in)."""
    return os.path.join(CONFIG_DIR, "serving_config_finetuning.yaml")


def _read_ttft_slo_from_yaml(yaml_path: str):
    """Return float(slo.ttft_slo) from `yaml_path`, or None on any failure."""
    try:
        import yaml  # PyYAML — already a project dep (used by dserve/server/config.py)
        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}
        v = data.get("slo", {}).get("ttft_slo")
        return float(v) if v is not None else None
    except Exception:
        return None

ALL_MODES = ("loose", "tight", "nutanix")

MODE_COLORS = {
    "loose":   "tab:blue",
    "tight":   "tab:red",
    "nutanix": "tab:green",
}

# Color used to shade the FT contribution in the throughput stack —
# kept stable across modes so the eye reads the FT band consistently.
FT_SHADE_COLOR = "tab:orange"


# ---------------- Token-distribution helpers ----------------
def _distribute_to_bins(t_starts, t_ends, weights, t_max, bin_s):
    """For each (t_start, t_end, weight), spread `weight` uniformly across
    1 s bins covering [t_start, t_end]. Returns a per-bin array."""
    n_bins = int(np.ceil(t_max / bin_s)) + 1
    bins = np.zeros(n_bins, dtype=float)
    for ts, te, w in zip(t_starts, t_ends, weights):
        if te <= ts:
            # Degenerate (no decode duration recorded) — drop weight in start bin.
            b = int(ts // bin_s)
            if 0 <= b < n_bins:
                bins[b] += float(w)
            continue
        rate = float(w) / (te - ts)
        b_start = max(0, int(np.floor(ts / bin_s)))
        b_end = min(n_bins, int(np.ceil(te / bin_s)))
        for b in range(b_start, b_end):
            lo = max(ts, b * bin_s)
            hi = min(te, (b + 1) * bin_s)
            if hi > lo:
                bins[b] += rate * (hi - lo)
    return bins


def _ft_per_bin(rel_t, cum_tok, t_max, bin_s):
    """Convert (timestamp, cumulative-tokens) sequence into per-bin token
    deltas via linear interpolation at bin edges."""
    n_bins = int(np.ceil(t_max / bin_s)) + 1
    if len(rel_t) == 0 or len(cum_tok) == 0:
        return np.zeros(n_bins)
    edges = np.arange(n_bins + 1) * bin_s
    cum_at_edges = np.interp(
        edges, rel_t, cum_tok, left=0.0, right=float(cum_tok[-1])
    )
    deltas = np.diff(cum_at_edges)
    deltas[deltas < 0] = 0.0  # cumulative shouldn't decrease; clip noise
    return deltas


def _smooth(y: np.ndarray, window_s: float, bin_s: float) -> np.ndarray:
    """Centered rolling mean over `window_s` seconds. Used to flatten the
    per-second spikes that come from chunky backward token release so the
    burst-shape signal stays readable on long workloads."""
    if y.size == 0 or window_s <= 0 or bin_s <= 0:
        return y
    w = max(1, int(round(window_s / bin_s)))
    if w <= 1 or y.size < w:
        return y
    kernel = np.ones(w, dtype=float) / w
    return np.convolve(y, kernel, mode="same")


def _auto_throughput_window_s(t_max: float) -> float:
    """Pick a smoothing window proportional to workload duration.
    Targets ~100 effective points across the chart: short runs get a
    narrow window so brief bursts stay visible, long runs get a wider
    window so per-second spikes don't dominate. Clamped to [5, 60] s."""
    if t_max <= 0:
        return 5.0
    return float(np.clip(t_max / 100.0, 5.0, 60.0))


# ---------------- Single-trace plot helpers ----------------
def plot_latency_vs_time_single(ax, results_df, label: str, color: str):
    ok = results_df[results_df["ok"]]
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


def plot_throughput_curves(
    ax, results_df, timeline_df, bwd_log,
    color_inf: str, bin_s: float = 1.0,
    smoothing_window_s: Optional[float] = None,
):
    """Two-curve throughput view: inference-only tokens/s and total
    (inference + finetune) tokens/s. The area BETWEEN the curves — which
    is the FT contribution at each instant — is shaded.

    Both series are on the request-timeline time origin; the FT origin is
    the bwd-log's first event, so very early seconds may be slightly
    misaligned by the FT-trigger latency (a few s).

    `smoothing_window_s` controls the rolling-mean window used to flatten
    per-second spikes (chunky backward token release). None → auto-pick
    based on workload duration (`_auto_throughput_window_s`). Raw per-bin
    values are still drawn as a faint background so true outliers stay
    visible.
    """
    # Join max_new_tokens onto results via row_id (timeline df.index == row_id).
    tl = timeline_df.copy()
    tl["row_id"] = tl.index
    merged = results_df.merge(
        tl[["row_id", "max_new_tokens"]],
        left_on="idx", right_on="row_id", how="left",
    )
    ok = merged[
        merged["ok"]
        & merged["ttft_s"].notna()
        & merged["latency_s"].notna()
        & merged["max_new_tokens"].notna()
    ]

    if len(ok) > 0:
        inf_t_start = (ok["t_rel_s"].astype(float) + ok["ttft_s"].astype(float)).to_numpy()
        inf_t_end   = (ok["t_rel_s"].astype(float) + ok["latency_s"].astype(float)).to_numpy()
        inf_tokens  = ok["max_new_tokens"].astype(float).to_numpy()
    else:
        inf_t_start = inf_t_end = inf_tokens = np.array([], dtype=float)

    rel_t, cum_tok, _ = bwd_log

    t_max = max(
        float(inf_t_end.max()) if len(inf_t_end) else 0.0,
        float(rel_t[-1]) if len(rel_t) else 0.0,
    )
    if t_max <= 0:
        ax.set_title("Throughput (no data)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Tokens / s")
        return

    inf_per_s = _distribute_to_bins(inf_t_start, inf_t_end, inf_tokens, t_max, bin_s) / bin_s
    ft_per_s  = _ft_per_bin(rel_t, cum_tok, t_max, bin_s) / bin_s

    # Pad both to a common length so the fill_between aligns.
    n = max(len(inf_per_s), len(ft_per_s))
    inf_per_s = np.pad(inf_per_s, (0, n - len(inf_per_s)))
    ft_per_s  = np.pad(ft_per_s,  (0, n - len(ft_per_s)))
    centers = (np.arange(n) + 0.5) * bin_s
    total = inf_per_s + ft_per_s

    # Averages from the *raw* per-bin signal — that's the right number
    # for the legend, smoothing must not change reported totals.
    inf_avg, ft_avg, tot_avg = inf_per_s.mean(), ft_per_s.mean(), total.mean()

    # Smoothing: auto-pick window if not provided. Apply to all three
    # series the same way so the shaded gap stays consistent.
    win_s = smoothing_window_s if smoothing_window_s is not None \
        else _auto_throughput_window_s(t_max)
    inf_smooth = _smooth(inf_per_s, win_s, bin_s)
    total_smooth = _smooth(total, win_s, bin_s)

    # Raw per-bin "Total" as a very faint background so true peaks are
    # still visible if they matter for debugging. Skip when smoothing is
    # effectively off so we don't double-draw the same line.
    if win_s > bin_s:
        ax.plot(
            centers, total, color="black", linewidth=0.4, alpha=0.20,
            zorder=1,
        )

    # Stacked shading: bottom band is the inference contribution (from
    # 0 up to the smoothed inference curve), top band is the FT
    # contribution (gap from inference to total). Together they fill the
    # area under the total curve, so the visual stack matches the
    # tok/s budget at every instant.
    ax.fill_between(
        centers, 0, inf_smooth,
        color=color_inf, alpha=0.20, linewidth=0,
        label=f"Inference contribution (avg {inf_avg:.0f} tok/s)",
        zorder=2,
    )
    ax.fill_between(
        centers, inf_smooth, total_smooth,
        color=FT_SHADE_COLOR, alpha=0.30, hatch="//", linewidth=0,
        label=f"Finetune contribution (avg {ft_avg:.0f} tok/s)",
        zorder=2,
    )
    # Two curves on top, plotted from the smoothed series. Inference
    # curve outlines the boundary between the two shaded bands; total
    # curve outlines the top.
    ax.plot(
        centers, inf_smooth, color=color_inf, linewidth=1.6,
        label=f"Inference (avg {inf_avg:.0f} tok/s)",
        zorder=3,
    )
    ax.plot(
        centers, total_smooth, color="black", linewidth=1.6,
        label=f"Total (avg {tot_avg:.0f} tok/s)",
        zorder=3,
    )
    title = "Throughput (tokens/s)"
    if win_s > bin_s:
        title += f" — {win_s:.0f}s rolling mean"
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tokens / s")
    ax.legend(loc="best", fontsize=8)


def plot_ttft_satisfaction_rate(
    ax, results_df, slo_s: float, window_s: float = 5.0,
):
    """Rolling % of requests in a `window_s` window whose TTFT ≤ `slo_s`.
    Attribution is by request arrival time (t_rel_s)."""
    ok = results_df[results_df["ok"] & results_df["ttft_s"].notna()]
    if len(ok) == 0:
        ax.set_title(f"TTFT Satisfaction Rate (no data)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("% of requests")
        return

    t = ok["t_rel_s"].astype(float).to_numpy()
    sat = (ok["ttft_s"].astype(float) <= slo_s).to_numpy().astype(int)

    t_max = float(t.max())
    grid = np.arange(0.0, t_max + 1.0, 1.0)
    rates = np.full(len(grid), np.nan)
    half = window_s / 2.0
    for i, t0 in enumerate(grid):
        mask = (t >= t0 - half) & (t <= t0 + half)
        n = int(mask.sum())
        if n > 0:
            rates[i] = sat[mask].mean() * 100.0

    overall = sat.mean() * 100.0

    ax.plot(
        grid, rates, color="tab:purple", linewidth=1.5,
        label=f"Satisfaction ({int(window_s)}s window) — overall {overall:.1f}%",
    )
    ax.axhline(95.0, color="tab:red", linestyle="--", linewidth=1, alpha=0.6,
               label="95% target")
    ax.set_ylim(0, 105)
    ax.set_title(f"TTFT Satisfaction Rate (SLO ≤ {slo_s:.2f}s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("% of requests")
    ax.legend(loc="best", fontsize=8)


# ---------------- Per-mode figure ----------------
def make_figure_for_mode(
    mode: str,
    base_suffix: str,
    output_dir: str,
    plots_dir: str,
    timeline_csv_dir: str,
    out_path: str,
    slo_s: float,
    window_s: float,
    throughput_window_s: Optional[float] = None,
) -> None:
    """Build one 4-panel PNG for `mode` ∈ {'loose', 'tight', 'nutanix'}."""
    full_suffix = f"{base_suffix}_{mode}"
    results_csv = os.path.join(output_dir, f"timeline_results{full_suffix}.csv")
    bwd_log_csv = os.path.join(output_dir, f"bwd_log{full_suffix}.csv")
    timeline_csv = os.path.join(timeline_csv_dir, f"timeline_{mode}.csv")

    for p in (timeline_csv, results_csv, bwd_log_csv):
        ensure_exists(p)

    timeline_df = load_timeline(timeline_csv)
    results_df = load_results(results_csv)
    bwd_log = parse_bwd_log_csv(bwd_log_csv)

    color = MODE_COLORS.get(mode, "tab:gray")

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    plot_request_timeline(axes[0], timeline_df)
    plot_latency_vs_time_single(axes[1], results_df, label=mode, color=color)
    plot_throughput_curves(axes[2], results_df, timeline_df, bwd_log,
                            color_inf=color,
                            smoothing_window_s=throughput_window_s)
    plot_ttft_satisfaction_rate(axes[3], results_df, slo_s=slo_s, window_s=window_s)

    config_tag = base_suffix.lstrip("_") or "baseline"
    fig.suptitle(f"{mode}  ({config_tag})")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[auto_plot] Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--suffix",
        default="_decode_prefill_bwd",
        help="Base configuration suffix from auto_benchmark, WITHOUT the "
             "trailing _<mode> tag. Default '_decode_prefill_bwd' matches "
             "'--co --graphs'.",
    )
    ap.add_argument("--output-dir", default=OUTPUT_DIR,
                    help="Where auto_benchmark writes its CSVs.")
    ap.add_argument("--plots-dir", default=PLOTS_DIR,
                    help="Where to write the PNGs.")
    ap.add_argument("--timeline-csv-dir", default=TIMELINES_DIR,
                    help=f"Directory containing timeline_<mode>.csv inputs. "
                         f"Default: timelines/<gpu>/ auto-resolved by GPU "
                         f"(currently '{_DEFAULT_GPU_SUBDIR}').")
    ap.add_argument(
        "--mode",
        choices=ALL_MODES + ("all",),
        default="all",
        help="Which mode(s) to plot. 'all' (default) emits one PNG per mode "
             "in {loose, tight, nutanix}; modes whose input CSVs are missing "
             "are skipped with a warning.",
    )
    ap.add_argument(
        "--slo", type=float, default=None,
        help="TTFT SLO threshold (seconds) for the satisfaction-rate subplot. "
             "Default: read slo.ttft_slo from "
             "serving_config_finetuning.yaml. Falls back to "
             f"{DEFAULT_SLO_FALLBACK}s if the YAML can't be read.",
    )
    ap.add_argument(
        "--config-yaml",
        default=None,
        help="Override the YAML the SLO is read from. Default: "
             "serving_config_finetuning.yaml.",
    )
    ap.add_argument(
        "--window", type=float, default=5.0,
        help="Rolling-window size (seconds) for the satisfaction-rate "
             "subplot. Default 5s.",
    )
    ap.add_argument(
        "--throughput-window", type=float, default=None,
        help="Rolling-mean window (seconds) for the throughput subplot. "
             "Default: auto-pick proportional to workload duration "
             "(clipped to [5, 60]s) — short runs get small windows so "
             "bursts stay visible, long runs get wider windows so "
             "per-second spikes don't dominate. Set to 0 to disable "
             "smoothing entirely (shows raw per-bin values).",
    )
    args = ap.parse_args()

    # Resolve SLO: explicit --slo wins; else read from YAML; else fallback.
    slo_s = args.slo
    if slo_s is None:
        yaml_path = args.config_yaml or _config_yaml_for_suffix(args.suffix)
        slo_s = _read_ttft_slo_from_yaml(yaml_path)
        if slo_s is not None:
            print(f"[auto_plot] SLO = {slo_s:.3f}s (read from {os.path.basename(yaml_path)})")
        else:
            slo_s = DEFAULT_SLO_FALLBACK
            print(
                f"[auto_plot] SLO = {slo_s:.3f}s (fallback — couldn't read "
                f"slo.ttft_slo from {yaml_path})",
                file=sys.stderr,
            )
    else:
        print(f"[auto_plot] SLO = {slo_s:.3f}s (from --slo)")

    modes = ALL_MODES if args.mode == "all" else (args.mode,)

    skipped = []
    wrote = 0
    for mode in modes:
        config_tag = args.suffix.lstrip("_") or "baseline"
        out_path = os.path.join(args.plots_dir, f"{mode}_{config_tag}.png")
        try:
            make_figure_for_mode(
                mode=mode,
                base_suffix=args.suffix,
                output_dir=args.output_dir,
                plots_dir=args.plots_dir,
                timeline_csv_dir=args.timeline_csv_dir,
                out_path=out_path,
                slo_s=slo_s,
                window_s=args.window,
                throughput_window_s=args.throughput_window,
            )
            wrote += 1
        except FileNotFoundError as e:
            skipped.append((mode, str(e)))

    for mode, msg in skipped:
        print(f"[auto_plot] {mode}: skipped (missing input — {msg})", file=sys.stderr)

    if wrote == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
