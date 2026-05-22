#!/usr/bin/env python3
"""auto_plot.py — 4-panel per-mode plots for DeltaServe-on-vLLM benchmarks.

Self-contained port of DeltaServe/eval/llama3/auto_plot.py (helpers inlined,
csv+numpy only — no pandas). For each workload-shape mode it builds one
4-subplot PNG:

  1. Scheduled request timeline (req/s bars + output tok/s line) from
     timelines/<gpu>/timeline_<mode>.csv
  2. Per-request E2E latency vs time (from timeline_results<suffix>.csv)
  3. Throughput tok/s: inference contribution + finetune contribution bands
     (FT band from bwd_log<suffix>.csv)
  4. TTFT SLO satisfaction rate vs the 95% target

Inputs (under eval/output/, suffix = '<base>_<mode>' from auto_benchmark.py):
  timeline_results<base>_<mode>.csv
  bwd_log<base>_<mode>.csv          (only present for --co runs)
  timelines/<gpu>/timeline_<mode>.csv

Run the benchmark first, e.g.:
  python eval/auto_benchmark.py --co --loose
  python eval/auto_benchmark.py --co --tight
Then:
  python eval/auto_plot.py            # default --suffix _co

Requires matplotlib (pip install matplotlib).
"""
import argparse
import csv
import datetime
import os
import subprocess
import sys
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
OUTPUT_DIR = os.path.join(_HERE, "output")
PLOTS_DIR = os.path.join(_HERE, "plots")
CONFIG_YAML = os.path.join(_ROOT, "configs", "serving_config_finetuning_llama3.yaml")
DEFAULT_SLO_FALLBACK = 1.0

ALL_MODES = ("loose", "tight", "nutanix")
MODE_COLORS = {"loose": "tab:blue", "tight": "tab:red", "nutanix": "tab:green"}
FT_SHADE_COLOR = "tab:orange"


def detect_gpu_subdir() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True, timeout=2.0)
        name = (out.strip().splitlines() or [""])[0].upper()
        if "A100" in name:
            return "A100"
        if "5090" in name:
            return "5090"
    except Exception:
        pass
    return "5090"


_DEFAULT_GPU_SUBDIR = detect_gpu_subdir()
TIMELINES_DIR = os.path.join(_HERE, "timelines", _DEFAULT_GPU_SUBDIR)


# ---------------- IO helpers (csv + numpy) ----------------
def ensure_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")


def _f(x) -> float:
    """Parse to float, '' / None -> nan."""
    try:
        if x is None or str(x).strip() == "":
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def load_results(csv_path: str) -> dict:
    """Columns: idx,t_rel_s,latency_s,status,ttft_s,avg_tbt_s,worst_tbt_s.
    Returns a dict of numpy arrays + a boolean `ok` mask."""
    cols = {k: [] for k in
            ["idx", "t_rel_s", "latency_s", "ttft_s", "avg_tbt_s", "worst_tbt_s"]}
    ok = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        required = ["idx", "t_rel_s", "latency_s", "status", "ttft_s"]
        missing = [c for c in required if c not in (r.fieldnames or [])]
        if missing:
            raise ValueError(f"{csv_path} missing columns: {missing}")
        for row in r:
            cols["idx"].append(_f(row.get("idx")))
            cols["t_rel_s"].append(_f(row.get("t_rel_s")))
            cols["latency_s"].append(_f(row.get("latency_s")))
            cols["ttft_s"].append(_f(row.get("ttft_s")))
            cols["avg_tbt_s"].append(_f(row.get("avg_tbt_s")))
            cols["worst_tbt_s"].append(_f(row.get("worst_tbt_s")))
            ok.append(str(row.get("status", "")).strip() == "ok")
    out = {k: np.asarray(v, dtype=float) for k, v in cols.items()}
    out["ok"] = np.asarray(ok, dtype=bool)
    return out


def load_timeline(csv_path: str) -> dict:
    """Columns: timestamp_s, prompt_length, max_new_tokens (+ extras).
    Returns {t_rel_s, max_new_tokens, row_id} numpy arrays (row_id == file
    order, matching the benchmark's request idx)."""
    ts, tok = [], []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for c in ("timestamp_s", "max_new_tokens"):
            if c not in (r.fieldnames or []):
                raise ValueError(f"{csv_path} missing column: {c}")
        for row in r:
            ts.append(_f(row.get("timestamp_s")))
            tok.append(_f(row.get("max_new_tokens")))
    ts = np.asarray(ts, dtype=float)
    tok = np.asarray(tok, dtype=float)
    t0 = float(ts.min()) if len(ts) else 0.0
    return {"t_rel_s": ts - t0, "max_new_tokens": tok,
            "row_id": np.arange(len(ts), dtype=float)}


def parse_bwd_log_csv(csv_path: str):
    """Returns (rel_time_s, cum_tokens, avg_tok_s). Columns:
    timestamp, ..., batch_tokens, total_processed_tokens. Missing/empty ->
    empty arrays (FT band just won't render)."""
    if not os.path.exists(csv_path):
        return np.array([]), np.array([]), float("nan")
    has_total = False
    rows = []  # (dt, total_val_or_nan, batch_tok)
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "timestamp" not in r.fieldnames:
            return np.array([]), np.array([]), float("nan")
        has_total = "total_processed_tokens" in r.fieldnames
        for row in r:
            try:
                dt = datetime.datetime.fromisoformat((row.get("timestamp") or "").strip())
            except Exception:
                continue
            total_val = _f(row.get("total_processed_tokens")) if has_total else float("nan")
            rows.append((dt, total_val, _f(row.get("batch_tokens"))))
    if not rows:
        return np.array([]), np.array([]), float("nan")
    # Collapse same-second rows to the last (1s timestamp resolution).
    dedup = {}
    for dt, total_val, batch_tok in rows:
        dedup[dt] = (total_val, batch_tok)
    items = sorted(dedup.items(), key=lambda x: x[0])
    t0 = items[0][0]
    rel = np.array([(dt - t0).total_seconds() for dt, _ in items], dtype=float)
    if has_total:
        cum = np.array([v[0] for _, v in items], dtype=float)
    else:
        cum = np.cumsum(np.array([v[1] for _, v in items], dtype=float))
    elapsed = float(rel[-1]) if len(rel) else 0.0
    total = float(cum[-1]) if len(cum) else 0.0
    return rel, cum, (total / elapsed if elapsed > 0 else float("nan"))


def read_slo(yaml_path: str, key: str) -> Optional[float]:
    try:
        import yaml
        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}
        v = (data.get("slo", {}) or {}).get(key)
        return float(v) if v is not None else None
    except Exception:
        return None


def read_ttft_slo(yaml_path: str) -> Optional[float]:
    return read_slo(yaml_path, "ttft_slo")


# ---------------- token-distribution helpers ----------------
def _distribute_to_bins(t_starts, t_ends, weights, t_max, bin_s):
    n_bins = int(np.ceil(t_max / bin_s)) + 1
    bins = np.zeros(n_bins, dtype=float)
    for ts, te, w in zip(t_starts, t_ends, weights):
        if te <= ts:
            b = int(ts // bin_s)
            if 0 <= b < n_bins:
                bins[b] += float(w)
            continue
        rate = float(w) / (te - ts)
        for b in range(max(0, int(np.floor(ts / bin_s))),
                       min(n_bins, int(np.ceil(te / bin_s)))):
            lo = max(ts, b * bin_s)
            hi = min(te, (b + 1) * bin_s)
            if hi > lo:
                bins[b] += rate * (hi - lo)
    return bins


def _ft_per_bin(rel_t, cum_tok, t_max, bin_s):
    n_bins = int(np.ceil(t_max / bin_s)) + 1
    if len(rel_t) == 0 or len(cum_tok) == 0:
        return np.zeros(n_bins)
    edges = np.arange(n_bins + 1) * bin_s
    cum_at_edges = np.interp(edges, rel_t, cum_tok, left=0.0, right=float(cum_tok[-1]))
    deltas = np.diff(cum_at_edges)
    deltas[deltas < 0] = 0.0
    return deltas


def _smooth(y, window_s, bin_s):
    if y.size == 0 or window_s <= 0 or bin_s <= 0:
        return y
    w = max(1, int(round(window_s / bin_s)))
    if w <= 1 or y.size < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="same")


def _auto_window(t_max):
    return 5.0 if t_max <= 0 else float(np.clip(t_max / 100.0, 5.0, 60.0))


# ---------------- panels ----------------
def plot_request_timeline(ax, tl):
    t = tl["t_rel_s"]
    tok = tl["max_new_tokens"]
    if len(t) == 0:
        ax.set_title("Request Timeline (empty)")
        return
    bucket_idx = np.floor(t).astype(int)
    n = int(bucket_idx.max()) + 1
    req_per_s = np.bincount(bucket_idx, minlength=n).astype(float)
    tok_per_s = np.bincount(bucket_idx, weights=tok, minlength=n).astype(float)
    centers = np.arange(n, dtype=float)
    ax.bar(centers, req_per_s, width=1.0, color="tab:gray", alpha=0.55,
           align="edge", label="req/s", edgecolor="none")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Requests / s", color="tab:gray")
    ax.tick_params(axis="y", labelcolor="tab:gray")
    ax.set_xlim(0, n)
    ax.set_ylim(bottom=0)
    ax.set_title("Scheduled Request Timeline")
    ax2 = ax.twinx()
    ax2.plot(centers + 0.5, tok_per_s, color="tab:green", marker="o",
             markersize=3, linewidth=1.5, label="tokens/s")
    ax2.set_ylabel("Output tokens / s", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    ax2.set_ylim(bottom=0)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)


def plot_latency_vs_time(ax, res, label, color):
    ok = res["ok"]
    t = res["t_rel_s"][ok]
    lat = res["latency_s"][ok]
    avg = float(np.nanmean(lat)) if lat.size else float("nan")
    series_label = f"{label} (avg {avg:.3f}s)" if np.isfinite(avg) else label
    ax.scatter(t, lat, s=10, color=color, alpha=0.7, label=series_label)
    if np.isfinite(avg):
        ax.axhline(avg, color=color, linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title("Request E2E Latency vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Latency (s)")
    ax.legend()


def plot_throughput_curves(ax, res, tl, bwd_log, color_inf,
                           bin_s=1.0, smoothing_window_s=None):
    # Map request idx -> timeline max_new_tokens via row_id.
    tok_by_row = {int(rid): tok for rid, tok in
                  zip(tl["row_id"], tl["max_new_tokens"])}
    ok = res["ok"]
    idx = res["idx"]
    t_rel = res["t_rel_s"]
    ttft = res["ttft_s"]
    lat = res["latency_s"]
    sel = ok & np.isfinite(ttft) & np.isfinite(lat)
    inf_t_start, inf_t_end, inf_tokens = [], [], []
    for i in np.nonzero(sel)[0]:
        tok = tok_by_row.get(int(idx[i]))
        if tok is None or not np.isfinite(tok):
            continue
        inf_t_start.append(t_rel[i] + ttft[i])
        inf_t_end.append(t_rel[i] + lat[i])
        inf_tokens.append(tok)
    inf_t_start = np.asarray(inf_t_start, dtype=float)
    inf_t_end = np.asarray(inf_t_end, dtype=float)
    inf_tokens = np.asarray(inf_tokens, dtype=float)

    rel_t, cum_tok, _ = bwd_log
    t_max = max(float(inf_t_end.max()) if len(inf_t_end) else 0.0,
                float(rel_t[-1]) if len(rel_t) else 0.0)
    if t_max <= 0:
        ax.set_title("Throughput (no data)")
        return
    inf_per_s = _distribute_to_bins(inf_t_start, inf_t_end, inf_tokens, t_max, bin_s) / bin_s
    ft_per_s = _ft_per_bin(rel_t, cum_tok, t_max, bin_s) / bin_s
    n = max(len(inf_per_s), len(ft_per_s))
    inf_per_s = np.pad(inf_per_s, (0, n - len(inf_per_s)))
    ft_per_s = np.pad(ft_per_s, (0, n - len(ft_per_s)))
    centers = (np.arange(n) + 0.5) * bin_s
    total = inf_per_s + ft_per_s
    inf_avg, ft_avg, tot_avg = inf_per_s.mean(), ft_per_s.mean(), total.mean()
    win_s = smoothing_window_s if smoothing_window_s is not None else _auto_window(t_max)
    inf_smooth = _smooth(inf_per_s, win_s, bin_s)
    total_smooth = _smooth(total, win_s, bin_s)
    if win_s > bin_s:
        ax.plot(centers, total, color="black", linewidth=0.4, alpha=0.20, zorder=1)
    ax.fill_between(centers, 0, inf_smooth, color=color_inf, alpha=0.20, linewidth=0,
                    label=f"Inference contribution (avg {inf_avg:.0f} tok/s)", zorder=2)
    ax.fill_between(centers, inf_smooth, total_smooth, color=FT_SHADE_COLOR, alpha=0.30,
                    hatch="//", linewidth=0,
                    label=f"Finetune contribution (avg {ft_avg:.0f} tok/s)", zorder=2)
    ax.plot(centers, inf_smooth, color=color_inf, linewidth=1.6,
            label=f"Inference (avg {inf_avg:.0f} tok/s)", zorder=3)
    ax.plot(centers, total_smooth, color="black", linewidth=1.6,
            label=f"Total (avg {tot_avg:.0f} tok/s)", zorder=3)
    title = "Throughput (tokens/s)"
    if win_s > bin_s:
        title += f" — {win_s:.0f}s rolling mean"
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tokens / s")
    ax.legend(loc="best", fontsize=8)


def plot_ttft_satisfaction(ax, res, slo_s, window_s=5.0, avg_tbt_slo=None):
    ok = res["ok"] & np.isfinite(res["ttft_s"])
    t = res["t_rel_s"][ok]
    ttft = res["ttft_s"][ok]
    if t.size == 0:
        ax.set_title("TTFT Satisfaction Rate (no data)")
        return
    sat = (ttft <= slo_s).astype(int)
    grid = np.arange(0.0, float(t.max()) + 1.0, 1.0)
    rates = np.full(len(grid), np.nan)
    half = window_s / 2.0
    for i, t0 in enumerate(grid):
        mask = (t >= t0 - half) & (t <= t0 + half)
        if mask.sum() > 0:
            rates[i] = sat[mask].mean() * 100.0
    overall = sat.mean() * 100.0
    avg_ttft = float(np.mean(ttft))
    p90_ttft = float(np.percentile(ttft, 90))
    # Avg time-between-tokens across requests (avg_tbt_s; NaN for single-token
    # responses → dropped). Flagged against avg_tbt_slo when available.
    tbt = res["avg_tbt_s"][ok]
    tbt = tbt[np.isfinite(tbt)]
    avg_tbt = float(np.mean(tbt)) if tbt.size else float("nan")
    ax.plot(grid, rates, color="tab:purple", linewidth=1.5,
            label=f"Satisfaction ({int(window_s)}s window) — overall {overall:.1f}%")
    ax.axhline(95.0, color="tab:red", linestyle="--", linewidth=1, alpha=0.6,
               label="95% target")
    ax.set_ylim(0, 105)
    ax.set_title(f"TTFT Satisfaction Rate (SLO ≤ {slo_s:.2f}s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("% of requests")
    # Summary TTFT stats vs the SLO (y-axis is %, so these go in a text box).
    # Green/red flags whether avg / p90 TTFT meet the SLO.
    _flag = lambda v, lim: (  # noqa: E731
        "" if lim is None or not np.isfinite(v)
        else (" ✓" if v <= lim else " ✗"))
    tbt_line = (f"avg TBT = {avg_tbt:.3f}s{_flag(avg_tbt, avg_tbt_slo)}"
                + (f"  (SLO {avg_tbt_slo:.2f}s)" if avg_tbt_slo else ""))
    ax.text(0.02, 0.04,
            f"avg TTFT = {avg_ttft:.3f}s{_flag(avg_ttft, slo_s)}\n"
            f"p90 TTFT = {p90_ttft:.3f}s{_flag(p90_ttft, slo_s)}\n"
            f"TTFT SLO = {slo_s:.2f}s\n"
            f"{tbt_line}",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75,
                      edgecolor="0.7"))
    ax.legend(loc="upper right", fontsize=8)


def make_figure_for_mode(mode, base_suffix, output_dir, plots_dir,
                         timeline_csv_dir, out_path, slo_s, window_s,
                         throughput_window_s=None, avg_tbt_slo=None):
    full = f"{base_suffix}_{mode}"
    results_csv = os.path.join(output_dir, f"timeline_results{full}.csv")
    bwd_log_csv = os.path.join(output_dir, f"bwd_log{full}.csv")
    timeline_csv = os.path.join(timeline_csv_dir, f"timeline_{mode}.csv")
    # results + timeline required; bwd_log optional (no-co runs).
    ensure_exists(timeline_csv)
    ensure_exists(results_csv)

    tl = load_timeline(timeline_csv)
    res = load_results(results_csv)
    bwd_log = parse_bwd_log_csv(bwd_log_csv)
    color = MODE_COLORS.get(mode, "tab:gray")

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    plot_request_timeline(axes[0], tl)
    plot_latency_vs_time(axes[1], res, label=mode, color=color)
    # If this is a co-serving run, overlay the inference-only (no-co) E2E
    # latency for the same mode when its results exist — only on the E2E panel,
    # so the co-serving overhead is visible at a glance.
    if base_suffix:
        infonly_csv = os.path.join(output_dir, f"timeline_results_{mode}.csv")
        if (os.path.exists(infonly_csv)
                and os.path.abspath(infonly_csv) != os.path.abspath(results_csv)):
            try:
                plot_latency_vs_time(axes[1], load_results(infonly_csv),
                                     label="inf-only", color="tab:gray")
            except Exception as e:
                print(f"[auto_plot] skip inf-only overlay for {mode}: {e}")
    plot_throughput_curves(axes[2], res, tl, bwd_log, color_inf=color,
                           smoothing_window_s=throughput_window_s)
    plot_ttft_satisfaction(axes[3], res, slo_s=slo_s, window_s=window_s,
                           avg_tbt_slo=avg_tbt_slo)

    config_tag = base_suffix.lstrip("_") or "baseline"
    fig.suptitle(f"{mode}  ({config_tag})")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[auto_plot] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suffix", default="_co",
                    help="Base config suffix WITHOUT the trailing _<mode>. "
                         "Default '_co' (matches --co); use '' for no-co runs.")
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    ap.add_argument("--plots-dir", default=PLOTS_DIR)
    ap.add_argument("--timeline-csv-dir", default=TIMELINES_DIR)
    ap.add_argument("--mode", choices=ALL_MODES + ("all",), default="all")
    ap.add_argument("--slo", type=float, default=None,
                    help="TTFT SLO (s). Default: slo.ttft_slo from the finetuning YAML.")
    ap.add_argument("--config-yaml", default=CONFIG_YAML)
    ap.add_argument("--window", type=float, default=5.0)
    ap.add_argument("--throughput-window", type=float, default=None)
    args = ap.parse_args()

    slo_s = args.slo
    if slo_s is None:
        slo_s = read_ttft_slo(args.config_yaml)
        if slo_s is not None:
            print(f"[auto_plot] SLO = {slo_s:.3f}s (from {os.path.basename(args.config_yaml)})")
        else:
            slo_s = DEFAULT_SLO_FALLBACK
            print(f"[auto_plot] SLO = {slo_s:.3f}s (fallback)", file=sys.stderr)
    else:
        print(f"[auto_plot] SLO = {slo_s:.3f}s (from --slo)")

    avg_tbt_slo = read_slo(args.config_yaml, "avg_tbt_slo")
    if avg_tbt_slo is not None:
        print(f"[auto_plot] avg-TBT SLO = {avg_tbt_slo:.3f}s")

    modes = ALL_MODES if args.mode == "all" else (args.mode,)
    skipped, wrote = [], 0
    for mode in modes:
        tag = args.suffix.lstrip("_") or "baseline"
        out_path = os.path.join(args.plots_dir, f"{mode}_{tag}.png")
        try:
            make_figure_for_mode(
                mode=mode, base_suffix=args.suffix, output_dir=args.output_dir,
                plots_dir=args.plots_dir, timeline_csv_dir=args.timeline_csv_dir,
                out_path=out_path, slo_s=slo_s, window_s=args.window,
                throughput_window_s=args.throughput_window,
                avg_tbt_slo=avg_tbt_slo)
            wrote += 1
        except FileNotFoundError as e:
            skipped.append((mode, str(e)))
    for mode, msg in skipped:
        print(f"[auto_plot] {mode}: skipped (missing input — {msg})", file=sys.stderr)
    if wrote == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
