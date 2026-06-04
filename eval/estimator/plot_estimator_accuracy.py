#!/usr/bin/env python3
"""plot_estimator_accuracy.py — SLO execution-time estimator error vs time.

Reads the per-batch predicted-vs-actual CSV written by the server in
``validate_estimator`` mode (see configs/serving_config_finetuning_llama3_
validate.yaml) and produces ONE figure:

  • a single combined **error-vs-time** plot mixing the selected regimes
    (default Prefill + Decode): signed relative error
    (predicted−actual)/actual ×100 % vs time, smoothed over a time window,
    plus a green overlay of the same curve shifted up by the safety-margin
    offset → ``<stem><OUTPUT_SUFFIX>.png`` (+PDF).

It also prints a per-regime MAE / RMSE / bias / MAPE table to stdout.

Input CSV columns (from FinetuneScheduler._append_validation_row):
  timestamp, regime, was_graph, t_in, p, t_ft, b_d, k, s,
  execution_duration, predicted_duration
(``predicted_duration`` is the RAW model prediction — the ×(1+1.5·RMSE) safety
margin is NOT applied, so this scores the estimator model itself.)

Rows with a blank ``predicted_duration`` (estimator cold-start, before the
first fit) are dropped — they have no prediction to score.

All styling / behaviour knobs live in the Settings block below — edit there.

Usage:
  python eval/estimator/plot_estimator_accuracy.py
  python eval/estimator/plot_estimator_accuracy.py --input <csv> --output <png>
"""
import argparse
import csv
import datetime
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(_HERE, "estimator_validation_nutanix.csv")

# ============================================================================
# Settings — edit here to retitle / resize / restyle / re-window.
# ============================================================================

# ---- Output ----
GENERATE_PDF = True
PNG_DPI = 130
OUTPUT_SUFFIX = "_err_timeseries"   # figure path = <input stem><suffix>.{png,pdf}

# ---- Figure ----
FIGSIZE = (15, 4.5)                 # wide aspect for the error-vs-time series

# ---- Regime mix (steps merged into the single error figure) ----
REGIMES = ["inf_prefill", "decode_only"]    # default → "Prefill + Decode"

# ---- Smoothing / windowing ----
SMOOTH_SEC = 5.0          # centered moving-average window (s); 0 disables
DROP_FIRST_SEC = 5.0      # drop the estimator cold-start (s) from curve + stats

# ---- Title ----
TITLE = "Batch Prediction Error"    # None → auto "<regimes> Prediction Error"

# ---- Axis labels ----
XLABEL = None             # None → "Time (s)" / "Step index" (auto)
YLABEL = "Rel. error (%)"

# ---- Curves ----
RAW_ERR_COLOR = "tab:orange"
RAW_ERR_LABEL = "Raw prediction"
MARGIN_COLOR = "tab:green"
MARGIN_LABEL = "Prediction with safety margin"
MARGIN_OFFSET_PCT = 5.0   # +pp vertical shift of the safety-margin overlay
LINE_WIDTH = 1.1

# ---- Reference lines ----
SHOW_MEAN_LINE = False
MEAN_LINE_COLOR = "tab:red"
ZERO_LINE_COLOR = "0.4"

# ---- Annotation box (top-left) ----
SHOW_ANNOTATION = True    # "Avg |Err|: X%"

# ---- Legend ----
LEGEND_LOC = "lower right"
LEGEND_FRAMEALPHA = 0.9

# ---- Font sizes ----
FONTSIZE_TITLE = 20
FONTSIZE_AXIS_LABEL = 14
FONTSIZE_LEGEND = 13
FONTSIZE_ANNOT = 14

# ---- Font weights ----
FONTWEIGHT_TITLE = "bold"
FONTWEIGHT_AXIS_LABEL = "bold"
FONTWEIGHT_LEGEND = "normal"

# ---- Metrics report ----
MS = True   # report metrics in milliseconds (estimator times are ~10-100 ms)

# ---- Regime → human label (used for the auto title when TITLE is None) ----
REGIME_LABEL = {
    "inf_prefill": "Prefill",
    "eager": "Eager (co-serving)",
    "decode_only": "Decode",
    "prefill": "Prefill",       # inf_prefill ∪ eager
    "all": "Step",
}
# ============================================================================


def load_rows(path):
    """Return parallel arrays (times, actual, predicted, regime, was_graph) for
    rows with a finite predicted value. Times are seconds relative to the first
    parseable ``timestamp`` (NaN where unparseable). Durations in seconds."""
    actual, predicted, regime, was_graph, ts = [], [], [], [], []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        need = {"execution_duration", "predicted_duration", "regime"}
        missing = need - set(r.fieldnames or [])
        if missing:
            sys.exit(f"[estimator-eval] {path}: missing columns {sorted(missing)}")
        for row in r:
            pred_s = (row.get("predicted_duration") or "").strip()
            act_s = (row.get("execution_duration") or "").strip()
            if not pred_s or not act_s:
                continue
            try:
                p, a = float(pred_s), float(act_s)
            except ValueError:
                continue
            if not (np.isfinite(p) and np.isfinite(a)) or a <= 0:
                continue
            predicted.append(p)
            actual.append(a)
            regime.append((row.get("regime") or "?").strip())
            was_graph.append((row.get("was_graph") or "").strip())
            try:
                ts.append(datetime.datetime.fromisoformat(
                    (row.get("timestamp") or "").strip()))
            except (ValueError, TypeError):
                ts.append(None)
    valid = [t for t in ts if t is not None]
    if valid:
        t0 = min(valid)
        times = np.array([(t - t0).total_seconds() if t is not None else np.nan
                          for t in ts], dtype=float)
    else:
        times = np.full(len(actual), np.nan)
    return (times, np.asarray(actual), np.asarray(predicted),
            np.asarray(regime), np.asarray(was_graph))


def _time_smooth(t, y, window_sec):
    """Centered time-windowed moving average. ``t`` must be sorted ascending.

    For each point i, averages every y[j] whose timestamp lies within
    ±window_sec/2 of t[i] — a true *time* window, so it behaves correctly even
    when steps are irregularly spaced in wall-clock time. O(n) via cumulative
    sums + searchsorted. Returns ``y`` unchanged if smoothing is disabled
    (window<=0). The window is in whatever units ``t`` carries (seconds, or
    step index in the timestamp-less fallback)."""
    if window_sec is None or window_sec <= 0 or t.size == 0:
        return y
    half = window_sec / 2.0
    lo = np.searchsorted(t, t - half, side="left")
    hi = np.searchsorted(t, t + half, side="right")
    csum = np.concatenate(([0.0], np.cumsum(y.astype(float))))
    counts = np.maximum(hi - lo, 1)
    return (csum[hi] - csum[lo]) / counts


def _padded_limits(arrs, include=(), frac=0.08):
    """(lo, hi) spanning all finite values in ``arrs`` (+ any ``include``
    points such as 0 / the mean line), padded by ``frac`` of the range. Used to
    autoscale the y-axis to the SMOOTHED trace."""
    vals = np.concatenate([np.asarray(a, float).ravel() for a in arrs]
                          + [np.asarray(include, float).ravel()])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        hi = lo + 1.0
    pad = frac * (hi - lo)
    return lo - pad, hi + pad


def _regime_mask(regime, sel):
    if sel == "all":
        return np.ones(regime.size, dtype=bool)
    if sel == "prefill":                    # any prefill-carrying step
        return np.isin(regime, ["inf_prefill", "eager"])
    return regime == sel


def _metrics(actual, predicted):
    """(n, mae, rmse, bias, mape%) for predicted−actual. NaN-safe on empty."""
    n = actual.size
    if n == 0:
        return 0, float("nan"), float("nan"), float("nan"), float("nan")
    err = predicted - actual
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    mape = float(np.mean(np.abs(err) / actual) * 100.0)
    return n, mae, rmse, bias, mape


def _fmt(x):
    if not np.isfinite(x):
        return "   nan"
    return f"{x * 1000:7.2f}ms" if MS else f"{x:7.4f}s"


def report(actual, predicted, regime, was_graph):
    unit = "ms" if MS else "s"
    print(f"\n[estimator-eval] {actual.size} scored steps (predicted − actual)\n")
    hdr = f"{'group':<16}{'n':>6}  {'MAE':>10}{'RMSE':>10}{'bias':>10}  {'MAPE':>8}"
    print(hdr)
    print("-" * len(hdr))

    def _line(label, mask):
        n, mae, rmse, bias, mape = _metrics(actual[mask], predicted[mask])
        if n == 0:
            return
        print(f"{label:<16}{n:>6}  {_fmt(mae):>10}{_fmt(rmse):>10}"
              f"{_fmt(bias):>10}  {mape:6.1f}%")

    for rg in ("eager", "inf_prefill", "decode_only"):
        _line(rg, regime == rg)
    other = ~np.isin(regime, ["eager", "inf_prefill", "decode_only"])
    if other.any():
        _line("(other)", other)
    print()
    for wg, lbl in (("True", "graph"), ("False", "eager-step"), ("", "unknown")):
        _line(lbl, was_graph == wg)
    print()
    _line("OVERALL", np.ones(actual.size, dtype=bool))
    print(f"\n(times shown in {unit}; bias>0 ⇒ estimator OVER-predicts)\n")


def plot_error_timeseries(times, actual, predicted, out_path, title, xlabel):
    """Signed relative prediction error vs time for the already-filtered subset.

    A single panel of (predicted−actual)/actual ×100 %, with a 0 % reference
    line (above ⇒ over-predict) and, optionally, the mean as a flat dashed
    line. A green overlay shows the same curve shifted up by
    ``MARGIN_OFFSET_PCT`` pp (the estimator's safety margin). The trace is a
    centered ``SMOOTH_SEC`` time-windowed moving average; the first
    ``DROP_FIRST_SEC`` seconds (estimator cold-start) are dropped from both the
    curve and the annotation, which is computed on the RAW error."""
    order = np.argsort(times)
    t = times[order]
    a, p = actual[order], predicted[order]
    # Drop the estimator cold-start from both the curve and the stats.
    keep = t >= DROP_FIRST_SEC
    if keep.any():
        t, a, p = t[keep], a[keep], p[keep]
    rel = (p - a) / a * 100.0           # signed % error
    mape = float(np.mean(np.abs(rel))) if rel.size else float("nan")
    mbias = float(np.mean(rel)) if rel.size else float("nan")
    sm = SMOOTH_SEC and SMOOTH_SEC > 0
    rel_s = _time_smooth(t, rel, SMOOTH_SEC) if sm else rel
    # Constant offset commutes with the moving average, so rel_s + k is the
    # smoothed margin curve.
    margin = rel_s + MARGIN_OFFSET_PCT

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    ax.axhline(0.0, color=ZERO_LINE_COLOR, linewidth=1.0, zorder=1)
    raw_line, = ax.plot(t, rel_s, color=RAW_ERR_COLOR, linewidth=LINE_WIDTH,
                        zorder=2, label=RAW_ERR_LABEL)
    margin_line, = ax.plot(t, margin, color=MARGIN_COLOR, linewidth=LINE_WIDTH,
                           zorder=2, label=MARGIN_LABEL)
    if SHOW_MEAN_LINE:
        ax.axhline(mbias, color=MEAN_LINE_COLOR, linestyle="--",
                   linewidth=1.1, zorder=3)
    ax.set_ylabel(YLABEL, fontsize=FONTSIZE_AXIS_LABEL,
                  fontweight=FONTWEIGHT_AXIS_LABEL)
    ax.set_xlabel(xlabel, fontsize=FONTSIZE_AXIS_LABEL,
                  fontweight=FONTWEIGHT_AXIS_LABEL)
    ax.set_title(title, fontsize=FONTSIZE_TITLE, fontweight=FONTWEIGHT_TITLE)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(handles=[raw_line, margin_line], loc=LEGEND_LOC,
              framealpha=LEGEND_FRAMEALPHA,
              prop={"weight": FONTWEIGHT_LEGEND, "size": FONTSIZE_LEGEND})
    if SHOW_ANNOTATION:
        ax.text(0.012, 0.94, f"Avg |Err|: {mape:.2f}%",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=FONTSIZE_ANNOT,
                bbox=dict(boxstyle="round", facecolor="white",
                          edgecolor="0.5", alpha=0.9))
    if t.size:
        ax.set_xlim(float(t.min()), float(t.max()))
    if sm:  # scale to both smoothed traces (+ 0 / mean reference)
        ax.set_ylim(*_padded_limits([rel_s, margin], include=(0.0, mbias)))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=PNG_DPI)
    print(f"[estimator-eval] wrote error-series → {out_path}")
    if GENERATE_PDF:
        pdf = os.path.splitext(out_path)[0] + ".pdf"
        fig.savefig(pdf, format="pdf", bbox_inches="tight", pad_inches=0.05)
        print(f"[estimator-eval] wrote error-series → {pdf}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=DEFAULT_INPUT,
                    help="Validation CSV. Default: "
                         "eval/estimator/estimator_validation_nutanix.csv")
    ap.add_argument("--output", default=None,
                    help="Error-series PNG path. Default: alongside the input, "
                         f"<stem>{OUTPUT_SUFFIX}.png.")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"[estimator-eval] input not found: {args.input}\n"
                 "Run a validate_estimator session first (see "
                 "configs/serving_config_finetuning_llama3_validate.yaml).")

    times, actual, predicted, regime, was_graph = load_rows(args.input)
    if actual.size == 0:
        sys.exit(f"[estimator-eval] no scored rows in {args.input} "
                 "(all predicted_duration blank? estimator never fitted).")

    report(actual, predicted, regime, was_graph)

    # ---- ONE combined error-vs-time figure mixing the selected regimes ----
    # (default: inf_prefill + decode_only → "Prefill + Decode"). Steps from all
    # selected regimes are merged and sorted by time, so the error trace is a
    # single continuous view across the run.
    union = np.zeros(regime.size, dtype=bool)
    for sel in REGIMES:
        union |= _regime_mask(regime, sel)
    if not union.any():
        sys.exit(f"[estimator-eval] no rows for regimes {REGIMES} "
                 f"(present: {sorted(set(regime))}).")

    t_sub = times[union]
    if np.isfinite(t_sub).all() and t_sub.size:
        xs, xlabel = t_sub, "Time (s)"
    else:
        # No usable timestamps → fall back to step index on the x-axis.
        xs, xlabel = np.arange(int(union.sum()), dtype=float), "Step index"
    if XLABEL is not None:
        xlabel = XLABEL

    title = TITLE or (" + ".join(REGIME_LABEL.get(s, s) for s in REGIMES)
                      + " Prediction Error")
    stem = os.path.splitext(args.input)[0]
    out = args.output or f"{stem}{OUTPUT_SUFFIX}.png"
    print(f"[estimator-eval] error series (mixed {REGIMES}): "
          f"{int(union.sum())} steps")
    plot_error_timeseries(xs, actual[union], predicted[union], out, title,
                          xlabel)


if __name__ == "__main__":
    main()
