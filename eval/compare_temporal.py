#!/usr/bin/env python3
"""compare_temporal.py — temporal-sharing comparison figure.

Reads the result bundle under ``eval/temporal_share_output/`` and builds ONE
figure with three panels:

  Row 0 (full width)  Scheduled request timeline (req/s bars + output tok/s
                      line) from ``temporal_share_output/timeline_<mode>.csv``.
  Row 1, left         5090 panel.
  Row 1, right        A100 panel.

Each per-GPU panel overlays, on a shared time axis that starts at t=0:
  * Left y-axis  — per-request E2E latency vs time for BOTH the co-serving
                   run (``co_factor_0``) and the inference-only run. The
                   co-serving legend reports its full mean (overhead vs the
                   inference-only full mean) and its slowest-1% tail (mean of
                   the worst 1%, overhead vs the inference-only slowest-1%),
                   both as text in the legend entry (no extra plotted lines).
  * Right y-axis — the co_factor_0 finetune (backward) throughput in tok/s,
                   derived from the bwd_log's ``total_processed_tokens``.

A GPU whose result files are absent (e.g. an empty ``A100/`` dir) renders as
a labeled blank panel, so the figure layout is stable across machines.

Expected files per GPU subdir (``<mode>`` defaults to ``nutanix``,
``co_factor_0`` from ``--factor 0``):
  timeline_results_co_factor_<F>_<mode>.csv   (co-serving E2E latency)
  timeline_results_<mode>.csv                 (inference-only E2E latency)
  bwd_log_co_factor_<F>_<mode>.csv            (FT backward throughput)
  bench_meta_co_factor_<F>_<mode>.json        (optional — FT time anchor)

Time alignment (assumes --real-timestamp results): the co results CSV carries
an absolute ``timestamp`` column on the same wall clock as the bwd_log, so the
FT curve is anchored to the co run's first-request wall clock — pinning FT to
inference exactly, no offset guessing. If that column is absent (older runs)
it falls back to a ``bench_meta`` ``t_first_wall_iso`` anchor, then to first-
backward anchoring. The anchor source is printed per GPU.

Finetune runs during warmup, so backwards complete BEFORE the first inference
request; that pre-inference FT lands in ``[0, tl_base)`` (timeline coords) and
is kept, not trimmed — only off-screen bins (coord < 0) are dropped.

Helpers (loaders, bwd-log parsing, FT-per-bin, the timeline panel) are reused
from ``auto_plot.py`` so the math matches the main plotting path exactly.

Usage:
  python eval/compare_temporal.py                 # nutanix, factor 0
  python eval/compare_temporal.py --mode nutanix --factor 0
"""
import argparse
import csv
import datetime
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# Reuse auto_plot.py's helpers verbatim (same dir). Insert _HERE so the import
# resolves regardless of the caller's CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from auto_plot import (  # noqa: E402
    FT_SHADE_COLOR,
    _auto_window,
    _smooth,
    load_results,
    load_timeline,
    plot_request_timeline,
)

DEFAULT_INPUT_DIR = os.path.join(_HERE, "temporal_share_output")
GPUS = ("5090", "A100")

# ============================================================================
# Settings — edit here to retitle / resize / restyle.
# ============================================================================

# ---- Output ----
GENERATE_PDF = True               # write a .pdf next to the .png
PNG_DPI = 130

# ---- Figure ----
FIGSIZE = (14, 13)                # overall figure size in inches
HEIGHT_RATIOS = (0.8, 1.0, 1.0)   # gridspec row heights: timeline / GPU1 / GPU2
SUPTITLE = None                   # None disables the figure-level title

# ---- Per-panel titles ----
# Set to None to drop the panel header above each GPU subplot. Use
# ``{gpu_name}`` placeholder for substitution.
PANEL_TITLE_TEMPLATE = "{gpu_name} — E2E latency & FT throughput"

# ---- Axis labels ----
XLABEL = "Time (s)"
YLABEL_LATENCY = "E2E latency (s)"
YLABEL_FT = "FT throughput (tok/s)"

# ---- Y-axis headroom (per-GPU panels only) ----
# Cap both the left (latency) and right (FT throughput) y-axes at
# ``peak × YMAX_HEADROOM``. Keeps the legend / annotation box from
# overlapping the highest data points.
YMAX_HEADROOM = 1.3

# ---- Display names (replace internal series ids in the legend) ----
DISPLAY_NAME_INF = "vLLM"
DISPLAY_NAME_CO = "DeltaServe-vLLM-Temp"
DISPLAY_NAME_TAIL_OVERHEAD = "Lowest 1% overhead"

# ---- Font sizes ----
FONTSIZE_PANEL_TITLE = 12
FONTSIZE_AXIS_LABEL = 11
FONTSIZE_TICK = 10
FONTSIZE_LEGEND = 9

# ---- Colors ----
INF_COLOR = "tab:blue"
CO_COLOR = "tab:red"
# FT_SHADE_COLOR comes from auto_plot.py (orange).

# ---- Scatter style ----
SCATTER_SIZE = 10
SCATTER_ALPHA = 0.7
MEAN_LINE_ALPHA = 0.5
MEAN_LINE_WIDTH = 1.0

# ============================================================================


def _load_or_none(loader, path: str):
    """Return ``loader(path)`` or ``None`` if the file is missing/unreadable."""
    if not os.path.exists(path):
        return None
    try:
        return loader(path)
    except Exception as e:  # noqa: BLE001 — a bad file shouldn't kill the figure
        print(f"[compare_temporal] failed to load {path}: {e}", file=sys.stderr)
        return None


def _timeline_base(path: str) -> float:
    """Smallest ``timestamp_s`` in the timeline file — the schedule's first-
    request offset (e.g. 15.0s for a 600-800 slice). The benchmark strips this
    when it normalizes latency ``t_rel_s`` to start at 0, so we add it back to
    put every series on the timeline file's own clock (first inference at 15s,
    not 0)."""
    import csv as _csv
    try:
        with open(path) as f:
            vals = [float(r["timestamp_s"]) for r in _csv.DictReader(f)]
    except Exception:
        return 0.0
    return min(vals) if vals else 0.0


def _results_first_wall(path: str):
    """First request's wall clock from a results CSV's ``timestamp`` column
    (written by ``auto_benchmark.py --real-timestamp``, ISO-millisecond, same
    clock as the bwd_log). This is the exact origin the FT curve should anchor
    to. Returns a ``datetime`` or ``None`` when the column is absent (run made
    without --real-timestamp)."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, newline="") as f:
            r = csv.DictReader(f)
            if not r.fieldnames or "timestamp" not in r.fieldnames:
                return None
            best = None
            for row in r:
                try:
                    dt = datetime.datetime.fromisoformat(
                        (row.get("timestamp") or "").strip())
                except (ValueError, TypeError):
                    continue
                if best is None or dt < best:
                    best = dt
            return best
    except Exception:
        return None


def _read_t0_wall(meta_path: str):
    """Pull ``t_first_wall_iso`` (the inference recording-phase t=0) from a
    bench_meta JSON so the FT series can anchor at the same origin as the
    latency series. Returns a ``datetime`` or ``None`` (→ legacy anchoring).
    Fallback for runs made without --real-timestamp."""
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f) or {}
        iso = meta.get("t_first_wall_iso")
        return datetime.datetime.fromisoformat(iso) if iso else None
    except Exception:
        return None


def _latency_stats(res):
    """Return ``(avg, avg95, avg99, slow1, n)`` over ok & finite E2E latencies.
    ``avg`` is the full mean (all 100% of requests — the "P100" baseline).
    ``avg95`` / ``avg99`` are the means of the lowest 95% / 99% (latencies <=
    p95 / p99) — tail-robust averages that drop the worst 5% / 1%.
    ``slow1`` is the mean of the SLOWEST 1% (latencies >= p99) — the tail."""
    ok = res["ok"]
    lat = res["latency_s"][ok]
    lat = lat[np.isfinite(lat)]
    if lat.size == 0:
        return (float("nan"),) * 4 + (0,)
    avg = float(lat.mean())
    p95 = float(np.percentile(lat, 95))
    p99 = float(np.percentile(lat, 99))
    below95 = lat[lat <= p95]
    below99 = lat[lat <= p99]
    above99 = lat[lat >= p99]
    avg95 = float(below95.mean()) if below95.size else avg
    avg99 = float(below99.mean()) if below99.size else avg
    slow1 = float(above99.mean()) if above99.size else float(lat.max())
    return avg, avg95, avg99, slow1, int(lat.size)


def _overhead_pct(a: float, b: float) -> float:
    """Percent overhead of ``a`` relative to baseline ``b`` ((a-b)/b·100)."""
    if np.isfinite(a) and np.isfinite(b) and b > 0:
        return (a - b) / b * 100.0
    return float("nan")


def _scatter_latency(ax, res, color: str, x_offset: float = 0.0,
                     mean_line: float = None) -> None:
    """Scatter E2E latency vs time on ``ax`` (x shifted by ``x_offset`` into
    timeline coords), plus an optional dashed line at ``mean_line``. The
    legend entry is built by the caller (so we can rename series and add
    overhead annotations) — this function deliberately does NOT push a
    label into the auto-legend handles."""
    ok = res["ok"]
    t = res["t_rel_s"][ok] + x_offset
    lat = res["latency_s"][ok]
    ax.scatter(t, lat, s=SCATTER_SIZE, color=color, alpha=SCATTER_ALPHA,
               zorder=3)
    if mean_line is not None and np.isfinite(mean_line):
        ax.axhline(mean_line, color=color, linestyle="--",
                   linewidth=MEAN_LINE_WIDTH, alpha=MEAN_LINE_ALPHA, zorder=2)


def _ft_throughput_curve(bwd_path: str, anchor_wall, tl_base: float,
                         ft_offset: float = 0.0, bin_s: float = 1.0,
                         window_s=None):
    """Read the bwd_log and bin each backward's ``batch_tokens`` at its
    completion time, mapped onto the timeline clock::

        coord = tl_base + ft_offset + (completion_wall − anchor_wall)

    Returns ``(centers, tok_per_s, last_x)`` smoothed, or ``None`` when there's
    no FT data.

    The finetune that completed BEFORE the first inference request lands in
    ``[0, tl_base)`` and is KEPT — only truly off-screen bins (coord < 0) are
    dropped, so the curve is not trimmed at the beginning. ``anchor_wall`` is
    the first inference request's wall clock (from the co results' real
    timestamps); ``None`` falls back to anchoring at the first backward row.
    Per-backward ``batch_tokens`` are binned directly (not the cumulative
    column), so there is no startup baseline spike."""
    if not os.path.exists(bwd_path):
        return None
    rows = []
    with open(bwd_path, newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "timestamp" not in r.fieldnames:
            return None
        for row in r:
            try:
                dt = datetime.datetime.fromisoformat(
                    (row.get("timestamp") or "").strip())
            except (ValueError, TypeError):
                continue
            try:
                bt = float(row.get("batch_tokens") or 0.0)
            except (ValueError, TypeError):
                bt = 0.0
            rows.append((dt, bt))
    if not rows:
        return None
    rows.sort(key=lambda x: x[0])
    if anchor_wall is None:
        anchor_wall = rows[0][0]
    base = tl_base + ft_offset
    coords = np.array([base + (dt - anchor_wall).total_seconds()
                       for dt, _ in rows], dtype=float)
    toks = np.array([bt for _, bt in rows], dtype=float)
    keep = coords >= 0.0          # drop only off-screen (pre-timeline-0) bins
    coords, toks = coords[keep], toks[keep]
    if coords.size == 0:
        return None
    t_max = float(coords.max())
    n_bins = int(np.ceil((t_max + bin_s) / bin_s))
    per_bin = np.zeros(n_bins, dtype=float)
    idx = np.minimum(np.floor(coords / bin_s).astype(int), n_bins - 1)
    np.add.at(per_bin, idx, toks)
    win_s = window_s if window_s is not None else _auto_window(t_max)
    tok_per_s = _smooth(per_bin / bin_s, win_s, bin_s)
    centers = (np.arange(n_bins) + 0.5) * bin_s
    return centers, tok_per_s, t_max


def plot_gpu_panel(ax, gpu_dir: str, gpu_name: str, mode: str, factor: float,
                   tl_base: float = 0.0, ft_offset: float = 0.0,
                   window_s=None) -> float:
    """E2E latency (vLLM vs DeltaServe-vLLM-Temp, left axis) + co FT
    throughput (right axis) for one GPU, all on the timeline file's clock.
    Inference is shifted by ``tl_base`` (so the first request lands at the
    timeline's first timestamp); FT is anchored at the timeline origin +
    ``ft_offset``. Blank labeled panel when neither result file exists.
    Returns the panel's right-most x (0 if blank) so the caller can set a
    shared x-limit."""
    fsuf = f"_co_factor_{factor:g}"
    co_path = os.path.join(gpu_dir, f"timeline_results{fsuf}_{mode}.csv")
    inf_path = os.path.join(gpu_dir, f"timeline_results_{mode}.csv")
    bwd_path = os.path.join(gpu_dir, f"bwd_log{fsuf}_{mode}.csv")
    meta_path = os.path.join(gpu_dir, f"bench_meta{fsuf}_{mode}.json")

    co = _load_or_none(load_results, co_path)
    inf = _load_or_none(load_results, inf_path)

    def _set_panel_title():
        if PANEL_TITLE_TEMPLATE:
            ax.set_title(PANEL_TITLE_TEMPLATE.format(gpu_name=gpu_name),
                         fontsize=FONTSIZE_PANEL_TITLE)

    if co is None and inf is None:
        ax.text(0.5, 0.5, f"{gpu_name}\n(no data)", ha="center", va="center",
                transform=ax.transAxes, fontsize=13, color="0.55")
        ax.set_xticks([])
        ax.set_yticks([])
        _set_panel_title()
        return 0.0

    # ---- left axis: E2E latency vs time (timeline coords), both runs ----
    t_ends = []
    inf_avg = inf_slow1 = float("nan")
    co_avg = co_slow1 = float("nan")
    if inf is not None:
        inf_avg, _, _, inf_slow1, _ = _latency_stats(inf)
        _scatter_latency(ax, inf, INF_COLOR, x_offset=tl_base,
                         mean_line=inf_avg)
        m = inf["ok"]
        t_ends.append(inf["t_rel_s"][m] + inf["latency_s"][m] + tl_base)
    if co is not None:
        co_avg, _, _, co_slow1, _ = _latency_stats(co)
        _scatter_latency(ax, co, CO_COLOR, x_offset=tl_base, mean_line=co_avg)
        m = co["ok"]
        t_ends.append(co["t_rel_s"][m] + co["latency_s"][m] + tl_base)

    ax.set_xlabel(XLABEL, fontsize=FONTSIZE_AXIS_LABEL)
    ax.set_ylabel(YLABEL_LATENCY, fontsize=FONTSIZE_AXIS_LABEL)
    ax.tick_params(axis="both", labelsize=FONTSIZE_TICK)
    ax.set_ylim(bottom=0)
    _set_panel_title()
    ax.grid(True, alpha=0.25)

    # ---- right axis: co_factor FT throughput, aligned via real timestamps ----
    co_first_wall = _results_first_wall(co_path) if co is not None else None
    anchor_wall = co_first_wall or _read_t0_wall(meta_path)
    if os.path.exists(bwd_path):
        src = ("co results real timestamp" if co_first_wall is not None
               else "bench_meta" if anchor_wall is not None
               else "first backward (legacy approx)")
        print(f"[compare_temporal] {gpu_name}: FT anchored via {src}.")

    t_max = 0.0
    for arr in t_ends:
        if len(arr):
            t_max = max(t_max, float(np.nanmax(arr)))

    ax_r = ax.twinx()
    ax_r.set_ylabel(YLABEL_FT, color=FT_SHADE_COLOR,
                    fontsize=FONTSIZE_AXIS_LABEL)
    ax_r.tick_params(axis="y", labelcolor=FT_SHADE_COLOR,
                     labelsize=FONTSIZE_TICK)
    ft = _ft_throughput_curve(bwd_path, anchor_wall, tl_base,
                              ft_offset=ft_offset, window_s=window_s)
    ft_peak = 0.0
    if ft is None:
        # No FT data: hide the empty twin so it doesn't draw a stray 0..1 axis.
        ax_r.set_yticks([])
    else:
        centers, tok_per_s, ft_last = ft
        ax_r.fill_between(centers, 0, tok_per_s, color=FT_SHADE_COLOR,
                          alpha=0.18, linewidth=0, zorder=1)
        ax_r.plot(centers, tok_per_s, color=FT_SHADE_COLOR, linewidth=1.7,
                  zorder=2)
        ft_peak = float(np.nanmax(tok_per_s)) if tok_per_s.size else 0.0
        t_max = max(t_max, ft_last)

    # ---- apply 1.3× headroom to BOTH y-axes ----
    lat_peaks = []
    if inf is not None:
        m = inf["ok"]
        if m.any():
            lat_peaks.append(float(np.nanmax(inf["latency_s"][m])))
    if co is not None:
        m = co["ok"]
        if m.any():
            lat_peaks.append(float(np.nanmax(co["latency_s"][m])))
    lat_peak = max((p for p in lat_peaks if np.isfinite(p) and p > 0),
                   default=0.0)
    if lat_peak > 0:
        ax.set_ylim(0, lat_peak * YMAX_HEADROOM)
    if ft_peak > 0:
        ax_r.set_ylim(0, ft_peak * YMAX_HEADROOM)
    else:
        ax_r.set_ylim(bottom=0)

    # ---- legend: three explicit entries (no auto-built scatter labels) ----
    #   1) vLLM (avg latency)
    #   2) DeltaServe-vLLM-Temp (avg latency, % overhead vs vLLM)
    #   3) Lowest 1% overhead: +/- X% — text-only entry, no marker
    handles, labels = [], []
    if inf is not None and np.isfinite(inf_avg):
        handles.append(_dot_handle(INF_COLOR))
        labels.append(f"{DISPLAY_NAME_INF} (avg {inf_avg:.3f}s)")
    if co is not None and np.isfinite(co_avg):
        avg_oh = _overhead_pct(co_avg, inf_avg)
        oh_str = (f", {avg_oh:+.1f}% vs {DISPLAY_NAME_INF}"
                  if np.isfinite(avg_oh) else "")
        handles.append(_dot_handle(CO_COLOR))
        labels.append(f"{DISPLAY_NAME_CO} (avg {co_avg:.3f}s{oh_str})")
    if (co is not None and inf is not None
            and np.isfinite(co_slow1) and np.isfinite(inf_slow1)):
        tail_oh = _overhead_pct(co_slow1, inf_slow1)
        if np.isfinite(tail_oh):
            handles.append(Line2D([0], [0], linestyle="none", marker=""))
            labels.append(f"{DISPLAY_NAME_TAIL_OVERHEAD}: {tail_oh:+.1f}%")
    if handles:
        ax.legend(handles, labels, loc="upper right",
                  fontsize=FONTSIZE_LEGEND)
    return t_max


def _dot_handle(color: str) -> Line2D:
    """Coloured-dot legend handle that matches the scatter style."""
    return Line2D([0], [0], marker="o", color="w",
                  markerfacecolor=color, markersize=8)


def build_figure(input_dir: str, mode: str, factor: float,
                 ft_offset: float = 0.0, window_s=None) -> plt.Figure:
    # constrained_layout (not tight_layout) so the per-panel twinx right axes
    # don't trip the "Axes not compatible with tight_layout" warning. Three
    # full-width rows: timeline, then one row per GPU.
    fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
    gs = GridSpec(3, 1, figure=fig, height_ratios=list(HEIGHT_RATIOS))

    # Row 0: scheduled request timeline (full width). The timeline file is the
    # x-axis reference: keep its native timestamps (first inference at e.g. 15s)
    # instead of normalizing to 0, and shift the latency series to match.
    ax_tl = fig.add_subplot(gs[0, 0])
    tl_path = os.path.join(input_dir, f"timeline_{mode}.csv")
    tl = _load_or_none(load_timeline, tl_path)
    tl_base = _timeline_base(tl_path) if tl is not None else 0.0
    if tl is not None:
        tl["t_rel_s"] = tl["t_rel_s"] + tl_base  # undo load_timeline's normalize
        plot_request_timeline(ax_tl, tl)
    else:
        ax_tl.text(0.5, 0.5, f"timeline_{mode}.csv\n(not found)",
                   ha="center", va="center", transform=ax_tl.transAxes,
                   fontsize=12, color="0.55")
        ax_tl.set_xticks([])
        ax_tl.set_yticks([])
        ax_tl.set_title("Scheduled Request Timeline (missing)")

    # Rows 1+2: one GPU per full-width row (5090, then A100). All inference
    # shifted into timeline coords by tl_base; FT anchored at the timeline
    # origin (+ ft_offset).
    gpu_axes = []
    t_max = float(tl["t_rel_s"].max()) if tl is not None and len(tl["t_rel_s"]) else 0.0
    for row, gpu in enumerate(GPUS, start=1):
        ax = fig.add_subplot(gs[row, 0])
        gpu_axes.append(ax)
        panel_tmax = plot_gpu_panel(
            ax, os.path.join(input_dir, gpu), gpu, mode, factor,
            tl_base=tl_base, ft_offset=ft_offset, window_s=window_s)
        t_max = max(t_max, panel_tmax)

    # Shared x-limit so request bursts line up vertically between the timeline
    # panel and the per-GPU panels (a twin shares its parent's x automatically).
    if t_max > 0:
        xlim = (0, t_max * 1.01)
        ax_tl.set_xlim(*xlim)
        for ax in gpu_axes:
            if ax.has_data():
                ax.set_xlim(*xlim)

    if SUPTITLE:
        fig.suptitle(SUPTITLE, fontsize=14)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default=DEFAULT_INPUT_DIR,
                    help="Result bundle dir. Default: eval/temporal_share_output.")
    ap.add_argument("--mode", default="nutanix",
                    help="Workload/timeline tag (file suffix). Default: nutanix.")
    ap.add_argument("--factor", type=float, default=0.0,
                    help="co_factor value to compare against inf-only. "
                         "Default: 0 (→ co_factor_0).")
    ap.add_argument("--window", type=float, default=None,
                    help="FT-throughput smoothing window (s). Default: auto "
                         "(~t_max/100, clamped 5–60s).")
    ap.add_argument("--ft-offset", type=float, default=0.0,
                    help="Extra manual shift (s) of the FT-throughput curve on "
                         "top of the real-timestamp anchor. Default 0 — with "
                         "--real-timestamp results the FT curve is already "
                         "aligned to inference exactly, so this is rarely "
                         "needed (kept for older runs lacking timestamps).")
    ap.add_argument("--output", default=None,
                    help="Output PNG path. Default: "
                         "<input-dir>/compare_temporal_<mode>.png.")
    args = ap.parse_args()

    if not os.path.isdir(args.input_dir):
        sys.exit(f"[compare_temporal] input dir not found: {args.input_dir}")

    out_path = args.output or os.path.join(
        args.input_dir, f"compare_temporal_{args.mode}.png")

    fig = build_figure(args.input_dir, args.mode, args.factor,
                       ft_offset=args.ft_offset, window_s=args.window)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=PNG_DPI)
    print(f"[compare_temporal] wrote figure → {out_path}")
    if GENERATE_PDF:
        # Mirror plot_multi_result / plot_full_timelines: no surrounding
        # margin so the PDF drops into a paper/poster context cleanly.
        pdf_path = os.path.splitext(out_path)[0] + ".pdf"
        fig.savefig(pdf_path, format="pdf",
                    bbox_inches="tight", pad_inches=0)
        print(f"[compare_temporal] wrote figure → {pdf_path}  (no-margin)")
    plt.close(fig)


if __name__ == "__main__":
    main()
