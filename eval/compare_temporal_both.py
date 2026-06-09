#!/usr/bin/env python3
"""compare_temporal_both.py — temporal-sharing comparison figure with the
unthrottled DeltaServe-vLLM run added.

Three-way version of ``compare_temporal.py``. Each per-GPU panel overlays
THREE inference E2E latency traces (plus two FT throughput traces) on a
shared time axis::

  * vLLM                  — inference-only baseline      (no co-serving)
  * DeltaServe-vLLM-Temp  — co_factor_<F> run            (throttled / temporal)
  * DeltaServe-vLLM       — co_factor_<F_full> run       (UNTHROTTLED, new)

For each co-serving series, the corresponding finetune-throughput curve
is also drawn on the right y-axis (orange for the throttled run, purple
for the unthrottled run by default).

Expected files per GPU subdir (``<mode>`` defaults to ``nutanix``,
``F`` from ``--factor 0`` → ``co_factor_0``, ``F_full`` from
``--factor-full off`` → ``co_factor_off`` by default — this matches the
``_factor_off`` naming convention auto_benchmark uses for the
unthrottled / "constraint disabled" run; pass ``--factor-full -1`` (or
any other value) if your bundle uses a different tag):

  timeline_results_<mode>.csv                    (vLLM, inf-only)
  timeline_results_co_factor_<F>_<mode>.csv      (Temp / throttled)
  timeline_results_co_factor_<F_full>_<mode>.csv (full / unthrottled)
  bwd_log_co_factor_<F>_<mode>.csv               (Temp FT throughput)
  bwd_log_co_factor_<F_full>_<mode>.csv          (full FT throughput)
  bench_meta_co_factor_<F>_<mode>.json           (optional anchors)
  bench_meta_co_factor_<F_full>_<mode>.json

Time alignment, the bwd_log anchor source, and the warmup-phase FT
handling are identical to ``compare_temporal.py`` — see that file's
header for the full reasoning. The only change here is the third trace.

By default both GPU panels are drawn (timeline + 5090 + A100 = 3 rows). Pass
``--5090`` or ``--A100`` to render only that GPU (timeline + one panel = 2
rows), dropping the other.

Usage:
  python eval/compare_temporal_both.py                                # both GPUs (3 rows)
  python eval/compare_temporal_both.py --5090                         # 5090 only (2 rows)
  python eval/compare_temporal_both.py --A100                         # A100 only (2 rows)
  python eval/compare_temporal_both.py --factor 0 --factor-full -1    # bundle uses _factor_-1
  python eval/compare_temporal_both.py --mode nutanix --factor 0
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
from matplotlib.patches import Patch

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

# Also reuse the small helpers from compare_temporal.py rather than
# re-implementing — keeps the time-alignment / stats math identical.
from compare_temporal import (  # noqa: E402
    _load_or_none,
    _timeline_base,
    _results_first_wall,
    _read_t0_wall,
    _latency_stats,
    _overhead_pct,
    _ft_throughput_curve,
)

DEFAULT_INPUT_DIR = os.path.join(_HERE, "temporal_share_output")
GPUS = ("5090", "A100")

# ============================================================================
# Settings — edit here to retitle / resize / restyle.
# ============================================================================

# ---- Output ----
GENERATE_PDF = True
PNG_DPI = 130

# ---- Figure ----
FIGSIZE = (15, 13)
HEIGHT_RATIOS = (1.0, 1.0, 1.0)
SUPTITLE = None                   # None disables the figure-level title

# Vertical gap between adjacent figure rows as a fraction of an average
# row height (matplotlib ``GridSpec.hspace``). Larger → bigger gap
# between the timeline panel and the first GPU panel (and between the
# two GPU panels when both are drawn). ``None`` lets constrained_layout
# auto-pick.
ROW_HSPACE = 0.05

# ---- Per-panel titles ----
PANEL_TITLE_TEMPLATE = "{gpu_name} — E2E latency & FT throughput"
# Display-name map for the panel title only. Keys are the directory
# names under ``input_dir/`` ("5090", "A100"); values are how the GPU
# is shown in the title text. Unknown keys fall through to the raw
# directory name.
GPU_DISPLAY_NAME = {
    "5090": "RTX 5090",
    "A100": "NVIDIA A100",
}

# ---- Axis labels ----
XLABEL = "Time (s)"
YLABEL_LATENCY = "E2E latency (s)"
YLABEL_FT = "FT throughput (tok/s)"

# ---- Y-axis headroom (per-GPU panels only) ----
YMAX_HEADROOM = 1.55

# ---- Display names (replace internal series ids in the legend) ----
DISPLAY_NAME_INF = "vLLM"
DISPLAY_NAME_CO_TEMP = "DeltaServe-vLLM-Temp"   # throttled run
DISPLAY_NAME_CO_FULL = "DeltaServe-vLLM"        # UNthrottled (new) run
DISPLAY_NAME_TAIL_OVERHEAD = "Lowest 1% overhead"

# ---- Font sizes ----
FONTSIZE_PANEL_TITLE = 20
FONTSIZE_AXIS_LABEL = 14
FONTSIZE_TICK = 14
FONTSIZE_LEGEND = 13

# ---- Font weights — matplotlib accepts "normal", "bold", "light", or a
# numeric value 0–1000. Drop these to "normal" to revert. ----
FONTWEIGHT_TITLE = "bold"
FONTWEIGHT_AXIS_LABEL = "bold"

# ---- Legend placement (per-GPU panels) ----
# Three-row layout: ``ncol = 2`` with a hidden spacer between the
# latency block and the FT-throughput block lays the entries out as::
#
#   Row 1: vLLM                 | DeltaServe-vLLM-Temp
#   Row 2: DeltaServe-vLLM      | (empty — spacer)
#   Row 3: Temp FT Throughput   | DeltaServe-vLLM FT Throughput
#
# i.e. the unthrottled DeltaServe-vLLM latency entry sits alone on
# row 2, separating the throttled / unthrottled latency lines from the
# FT-throughput comparison underneath. Drop ``LEGEND_NCOL`` back to 3
# (and remove the spacer in ``plot_gpu_panel``) to revert to the older
# 2-row layout. Matplotlib fills the legend column-major; the natural
# row-major handle order is fixed up via ``_row_major_reorder``.
LEGEND_LOC = "upper center"
LEGEND_BBOX_TO_ANCHOR = (0.5, 1.0)
LEGEND_NCOL = 2

# ---- Timeline panel (row 0) ----
# auto_plot.plot_request_timeline adds its own legend; set False to drop it.
SHOW_TIMELINE_LEGEND = False

# ---- Colors ----
INF_COLOR = "tab:blue"
CO_TEMP_COLOR = "tab:red"
CO_FULL_COLOR = "tab:green"

# FT throughput curves: orange for the Temp run (matches the single-series
# version), a contrasting purple for the unthrottled run so the two FT
# bands can be told apart on the shared right axis.
FT_TEMP_COLOR = FT_SHADE_COLOR     # "tab:orange" via auto_plot
FT_FULL_COLOR = "tab:purple"
FT_FILL_ALPHA = 0.15               # band alpha (applied to both FT fills)
FT_LINE_WIDTH = 1.7

# ---- Scatter style ----
SCATTER_SIZE = 10
SCATTER_ALPHA = 0.7
# Set ``SHOW_MEAN_LINE = True`` to overlay a dashed per-series mean
# (full-mean latency) on each scatter — disabled by default to keep the
# panel clean; the mean values still appear in the legend annotations.
SHOW_MEAN_LINE = False
MEAN_LINE_ALPHA = 0.5
MEAN_LINE_WIDTH = 1.0

# ============================================================================


def _format_factor_suffix(value: str) -> str:
    """Convert a CLI ``--factor*`` value to the file-name suffix segment.
    Accepts either a numeric string (``"-1"``, ``"0"``, ``"1.5"``) which
    gets formatted as ``f"{float(value):g}"`` (matching auto_benchmark's
    naming), or a literal tag (``"off"``) which is used verbatim.
    Returns the full prefix ``"_co_factor_<TAG>"``."""
    s = str(value).strip()
    try:
        return f"_co_factor_{float(s):g}"
    except ValueError:
        return f"_co_factor_{s}"


def _scatter_latency(ax, res, color: str, x_offset: float = 0.0,
                     mean_line: float = None) -> None:
    """Scatter E2E latency vs time on ``ax`` (x shifted by ``x_offset``
    into timeline coords) + an optional dashed mean line. No auto-legend
    label — the caller builds the legend manually so series can be
    renamed and overhead annotations can be added."""
    ok = res["ok"]
    t = res["t_rel_s"][ok] + x_offset
    lat = res["latency_s"][ok]
    ax.scatter(t, lat, s=SCATTER_SIZE, color=color, alpha=SCATTER_ALPHA,
               zorder=3)
    if (SHOW_MEAN_LINE and mean_line is not None
            and np.isfinite(mean_line)):
        ax.axhline(mean_line, color=color, linestyle="--",
                   linewidth=MEAN_LINE_WIDTH, alpha=MEAN_LINE_ALPHA, zorder=2)


def _dot_handle(color: str) -> Line2D:
    """Coloured-dot legend handle that matches the scatter style."""
    return Line2D([0], [0], marker="o", color="w",
                  markerfacecolor=color, markersize=8)


def _row_major_reorder(items, ncol):
    """Reorder ``items`` so matplotlib's column-major legend fill
    DISPLAYS them in row-major order.

    matplotlib fills ``ncol`` columns top-to-bottom (col 1 first, then
    col 2, …). With ``ncol = 3`` and 5 items the natural input
    ``[A, B, C, D, E]`` therefore renders as::

        A C E
        B D

    Re-permuting the input by column-then-row pulls the row-major
    visual back::

        A B C
        D E

    The transform is: for each column ``c`` walk down its rows ``r`` and
    pull ``items[r*ncol + c]``. Items shorter than ``nrow*ncol`` just
    leave the trailing slots empty (matplotlib renders them blank)."""
    n = len(items)
    if n == 0 or ncol <= 1:
        return list(items)
    nrow = (n + ncol - 1) // ncol
    out = []
    for c in range(ncol):
        for r in range(nrow):
            idx = r * ncol + c
            if idx < n:
                out.append(items[idx])
    return out


def _draw_ft_curve(ax_r, bwd_path: str, anchor_wall, tl_base: float,
                   ft_offset: float, window_s, color: str,
                   gpu_name: str, series_name: str):
    """Plot one FT throughput curve onto ``ax_r``. Returns
    ``(peak, mean)`` of the smoothed per-second curve in tok/s, both
    ``0.0`` when there's no data. The mean is over the curve's active
    range (centers >= 0), so it represents the average FT throughput
    during the inference window — directly comparable across runs."""
    if not os.path.exists(bwd_path):
        return 0.0, 0.0
    ft = _ft_throughput_curve(bwd_path, anchor_wall, tl_base,
                              ft_offset=ft_offset, window_s=window_s)
    if ft is None:
        return 0.0, 0.0
    centers, tok_per_s, _ft_last = ft
    ax_r.fill_between(centers, 0, tok_per_s, color=color,
                      alpha=FT_FILL_ALPHA, linewidth=0, zorder=1)
    ax_r.plot(centers, tok_per_s, color=color, linewidth=FT_LINE_WIDTH,
              zorder=2)
    peak = float(np.nanmax(tok_per_s)) if tok_per_s.size else 0.0
    mean = float(np.mean(tok_per_s)) if tok_per_s.size else 0.0
    print(f"[compare_temporal_both] {gpu_name}: FT {series_name}: "
          f"peak={peak:.0f} tok/s, mean={mean:.0f} tok/s, "
          f"t_last={_ft_last:.1f}s")
    return peak, mean


def plot_gpu_panel(ax, gpu_dir: str, gpu_name: str, mode: str,
                   factor_temp: str, factor_full: str,
                   tl_base: float = 0.0, ft_offset: float = 0.0,
                   window_s=None) -> float:
    """Three-series E2E latency overlay on the left axis + two FT-
    throughput curves on the right axis, all on the timeline file's
    clock. Returns the right-most x (0 if blank) so the caller can pin a
    shared x-limit."""
    fsuf_temp = _format_factor_suffix(factor_temp)
    fsuf_full = _format_factor_suffix(factor_full)

    inf_path = os.path.join(gpu_dir, f"timeline_results_{mode}.csv")
    co_temp_path = os.path.join(gpu_dir, f"timeline_results{fsuf_temp}_{mode}.csv")
    co_full_path = os.path.join(gpu_dir, f"timeline_results{fsuf_full}_{mode}.csv")
    bwd_temp_path = os.path.join(gpu_dir, f"bwd_log{fsuf_temp}_{mode}.csv")
    bwd_full_path = os.path.join(gpu_dir, f"bwd_log{fsuf_full}_{mode}.csv")
    meta_temp_path = os.path.join(gpu_dir, f"bench_meta{fsuf_temp}_{mode}.json")
    meta_full_path = os.path.join(gpu_dir, f"bench_meta{fsuf_full}_{mode}.json")

    inf = _load_or_none(load_results, inf_path)
    co_temp = _load_or_none(load_results, co_temp_path)
    co_full = _load_or_none(load_results, co_full_path)

    def _set_panel_title():
        if PANEL_TITLE_TEMPLATE:
            ax.set_title(PANEL_TITLE_TEMPLATE.format(
                gpu_name=GPU_DISPLAY_NAME.get(gpu_name, gpu_name)),
                         fontsize=FONTSIZE_PANEL_TITLE,
                         fontweight=FONTWEIGHT_TITLE)

    if inf is None and co_temp is None and co_full is None:
        ax.text(0.5, 0.5, f"{gpu_name}\n(no data)", ha="center", va="center",
                transform=ax.transAxes, fontsize=13, color="0.55")
        ax.set_xticks([])
        ax.set_yticks([])
        _set_panel_title()
        return 0.0

    # ---- left axis: three E2E latency scatter overlays ----
    t_ends = []
    inf_avg = inf_slow1 = float("nan")
    co_temp_avg = co_temp_slow1 = float("nan")
    co_full_avg = co_full_slow1 = float("nan")

    if inf is not None:
        inf_avg, _, _, inf_slow1, _ = _latency_stats(inf)
        _scatter_latency(ax, inf, INF_COLOR, x_offset=tl_base,
                         mean_line=inf_avg)
        m = inf["ok"]
        t_ends.append(inf["t_rel_s"][m] + inf["latency_s"][m] + tl_base)
    if co_temp is not None:
        co_temp_avg, _, _, co_temp_slow1, _ = _latency_stats(co_temp)
        _scatter_latency(ax, co_temp, CO_TEMP_COLOR, x_offset=tl_base,
                         mean_line=co_temp_avg)
        m = co_temp["ok"]
        t_ends.append(co_temp["t_rel_s"][m] + co_temp["latency_s"][m] + tl_base)
    if co_full is not None:
        co_full_avg, _, _, co_full_slow1, _ = _latency_stats(co_full)
        _scatter_latency(ax, co_full, CO_FULL_COLOR, x_offset=tl_base,
                         mean_line=co_full_avg)
        m = co_full["ok"]
        t_ends.append(co_full["t_rel_s"][m] + co_full["latency_s"][m] + tl_base)

    ax.set_xlabel(XLABEL, fontsize=FONTSIZE_AXIS_LABEL,
                  fontweight=FONTWEIGHT_AXIS_LABEL)
    ax.set_ylabel(YLABEL_LATENCY, fontsize=FONTSIZE_AXIS_LABEL,
                  fontweight=FONTWEIGHT_AXIS_LABEL)
    ax.tick_params(axis="both", labelsize=FONTSIZE_TICK)
    ax.set_ylim(bottom=0)
    _set_panel_title()
    ax.grid(True, alpha=0.25)

    # ---- right axis: TWO FT throughput curves ----
    # Anchor each curve to its OWN co-run's first-request wall clock
    # (read from that run's results CSV via --real-timestamp), falling
    # back to its own bench_meta, then first-backward anchoring. This
    # keeps the two curves on the same timeline-coords axis regardless
    # of whether the runs started at the same wall time.
    t_max = max((float(np.nanmax(arr)) for arr in t_ends if len(arr)),
                default=0.0)

    ax_r = ax.twinx()
    ax_r.set_ylabel(YLABEL_FT, color=FT_TEMP_COLOR,
                    fontsize=FONTSIZE_AXIS_LABEL,
                    fontweight=FONTWEIGHT_AXIS_LABEL)
    ax_r.tick_params(axis="y", labelcolor=FT_TEMP_COLOR,
                     labelsize=FONTSIZE_TICK)

    ft_peak = 0.0
    ft_temp_mean = 0.0
    ft_full_mean = 0.0
    if co_temp is not None:
        co_temp_first_wall = _results_first_wall(co_temp_path)
        anchor = co_temp_first_wall or _read_t0_wall(meta_temp_path)
        peak, ft_temp_mean = _draw_ft_curve(
            ax_r, bwd_temp_path, anchor, tl_base, ft_offset, window_s,
            FT_TEMP_COLOR, gpu_name, DISPLAY_NAME_CO_TEMP)
        ft_peak = max(ft_peak, peak)
    if co_full is not None:
        co_full_first_wall = _results_first_wall(co_full_path)
        anchor = co_full_first_wall or _read_t0_wall(meta_full_path)
        peak, ft_full_mean = _draw_ft_curve(
            ax_r, bwd_full_path, anchor, tl_base, ft_offset, window_s,
            FT_FULL_COLOR, gpu_name, DISPLAY_NAME_CO_FULL)
        ft_peak = max(ft_peak, peak)
    if ft_peak == 0.0:
        ax_r.set_yticks([])

    # ---- 1.3× headroom on BOTH y-axes ----
    lat_peaks = []
    for r in (inf, co_temp, co_full):
        if r is None:
            continue
        m = r["ok"]
        if m.any():
            lat_peaks.append(float(np.nanmax(r["latency_s"][m])))
    lat_peak = max((p for p in lat_peaks if np.isfinite(p) and p > 0),
                   default=0.0)
    if lat_peak > 0:
        ax.set_ylim(0, lat_peak * YMAX_HEADROOM)
    if ft_peak > 0:
        ax_r.set_ylim(0, ft_peak * YMAX_HEADROOM)
    else:
        ax_r.set_ylim(bottom=0)

    # ---- legend: up to five entries ----
    #   1) vLLM                                 (avg latency)
    #   2) DeltaServe-vLLM-Temp                 (avg latency, % vs vLLM)
    #   3) DeltaServe-vLLM                      (avg latency, % vs vLLM)
    #   4) FT throughput (DeltaServe-vLLM-Temp) (mean tok/s, % vs full)
    #   5) FT throughput (DeltaServe-vLLM)      (mean tok/s)
    handles, labels = [], []
    if inf is not None and np.isfinite(inf_avg):
        handles.append(_dot_handle(INF_COLOR))
        labels.append(f"{DISPLAY_NAME_INF} (avg {inf_avg:.3f}s)")
    if co_temp is not None and np.isfinite(co_temp_avg):
        parts = [f"avg {co_temp_avg:.3f}s"]
        avg_oh = _overhead_pct(co_temp_avg, inf_avg)
        if np.isfinite(avg_oh):
            # Drop the explicit "vs <baseline>" — vLLM is the obvious
            # reference and trimming the suffix tightens the legend.
            parts.append(f"{avg_oh:+.1f}%")
        handles.append(_dot_handle(CO_TEMP_COLOR))
        labels.append(f"{DISPLAY_NAME_CO_TEMP} ({', '.join(parts)})")
    if co_full is not None and np.isfinite(co_full_avg):
        parts = [f"avg {co_full_avg:.3f}s"]
        avg_oh = _overhead_pct(co_full_avg, inf_avg)
        if np.isfinite(avg_oh):
            parts.append(f"{avg_oh:+.1f}%")
        handles.append(_dot_handle(CO_FULL_COLOR))
        labels.append(f"{DISPLAY_NAME_CO_FULL} ({', '.join(parts)})")
    # Three-row layout: drop a hidden spacer between the latency block
    # and the FT-throughput block so ncol=2 lays out as
    #   row 1: vLLM | Temp
    #   row 2: full | <spacer>
    #   row 3: Temp FT | full FT
    # i.e. the unthrottled DeltaServe-vLLM latency entry sits alone on
    # row 2. Drop the spacer (and set LEGEND_NCOL back to 3) to revert
    # to the older 2-row layout.
    if LEGEND_NCOL == 2 and len(handles) == 3:
        handles.append(Line2D([0], [0], linestyle="none", marker=""))
        labels.append("")
    # Dedicated FT-throughput entries — coloured patches match the
    # filled bands on the right axis so the legend doubles as a curve
    # key. The relative comparison (full vs Temp) lives on the
    # unthrottled entry so it reads as a positive gain ("how much more
    # FT throughput the unthrottled run delivers vs the Temp baseline").
    if ft_temp_mean > 0:
        handles.append(Patch(facecolor=FT_TEMP_COLOR, alpha=FT_FILL_ALPHA,
                             edgecolor=FT_TEMP_COLOR))
        labels.append(
            f"{DISPLAY_NAME_CO_TEMP} FT Throughput: {ft_temp_mean:.0f} tok/s")
    if ft_full_mean > 0:
        label = (f"{DISPLAY_NAME_CO_FULL} FT Throughput: "
                 f"{ft_full_mean:.0f} tok/s")
        if ft_temp_mean > 0:
            delta_pct = (ft_full_mean - ft_temp_mean) / ft_temp_mean * 100.0
            label += f" ({delta_pct:+.1f}%)"
        handles.append(Patch(facecolor=FT_FULL_COLOR, alpha=FT_FILL_ALPHA,
                             edgecolor=FT_FULL_COLOR))
        labels.append(label)
    if handles:
        legend_kwargs = dict(loc=LEGEND_LOC, fontsize=FONTSIZE_LEGEND,
                             ncol=LEGEND_NCOL)
        if LEGEND_BBOX_TO_ANCHOR is not None:
            legend_kwargs["bbox_to_anchor"] = LEGEND_BBOX_TO_ANCHOR
        # matplotlib fills the legend column-major; permute so the
        # visual ends up row-major. For LEGEND_NCOL=2 (3-row layout) a
        # hidden spacer above pads the latency block to 4 entries so
        # the FT-throughput pair lands cleanly on row 3.
        handles = _row_major_reorder(handles, LEGEND_NCOL)
        labels = _row_major_reorder(labels, LEGEND_NCOL)
        ax.legend(handles, labels, **legend_kwargs)
    return t_max


def build_figure(input_dir: str, mode: str,
                 factor_temp: str, factor_full: str,
                 ft_offset: float = 0.0, window_s=None,
                 gpus=GPUS) -> plt.Figure:
    # Rows: timeline (HEIGHT_RATIOS[0]) + 1.0 per GPU in ``gpus``. Scale the
    # figure height so each row keeps its default per-row height regardless of
    # how many GPUs are shown (one GPU => 2 rows, both => 3).
    gpus = tuple(gpus)
    height_ratios = [HEIGHT_RATIOS[0]] + [1.0] * len(gpus)
    row_unit = FIGSIZE[1] / sum(HEIGHT_RATIOS)
    fig = plt.figure(figsize=(FIGSIZE[0], row_unit * sum(height_ratios)),
                     constrained_layout=True)
    gs_kwargs = dict(figure=fig, height_ratios=height_ratios)
    if ROW_HSPACE is not None:
        gs_kwargs["hspace"] = ROW_HSPACE
    gs = GridSpec(1 + len(gpus), 1, **gs_kwargs)

    # Row 0: scheduled request timeline (full width). The timeline file
    # is the x-axis reference: keep its native timestamps (first inference
    # at e.g. 15s) instead of normalizing to 0, and shift the latency
    # series to match.
    ax_tl = fig.add_subplot(gs[0, 0])
    tl_path = os.path.join(input_dir, f"timeline_{mode}.csv")
    tl = _load_or_none(load_timeline, tl_path)
    tl_base = _timeline_base(tl_path) if tl is not None else 0.0
    if tl is not None:
        tl["t_rel_s"] = tl["t_rel_s"] + tl_base
        plot_request_timeline(ax_tl, tl)
        # Drop the timeline panel's auto-legend (auto_plot adds one on ax_tl).
        if not SHOW_TIMELINE_LEGEND:
            leg = ax_tl.get_legend()
            if leg is not None:
                leg.remove()
        # auto_plot.plot_request_timeline sets the title + axis labels
        # with matplotlib's defaults — re-style each artist in place so
        # this panel matches the per-GPU panels below (bold + tuned size).
        if ax_tl.get_title():
            ax_tl.title.set_fontsize(FONTSIZE_PANEL_TITLE)
            ax_tl.title.set_fontweight(FONTWEIGHT_TITLE)
        ax_tl.xaxis.label.set_fontsize(FONTSIZE_AXIS_LABEL)
        ax_tl.xaxis.label.set_fontweight(FONTWEIGHT_AXIS_LABEL)
        ax_tl.yaxis.label.set_fontsize(FONTSIZE_AXIS_LABEL)
        ax_tl.yaxis.label.set_fontweight(FONTWEIGHT_AXIS_LABEL)
        # auto_plot uses matplotlib default tick sizes — match the per-GPU
        # panels' FONTSIZE_TICK on the timeline panel too.
        ax_tl.tick_params(axis="both", labelsize=FONTSIZE_TICK)
        # plot_request_timeline draws a tokens/s twin axis (right side)
        # — find it on the figure and bold its y label + size its ticks too.
        # Twin axes share the parent's x range, which is the cheapest way to
        # match them without auto_plot exposing the handle.
        for child in ax_tl.figure.axes:
            if child is ax_tl:
                continue
            if (child.get_position().bounds == ax_tl.get_position().bounds
                    and child.yaxis.label.get_text()):
                child.yaxis.label.set_fontsize(FONTSIZE_AXIS_LABEL)
                child.yaxis.label.set_fontweight(FONTWEIGHT_AXIS_LABEL)
                child.tick_params(axis="y", labelsize=FONTSIZE_TICK)
    else:
        ax_tl.text(0.5, 0.5, f"timeline_{mode}.csv\n(not found)",
                   ha="center", va="center", transform=ax_tl.transAxes,
                   fontsize=12, color="0.55")
        ax_tl.set_xticks([])
        ax_tl.set_yticks([])
        ax_tl.set_title("Scheduled Request Timeline (missing)",
                        fontsize=FONTSIZE_PANEL_TITLE,
                        fontweight=FONTWEIGHT_TITLE)

    # Rows 1..N: one GPU per full-width row (whichever GPUs are in ``gpus``).
    gpu_axes = []
    t_max = float(tl["t_rel_s"].max()) if tl is not None and len(tl["t_rel_s"]) else 0.0
    for row, gpu in enumerate(gpus, start=1):
        ax = fig.add_subplot(gs[row, 0])
        gpu_axes.append(ax)
        panel_tmax = plot_gpu_panel(
            ax, os.path.join(input_dir, gpu), gpu, mode,
            factor_temp=factor_temp, factor_full=factor_full,
            tl_base=tl_base, ft_offset=ft_offset, window_s=window_s)
        t_max = max(t_max, panel_tmax)

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
    ap.add_argument("--factor", default="0",
                    help="co_factor for the throttled (DeltaServe-vLLM-Temp) "
                         "run. Default: 0 (→ co_factor_0).")
    ap.add_argument("--factor-full", default="off",
                    help="co_factor for the unthrottled (DeltaServe-vLLM) "
                         "run. Default: off (→ co_factor_off, matches "
                         "auto_benchmark's _factor_off naming for the "
                         "'admission constraint disabled' run). Pass a "
                         "numeric value (e.g. '-1', '1.5') if your bundle "
                         "uses _factor_<N> instead.")
    ap.add_argument("--window", type=float, default=None,
                    help="FT-throughput smoothing window (s). Default: auto "
                         "(~t_max/100, clamped 5–60s).")
    ap.add_argument("--ft-offset", type=float, default=0.0,
                    help="Extra manual shift (s) of the FT-throughput curves "
                         "on top of the real-timestamp anchor. Default 0.")
    ap.add_argument("--output", default=None,
                    help="Output PNG path. Default: "
                         "<input-dir>/compare_temporal_both[_<gpu>]_<mode>.png.")
    # GPU selection: with --5090 or --A100 the figure shows ONLY that GPU
    # (timeline + one panel = 2 rows), dropping the other. With neither, both
    # GPUs are shown (timeline + 2 panels = 3 rows), as before.
    gpu_sel = ap.add_mutually_exclusive_group()
    gpu_sel.add_argument("--5090", dest="gpu", action="store_const",
                         const="5090",
                         help="Plot only the 5090 panel (drop A100).")
    gpu_sel.add_argument("--A100", dest="gpu", action="store_const",
                         const="A100",
                         help="Plot only the A100 panel (drop 5090).")
    args = ap.parse_args()

    if not os.path.isdir(args.input_dir):
        sys.exit(f"[compare_temporal_both] input dir not found: {args.input_dir}")

    gpus = (args.gpu,) if args.gpu else GPUS
    gpu_tag = f"_{args.gpu}" if args.gpu else ""
    out_path = args.output or os.path.join(
        args.input_dir, f"compare_temporal_both{gpu_tag}_{args.mode}.png")

    fig = build_figure(args.input_dir, args.mode,
                       factor_temp=args.factor,
                       factor_full=args.factor_full,
                       ft_offset=args.ft_offset, window_s=args.window,
                       gpus=gpus)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=PNG_DPI)
    print(f"[compare_temporal_both] wrote figure → {out_path}")
    if GENERATE_PDF:
        pdf_path = os.path.splitext(out_path)[0] + ".pdf"
        fig.savefig(pdf_path, format="pdf",
                    bbox_inches="tight", pad_inches=0)
        print(f"[compare_temporal_both] wrote figure → {pdf_path}  (no-margin)")
    plt.close(fig)


if __name__ == "__main__":
    main()
