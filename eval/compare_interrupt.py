#!/usr/bin/env python3
"""compare_interrupt.py — interrupt-vs-no-interrupt comparison figure (5090 only).

Mirrors ``compare_temporal.py`` / ``compare_temporal_both.py`` but pares
the figure down to a single per-GPU panel for the 5090 box and drops the
scheduled-request-timeline strip. The two co-serving series being
compared on that panel:

  * vLLM                            — inference-only baseline
  * DeltaServe-vLLM-Temp             — throttled co-serving with
                                       ``finetune.forward_interruptible = True``
                                       (the default; "interrupt true")
  * DeltaServe-vLLM-Temp-No-Interrupt — throttled co-serving with
                                       ``finetune.forward_interruptible = False``

E2E latency for all three sits on the left y-axis; the two co-serving
runs' FT throughput curves go on the right y-axis (orange for the
"interrupt true" run, purple for the no-interrupt run by default).

Expected files under ``eval/interrupt_output/5090/`` (``<mode>`` defaults
to ``nutanix``):

  timeline_results_<mode>.csv                            (vLLM, inf-only)
  timeline_results_interrupt_true_<mode>.csv             (Temp / interrupt=true)
  bwd_log_interrupt_true_<mode>.csv                      (Temp FT throughput)
  timeline_results_co_interrupt_false_<mode>.csv         (Temp-No-Interrupt)
  bwd_log_co_factor_interrupt_false_<mode>.csv           (Temp-No-Interrupt FT)
  bench_meta_*_<mode>.json                               (optional anchors)

The file-name shape is irregular (the no-interrupt bwd_log carries an
extra ``co_factor_`` segment that the timeline file lacks) — the file
templates at the top of this module make every path overridable so
future runs can rename without code changes.

Time alignment, the bwd_log anchor source, the warmup-phase FT handling,
and the legend / annotation helpers are imported from
``compare_temporal_both.py`` (and via that, ``compare_temporal.py``) so
the math stays identical across all three scripts.

Usage:
  python eval/compare_interrupt.py                       # nutanix, 5090
  python eval/compare_interrupt.py --mode nutanix
  python eval/compare_interrupt.py --output /tmp/x.png
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

# Reuse auto_plot.py helpers + compare_temporal.py + compare_temporal_both.py.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from auto_plot import (  # noqa: E402
    FT_SHADE_COLOR,
    load_results,
)
from compare_temporal import (  # noqa: E402
    _load_or_none,
    _results_first_wall,
    _read_t0_wall,
    _latency_stats,
    _overhead_pct,
    _timeline_base,
)
from compare_temporal_both import (  # noqa: E402
    _scatter_latency,
    _dot_handle,
    _draw_ft_curve,
    _row_major_reorder,
)

DEFAULT_INPUT_DIR = os.path.join(_HERE, "interrupt_output")
GPU = "5090"
GPU_DISPLAY_NAME = "RTX 5090"      # how the GPU is shown in the panel title

# ============================================================================
# Settings — edit here to retitle / resize / restyle.
# ============================================================================

# ---- Output ----
GENERATE_PDF = True
PNG_DPI = 130

# ---- Figure ----
FIGSIZE = (15, 4.5)              # single panel; height is enough for legend + data
SUPTITLE = None                  # None disables the figure-level title
XMAX = 40.0                      # x-axis upper limit (s); None = full timeline span
                                 # (overridable with --xmax)

# ---- Per-panel title ----
PANEL_TITLE = f"{GPU_DISPLAY_NAME} — E2E latency & FT throughput"
# Pass None to drop the panel header.

# ---- Axis labels ----
XLABEL = "Time (s)"
YLABEL_LATENCY = "E2E latency (s)"
YLABEL_FT = "FT throughput (tok/s)"

# ---- Y-axis headroom ----
YMAX_HEADROOM = 1.55

# ---- Y-axis tick spacing (None = matplotlib auto) ----
YTICK_LATENCY = 0.4       # left axis (E2E latency, s): a tick every 0.4
YTICK_FT = 500            # right axis (FT throughput, tok/s): a tick every 500

# ---- Display names (replace internal series ids in the legend) ----
DISPLAY_NAME_INF = "vLLM"
DISPLAY_NAME_CO_TEMP = "DeltaServe-vLLM-Temp"
DISPLAY_NAME_CO_TEMP_NOINT = "DeltaServe-vLLM-No-Interrupt"

# ---- File-name templates ----
# ``{mode}`` is the workload tag (e.g. "nutanix"). Edit individual
# entries if your bundle uses a different naming convention.
FILE_INF_RESULTS = "timeline_results_{mode}.csv"
FILE_TEMP_RESULTS = "timeline_results_interrupt_true_{mode}.csv"
FILE_TEMP_BWD = "bwd_log_interrupt_true_{mode}.csv"
FILE_TEMP_META = "bench_meta_interrupt_true_{mode}.json"
FILE_NOINT_RESULTS = "timeline_results_co_interrupt_false_{mode}.csv"
FILE_NOINT_BWD = "bwd_log_co_factor_interrupt_false_{mode}.csv"
FILE_NOINT_META = "bench_meta_co_factor_interrupt_false_{mode}.json"

# ---- Font sizes ----
FONTSIZE_PANEL_TITLE = 20
FONTSIZE_AXIS_LABEL = 14
FONTSIZE_TICK = 14
FONTSIZE_LEGEND = 13

# ---- Font weights ----
FONTWEIGHT_TITLE = "bold"
FONTWEIGHT_AXIS_LABEL = "bold"

# ---- Legend placement ----
# Two legend boxes on the same axes:
#
#   TOP (latency) — 3 rows x 2 cols, anchored at the top edge of the
#   panel where the legend has always sat:
#
#       Row 1: vLLM                  | (blank)
#       Row 2: DeltaServe-vLLM-Temp  | 5% tail overhead vs vLLM
#       Row 3: DeltaServe-vLLM-No-Interrupt | 5% tail overhead vs vLLM
#
#   The col-2 entries on rows 2/3 share the row's dot color so the eye
#   pairs them with the system on col 1. Row 1 col 2 is a hidden
#   ``Line2D`` spacer (no marker, empty label). Matplotlib fills the
#   legend column-major; the natural row-major handle order is fixed
#   up by ``_row_major_reorder`` at render time.
#
#   BOTTOM (FT throughput) — 1 row x 2 cols, anchored at the bottom
#   center of the panel. ``framealpha`` keeps it readable when the
#   coloured FT band sits underneath. Drop the bbox y-coord below 0 to
#   push the box outside the axes (e.g. ``(0.5, -0.18)``).
LEGEND_LOC = "upper center"               # top (latency) box
LEGEND_BBOX_TO_ANCHOR = (0.5, 1.0)
LEGEND_NCOL = 2

LEGEND_FT_LOC = "lower center"            # bottom (FT throughput) box
LEGEND_FT_BBOX_TO_ANCHOR = (0.5, 0.0)
LEGEND_FT_NCOL = 2
LEGEND_FT_FRAMEALPHA = 0.9

# ---- Colors ----
INF_COLOR = "tab:blue"
CO_TEMP_COLOR = "tab:red"
CO_NOINT_COLOR = "tab:green"

# FT throughput curves: orange for the Temp run, purple for the
# no-interrupt run so the two bands are distinct on the shared right
# axis.
FT_TEMP_COLOR = FT_SHADE_COLOR     # "tab:orange" via auto_plot
FT_NOINT_COLOR = "tab:purple"
FT_FILL_ALPHA = 0.15
FT_LINE_WIDTH = 1.7

# ---- Scatter style ----
SCATTER_SIZE = 10
SCATTER_ALPHA = 0.7
SHOW_MEAN_LINE = False             # set True to overlay dashed per-series mean
MEAN_LINE_ALPHA = 0.5
MEAN_LINE_WIDTH = 1.0

# ============================================================================


def _slow_mean(res, pct: float = 5.0) -> float:
    """Mean of the slowest ``pct``% of ok & finite E2E latencies in
    ``res`` — i.e. the tail-overhead reference. Returns ``nan`` when
    there's no data. Generalises ``compare_temporal._latency_stats``'s
    ``slow1`` (which is fixed at 1%)."""
    ok = res["ok"]
    lat = res["latency_s"][ok]
    lat = lat[np.isfinite(lat)]
    if lat.size == 0:
        return float("nan")
    cutoff = float(np.percentile(lat, 100.0 - pct))
    tail = lat[lat >= cutoff]
    if tail.size == 0:
        return float(lat.max())
    return float(tail.mean())


def plot_panel(ax, gpu_dir: str, mode: str,
               tl_base: float = 0.0,
               ft_offset: float = 0.0, window_s=None) -> float:
    """Single-panel rendering: 3 latency scatters on the left axis +
    2 FT throughput curves on the right axis. Returns the panel's
    right-most x so the caller can pin a shared x-limit (vestigial
    since we only draw one panel, but it keeps the code shape close
    to compare_temporal_both)."""
    inf_path = os.path.join(gpu_dir, FILE_INF_RESULTS.format(mode=mode))
    temp_path = os.path.join(gpu_dir, FILE_TEMP_RESULTS.format(mode=mode))
    temp_bwd = os.path.join(gpu_dir, FILE_TEMP_BWD.format(mode=mode))
    temp_meta = os.path.join(gpu_dir, FILE_TEMP_META.format(mode=mode))
    noint_path = os.path.join(gpu_dir, FILE_NOINT_RESULTS.format(mode=mode))
    noint_bwd = os.path.join(gpu_dir, FILE_NOINT_BWD.format(mode=mode))
    noint_meta = os.path.join(gpu_dir, FILE_NOINT_META.format(mode=mode))

    inf = _load_or_none(load_results, inf_path)
    temp = _load_or_none(load_results, temp_path)
    noint = _load_or_none(load_results, noint_path)

    def _set_panel_title():
        if PANEL_TITLE:
            ax.set_title(PANEL_TITLE,
                         fontsize=FONTSIZE_PANEL_TITLE,
                         fontweight=FONTWEIGHT_TITLE)

    if inf is None and temp is None and noint is None:
        ax.text(0.5, 0.5, f"{GPU}\n(no data)", ha="center", va="center",
                transform=ax.transAxes, fontsize=13, color="0.55")
        ax.set_xticks([])
        ax.set_yticks([])
        _set_panel_title()
        return 0.0

    # ``tl_base`` is the first scheduled request's offset (the smallest
    # ``timestamp_s`` in the timeline file — e.g. 5.0s when the schedule
    # starts at second 5). The benchmark strips it when normalizing
    # latency ``t_rel_s`` to start at 0; we add it back so the scatter
    # and FT curves sit on the same clock the requests were actually
    # scheduled against. Caller is responsible for loading the timeline
    # and computing the offset — we don't draw the timeline strip here.

    # ---- left axis: three latency scatters ----
    t_ends = []
    inf_avg = inf_slow5 = float("nan")
    temp_avg = temp_slow5 = float("nan")
    noint_avg = noint_slow5 = float("nan")

    if inf is not None:
        inf_avg, _, _, _, _ = _latency_stats(inf)
        inf_slow5 = _slow_mean(inf, 5.0)
        _scatter_latency(ax, inf, INF_COLOR, x_offset=tl_base,
                         mean_line=inf_avg)
        m = inf["ok"]
        t_ends.append(inf["t_rel_s"][m] + inf["latency_s"][m] + tl_base)
    if temp is not None:
        temp_avg, _, _, _, _ = _latency_stats(temp)
        temp_slow5 = _slow_mean(temp, 5.0)
        _scatter_latency(ax, temp, CO_TEMP_COLOR, x_offset=tl_base,
                         mean_line=temp_avg)
        m = temp["ok"]
        t_ends.append(temp["t_rel_s"][m] + temp["latency_s"][m] + tl_base)
    if noint is not None:
        noint_avg, _, _, _, _ = _latency_stats(noint)
        noint_slow5 = _slow_mean(noint, 5.0)
        _scatter_latency(ax, noint, CO_NOINT_COLOR, x_offset=tl_base,
                         mean_line=noint_avg)
        m = noint["ok"]
        t_ends.append(noint["t_rel_s"][m] + noint["latency_s"][m] + tl_base)

    ax.set_xlabel(XLABEL, fontsize=FONTSIZE_AXIS_LABEL,
                  fontweight=FONTWEIGHT_AXIS_LABEL)
    ax.set_ylabel(YLABEL_LATENCY, fontsize=FONTSIZE_AXIS_LABEL,
                  fontweight=FONTWEIGHT_AXIS_LABEL)
    ax.tick_params(axis="both", labelsize=FONTSIZE_TICK)
    ax.set_ylim(bottom=0)
    _set_panel_title()
    ax.grid(True, alpha=0.25)

    # ---- right axis: two FT throughput curves ----
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
    ft_noint_mean = 0.0
    if temp is not None:
        anchor = _results_first_wall(temp_path) or _read_t0_wall(temp_meta)
        peak, ft_temp_mean = _draw_ft_curve(
            ax_r, temp_bwd, anchor, tl_base, ft_offset, window_s,
            FT_TEMP_COLOR, GPU, DISPLAY_NAME_CO_TEMP)
        ft_peak = max(ft_peak, peak)
    if noint is not None:
        anchor = _results_first_wall(noint_path) or _read_t0_wall(noint_meta)
        peak, ft_noint_mean = _draw_ft_curve(
            ax_r, noint_bwd, anchor, tl_base, ft_offset, window_s,
            FT_NOINT_COLOR, GPU, DISPLAY_NAME_CO_TEMP_NOINT)
        ft_peak = max(ft_peak, peak)
    if ft_peak == 0.0:
        ax_r.set_yticks([])

    # ---- 1.55× headroom on both axes ----
    lat_peaks = []
    for r in (inf, temp, noint):
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

    # ---- y-axis tick spacing ----
    if YTICK_LATENCY:
        ax.yaxis.set_major_locator(MultipleLocator(YTICK_LATENCY))
    if YTICK_FT and ft_peak > 0:
        ax_r.yaxis.set_major_locator(MultipleLocator(YTICK_FT))

    # ---- legends ----
    # Top box (3 rows x 2 cols, ROW-major intent):
    #   Row 1: vLLM                            | (blank spacer)
    #   Row 2: DeltaServe-vLLM-Temp            | 5% tail overhead vs vLLM
    #   Row 3: DeltaServe-vLLM-No-Interrupt    | 5% tail overhead vs vLLM
    # The col-2 dot on rows 2/3 mirrors the row's system colour so the
    # eye pairs the tail-overhead entry with the latency-avg entry.
    def _spacer():
        return Line2D([0], [0], linestyle="none", marker="")

    def _tail_label(co_slow5):
        """Format the 5% tail-overhead entry for one co-serving system.
        Falls back to the spacer when vLLM's tail or the system's tail
        is missing so the row doesn't carry a half-meaningful number."""
        if not (np.isfinite(co_slow5) and np.isfinite(inf_slow5)):
            return None
        oh = _overhead_pct(co_slow5, inf_slow5)
        if not np.isfinite(oh):
            return None
        return f"5% tail {co_slow5:.3f}s ({oh:+.1f}% vs vLLM)"

    lat_handles, lat_labels = [], []
    # --- Row 1: vLLM + blank spacer
    if inf is not None and np.isfinite(inf_avg):
        lat_handles.append(_dot_handle(INF_COLOR))
        lat_labels.append(f"{DISPLAY_NAME_INF} (avg {inf_avg:.3f}s)")
        lat_handles.append(_spacer())
        lat_labels.append("")
    # --- Row 2: Temp + Temp's 5% tail overhead
    if temp is not None and np.isfinite(temp_avg):
        parts = [f"avg {temp_avg:.3f}s"]
        avg_oh = _overhead_pct(temp_avg, inf_avg)
        if np.isfinite(avg_oh):
            parts.append(f"{avg_oh:+.1f}%")
        lat_handles.append(_dot_handle(CO_TEMP_COLOR))
        lat_labels.append(f"{DISPLAY_NAME_CO_TEMP} ({', '.join(parts)})")
        tail_lbl = _tail_label(temp_slow5)
        if tail_lbl is not None:
            lat_handles.append(_dot_handle(CO_TEMP_COLOR))
            lat_labels.append(tail_lbl)
        else:
            lat_handles.append(_spacer())
            lat_labels.append("")
    # --- Row 3: No-Interrupt + No-Interrupt's 5% tail overhead
    if noint is not None and np.isfinite(noint_avg):
        parts = [f"avg {noint_avg:.3f}s"]
        avg_oh = _overhead_pct(noint_avg, inf_avg)
        if np.isfinite(avg_oh):
            parts.append(f"{avg_oh:+.1f}%")
        lat_handles.append(_dot_handle(CO_NOINT_COLOR))
        lat_labels.append(f"{DISPLAY_NAME_CO_TEMP_NOINT} ({', '.join(parts)})")
        tail_lbl = _tail_label(noint_slow5)
        if tail_lbl is not None:
            lat_handles.append(_dot_handle(CO_NOINT_COLOR))
            lat_labels.append(tail_lbl)
        else:
            lat_handles.append(_spacer())
            lat_labels.append("")

    # Bottom box (1 row x 2 cols): FT throughput entries — coloured
    # patches matching the right-axis bands. Temp shows its absolute
    # mean; No-Interrupt reports the relative delta vs Temp so the cost
    # of disabling forward_interruptible reads at a glance.
    ft_handles, ft_labels = [], []
    if ft_temp_mean > 0:
        ft_handles.append(Patch(facecolor=FT_TEMP_COLOR,
                                alpha=FT_FILL_ALPHA,
                                edgecolor=FT_TEMP_COLOR))
        ft_labels.append(
            f"{DISPLAY_NAME_CO_TEMP} FT Throughput: "
            f"{ft_temp_mean:.0f} tok/s")
    if ft_noint_mean > 0:
        label = (f"{DISPLAY_NAME_CO_TEMP_NOINT} FT Throughput: "
                 f"{ft_noint_mean:.0f} tok/s")
        if ft_temp_mean > 0:
            delta_pct = (ft_noint_mean - ft_temp_mean) / ft_temp_mean * 100.0
            label += f" ({delta_pct:+.1f}%)"
        ft_handles.append(Patch(facecolor=FT_NOINT_COLOR,
                                alpha=FT_FILL_ALPHA,
                                edgecolor=FT_NOINT_COLOR))
        ft_labels.append(label)

    # Render the top (latency) legend first; ``ax.add_artist`` keeps
    # it on the axes when the next ``ax.legend`` call would otherwise
    # replace it.
    if lat_handles:
        lat_kwargs = dict(loc=LEGEND_LOC, fontsize=FONTSIZE_LEGEND,
                          ncol=LEGEND_NCOL)
        if LEGEND_BBOX_TO_ANCHOR is not None:
            lat_kwargs["bbox_to_anchor"] = LEGEND_BBOX_TO_ANCHOR
        lat_h = _row_major_reorder(lat_handles, LEGEND_NCOL)
        lat_l = _row_major_reorder(lat_labels, LEGEND_NCOL)
        leg_lat = ax.legend(lat_h, lat_l, **lat_kwargs)
        ax.add_artist(leg_lat)
    if ft_handles:
        ft_kwargs = dict(loc=LEGEND_FT_LOC, fontsize=FONTSIZE_LEGEND,
                         ncol=LEGEND_FT_NCOL,
                         framealpha=LEGEND_FT_FRAMEALPHA)
        if LEGEND_FT_BBOX_TO_ANCHOR is not None:
            ft_kwargs["bbox_to_anchor"] = LEGEND_FT_BBOX_TO_ANCHOR
        ft_h = _row_major_reorder(ft_handles, LEGEND_FT_NCOL)
        ft_l = _row_major_reorder(ft_labels, LEGEND_FT_NCOL)
        ax.legend(ft_h, ft_l, **ft_kwargs)
    return t_max


def build_figure(input_dir: str, mode: str,
                 ft_offset: float = 0.0, window_s=None, xmax=XMAX) -> plt.Figure:
    fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
    gs = GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    # Timeline file is shared across GPUs at the bundle root (same shape
    # as compare_temporal_both). Use its first ``timestamp_s`` as the
    # offset so latency / FT curves line up with the requests'
    # scheduled arrival times, not the benchmark's t=0. The timeline
    # itself is not drawn — only its first-timestamp value is read.
    tl_path = os.path.join(input_dir, f"timeline_{mode}.csv")
    tl_base = _timeline_base(tl_path) if os.path.isfile(tl_path) else 0.0
    t_max = plot_panel(ax, os.path.join(input_dir, GPU), mode,
                       tl_base=tl_base,
                       ft_offset=ft_offset, window_s=window_s)
    if xmax is not None:
        ax.set_xlim(0, xmax)
    elif t_max > 0:
        ax.set_xlim(0, t_max * 1.01)
    if SUPTITLE:
        fig.suptitle(SUPTITLE, fontsize=14)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default=DEFAULT_INPUT_DIR,
                    help="Result bundle dir. Default: eval/interrupt_output.")
    ap.add_argument("--mode", default="nutanix",
                    help="Workload/timeline tag (file suffix). Default: nutanix.")
    ap.add_argument("--window", type=float, default=None,
                    help="FT-throughput smoothing window (s). Default: auto "
                         "(~t_max/100, clamped 5–60s).")
    ap.add_argument("--ft-offset", type=float, default=0.0,
                    help="Extra manual shift (s) of the FT-throughput curves "
                         "on top of the real-timestamp anchor. Default 0.")
    ap.add_argument("--xmax", type=float, default=XMAX,
                    help=f"x-axis upper limit in seconds (default {XMAX}); "
                         "pass a negative value for the full timeline span.")
    ap.add_argument("--output", default=None,
                    help="Output PNG path. Default: "
                         "<input-dir>/compare_interrupt_<mode>.png.")
    args = ap.parse_args()

    if not os.path.isdir(args.input_dir):
        sys.exit(f"[compare_interrupt] input dir not found: {args.input_dir}")

    out_path = args.output or os.path.join(
        args.input_dir, f"compare_interrupt_{args.mode}.png")

    # Negative --xmax means "full span" (xmax=None).
    xmax = None if (args.xmax is not None and args.xmax < 0) else args.xmax
    fig = build_figure(args.input_dir, args.mode,
                       ft_offset=args.ft_offset, window_s=args.window, xmax=xmax)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=PNG_DPI)
    print(f"[compare_interrupt] wrote figure → {out_path}")
    if GENERATE_PDF:
        pdf_path = os.path.splitext(out_path)[0] + ".pdf"
        fig.savefig(pdf_path, format="pdf",
                    bbox_inches="tight", pad_inches=0)
        print(f"[compare_interrupt] wrote figure → {pdf_path}  (no-margin)")
    plt.close(fig)


if __name__ == "__main__":
    main()
