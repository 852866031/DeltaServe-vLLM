#!/usr/bin/env python3
"""plot_production.py — recreate the two-panel production-workload figure:

  • TOP  — "Production Inference Workload Trace": Tokens/sec (left axis) and
    Requests/sec (right axis) vs time, derived from the replayed timeline
    (offered load), binned into fixed windows.
  • BOTTOM — "GPU Utilization Under Production Workload": whole-device GPU
    utilization (%) vs time from the gpu_util CSV produced by
    ``production_gpu_bench.py``, with a dashed reference line (default 60%).

The two panels share a time axis. The GPU samples (wall-clock) are anchored to
the benchmark's recording t=0 (``bench_meta``'s ``t_first_wall_iso``), which is
the same origin the timeline's first arrival maps to — so the workload trace and
the GPU trace line up.

Tokens/sec is the OFFERED token rate: per request, ``est_prompt_tokens`` (from
the timeline's char-based ``prompt_length``, see CHARS_PER_TOKEN) +
``max_new_tokens``, binned by arrival time. Requests/sec is the offered arrival
rate.

Inputs default to ``eval/production_trace/`` (written by production_gpu_bench.py)
and the replayed timeline under ``eval/timelines/<gpu>/``. The figure is written
to ``eval/production_trace/``.

Usage (defaults assume an inference-only nutanix run on the 5090):
  python eval/plot_production.py
  python eval/plot_production.py --mode nutanix --timeline-gpu 5090
  python eval/plot_production.py --timeline <t.csv> --gpu-util <g.csv> \
      --bench-meta <m.json> --output <fig.png>
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

_HERE = os.path.dirname(os.path.abspath(__file__))
PROD_DIR = os.path.join(_HERE, "production_trace")

# ============================================================================
# Settings — edit here to retitle / resize / restyle / re-bin.
# ============================================================================

# ---- Output ----
GENERATE_PDF = True
PNG_DPI = 130

# ---- Figure ----
FIGSIZE = (10, 5)
SUBPLOT_HSPACE = 0.09    # vertical gap between the two panels (fraction of panel
                          # height); larger = more space, smaller = tighter

# ---- Binning ----
BIN_S = 5.0               # workload-trace bin width (s); gives the staircase look
CHARS_PER_TOKEN = 4.0     # prompt_length is chars; Llama-3 BPE ≈ 4 chars/token
INCLUDE_PROMPT_TOKENS = True   # tokens/sec = prompt_est + max_new_tokens (vs just output)

# ---- Titles / labels ----
TITLE_TOP = "Company X Production Inference Workload Trace"
TITLE_BOTTOM = "GPU Utilization Under Production Workload"
XLABEL = "Time (s)"
YLABEL_TOKENS = "Tokens/sec"
YLABEL_REQS = "Requests/sec"
YLABEL_GPU = "GPU Utilization (%)"

# ---- GPU average line ----
SHOW_AVG_LINE = True      # dashed line at the MEAN GPU util computed from the CSV
                          # (labelled as a tick on the right axis)

# ---- Colors ----
TOKENS_FILL = "0.82"          # light gray area under tokens/sec
TOKENS_EDGE = "#6a5acd"       # slate-blue tokens/sec line (matches left axis)
TOKENS_LABEL_COLOR = "#6a5acd"  # slate-blue left-axis label (matches reference)
REQS_COLOR = "#1a1a1a"        # near-black requests/sec line (right axis)
GPU_COLOR = "tab:blue"
AVG_COLOR = "tab:blue"        # mean-util dashed line + right-axis tick

# ---- Legend (top panel) ----
SHOW_LEGEND = False           # label both the tokens/sec and requests/sec curves

# ---- Fonts ----
FONTSIZE_TITLE = 20
FONTSIZE_AXIS_LABEL = 16
FONTSIZE_TICK = 14
FONTWEIGHT_TITLE = "normal"

# ============================================================================


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_timeline(path: str):
    """Return (t_rel, tokens_per_req) numpy arrays from the replayed timeline.
    t_rel = arrival time relative to the first arrival. tokens_per_req =
    est_prompt_tokens + max_new_tokens (or just max_new_tokens)."""
    ts, mnt, plen = [], [], []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        cols = set(r.fieldnames or [])
        if "timestamp_s" not in cols or "max_new_tokens" not in cols:
            sys.exit(f"[production] {path}: need timestamp_s + max_new_tokens cols")
        has_plen = "prompt_length" in cols
        for row in r:
            ts.append(_f(row.get("timestamp_s")))
            mnt.append(_f(row.get("max_new_tokens")))
            plen.append(_f(row.get("prompt_length")) if has_plen else 0.0)
    ts = np.asarray(ts, dtype=float)
    mnt = np.asarray(mnt, dtype=float)
    plen = np.asarray(plen, dtype=float)
    t_rel = ts - (ts.min() if ts.size else 0.0)
    prompt_tok = np.ceil(np.nan_to_num(plen) / CHARS_PER_TOKEN) if INCLUDE_PROMPT_TOKENS \
        else np.zeros_like(plen)
    tokens = np.nan_to_num(mnt) + prompt_tok
    return t_rel, tokens


def load_gpu_util(path: str, t0: datetime.datetime | None):
    """Return (rel_seconds, util_pct) from the gpu_util CSV, anchored to t0.
    If t0 is None, fall back to the first sample as the origin."""
    times, util = [], []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                t = datetime.datetime.fromisoformat((row["timestamp"]).strip())
            except (KeyError, ValueError, TypeError):
                continue
            times.append(t)
            util.append(_f(row.get("util_pct")))
    if not times:
        sys.exit(f"[production] no usable rows in {path}")
    origin = t0 or min(times)
    rel = np.array([(t - origin).total_seconds() for t in times], dtype=float)
    return rel, np.asarray(util, dtype=float)


def read_t0_wall(meta_path: str) -> datetime.datetime | None:
    """Recording t=0 wall clock from bench_meta's t_first_wall_iso."""
    try:
        with open(meta_path) as f:
            meta = json.load(f) or {}
        iso = meta.get("t_first_wall_iso")
        return datetime.datetime.fromisoformat(iso) if iso else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _binned_rate(t_rel, weights, t_max, bin_s):
    """Per-bin rate: counts (or summed weights) / bin_s. Returns (centers, rate)."""
    edges = np.arange(0.0, t_max + bin_s, bin_s)
    if edges.size < 2:
        edges = np.array([0.0, bin_s])
    counts, _ = np.histogram(t_rel, bins=edges, weights=weights)
    centers = edges[:-1] + bin_s / 2.0
    return centers, counts / bin_s


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="nutanix",
                    help="Workload tag for default file names. Default nutanix.")
    ap.add_argument("--timeline-gpu", default="5090", choices=["5090", "A100"],
                    help="Which timelines/<gpu>/ dir the run replayed. Default 5090.")
    ap.add_argument("--timeline", default=None,
                    help="Replayed timeline CSV. Default: "
                         "eval/timelines/<gpu>/timeline_<mode>.csv")
    ap.add_argument("--gpu-util", default=None,
                    help="GPU-util CSV. Default: "
                         "eval/production_trace/gpu_util_<mode>.csv")
    ap.add_argument("--bench-meta", default=None,
                    help="bench_meta JSON (t0 anchor). Default: "
                         "eval/production_trace/bench_meta_<mode>.json")
    ap.add_argument("--output", default=None,
                    help="Figure PNG path. Default: "
                         "eval/production_trace/production_trace.png")
    ap.add_argument("--bin-s", type=float, default=BIN_S,
                    help=f"Workload bin width (s). Default {BIN_S}.")
    args = ap.parse_args()

    timeline = args.timeline or os.path.join(
        _HERE, "timelines", args.timeline_gpu, f"timeline_{args.mode}.csv")
    gpu_util = args.gpu_util or os.path.join(PROD_DIR, f"gpu_util_{args.mode}.csv")
    bench_meta = args.bench_meta or os.path.join(
        PROD_DIR, f"bench_meta_{args.mode}.json")
    out_path = args.output or os.path.join(PROD_DIR, "production_trace.png")

    for p, what in ((timeline, "timeline"), (gpu_util, "gpu_util")):
        if not os.path.exists(p):
            sys.exit(f"[production] {what} not found: {p}\n"
                     "Run eval/production_gpu_bench.py first.")

    t_rel, tokens = load_timeline(timeline)
    t0 = read_t0_wall(bench_meta)
    if t0 is None:
        print(f"[production] no bench_meta t0 ({bench_meta}); anchoring GPU to "
              "its first sample (panels may be slightly offset).", flush=True)
    gpu_rel, gpu_util_pct = load_gpu_util(gpu_util, t0)

    # Shared time window: the offered workload span, extended to cover any GPU
    # tail. Drop GPU samples before t=0 (warmup) and far past the window.
    t_max = float(max(t_rel.max() if t_rel.size else 0.0,
                      gpu_rel[gpu_rel >= 0].max() if np.any(gpu_rel >= 0) else 0.0))
    gmask = (gpu_rel >= 0.0) & (gpu_rel <= t_max + args.bin_s)
    gpu_rel, gpu_util_pct = gpu_rel[gmask], gpu_util_pct[gmask]

    centers, tok_s = _binned_rate(t_rel, tokens, t_max, args.bin_s)
    _, req_s = _binned_rate(t_rel, np.ones_like(t_rel), t_max, args.bin_s)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=FIGSIZE, constrained_layout=True)
    # Vertical gap between the two panels (constrained_layout's hspace).
    try:
        fig.get_layout_engine().set(hspace=SUBPLOT_HSPACE)
    except Exception:  # older matplotlib
        try:
            fig.set_constrained_layout_pads(hspace=SUBPLOT_HSPACE)
        except Exception:
            pass

    # ---- TOP: tokens/sec (left, slate-blue, gray fill) + requests/sec (right, black) ----
    # The two curves nearly coincide (tokens ≈ requests × avg-tokens/req), so
    # distinct colors + requests-on-top keep BOTH visible where they diverge.
    ax_top.fill_between(centers, tok_s, step="mid", color=TOKENS_FILL, zorder=1)
    (tok_line,) = ax_top.plot(centers, tok_s, drawstyle="steps-mid",
                              color=TOKENS_EDGE, linewidth=1.5, zorder=2,
                              label=YLABEL_TOKENS)
    ax_top.set_ylabel(YLABEL_TOKENS, fontsize=FONTSIZE_AXIS_LABEL,
                      color=TOKENS_LABEL_COLOR)
    ax_top.tick_params(axis="y", labelsize=FONTSIZE_TICK,
                       labelcolor=TOKENS_LABEL_COLOR)
    ax_top.tick_params(axis="x", labelsize=FONTSIZE_TICK)
    ax_top.set_ylim(bottom=0)
    ax_top.set_title(TITLE_TOP, fontsize=FONTSIZE_TITLE,
                     fontweight=FONTWEIGHT_TITLE)
    ax_top.grid(True, axis="y", alpha=0.25)

    ax_req = ax_top.twinx()
    (req_line,) = ax_req.plot(centers, req_s, drawstyle="steps-mid",
                              color=REQS_COLOR, linewidth=1.0, alpha=0.9,
                              zorder=3, label=YLABEL_REQS)
    ax_req.set_ylabel(YLABEL_REQS, fontsize=FONTSIZE_AXIS_LABEL, color=REQS_COLOR)
    ax_req.tick_params(axis="y", labelsize=FONTSIZE_TICK, labelcolor=REQS_COLOR)
    ax_req.set_ylim(bottom=0)

    if SHOW_LEGEND:
        ax_top.legend([tok_line, req_line], [YLABEL_TOKENS, YLABEL_REQS],
                      loc="upper right", fontsize=FONTSIZE_TICK, framealpha=0.9)

    # ---- BOTTOM: GPU utilization (%) + mean line ----
    ax_bot.plot(gpu_rel, gpu_util_pct, color=GPU_COLOR, linewidth=1.0)
    ax_bot.set_ylabel(YLABEL_GPU, fontsize=FONTSIZE_AXIS_LABEL)
    ax_bot.set_xlabel(XLABEL, fontsize=FONTSIZE_AXIS_LABEL)
    ax_bot.set_title(TITLE_BOTTOM, fontsize=FONTSIZE_TITLE,
                     fontweight=FONTWEIGHT_TITLE)
    ax_bot.set_ylim(0, 105)
    ax_bot.tick_params(labelsize=FONTSIZE_TICK)
    ax_bot.grid(True, alpha=0.25)

    for ax in (ax_top, ax_bot):
        ax.set_xlim(0, t_max)

    # Dashed line at the REAL mean GPU util (computed from the CSV), with the
    # value as a tick on the RIGHT axis (outside the plot, not floating text).
    if SHOW_AVG_LINE and gpu_util_pct.size:
        avg_util = float(np.nanmean(gpu_util_pct))
        ax_bot.axhline(avg_util, color=AVG_COLOR, linestyle="--",
                       linewidth=1.3, alpha=0.7)
        ax_avg = ax_bot.twinx()
        ax_avg.set_ylim(ax_bot.get_ylim())
        ax_avg.set_yticks([avg_util])
        ax_avg.set_yticklabels([f"{avg_util:.1f}%"])
        ax_avg.tick_params(axis="y", colors=AVG_COLOR, labelsize=FONTSIZE_TICK)
        print(f"[production] mean GPU util over the recording window: "
              f"{avg_util:.1f}%")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=PNG_DPI)
    print(f"[production] wrote figure → {out_path}")
    if GENERATE_PDF:
        pdf = os.path.splitext(out_path)[0] + ".pdf"
        fig.savefig(pdf, format="pdf", bbox_inches="tight", pad_inches=0.05)
        print(f"[production] wrote figure → {pdf}")
    plt.close(fig)


if __name__ == "__main__":
    main()
