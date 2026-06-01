#!/usr/bin/env python3
"""auto_plot_schedulers.py — A/B-compare co-serving schedulers + inf-only.

Emits TWO focused PNGs per run, both centered on the ``both`` (unified-phase)
scheduler — the variant we want to A/B against the other two anchors:

  * ``scheduler_compare_<mode>_factor_<tag>_both_vs_inf-only.png``
    Co-serving (both-phase) vs the no-co baseline — shows the inference
    overhead of running co-serving at all.
  * ``scheduler_compare_<mode>_factor_<tag>_both_vs_prefill.png``
    Both-phase scheduler vs the prefill-only scheduler — shows the
    incremental effect of admitting FT to decode-only / mixed steps.

Each file uses the same 4-panel layout as ``auto_plot.py``:
  1. Scheduled request timeline (req/s bars + output tok/s line). Plotted
     once — the request schedule is identical across runs.
  2. Per-request E2E latency vs time — one scatter per run.
  3. Throughput tok/s — inference (solid) + FT (dashed) lines per run.
     Inf-only contributes only the inference line (no FT log).
  4. TTFT SLO satisfaction rate (rolling-window %) — one line per run.

Inputs (under eval/output/, suffix scheme from auto_benchmark.py):
  timeline_results_co_factor_<tag>_phase_<phase>_<mode>.csv   (per scheduler)
  bwd_log_co_factor_<tag>_phase_<phase>_<mode>.csv            (per scheduler)
  bench_meta_co_factor_<tag>_phase_<phase>_<mode>.json        (per scheduler)
  timeline_results_<mode>.csv                                 (inf-only baseline)
  timelines/<gpu>/timeline_<mode>.csv                         (request schedule)

To produce the three input runs (all required for both PNGs):
  python eval/auto_benchmark.py --loose                         # inf-only baseline
  python eval/auto_benchmark.py --co --loose --scheduler prefill  # phase=prefill
  python eval/auto_benchmark.py --co --loose --scheduler both     # phase=both

Then plot:
  python eval/auto_plot_schedulers.py --mode loose
"""
import argparse
import datetime
import glob
import os
import re
import sys
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the existing single-run plotter's helpers — same parse semantics
# (CSV columns, t0 anchoring, bin/smooth helpers, etc.) keep the two plotters
# in lock-step.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from auto_plot import (  # noqa: E402
    ALL_MODES,
    FT_SHADE_COLOR,
    _auto_window,
    _distribute_to_bins,
    _ft_per_bin,
    _smooth,
    detect_gpu_subdir,
    load_results,
    load_timeline,
    parse_bwd_log_csv,
    plot_latency_vs_time,
    plot_request_timeline,
    read_slo,
    read_ttft_slo,
)

_ROOT = os.path.dirname(_HERE)
OUTPUT_DIR = os.path.join(_HERE, "output")
PLOTS_DIR = os.path.join(_HERE, "plots")
CONFIG_YAML_DEFAULT = os.path.join(
    _ROOT, "configs", "serving_config_finetuning_llama3.yaml")

# Color per phase; inf-only stays neutral gray so the co-serving phases stand
# out as the comparison axis. Add a color here if a future phase value lands.
PHASE_COLORS = {
    "prefill": "tab:blue",
    "both": "tab:orange",
}
INF_ONLY_COLOR = "tab:gray"


# --------------------------------------------------------------------------
# File-suffix discovery
# --------------------------------------------------------------------------

def _suffix_for_phase(factor_tag: str, phase: str, mode: str) -> str:
    """Reconstruct the auto_benchmark suffix for a (factor, phase, mode)
    tuple: ``_co_factor_<tag>_phase_<phase>_<mode>``. Matches what
    ``auto_benchmark.py:main()`` writes."""
    return f"_co_factor_{factor_tag}_phase_{phase}_{mode}"


def _discover_phase_tags(output_dir: str, factor_tag: str,
                         mode: str) -> list[str]:
    """Scan ``output_dir`` for ``timeline_results_co_factor_<tag>_phase_*_<mode>.csv``
    and return the set of distinct phase values found, lexicographic."""
    pattern = os.path.join(
        output_dir,
        f"timeline_results_co_factor_{factor_tag}_phase_*_{mode}.csv")
    name_re = re.compile(
        r"^timeline_results_co_factor_" + re.escape(factor_tag)
        + r"_phase_(?P<phase>[A-Za-z0-9]+)_" + re.escape(mode) + r"\.csv$")
    found: set[str] = set()
    for path in glob.glob(pattern):
        m = name_re.match(os.path.basename(path))
        if m is not None:
            found.add(m.group("phase"))
    return sorted(found)


def _discover_factor_tags_with_phase(output_dir: str, mode: str) -> list[str]:
    """Scan ``output_dir`` for ``timeline_results_co_factor_<tag>_phase_*_<mode>.csv``
    and return the set of distinct ``<tag>`` strings, sorted by numeric value
    (``off`` sorts as -1.0). Used for ``--factor`` autodetection."""
    pattern = os.path.join(
        output_dir, f"timeline_results_co_factor_*_phase_*_{mode}.csv")
    name_re = re.compile(
        r"^timeline_results_co_factor_(?P<tag>[^_]+(?:\.[^_]+)?)"
        r"_phase_[A-Za-z0-9]+_" + re.escape(mode) + r"\.csv$")
    tags: set[str] = set()
    for path in glob.glob(pattern):
        m = name_re.match(os.path.basename(path))
        if m is not None:
            tags.add(m.group("tag"))

    def _val(t: str) -> float:
        return -1.0 if t == "off" else float(t)
    try:
        return sorted(tags, key=_val)
    except ValueError:
        return sorted(tags)


def _maybe_load_meta(meta_path: str) -> Optional[datetime.datetime]:
    """Same t0-wall lookup as ``auto_plot.make_figure_for_mode``."""
    if not os.path.exists(meta_path):
        return None
    try:
        import json as _json
        with open(meta_path) as f:
            meta = _json.load(f)
        iso = meta.get("t_first_wall_iso")
        return datetime.datetime.fromisoformat(iso) if iso else None
    except Exception as e:
        print(f"[plot] meta load failed ({meta_path}): {e}")
        return None


# --------------------------------------------------------------------------
# Custom multi-series panel renderers
# (auto_plot.py helpers are single-series — fill_between bands occlude when
# overlaid; the TTFT text box uses fixed axes coordinates. Rewritten here
# to overlay cleanly across runs.)
# --------------------------------------------------------------------------

# Hatch patterns cycle per run for the FT band, so overlapping FT regions
# stay visually distinguishable when both runs have FT (the "both vs prefill"
# figure). The inf-only baseline never reaches the FT branch, so its slot is
# unused. Kept short on purpose — matplotlib's hatch renderer thrashes with
# dense patterns.
_FT_HATCHES = ("//", "\\\\")


def _plot_throughput_multi(ax, runs: list[dict], inf_run: Optional[dict],
                           tl: dict, bin_s: float = 1.0,
                           smoothing_window_s: Optional[float] = None) -> None:
    """Inference + FT throughput in tok/s, overlaid per scheduler.

    Same visual identity as ``auto_plot.plot_throughput_curves`` (single
    inference fill from 0 → inf_smooth, FT fill from inf_smooth → total,
    inference line, total line), repeated per run with per-run color +
    different hatches on the FT band so overlapping regions stay readable.
    Inf-only contributes only the inference fill + line (no FT log)."""
    tok_by_row = {int(rid): tok for rid, tok in
                  zip(tl["row_id"], tl["max_new_tokens"])}

    def _inf_tokens(res):
        ok = res["ok"]
        sel = ok & np.isfinite(res["ttft_s"]) & np.isfinite(res["latency_s"])
        t_start, t_end, tokens = [], [], []
        for i in np.nonzero(sel)[0]:
            tok = tok_by_row.get(int(res["idx"][i]))
            if tok is None or not np.isfinite(tok):
                continue
            t_start.append(res["t_rel_s"][i] + res["ttft_s"][i])
            t_end.append(res["t_rel_s"][i] + res["latency_s"][i])
            tokens.append(tok)
        return (np.asarray(t_start, dtype=float),
                np.asarray(t_end, dtype=float),
                np.asarray(tokens, dtype=float))

    # Order: phase runs first (they get the FT bands), inf-only last so its
    # inference fill draws on top — easier to see the no-co baseline shape.
    all_runs: list[tuple[dict, bool]] = [(r, True) for r in runs]
    if inf_run is not None:
        all_runs.append((inf_run, False))
    if not all_runs:
        ax.set_title("Throughput (no data)")
        return

    # Pre-compute per-run inference series + a shared t_max so every band
    # uses the same binning grid (matches auto_plot's single-run behaviour).
    inf_series: list[tuple[dict, bool, np.ndarray, np.ndarray, np.ndarray]] = []
    t_maxes = [0.0]
    for run, is_phase in all_runs:
        ts, te, tk = _inf_tokens(run["res"])
        inf_series.append((run, is_phase, ts, te, tk))
        if len(te):
            t_maxes.append(float(te.max()))
        if is_phase:
            rel_t, _, _ = run["bwd"]
            if len(rel_t):
                t_maxes.append(float(rel_t[-1]))
    t_max = max(t_maxes)
    if t_max <= 0:
        ax.set_title("Throughput (no data)")
        return
    win_s = smoothing_window_s if smoothing_window_s is not None \
        else _auto_window(t_max)

    drew_any = False
    for hatch_idx, (run, is_phase, ts, te, tk) in enumerate(inf_series):
        inf_per_s = _distribute_to_bins(ts, te, tk, t_max, bin_s) / bin_s
        inf_smooth = _smooth(inf_per_s, win_s, bin_s)
        centers = (np.arange(len(inf_smooth)) + 0.5) * bin_s
        inf_avg = float(inf_per_s.mean()) if inf_per_s.size else 0.0

        # Inference contribution band (color per run; same alpha as auto_plot).
        ax.fill_between(centers, 0, inf_smooth, color=run["color"], alpha=0.20,
                        linewidth=0,
                        label=f"{run['label']} inference contribution "
                              f"(avg {inf_avg:.0f} tok/s)",
                        zorder=2)
        # Inference line on the band edge — matches auto_plot.
        ax.plot(centers, inf_smooth, color=run["color"], linewidth=1.6,
                label=f"{run['label']} inference (avg {inf_avg:.0f} tok/s)",
                zorder=3)
        drew_any = True

        if not is_phase:
            continue   # inf-only: skip FT band + total

        rel_t, cum_tok, _ = run["bwd"]
        if rel_t.size == 0:
            continue
        ft_per_s = _ft_per_bin(rel_t, cum_tok, t_max, bin_s) / bin_s
        ft_smooth = _smooth(ft_per_s, win_s, bin_s)
        # Re-bin total in case ft_per_s is shorter (pad to inf_smooth length).
        n = max(len(inf_smooth), len(ft_smooth))
        inf_padded = np.pad(inf_smooth, (0, n - len(inf_smooth)))
        ft_padded = np.pad(ft_smooth, (0, n - len(ft_smooth)))
        centers_t = (np.arange(n) + 0.5) * bin_s
        total = inf_padded + ft_padded
        ft_avg = float(ft_per_s.mean()) if ft_per_s.size else 0.0
        tot_avg = float(total.mean())
        hatch = _FT_HATCHES[hatch_idx % len(_FT_HATCHES)]

        # FT contribution band (orange, hatched — color matches auto_plot;
        # hatch varies per run so overlapping FT regions read separately).
        ax.fill_between(centers_t, inf_padded, total, color=FT_SHADE_COLOR,
                        alpha=0.30, hatch=hatch, linewidth=0,
                        label=f"{run['label']} finetune contribution "
                              f"(avg {ft_avg:.0f} tok/s)",
                        zorder=2)
        # Total line — per-run color (not black) so it pairs with the
        # inference line above. Distinguishable across schedulers.
        ax.plot(centers_t, total, color=run["color"], linewidth=1.8,
                linestyle="--",
                label=f"{run['label']} total (avg {tot_avg:.0f} tok/s)",
                zorder=3)

    if not drew_any:
        ax.set_title("Throughput (no data)")
        return
    title = "Throughput (tokens/s)"
    if win_s > bin_s:
        title += f" — {win_s:.0f}s rolling mean"
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tokens / s")
    ax.legend(loc="best", fontsize=7)


def _plot_ttft_satisfaction_multi(ax, runs: list[dict], slo_s: float,
                                  window_s: float = 5.0) -> None:
    """Multi-series TTFT-satisfaction overlay (one rolling-% line per run).

    Each ``run`` is a dict with keys: ``label``, ``color``, ``res`` (the
    ``load_results`` dict). The original ``plot_ttft_satisfaction`` in
    ``auto_plot.py`` writes a single line + a text box at fixed coords;
    multi-call would overlap. This version draws a line per run and folds
    per-run summary stats into a single combined text box."""
    drawn_any = False
    summary_lines: list[str] = []
    for run in runs:
        res = run["res"]
        ok = res["ok"] & np.isfinite(res["ttft_s"])
        t = res["t_rel_s"][ok]
        ttft = res["ttft_s"][ok]
        if t.size == 0:
            continue
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
        ax.plot(grid, rates, color=run["color"], linewidth=1.5,
                label=f"{run['label']} — overall {overall:.1f}%")
        summary_lines.append(
            f"{run['label']}: avg {avg_ttft:.3f}s · p90 {p90_ttft:.3f}s")
        drawn_any = True
    if not drawn_any:
        ax.set_title("TTFT Satisfaction Rate (no data)")
        return
    ax.axhline(95.0, color="tab:red", linestyle="--", linewidth=1, alpha=0.6,
               label="95% target")
    ax.set_ylim(0, 105)
    ax.set_title(f"TTFT Satisfaction Rate (SLO ≤ {slo_s:.2f}s, "
                 f"{int(window_s)}s window)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("% of requests")
    ax.text(0.02, 0.04, f"TTFT SLO = {slo_s:.2f}s\n" + "\n".join(summary_lines),
            transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75,
                      edgecolor="0.7"))
    ax.legend(loc="upper right", fontsize=8)


# --------------------------------------------------------------------------
# Figure assembly
# --------------------------------------------------------------------------

def _load_phase_run(output_dir: str, factor_tag: str, phase: str,
                    mode: str) -> Optional[dict]:
    """Load a per-phase run from disk. Returns None if its results CSV is
    missing — caller decides whether that's a hard error."""
    suffix = _suffix_for_phase(factor_tag, phase, mode)
    results_csv = os.path.join(output_dir, f"timeline_results{suffix}.csv")
    bwd_csv = os.path.join(output_dir, f"bwd_log{suffix}.csv")
    meta_json = os.path.join(output_dir, f"bench_meta{suffix}.json")
    if not os.path.exists(results_csv):
        return None
    print(f"[plot] phase={phase}: {results_csv}")
    print(f"[plot]            + {bwd_csv} "
          f"({'present' if os.path.exists(bwd_csv) else 'MISSING'})")
    t0_wall = _maybe_load_meta(meta_json)
    return {
        "phase": phase,
        "label": f"phase={phase}",
        "color": PHASE_COLORS.get(phase, "tab:purple"),
        "res": load_results(results_csv),
        "bwd": parse_bwd_log_csv(bwd_csv, t0_wall=t0_wall),
    }


def _load_inf_only_run(output_dir: str, mode: str) -> Optional[dict]:
    """Load the inference-only baseline (timeline_results_<mode>.csv, no
    suffix/factor/phase tags). Returns None if absent. No bwd_log."""
    infonly_csv = os.path.join(output_dir, f"timeline_results_{mode}.csv")
    if not os.path.exists(infonly_csv):
        return None
    print(f"[plot] inf-only:   {infonly_csv}")
    return {
        "label": "inf-only",
        "color": INF_ONLY_COLOR,
        "res": load_results(infonly_csv),
    }


def _make_one_figure(mode: str, factor_tag: str, compare_tag: str,
                     phase_runs: list[dict], inf_run: Optional[dict],
                     tl: dict, plots_dir: str, ttft_slo: float,
                     window_s: float = 5.0) -> str:
    """Render one A/B figure with the given runs (4 panels, same layout as
    ``auto_plot.py``). ``compare_tag`` differentiates the output PNG, e.g.
    ``both_vs_inf-only`` or ``both_vs_prefill``."""
    fig = plt.figure(figsize=(24, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, 4)
    ax_timeline = fig.add_subplot(gs[0, 0])
    ax_latency = fig.add_subplot(gs[0, 1])
    ax_throughput = fig.add_subplot(gs[0, 2])
    ax_ttft = fig.add_subplot(gs[0, 3])

    # Panel 1: Request timeline — shared input, plot once.
    plot_request_timeline(ax_timeline, tl)

    # Panel 2: E2E latency vs time — overlay phases + (optional) inf-only.
    for run in phase_runs:
        plot_latency_vs_time(ax_latency, run["res"], label=run["label"],
                             color=run["color"])
    if inf_run is not None:
        plot_latency_vs_time(ax_latency, inf_run["res"],
                             label=inf_run["label"], color=inf_run["color"])

    # Panel 3: Throughput — multi-series inference + FT lines.
    _plot_throughput_multi(ax_throughput, phase_runs, inf_run, tl)

    # Panel 4: TTFT SLO satisfaction — multi-series.
    ttft_runs = list(phase_runs)
    if inf_run is not None:
        ttft_runs.append(inf_run)
    _plot_ttft_satisfaction_multi(ax_ttft, ttft_runs, slo_s=ttft_slo,
                                  window_s=window_s)

    fig.suptitle(
        f"Scheduler A/B [{compare_tag}] — mode={mode}, factor={factor_tag}",
        fontsize=13)

    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(
        plots_dir,
        f"scheduler_compare_{mode}_factor_{factor_tag}_{compare_tag}.png")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def make_figures(mode: str, factor_tag: str, output_dir: str, plots_dir: str,
                 timeline_csv_dir: str, ttft_slo: float,
                 window_s: float = 5.0) -> list[str]:
    """Build the two centered-on-``both`` comparison figures:

      1. ``both`` vs ``inf-only`` — co-serving overhead vs no-co baseline.
      2. ``both`` vs ``prefill``  — unified-phase admission vs prefill-only.

    Missing inputs skip the corresponding figure (with a clear print) rather
    than crashing — partial A/B data still produces something useful.
    Returns the list of PNG paths actually written."""
    # Shared request timeline.
    timeline_csv = os.path.join(timeline_csv_dir, f"timeline_{mode}.csv")
    print(f"[plot] timeline:    {timeline_csv}")
    tl = load_timeline(timeline_csv)

    both_run = _load_phase_run(output_dir, factor_tag, "both", mode)
    prefill_run = _load_phase_run(output_dir, factor_tag, "prefill", mode)
    inf_run = _load_inf_only_run(output_dir, mode)

    written: list[str] = []

    # Figure 1: both vs inf-only.
    if both_run is None:
        print("[plot] skip both_vs_inf-only: phase=both run not found")
    elif inf_run is None:
        print("[plot] skip both_vs_inf-only: inf-only baseline not found")
    else:
        path = _make_one_figure(
            mode=mode, factor_tag=factor_tag,
            compare_tag="both_vs_inf-only",
            phase_runs=[both_run], inf_run=inf_run,
            tl=tl, plots_dir=plots_dir,
            ttft_slo=ttft_slo, window_s=window_s)
        print(f"[plot] wrote {path}")
        written.append(path)

    # Figure 2: both vs prefill (no inf-only — head-to-head schedulers).
    if both_run is None:
        print("[plot] skip both_vs_prefill: phase=both run not found")
    elif prefill_run is None:
        print("[plot] skip both_vs_prefill: phase=prefill run not found")
    else:
        path = _make_one_figure(
            mode=mode, factor_tag=factor_tag,
            compare_tag="both_vs_prefill",
            phase_runs=[both_run, prefill_run], inf_run=None,
            tl=tl, plots_dir=plots_dir,
            ttft_slo=ttft_slo, window_s=window_s)
        print(f"[plot] wrote {path}")
        written.append(path)

    if not written:
        raise FileNotFoundError(
            f"no figures produced for mode={mode} factor={factor_tag}; "
            "need at least phase=both + (inf-only OR phase=prefill) on disk. "
            "Run auto_benchmark.py first.")
    return written


def main() -> None:
    gpu_default = detect_gpu_subdir()
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=ALL_MODES, default="loose",
                    help="Which workload-shape timeline the A/B was run on.")
    ap.add_argument("--factor", default=None,
                    help="Factor tag (matches the auto_benchmark suffix; "
                         "e.g. 'off', '0.5', '2'). Default: autodetect the "
                         "smallest factor present in output/.")
    ap.add_argument("--output_dir", default=OUTPUT_DIR,
                    help=f"Where the auto_benchmark CSVs live. "
                         f"Default {OUTPUT_DIR}.")
    ap.add_argument("--plots_dir", default=PLOTS_DIR,
                    help=f"Where to write the comparison PNGs. "
                         f"Default {PLOTS_DIR}.")
    ap.add_argument("--timeline-gpu", default=gpu_default,
                    choices=["5090", "A100"],
                    help="timelines/<gpu>/ subdir for the request-schedule CSV.")
    ap.add_argument("--config", default=CONFIG_YAML_DEFAULT,
                    help=f"YAML to read the TTFT SLO from. Default "
                         f"{os.path.relpath(CONFIG_YAML_DEFAULT, _ROOT)}.")
    ap.add_argument("--window_s", type=float, default=5.0,
                    help="Rolling-window seconds for the TTFT-satisfaction "
                         "panel. Default 5.")
    args = ap.parse_args()

    # Autodetect the factor tag if not given.
    factor_tag = args.factor
    if factor_tag is None:
        tags = _discover_factor_tags_with_phase(args.output_dir, args.mode)
        if not tags:
            ap.error(
                f"no timeline_results_co_factor_*_phase_*_{args.mode}.csv "
                f"files found in {args.output_dir}; pass --factor explicitly "
                f"or run auto_benchmark first")
        factor_tag = tags[0]
        print(f"[plot] auto-detected factor tag: {factor_tag} "
              f"(available: {tags})")

    # SLO from the YAML.
    ttft_slo = read_ttft_slo(args.config) or 1.0
    avg_tbt_slo = read_slo(args.config, "avg_tbt_slo")
    timeline_dir = os.path.join(_HERE, "timelines", args.timeline_gpu)

    written = make_figures(
        mode=args.mode,
        factor_tag=factor_tag,
        output_dir=args.output_dir,
        plots_dir=args.plots_dir,
        timeline_csv_dir=timeline_dir,
        ttft_slo=ttft_slo,
        window_s=args.window_s,
    )
    summary = "; ".join(os.path.basename(p) for p in written)
    print(f"[plot] done — {len(written)} file(s): {summary} "
          f"(ttft_slo={ttft_slo:.2f}s"
          + (f", avg_tbt_slo={avg_tbt_slo:.3f}s" if avg_tbt_slo else "")
          + ")")


if __name__ == "__main__":
    main()
