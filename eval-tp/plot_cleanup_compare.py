#!/usr/bin/env python3
"""Baseline vs final comparison plots for the 2026-10 cleanup (CLEANUP_LOG.md).

Reads the validation runs under eval-tp/output/cleanup/ (Qwen3-14B TP=2,
co-serving) and writes two figures to eval-tp/plots/:

  cleanup_compare_summary.png       TTFT percentiles + FT throughput per workload
  cleanup_compare_distributions.png per workload: TTFT CDF, worst-TBT CDF,
                                    cumulative FT tokens over the timeline

    python eval-tp/plot_cleanup_compare.py [--base 00_base,00_base_r2] [--final 11_arrival,11_arrival_r2]

A run directory missing a workload is skipped for that workload.
"""

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_HERE = Path(__file__).resolve().parent
_RUNS = _HERE / "output" / "cleanup"
_OUT = _HERE / "plots"
_SFX = "_qwen3-14b_tp2_co_factor_off_phase_prefill_"
WORKLOADS = [("tight-half", "tight (first half)"),
             ("loose-half", "loose (first half)"),
             ("nutanix-600-800", "nutanix 600–800 s")]
TTFT_SLO_MS = 400
# Categorical slots 1 and 2 of the reference palette (validated: CVD ΔE 24.7).
COLOR = {"baseline": "#eb6834", "final": "#2a78d6"}
LABEL = {"baseline": "baseline (original fork)", "final": "final (plugin, stock vLLM)"}
INK, MUTED, GRID = "#1f2328", "#59636e", "#e4e6e9"


def load(run: str, mode: str) -> dict | None:
    d = _RUNS / run
    res_p = d / f"timeline_results{_SFX}{mode}.csv"
    if not res_p.exists():
        return None
    res = list(csv.DictReader(open(res_p)))
    t0 = dt.datetime.fromisoformat(
        json.load(open(d / f"bench_meta{_SFX}{mode}.json"))["t_first_wall_iso"]).timestamp()
    w0 = t0 + min(float(r["t_rel_s"]) for r in res)
    w1 = t0 + max(float(r["t_rel_s"]) + float(r["latency_s"]) for r in res)
    bwd = [(dt.datetime.fromisoformat(r["timestamp"]).timestamp(), int(r["batch_tokens"]))
           for r in csv.DictReader(open(d / f"bwd_log{_SFX}{mode}.csv"))]
    bwd = [(t - w0, n) for t, n in bwd if w0 <= t <= w1]
    return {
        "ttft": np.array(sorted(float(r["ttft_s"]) * 1e3 for r in res)),
        "wtbt": np.array(sorted(float(r["worst_tbt_s"]) * 1e3 for r in res)),
        "ft_t": np.array([t for t, _ in bwd]),
        "ft_cum": np.cumsum([n for _, n in bwd]),
        "ft_rate": sum(n for _, n in bwd) / (w1 - w0),
        "span": w1 - w0,
    }


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def cdf(ax, arrs, kind):
    for i, a in enumerate(arrs):
        y = np.arange(1, len(a) + 1) / len(a) * 100
        ax.step(a, y, where="post", color=COLOR[kind], linewidth=2,
                alpha=1.0 if i == 0 else 0.45,
                label=LABEL[kind] if i == 0 else None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="00_base,00_base_r2")
    ap.add_argument("--final", default="11_arrival,11_arrival_r2")
    args = ap.parse_args()
    runs = {"baseline": args.base.split(","), "final": args.final.split(",")}
    data = {m: {k: [x for x in (load(r, m) for r in rs) if x] for k, rs in runs.items()}
            for m, _ in WORKLOADS}
    _OUT.mkdir(parents=True, exist_ok=True)

    # ── summary: TTFT p50 / p95 / p99 / max, and FT throughput ──
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2),
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.9]})
    stats = [("p50", 50), ("p95", 95), ("p99", 99), ("max", 100)]
    x = np.arange(len(stats))
    w = 0.38
    for ax, (mode, title) in zip(axes[:3], WORKLOADS):
        for j, kind in enumerate(("baseline", "final")):
            ds = data[mode][kind]
            vals = [np.mean([np.percentile(d["ttft"], p) for d in ds]) for _, p in stats]
            bars = ax.bar(x + (j - 0.5) * w, vals, w - 0.04, color=COLOR[kind],
                          label=LABEL[kind])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 6, f"{v:.0f}",
                        ha="center", va="bottom", fontsize=8, color=INK)
        ax.axhline(TTFT_SLO_MS, color=MUTED, linewidth=1, linestyle="--")
        ax.text(len(stats) - 0.5, TTFT_SLO_MS + 6, "TTFT SLO", ha="right",
                va="bottom", fontsize=8, color=MUTED)
        ax.set_xticks(x, [s for s, _ in stats])
        ax.set_ylim(0, 460)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_ylabel("TTFT (ms)" if mode == WORKLOADS[0][0] else "")
        style(ax)
    ax = axes[3]
    xi = np.arange(len(WORKLOADS))
    for j, kind in enumerate(("baseline", "final")):
        vals = [np.mean([d["ft_rate"] for d in data[m][kind]]) for m, _ in WORKLOADS]
        bars = ax.bar(xi + (j - 0.5) * w, vals, w - 0.04, color=COLOR[kind])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 10, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(xi, ["tight", "loose", "nutanix"])
    ax.set_ylabel("finetuning throughput (tok/s)")
    ax.set_title("FT throughput", fontsize=11, color=INK, loc="left")
    style(ax)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Qwen3-14B TP=2 co-serving — baseline vs final "
                 "(mean of the repeat runs; TTFT satisfaction 100 % in all)",
                 y=1.10, fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(_OUT / "cleanup_compare_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── distributions: per workload TTFT CDF, worst-TBT CDF, cumulative FT ──
    fig, axes = plt.subplots(len(WORKLOADS), 3, figsize=(16, 4 * len(WORKLOADS)))
    for row, (mode, title) in enumerate(WORKLOADS):
        a_ttft, a_tbt, a_ft = axes[row]
        for kind in ("baseline", "final"):
            ds = data[mode][kind]
            cdf(a_ttft, [d["ttft"] for d in ds], kind)
            cdf(a_tbt, [d["wtbt"] for d in ds], kind)
            for i, d in enumerate(ds):
                a_ft.plot(d["ft_t"], d["ft_cum"] / 1e3, color=COLOR[kind],
                          linewidth=2, alpha=1.0 if i == 0 else 0.45,
                          label=LABEL[kind] if i == 0 else None)
        a_ttft.axvline(TTFT_SLO_MS, color=MUTED, linewidth=1, linestyle="--")
        a_ttft.set_xlim(0, 450)
        a_ttft.set_xlabel("TTFT (ms)")
        a_ttft.set_ylabel(f"{title}\n% of requests")
        a_tbt.set_xlim(0, 350)
        a_tbt.set_xlabel("worst time-between-tokens per request (ms)")
        a_ft.set_xlabel("time in the timeline (s)")
        a_ft.set_ylabel("FT tokens trained (thousands)")
        if row == 0:
            a_ttft.set_title("TTFT distribution (dashed: SLO)", fontsize=11,
                             color=INK, loc="left")
            a_tbt.set_title("Worst TBT per request", fontsize=11, color=INK,
                            loc="left")
            a_ft.set_title("Cumulative finetuning tokens", fontsize=11, color=INK,
                           loc="left")
            a_ttft.legend(frameon=False, fontsize=9, loc="lower right")
        for a in (a_ttft, a_tbt, a_ft):
            style(a)
    fig.suptitle("Baseline vs final — distributions (lighter line: repeat run)",
                 y=1.0, fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(_OUT / "cleanup_compare_distributions.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {_OUT / 'cleanup_compare_summary.png'}\n"
          f"wrote {_OUT / 'cleanup_compare_distributions.png'}")


if __name__ == "__main__":
    main()
