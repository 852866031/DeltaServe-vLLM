#!/usr/bin/env python
"""Plots for the arxiv 0.4 rps trace (reads output/, writes plots/).
Blue = inference, orange = finetuning / co-serving, grey = inference-only baseline.

    python eval-arxiv/plot_arxiv.py          # trace_co (+ trace_inf when present)
"""
import csv, datetime as dt, json, statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"; PLOTS = HERE / "plots"; PLOTS.mkdir(exist_ok=True)
C_INF, C_CO, C_ALT, C_TEXT = "#1f77b4", "#ff7f0e", "#7f7f7f", "#333333"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#bbbbbb", "legend.frameon": False, "axes.axisbelow": True})

def rows(name):
    with open(OUT / name) as f:
        return list(csv.DictReader(f))

def f(r, k):
    try: return float(r[k])
    except (ValueError, KeyError, TypeError): return None

def load_trace(mode):
    res = rows(f"trace_{mode}.csv")
    meta = json.load(open(OUT / f"trace_{mode}_meta.json"))
    return res, meta

# ── per-mode timeline: requests, tokens/s, step times ────────────────────────
def plot_timeline(mode):
    res, meta = load_trace(mode); t0 = meta["t0_wall"]
    T = max(f(r, "arrival_s") + f(r, "e2e_s") for r in res) + 2.0
    has_steps = (OUT / f"step_trace_trace_{mode}.csv").exists()
    n_ax = 3 if has_steps else 1
    fig, axes = plt.subplots(n_ax, 1, figsize=(11, 2.6 * n_ax + 0.8), sharex=True,
                             gridspec_kw={"height_ratios": [1.0] * n_ax})
    axes = np.atleast_1d(axes)
    # (1) requests: bar from arrival to first token (TTFT) then to completion (decode)
    a = axes[0]
    for r in res:
        y = int(r["req"]); ta = f(r, "arrival_s"); tt = f(r, "ttft_s"); te = f(r, "e2e_s")
        a.plot([ta, ta + tt], [y, y], color=C_INF, lw=3, solid_capstyle="butt")
        a.plot([ta + tt, ta + te], [y, y], color=C_INF, lw=3, alpha=0.35, solid_capstyle="butt")
        a.plot(ta, y, marker="|", color=C_TEXT, ms=9)
    a.plot([], [], color=C_INF, lw=3, label="prefill (arrival → first token)")
    a.plot([], [], color=C_INF, lw=3, alpha=0.35, label="decode (→ last token)")
    a.set_ylabel("request #"); a.set_ylim(-1, len(res)); a.legend(loc="lower right", ncol=2, fontsize=9)
    tt = [f(r, "ttft_s") for r in res]
    a.set_title(f"arxiv trace — {meta['rps']} rps Poisson, {meta['n']} requests, ~{meta['len']} prompt tokens — "
                f"{'co-serving' if mode == 'co' else 'inference only'}  "
                f"(TTFT p50 {st.median(tt):.2f} s, max {max(tt):.2f} s)")
    if has_steps:
        tr = [r for r in rows(f"step_trace_trace_{mode}.csv") if r["kind"] == "step"
              and -1 <= f(r, "t_sched") - t0 <= T]
        bin_s = 0.5; nb = int(np.ceil(T / bin_s)) + 2
        inf_tok = np.zeros(nb); ft_tok = np.zeros(nb)
        for r in tr:
            b = int((f(r, "t_sched") - t0) / bin_s) + 1
            tft = f(r, "t_ft") or 0.0; tin = f(r, "t_in") or 0.0
            ft_tok[b] += tft; inf_tok[b] += max(0.0, tin - tft)
        centers = (np.arange(nb) - 0.5) * bin_s
        a2 = axes[1]
        a2.bar(centers, ft_tok / bin_s, width=bin_s, color=C_CO, label="finetuning tokens / s")
        a2.bar(centers, inf_tok / bin_s, width=bin_s, bottom=ft_tok / bin_s, color=C_INF, label="inference prefill tokens / s")
        a2.set_ylabel("tokens / s (0.5 s bins)"); a2.legend(loc="upper right", ncol=2, fontsize=9)
        a3 = axes[2]
        pts = defaultdict(list)
        for r in tr:
            tft = f(r, "t_ft") or 0.0; tin = f(r, "t_in") or 0.0; bd = f(r, "b_d") or 0.0; act = f(r, "actual")
            if act is None: continue
            k = ("FT-only step" if (tft > 0 and tin == tft and bd == 0) else
                 "inference prefill chunk" if (tin - tft >= 1900) else "mixed / decode")
            pts[k].append((f(r, "t_sched") - t0, act))
        ymax = max((y for v in pts.values() for _, y in v), default=0.5)
        for k, col, mk, s in (("inference prefill chunk", C_INF, "o", 14), ("FT-only step", C_CO, "^", 14),
                              ("mixed / decode", C_ALT, ".", 8)):
            if pts[k]:
                xs, ys = zip(*pts[k]); a3.scatter(xs, ys, s=s, color=col, marker=mk, label=k, alpha=0.85)
        bl_path = OUT / f"bwd_log_trace_{mode}.csv"
        if bl_path.exists():
            cyc = [dt.datetime.fromisoformat(r["timestamp"]).timestamp() - t0 for r in rows(bl_path.name)]
            cyc = [t for t in cyc if -1 <= t <= T]
            a3.scatter(cyc, [-0.02 * ymax / 0.5] * len(cyc), marker="|", s=40, color=C_CO, label=f"backward cycle (n={len(cyc)})")
        a3.set_ylabel("step time (s)"); a3.set_ylim(-0.05 * ymax / 0.5, ymax * 1.25); a3.legend(loc="upper right", ncol=4, fontsize=8)
    axes[-1].set_xlabel("time since first arrival (s)"); axes[-1].set_xlim(-1, T)
    fig.tight_layout(); fig.savefig(PLOTS / f"trace_timeline_{mode}.png", dpi=150); plt.close(fig)

# ── co vs inference-only per request ─────────────────────────────────────────
def plot_compare():
    co, _ = load_trace("co"); inf, _ = load_trace("inf")
    n = min(len(co), len(inf)); x = np.arange(n); w = 0.38
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, key, lab in ((a1, "ttft_s", "TTFT (s)"), (a2, "e2e_s", "E2E latency (s)")):
        vi = [f(r, key) for r in inf[:n]]; vc = [f(r, key) for r in co[:n]]
        ax.bar(x - w / 2, vi, w, color=C_ALT, label="inference only")
        ax.bar(x + w / 2, vc, w, color=C_CO, label="co-serving")
        ax.set_xticks(x, [str(i) for i in x]); ax.set_xlabel("request # (same arrival schedule)"); ax.set_ylabel(lab)
        ax.set_title(f"{lab.split(' (')[0]}: p50 {st.median(vi):.2f} → {st.median(vc):.2f} s, "
                     f"max {max(vi):.2f} → {max(vc):.2f} s", fontsize=10)
        ax.legend(loc="upper left", fontsize=9)
    fig.suptitle("arxiv 0.4 rps trace — inference only vs co-serving", fontsize=11)
    fig.tight_layout(); fig.savefig(PLOTS / "trace_compare.png", dpi=150); plt.close(fig)

if __name__ == "__main__":
    done = []
    for mode in ("co", "inf"):
        if (OUT / f"trace_{mode}.csv").exists():
            plot_timeline(mode); done.append(f"trace_timeline_{mode}.png")
    if (OUT / "trace_co.csv").exists() and (OUT / "trace_inf.csv").exists():
        plot_compare(); done.append("trace_compare.png")
    print(f"[plot] wrote {done}")
