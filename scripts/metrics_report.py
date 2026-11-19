#!/usr/bin/env python
"""Summarize a telemetry.metrics_dir: text report (+ an optional figure).

    python scripts/metrics_report.py logs/qwen3-14b/metrics
    python scripts/metrics_report.py logs/qwen3-14b/metrics --png report.png

Reads requests.csv, backward.csv, summary.jsonl and memory_rank*.json (see
deltaserve-plugin/deltaserve/metrics.py) and prints, over the whole run:
inference latency (TTFT / TBT percentiles, SLO attainment) and throughput,
finetuning throughput and per-step LoRA backward time, the GPU time split, and
GPU memory per rank. ``--png`` adds four panels: TTFT per request, throughput,
GPU busy time, memory.
"""

import argparse
import csv
import json
import math
import os
import sys

INF, FT, BWD = "#2a78d6", "#eb6834", "#1baf7a"           # fixed per entity
MEM = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def pct(vals, q):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    return v[min(len(v) - 1, max(0, math.ceil(q / 100 * len(v)) - 1))]


def fmt(x, unit="", k=3):
    return "—" if x is None else f"{x:.{k}f}{unit}"


def load(root):
    req = read_csv(os.path.join(root, "requests.csv"))
    bwd = read_csv(os.path.join(root, "backward.csv"))
    summ = []
    p = os.path.join(root, "summary.jsonl")
    if os.path.exists(p):
        with open(p) as f:
            summ = [json.loads(line) for line in f if line.strip()]
    mem = {}
    for name in sorted(os.listdir(root)):
        if name.startswith("memory_rank") and name.endswith(".json"):
            with open(os.path.join(root, name)) as f:
                mem[name[len("memory_"):-5]] = json.load(f)["gib"]
    return req, bwd, summ, mem


def report(root) -> dict:
    req, bwd, summ, mem = load(root)
    ttft = [fnum(r["ttft_s"]) for r in req]
    tbt_max = [fnum(r["tbt_max_s"]) for r in req]
    tbt_mean = [fnum(r["tbt_mean_s"]) for r in req]
    span = None
    if summ:
        span = summ[-1]["unix_time"] - (summ[0]["unix_time"] - summ[0]["window_s"])
    last = summ[-1] if summ else {}
    slo = (last.get("inference") or {}).get("ttft_slo")
    out_tok = sum(int(r["output_tokens"]) for r in req)
    in_tok = sum(int(r["prompt_tokens"]) for r in req)
    gpu = [fnum(b["gpu_ms"]) for b in bwd]
    wall = [fnum(b["wall_ms"]) for b in bwd]
    trained = sum(int(b["tokens"]) for b in bwd)
    fw = {"inference_only": 0.0, "mixed": 0.0, "ft_only": 0.0}
    inf_g = ft_g = bwd_g = 0.0
    for s in summ:
        g = s["gpu_time_s"]
        for k in fw:
            fw[k] += g["forward"][k]["gpu_s"]
        tot = g["forward_total"]
        if tot:
            inf_g += tot * (g["forward_inference_share"] or 0)
            ft_g += tot * (g["forward_ft_share"] or 0)
        bwd_g += g["backward"]
    r = {
        "requests": len(req), "span_s": span,
        "ttft_p50": pct(ttft, 50), "ttft_p90": pct(ttft, 90), "ttft_p99": pct(ttft, 99),
        "ttft_max": pct(ttft, 100), "ttft_slo": slo,
        "ttft_slo_attainment": (sum(1 for x in ttft if x is not None and x <= slo) / len(ttft)
                                if slo and ttft else None),
        "tbt_mean_p50": pct(tbt_mean, 50), "tbt_max_p99": pct(tbt_max, 99),
        "prompt_tokens": in_tok, "output_tokens": out_tok,
        "backwards": len(bwd), "trained_tokens": trained,
        "bwd_gpu_ms_p50": pct(gpu, 50), "bwd_gpu_ms_p99": pct(gpu, 99),
        "bwd_wall_ms_p50": pct(wall, 50),
        "loss_first": fnum(bwd[0]["loss"]) if bwd else None,
        "loss_last": fnum(bwd[-1]["loss"]) if bwd else None,
        "fwd_gpu_s": fw, "inf_gpu_s": inf_g, "ft_fwd_gpu_s": ft_g, "bwd_gpu_s": bwd_g,
        "memory": mem,
        "child_mem": (last.get("memory") or {}).get("backward_child_rank0_gib"),
    }
    return r


def print_report(root, r) -> None:
    span = r["span_s"]
    per_s = (lambda n: f"{n / span:.1f}/s") if span else (lambda n: "—")
    print(f"metrics: {root}   (span {fmt(span, ' s', 0)})\n")
    print("inference")
    print(f"  requests finished   {r['requests']}   prompt {r['prompt_tokens']} tok "
          f"({per_s(r['prompt_tokens'])}), output {r['output_tokens']} tok "
          f"({per_s(r['output_tokens'])})")
    print(f"  TTFT                p50 {fmt(r['ttft_p50'], ' s')}  p90 {fmt(r['ttft_p90'], ' s')}"
          f"  p99 {fmt(r['ttft_p99'], ' s')}  max {fmt(r['ttft_max'], ' s')}")
    if r["ttft_slo"]:
        att = r["ttft_slo_attainment"]
        print(f"  TTFT SLO {r['ttft_slo']} s    met by {fmt(att * 100 if att is not None else None, ' %', 1)}")
    print(f"  TBT                 per-request mean p50 {fmt(r['tbt_mean_p50'], ' s', 4)}"
          f"   per-request max p99 {fmt(r['tbt_max_p99'], ' s', 4)}")
    print("finetuning")
    print(f"  backward cycles     {r['backwards']}   trained {r['trained_tokens']} tok "
          f"({per_s(r['trained_tokens'])})")
    print(f"  LoRA step time      GPU p50 {fmt(r['bwd_gpu_ms_p50'], ' ms', 1)}"
          f"  p99 {fmt(r['bwd_gpu_ms_p99'], ' ms', 1)}   wall p50 "
          f"{fmt(r['bwd_wall_ms_p50'], ' ms', 1)} (incl. pauses for inference)")
    print(f"  loss                first {fmt(r['loss_first'], '', 4)}  last {fmt(r['loss_last'], '', 4)}")
    fw = r["fwd_gpu_s"]
    tot = r["inf_gpu_s"] + r["ft_fwd_gpu_s"] + r["bwd_gpu_s"]
    print("GPU time (rank 0, seconds of kernel time)")
    print(f"  forward steps       inference-only {fw['inference_only']:.1f}  mixed "
          f"{fw['mixed']:.1f}  FT-only {fw['ft_only']:.1f}")
    if tot:
        print(f"  by owner            inference {r['inf_gpu_s']:.1f} ({r['inf_gpu_s'] / tot:.0%})"
              f"   FT forward {r['ft_fwd_gpu_s']:.1f} ({r['ft_fwd_gpu_s'] / tot:.0%})"
              f"   LoRA backward {r['bwd_gpu_s']:.1f} ({r['bwd_gpu_s'] / tot:.0%})")
        print("                      (mixed steps split by token share; forward and"
              " backward overlap in time)")
    print("GPU memory (GiB)")
    for rank, g in r["memory"].items():
        keys = ("model_weights", "kv_cache", "ft_activation_buffers")
        other = {k: v for k, v in g.items() if k not in keys}
        print(f"  {rank:<8} weights {g.get('model_weights')}  KV cache {g.get('kv_cache')}"
              f"  FT buffers {g.get('ft_activation_buffers')}  | "
              + "  ".join(f"{k} {v}" for k, v in other.items()))
    if r["child_mem"]:
        c = r["child_mem"]
        print(f"  backward child (rank 0): allocated {c.get('alloc')}  peak {c.get('peak')}"
              f"  reserved {c.get('reserved')}  of which LoRA masters+grads+Adam "
              f"{c.get('lora_train_state')}")


def plot(root, r, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    req, bwd, summ, mem = load(root)
    t0 = (summ[0]["unix_time"] - summ[0]["window_s"]) if summ else 0
    fig, axs = plt.subplots(2, 2, figsize=(12, 7.5), dpi=120)
    for ax in axs.flat:
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
    fig.patch.set_facecolor("#fcfcfb")

    ax = axs[0, 0]
    xs = [float(q["finish_unix_time"]) - float(q["e2e_s"]) - t0 for q in req]
    ys = [float(q["ttft_s"]) for q in req]
    ax.scatter(xs, ys, s=10, color=INF, lw=0)
    if r["ttft_slo"]:
        ax.axhline(r["ttft_slo"], color=INK2, lw=1, ls=(0, (4, 3)))
        ax.text(ax.get_xlim()[1] if xs else 1, r["ttft_slo"], f" SLO {r['ttft_slo']} s",
                va="bottom", ha="right", color=INK2, fontsize=9)
    ax.set_title("TTFT per request (s)", loc="left", color=INK, fontsize=11)
    ax.set_xlabel("arrival (s since start)", color=INK2, fontsize=9)

    ax = axs[0, 1]
    ts = [s["unix_time"] - t0 for s in summ]
    ax.plot(ts, [s["inference"]["output_tokens_per_s"] for s in summ], color=INF, lw=2,
            label="inference output")
    ax.plot(ts, [s["finetune"]["trained_tokens_per_s"] for s in summ], color=FT, lw=2,
            label="finetuning trained")
    ax.set_title("throughput (tokens/s)", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    ax.set_xlabel("time (s)", color=INK2, fontsize=9)

    ax = axs[1, 0]
    def share(s, key):
        g = s["gpu_time_s"]
        return (g["forward_total"] * (g[key] or 0)) / s["window_s"]
    ax.plot(ts, [share(s, "forward_inference_share") for s in summ], color=INF, lw=2,
            label="inference forward")
    ax.plot(ts, [share(s, "forward_ft_share") for s in summ], color=FT, lw=2,
            label="FT forward")
    ax.plot(ts, [s["gpu_time_s"]["backward_busy_fraction"] for s in summ], color=BWD, lw=2,
            label="LoRA backward")
    ax.set_title("GPU busy fraction (rank 0, kernel time / wall)", loc="left", color=INK,
                 fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    ax.set_xlabel("time (s)", color=INK2, fontsize=9)

    ax = axs[1, 1]
    parts = [("model_weights", "weights"), ("kv_cache", "KV cache"),
             ("ft_activation_buffers", "FT buffers")]
    rows = []
    for rank, g in mem.items():
        other = (g.get("inference_process_reserved", 0) - sum(g.get(k, 0) for k, _ in parts))
        rows.append((f"{rank} inference", [g.get(k, 0) for k, _ in parts] + [max(0, other), 0]))
    if r["child_mem"]:
        rows.append(("rank0 backward child",
                     [0, 0, 0, 0, r["child_mem"].get("reserved", 0)]))
    labels = [p[1] for p in parts] + ["other inference", "backward child"]
    for i, (name, vals) in enumerate(rows):
        left = 0
        for j, v in enumerate(vals):
            if v <= 0:
                continue
            ax.barh(i, v - 0.04, left=left, color=MEM[j], height=0.55,
                    label=labels[j] if i == 0 or (j == 4) else None)
            if v >= 0.06 * max(sum(vv) for _, vv in rows):   # label only if it fits
                ax.text(left + v / 2, i, f"{v:.1f}", ha="center", va="center",
                        color="white", fontsize=8)
            left += v
    ax.set_yticks(range(len(rows)), [n for n, _ in rows], color=INK2, fontsize=9)
    ax.invert_yaxis()
    ax.set_title("GPU memory (GiB)", loc="left", color=INK, fontsize=11)
    h, l = ax.get_legend_handles_labels()
    seen = dict(zip(l, h))
    ax.legend(seen.values(), seen.keys(), frameon=False, fontsize=8, labelcolor=INK2,
              loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    print(f"\nfigure: {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metrics_dir")
    ap.add_argument("--png", help="also write a 4-panel figure here")
    ap.add_argument("--json", action="store_true", help="print the numbers as JSON")
    a = ap.parse_args()
    if not os.path.isdir(a.metrics_dir):
        print(f"no such directory: {a.metrics_dir}", file=sys.stderr)
        return 1
    r = report(a.metrics_dir)
    if a.json:
        print(json.dumps(r, indent=2, default=str))
    else:
        print_report(a.metrics_dir, r)
    if a.png:
        plot(a.metrics_dir, r, a.png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
