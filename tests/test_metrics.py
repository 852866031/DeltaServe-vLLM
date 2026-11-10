#!/usr/bin/env python
"""System metrics recorder (telemetry.metrics_dir), on CPU.

  A. requests: TTFT / inter-token gaps / e2e / tokens computed exactly from a
     simulated clock; FT requests (never in the outputs) are not counted;
     requests.csv rows.
  B. steps: inference-only / mixed / FT-only classification, the mixed steps'
     GPU time split by tokens, empty steps ignored.
  C. backwards: backward.csv rows (GPU / wall ms, tokens/s, child memory).
  D. summary: windows close every interval, percentiles + SLO attainment,
     throughputs, GPU shares, totals; memory files merged; close() flushes.
  E. wiring: the child's ack carries gpu_ms / wall_ms / samples (+ mem when on);
     the coordinator forwards acks; metrics_dir resolves relative to the YAML;
     GET /system_metrics.

    python tests/test_metrics.py
"""

import csv
import json
import os
import sys
import tempfile
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bwd_harness as H  # noqa: E402

sys.path[:] = [p for p in sys.path
               if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]

from deltaserve.metrics import MetricsRecorder  # noqa: E402

C = H.Checker(tol=1e-9)


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def out(rid, n=1, finish=None):
    return types.SimpleNamespace(request_id=rid, new_token_ids=[7] * n,
                                 finish_reason=finish)


def outputs(*os_):
    return {0: types.SimpleNamespace(outputs=list(os_))}


def feats(t_in=0, t_ft=0, b_d=0):
    return types.SimpleNamespace(t_in=t_in, t_ft=t_ft, b_d=b_d)


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def part_a_to_d(root):
    print("A-D. recorder")
    clk = Clock()
    m = MetricsRecorder(root, interval_s=10, ttft_slo=0.4, max_tbt_slo=0.05, clock=clk,
                        start_ticker=False)
    reqs = {"a": types.SimpleNamespace(arrival_time=1000.0, num_prompt_tokens=50),
            "b": types.SimpleNamespace(arrival_time=1000.5, num_prompt_tokens=20)}
    # a: arrives 1000.0, first token 1000.3, then 1000.32, 1000.36 (finished)
    # b: arrives 1000.5, first token 1001.0 (TTFT 0.5, misses the SLO), 1 more
    clk.t = 1000.3
    m.on_outputs(outputs(out("a")), reqs)
    clk.t = 1000.32
    m.on_outputs(outputs(out("a")), reqs)
    clk.t = 1000.36
    m.on_outputs(outputs(out("a", finish=types.SimpleNamespace(name="LENGTH"))), reqs)
    clk.t = 1001.0
    m.on_outputs(outputs(out("b", n=2)), reqs)     # first step yields 2 tokens
    clk.t = 1001.1
    m.on_outputs(outputs(out("b", n=1, finish="stop")), reqs)
    m.on_outputs(outputs(out("never", n=0)), reqs)  # no tokens: not tracked
    C.ok("in-flight requests cleared", not m._live)
    # steps
    m.on_step(feats(t_in=100), 0.010)                   # inference prefill
    m.on_step(feats(b_d=8), 0.004)                      # decode
    m.on_step(feats(t_in=300, t_ft=200, b_d=0), 0.030)  # mixed: 100 inf + 200 FT
    m.on_step(feats(t_in=50, t_ft=50, b_d=6), 0.012)    # mixed: 6 decode + 50 FT
    m.on_step(feats(t_in=256, t_ft=256), 0.020)         # FT-only
    m.on_step(feats(), 0.001)                           # empty step: ignored
    m.on_step(feats(t_in=10), 0.0)                      # untimed: ignored
    # backwards
    for i, (gpu, wall, loss) in enumerate([(80.0, 120.0, 2.0), (90.0, 300.0, 1.0)]):
        m.on_backward({"event": "activations_processed", "n": 256, "loss": loss,
                       "samples": 12, "gpu_ms": gpu, "wall_ms": wall,
                       "mem": {"alloc": 2 << 30, "peak": 3 << 30, "reserved": 4 << 30,
                               "lora_train_state": 1 << 28}}, epoch=1)
    m.on_backward({"event": "something_else"})          # not an ack: ignored
    with open(os.path.join(root, "memory_rank0.json"), "w") as f:
        json.dump({"gib": {"kv_cache": 10.0}}, f)
    clk.t = 1005.0
    m.maybe_flush()
    C.ok("no summary before the interval", not os.path.exists(
        os.path.join(root, "summary.jsonl")) or not open(
        os.path.join(root, "summary.jsonl")).read())
    clk.t = 1010.0
    snap = m.snapshot()
    m.maybe_flush()
    inf, ft, g = snap["inference"], snap["finetune"], snap["gpu_time_s"]
    C.ok("TTFT p50 / max", abs(inf["ttft_s"]["p50"] - 0.3) < 1e-9
         and abs(inf["ttft_s"]["max"] - 0.5) < 1e-9, json.dumps(inf["ttft_s"]))
    C.ok("TTFT SLO attainment 1/2", inf["ttft_slo_attainment"] == 0.5)
    # gaps: a: 0.02, 0.04; b: 0.1  → TBT SLO 0.05: 2/3
    C.ok("inter-token gaps (max exact, percentiles within a 2 % bucket)",
         abs(inf["tbt_s"]["max"] - 0.1) < 1e-9 and inf["tbt_s"]["tokens"] == 3
         and 0.04 <= inf["tbt_s"]["p50"] <= 0.04 * 1.0201
         and abs(inf["tbt_slo_attainment"] - 0.6667) < 1e-4, json.dumps(inf["tbt_s"]))
    C.ok("throughputs over the 10 s window",
         inf["prompt_tokens_per_s"] == 7.0 and inf["output_tokens_per_s"] == 0.6
         and ft["trained_tokens_per_s"] == 51.2
         and ft["ft_forward_tokens_per_s"] == 50.6, json.dumps(ft))
    fw = g["forward"]
    C.ok("step kinds", fw["inference_only"]["steps"] == 2 and fw["mixed"]["steps"] == 2
         and fw["ft_only"]["steps"] == 1 and abs(g["forward_total"] - 0.076) < 1e-9,
         json.dumps(fw))
    # inference GPU = 0.014 + 0.030*100/300 + 0.012*6/56 ; FT = rest
    inf_gpu = 0.014 + 0.010 + 0.012 * 6 / 56
    C.ok("mixed steps split by tokens",
         abs(g["forward_inference_share"] - round(inf_gpu / 0.076, 4)) < 1e-4
         and abs(g["forward_inference_share"] + g["forward_ft_share"] - 1) < 2e-4,
         f"inf={g['forward_inference_share']} ft={g['forward_ft_share']}")
    C.ok("backward means", ft["backwards"] == 2 and ft["backward_gpu_ms_mean"] == 85.0
         and ft["backward_wall_ms_mean"] == 210.0 and ft["loss_mean"] == 1.5
         and abs(g["backward"] - 0.17) < 1e-9 and g["backward_busy_fraction"] == 0.017)
    mem = snap["memory"]
    C.ok("memory: worker file + child", mem.get("rank0", {}).get("gib", {}).get(
        "kv_cache") == 10.0 and mem["backward_child_rank0_gib"]["peak"] == 3.0
         and mem["backward_child_rank0_gib"]["lora_train_state"] == 0.25,
         json.dumps(mem)[:120])
    t = snap["totals"]
    C.ok("totals", t["requests"] == 2 and t["output_tokens"] == 6
         and t["ttft_slo_met"] == 1 and t["backwards"] == 2
         and t["trained_tokens"] == 512, json.dumps(t))
    # a new window starts empty
    clk.t = 1012.0
    s2 = m.snapshot()
    C.ok("next window starts empty", s2["inference"]["requests_finished"] == 0
         and s2["finetune"]["backwards"] == 0 and s2["totals"]["requests"] == 2)
    cur = m.current()
    C.ok("current(): last + in-progress window",
         cur["last_window"]["inference"]["requests_finished"] == 2
         and cur["current_window"]["window_s"] == 2.0)
    m.close()
    r = rows(os.path.join(root, "requests.csv"))
    C.ok("requests.csv rows", len(r) == 2 and r[0]["request_id"] == "a"
         and r[0]["prompt_tokens"] == "50" and r[0]["output_tokens"] == "3"
         and float(r[0]["ttft_s"]) == 0.3 and abs(float(r[0]["tbt_mean_s"]) - 0.03) < 1e-9
         and float(r[0]["e2e_s"]) == 0.36 and r[0]["finish_reason"] == "LENGTH"
         and r[1]["output_tokens"] == "3" and r[1]["tbt_max_s"] == "0.1", str(r[0]))
    b = rows(os.path.join(root, "backward.csv"))
    C.ok("backward.csv rows", len(b) == 2 and b[0]["gpu_ms"] == "80.0"
         and b[0]["wall_ms"] == "120.0" and b[0]["train_tokens_per_gpu_s"] == "3200.0"
         and b[1]["child_mem_peak_gib"] == "3.0" and b[0]["epoch"] == "1", str(b[0]))
    lines = open(os.path.join(root, "summary.jsonl")).read().splitlines()
    C.ok("summary.jsonl: one line per window + the final flush",
         len(lines) == 2 and json.loads(lines[0])["inference"]["requests_finished"] == 2)


def part_e(root):
    print("\nE. wiring")
    # child ack fields, through the real cycle handler (CPU)
    from deltaserve.bwd_services.llama3 import Llama3BackwardService
    svc, embed = H.build_synthetic_service(
        Llama3BackwardService, hidden=32, n_layers=2, Hq=4, Hkv=2, Hd=8,
        inter=64, vocab=40, r=4, lr=1e-3)
    svc.shared["meta"]["metrics"] = True
    seq = [9, 7]
    ids = torch.randint(0, 40, (16,), generator=torch.Generator().manual_seed(0))
    svc.activations = H.forward_capture(svc, embed, ids, seq, [0, 9])
    sent = []
    svc._handle_process_activations(types.SimpleNamespace(send=sent.append),
                                    {"n": 16, "sample_lens": seq, "epoch": 0})
    a = sent[0]
    C.ok("child ack: gpu_ms / wall_ms / samples", a["samples"] == 2
         and a["gpu_ms"] is not None and a["gpu_ms"] > 0
         and a["wall_ms"] >= a["gpu_ms"] * 0.5, json.dumps(
             {k: a[k] for k in ("samples", "gpu_ms", "wall_ms")}))
    C.ok("child knows its training-state size",
         svc._train_state_bytes == 16 * sum(p.numel() for ld in svc.lora.values()
                                            for pd in ld.values() for p in pd.values()))
    # coordinator → recorder
    from deltaserve.coordinator import FinetuneCoordinator
    got = []
    c = FinetuneCoordinator(capacity=64)
    c.metrics = types.SimpleNamespace(on_backward=lambda r, e: got.append((r["n"], e)))
    c.current_epoch = 3
    c.note_backward_done({"event": "activations_processed", "n": 16, "loss": 1.0,
                          "gpu_ms": 5.0})
    C.ok("coordinator forwards acks + keeps last gpu_ms",
         got == [(16, 3)] and c.status()["last_backward_gpu_ms"] == 5.0)
    # config: metrics_dir is a relative *_dir path
    from deltaserve.config_loader import SECTION_OF, finetune_kwargs, load_yaml_config
    y = os.path.join(root, "cfg.yaml")
    with open(y, "w") as f:
        f.write("finetune_param:\n  enable_finetuning: true\n"
                "telemetry:\n  metrics_dir: out/m\n  metrics_interval_s: 5\n")
    kw = finetune_kwargs(load_yaml_config(y, verbose=False))
    C.ok("metrics_dir resolved against the YAML, in telemetry",
         kw["metrics_dir"] == os.path.join(root, "out", "m")
         and kw["metrics_interval_s"] == 5 and SECTION_OF["metrics_dir"] == "telemetry")
    from deltaserve.config import FinetuneConfig
    C.ok("off by default", FinetuneConfig().metrics_dir is None)
    # route
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from deltaserve import api

    for enabled, code in ((True, 200), (False, 409)):
        app = FastAPI()

        class Core:
            async def call_utility_async(self, method, *args):
                assert method == "deltaserve_metrics_summary"
                return {"enabled": enabled}
        app.state.engine_client = types.SimpleNamespace(engine_core=Core())
        app.state.args = types.SimpleNamespace(
            additional_config={"deltaserve": {"enable_finetuning": True}})
        api.attach_router(app)
        r = TestClient(app).get("/system_metrics")
        C.ok(f"GET /system_metrics ({'on' if enabled else 'off'}) → {code}",
             r.status_code == code)
    from deltaserve.engine_core import metrics_summary
    C.ok("utility without a FinetuneScheduler",
         metrics_summary(types.SimpleNamespace(scheduler=object()))["enabled"] is False)


def part_idle(root):
    print("\nH. idle engine + shutdown")
    import time as _t
    m = MetricsRecorder(root, interval_s=0.5)          # real clock, ticker on
    p = os.path.join(root, "summary.jsonl")
    _t.sleep(1.2)                                      # "profiling pass"
    C.ok("nothing is written before serving starts", not os.path.exists(p)
         or not open(p).read())
    m.start_serving()
    m.on_step(feats(t_in=10), 0.01)
    _t.sleep(1.6)                                      # no engine steps at all
    n_idle = len(open(p).read().splitlines()) if os.path.exists(p) else 0
    C.ok("the ticker closes windows while the engine idles", n_idle >= 2,
         f"{n_idle} windows in 1.6 s")
    m.on_step(feats(t_in=20), 0.02)
    m.final_flush()
    last = json.loads(open(p).read().splitlines()[-1])
    C.ok("final_flush writes the window in progress, synchronously",
         last.get("final") is True
         and last["gpu_time_s"]["forward"]["inference_only"]["steps"] == 1)
    m.close()


def part_hist():
    print("\nG. histogram percentiles")
    import random

    from deltaserve.metrics import LogHistogram
    rng = random.Random(0)
    xs = [rng.lognormvariate(-4, 1) for _ in range(20000)]
    h = LogHistogram(0.05)
    for x in xs:
        h.add(x)
    v = sorted(xs)
    worst = max(abs(h.pct(q) - v[max(0, -(-q * len(v) // 100) - 1)])
                / v[max(0, -(-q * len(v) // 100) - 1)] for q in (50, 90, 99))
    C.ok("p50 / p90 / p99 within 2 % of exact", worst <= 0.0201, f"worst {worst:.4f}")
    C.ok("count, mean, max, SLO share exact",
         h.n == len(xs) and abs(h.mean() - sum(xs) / len(xs)) < 1e-12
         and h.max == max(xs)
         and h.at_or_below == sum(1 for x in xs if x <= 0.05))


def part_f(root):
    print("\nF. hot-path cost")
    import time as _t
    m = MetricsRecorder(root, interval_s=10.0, max_tbt_slo=0.05, start_ticker=False)
    clk_t = [0.0]
    m._clock = lambda: clk_t[0]
    reqs = {f"r{i}": types.SimpleNamespace(arrival_time=0.0, num_prompt_tokens=100)
            for i in range(256)}
    batch = outputs(*[out(f"r{i}") for i in range(256)])
    m.on_outputs(batch, reqs)                    # first tokens
    f = feats(t_in=0, b_d=256)
    n = 2000
    t0 = _t.perf_counter()
    worst_us = 0.0
    for k in range(n):
        clk_t[0] = k * 0.015                     # 15 ms steps: a window per 667
        s0 = _t.perf_counter()
        m.on_outputs(batch, reqs)
        m.on_step(f, 0.015)
        m.maybe_flush()
        worst_us = max(worst_us, (_t.perf_counter() - s0) * 1e6)
    per_step_us = (_t.perf_counter() - t0) / n * 1e6
    C.ok("256-request decode step: recording cost < 500 µs (vs a ~15 ms step)",
         per_step_us < 500, f"{per_step_us:.0f} µs/step")
    C.ok("a window close (170k gaps) stays < 5 ms", worst_us < 5000,
         f"worst step {worst_us:.0f} µs")
    m.close()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as root:
        part_a_to_d(os.path.join(root, "m"))
        part_e(root)
        part_hist()
        part_idle(os.path.join(root, "idle"))
        part_f(os.path.join(root, "perf"))
    C.finish()
