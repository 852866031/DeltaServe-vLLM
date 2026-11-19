#!/usr/bin/env python
"""Overhead of telemetry.metrics_dir, measured two ways in ~20 minutes.

1. Interleaved on/off A/B (ABBAAB order, so drift cancels) on a template, under
   a fixed closed-loop load from ``vllm bench serve`` while finetuning runs:
   request throughput, TTFT / TPOT / ITL percentiles, FT trained tokens/s.
   Qwen3-0.6B (the default) is host-bound, so engine-loop CPU overhead shows
   there most: a worst case for the larger models.
2. Accounting in the metrics-on runs: every recorder call is timed in the
   EngineCore (a sitecustomize wraps the methods), giving the CPU time the
   recorder adds per engine step and as a share of wall time.

    python eval-tp/bench_metrics_overhead.py                        # 0.6B, 3+3 runs
    python eval-tp/bench_metrics_overhead.py --config configs/template_qwen3_14b_tp2.yaml \\
        --order A --concurrency 16 --num-prompts 400                # accounting only
    python eval-tp/bench_metrics_overhead.py --order ABTTBA          # + dserve-top attached

Arms: A = metrics on, B = off, T = metrics on with a ``dserve-top`` polling the
server the whole run (its live feed ~4×/s, memory / adapter snapshot every 10 s).
"""

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parent.parent
PY = sys.executable

_SITE = r'''
import json, os, threading, time
try:
    import deltaserve.metrics as _m
    _stats, _lock, _t0 = {}, threading.Lock(), [None]
    def _wrap(name):
        orig = getattr(_m.MetricsRecorder, name)
        def w(self, *a, **k):
            s = time.perf_counter()
            try:
                return orig(self, *a, **k)
            finally:
                d = time.perf_counter() - s
                key = name + ("@ticker" if threading.current_thread().name == "ds-metrics-tick" else "")
                with _lock:
                    if _t0[0] is None:
                        _t0[0] = time.time()
                    st = _stats.setdefault(key, [0, 0.0, 0.0])
                    st[0] += 1; st[1] += d; st[2] = max(st[2], d)
        setattr(_m.MetricsRecorder, name, w)
    for _n in ("on_outputs", "on_step", "on_backward", "maybe_flush", "snapshot", "current",
               "live", "memory"):
        _wrap(_n)
    def _dump():
        path = os.environ["DS_ACCT_FILE"]
        while True:
            time.sleep(2)
            with _lock:
                d = {"since": _t0[0], "now": time.time(), "stats": dict(_stats)}
            if _stats:
                with open(path + ".tmp", "w") as f:
                    json.dump(d, f)
                os.replace(path + ".tmp", path)
    if os.environ.get("DS_ACCT_FILE"):
        threading.Thread(target=_dump, daemon=True).start()
except Exception:
    pass
'''


def http(path, port, method="GET", timeout=10):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except Exception:  # noqa: BLE001
        return None


def config_for(template: Path, tmp: Path, metrics: bool, tag: str) -> Path:
    raw = yaml.safe_load(template.read_text())
    for sec in raw.values():
        if isinstance(sec, dict):
            for k, v in list(sec.items()):
                if ("path" in k or k.endswith("_dir")) and isinstance(v, str) \
                        and not os.path.isabs(v):
                    sec[k] = str((template.parent / v).resolve())
    fp = raw["finetune_param"]
    fp.update(adapter_save_path=str(tmp / "ck"), adapter_save_every_backwards=0,
              adapter_save_every_epoch=False)
    raw["telemetry"] = ({"metrics_dir": str(tmp / f"metrics_{tag}"),
                         "metrics_interval_s": 10} if metrics else {})
    out = tmp / f"cfg_{tag}.yaml"
    out.write_text(yaml.safe_dump(raw))
    return out


def run_one(template, tmp, metrics, tag, port, args, top=False):
    cfg = config_for(template, tmp, metrics, tag)
    env = dict(os.environ, HF_HUB_OFFLINE="1", VLLM_USE_FLASHINFER_SAMPLER="0")
    env.setdefault("HF_HOME", "/mnt/storage/huggingface")
    if metrics:
        site = tmp / "site"
        site.mkdir(exist_ok=True)
        (site / "sitecustomize.py").write_text(_SITE)
        env["PYTHONPATH"] = f"{site}:{env.get('PYTHONPATH', '')}"
        env["DS_ACCT_FILE"] = str(tmp / f"acct_{tag}.json")
    log = open(tmp / f"server_{tag}.log", "w")
    top_proc = None
    proc = subprocess.Popen([PY, str(_REPO / "eval-tp/launch_deltaserve.py"),
                             "--config", str(cfg), "--port", str(port)],
                            stdout=log, stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)
    try:
        t0 = time.monotonic()
        while http("/health", port) is None:
            if proc.poll() is not None or time.monotonic() - t0 > 1200:
                raise RuntimeError(f"server {tag} did not come up ({tmp}/server_{tag}.log)")
            time.sleep(2)
        while (http("/finetuning_status", port) or {}).get("backwards_done", 0) < 3:
            time.sleep(1)                      # FT warmed up in both arms
        if top:                                # the dashboard's poller, no browser needed
            top_proc = subprocess.Popen(
                [PY, "-m", "deltaserve.top", "--server", f"http://127.0.0.1:{port}",
                 "--port", str(port + 1000), "--dir", str(tmp / f"top_{tag}")],
                stdout=open(tmp / f"top_{tag}.log", "w"), stderr=subprocess.STDOUT, env=env)
            time.sleep(5)
        model = http("/v1/models", port)["data"][0]["id"]
        raw = yaml.safe_load(cfg.read_text())
        st0 = http("/finetuning_status", port)
        t_b0 = time.time()
        bench = subprocess.run(
            [str(Path(PY).parent / "vllm"), "bench", "serve", "--backend", "vllm",
             "--port", str(port), "--model", raw["model"]["model"],
             "--served-model-name", model, "--dataset-name", "random",
             "--random-input-len", str(args.input_len),
             "--random-output-len", str(args.output_len), "--ignore-eos",
             "--num-prompts", str(args.num_prompts),
             "--max-concurrency", str(args.concurrency), "--request-rate", "inf",
             "--percentile-metrics", "ttft,tpot,itl", "--metric-percentiles", "50,90,99",
             "--seed", "0", "--save-result", "--result-dir", str(tmp),
             "--result-filename", f"bench_{tag}.json"],
            capture_output=True, text=True, env=env, timeout=1800)
        t_b1 = time.time()
        st1 = http("/finetuning_status", port)
        if bench.returncode != 0:
            raise RuntimeError(bench.stdout[-1500:] + bench.stderr[-1500:])
        res = json.load(open(tmp / f"bench_{tag}.json"))
        time.sleep(3)
        acct = None
        if metrics and (tmp / f"acct_{tag}.json").exists():
            acct = json.load(open(tmp / f"acct_{tag}.json"))
        return {
            "tag": tag, "metrics": metrics, "top": top,
            "req_per_s": res["request_throughput"],
            "out_tok_per_s": res["output_throughput"],
            **{f"{m}_{p}": res.get(f"p{p}_{m}_ms") for m in ("ttft", "tpot", "itl")
               for p in (50, 90, 99)},
            "ft_tok_per_s": (st1["tokens_trained"] - st0["tokens_trained"]) / (t_b1 - t_b0),
            "bench_s": t_b1 - t_b0, "acct": acct,
        }
    finally:
        if top_proc is not None:
            top_proc.send_signal(signal.SIGINT)
            try:
                top_proc.wait(15)
            except Exception:  # noqa: BLE001
                top_proc.kill()
        try:
            os.killpg(proc.pid, signal.SIGINT)
            proc.wait(120)
        except Exception:  # noqa: BLE001
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(_REPO / "configs/template_qwen3_0.6b.yaml"))
    ap.add_argument("--order", default="ABBAAB",
                    help="A = metrics on, B = off, T = metrics on + dserve-top attached")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--num-prompts", type=int, default=800)
    ap.add_argument("--input-len", type=int, default=512)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--port", type=int, default=8021)
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="ds-ovh-"))
    print(f"work dir {tmp}", flush=True)
    runs = []
    for i, arm in enumerate(args.order):
        r = run_one(Path(args.config), tmp, arm in "AT", f"{i}{arm}", args.port, args,
                    top=arm == "T")
        runs.append(r)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                          for k, v in r.items() if k != "acct"}), flush=True)
    json.dump(runs, open(tmp / "runs.json", "w"), indent=1)
    keys = ["req_per_s", "out_tok_per_s", "ttft_50", "ttft_99", "tpot_50", "tpot_99",
            "itl_50", "itl_99", "ft_tok_per_s"]
    off = [r for r in runs if not r["metrics"]]
    for name, on in (("on", [r for r in runs if r["metrics"] and not r["top"]]),
                     ("on+top", [r for r in runs if r["top"]])):
        if not (on and off):
            continue
        print(f"\n{'metric':<14}{name + ' (mean ± sd)':>22}{'off (mean ± sd)':>22}"
              f"{'Δ vs off':>12}")
        for k in keys:
            a = [r[k] for r in on if r[k] is not None]
            b = [r[k] for r in off if r[k] is not None]
            sa = statistics.pstdev(a) if len(a) > 1 else 0
            sb = statistics.pstdev(b) if len(b) > 1 else 0
            ma, mb = statistics.mean(a), statistics.mean(b)
            print(f"{k:<14}{ma:>14.2f} ± {sa:<6.2f}{mb:>14.2f} ± {sb:<6.2f}"
                  f"{(ma - mb) / mb * 100:>+11.2f}%")
    for r in (r for r in runs if r["metrics"]):
        a = r["acct"]
        if not a:
            continue
        st = a["stats"]
        wall = a["now"] - a["since"]
        eng = {k: v for k, v in st.items() if not k.endswith("@ticker")}
        eng_s = sum(v[1] for v in eng.values())
        steps = st.get("on_outputs", [0])[0]
        tick_s = sum(v[1] for k, v in st.items() if k.endswith("@ticker"))
        worst = max(v[2] for v in st.values())
        print(f"\n[{r['tag']}] recorder CPU on the engine thread: {eng_s * 1000:.0f} ms over "
              f"{wall:.0f} s wall = {eng_s / wall * 100:.3f} %  |  {steps} steps → "
              f"{eng_s / max(1, steps) * 1e6:.1f} µs/step  |  ticker thread "
              f"{tick_s * 1000:.0f} ms  |  slowest single call {worst * 1000:.2f} ms")
        for k, v in sorted(st.items()):
            print(f"      {k:<22} calls {v[0]:>8}  total {v[1] * 1000:>8.1f} ms  "
                  f"mean {v[1] / v[0] * 1e6:>7.1f} µs  max {v[2] * 1000:>6.2f} ms")


if __name__ == "__main__":
    main()
