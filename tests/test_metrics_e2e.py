#!/usr/bin/env python
"""System metrics end to end: a real server from a template, real traffic.

Launches a template (default Qwen3-0.6B; ``--tp2`` = the Qwen3-14B TP=2
template) with telemetry.metrics_dir in a temp dir and a 5 s summary window,
sends streaming requests (ignore_eos, fixed output length) while finetuning
runs, then checks what was recorded against what the client saw:

  - requests.csv: one row per request; output tokens == max_tokens; prompt
    tokens == the server's usage; server TTFT ≤ client TTFT and close to it
    (the client also pays HTTP + detokenization).
  - backward.csv: one row per backward the status reported; GPU ms > 0,
    wall ms ≥ GPU ms, child memory filled, peak ≥ allocated.
  - summary.jsonl: windows with forward / backward GPU time, inference + FT
    shares summing to 1, throughputs > 0; GET /system_metrics live.
  - memory_rank<r>.json per rank: weights ≈ checkpoint size / TP, KV cache ≈
    the "Available KV cache memory" vLLM logged, FT buffers > 0.
  - scripts/metrics_report.py prints the report and draws the figure.

    python tests/test_metrics_e2e.py           # Qwen3-0.6B, GPU 0
    python tests/test_metrics_e2e.py --tp2     # Qwen3-14B, both GPUs
"""

import argparse
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parent.parent
_PORT = 8019
_TTY = sys.stdout.isatty()
_G, _R, _X = ("\033[92m", "\033[91m", "\033[0m") if _TTY else ("", "", "")
_passed = _failed = 0
N_REQ, MAX_TOK, CONC = 40, 32, 4


def check(name, cond, detail=""):
    global _passed, _failed
    print(f"  {_G + '[PASS]' if cond else _R + '[FAIL]'}{_X} {name}  {detail}", flush=True)
    _passed += bool(cond)
    _failed += not cond
    return cond


def http(method, path, body=None, timeout=60):
    req = urllib.request.Request(
        f"http://127.0.0.1:{_PORT}{path}", method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            try:
                return r.status, json.loads(body or b"{}")
            except ValueError:
                return r.status, {"error": "not JSON", "body": body[:300].decode(errors="replace")}
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")
    except Exception as e:  # noqa: BLE001
        return None, {"error": f"{type(e).__name__}: {e}"}


def stream_one(model, prompt, res, i):
    """One streaming completion: client TTFT, usage, response id."""
    body = {"model": model, "prompt": prompt, "max_tokens": MAX_TOK,
            "temperature": 0, "ignore_eos": True, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(f"http://127.0.0.1:{_PORT}/v1/completions",
                                 data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first, usage, rid = None, None, None
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            rid = rid or d.get("id")
            if first is None and d.get("choices") and d["choices"][0].get("text"):
                first = time.time()
            if d.get("usage"):
                usage = d["usage"]
    res[i] = {"t0": t0, "ttft": (first - t0) if first else None, "usage": usage,
              "id": rid}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp2", action="store_true")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    tmpl = _REPO / "configs" / ("template_qwen3_14b_tp2.yaml" if a.tp2
                                else "template_qwen3_0.6b.yaml")
    raw = yaml.safe_load(tmpl.read_text())
    tp = int(raw["parallel"]["tensor_parallel_size"])
    print(f"=== metrics end to end: {tmpl.name} (TP={tp}) ===", flush=True)
    tmp = Path(tempfile.mkdtemp(prefix="ds-metrics-"))
    for sec in raw.values():
        if isinstance(sec, dict):
            for k, v in list(sec.items()):
                if ("path" in k or k.endswith("_dir")) and isinstance(v, str) \
                        and not os.path.isabs(v):
                    sec[k] = str((tmpl.parent / v).resolve())
    mdir = tmp / "metrics"
    raw["telemetry"] = {"metrics_dir": str(mdir), "metrics_interval_s": 5}
    raw["finetune_param"]["adapter_save_path"] = str(tmp / "ck")
    cfg = tmp / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(raw))
    env = dict(os.environ, HF_HUB_OFFLINE="1", VLLM_USE_FLASHINFER_SAMPLER="0")
    env.setdefault("HF_HOME", "/mnt/storage/huggingface")
    log = open(tmp / "server.log", "w")
    proc = subprocess.Popen(
        [sys.executable, str(_REPO / "eval-tp/launch_deltaserve.py"),
         "--config", str(cfg), "--port", str(_PORT)],
        stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    st_before_stop = {}
    try:
        deadline = time.monotonic() + 1200
        while http("GET", "/health", timeout=3)[0] != 200:
            if proc.poll() is not None or time.monotonic() > deadline:
                check(f"server healthy (log {tmp / 'server.log'})", False)
                return
            time.sleep(3)
        check("server healthy", True)
        _, models = http("GET", "/v1/models")
        model = models["data"][0]["id"]
        lines = [l.strip() for l in open(_REPO / "data/alpaca_1000.txt") if l.strip()]
        prompts = [" ".join(lines[i * 3:(i * 3) + 1 + (i % 4)]) for i in range(N_REQ)]
        res = [None] * N_REQ
        sem = threading.Semaphore(CONC)

        def run(i):
            with sem:
                try:
                    stream_one(model, prompts[i], res, i)
                except Exception as e:  # noqa: BLE001
                    res[i] = {"error": str(e)}
                time.sleep(0.2)
        ths = [threading.Thread(target=run, args=(i,)) for i in range(N_REQ)]
        for t in ths:
            t.start()
            time.sleep(0.15)
        for t in ths:
            t.join()
        ok = [x for x in res if x and x.get("ttft") is not None]
        check(f"{N_REQ} streaming requests answered", len(ok) == N_REQ,
              f"{len(ok)} ok; errors={[x.get('error') for x in res if x and 'error' in x][:2]}")
        time.sleep(12)                          # ≥ 2 summary windows, backwards
        code, ms = http("GET", "/system_metrics")
        check("GET /system_metrics live", code == 200 and ms.get("enabled")
              and ms["current_window"]["totals"]["requests"] == N_REQ,
              json.dumps(ms.get("current_window", {}).get("totals", ms))[:160])
        _, st_before_stop = http("GET", "/finetuning_status")
        http("POST", "/stop_finetuning")
        for _ in range(60):
            _, st = http("GET", "/finetuning_status")
            if not st.get("backward_in_flight"):
                break
            time.sleep(1)
        st_before_stop = st
    finally:
        try:
            os.killpg(proc.pid, signal.SIGINT)
            proc.wait(120)
        except Exception:  # noqa: BLE001
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        log.close()

    sys.path.insert(0, str(_REPO / "scripts"))
    import csv
    reqs = list(csv.DictReader(open(mdir / "requests.csv")))
    check("requests.csv: one row per request", len(reqs) == N_REQ, f"{len(reqs)} rows")
    check("output tokens == max_tokens (ignore_eos)",
          all(int(r["output_tokens"]) == MAX_TOK for r in reqs))
    by_id = {}
    for r in reqs:
        for x in ok:
            if x["id"] and r["request_id"].startswith(x["id"]):
                by_id[r["request_id"]] = x
    check("server rows map to client requests by id", len(by_id) == N_REQ,
          f"{len(by_id)} mapped (e.g. {reqs[0]['request_id']} vs {ok[0]['id']})")
    if by_id:
        prompt_ok = all(int(r["prompt_tokens"]) == by_id[r["request_id"]]["usage"]["prompt_tokens"]
                        for r in reqs if r["request_id"] in by_id)
        check("prompt tokens == the server's usage", prompt_ok)
        diffs = [by_id[r["request_id"]]["ttft"] - float(r["ttft_s"])
                 for r in reqs if r["request_id"] in by_id]
        check("server TTFT ≤ client TTFT, and close (HTTP + detokenize only)",
              min(diffs) > -0.005 and sorted(diffs)[len(diffs) // 2] < 0.05,
              f"client−server: min {min(diffs) * 1000:.1f} ms, median "
              f"{sorted(diffs)[len(diffs) // 2] * 1000:.1f} ms, max {max(diffs) * 1000:.1f} ms")
    tbt = [float(r["tbt_mean_s"]) for r in reqs if r["tbt_mean_s"]]
    check("TBT recorded for every request", len(tbt) == N_REQ and all(x > 0 for x in tbt),
          f"mean of means {sum(tbt) / max(1, len(tbt)) * 1000:.1f} ms")
    bwd = list(csv.DictReader(open(mdir / "backward.csv")))
    check("backward.csv: one row per backward", len(bwd) == st_before_stop.get("backwards_done"),
          f"{len(bwd)} rows vs status {st_before_stop.get('backwards_done')}")
    check("backward rows: GPU ms > 0, wall ≥ GPU, child memory",
          bwd and all(float(b["gpu_ms"]) > 0 and float(b["wall_ms"]) >= 0.95 * float(b["gpu_ms"])
                      and float(b["child_mem_peak_gib"]) >= float(b["child_mem_alloc_gib"]) > 0
                      for b in bwd),
          f"gpu_ms p50 {sorted(float(b['gpu_ms']) for b in bwd)[len(bwd) // 2]:.1f}" if bwd else "")
    summ = [json.loads(l) for l in open(mdir / "summary.jsonl")]
    fwd = [s for s in summ if s["gpu_time_s"]["forward_total"] > 0]
    check("summary.jsonl windows", len(summ) >= 2 and fwd, f"{len(summ)} windows")
    shares_ok = all(abs((s["gpu_time_s"]["forward_inference_share"] or 0)
                        + (s["gpu_time_s"]["forward_ft_share"] or 0) - 1) < 1e-3 for s in fwd)
    check("inference + FT forward shares sum to 1", shares_ok)
    tot = summ[-1]["totals"]
    check("the stop wrote a final window (totals: requests, every backward, GPU time)",
          summ[-1].get("final") is True
          and tot["requests"] == N_REQ and tot["backwards"] == len(bwd)
          and tot["fwd_gpu_s"] > 0 and tot["bwd_gpu_s"] > 0, json.dumps(tot))
    slog = (tmp / "server.log").read_text()
    kv_logged = [float(x) for x in re.findall(r"Available KV cache memory: ([0-9.]+) GiB", slog)]
    snap = glob.glob(f"{os.environ.get('HF_HOME', '/mnt/storage/huggingface')}/hub/models--"
                     f"{raw['model']['model'].replace('/', '--')}/snapshots/*")[0]
    ckpt_gib = sum(os.path.getsize(p) for p in glob.glob(f"{snap}/*.safetensors")) / 2**30
    for rank in range(tp):
        p = mdir / f"memory_rank{rank}.json"
        if not check(f"memory_rank{rank}.json written", p.exists()):
            continue
        g = json.load(open(p))["gib"]
        check(f"rank {rank}: weights ≈ checkpoint / TP",
              abs(g["model_weights"] - ckpt_gib / tp) / (ckpt_gib / tp) < 0.12,
              f"{g['model_weights']} vs {ckpt_gib / tp:.2f} GiB")
        check(f"rank {rank}: KV cache ≈ what vLLM reserved",
              kv_logged and abs(g["kv_cache"] - kv_logged[0]) / kv_logged[0] < 0.08,
              f"{g['kv_cache']} vs logged {kv_logged[:1]} GiB")
        check(f"rank {rank}: FT activation buffers > 0", g["ft_activation_buffers"] > 0,
              f"{g['ft_activation_buffers']} GiB")
    rep = subprocess.run([sys.executable, str(_REPO / "scripts/metrics_report.py"),
                          str(mdir), "--png", str(tmp / "report.png")],
                         capture_output=True, text=True)
    check("metrics_report.py: report + figure", rep.returncode == 0
          and "TTFT" in rep.stdout and (tmp / "report.png").exists(),
          rep.stderr[-200:])
    print(rep.stdout)
    keep = a.keep or _failed
    if keep:
        print(f"(kept {tmp})")
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    color = _R if _failed else _G
    print(f"\n{color}{_passed} passed, {_failed} failed{_X}")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
