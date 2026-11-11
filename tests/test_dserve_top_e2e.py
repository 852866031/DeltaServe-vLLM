#!/usr/bin/env python
"""dserve-top end to end: a real server with telemetry OFF in its config.

Launches a template with the telemetry section removed, starts dserve-top (it
must switch recording on at runtime), drives load with ``vllm bench serve``
while finetuning runs, and checks:

  - POST /telemetry/start happened: the server records into dserve-top's dir;
  - the feed through dserve-top carries real steps: folded decode runs, mixed /
    FT steps with FT admission codes from the scheduler, GPU ms, graph flags;
    per-second TTFT / TPOT rows; FT status; a memory snapshot per GPU;
  - the recording dir gets run.json, memory_rank<r>.json, requests.csv,
    backward.csv, summary.jsonl and the rolling steps.*.jsonl.gz;
  - a headless Chrome screenshot of the page under load;
  - on exit dserve-top stops the recording (the server reports inactive) and
    the server keeps serving.

    python tests/test_dserve_top_e2e.py            # Qwen3-0.6B, GPU 0
    python tests/test_dserve_top_e2e.py --tp2      # Qwen3-14B TP=2, both GPUs
"""

import argparse
import glob
import gzip
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parent.parent
_PORT = 8023
_TTY = sys.stdout.isatty()
_G, _R, _X = ("\033[92m", "\033[91m", "\033[0m") if _TTY else ("", "", "")
_passed = _failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    print(f"  {_G + '[PASS]' if cond else _R + '[FAIL]'}{_X} {name}  {detail}", flush=True)
    _passed += bool(cond)
    _failed += not cond
    return cond


def get(url, timeout=10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, {}
    except Exception as e:  # noqa: BLE001
        return None, {"error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp2", action="store_true")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    tmpl = _REPO / "configs" / ("template_qwen3_14b_tp2.yaml" if a.tp2
                                else "template_qwen3_0.6b.yaml")
    raw = yaml.safe_load(tmpl.read_text())
    tp = int(raw["parallel"]["tensor_parallel_size"])
    print(f"=== dserve-top end to end: {tmpl.name} (TP={tp}), telemetry off ===", flush=True)
    tmp = Path(tempfile.mkdtemp(prefix="ds-top-e2e-"))
    for sec in raw.values():
        if isinstance(sec, dict):
            for k, v in list(sec.items()):
                if ("path" in k or k.endswith("_dir")) and isinstance(v, str) \
                        and not os.path.isabs(v):
                    sec[k] = str((tmpl.parent / v).resolve())
    raw.pop("telemetry", None)
    raw["finetune_param"].update(adapter_save_path=str(tmp / "ck"),
                                 adapter_save_every_backwards=0,
                                 adapter_save_every_epoch=False)
    cfg = tmp / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(raw))
    env = dict(os.environ, HF_HUB_OFFLINE="1", VLLM_USE_FLASHINFER_SAMPLER="0",
               VLLM_ALLOW_RUNTIME_LORA_UPDATING="True")
    env.setdefault("HF_HOME", "/mnt/storage/huggingface")
    slog = open(tmp / "server.log", "w")
    server = subprocess.Popen(
        [sys.executable, str(_REPO / "eval-tp/launch_deltaserve.py"), "--config", str(cfg),
         "--port", str(_PORT)], stdout=slog, stderr=subprocess.STDOUT, env=env,
        start_new_session=True)
    top = None
    rec = tmp / "rec"
    try:
        t0 = time.monotonic()
        while get(f"http://127.0.0.1:{_PORT}/health", 3)[0] != 200:
            if server.poll() is not None or time.monotonic() - t0 > 1200:
                check(f"server healthy ({tmp}/server.log)", False)
                return
            time.sleep(3)
        check("server healthy", True)
        _, d = get(f"http://127.0.0.1:{_PORT}/telemetry/data?steps_since=0&series_since=0")
        check("telemetry off at launch", d.get("active") is False, json.dumps(d)[:80])
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            tport = s.getsockname()[1]
        top = subprocess.Popen(
            [sys.executable, "-m", "deltaserve.top", "--server", f"http://127.0.0.1:{_PORT}",
             "--port", str(tport), "--dir", str(rec), "--interval", "5"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        out = ""
        while "connected;" not in out:
            line = top.stdout.readline()
            if not line:
                break
            out += line
        check("dserve-top started the recording and printed the guide",
              "started telemetry recording" in out and "ssh -N -L" in out, str(rec))
        _, d = get(f"http://127.0.0.1:{_PORT}/telemetry/data?steps_since=0&series_since=0")
        check("server now records into dserve-top's dir", d.get("active")
              and d.get("runtime") and d.get("dir") == str(rec), d.get("dir"))
        _, models = get(f"http://127.0.0.1:{_PORT}/v1/models")
        bench = subprocess.Popen(
            [str(Path(sys.executable).parent / "vllm"), "bench", "serve", "--backend", "vllm",
             "--port", str(_PORT), "--model", raw["model"]["model"],
             "--served-model-name", models["data"][0]["id"], "--dataset-name", "random",
             "--random-input-len", "512", "--random-output-len", "96", "--ignore-eos",
             "--num-prompts", "300", "--max-concurrency", "8", "--request-rate", "6",
             "--seed", "0"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        time.sleep(25)
        shot = tmp / "dserve-top.png"
        chrome = shutil.which("google-chrome") or shutil.which("chromium")
        if chrome:
            subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
                            "--hide-scrollbars", f"--screenshot={shot}",
                            "--window-size=1440,1250", "--virtual-time-budget=4000",
                            f"http://127.0.0.1:{tport}/"], capture_output=True, timeout=60)
            check("headless Chrome screenshot under load", shot.exists()
                  and shot.stat().st_size > 20000, str(shot))
        _, st = get(f"http://127.0.0.1:{tport}/api/state?steps_since=0&series_since=0")
        steps = st.get("steps", [])
        kinds = {e["k"] for e in steps}
        codes = {e.get("a") for e in steps if e["k"] == "x"}
        runs = [e for e in steps if e["k"] == "d"]
        check("feed has folded decode runs and mixed / FT steps",
              "d" in kinds and "x" in kinds and max((e["n"] for e in runs), default=0) > 1,
              f"{len(steps)} entries, longest run d×{max((e['n'] for e in runs), default=0)}")
        check("steps carry FT admission codes from the scheduler",
              bool(codes - {""}) and codes <= set("ASFCBTXEDK") | {""},
              str(sorted(c for c in codes if c)))
        xs = [e for e in steps if e["k"] == "x"]
        check("steps carry GPU ms and graph mode",
              xs and all(e["ms"] > 0 for e in xs) and any(e["g"] is not None for e in xs))
        ser = [r for r in st.get("series", []) if r["nt"]]
        check("per-second TTFT / TPOT rows", ser and any(r["tpot50"] for r in st["series"]),
              f"{len(st.get('series', []))} rows, {len(ser)} with first tokens")
        mem = (st.get("window") or {}).get("memory") or {}
        check(f"memory snapshot for {tp} GPU(s)", all(f"rank{r}" in mem for r in range(tp)),
              str(sorted(mem)))
        check("FT status in the feed", (st.get("finetuning") or {}).get("backwards_done", 0) > 0)
        check("history for the scrubber: status snapshots + memory",
              len(st.get("snaps", [])) >= 5 and st.get("mems"), f"{len(st.get('snaps', []))} snaps")
        with urllib.request.urlopen(f"http://127.0.0.1:{tport}/api/export", timeout=60) as r:
            html, cd = r.read(), r.headers.get("Content-Disposition", "")
        import base64
        import re
        m = re.search(rb'id="ds-embedded" type="application/gzip-base64">([^<]*)<', html)
        emb = json.loads(gzip.decompress(base64.b64decode(m.group(1)))) if m else {}
        check("single-HTML export of the live run (≤ 50 MB, steps + series + memory)",
              cd.startswith("attachment;") and len(html) <= 50 * 2**20
              and any(e["k"] == "x" and e.get("a") for e in emb.get("steps", []))
              and emb.get("series") and emb.get("mems"),
              f"{len(html) / 1024:.0f} KiB, {len(emb.get('steps', []))} steps")
        if chrome:
            ex = tmp / "export.html"
            ex.write_bytes(html)
            dom = subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
                                  "--virtual-time-budget=5000", "--dump-dom", f"file://{ex}"],
                                 capture_output=True, text=True, timeout=60).stdout
            check("the export replays offline", "replay · exported" in dom
                  and "GPU slot" in dom and "waiting for steps" not in dom, str(ex))
        bench.wait(600)
        # On-demand adapters: register a second inference adapter at runtime
        # and use it — with max_loras=2 (FT pinned) it must take slot 1.
        lo = ((st.get("window") or {}).get("memory") or {}).get("rank0", {}).get("lora", {})
        names = {a["name"] for a in lo.get("adapters", [])}
        check("adapter names tracked (FT + inference)",
              "__finetune__" in names and len(names) >= 2, str(sorted(names)))
        src = raw["adapters"]["lora_path_0"]
        req = urllib.request.Request(
            f"http://127.0.0.1:{_PORT}/v1/load_lora_adapter", method="POST",
            data=json.dumps({"lora_name": "toy-copy", "lora_path": src}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            loaded = r.status == 200
        req = urllib.request.Request(
            f"http://127.0.0.1:{_PORT}/v1/completions", method="POST",
            data=json.dumps({"model": "toy-copy", "prompt": "Hello", "max_tokens": 8}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            answered = r.status == 200
        time.sleep(14)                          # dserve-top refreshes every 10 s
        _, st2 = get(f"http://127.0.0.1:{tport}/api/state?steps_since=999999999&series_since=1e18")
        lo2 = ((st2.get("window") or {}).get("memory") or {}).get("rank0", {}).get("lora", {})
        slots = {x["slot"]: x["name"] for x in lo2.get("in_slots", [])}
        served = {m["id"] for m in st2.get("models", []) if m.get("parent")}
        check("runtime-loaded adapter served and shown in its GPU slot",
              loaded and answered and "toy-copy" in slots.values()
              and "__finetune__" in slots.values() and "toy-copy" in served,
              f"slots {slots} · served {sorted(served)}")
    finally:
        if top is not None:
            top.send_signal(signal.SIGINT)
            try:
                top.wait(30)
            except subprocess.TimeoutExpired:
                top.kill()
        time.sleep(2)
        _, d = get(f"http://127.0.0.1:{_PORT}/telemetry/data?steps_since=0&series_since=0")
        check("dserve-top's exit stopped the recording", d.get("active") is False)
        code, _ = get(f"http://127.0.0.1:{_PORT}/v1/models")
        check("server still serving", code == 200)
        try:
            os.killpg(server.pid, signal.SIGINT)
            server.wait(120)
        except Exception:  # noqa: BLE001
            os.killpg(server.pid, signal.SIGKILL)
            server.wait()
        slog.close()

    for name in ["run.json", "requests.csv", "backward.csv", "summary.jsonl"] + \
            [f"memory_rank{r}.json" for r in range(tp)]:
        check(f"recorded {name}", (rec / name).exists())
    segs = sorted(glob.glob(str(rec / "steps.*.jsonl.gz")))
    lines = []
    for p in segs:
        with gzip.open(p, "rt") as f:
            lines += [json.loads(x) for x in f]
    check("rolling step log written and readable",
          segs and any(e["k"] == "x" and e.get("a") for e in lines),
          f"{len(segs)} segment(s), {len(lines)} entries")
    run = json.load(open(rec / "run.json")) if (rec / "run.json").exists() else {}
    check("run.json describes the run", run.get("tensor_parallel_size") == tp
          and "admit_codes" in run, json.dumps({k: run.get(k) for k in
                                                ("model", "tensor_parallel_size")}))
    if a.keep or _failed:
        print(f"(kept {tmp})")
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    color = _R if _failed else _G
    print(f"\n{color}{_passed} passed, {_failed} failed{_X}")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
