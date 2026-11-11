#!/usr/bin/env python
"""dserve-top and the live feed behind it, on CPU.

  A. LiveFeed: plain decode steps fold into one growing ``d×N`` entry that any
     other step closes; the open run is re-sent with its new count; seqs stay
     contiguous through ring trimming; per-second rows (TTFT / TPOT
     percentiles, token rates, GPU split, loss) incl. idle seconds.
  B. Rolling step log: gzip segments rotate, the oldest are deleted to stay
     under the cap, every line is valid JSON.
  C. dserve-top against a fake dserve server (a real MetricsRecorder behind
     the same /telemetry routes as deltaserve/api.py, fed synthetic traffic):
     it starts the recording, prints the SSH guide (from SSH_CONNECTION),
     serves the page, merges the feed (incl. the re-sent open decode run),
     and on exit stops only a recording it started. A headless Chrome
     screenshot of the page is saved when Chrome is available.

    python tests/test_dserve_top.py [--keep]
"""

import gzip
import json
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bwd_harness as H  # noqa: E402

sys.path[:] = [p for p in sys.path
               if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]

from deltaserve.live import LiveFeed, RollingGz  # noqa: E402
from deltaserve.metrics import MetricsRecorder  # noqa: E402

C = H.Checker(tol=1e-9)
_REPO = Path(__file__).resolve().parent.parent


def feats(t_in=0, t_ft=0, b_d=0, admit=("", 0), bwd=False):
    f = types.SimpleNamespace(t_in=t_in, t_ft=t_ft, b_d=b_d, bwd=bwd)
    f.admit = admit
    return f


def part_a():
    print("A. live feed")
    log = []
    fd = LiveFeed(write=log.append, steps_ring=50)
    t = 100.0
    for i in range(5):                                   # 5 plain decodes
        fd.step(t + i * .01, feats(b_d=8), 0.004)
    s = fd.steps_since(0)
    C.ok("plain decodes fold into one entry", len(s) == 1 and s[0]["k"] == "d"
         and s[0]["n"] == 5 and abs(s[0]["ms"] - 20.0) < 1e-6 and not log)
    fd.step(t + .1, feats(b_d=9), 0.004)
    again = fd.steps_since(s[0]["s"])
    C.ok("the open run is re-sent with its new count",
         len(again) == 1 and again[0]["s"] == s[0]["s"] and again[0]["n"] == 6)
    fd.step(t + .2, feats(t_in=600, t_ft=256, b_d=4, admit=("S", 256), bwd=True), 0.04,
            graph=True, paused=False)
    st = fd.steps_since(0)
    C.ok("a mixed step closes the run and carries its admission",
         len(st) == 2 and st[1]["k"] == "x" and st[1]["pf"] == 344 and st[1]["ft"] == 256
         and st[1]["a"] == "S" and st[1]["at"] == 256 and st[1]["g"] == 1 and st[1]["bw"] == 1
         and len(log) == 2 and log[0]["k"] == "d" and log[0]["n"] == 6, json.dumps(st[1]))
    fd.step(t + .3, feats(), 0.001)                      # empty: ignored
    fd.preempt(t + .4, feats(t_in=256, t_ft=256))
    C.ok("empty steps ignored, pre-empted steps recorded",
         [e["k"] for e in fd.steps_since(0)] == ["d", "x", "p"])
    for i in range(200):                                 # trim the ring
        fd.step(t + 1 + i, feats(t_in=10), 0.001)
    st = fd.steps_since(0)
    seqs = [e["s"] for e in st]
    C.ok("ring trimmed, seqs contiguous, newest kept",
         len(st) <= 100 and seqs == list(range(seqs[0], seqs[-1] + 1))
         and seqs[-1] == fd.last_seq and fd.steps_since(fd.last_seq)[0]["s"] == fd.last_seq)
    # per-second series
    fs = LiveFeed()
    fs.first_token(10.2, 0.05)
    fs.first_token(10.5, 0.30)
    fs.tokens(10.6, 7)
    fs.finished(10.7, 40, 0.02)
    fs.fwd(10.8, 0.010, 0.020, 128)
    fs.backward(10.9, 256, 0.15, 2.0)
    fs.tick(14.0)                                        # 11, 12, 13 idle
    rows = fs.series_since(0)
    r = rows[0]
    C.ok("per-second row", r["t"] == 10 and r["nt"] == 2 and r["ttft50"] == 0.05
         and r["ttft99"] == 0.3 and r["out"] == 7 and r["inp"] == 40 and r["tr"] == 256
         and r["ftf"] == 128 and r["loss"] == 2.0 and r["fi"] == 0.01, json.dumps(r))
    C.ok("raw TTFTs ride along for client-side percentiles", r["tv"] == [0.05, 0.3])
    many = LiveFeed()
    for i in range(500):
        many.first_token(20.5, i / 1000)
    many.tick(22)
    tv = many.series_since(0)[0]["tv"]
    C.ok("…at most 64 per second, spanning the range", len(tv) <= 64
         and tv[0] == 0.0 and tv[-1] >= 0.49, f"{len(tv)} values")
    C.ok("idle seconds filled", [x["t"] for x in rows] == [10, 11, 12, 13]
         and rows[1]["nt"] == 0 and rows[1]["ttft50"] is None)
    C.ok("series_since", [x["t"] for x in fs.series_since(11)] == [12, 13])


def part_b(root):
    print("\nB. rolling step log")
    g = RollingGz(root, "steps", max_mb=0.5)            # 64 KiB segments
    rnd = random.Random(0)
    for i in range(60000):
        g.write({"k": "x", "t": i * 0.01, "pf": rnd.randint(0, 2048),
                 "ft": rnd.randint(0, 512), "ms": rnd.random() * 50,
                 "a": rnd.choice("ASFBT"), "r": rnd.getrandbits(64)})
    g.close()
    g.rotate()
    segs = sorted(p for p in os.listdir(root) if p.startswith("steps."))
    total = sum(os.path.getsize(os.path.join(root, p)) for p in segs)
    C.ok("rotated into segments, total under the cap",
         len(segs) >= 3 and total <= 0.5 * (1 << 20), f"{len(segs)} segments, {total} B")
    C.ok("oldest segments deleted", not any(p.endswith("000001.jsonl.gz") for p in segs),
         str(segs[:2]))
    lines = 0
    for p in segs:
        with gzip.open(os.path.join(root, p), "rt") as f:
            for line in f:
                json.loads(line)
                lines += 1
    C.ok("every kept line is valid JSON, newest kept", lines > 1000, f"{lines} lines")


class FakeDServe:
    """The /health and /telemetry/* routes of deltaserve/api.py over a real
    MetricsRecorder, plus a thread producing synthetic co-serving traffic."""

    def __init__(self, root, active=False):
        self.root = root
        self.calls = []
        self.rec = None
        if active:
            self._start(os.path.join(root, "from-config"))
        self.stop_evt = threading.Event()
        outer = self

        class Hd(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _j(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                u = urllib.parse.urlparse(self.path)
                outer.calls.append(("GET", u.path))
                if u.path == "/health":
                    return self._j(200, {})
                if u.path == "/telemetry/data":
                    q = urllib.parse.parse_qs(u.query)
                    if outer.rec is None:
                        return self._j(200, {"active": False})
                    d = outer.rec.live(int(q["steps_since"][0]),
                                       float(q["series_since"][0]))
                    return self._j(200, {"active": True, **d, "finetuning": {
                        "started": True, "backwards_done": outer.rec.tot["backwards"],
                        "buffer_fill": 120, "buffer_capacity": 256, "total_epochs": 20,
                        "store_epoch": 2, "corpus_samples": 1000,
                        "samples_trained_this_epoch": 412, "tokens_trained": 1923000,
                        "mean_loss_last_20": 1.67, "admission_open": True,
                        "estimator_fitted": True, "corpus_tokens_per_epoch": 140000}})
                self._j(404, {})

            def do_POST(self):
                u = urllib.parse.urlparse(self.path)
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                outer.calls.append(("POST", u.path, body))
                if u.path == "/telemetry/start":
                    started = outer.rec is None
                    if started:
                        outer._start(body["dir"])
                    return self._j(200, {"active": True, "started": started,
                                         "dir": outer.rec.root})
                if u.path == "/telemetry/stop":
                    stopped = outer.rec is not None and outer.runtime
                    if stopped:
                        outer.rec.close()
                        outer.rec = None
                    return self._j(200, {"stopped": stopped})
                if u.path == "/telemetry/memory":
                    return self._j(200 if outer.rec else 409, {})
                self._j(404, {})

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.httpd = ThreadingHTTPServer(("127.0.0.1", self.port), Hd)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        threading.Thread(target=self._traffic, daemon=True).start()

    def _start(self, root):
        self.runtime = not root.endswith("from-config")
        self.rec = MetricsRecorder(root, interval_s=2, ttft_slo=0.4, max_tbt_slo=0.05,
                                   meta={"model": "Qwen/Qwen3-14B-Base",
                                         "tensor_parallel_size": 2,
                                         "max_num_batched_tokens": 2048,
                                         "admit_codes": {"A": "admitted", "S": "SLO",
                                                         "B": "backward running"}})
        self.rec.start_serving()
        os.makedirs(root, exist_ok=True)
        for r in (0, 1):
            with open(os.path.join(root, f"memory_rank{r}.json"), "w") as f:
                json.dump({"lora": {"slots": 2, "slot_gib": 0.031, "in_slots": [
                               {"slot": 0, "id": 1000, "name": "__finetune__", "role": "FT adapter (trained; pinned)"},
                               {"slot": 1, "id": 1, "name": "qwen3-14b-toy-lora", "role": "inference"}],
                               "adapters": [
                                   {"id": 1000, "name": "__finetune__", "path": "qwen3-14b-toy-lora-ft",
                                    "rank": 16, "cpu_mib": 30.0, "slot": 0, "pinned": True, "role": "finetune"},
                                   {"id": 1, "name": "qwen3-14b-toy-lora", "path": "qwen3-14b-toy-lora",
                                    "rank": 16, "cpu_mib": 30.0, "slot": 1, "pinned": False, "role": "inference"}],
                               "cpu_cached": 2, "max_cpu_loras": 2, "cpu_cached_gib": 0.06},
                           "unix_time": time.time(),
                           "gib": {"model_weights": 13.75, "kv_cache": 3.06,
                                   "ft_activation_buffers": 0.65, "lora_slots": 0.062,
                                   "other_allocated (lm_head copy, temporaries)": 1.94,
                                   "torch_reserved_not_allocated (incl. CUDA graph pools)": 0.76,
                                   "inference_process_reserved": 20.24,
                                   "device_total": 31.35,
                                   "device_used_all_processes": 26.3}}, f)

    def _traffic(self):
        rnd = random.Random(1)
        rid = 0
        live = {}
        while not self.stop_evt.is_set():
            time.sleep(0.02)
            rec = self.rec
            if rec is None:
                continue
            now = time.time()
            if rnd.random() < 0.8:
                rec.on_step(feats(b_d=rnd.randint(4, 16)), 0.015, True, (0, now, 0, False))
            else:
                ft = rnd.choice([0, 128, 256])
                code = rnd.choice("AS") if ft else rnd.choice("BT")
                rec.on_step(feats(t_in=rnd.randint(200, 1500) + ft, t_ft=ft, b_d=6,
                                  admit=(code, ft)), 0.04, False, (0, now, 0, True))
            if rnd.random() < 0.05:
                rid += 1
                live[f"r{rid}"] = types.SimpleNamespace(arrival_time=now - rnd.random() * 0.2,
                                                        num_prompt_tokens=300)
            outs = [types.SimpleNamespace(request_id=k, new_token_ids=[1],
                                          finish_reason="stop" if rnd.random() < 0.03 else None)
                    for k in list(live)]
            rec.on_outputs({0: types.SimpleNamespace(outputs=outs)}, live)
            for o in outs:
                if o.finish_reason:
                    live.pop(o.request_id)
            if rnd.random() < 0.02:
                rec.on_backward({"event": "activations_processed", "n": 256,
                                 "loss": 2 - rid * 0.002, "gpu_ms": 147, "wall_ms": 152,
                                 "mem": {"alloc": 4e8, "peak": 8e8, "reserved": 1.3e9,
                                         "lora_train_state": 2.7e8}})

    def close(self):
        self.stop_evt.set()
        self.httpd.shutdown()
        if self.rec:
            self.rec.close()


def get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
        return r.status, r.read()


def run_top(fake, root, extra_env=None):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(os.environ, **(extra_env or {}))
    p = subprocess.Popen([sys.executable, "-m", "deltaserve.top", "--server",
                          f"http://127.0.0.1:{fake.port}", "--port", str(port),
                          "--dir", os.path.join(root, "rec"), "--interval", "2"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env=env)
    out = []
    t0 = time.time()
    while time.time() - t0 < 20:
        line = p.stdout.readline()
        out.append(line)
        if "Ctrl-C to quit" in line:
            break
    return p, port, "".join(out)


def part_c(root, keep):
    print("\nC. dserve-top against a (fake) dserve server")
    fake = FakeDServe(root)
    p, port, out = run_top(fake, root, {"SSH_CONNECTION": "10.0.0.5 51234 10.1.2.3 2222",
                                        "USER": "alice", "LOGNAME": "alice"})
    print("    " + out.replace("\n", "\n    ").rstrip())
    C.ok("started the recording (telemetry was off)",
         any(c[:2] == ("POST", "/telemetry/start") for c in fake.calls) and fake.rec is not None
         and fake.rec.root == os.path.join(root, "rec"))
    C.ok("SSH guide uses the address you came in on",
         "ssh -N -L" in out and "-p 2222" in out and "@10.1.2.3" in out
         and f"http://localhost:" in out)
    code, page = get(port, "/")
    C.ok("serves the page", code == 200 and b"dserve-top" in page and b"/api/state" in page)
    node = shutil.which("node")
    if node:
        import re
        js = re.findall(rb"<script>(.*?)</script>", page, re.S)[0]
        jsf = os.path.join(root, "page.js")
        open(jsf, "wb").write(js)
        r = subprocess.run([node, "--check", jsf], capture_output=True, text=True)
        C.ok("page script parses (node --check)", r.returncode == 0, r.stderr[:160])
    time.sleep(4)
    _, b = get(port, "/api/state?steps_since=0&series_since=0")
    st = json.loads(b)
    kinds = {e["k"] for e in st["steps"]}
    C.ok("feed merged: steps (folded decodes + mixed), series, FT, memory",
         st["connected"] and kinds >= {"d", "x"} and len(st["series"]) >= 2
         and st["finetuning"]["buffer_capacity"] == 256
         and "rank1" in st["window"]["memory"]
         and st["top"]["started_by_top"] is True, f"{len(st['steps'])} steps, {kinds}")
    seqs = [e["s"] for e in st["steps"]]
    C.ok("no duplicate seqs after merging re-sent runs", len(seqs) == len(set(seqs))
         and seqs == sorted(seqs))
    last = seqs[-1]
    _, b = get(port, f"/api/state?steps_since={last}&series_since={st['series'][-1]['t']}")
    inc = json.loads(b)
    C.ok("incremental poll returns only what is new",
         all(e["s"] >= last for e in inc["steps"])
         and all(r["t"] > st["series"][-1]["t"] for r in inc["series"]))
    shot = os.path.join(root, "dserve-top.png")
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if chrome:
        time.sleep(4)
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
                        "--hide-scrollbars", f"--screenshot={shot}",
                        "--window-size=1440,1250", "--virtual-time-budget=6000",
                        f"http://127.0.0.1:{port}/"], capture_output=True, timeout=60)
        dom = subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
                              "--virtual-time-budget=6000", "--dump-dom",
                              f"http://127.0.0.1:{port}/"], capture_output=True, text=True,
                             timeout=60).stdout
        C.ok("headless Chrome renders live data in every panel",
             os.path.exists(shot) and "connecting" not in dom.split("ENGINE")[0]
             and "waiting for steps" not in dom and "waiting for the adapter" not in dom
             and "waiting for the per-GPU" not in dom and "GPU slot" in dom
             and "TTFT p99" in dom, shot)
    p.send_signal(signal.SIGINT)
    p.wait(15)
    C.ok("on exit it stops the recording it started",
         any(c[:2] == ("POST", "/telemetry/stop") for c in fake.calls) and fake.rec is None)
    fake.close()

    print("\n   telemetry already on (config)")
    fake2 = FakeDServe(os.path.join(root, "b"), active=True)
    p2, port2, out2 = run_top(fake2, os.path.join(root, "b"))
    C.ok("does not start a second recording", "already records" in out2
         and not any(c[:2] == ("POST", "/telemetry/start") for c in fake2.calls))
    p2.send_signal(signal.SIGINT)
    p2.wait(15)
    C.ok("and leaves it running on exit",
         not any(c[:2] == ("POST", "/telemetry/stop") for c in fake2.calls)
         and fake2.rec is not None)
    fake2.close()

    print("\n   server down")
    r = subprocess.run([sys.executable, "-m", "deltaserve.top", "--server",
                        "http://127.0.0.1:1"], capture_output=True, text=True, timeout=20)
    C.ok("clear error when the server is unreachable",
         r.returncode == 1 and "cannot reach the server" in r.stdout, r.stdout[:80])
    return shot


if __name__ == "__main__":
    keep = "--keep" in sys.argv
    root = tempfile.mkdtemp(prefix="ds-top-")
    part_a()
    part_b(os.path.join(root, "roll") if os.makedirs(os.path.join(root, "roll")) is None else "")
    shot = part_c(root, keep)
    if keep:
        print(f"\n(kept {root}; screenshot {shot})")
    else:
        shutil.rmtree(root, ignore_errors=True)
    C.finish()
