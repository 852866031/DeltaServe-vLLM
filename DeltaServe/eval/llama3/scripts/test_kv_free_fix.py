#!/usr/bin/env python3
"""test_kv_free_fix.py — verify the infer_batch KV-free off-by-one fix.

What this checks
----------------
1. Functional equivalence — greedy generation (do_sample=False) of a small
   prompt set produces identical token strings before vs. after the fix.
   The fix lives in cleanup paths that run AFTER a request returns its
   tokens, so the per-request output bytes *should* be unchanged. This
   asserts that empirically.
2. Steady-state pool occupancy — repeating the prompt batch multiple
   rounds should not grow `used` pages without bound. With the leak
   present we expect a monotonic drift; with the fix in place the drift
   should plateau within rounding noise.

Output
------
Writes a JSON snapshot of {round: [response_dicts]} to the file given by
--out (default test_kv_free.json). Pass --baseline <path> to compare
against a prior snapshot.
"""

import argparse
import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent          # eval/llama3/
REPO_ROOT = PROJECT_ROOT.parent.parent    # repo root
CONFIG = PROJECT_ROOT / "config" / "serving_config_no_finetuning.yaml"

# Greedy prompts — pick varied lengths to exercise multiple decode
# steps, finished-batch teardown, and partial-finish filters.
PROMPTS = [
    ("short",  "The capital of France is",                          16),
    ("med",    "Once upon a time in a quiet village, there lived",  24),
    ("long",   "Explain the difference between supervised and unsupervised "
               "learning in three concise paragraphs.",             32),
    ("trivia", "Who wrote the play Hamlet? Answer in one sentence.", 16),
    ("math",   "What is 17 times 13? Show the steps briefly.",      24),
]

ROUNDS = 3       # how many times to replay the prompt set
SERVER_PORT = 9100
BASE_MODEL = "meta-llama/Meta-Llama-3-8B"
ADAPTER_DIR = str(PROJECT_ROOT / "adapters" / "llama3-toy-lora")


def make_payload(prompt: str, max_new_tokens: int) -> Dict:
    return {
        "model_dir": BASE_MODEL,
        "lora_dir": ADAPTER_DIR,
        "inputs": prompt,
        "parameters": {
            "do_sample": False,
            "ignore_eos": True,
            "max_new_tokens": max_new_tokens,
        },
    }


async def wait_for_server(server: str, max_wait_s: float = 300.0) -> bool:
    t0 = time.monotonic()
    async with aiohttp.ClientSession() as session:
        while time.monotonic() - t0 < max_wait_s:
            try:
                async with session.post(
                    f"{server}/generate",
                    json=make_payload("ping", 2),
                    timeout=5.0,
                ) as r:
                    if r.status == 200:
                        return True
            except Exception:
                pass
            await asyncio.sleep(1.0)
    return False


async def send_one(session, server, label, prompt, max_new):
    async with session.post(
        f"{server}/generate", json=make_payload(prompt, max_new),
    ) as resp:
        body = await resp.read()
    try:
        data = json.loads(body)
        text = data.get("generated_text", ["<none>"])[0]
    except Exception:
        text = body.decode(errors="replace")
    return {"label": label, "prompt": prompt, "max_new": max_new, "text": text}


async def run_rounds(server: str) -> List[List[Dict]]:
    """Replay the prompt set ROUNDS times, sequentially. Each round waits
    for all responses so finished-batch cleanup runs fully between rounds."""
    all_rounds: List[List[Dict]] = []
    async with aiohttp.ClientSession() as session:
        for r in range(ROUNDS):
            print(f"[test] round {r + 1}/{ROUNDS}: sending {len(PROMPTS)} prompts", flush=True)
            tasks = [
                send_one(session, server, lbl, pr, mn)
                for lbl, pr, mn in PROMPTS
            ]
            results = await asyncio.gather(*tasks)
            results.sort(key=lambda r: r["label"])
            all_rounds.append(results)
            # Brief pause so the server fully drains the finished batch
            # and any used-page drift becomes visible in its logs.
            await asyncio.sleep(2.0)
    return all_rounds


def parse_used_pages_from_log(log_path: Path) -> List[Tuple[float, int]]:
    """Pull `[mem_free] … used=N` lines out of the captured server log so
    we can plot/inspect drift. Returns a list of (rough_timestamp, used)."""
    if not log_path.exists():
        return []
    pattern = re.compile(r"used=(\d+),")
    out = []
    t0 = log_path.stat().st_ctime
    for line in log_path.read_text(errors="replace").splitlines():
        m = pattern.search(line)
        if m:
            out.append((time.time() - t0, int(m.group(1))))
    return out


def summarize_drift(used_series: List[Tuple[float, int]]) -> str:
    if not used_series:
        return "(no [mem_free] traces captured — was _DEBUG_FREE on?)"
    n = len(used_series)
    first = used_series[0][1]
    last = used_series[-1][1]
    peak = max(u for _, u in used_series)
    return (
        f"{n} free events | first.used={first} | last.used={last} | "
        f"peak.used={peak} | net drift={last - first:+d}"
    )


def diff_rounds(a: List[List[Dict]], b: List[List[Dict]]) -> List[str]:
    diffs: List[str] = []
    if len(a) != len(b):
        diffs.append(f"round count: {len(a)} vs {len(b)}")
        return diffs
    for r, (ra, rb) in enumerate(zip(a, b)):
        if len(ra) != len(rb):
            diffs.append(f"round {r}: response count {len(ra)} vs {len(rb)}")
            continue
        for x, y in zip(ra, rb):
            if x["label"] != y["label"]:
                diffs.append(f"round {r}: label mismatch {x['label']} vs {y['label']}")
                continue
            if x["text"] != y["text"]:
                diffs.append(
                    f"round {r} / {x['label']}:\n"
                    f"  baseline: {x['text']!r}\n"
                    f"  current : {y['text']!r}"
                )
    return diffs


def spawn_server(log_path: Path) -> subprocess.Popen:
    """Start api_server pointing at the no-finetuning yaml. Stdout/stderr
    are tee'd to `log_path` so we can grep [mem_free] traces afterward.

    Uses the dserve-1 conda env's Python explicitly — the harness may
    be launched from any interpreter, but the server requires rpyc +
    the full PyTorch stack from that env."""
    server_python = os.environ.get(
        "DSERVE_PYTHON",
        "/mnt/storage/conda/envs/dserve-1/bin/python",
    )
    cmd = [
        server_python, "-u", "-m", "dserve.server.api_server",
        "--config", str(CONFIG),
        "--port", str(SERVER_PORT),
        "--rank_id", "0",
    ]
    print(f"[test] spawning: {' '.join(cmd)}", flush=True)
    log_f = open(log_path, "w")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=log_f, stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
    )


def kill_server(p: subprocess.Popen) -> None:
    if p.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGINT)
    except Exception:
        pass
    for _ in range(40):
        if p.poll() is not None:
            return
        time.sleep(0.25)
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception:
        pass


async def amain(args) -> int:
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    server_proc = spawn_server(log_path)
    server = f"http://127.0.0.1:{SERVER_PORT}"
    try:
        if not await wait_for_server(server):
            print("[test] server didn't come up in time", flush=True)
            return 2
        print("[test] server ready ✓", flush=True)
        rounds = await run_rounds(server)
    finally:
        print("[test] shutting down server", flush=True)
        kill_server(server_proc)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rounds, indent=2))
    print(f"[test] wrote responses → {out_path}", flush=True)

    drift = summarize_drift(parse_used_pages_from_log(log_path))
    print(f"[test] used-pages drift: {drift}", flush=True)

    if args.baseline:
        try:
            baseline = json.loads(Path(args.baseline).read_text())
        except FileNotFoundError:
            print(f"[test] no baseline at {args.baseline}; skipping diff", flush=True)
            return 0
        diffs = diff_rounds(baseline, rounds)
        if diffs:
            print(f"\n[test] ❌ {len(diffs)} output diff(s) vs baseline:", flush=True)
            for d in diffs:
                print("  - " + d)
            return 1
        else:
            print(f"\n[test] ✅ all {sum(len(r) for r in rounds)} responses match "
                  f"baseline at {args.baseline}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(SCRIPT_DIR / "test_kv_free.json"))
    ap.add_argument("--log", default=str(SCRIPT_DIR / "test_kv_free.server.log"))
    ap.add_argument("--baseline", default=None,
                    help="If set, compare new outputs to this JSON snapshot.")
    args = ap.parse_args()
    sys.exit(asyncio.run(amain(args)))


if __name__ == "__main__":
    main()
