#!/usr/bin/env python
"""ft_bench_tp.py — TP co-serving finetuning smoke/benchmark (Phase 7 / M4).

Launches the TP=2 co-serving server, opens FT admission, drives a steady stream
of inference requests (so FT rides prefill under the default scheduler), then
tears down and reports whether the backward actually trained under tensor
parallelism: the per-cycle loss trend (first/last/min/max), a NaN/Inf guard, and
the FT cycle count. This is the M4 live-validation harness — the analogue of
eval/auto_benchmark.py, trimmed to the one question "does co-serving FT train
correctly under TP=2".

It reuses eval-tp/launch_deltaserve.build_server_cmd (same dir) so the server is
built from configs/serving_config_finetuning_llama3_tp2.yaml exactly like the
launch script, then appends a bwd_log path.

Usage (dserve-vllm env, CUDA env per README.md):

    python eval-tp/ft_bench_tp.py                      # 60s, ~4 req/s traffic, TP from YAML
    python eval-tp/ft_bench_tp.py --duration 120       # longer, more backward cycles
    python eval-tp/ft_bench_tp.py --tp 2 --rps 6       # override TP + request rate
    python eval-tp/ft_bench_tp.py --dry-run            # print the server cmd, don't launch

Watch the server log stream for lines like:
    [deltaserve] [backward] 42.1ms (eager) loss=2.83 total_trained=... n=... epoch=...
The summary at the end confirms loss decreased with no NaN under TP=2.
"""

import argparse
import asyncio
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import aiohttp

_HERE = Path(__file__).resolve().parent          # eval-tp/
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))                   # import launch_deltaserve (same dir)

from launch_deltaserve import (  # noqa: E402
    _DEFAULT_CONFIG, _SERVED_NAME, _BASE_MODEL_DEFAULT, _HF_HOME_DEFAULT,
    build_server_cmd, terminate,
)

OUTPUT_DIR = _HERE / "output"
_LOSS_RE = re.compile(r"\[backward\].*?loss=([0-9eE.+-]+)")


async def wait_for_health(server: str, max_wait_s: float, proc) -> bool:
    deadline = time.time() + max_wait_s
    async with aiohttp.ClientSession() as s:
        while time.time() < deadline:
            if proc.poll() is not None:
                print(f"[ft-tp] server exited early (code {proc.returncode})",
                      flush=True)
                return False
            try:
                async with s.get(f"{server}/health", timeout=2) as r:
                    if r.status == 200:
                        print(f"[ft-tp] server healthy at {server}", flush=True)
                        return True
            except Exception:
                pass
            await asyncio.sleep(1.0)
    print(f"[ft-tp] server not healthy within {max_wait_s}s", flush=True)
    return False


async def start_finetuning(server: str) -> bool:
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(f"{server}/start_finetuning", timeout=10) as r:
                ok = r.status == 200
                print(f"[ft-tp] POST /start_finetuning -> {r.status}", flush=True)
                return ok
    except Exception as e:  # noqa: BLE001
        print(f"[ft-tp] start_finetuning failed: {e}", flush=True)
        return False


async def _one_request(session, server, idx):
    prompt = f"Summarize in one sentence (req {idx}): the quick brown fox " \
             f"jumps over the lazy dog and then keeps running for a while."
    try:
        async with session.post(
            f"{server}/v1/completions",
            json={"model": _SERVED_NAME, "prompt": prompt, "max_tokens": 8,
                  "temperature": 0.0},
            timeout=30,
        ) as r:
            await r.read()
            return r.status == 200
    except Exception:
        return False


async def drive_traffic(server: str, duration_s: float, rps: float,
                        stop: asyncio.Event) -> tuple[int, int]:
    """Fire ~rps requests/s for duration_s. Returns (sent, ok)."""
    sent = ok = 0
    interval = 1.0 / max(rps, 0.1)
    t0 = time.monotonic()
    inflight: set = set()
    async with aiohttp.ClientSession() as session:
        while time.monotonic() - t0 < duration_s and not stop.is_set():
            t = asyncio.create_task(_one_request(session, server, sent))
            inflight.add(t)
            t.add_done_callback(inflight.discard)
            sent += 1
            done = [x for x in list(inflight) if x.done()]
            for x in done:
                ok += 1 if x.result() else 0
            await asyncio.sleep(interval)
        if inflight:
            results = await asyncio.gather(*inflight, return_exceptions=True)
            ok += sum(1 for x in results if x is True)
    return sent, ok



def preflight_stale_check(kill: bool) -> bool:
    """Refuse to launch when a previous run left GPU-resident processes behind.

    Finetuning runs spawn workers NON-daemonic (so each rank can fork its
    backward child), so a hard-killed run can strand ``VLLM::Worker_TP``
    processes holding the whole model. The next launch then dies ~3 minutes in
    with a misleading "Free memory on device cuda:N is less than desired GPU
    memory utilization" — the real cause being invisible. Surface it up front.

    Keys off processes actually holding GPU memory (nvidia-smi), not a process
    name grep: a grep for the pattern matches its own invoking shell, and what
    we care about is memory residency, not naming. Returns True if safe to
    launch."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return True          # no nvidia-smi / no GPUs — nothing to check
    rows = [r.strip() for r in out.splitlines() if r.strip()]
    if not rows:
        return True

    def cmd_of(pid: str) -> str:
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                return f.read().replace(b"\0", b" ").decode(errors="ignore").strip()
        except Exception:
            return "<gone>"

    stale = []
    for row in rows:
        pid = row.split(",")[0].strip()
        if pid.isdigit() and pid != str(os.getpid()):
            stale.append((pid, row, cmd_of(pid)))
    if not stale:
        return True

    print(f"[ft-tp] {len(stale)} process(es) are already holding GPU memory:",
          flush=True)
    for pid, row, cmdline in stale[:8]:
        print(f"[ft-tp]   {row}  {cmdline[:100]}", flush=True)
    if not kill:
        print("[ft-tp] REFUSING to launch — this run would fail with a "
              "confusing out-of-memory error a few minutes in.\n"
              "[ft-tp] Re-run with --kill-stale, or clean up manually.",
              flush=True)
        return False
    print(f"[ft-tp] --kill-stale: killing {' '.join(p for p, _, _ in stale)}",
          flush=True)
    for pid, _, _ in stale:
        try:
            os.kill(int(pid), signal.SIGKILL)
        except Exception as e:
            print(f"[ft-tp]   could not kill {pid}: {e}", flush=True)
    time.sleep(5)
    return True


def summarize(server_log: str, sent: int, ok: int, tp: int,
              bwd_log: str | None = None) -> int:
    """Report the per-cycle loss trend. Returns exit code (0 = looks-trained).

    Prefers the bwd CSV: it has exactly ONE row per backward cycle (the
    scheduler-side coordinator is the single writer). The server log is only a
    fallback — under TP>1 EVERY rank prints its own ``[backward] … loss=`` line,
    so counting log lines reports tp_size x the real cycle count."""
    losses: list[float] = []
    if bwd_log and os.path.exists(bwd_log):
        with open(bwd_log, "r", errors="ignore") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 5 and not line.startswith("timestamp"):
                    try:
                        losses.append(float(parts[4]))
                    except ValueError:
                        pass
    if not losses and os.path.exists(server_log):
        seen: set[str] = set()
        with open(server_log, "r", errors="ignore") as f:
            for line in f:
                m = _LOSS_RE.search(line)
                if m:
                    # de-dup the per-rank duplicates of the same cycle
                    key = line.split("[backward]", 1)[-1].strip()
                    if key in seen:
                        continue
                    seen.add(key)
                    try:
                        losses.append(float(m.group(1)))
                    except ValueError:
                        pass
    print("\n" + "=" * 62, flush=True)
    print(f"[ft-tp] M4 co-serving FT under TP={tp} — summary", flush=True)
    print("=" * 62, flush=True)
    print(f"  inference requests: {ok}/{sent} ok", flush=True)
    print(f"  backward cycles:    {len(losses)}", flush=True)
    if not losses:
        print("  RESULT: NO backward cycles fired — FT never admitted (traffic "
              "too light? duration too short? admission closed?)", flush=True)
        print("=" * 62, flush=True)
        return 1
    has_bad = any(l != l or l in (float("inf"), float("-inf")) for l in losses)
    finite = [l for l in losses if l == l and l not in (float("inf"), float("-inf"))]
    print(f"  loss first/last:    {losses[0]:.4f} -> {losses[-1]:.4f}", flush=True)
    if finite:
        print(f"  loss min/max:       {min(finite):.4f} / {max(finite):.4f}",
              flush=True)
    decreased = len(finite) >= 2 and finite[-1] < finite[0]
    print(f"  NaN/Inf in loss:    {'YES (BAD)' if has_bad else 'none'}", flush=True)
    print(f"  loss decreased:     {'yes' if decreased else 'no (may need longer run)'}",
          flush=True)
    ok_result = (not has_bad) and len(losses) >= 1
    print(f"  RESULT: {'TRAINED under TP (finite loss, cycles fired)' if ok_result else 'PROBLEM — see above'}",
          flush=True)
    print("=" * 62, flush=True)
    return 0 if ok_result else 1


async def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(_DEFAULT_CONFIG))
    ap.add_argument("--model", default=_BASE_MODEL_DEFAULT)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tp", type=int, default=None, help="Override tensor_parallel_size.")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--rps", type=float, default=4.0, help="Inference requests/s.")
    ap.add_argument("--hf-home", default=None)
    ap.add_argument("--startup-timeout", type=float, default=600.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--kill-stale", action="store_true",
                    help="Kill leftover vLLM processes from a previous "
                         "run instead of refusing to launch.")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    cmd, cfg = build_server_cmd(args.config, args.port, args.model, args.tp)
    tp = args.tp if args.tp is not None else int(
        (cfg.get("parallel") or {}).get("tensor_parallel_size", 1) or 1)
    # Per-TP filenames so a tp=1 control and a tp=2 run never overwrite or
    # append into each other, and TRUNCATE the bwd log up front: the server
    # opens it in APPEND mode, so without this each run silently concatenates
    # onto the previous one's rows (and total_processed_tokens keeps climbing).
    bwd_log = str(OUTPUT_DIR / f"bwd_log_tp{tp}.csv")
    server_log = str(OUTPUT_DIR / f"server_tp{tp}.log")
    open(bwd_log, "w").close()
    cmd.append(f"--finetune-config.bwd_log_path={bwd_log}")

    print(f"[ft-tp] config          = {args.config}", flush=True)
    print(f"[ft-tp] tensor_parallel = {tp}", flush=True)
    print(f"[ft-tp] server log      = {server_log}", flush=True)
    print(f"[ft-tp] bwd log         = {bwd_log}", flush=True)
    print(f"[ft-tp] server cmd      = {' '.join(cmd)}", flush=True)
    if args.dry_run:
        print("[ft-tp] --dry-run: not launching", flush=True)
        return 0

    if not preflight_stale_check(args.kill_stale):
        return 2

    hf_home = args.hf_home or os.environ.get("HF_HOME") or _HF_HOME_DEFAULT
    env = dict(os.environ)
    env.setdefault("HF_HOME", hf_home)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    env["PYTHONUNBUFFERED"] = "1"

    # Capture server stdout+stderr to a file (the [backward] loss lines live
    # here) AND stream to this terminal via tee-like duplication.
    logf = open(server_log, "w")
    proc = subprocess.Popen(cmd, env=env, cwd="/tmp",
                            stdout=logf, stderr=subprocess.STDOUT,
                            start_new_session=True)
    server = f"http://127.0.0.1:{args.port}"
    stop = asyncio.Event()
    rc = 1
    try:
        if not await wait_for_health(server, args.startup_timeout, proc):
            return 1
        if not await start_finetuning(server):
            print("[ft-tp] could not open FT admission; aborting", flush=True)
            return 1
        print(f"[ft-tp] driving {args.rps} req/s for {args.duration}s "
              f"(FT rides prefill)…", flush=True)
        sent, ok = await drive_traffic(server, args.duration, args.rps, stop)
        # Give the last in-flight backward a moment to flush its log line.
        await asyncio.sleep(3.0)
        rc = summarize(server_log, sent, ok, tp, bwd_log)
    except KeyboardInterrupt:
        print("\n[ft-tp] Ctrl-C — shutting down…", flush=True)
    finally:
        terminate(proc)
        logf.close()
    return rc


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
