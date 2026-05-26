#!/usr/bin/env python3
"""pure_ft_bench.py — pure-finetuning workload benchmark.

Spins up the DeltaServe-on-vLLM server with finetuning enabled but holds FT
admission closed at launch, POSTs ``/start_finetuning`` to open admission,
sleeps for a configurable duration with NO inference traffic, then tears the
server down and summarizes the finetune-throughput log.

This is the "isolated" backward measurement — useful for:
  - Measuring peak FT tokens/s without inference interference.
  - Validating Phase 5 backward-CUDA-graph speedups (compare runs with
    ``finetune.backward_cuda_graph`` on vs off in the YAML).
  - Profiling backward wall-clock with the existing per-cycle log line
    ``[backward] X.Xms (graph|eager) loss=... total_trained=...``.

Usage (dserve-vllm env):
    python eval/pure_ft_bench.py                 # default 100s
    python eval/pure_ft_bench.py --duration 30   # quick smoke
    python eval/pure_ft_bench.py --duration 600 -f  # 10 min, log to file

The server writes ``output/bwd_log_pure_ft.csv`` (timestamp, batch_tokens,
total_processed_tokens, …) and this script trims it to the window between
``/start_finetuning`` and shutdown.
"""

import argparse
import asyncio
import csv
import datetime
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

# Re-use the server-launch + health-poll + FT-start helpers from auto_benchmark
# rather than duplicating. They're imported lazily inside main() to avoid the
# sys.path side effect at module load time.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
OUTPUT_DIR = _HERE / "output"

# Per-GPU base model + HF cache root — mirrors auto_benchmark._GPU_ENV so a
# user's flags / env behave identically across the two scripts.
_GPU_ENV = {
    "5090": {
        "model": "meta-llama/Meta-Llama-3-8B",
        "hf_home": "/mnt/storage/huggingface",
        "hf_hub_cache": None,
    },
    "A100": {
        "model": "meta-llama/Meta-Llama-3.1-8B",
        "hf_home": "/home/jiaxuan_chen/scratch",
        "hf_hub_cache": "/home/jiaxuan_chen/scratch",
    },
}


def detect_gpu_subdir() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True, timeout=2.0)
        name = (out.strip().splitlines() or [""])[0].upper()
        if "A100" in name:
            return "A100"
        if "5090" in name:
            return "5090"
    except Exception:
        pass
    return "5090"


def trim_bwd_log_before(path: str, cutoff: datetime.datetime) -> int:
    """Drop bwd_log rows whose ISO ``timestamp`` is strictly before ``cutoff``.
    Returns the number of kept rows. Same-second rows are kept (the server
    log writes ms-precision timestamps but we cut on second boundary to
    match the auto_benchmark trim semantics)."""
    if not os.path.exists(path):
        print(f"[pure_ft] no bwd_log to trim at {path}", flush=True)
        return 0
    cutoff_iso = cutoff.replace(microsecond=0).isoformat(timespec="seconds")
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        all_rows = list(reader)
    if not fieldnames or "timestamp" not in fieldnames:
        return len(all_rows)
    kept = [r for r in all_rows if (r.get("timestamp") or "") >= cutoff_iso]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(kept)
    print(f"[pure_ft] trimmed bwd_log: kept {len(kept)}/{len(all_rows)} rows "
          f"(cutoff {cutoff_iso})", flush=True)
    return len(kept)


def summarize_bwd_log(path: str) -> None:
    """Print a one-screen summary of the FT throughput log: total tokens
    trained, cycle count, avg tok/s over the run window, per-cycle duration
    stats. No-op if the log is missing or empty."""
    if not os.path.exists(path):
        print(f"[pure_ft] no bwd_log at {path} — server may not have run any "
              f"backward (start_finetuning didn't land? duration too short?)",
              flush=True)
        return
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        print(f"[pure_ft] bwd_log at {path} is empty", flush=True)
        return
    # Parse what we need; tolerate missing columns from older log writers.
    def _f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")
    timestamps = []
    batch_tokens = []
    total_tokens = []
    for r in rows:
        try:
            timestamps.append(datetime.datetime.fromisoformat(
                (r.get("timestamp") or "").strip()))
        except Exception:
            timestamps.append(None)
        batch_tokens.append(_f(r.get("batch_tokens")))
        total_tokens.append(_f(r.get("total_processed_tokens")))
    ts_valid = [t for t in timestamps if t is not None]
    if len(ts_valid) >= 2:
        span_s = (ts_valid[-1] - ts_valid[0]).total_seconds()
    else:
        span_s = 0.0
    # Pick the right totals column. total_processed_tokens is cumulative;
    # if absent or all NaN we fall back to summing batch_tokens.
    if any(t == t for t in total_tokens):  # any non-NaN
        # last finite value
        last = next((t for t in reversed(total_tokens) if t == t), float("nan"))
        total_trained = int(last) if last == last else 0
    else:
        total_trained = int(sum(b for b in batch_tokens if b == b))
    n_cycles = len(rows)
    avg_tok_s = (total_trained / span_s) if span_s > 0 else float("nan")
    # Per-cycle dt + batch_tokens stats
    cycle_dts = []
    for a, b in zip(ts_valid, ts_valid[1:]):
        cycle_dts.append((b - a).total_seconds() * 1000.0)  # ms
    print("", flush=True)
    print("=" * 60, flush=True)
    print("[pure_ft] FT throughput summary", flush=True)
    print("=" * 60, flush=True)
    print(f"  bwd_log:        {path}", flush=True)
    print(f"  cycles:         {n_cycles}", flush=True)
    print(f"  span:           {span_s:.1f}s", flush=True)
    print(f"  total trained:  {total_trained} tokens", flush=True)
    print(f"  avg tok/s:      {avg_tok_s:.1f}", flush=True)
    bt = [b for b in batch_tokens if b == b and b > 0]
    if bt:
        print(f"  batch_tokens:   "
              f"min={int(min(bt))}  avg={sum(bt) / len(bt):.0f}  max={int(max(bt))}",
              flush=True)
    if cycle_dts:
        print(f"  cycle dt (ms):  "
              f"min={min(cycle_dts):.1f}  avg={sum(cycle_dts) / len(cycle_dts):.1f}  "
              f"max={max(cycle_dts):.1f}", flush=True)
    print("=" * 60, flush=True)


async def _idle_with_progress(stop: asyncio.Event, duration_s: float,
                              tick_s: float = 5.0) -> None:
    """Sleep for ``duration_s`` total, printing an elapsed-time tick every
    ``tick_s``. Wakes immediately if ``stop`` is set (Ctrl+C)."""
    t0 = time.monotonic()
    while True:
        elapsed = time.monotonic() - t0
        if elapsed >= duration_s:
            return
        remaining = duration_s - elapsed
        sleep_for = min(tick_s, remaining)
        try:
            await asyncio.wait_for(stop.wait(), timeout=sleep_for)
            print("[pure_ft] stop requested mid-idle", flush=True)
            return
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - t0
            print(f"[pure_ft] idle t={elapsed:.0f}s / {duration_s:.0f}s",
                  flush=True)


async def main() -> None:
    # Import auto_benchmark's helpers lazily so its sys.path side effects (in
    # build_server_cmd) don't fire on this module's import.
    sys.path.insert(0, str(_HERE))
    from auto_benchmark import (  # noqa: E402
        build_server_cmd,
        start_finetuning,
        terminate,
        wait_for_health,
    )

    gpu_default = detect_gpu_subdir()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duration", type=float, default=100.0,
                    help="Pure-FT wall-clock duration (s) AFTER "
                         "/start_finetuning lands. Default 100s.")
    ap.add_argument("--output-name", default="pure_ft",
                    help="Tag for output files. Server writes "
                         "output/bwd_log_<tag>.csv; with -f the server log "
                         "lands at output/server_<tag>.log. Default "
                         "'pure_ft'.")
    ap.add_argument("--startup-timeout", type=float, default=600.0)
    ap.add_argument("--api-server-count", type=int, default=None,
                    help="Frontend API server processes. Default reads "
                         "server.api_server_count from the YAML.")
    ap.add_argument("--model", default=None,
                    help="Base model id (HF) or local path. Default per-GPU.")
    ap.add_argument("--hf-home", default=None)
    ap.add_argument("--hf-hub-cache", default=None)
    ap.add_argument("--gpu", default=gpu_default, choices=["5090", "A100"],
                    help="Pick per-GPU model / HF defaults. Default "
                         f"auto-detected ({gpu_default}).")
    ap.add_argument("--f", "-f", dest="log_to_file", action="store_true",
                    help="Capture server stdout/stderr to "
                         "output/server_<tag>.log. Off by default (stream "
                         "to terminal so you can watch [backward] cycle logs "
                         "live).")
    args = ap.parse_args()

    if args.duration <= 0:
        ap.error("--duration must be > 0")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.output_name
    bwd_log = str(OUTPUT_DIR / f"bwd_log_{tag}.csv")
    if os.path.exists(bwd_log):
        os.remove(bwd_log)  # fresh log per run

    # Resolve per-GPU defaults the same way auto_benchmark does.
    gpu_env = _GPU_ENV.get(args.gpu, _GPU_ENV["5090"])
    base_model = args.model or gpu_env["model"]
    hf_home = args.hf_home or gpu_env["hf_home"]
    hf_hub_cache = args.hf_hub_cache or gpu_env.get("hf_hub_cache")
    print(f"[pure_ft] base model: {base_model}", flush=True)
    print(f"[pure_ft] HF_HOME:    {hf_home}", flush=True)
    if hf_hub_cache:
        print(f"[pure_ft] HF_HUB_CACHE: {hf_hub_cache}", flush=True)

    # Always co-serving (FT on); pure-FT means no inference traffic, not
    # that the engine starts without the FT subprocess.
    cmd = build_server_cmd(co=True, bwd_log_path=bwd_log,
                           api_server_count=args.api_server_count,
                           base_model=base_model)
    print("[pure_ft] launching:", " ".join(cmd), flush=True)
    print(f"[pure_ft] bwd_log -> {bwd_log}", flush=True)

    env = dict(os.environ)
    env.setdefault("HF_HOME", hf_home)
    if hf_hub_cache:
        env.setdefault("HF_HUB_CACHE", hf_hub_cache)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    env["PYTHONSAFEPATH"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    logf = None
    server_log: Optional[str] = None
    if args.log_to_file:
        server_log = str(OUTPUT_DIR / f"server_{tag}.log")
        logf = open(server_log, "w")
        popen_kwargs = dict(stdout=logf, stderr=subprocess.STDOUT)
        print(f"[pure_ft] server log -> {server_log}  (tail -f to watch live)",
              flush=True)
    else:
        popen_kwargs = dict()
        print("[pure_ft] server log -> terminal (pass --f to capture to file)",
              flush=True)
    proc = subprocess.Popen(cmd, env=env, cwd=tempfile.gettempdir(),
                            start_new_session=True, **popen_kwargs)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    _ctrlc = {"count": 0}

    def _on_sigint() -> None:
        _ctrlc["count"] += 1
        if _ctrlc["count"] == 1:
            print("\n[pure_ft] Ctrl+C — shutting down "
                  "(press again to force kill)", flush=True)
            stop.set()
        else:
            print("\n[pure_ft] Ctrl+C (force) — SIGKILLing server now",
                  flush=True)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
            loop.remove_signal_handler(signal.SIGINT)
            raise KeyboardInterrupt

    loop.add_signal_handler(signal.SIGINT, _on_sigint)

    t_ft_start_wall: Optional[datetime.datetime] = None
    server = f"http://127.0.0.1:8000"  # _PORT in auto_benchmark
    try:
        await wait_for_health(server, args.startup_timeout, stop=stop)
        if stop.is_set():
            return

        # Open FT admission. Server YAML sets start_on_launch=false (via
        # build_server_cmd) so nothing fires until this POST lands. Record
        # the wall clock so we can trim out any pre-POST profiling rows.
        t_ft_start_wall = datetime.datetime.now()
        ok = await start_finetuning(server)
        if not ok:
            print("[pure_ft] start_finetuning failed — nothing to measure, "
                  "shutting down", flush=True)
            return

        print(f"[pure_ft] idling for {args.duration:.0f}s — server is doing "
              f"pure FT backward (watch [backward] cycle logs)", flush=True)
        await _idle_with_progress(stop, args.duration)
    finally:
        stop.set()
        print("[pure_ft] shutting down server", flush=True)
        terminate(proc)
        if logf is not None:
            try:
                logf.close()
            except Exception:
                pass

        # Trim the bwd_log to rows in the FT window (cuts any profiling-pass
        # backward rows that ran before /start_finetuning), then summarize.
        if t_ft_start_wall is not None:
            # Give the OS a moment for the server's final log flush.
            await asyncio.sleep(0.5)
            trim_bwd_log_before(bwd_log, t_ft_start_wall)
        summarize_bwd_log(bwd_log)
        if server_log is not None:
            print(f"[pure_ft] server log: {server_log}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
