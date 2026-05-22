#!/usr/bin/env python3
"""
Benchmark orchestrator (consumer) that sends requests to a running server
following a given timeline.csv file.

Each row in timeline.csv specifies:
  timestamp_s, prompt_length, max_new_tokens, second, index_in_second

Requests are scheduled according to 'timestamp_s' (relative seconds since t=0).
"""

from __future__ import annotations
import argparse
import asyncio
import json
import os
import random
import signal

import string
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp
import socket
import requests
import pandas as pd
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))

# ---------------- Defaults ----------------
DEFAULTS = {
    "server": "http://localhost:9000",
    "timeline_csv": f"{current_dir}/timelines/timeline_live.csv",  
    "max_wait": 120.0,
    "ft_poll_interval": 3.0,
    "ft_max_wait": 60.0,
}
def internet_available(timeout=2):
    """Check internet by pinging HuggingFace DNS & HTTPS."""
    try:
        # DNS check
        socket.gethostbyname("huggingface.co")
        # HTTPS check
        requests.head("https://huggingface.co", timeout=timeout)
        return True
    except Exception:
        return False

if internet_available():
    base_model = "huggyllama/llama-7b"
    DEFAULT_LORA = "tloen/alpaca-lora-7b"
else:
    base_model = "/projects/I20240005/jchen/hf_cache/models--huggyllama--llama-7b/snapshots/llama-7b"
    DEFAULT_LORA = "/projects/I20240005/jchen/hf_cache/hub/models--tloen--alpaca-lora-7b/snapshots/12103d6baae1b320aa60631b38acb6ea094a0539"
# ---------------- Prompt generation ----------------
def generate_random_sentence(length: int) -> str:
    """Generate a random sentence of `length` words."""
    words = []
    for _ in range(max(1, length)):
        word = "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 5)))
        words.append(word)
    return " ".join(words).capitalize() + "."


# ---------------- Request building ----------------
def make_payload(prompt: str, max_new_tokens: int) -> Dict:
    """Build JSON payload for /generate."""
    return {
        "model_dir": base_model,
        "lora_dir": DEFAULT_LORA,
        "inputs": prompt,
        "parameters": {
            "do_sample": False,
            "ignore_eos": True,
            "max_new_tokens": int(max_new_tokens),
        },
    }

import subprocess
from datetime import datetime

# ---------------- Load schedule from CSV ----------------
def load_schedule_from_csv(csv_path: str) -> List[Tuple[int, float, Dict]]:
    """
    Load timeline.csv and build (idx, t_off, payload) list for run_schedule().
    Columns required: timestamp_s, prompt_length, max_new_tokens
    """
    df = pd.read_csv(csv_path)
    if not all(col in df.columns for col in ["timestamp_s", "prompt_length", "max_new_tokens"]):
        raise ValueError("timeline.csv missing required columns.")

    schedule = []
    for idx, row in enumerate(df.itertuples(index=False)):
        t_off = float(row.timestamp_s)
        prompt_len = int(row.prompt_length)
        max_new = int(row.max_new_tokens)
        prompt = generate_random_sentence(prompt_len)
        schedule.append((idx, t_off, make_payload(prompt, max_new)))

    schedule.sort(key=lambda x: x[1])
    print(f"[orchestrator] Loaded {len(schedule)} requests from {csv_path}")
    return schedule


# ---------------- HTTP helpers ----------------
async def try_health(session: aiohttp.ClientSession, server: str, timeout_s: float = 1.0) -> bool:
    try:
        async with session.get(f"{server.rstrip('/')}/health", timeout=timeout_s) as resp:
            return resp.status == 200
    except Exception:
        return False


async def try_generate_probe(session: aiohttp.ClientSession, server: str, timeout_s: float = 2.0) -> bool:
    try:
        async with session.post(
            f"{server.rstrip('/')}/generate",
            json=make_payload("ping", max_new_tokens=4),
            timeout=timeout_s,
        ) as resp:
            return resp.status == 200
    except Exception:
        return False


async def wait_for_server(server: str, max_wait_s: float = 120.0, poll_period_s: float = 0.5) -> None:
    t0 = time.time()
    async with aiohttp.ClientSession() as session:
        while True:
            if await try_health(session, server) or await try_generate_probe(session, server):
                print("[orchestrator] Server is up ✅ at ", server)
                return
            if time.time() - t0 > max_wait_s:
                raise TimeoutError(f"Server didn't become healthy within {max_wait_s:.1f}s")
            await asyncio.sleep(poll_period_s)

async def start_finetuning(session: aiohttp.ClientSession, server: str, timeout_s: float = 1.0) -> bool:
    """
    Call POST /start_finetuning on the given server.
    Returns True on success, False on failure.
    """
    url = f"{server.rstrip('/')}/start_finetuning"

    try:
        async with session.post(url, timeout=timeout_s) as resp:
            if resp.status == 200:
                return True
            return False
    except Exception:
        return False


# ---------- wait for finetuning finished ----------
async def wait_for_finetuning(
    server: str,
    poll_interval_s: float = DEFAULTS["ft_poll_interval"],
    max_wait_s: float = DEFAULTS["ft_max_wait"],
) -> None:
    url = f"{server.rstrip('/')}/finetuning_status"
    t0 = time.time()
    n = 0
    async with aiohttp.ClientSession(headers={"User-Agent": "OrchestratorFTPoller"}) as session:
        while True:
            n += 1
            try:
                async with session.post(url) as resp:
                    text = await resp.text()
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        payload = {}
                    finished = isinstance(payload, dict) and str(payload.get("finished", "")).lower() == "true"
                    if finished:
                        elapsed = time.time() - t0
                        print(f"[orchestrator] Finetuning finished after {elapsed:.1f}s (poll #{n}). ✅")
                        return
                    print(f"[orchestrator] Finetuning not finished yet (poll #{n}).")
            except Exception as e:
                print(f"[orchestrator] finetuning_status error: {e}")

            if time.time() - t0 > max_wait_s:
                raise TimeoutError(f"Finetuning didn't finish within {max_wait_s:.1f}s")
            await asyncio.sleep(poll_interval_s)


# ---------- include relative send time (t_rel) ----------
async def send_one(
    session: aiohttp.ClientSession,
    server: str,
    idx: int,
    payload: Dict,
    t0_ref: float,
) -> Tuple[int, float, float, str, float | None, float | None, float | None]:
    """
    Return (idx, t_rel, latency, status, ttft, avg_tbt, worst_tbt)
      t_rel   = time.monotonic() - t0_ref at send time
      latency = total response time
    """
    url = f"{server.rstrip('/')}/generate"
    t_send = time.monotonic()
    t_rel = t_send - t0_ref
    try:
        async with session.post(url, json=payload) as resp:
            body = await resp.read()
            latency = time.monotonic() - t_send
            ttft = avg_tbt = worst_tbt = None
            try:
                data = json.loads(body)
                ttft = data.get("ttft")
                avg_tbt = data.get("avg_tbt")
                worst_tbt = data.get("worst_tbt")
                out = data.get("generated_text", ["<no-text>"])[0]
            except Exception:
                out = body.decode(errors="replace")

            return (idx, t_rel, latency, "ok", ttft, avg_tbt, worst_tbt)
    except Exception as e:
        latency = time.monotonic() - t_send
        #print(f"[req {idx:04d}] @{t_rel:6.3f}s  FAILED after {latency*1000:7.1f} ms: {e}", flush=True)
        return (idx, t_rel, latency, "err", None, None, None)


# ---------------- Process control ----------------
def launch_server(enable_finetuning: bool):
    import subprocess
    cmd = [sys.executable, f"{current_dir}/launch_server.py"]
    if enable_finetuning:
        cmd.append("--enable-finetuning")
    print(f"[orchestrator] Launching server:\n  {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, preexec_fn=os.setsid)
    return proc


def kill_server(proc) -> None:
    """Gracefully stop launched server."""
    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        pgid = None
    try:
        if pgid:
            os.killpg(pgid, signal.SIGINT)
        else:
            proc.send_signal(signal.SIGINT)
        proc.wait(timeout=10)
    except Exception:
        pass

async def run_schedule(
    server: str,
    schedule: List[Tuple[int, float, Dict]],
) -> List[Tuple[int, float, float, str, float | None, float | None, float | None]]:
    """
    Execute the precomputed schedule with precise arrival times.
    Each task sleeps until its absolute due time (t0 + t_off) before POSTing.
    """
    if not schedule:
        return []

    total = len(schedule)
    finished_count = 0
    lock = asyncio.Lock()
    time_len = schedule[-1][1]  # last timestamp_s → end of schedule
    t0 = time.monotonic()

    # tqdm progress bar at the bottom
    pbar = tqdm(
        total=total,
        position=0,
        leave=True,
        desc="Requests",
        unit="req",
    )

    async with aiohttp.ClientSession(headers={"User-Agent": "OrchestratorLoad"}) as session:
        async def fire_at(idx: int, t_off: float, payload: Dict):
            nonlocal finished_count

            # Sleep until scheduled send time
            delay = (t0 + t_off) - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)

            # Send request
            result = await send_one(session, server, idx, payload, t0)
            if result[3] != "ok":
                print(f"[orchestrator] Request {idx} failed.")
                exit(1)
                
            # Update progress bar
            async with lock:
                finished_count += 1
                pbar.update(1)
                remaining = max((t0 + time_len) - time.monotonic(), 0.0)
                mins = int(remaining // 60)
                secs = remaining % 60
                pbar.set_postfix_str(f"{mins}m {secs:04.1f}s remaining")

            return result

        tasks = [
            asyncio.create_task(fire_at(idx, t_off, payload))
            for idx, t_off, payload in schedule
        ]
        results = await asyncio.gather(*tasks)

    pbar.close()
    return results



# ---------------- Summaries ----------------
def summarize(results) -> None:
    """
    Works with either:
      (idx, t_rel, latency, status)
    or
      (idx, t_rel, latency, status, ttft, avg_tbt, worst_tbt)
    """
    if not results:
        print("[orchestrator] No results collected.")
        return

    # indices: 0 idx, 1 t_rel, 2 latency, 3 status, 4 ttft, 5 avg_tbt, 6 worst_tbt
    lat_ok = [r[2] for r in results if len(r) >= 4 and r[3] == "ok"]
    if not lat_ok:
        print("[orchestrator] No successful requests.")
        return

    print("\n== Latency Summary ==")
    print(f"Requests (ok/total): {len(lat_ok)}/{len(results)}")
    print(f"Mean    : {np.mean(lat_ok):.4f} s")
    print(f"P50/P90 : {np.percentile(lat_ok,50):.4f} / {np.percentile(lat_ok,90):.4f} s")
    print(f"Min/Max : {np.min(lat_ok):.4f} / {np.max(lat_ok):.4f} s")

    # Optional: summarize ttft / avg_tbt / worst_tbt if present
    if any(len(r) >= 7 for r in results):
        ttft_ok      = [r[4] for r in results if len(r) >= 7 and r[3] == "ok" and r[4] is not None]
        avg_tbt_ok   = [r[5] for r in results if len(r) >= 7 and r[3] == "ok" and r[5] is not None]
        worst_tbt_ok = [r[6] for r in results if len(r) >= 7 and r[3] == "ok" and r[6] is not None]

        if ttft_ok:
            print("\n-- TTFT (s) --")
            print(f"Mean {np.mean(ttft_ok):.4f} | P50 {np.percentile(ttft_ok,50):.4f} | P90 {np.percentile(ttft_ok,90):.4f}")
        if avg_tbt_ok:
            print("-- Avg TBT (s) --")
            print(f"Mean {np.mean(avg_tbt_ok):.4f} | P50 {np.percentile(avg_tbt_ok,50):.4f} | P90 {np.percentile(avg_tbt_ok,90):.4f}")
        if worst_tbt_ok:
            print("-- Worst TBT (s) --")
            print(f"Mean {np.mean(worst_tbt_ok):.4f} | P50 {np.percentile(worst_tbt_ok,50):.4f} | P90 {np.percentile(worst_tbt_ok,90):.4f}")


def write_latency_csv(results, out_stem: str) -> Path:
    """
    Save detailed latency metrics to CSV.
    Columns:
      idx,t_rel_s,latency_s,status,ttft_s,avg_tbt_s,worst_tbt_s
    """
    path = Path(f"{out_stem}.csv").resolve()
    with path.open("w") as f:
        f.write("idx,t_rel_s,latency_s,status,ttft_s,avg_tbt_s,worst_tbt_s\n")
        for idx, t_rel, lat, st, ttft, avg_tbt, worst_tbt in results:
            ttft = "" if ttft is None else f"{ttft:.6f}"
            avg_tbt = "" if avg_tbt is None else f"{avg_tbt:.6f}"
            worst_tbt = "" if worst_tbt is None else f"{worst_tbt:.6f}"
            f.write(f"{idx},{t_rel:.6f},{lat:.6f},{st},{ttft},{avg_tbt},{worst_tbt}\n")

    print(f"[orchestrator] Wrote CSV: {path}")
    return path

from pynvml import (
    nvmlInit, nvmlShutdown, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetPowerUsage, nvmlDeviceGetTemperature,
    NVML_TEMPERATURE_GPU,
)
from typing import Optional, List
from datetime import datetime
import asyncio
import math


async def monitor_gpu_usage(
    log_path: str = "gpu_usage_log.csv",
    interval_s: float = 0.2,
    aggregation_period_s: float = 0.5,     # NEW ✓ group samples into windows
    stop_event: Optional[asyncio.Event] = None,
):
    """
    High-resolution GPU monitor:
      • Samples GPU every interval_s
      • Aggregates samples into windows of aggregation_period_s
      • Writes only the peak values (per GPU) for each window
      • Column name = timestamp (string)
    """

    print(f"[monitor] Starting GPU monitor → aggregation window = {aggregation_period_s}s")

    # CSV header
    buffer: List[str] = []
    buffer.append(
        "timestamp,gpu_index,gpu_util,"
        "memory_used_mb,memory_total_mb,power_w,temperature_c\n"
    )

    try:
        nvmlInit()
        device_count = nvmlDeviceGetCount()
        print(f"[monitor] {device_count} GPU(s) detected → writing to {log_path}")

        # Track window boundaries
        t0 = datetime.now().timestamp()
        window_index = 0  # incremented each window

        # Accumulators: gpu_id → dict of peak metrics
        accumulators = {}

        while True:
            if stop_event is not None and stop_event.is_set():
                break

            now = datetime.now()
            now_ts = now.timestamp()

            # Determine current window index
            current_window = int((now_ts - t0) // aggregation_period_s)

            # If window changed → flush previous window
            if current_window != window_index and accumulators:
                # Window closing → write peaks
                timestamp_str = datetime.fromtimestamp(
                    t0 + window_index * aggregation_period_s
                ).strftime("%Y-%m-%d %H:%M:%S")

                for gpu_id, peak in accumulators.items():
                    buffer.append(
                        f"{timestamp_str},{gpu_id},{peak['util']},"
                        f"{peak['mem_used']:.1f},{peak['mem_total']:.1f},"
                        f"{peak['power']:.1f},{peak['temp']}\n"
                    )

                # Reset for next window
                window_index = current_window
                accumulators = {}

            # Sample GPUs and record peaks
            for i in range(device_count):
                handle = nvmlDeviceGetHandleByIndex(i)
                util = nvmlDeviceGetUtilizationRates(handle).gpu
                mem = nvmlDeviceGetMemoryInfo(handle)
                power = nvmlDeviceGetPowerUsage(handle) / 1000.0
                temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)

                acc = accumulators.setdefault(i, {
                    "util": 0,
                    "mem_used": 0,
                    "mem_total": mem.total / 1024**2,
                    "power": 0,
                    "temp": 0,
                })

                acc["util"] = max(acc["util"], util)
                acc["mem_used"] = max(acc["mem_used"], mem.used / 1024**2)
                acc["power"] = max(acc["power"], power)
                acc["temp"] = max(acc["temp"], temp)

            # Sleep while remaining cancellable
            try:
                await asyncio.sleep(interval_s)
            except asyncio.CancelledError:
                print("[monitor] Cancelled mid-sleep.")
                break

    except KeyboardInterrupt:
        print("[monitor] KeyboardInterrupt → stopping.")
    finally:
        try:
            nvmlShutdown()
        except:
            pass

        # Flush last partial window if any
        try:
            if accumulators:
                timestamp_str = datetime.fromtimestamp(
                    t0 + window_index * aggregation_period_s
                ).strftime("%Y-%m-%d %H:%M:%S")

                for gpu_id, peak in accumulators.items():
                    buffer.append(
                        f"{timestamp_str},{gpu_id},{peak['util']},"
                        f"{peak['mem_used']:.1f},{peak['mem_total']:.1f},"
                        f"{peak['power']:.1f},{peak['temp']}\n"
                    )
        except:
            pass

        # Write log file
        try:
            with open(log_path, "w") as f:
                f.writelines(buffer)

            print(f"[monitor] Stopped — wrote {len(buffer)-1} rows → {log_path}")
        except Exception as e:
            print(f"[monitor] Failed to write log: {e}")

async def run_warmup(
    server: str,
    schedule: List[Tuple[int, float, Dict]],
    warmup_end: float = 3.0
) -> List[Tuple[int, float, float, str]]:
    """
    Run warmup requests (t_off < warmup_end) using same aiohttp pattern as run_schedule,
    but without enforcing scheduled delays—warmup fires immediately.

    Returns warmup results.
    """
    warmup = [(idx, t_off, payload) for idx, t_off, payload in schedule if t_off < warmup_end]
    if not warmup:
        print("[warmup] No warmup requests found.")
        return []

    print(f"[warmup] Running {len(warmup)} warmup requests…")

    total = len(warmup)
    finished_count = 0
    lock = asyncio.Lock()
    t0 = time.monotonic()  # reference for t_rel
    print(f"[warmup] total warmup requests: {total}")
    async with aiohttp.ClientSession(headers={"User-Agent": "OrchestratorLoad"}) as session:
        async def fire_at(idx: int, t_off: float, payload: Dict):
            nonlocal finished_count
            delay = (t0 + t_off) - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            if idx % 5 == 0:
                print("[Orchestrator] Request {}/{} fired at server".format(idx + 1, total))
            result = await send_one(session, server, idx, payload, t0)
            async with lock:
                finished_count += 1
            return result

        tasks = [asyncio.create_task(fire_at(idx, t_off, payload)) for idx, t_off, payload in warmup]
        results = await asyncio.gather(*tasks)

    print("[warmup] Warmup complete.")
    return results

# ---------------- Main ----------------
async def main():
    ap = argparse.ArgumentParser(description="Replay requests from timeline.csv against a server.")
    ap.add_argument("--server", default=DEFAULTS["server"])
    ap.add_argument("--timeline-csv", default=DEFAULTS["timeline_csv"])
    ap.add_argument("--max-wait", type=float, default=DEFAULTS["max_wait"])
    ap.add_argument("--co", action="store_true")
    ap.add_argument("--inf", action="store_true")
    ap.add_argument("--enable_nsys", action="store_true", help="Launch server with Nsight Systems")
    args = ap.parse_args()

    if args.co:
        args.enable_finetuning = True
    elif args.inf:
        args.enable_finetuning = False

    schedule = load_schedule_from_csv(args.timeline_csv)
    print(f"[orchestrator] Schedule loaded: {len(schedule)} requests from {args.timeline_csv}")

    proc = launch_server(args.enable_finetuning)

     # ---------------- Warmup (first 3 seconds) ----------------
    WARMUP_END = 3
    await wait_for_server(args.server, max_wait_s=args.max_wait)
    await asyncio.sleep(1.0)
    print(f"[orchestrator] Starting warmup phase (first {WARMUP_END}s)…")
    await run_warmup(args.server, schedule, warmup_end=WARMUP_END)
    print("[orchestrator] Waiting 3s before starting measured schedule…")
    await asyncio.sleep(3.0)
    print(f"[orchestrator] Schedule after warmup: {len(schedule)} requests")

    stop_gpu_event = asyncio.Event()
    suffix = "co-serving" if args.enable_finetuning else "inference"
    gpu_log_path = f"{current_dir}/results/gpu_usage_{suffix}.csv"

    # Start GPU monitor in background
    gpu_monitor_task = asyncio.create_task(monitor_gpu_usage(gpu_log_path, interval_s=0.2, stop_event=stop_gpu_event))

    try:
        #await wait_for_server(args.server, max_wait_s=args.max_wait)
        if args.enable_finetuning:
            async with aiohttp.ClientSession() as session:
                await start_finetuning(session, args.server)
        results = await run_schedule(args.server, schedule)
        stop_gpu_event.set()
        await gpu_monitor_task
        #await wait_for_finetuning(args.server)
    finally:
        print("[orchestrator] Stopping server…")
        kill_server(proc)
        summarize(results)
        out_stem = f"{current_dir}/results/latency_{suffix}"
        write_latency_csv(results, out_stem)
        print("[orchestrator] Done.")


if __name__ == "__main__":
    try:
        import numpy as np
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[orchestrator] Interrupted — exiting…", flush=True)