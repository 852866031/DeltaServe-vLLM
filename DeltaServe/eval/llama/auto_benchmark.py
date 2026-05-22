#!/usr/bin/env python3
import asyncio
import argparse
import aiohttp
import signal
import sys
import os
from pathlib import Path
from benchmark_utils.gpu_monitor import monitor_gpu_usage
import subprocess

from benchmark_utils.common import (
    load_schedule, summarize, write_latency_csv,
    wait_for_server, start_finetuning
)
from benchmark_utils.multi_instance import (
    launch_instance, kill_instance,
    warmup_instances, run_rr_schedule
)

current_dir = os.path.dirname(os.path.abspath(__file__))

INSTANCE_CONFIG = [
    {"port": 8000, "enable_finetuning": False},
    #{"port": 9000, "enable_finetuning": True},
]

DEFAULTS = {
    "timeline_csv": f"{current_dir}/timelines/timeline_live.csv",
    "max_wait": 120.0,
    "out_csv": f"{current_dir}/results/latency_multi_co.csv",
    "out_usage": f"{current_dir}/results/gpu_usage_multi_co.csv"
}

# ---------------- GLOBAL STATE FOR CLEAN EXIT ----------------
running_process = None

def handle_sigint(signum, frame):
    """Capture CTRL-C and kill the launched instance."""
    print("\n[auto] Caught CTRL-C — cleaning up ...")
    global running_process
    if running_process is not None:
        print("[auto] Killing server instance...")
        kill_instance(running_process)
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)


# ==========================================================
# Main
# ==========================================================
async def main():
    global running_process

    ap = argparse.ArgumentParser(description="Auto Benchmark / Instance Launcher")
    ap.add_argument("instance", type=int, nargs="?", default=None,
                    help="Instance index to launch. If omitted: run scheduler only.")
    ap.add_argument("--timeline-csv", default=DEFAULTS["timeline_csv"])
    args = ap.parse_args()

    # ======================================================
    # CASE 1 — Launch a single system instance and wait
    # ======================================================
    if args.instance is not None:
        idx = args.instance
        if idx < 0 or idx >= len(INSTANCE_CONFIG):
            print(f"[auto] Invalid instance index {idx}")
            return

        cfg = INSTANCE_CONFIG[idx]
        port = cfg["port"]
        gpu = idx
        enable_ft = cfg["enable_finetuning"]

        print(f"[auto] Launching instance #{idx}: port={port}, gpu={gpu}, ft={enable_ft}")

        running_process = launch_instance(
            port=port,
            gpu=gpu,
            enable_finetuning=enable_ft,
            current_dir=current_dir,
            instance_index=idx
        )

        print("[auto] Instance launched. Press CTRL-C to stop it.")
        # Keep script alive until CTRL-C
        while True:
            await asyncio.sleep(1)

    # ======================================================
    # CASE 2 — Run request scheduler (multi-instance)
    # ======================================================
    schedule = load_schedule(args.timeline_csv)
    print(f"[auto] Loaded {len(schedule)} requests")

    # Wait for all servers
    servers = [f"http://localhost:{cfg['port']}" for cfg in INSTANCE_CONFIG]
    print(f"[auto] Waiting for {len(servers)} servers...")

    ready = 0
    for s in servers:
        try:
            await wait_for_server(s)
            ready += 1
            print(f"[auto] {ready}/{len(servers)} ready")
        except:
            print(f"[auto] Could not connect to {s}")
            sys.exit(1)

    print("[auto] All instances alive.")

    # Warmup
    print("[auto] Warmup...")
    await warmup_instances(servers, schedule)
    await asyncio.sleep(3)

     # ---------------- GPU MONITOR START (🔥) ----------------
    gpu_log_path = DEFAULTS["out_usage"]
    print(f"[auto] Starting GPU monitor: {gpu_log_path}")

    stop_gpu_event = asyncio.Event()
    gpu_monitor_task = asyncio.create_task(
        monitor_gpu_usage(
            log_path=gpu_log_path,
            interval_s=0.2,
            stop_event=stop_gpu_event,
        )
    )
    # -------------------------------------------------------
    # Finetuning triggers
    async with aiohttp.ClientSession() as session:
        for cfg, server in zip(INSTANCE_CONFIG, servers):
            if cfg["enable_finetuning"]:
                print(f"[auto] Starting finetuning for {server}")
                await start_finetuning(session, server)

    # Run actual benchmark
    print("[auto] Running schedule (RR)...")
    results = await run_rr_schedule(servers, schedule)

    summarize(results)
    out_stem = DEFAULTS["out_csv"]
    write_latency_csv(results, out_stem)
    print("[auto] Benchmark completed.")

    # ======================================================
    # RUN clean.sh AFTER scheduler completes
    # ======================================================
    # os.system("ps aux | grep 'jiaxuan' | grep 'slora' | grep -v grep | awk '{print $2}' | xargs -r kill -9")
    # os.system("ps aux | grep 'jiaxuan' | grep 'auto_benchmark.py ' | grep -v grep | awk '{print $2}' | xargs -r kill -9")
    print("[auto] Done.")


if __name__ == "__main__":
    import numpy as np
    asyncio.run(main())