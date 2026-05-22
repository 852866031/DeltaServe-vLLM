# simple_benchmark.py
"""A minimal async benchmark script that sends 20 inference requests to an
HTTP server exposing the /generate endpoint used by the Slora examples.

* The first 10 requests are fired immediately.
* The script then waits **4 seconds** before firing the next 10 requests.

Only two external dependencies are required:
    aiohttp  (pip install aiohttp)
    tqdm     (optional — progress bar)

Usage
-----
$ python simple_benchmark.py --server http://localhost:8000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List

import aiohttp
from tqdm import tqdm
import time
from launch_server import base_model, adapter_dirs

# -----------------------------------------------------------------------------
# Helper – build a request payload
# -----------------------------------------------------------------------------

def make_payload(prompt: str, output_len: int) -> Dict:
    """Return the JSON body expected by the /generate route."""
    return {
        "model_dir": base_model,         # adapt as needed
        #"lora_dir": "/home/jiaxuan/Documents/Projects/slora-plus/S-LoRA/test/test_e2e/finetuning_adapter",               # adapt as needed
        "lora_dir": "tloen/alpaca-lora-7b",
        "inputs": prompt,
        "parameters": {
            "do_sample": False,
            "ignore_eos": True,
            "max_new_tokens": 10,
        },
    }

# -----------------------------------------------------------------------------
# Core request coroutine
# -----------------------------------------------------------------------------

async def send_request(session: aiohttp.ClientSession, server: str, idx: int, prompt: str, output_len: int) -> float:
    """Send *one* generation request and return the latency (s)."""
    url = f"{server.rstrip('/')}/generate"
    payload = make_payload(prompt, output_len)

    start = time.time()
    async with session.post(url, json=payload) as resp:
        # We assume the server streams chunks but reading the whole body is fine
        body = await resp.read()
        try:
            result = json.loads(body)
            generated = result.get("generated_text", ["<no-text>"])[0]
        except json.JSONDecodeError:
            generated = body.decode(errors="replace")
    latency = time.time() - start

    print(f"[req {idx:02d}] prompt: {prompt} latency={latency*1000:7.1f} ms  →  {generated}")
    return latency


async def run_benchmark(
    server: str,
    prompts: List[str] = ["Capital of France is", "i am feeling a bit restless these days <label>"],
    per_wave: int = 10,
    wait_interval: float = 1,
    num_waves: int = 5
):
    """Fire N requests in multiple waves with a delay between them, using rotating prompts."""

    requests: List[tuple[int, str, int]] = [
        (i, prompts[i % len(prompts)], 32) for i in range(per_wave * num_waves)
    ]

    async with aiohttp.ClientSession(headers={"User-Agent": "SimpleBenchmark"}) as session:
        for wave_idx in range(num_waves):
            print(f"\nStarting wave {wave_idx + 1}/{num_waves}…")
            wave_reqs = [
                send_request(session, server, idx, prompt, out_len)
                for idx, prompt, out_len in requests[wave_idx * per_wave : (wave_idx + 1) * per_wave]
            ]
            await asyncio.gather(*wave_reqs)

            if wave_idx < num_waves - 1:
                print(f"\nWaiting {wait_interval} seconds before the next wave…\n")
                await asyncio.sleep(wait_interval)

    print("\nBenchmark finished ✅")
# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal async benchmark for /generate endpoint")
    parser.add_argument("--server", type=str, default="http://localhost:8000", help="Base URL of inference server")
    args = parser.parse_args()

    try:
        asyncio.run(run_benchmark(args.server))
    except KeyboardInterrupt:
        print("Interrupted — exiting…")
