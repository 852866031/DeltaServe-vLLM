#!/usr/bin/env python3
"""
bench_cuda_graph.py

Compares decode performance with and without CUDA graph.
Launches the API server directly (no dependency on launch_llama3.py),
sends sequential requests, and reports timing comparison.

Usage:
    python bench_cuda_graph.py
    python bench_cuda_graph.py --num-requests 20 --max-new-tokens 64
    python bench_cuda_graph.py --python /path/to/python  # custom python binary
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from typing import Dict, List

# ─── Configuration ───────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

BASE_MODEL = "meta-llama/Meta-Llama-3-8B"
ADAPTER_DIR = os.path.join(SCRIPT_DIR, "adapters", "llama3-toy-lora")
FT_LOG_PATH = os.path.join(SCRIPT_DIR, "bwd_log.csv")
NO_FT_CONFIG = os.path.join(SCRIPT_DIR, "config", "no_finetuning_config.json")

DEFAULT_SERVER_ARGS = [
    "--max_total_token_num", "25000",
    "--model", BASE_MODEL,
    "--tokenizer_mode", "auto",
    "--pool-size-lora", "0",
    "--rank_id", "0",
    "--ft_log_path", FT_LOG_PATH,
    "--finetuning_config_path", NO_FT_CONFIG,
    "--lora", ADAPTER_DIR,
    "--swap",
    "--enable_unified_mem_manager",
    "--unified_mem_manager_max_size", "6",
]


# ─── Server management ──────────────────────────────────────────────────

def find_python():
    """Try to find the dserve conda python that has slora installed."""
    candidates = [
        os.path.expanduser("~/miniconda3/envs/dserve/bin/python"),
        sys.executable,
    ]
    for p in candidates:
        if os.path.isfile(p):
            ret = subprocess.run([p, "-c", "import dserve"], capture_output=True)
            if ret.returncode == 0:
                return p
    print("ERROR: Cannot find a python with slora installed.")
    print("  Use --python /path/to/python to specify one.")
    sys.exit(1)


def wait_for_server(url: str, timeout_s: float = 180.0):
    """Poll the server until it responds or timeout."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            req = urllib.request.Request(f"{url}/generate", method="GET")
            urllib.request.urlopen(req, timeout=2)
            return True
        except urllib.error.HTTPError:
            # Server is up (returned an HTTP error, e.g., 405 Method Not Allowed)
            return True
        except Exception:
            time.sleep(2)
    return False


def wait_for_port_free(port: int, timeout_s: float = 30.0):
    """Wait until the port is free."""
    import socket
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return True
        time.sleep(1)
    return False


def launch_server(python_bin: str, port: int, enable_cuda_graph: bool) -> subprocess.Popen:
    cmd = [python_bin, "-m", "slora.server.api_server",
           "--port", str(port)] + DEFAULT_SERVER_ARGS
    if enable_cuda_graph:
        cmd.append("--enable-cuda-graph")

    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT + ((":" + env.get("PYTHONPATH", "")) if env.get("PYTHONPATH") else "")

    label = "CUDA Graph ON" if enable_cuda_graph else "CUDA Graph OFF"
    log_file = f"/tmp/bench_cg_{'on' if enable_cuda_graph else 'off'}.log"
    print(f"  [{label}] Launching server on port {port} ...")
    print(f"  [{label}] Log: {log_file}")

    f_log = open(log_file, "w")
    proc = subprocess.Popen(
        cmd, stdout=f_log, stderr=subprocess.STDOUT,
        env=env, preexec_fn=os.setsid,
    )
    proc._log_file = f_log  # keep reference to close later
    return proc


def kill_server(proc: subprocess.Popen):
    try:
        proc._log_file.close()
    except Exception:
        pass
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=10)
    except Exception:
        pass


# ─── Benchmark ───────────────────────────────────────────────────────────

def send_request(server: str, prompt: str, max_new_tokens: int) -> Dict:
    payload = json.dumps({
        "model_dir": BASE_MODEL,
        "lora_dir": ADAPTER_DIR,
        "inputs": prompt,
        "parameters": {
            "do_sample": False,
            "ignore_eos": True,
            "max_new_tokens": max_new_tokens,
        },
    }).encode()
    req = urllib.request.Request(
        f"{server}/generate", data=payload,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    wall = time.time() - t0
    return {
        "wall": wall,
        "ttft": data.get("ttft"),
        "avg_tbt": data.get("avg_tbt"),
        "worst_tbt": data.get("worst_tbt"),
        "tokens": data.get("count_output_tokens"),
        "text": data.get("generated_text", [""])[0][:80],
    }


def run_benchmark(server: str, label: str, num_warmup: int, num_requests: int,
                  prompt: str, max_new_tokens: int) -> Dict:
    # Warmup
    print(f"  [{label}] Warming up ({num_warmup} requests) ...")
    for i in range(num_warmup):
        r = send_request(server, prompt, max_new_tokens)
        print(f"    warmup {i}: wall={r['wall']:.3f}s avg_tbt={r['avg_tbt']:.4f}s")

    # Benchmark
    print(f"  [{label}] Benchmarking ({num_requests} requests) ...")
    results = []
    for i in range(num_requests):
        r = send_request(server, prompt, max_new_tokens)
        results.append(r)
        print(f"    req {i:2d}: wall={r['wall']:.3f}s  avg_tbt={r['avg_tbt']:.4f}s  worst_tbt={r['worst_tbt']:.4f}s")

    avg_wall = sum(r['wall'] for r in results) / len(results)
    avg_tbt = sum(r['avg_tbt'] for r in results) / len(results)
    avg_worst = sum(r['worst_tbt'] for r in results) / len(results)
    avg_ttft = sum(r['ttft'] for r in results) / len(results)
    throughput = max_new_tokens / avg_wall

    print(f"\n  {'=' * 55}")
    print(f"  {label}  ({num_requests} requests, {max_new_tokens} tokens each)")
    print(f"  {'=' * 55}")
    print(f"  Avg wall time:  {avg_wall:.4f}s")
    print(f"  Avg TTFT:       {avg_ttft:.4f}s")
    print(f"  Avg TBT:        {avg_tbt:.4f}s")
    print(f"  Avg worst TBT:  {avg_worst:.4f}s")
    print(f"  Decode tok/s:   {throughput:.1f}")
    print(f"  Sample output:  {results[0]['text']!r}")

    return {
        "avg_wall": avg_wall,
        "avg_ttft": avg_ttft,
        "avg_tbt": avg_tbt,
        "avg_worst_tbt": avg_worst,
        "throughput": throughput,
    }


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark CUDA graph decode speedup")
    parser.add_argument("--python", type=str, default=None,
                        help="Path to python binary with slora installed (auto-detected if omitted)")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--num-warmup", type=int, default=3,
                        help="Number of warmup requests before benchmarking")
    parser.add_argument("--num-requests", type=int, default=10,
                        help="Number of benchmark requests per mode")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prompt", type=str,
                        default="Hello, tell me about machine learning.",
                        help="Prompt text for benchmark requests")
    args = parser.parse_args()

    python_bin = args.python or find_python()
    print(f"Using python: {python_bin}")
    print(f"Repo root:    {REPO_ROOT}")
    print(f"Adapter:      {ADAPTER_DIR}")
    print()

    all_results = {}

    for enable_cg in [False, True]:
        label = "CUDA Graph ON" if enable_cg else "Baseline (OFF)"
        print(f"\n{'#' * 60}")
        print(f"# {label}")
        print(f"{'#' * 60}\n")

        # Make sure port is free
        if not wait_for_port_free(args.port):
            print(f"  ERROR: Port {args.port} is still in use. Kill the process and retry.")
            sys.exit(1)

        proc = launch_server(python_bin, args.port, enable_cg)
        server = f"http://127.0.0.1:{args.port}"

        try:
            print(f"  [{label}] Waiting for server (up to 180s) ...")
            if not wait_for_server(server, timeout_s=180):
                print(f"  ERROR: Server failed to start. Check log file.")
                continue

            print(f"  [{label}] Server is ready.\n")
            stats = run_benchmark(
                server, label,
                args.num_warmup, args.num_requests,
                args.prompt, args.max_new_tokens,
            )
            all_results[label] = stats

        except Exception as e:
            print(f"  ERROR: {e}")
        finally:
            print(f"\n  [{label}] Stopping server ...")
            kill_server(proc)
            # Wait for port to be freed (NCCL port 28765 can linger)
            time.sleep(5)
            # Kill any remaining child processes on NCCL port
            os.system("ss -tlnp 2>/dev/null | grep 28765 | grep -oP 'pid=\\K[0-9]+' | xargs kill -9 2>/dev/null")
            time.sleep(3)

    # ─── Comparison ──────────────────────────────────────────────────────
    if len(all_results) == 2:
        off = all_results["Baseline (OFF)"]
        on = all_results["CUDA Graph ON"]

        print(f"\n{'=' * 60}")
        print(f"  COMPARISON SUMMARY")
        print(f"{'=' * 60}")
        print(f"  {'Metric':<20s}  {'Baseline':>10s}  {'CUDA Graph':>10s}  {'Speedup':>8s}")
        print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*8}")

        for key, label, fmt in [
            ("avg_wall",      "Avg wall time",  ".4f"),
            ("avg_ttft",      "Avg TTFT",       ".4f"),
            ("avg_tbt",       "Avg TBT",        ".4f"),
            ("avg_worst_tbt", "Avg worst TBT",  ".4f"),
            ("throughput",    "Decode tok/s",    ".1f"),
        ]:
            v_off = off[key]
            v_on = on[key]
            if key == "throughput":
                speedup = v_on / v_off if v_off > 0 else 0
            else:
                speedup = v_off / v_on if v_on > 0 else 0
            print(f"  {label:<20s}  {v_off:>10{fmt}}  {v_on:>10{fmt}}  {speedup:>7.2f}x")

        print()
    else:
        print("\nNot enough data for comparison.")


if __name__ == "__main__":
    main()
