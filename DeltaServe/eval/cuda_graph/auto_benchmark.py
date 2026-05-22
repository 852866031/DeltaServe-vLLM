#!/usr/bin/env python3
"""
orchestrate_run_timeline.py

- Launches launch_llama3.py in a separate process group (Linux/POSIX only)
- Streams server logs live
- Waits for server readiness
- Warmup: first N timeline rows, ignore timestamps, spread over warmup_duration_s
- Rest, then start finetuning (optional), then run full timeline (including warmup rows)
- Writes scheduled-phase request metrics to a CSV (warmup requests are NOT recorded)
"""

import argparse
import asyncio
import csv
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import aiohttp

from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent


# ----------------------------
# Request helpers
# ----------------------------
def make_payload(prompt: str, base_model: str, lora_dir: str, max_new_tokens: int) -> Dict:
    return {
        "model_dir": base_model,
        "lora_dir": lora_dir,
        "inputs": prompt,
        "parameters": {
            "do_sample": False,
            "ignore_eos": True,
            "max_new_tokens": max_new_tokens,
        },
    }


def make_prompt_from_length(prompt_length: int) -> str:
    """
    Build a deterministic prompt approximately matching prompt_length.
    Replace this with your own prompt source if needed.
    """
    base = "Instruction:\n"
    tail = "\n### Response: "
    filler_needed = max(0, prompt_length - len(base) - len(tail))
    filler = ("hello " * 10000)[:filler_needed]
    prompt = base + filler + tail
    if len(prompt) < prompt_length:
        prompt += "x" * (prompt_length - len(prompt))
    return prompt[:prompt_length]


async def try_health(session: aiohttp.ClientSession, server: str, timeout_s: float = 1.0) -> bool:
    try:
        async with session.get(f"{server.rstrip('/')}/health", timeout=timeout_s) as resp:
            return resp.status == 200
    except Exception:
        return False


async def try_generate_probe(
    session: aiohttp.ClientSession,
    server: str,
    base_model: str,
    lora_dir: str,
    timeout_s: float = 2.0,
) -> bool:
    try:
        async with session.post(
            f"{server.rstrip('/')}/generate",
            json=make_payload("ping", base_model=base_model, lora_dir=lora_dir, max_new_tokens=2),
            timeout=timeout_s,
        ) as resp:
            return resp.status == 200
    except Exception:
        return False


async def wait_for_server(
    server: str,
    base_model: str,
    lora_dir: str,
    max_wait_s: float = 180.0,
    poll_period_s: float = 0.5,
) -> None:
    t0 = time.monotonic()
    async with aiohttp.ClientSession() as session:
        while True:
            if await try_health(session, server) or await try_generate_probe(session, server, base_model, lora_dir):
                print(f"[orchestrator] Server is up ✅ at {server}", flush=True)
                return
            if time.monotonic() - t0 > max_wait_s:
                raise TimeoutError(f"Server didn't become healthy within {max_wait_s:.1f}s")
            await asyncio.sleep(poll_period_s)


async def send_one_request(
    session: aiohttp.ClientSession,
    server: str,
    idx: int,
    t_rel: float,
    prompt: str,
    base_model: str,
    lora_dir: str,
    max_new_tokens: int,
) -> Tuple[int, float, float, str, Optional[float], Optional[float], Optional[float], str]:
    """
    Returns:
      (idx, t_rel_s, latency_s, status, ttft_s, avg_tbt_s, worst_tbt_s, preview_text)
    """
    url = f"{server.rstrip('/')}/generate"
    payload = make_payload(prompt, base_model=base_model, lora_dir=lora_dir, max_new_tokens=max_new_tokens)

    t_send = time.monotonic()
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

        return (idx, t_rel, latency, "ok", ttft, avg_tbt, worst_tbt, out)
    except Exception as e:
        latency = time.monotonic() - t_send
        return (idx, t_rel, latency, f"error:{type(e).__name__}", None, None, None, str(e))


async def start_finetuning(session: aiohttp.ClientSession, server: str, timeout_s: float = 5.0) -> bool:
    try:
        async with session.post(f"{server.rstrip('/')}/start_finetuning", timeout=timeout_s) as resp:
            print(f"[orchestrator] start_finetuning status={resp.status}", flush=True)
            return resp.status == 200
    except Exception:
        return False


async def exit_finetuning(session: aiohttp.ClientSession, server: str) -> bool:
    try:
        async with session.post(f"{server.rstrip('/')}/exit_finetuning") as resp:
            return resp.status == 200
    except Exception:
        return False


# ----------------------------
# Timeline loading + scheduling
# ----------------------------
@dataclass
class TimelineRow:
    timestamp_s: float
    prompt_length: int
    max_new_tokens: int
    row_id: int


def load_timeline_csv(path: str) -> List[TimelineRow]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Timeline CSV not found: {path}")

    rows: List[TimelineRow] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"timestamp_s", "prompt_length", "max_new_tokens"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Timeline CSV missing required columns: {sorted(missing)}")

        for i, r in enumerate(reader):
            rows.append(
                TimelineRow(
                    timestamp_s=float(r["timestamp_s"]),
                    prompt_length=int(float(r["prompt_length"])),
                    max_new_tokens=int(float(r["max_new_tokens"])),
                    row_id=i,
                )
            )

    rows.sort(key=lambda x: x.timestamp_s)
    return rows


def write_results_csv(path: str, rows: List[Tuple[int, float, float, str, Optional[float], Optional[float], Optional[float]]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "t_rel_s", "latency_s", "status", "ttft_s", "avg_tbt_s", "worst_tbt_s"])
        for idx, t_rel, latency, status, ttft, avg_tbt, worst_tbt in rows:
            w.writerow([idx, t_rel, latency, status, ttft, avg_tbt, worst_tbt])
    print(f"[orchestrator] Wrote results CSV: {path}", flush=True)


async def run_warmup_requests(
    server: str,
    base_model: str,
    lora_dir: str,
    warmup_rows: List[TimelineRow],
    stop_event: asyncio.Event,
    warmup_duration_s: float = 2.0,
    request_timeout_s: float = 600.0,
) -> None:
    """
    Warmup phase:
    - Ignores row.timestamp_s
    - Spreads requests uniformly over warmup_duration_s
    - Waits for all to finish
    - Warmup requests are NOT recorded to the output CSV
    """
    if not warmup_rows:
        return

    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=request_timeout_s)

    async with aiohttp.ClientSession(
        headers={"User-Agent": "WarmupClient"},
        connector=connector,
        timeout=timeout,
    ) as session:
        t0 = time.monotonic()
        n = len(warmup_rows)
        spacing = warmup_duration_s / max(n, 1)

        print(f"[orchestrator] Warmup: {n} requests over {warmup_duration_s:.2f}s", flush=True)

        async def _run_one(slot: int, row: TimelineRow) -> None:
            target = t0 + slot * spacing
            delay = target - time.monotonic()
            if delay > 0:
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=delay)
                    return
                except asyncio.TimeoutError:
                    pass
            if stop_event.is_set():
                return

            t_rel = time.monotonic() - t0
            prompt = make_prompt_from_length(row.prompt_length)
            idx, _, latency, status, ttft, avg_tbt, worst_tbt, _ = await send_one_request(
                session=session,
                server=server,
                idx=row.row_id,
                t_rel=t_rel,
                prompt=prompt,
                base_model=base_model,
                lora_dir=lora_dir,
                max_new_tokens=row.max_new_tokens,
            )
            print(
                f"[warmup] idx={idx} t_rel={t_rel:.6f}s latency={latency:.6f}s status={status} "
                f"ttft={ttft} avg_tbt={avg_tbt} worst_tbt={worst_tbt}",
                flush=True,
            )

        tasks = [asyncio.create_task(_run_one(i, row)) for i, row in enumerate(warmup_rows)]
        await asyncio.gather(*tasks)
        print("[orchestrator] Warmup completed ✅", flush=True)


async def run_timeline_requests(
    server: str,
    base_model: str,
    lora_dir: str,
    timeline_rows: List[TimelineRow],
    stop_event: asyncio.Event,
    normalize_start: bool = True,
    request_timeout_s: float = 600.0,
) -> List[Tuple[int, float, float, str, Optional[float], Optional[float], Optional[float]]]:
    """
    Schedule requests according to timeline timestamp_s.

    Returns list of rows for results CSV:
      (idx, t_rel_s, latency_s, status, ttft_s, avg_tbt_s, worst_tbt_s)
    """
    if not timeline_rows:
        print("[orchestrator] No timeline rows to run.", flush=True)
        return []

    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=request_timeout_s)
    results: List[Tuple[int, float, float, str, Optional[float], Optional[float], Optional[float]]] = []

    base_ts = min(r.timestamp_s for r in timeline_rows) if normalize_start else 0.0

    async with aiohttp.ClientSession(
        headers={"User-Agent": "TimelineClient"},
        connector=connector,
        timeout=timeout,
    ) as session:
        t0 = time.monotonic()
        print(f"[orchestrator] Timeline start (normalize={normalize_start}, base_ts={base_ts:.6f})", flush=True)
        print(f"[orchestrator] Scheduling {len(timeline_rows)} requests", flush=True)

        async def _run_one(row: TimelineRow) -> None:
            target_rel = row.timestamp_s - base_ts
            target_abs = t0 + target_rel

            delay = target_abs - time.monotonic()
            if delay > 0:
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=delay)
                    return
                except asyncio.TimeoutError:
                    pass
            if stop_event.is_set():
                return

            t_rel = time.monotonic() - t0
            prompt = make_prompt_from_length(row.prompt_length)

            idx, t_rel_out, latency, status, ttft, avg_tbt, worst_tbt, out = await send_one_request(
                session=session,
                server=server,
                idx=row.row_id,
                t_rel=t_rel,
                prompt=prompt,
                base_model=base_model,
                lora_dir=lora_dir,
                max_new_tokens=row.max_new_tokens,
            )

            results.append((idx, t_rel_out, latency, status, ttft, avg_tbt, worst_tbt))

        tasks = [asyncio.create_task(_run_one(row)) for row in timeline_rows]
        await asyncio.gather(*tasks)

    # Sort by idx for stable CSV order like the sample
    results.sort(key=lambda x: x[0])
    print("[orchestrator] Timeline completed ✅", flush=True)
    return results


# ----------------------------
# Process orchestration (POSIX only)
# ----------------------------
def terminate_process_tree_fast(p: subprocess.Popen, grace_s: float = 0.15) -> None:
    if p is None or p.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGINT)
    except Exception:
        try:
            p.terminate()
        except Exception:
            pass

    t0 = time.monotonic()
    while time.monotonic() - t0 < grace_s:
        if p.poll() is not None:
            return
        time.sleep(0.01)

    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass


async def main() -> None:
    ap = argparse.ArgumentParser()

    # minimal required args
    ap.add_argument("--timeline_csv", default=str(SCRIPT_DIR / "timeline_live.csv"))
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B")
    ap.add_argument("--lora_dir", default=str(SCRIPT_DIR / "adapters" / "llama3-toy-lora"))

    # small set of useful knobs
    ap.add_argument("--launcher", default="../llama3/launch_llama3.py")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--rank_id", type=int, default=0)
    ap.add_argument("--co", action="store_true")  # enable finetuning mode
    ap.add_argument("--graph", action="store_true", default=False)  # enable CUDA graph
    ap.add_argument("--out_csv", default="timeline_results.csv")

    # warmup config
    ap.add_argument("--warmup_count", type=int, default=15)
    ap.add_argument("--warmup_duration_s", type=float, default=3.0)
    ap.add_argument("--warmup_rest_s", type=float, default=2.0)

    args = ap.parse_args()

    timeline_rows = load_timeline_csv(args.timeline_csv)
    print(f"[orchestrator] Loaded {len(timeline_rows)} rows from {args.timeline_csv}", flush=True)
    if timeline_rows:
        print(
            f"[orchestrator] Timeline range: {timeline_rows[0].timestamp_s:.6f}s -> {timeline_rows[-1].timestamp_s:.6f}s",
            flush=True,
        )

    server = f"http://127.0.0.1:{args.port}"

    cmd = [
        sys.executable,
        "-u",
        args.launcher,
        "--port",
        str(args.port),
        "--rank_id",
        str(args.rank_id),
    ]
    if args.co:
        cmd.append("--enable-finetuning")
    if args.graph:
        cmd.append("--enable-cuda-graph")
        args.out_csv = args.out_csv.replace(".csv", "_graph.csv")

    print("[orchestrator] launching:", " ".join(cmd), flush=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    p: Optional[subprocess.Popen] = None
    log_task: Optional[asyncio.Task] = None
    stop_event = asyncio.Event()

    try:
        # POSIX/Linux only
        p = subprocess.Popen(
            cmd,
            preexec_fn=os.setsid,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        async def pump_logs() -> None:
            assert p is not None and p.stdout is not None
            loop = asyncio.get_running_loop()
            while True:
                line = await loop.run_in_executor(None, p.stdout.readline)
                if not line:
                    break
                print("[server]", line.rstrip(), flush=True)

        log_task = asyncio.create_task(pump_logs())

        loop = asyncio.get_running_loop()

        def _on_sigint() -> None:
            if p is not None and p.poll() is None:
                terminate_process_tree_fast(p, grace_s=0.1)
            stop_event.set()

        loop.add_signal_handler(signal.SIGINT, _on_sigint)

        # Wait for server
        waiter = asyncio.create_task(
            wait_for_server(
                server=server,
                base_model=args.base_model,
                lora_dir=args.lora_dir,
                max_wait_s=240.0,
                poll_period_s=0.5,
            )
        )
        stopper = asyncio.create_task(stop_event.wait())
        done, pending = await asyncio.wait({waiter, stopper}, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()

        if stop_event.is_set():
            return

        # Warmup (first N rows, ignored timestamps)
        warmup_count = max(0, min(args.warmup_count, len(timeline_rows)))
        warmup_rows = timeline_rows[:warmup_count]

        if warmup_rows:
            await run_warmup_requests(
                server=server,
                base_model=args.base_model,
                lora_dir=args.lora_dir,
                warmup_rows=warmup_rows,
                stop_event=stop_event,
                warmup_duration_s=args.warmup_duration_s,
            )
            if stop_event.is_set():
                return

            print(f"[orchestrator] Resting {args.warmup_rest_s:.2f}s after warmup...", flush=True)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=args.warmup_rest_s)
                return
            except asyncio.TimeoutError:
                pass

        # Start finetuning after warmup+rest, before schedule
        if args.co:
            print("[orchestrator] Starting finetuning...", flush=True)
            async with aiohttp.ClientSession() as session:
                ok = await start_finetuning(session, server)
                if not ok:
                    print("[orchestrator] Failed to start finetuning", flush=True)
                    return
                print("[orchestrator] Finetuning started ✅", flush=True)

        # Run full timeline (including warmup rows again)
        results = await run_timeline_requests(
            server=server,
            base_model=args.base_model,
            lora_dir=args.lora_dir,
            timeline_rows=timeline_rows,
            stop_event=stop_event,
            normalize_start=True,
        )

        # Write CSV (scheduled phase only)
        write_results_csv(args.out_csv, results)

        # Exit finetuning after schedule
        if args.co and not stop_event.is_set():
            print("[orchestrator] Exiting finetuning...", flush=True)
            async with aiohttp.ClientSession() as session:
                ok = await exit_finetuning(session, server)
                print("[orchestrator] Exited finetuning ✅" if ok else "[orchestrator] Failed to exit finetuning", flush=True)

        await asyncio.sleep(0.5)

    except KeyboardInterrupt:
        if p is not None:
            terminate_process_tree_fast(p, grace_s=0.1)
    finally:
        if p is not None and p.poll() is None:
            print("[orchestrator] shutting down server…", flush=True)
            terminate_process_tree_fast(p, grace_s=0.1)

        if log_task is not None:
            log_task.cancel()
            try:
                await asyncio.wait_for(log_task, timeout=0.5)
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())