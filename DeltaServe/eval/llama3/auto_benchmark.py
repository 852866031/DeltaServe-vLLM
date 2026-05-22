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
import datetime
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import aiohttp
from tqdm import tqdm

from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent


def _detect_gpu_subdir() -> str:
    """Return the timelines/ subdirectory name matching the local GPU.

    Greps `nvidia-smi -L` for the GPU model. Falls back to '5090' if
    detection fails — that's the workstation default; override via
    --timeline-gpu when running on a different machine.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True, timeout=2.0,
        )
        name = (out.strip().splitlines() or [""])[0].upper()
        if "A100" in name:
            return "A100"
        if "5090" in name:
            return "5090"
    except Exception:
        pass
    return "5090"


# Resolved once at import; can be overridden by the --timeline-gpu CLI flag.
_DEFAULT_GPU_SUBDIR = _detect_gpu_subdir()
TIMELINES_DIR = SCRIPT_DIR / "timelines" / _DEFAULT_GPU_SUBDIR


# Module-level reference to the active progress bar so the log pump can
# route output through tqdm.write() and keep the bar anchored at the
# bottom of the terminal while the timeline runs.
_active_pbar: Optional["tqdm"] = None


def _emit_line(line: str) -> None:
    """Write a complete log line. Goes through tqdm.write when the
    timeline progress bar is active so the bar stays at the bottom."""
    if _active_pbar is not None:
        tqdm.write(line, file=sys.stdout)
    else:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


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


def trim_bwd_log_before(path: str, cutoff: datetime.datetime) -> Tuple[int, int]:
    """Strip rows from `path` whose `timestamp` column is strictly before
    `cutoff` (compared as ISO-8601-second strings — the same format the
    server uses in `finetuning_store.write_bwd_logs_csv`). Rewrites the
    file in place. Returns (kept, total).

    Why string comparison: bwd_log writes `timestamp` via
    `datetime.now().isoformat(timespec="seconds")` → `YYYY-MM-DDTHH:MM:SS`.
    These strings sort lexicographically the same as their datetime
    counterparts, so a `>= cutoff_str` comparison is exact and avoids
    parsing every row.

    Inclusivity: the cutoff is floored to the same-second precision the
    bwd log uses, so rows recorded in the *same second* as `cutoff` are
    kept (post-warmup batches that happened to complete just before the
    first request landed). Earlier-second rows are dropped.
    """
    if not os.path.exists(path):
        print(f"[orchestrator] Skipping bwd-log trim: {path} does not exist", flush=True)
        return (0, 0)

    cutoff_iso = cutoff.replace(microsecond=0).isoformat(timespec="seconds")
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        all_rows = list(reader)
    total = len(all_rows)
    if not fieldnames or "timestamp" not in fieldnames:
        print(f"[orchestrator] Skipping bwd-log trim: {path} has no 'timestamp' column", flush=True)
        return (total, total)

    kept = [r for r in all_rows if (r.get("timestamp") or "") >= cutoff_iso]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    print(
        f"[orchestrator] Trimmed bwd log {path}: kept {len(kept)}/{total} rows "
        f"(cutoff T = {cutoff_iso}, dropped {total - len(kept)} pre-warmup row(s))",
        flush=True,
    )
    return (len(kept), total)


WARMUP_START_OFFSET_S = 1.0


async def run_warmup_requests(
    server: str,
    base_model: str,
    lora_dir: str,
    warmup_rows: List[TimelineRow],
    stop_event: asyncio.Event,
    request_timeout_s: float = 600.0,
) -> None:
    """
    Warmup phase: replays the same timestamp-based schedule as the main
    timeline, but with the first row's timestamp normalized to
    WARMUP_START_OFFSET_S (so a first row at t=45s doesn't wait 45s).
    Warmup requests are NOT recorded to the output CSV.
    """
    if not warmup_rows:
        return

    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=request_timeout_s)

    base_ts = min(r.timestamp_s for r in warmup_rows)
    span = max(r.timestamp_s for r in warmup_rows) - base_ts

    async with aiohttp.ClientSession(
        headers={"User-Agent": "WarmupClient"},
        connector=connector,
        timeout=timeout,
    ) as session:
        t0 = time.monotonic()
        n = len(warmup_rows)
        print(
            f"[orchestrator] Warmup: {n} requests following timeline "
            f"(first at +{WARMUP_START_OFFSET_S:.2f}s, span {span:.2f}s)",
            flush=True,
        )

        async def _run_one(row: TimelineRow) -> None:
            target_rel = (row.timestamp_s - base_ts) + WARMUP_START_OFFSET_S
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
            if idx % 10 == 0:
                print(
                    f"[warmup] idx={idx} t_rel={t_rel:.6f}s latency={latency:.6f}s status={status} "
                    f"ttft={ttft} avg_tbt={avg_tbt} worst_tbt={worst_tbt}",
                    flush=True,
                )

        tasks = [asyncio.create_task(_run_one(row)) for row in warmup_rows]
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
) -> Tuple[
    List[Tuple[int, float, float, str, Optional[float], Optional[float], Optional[float]]],
    Optional[datetime.datetime],
]:
    """
    Schedule requests according to timeline timestamp_s.

    Returns:
      results: list of rows for results CSV
        (idx, t_rel_s, latency_s, status, ttft_s, avg_tbt_s, worst_tbt_s)
      t_first_wall: wall-clock datetime captured at `t0`, i.e. the moment
        the scheduler arms the first request. Used by the caller to trim
        pre-timeline rows from the bwd-log CSV (which timestamps in
        wall-clock ISO-second strings). None if there were no rows.
    """
    if not timeline_rows:
        print("[orchestrator] No timeline rows to run.", flush=True)
        return [], None

    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=request_timeout_s)
    results: List[Tuple[int, float, float, str, Optional[float], Optional[float], Optional[float]]] = []

    base_ts = min(r.timestamp_s for r in timeline_rows) if normalize_start else 0.0
    # Total schedule duration is the last row's relative timestamp — the
    # progress bar tracks wall-clock progress against this.
    total_duration = max(r.timestamp_s for r in timeline_rows) - base_ts

    async with aiohttp.ClientSession(
        headers={"User-Agent": "TimelineClient"},
        connector=connector,
        timeout=timeout,
    ) as session:
        t0 = time.monotonic()
        # Wall-clock anchor for filtering the bwd_log later. Captured at
        # the same instant as `t0`; for normalized timelines the first
        # row has target_rel == 0, so this is within microseconds of the
        # first request actually going out.
        t_first_wall = datetime.datetime.now()
        print(f"[orchestrator] Timeline start (normalize={normalize_start}, base_ts={base_ts:.6f})", flush=True)
        print(f"[orchestrator]   wall-clock anchor T = {t_first_wall.isoformat(timespec='seconds')}", flush=True)
        print(f"[orchestrator] Scheduling {len(timeline_rows)} requests "
              f"(total schedule duration: {total_duration:.2f}s)", flush=True)

        global _active_pbar
        pbar = tqdm(
            total=max(total_duration, 1e-3),
            desc="Timeline",
            unit="s",
            bar_format="{desc}: {percentage:5.1f}%|{bar}| {n:6.1f}/{total:.1f}s "
                       "[{elapsed}<{remaining}]",
            leave=True,
            position=0,
            dynamic_ncols=True,
            file=sys.stdout,
            mininterval=0.1,
        )
        _active_pbar = pbar

        async def _progress_updater() -> None:
            last = 0.0
            while not stop_event.is_set():
                elapsed = min(time.monotonic() - t0, total_duration)
                delta = elapsed - last
                if delta > 0:
                    pbar.update(delta)
                    last = elapsed
                if elapsed >= total_duration:
                    return
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=0.5)
                    return
                except asyncio.TimeoutError:
                    continue

        progress_task = asyncio.create_task(_progress_updater())

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
        try:
            await asyncio.gather(*tasks)
        finally:
            progress_task.cancel()
            try:
                await progress_task
            except (asyncio.CancelledError, Exception):
                pass
            # Snap bar to 100% on clean completion so the final render
            # reflects the full schedule.
            if not stop_event.is_set() and pbar.n < pbar.total:
                pbar.update(pbar.total - pbar.n)
            pbar.refresh()
            pbar.close()
            _active_pbar = None

    # Sort by idx for stable CSV order like the sample
    results.sort(key=lambda x: x[0])
    print("[orchestrator] Timeline completed ✅", flush=True)
    return results, t_first_wall


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
    ap.add_argument("--timeline_csv", default=None,
                    help="Path to the request schedule CSV. Default: "
                         "timelines/<gpu>/timeline_live.csv resolved by GPU "
                         "auto-detection (or --timeline-gpu).")
    ap.add_argument("--timeline-gpu", default=_DEFAULT_GPU_SUBDIR,
                    choices=["5090", "A100"],
                    help=f"Which timelines/<gpu>/ subdirectory to read schedules "
                         f"from. Default: auto-detected ({_DEFAULT_GPU_SUBDIR}).")
    ap.add_argument("--loose", default=False, action="store_true")
    ap.add_argument("--tight", default=False, action="store_true")
    ap.add_argument("--nutanix", default=False, action="store_true",
                    help="Use timeline_nutanix.csv as the request schedule.")
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B")
    ap.add_argument("--lora_dir", default=str(SCRIPT_DIR / "adapters" / "llama3-toy-lora"))

    # small set of useful knobs
    ap.add_argument("--launcher", default="launch_llama3.py")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--rank_id", type=int, default=0)
    ap.add_argument("--co", action="store_true")  # enable finetuning mode
    ap.add_argument("--decode_graph", action="store_true", default=False)     # decode  CUDA graph
    ap.add_argument("--prefill_graph", action="store_true", default=False)    # prefill CUDA graph
    ap.add_argument("--bwd_graph", action="store_true", default=False)        # backward CUDA graph
    ap.add_argument("--graphs", action="store_true", default=False,
                    help="Shorthand for --decode_graph --prefill_graph "
                         "--bwd_graph (enables all three CUDA graphs).")
    ap.add_argument("--fold", type=int, default=1,
                    help="Subsample the timeline: send only every N-th request "
                         "(those whose 0-indexed row_id is divisible by N). "
                         "Default 1 (send all).")
    ap.add_argument("--track_occupancy", action="store_true", default=False,
                    help="Enable allocator page-occupancy sampling. Writes "
                         "output/occupancy<suffix>.csv (one row per second).")
    # ft_log_path and out_csv default to base names; final paths are composed
    # below under OUTPUT_DIR with a suffix encoding which graphs are enabled.
    ap.add_argument("--ft_log_path", type=str, default="bwd_log.csv")
    ap.add_argument("--out_csv", default="timeline_results.csv")

    # warmup config
    ap.add_argument("--warmup_count", type=int, default=1000)
    ap.add_argument("--warmup_duration_s", type=float, default=20.0)
    ap.add_argument("--warmup_rest_s", type=float, default=0.0)

    args = ap.parse_args()

    shape_flags = sum(1 for f in (args.tight, args.loose, args.nutanix) if f)
    if shape_flags > 1:
        ap.error("--tight, --loose, and --nutanix are mutually exclusive")
    if args.fold < 1:
        ap.error("--fold must be a positive integer")
    if args.graphs:
        args.decode_graph = True
        args.prefill_graph = True
        args.bwd_graph = True
    timelines_dir = SCRIPT_DIR / "timelines" / args.timeline_gpu
    if args.loose:
        args.timeline_csv = str(timelines_dir / "timeline_loose.csv")
    elif args.tight:
        args.timeline_csv = str(timelines_dir / "timeline_tight.csv")
    elif args.nutanix:
        args.timeline_csv = str(timelines_dir / "timeline_nutanix.csv")
    elif args.timeline_csv is None:
        args.timeline_csv = str(timelines_dir / "timeline_live.csv")

    timeline_rows = load_timeline_csv(args.timeline_csv)
    print(f"[orchestrator] Loaded {len(timeline_rows)} rows from {args.timeline_csv}", flush=True)
    if args.fold > 1:
        before = len(timeline_rows)
        timeline_rows = [r for r in timeline_rows if r.row_id % args.fold == 0]
        print(
            f"[orchestrator] --fold {args.fold}: kept {len(timeline_rows)}/{before} rows "
            f"(every {args.fold}-th request by row_id)",
            flush=True,
        )
    if timeline_rows:
        print(
            f"[orchestrator] Timeline range: {timeline_rows[0].timestamp_s:.6f}s -> {timeline_rows[-1].timestamp_s:.6f}s",
            flush=True,
        )

    server = f"http://127.0.0.1:{args.port}"

    # Output paths: everything goes under eval/llama3/output/, with a suffix
    # tag that encodes which CUDA graphs are enabled so multiple runs don't
    # overwrite each other. Order is fixed: decode < prefill < bwd.
    output_dir = SCRIPT_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    tags = []
    if args.decode_graph:
        tags.append("decode")
    if args.prefill_graph:
        tags.append("prefill")
    if args.bwd_graph:
        tags.append("bwd")
    # Schedule-shape tag.
    if args.tight:
        tags.append("tight")
    elif args.loose:
        tags.append("loose")
    elif args.nutanix:
        tags.append("nutanix")
   
    suffix = ("_" + "_".join(tags)) if tags else ""

    def _tagged(filename: str, prefer_arg_path: bool = False) -> str:
        """Stamp the suffix into filename.stem and anchor under output_dir,
        unless the caller passed an absolute/relative-with-dir path.
        """
        p = Path(filename)
        if prefer_arg_path and (p.is_absolute() or len(p.parts) > 1):
            # User gave an explicit path — keep it verbatim, just stamp the tag.
            return str(p.with_name(p.stem + suffix + p.suffix))
        return str(output_dir / (p.stem + suffix + p.suffix))

    args.out_csv = _tagged(args.out_csv, prefer_arg_path=True)
    args.ft_log_path = _tagged(args.ft_log_path, prefer_arg_path=True)
    occupancy_log = _tagged("occupancy.csv") if args.track_occupancy else None

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
        if args.timeline_gpu == "A100":
            a100_cfg = SCRIPT_DIR / "config" / "serving_config_finetuning_A100.yaml"
            cmd.append("--config")
            cmd.append(str(a100_cfg))
            print(f"[orchestrator] A100 detected — using {a100_cfg.name}", flush=True)
    if args.decode_graph:
        cmd.append("--enable-cuda-graph")
    if args.prefill_graph:
        cmd.append("--enable-prefill-cuda-graph")
    if args.bwd_graph:
        cmd.append("--enable-bwd-cuda-graph")
    if occupancy_log is not None:
        cmd.append("--occupancy_log")
        cmd.append(occupancy_log)

    cmd.append("--ft_log_path")
    cmd.append(args.ft_log_path)

    print("[orchestrator] launching:", " ".join(cmd), flush=True)
    print(f"[orchestrator] writing results CSV to: {args.out_csv}", flush=True)
    print(f"[orchestrator] writing bwd log CSV to: {args.ft_log_path}", flush=True)
    if occupancy_log is not None:
        print(f"[orchestrator] writing occupancy CSV to: {occupancy_log}", flush=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    p: Optional[subprocess.Popen] = None
    log_task: Optional[asyncio.Task] = None
    stop_event = asyncio.Event()

    try:
        # POSIX/Linux only.
        # Read raw bytes (no text=True) so '\r' from tqdm progress bars is
        # preserved instead of being silently rewritten to '\n' by Python's
        # universal-newline translation. The pumper decodes manually and
        # splits on '\r' (redraw) and '\n' (finalize) separately.
        p = subprocess.Popen(
            cmd,
            preexec_fn=os.setsid,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )

        async def pump_logs() -> None:
            """
            Forward child output to our stdout, prefixing each logical line
            with '[server]'. Splits on either '\\n' (terminate line) or
            '\\r' (in-place redraw), so tqdm progress bars from the child
            render as a single updating line on our terminal.
            """
            assert p is not None and p.stdout is not None
            loop = asyncio.get_running_loop()
            stream = p.stdout
            buf = []
            redrawing = False  # last emitted output was a CR-update (no newline yet)

            def read_chunk() -> bytes:
                return stream.read(256)

            while True:
                raw = await loop.run_in_executor(None, read_chunk)
                if not raw:
                    if buf:
                        _emit_line(("\r" if redrawing else "") + "[server] " + "".join(buf))
                    break
                chunk = raw.decode("utf-8", errors="replace")
                for ch in chunk:
                    if ch == "\n":
                        prefix = "\r" if redrawing else ""
                        _emit_line(prefix + "[server] " + "".join(buf))
                        buf.clear()
                        redrawing = False
                    elif ch == "\r":
                        # In-place redraw (e.g. server-side tqdm bar). When
                        # our timeline progress bar is active, fold these
                        # into normal lines so they don't fight the bottom
                        # bar; otherwise keep the redraw behavior.
                        if _active_pbar is not None:
                            _emit_line("[server] " + "".join(buf))
                        else:
                            sys.stdout.write("\r[server] " + "".join(buf))
                            sys.stdout.flush()
                        buf.clear()
                        redrawing = _active_pbar is None
                    else:
                        buf.append(ch)

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

        # Warmup row selection: take min of two caps starting from row 0:
        #   (1) warmup_count rows
        #   (2) all rows whose timestamp - first_ts <= warmup_duration_s
        # The warmup itself replays those rows on the timeline schedule
        # (with the first row normalized to WARMUP_START_OFFSET_S).
        if timeline_rows and args.warmup_count > 0 and args.warmup_duration_s > 0:
            base_ts = timeline_rows[0].timestamp_s
            by_count = min(args.warmup_count, len(timeline_rows))
            by_duration = 0
            for r in timeline_rows:
                if r.timestamp_s - base_ts <= args.warmup_duration_s:
                    by_duration += 1
                else:
                    break
            warmup_n = min(by_count, by_duration)
        else:
            warmup_n = 0
        warmup_rows = timeline_rows[:warmup_n]
        print(
            f"[orchestrator] Warmup row selection: count_cap={args.warmup_count}, "
            f"duration_cap={args.warmup_duration_s:.2f}s -> {len(warmup_rows)} rows",
            flush=True,
        )

        # Start finetuning BEFORE warmup. Warmup requests + the post-warmup
        # timeline both run with FT live in the background. The bwd_log
        # written by the server therefore covers the entire run; we trim
        # the warmup-window rows out after the benchmark using the
        # wall-clock anchor `T` recorded at the timeline start.
        if args.co:
            print("[orchestrator] Starting finetuning (pre-warmup)...", flush=True)
            async with aiohttp.ClientSession() as session:
                ok = await start_finetuning(session, server)
                if not ok:
                    print("[orchestrator] Failed to start finetuning", flush=True)
                    return
                print("[orchestrator] Finetuning started ✅", flush=True)

        if warmup_rows:
            await run_warmup_requests(
                server=server,
                base_model=args.base_model,
                lora_dir=args.lora_dir,
                warmup_rows=warmup_rows,
                stop_event=stop_event,
            )
            if stop_event.is_set():
                return

            if args.warmup_rest_s > 0:
                print(f"[orchestrator] Resting {args.warmup_rest_s:.2f}s after warmup...", flush=True)
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=args.warmup_rest_s)
                    return
                except asyncio.TimeoutError:
                    pass

        # Run full timeline (including warmup rows again). `t_first_wall`
        # is the wall-clock at the timeline's t0 — used below to trim
        # warmup-window rows out of the bwd_log.
        results, t_first_wall = await run_timeline_requests(
            server=server,
            base_model=args.base_model,
            lora_dir=args.lora_dir,
            timeline_rows=timeline_rows,
            stop_event=stop_event,
            normalize_start=True,
        )

        # Write CSV (scheduled phase only)
        write_results_csv(args.out_csv, results)

        # Exit finetuning after schedule. This also triggers the server to
        # flush `bwd_log.csv` (via finetuning_manager.write_bwd_logs_csv),
        # so the trim below sees a complete file.
        if args.co and not stop_event.is_set():
            print("[orchestrator] Exiting finetuning...", flush=True)
            async with aiohttp.ClientSession() as session:
                ok = await exit_finetuning(session, server)
                print("[orchestrator] Exited finetuning ✅" if ok else "[orchestrator] Failed to exit finetuning", flush=True)

        await asyncio.sleep(0.5)

        # Trim pre-warmup rows out of the bwd_log. The server records bwd
        # batches across the whole run (warmup + timeline); for the
        # benchmark we only want the timeline-phase rows.
        if args.co and t_first_wall is not None:
            trim_bwd_log_before(args.ft_log_path, t_first_wall)

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