#!/usr/bin/env python3
"""auto_benchmark.py — timeline benchmark for DeltaServe-on-vLLM.

Port of DeltaServe/eval/llama3/auto_benchmark.py adapted to the vLLM OpenAI
server:

- Launches `vllm serve` (Llama-3-8B + the inference LoRA adapter, finetuning
  enabled when --co) in its own process group; streams server logs live.
- Waits for /health, then runs a warmup phase (first N timeline rows, replayed
  on the timeline schedule, NOT recorded), an optional rest, then the full
  timeline (recorded).
- Each request is a STREAMING /v1/completions call: ttft_s = time from send to
  the first streamed chunk; latency_s = time to the last chunk; avg/worst tbt
  from inter-chunk gaps.
- Writes per-request metrics to output/timeline_results<suffix>.csv (same
  columns as the DeltaServe harness). When --co, the server writes a
  finetune-throughput log (bwd_log<suffix>.csv) which is trimmed to the
  timeline window afterwards.

Hardcoded (per the eval setup): base model Llama-3-8B, FT adapter
adapters/llama3-toy-lora-ft (via the finetuning YAML), inference adapter
adapters/llama3-toy-lora, port/rank matching ft_experiment_llama3.py.

Usage (dserve-vllm env, CUDA env per vllm_setup_5090.md):
    python eval/auto_benchmark.py --co --loose
    python eval/auto_benchmark.py --tight        # inference-only (no FT)
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
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import aiohttp

_HERE = Path(__file__).resolve().parent          # eval/
_ROOT = _HERE.parent                             # repo root
_CONFIG = _ROOT / "configs" / "serving_config_finetuning_llama3.yaml"
_BASE_MODEL = "meta-llama/Meta-Llama-3-8B"
_SERVED_NAME = "llama3"
_INFER_LORA_DIR = _ROOT / "adapters" / "llama3-toy-lora"
_INFER_LORA_NAME = "llama3-toy-lora"
_PORT = 8000          # matches ft_experiment_llama3.py
_RANK_ID = 0
OUTPUT_DIR = _HERE / "output"


# ----------------------------------------------------------------------
# GPU autodetect (which timelines/<gpu>/ subdir to read)
# ----------------------------------------------------------------------
def detect_gpu_subdir() -> str:
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


# ----------------------------------------------------------------------
# Server launch (vllm serve, from the YAML — mirrors ft_experiment_llama3.py)
# ----------------------------------------------------------------------
def _engine_cli_args(engine_kwargs: dict) -> list[str]:
    args: list[str] = []
    for key, value in engine_kwargs.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(flag)
        else:
            args += [flag, str(value)]
    return args


def _finetune_cli_args(section: dict) -> list[str]:
    args: list[str] = []
    for key, value in (section or {}).items():
        if value is None:
            continue
        val = "true" if value is True else "false" if value is False else str(value)
        args.append(f"--finetune-config.{key}={val}")
    return args


def build_server_cmd(co: bool, bwd_log_path: Optional[str],
                     api_server_count: Optional[int] = None) -> list[str]:
    """Build the `vllm serve` command. Imports config_loader with the repo root
    stripped from sys.path so `vllm` resolves to the installed package, not the
    ./vllm checkout."""
    sys.path[:] = [p for p in sys.path
                   if os.path.abspath(p or ".") not in {str(_HERE), str(_ROOT)}]
    from vllm.deltaserve.config_loader import load_yaml_config, split_config

    cfg = load_yaml_config(str(_CONFIG))
    engine_kwargs, _, _ = split_config(cfg)
    engine_kwargs.pop("model", None)  # positional to `vllm serve`

    vllm_bin = str(Path(sys.executable).parent / "vllm")
    cmd = [vllm_bin, "serve", _BASE_MODEL]
    cmd += _engine_cli_args(engine_kwargs)
    # Serve the inference LoRA adapter by name (requests target it).
    cmd += ["--lora-modules", f"{_INFER_LORA_NAME}={_INFER_LORA_DIR}"]
    if co:
        # Finetuning on: pass the finetune / slo / debug sections, plus the
        # bwd-throughput log path.
        cmd += _finetune_cli_args(cfg.get("finetune") or {})
        cmd += _finetune_cli_args(cfg.get("debug") or {})
        cmd += _finetune_cli_args(cfg.get("slo") or {})
        if bwd_log_path:
            cmd.append(f"--finetune-config.bwd_log_path={bwd_log_path}")
        # Hold FT admission closed at launch; the benchmark opens it via
        # POST /start_finetuning after warmup. (Profiling at launch is
        # unaffected — it injects FT directly.)
        cmd.append("--finetune-config.start_on_launch=false")
    # else: finetune sections omitted -> enable_finetuning defaults False.
    cmd += ["--host", "127.0.0.1", "--port", str(_PORT),
            "--served-model-name", _SERVED_NAME]
    # Shard the frontend (HTTP + tokenize + detok + output processing) across
    # N processes; the single EngineCore (and all DeltaServe state) is shared.
    # CLI --api-server-count overrides; otherwise read server.api_server_count
    # from the YAML (default 1).
    if api_server_count is None:
        api_server_count = int((cfg.get("server") or {}).get(
            "api_server_count", 1) or 1)
    if api_server_count > 1:
        cmd += ["--api-server-count", str(api_server_count)]
    return cmd


# ----------------------------------------------------------------------
# Request helpers
# ----------------------------------------------------------------------
def make_prompt_from_length(prompt_length: int) -> str:
    """Deterministic char-length prompt (matches the DeltaServe harness)."""
    base = "Instruction:\n"
    tail = "\n### Response: "
    filler_needed = max(0, prompt_length - len(base) - len(tail))
    filler = ("hello " * 10000)[:filler_needed]
    prompt = base + filler + tail
    if len(prompt) < prompt_length:
        prompt += "x" * (prompt_length - len(prompt))
    return prompt[:prompt_length]


async def start_finetuning(server: str) -> bool:
    """POST /start_finetuning to open FT admission (after warmup)."""
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(f"{server}/start_finetuning", timeout=10) as r:
                ok = r.status == 200
                print(f"[bench] start_finetuning -> HTTP {r.status}", flush=True)
                return ok
    except Exception as e:
        print(f"[bench] start_finetuning failed: {e}", flush=True)
        return False


async def wait_for_health(server: str, max_wait_s: float = 600.0,
                          poll_s: float = 1.0) -> None:
    t0 = time.monotonic()
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get(f"{server}/health", timeout=2) as r:
                    if r.status == 200:
                        print(f"[bench] server up at {server}", flush=True)
                        return
            except Exception:
                pass
            if time.monotonic() - t0 > max_wait_s:
                raise TimeoutError(f"server not healthy within {max_wait_s}s")
            await asyncio.sleep(poll_s)


async def send_one_request(
    session: aiohttp.ClientSession, server: str, idx: int, t_rel: float,
    prompt: str, max_new_tokens: int, model: str,
) -> Tuple[int, float, float, str, Optional[float], Optional[float], Optional[float]]:
    """Streaming /v1/completions. Returns
    (idx, t_rel, latency_s, status, ttft_s, avg_tbt_s, worst_tbt_s).

    ttft_s = send -> first streamed chunk; latency_s = send -> last chunk;
    tbt from inter-chunk gaps."""
    url = f"{server}/v1/completions"
    payload = {
        "model": model, "prompt": prompt, "max_tokens": max_new_tokens,
        "temperature": 0.0, "stream": True, "ignore_eos": True,
    }
    t_send = time.monotonic()
    ttft: Optional[float] = None
    chunk_times: List[float] = []
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                await resp.read()
                return (idx, t_rel, time.monotonic() - t_send,
                        f"error:http{resp.status}", None, None, None)
            async for raw in resp.content:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                now = time.monotonic()
                if ttft is None:
                    ttft = now - t_send
                chunk_times.append(now)
        latency = time.monotonic() - t_send
        avg_tbt = worst_tbt = None
        if len(chunk_times) >= 2:
            gaps = [chunk_times[i + 1] - chunk_times[i]
                    for i in range(len(chunk_times) - 1)]
            avg_tbt = sum(gaps) / len(gaps)
            worst_tbt = max(gaps)
        return (idx, t_rel, latency, "ok", ttft, avg_tbt, worst_tbt)
    except Exception as e:
        return (idx, t_rel, time.monotonic() - t_send,
                f"error:{type(e).__name__}", None, None, None)


# ----------------------------------------------------------------------
# Timeline
# ----------------------------------------------------------------------
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
            raise ValueError(f"Timeline CSV missing columns: {sorted(missing)}")
        for i, r in enumerate(reader):
            rows.append(TimelineRow(
                timestamp_s=float(r["timestamp_s"]),
                prompt_length=int(float(r["prompt_length"])),
                max_new_tokens=int(float(r["max_new_tokens"])),
                row_id=i))
    rows.sort(key=lambda x: x.timestamp_s)
    return rows


def write_results_csv(path: str, rows) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "t_rel_s", "latency_s", "status",
                    "ttft_s", "avg_tbt_s", "worst_tbt_s"])
        for idx, t_rel, latency, status, ttft, avg_tbt, worst_tbt in rows:
            w.writerow([idx, t_rel, latency, status, ttft, avg_tbt, worst_tbt])
    print(f"[bench] wrote results CSV: {path}", flush=True)


def trim_bwd_log_before(path: str, cutoff: datetime.datetime) -> None:
    """Drop bwd_log rows whose ISO-second `timestamp` is strictly before
    `cutoff` (the timeline-start wall clock), keeping only timeline-phase
    backward rows. Same-second rows are kept."""
    if not os.path.exists(path):
        print(f"[bench] no bwd_log to trim at {path}", flush=True)
        return
    cutoff_iso = cutoff.replace(microsecond=0).isoformat(timespec="seconds")
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        all_rows = list(reader)
    if not fieldnames or "timestamp" not in fieldnames:
        return
    kept = [r for r in all_rows if (r.get("timestamp") or "") >= cutoff_iso]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(kept)
    print(f"[bench] trimmed bwd_log {path}: kept {len(kept)}/{len(all_rows)} "
          f"rows (cutoff {cutoff_iso})", flush=True)


WARMUP_START_OFFSET_S = 1.0


async def _run_rows(server: str, rows: List[TimelineRow], stop: asyncio.Event,
                    model: str, record: bool, label: str,
                    request_timeout_s: float = 600.0):
    """Replay `rows` on their (normalized) timeline schedule. Returns
    (recorded result tuples, timeline-start wall clock)."""
    results = []
    if not rows:
        return results, None
    base_ts = min(r.timestamp_s for r in rows)
    offset = 0.0 if record else WARMUP_START_OFFSET_S
    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=request_timeout_s)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        t0 = time.monotonic()
        t_first_wall = datetime.datetime.now()
        total = max(r.timestamp_s for r in rows) - base_ts
        print(f"[bench] {label}: {len(rows)} requests, span {total:.1f}s", flush=True)

        async def _one(row: TimelineRow):
            target = t0 + (row.timestamp_s - base_ts) + offset
            delay = target - time.monotonic()
            if delay > 0:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=delay)
                    return
                except asyncio.TimeoutError:
                    pass
            if stop.is_set():
                return
            t_rel = time.monotonic() - t0
            prompt = make_prompt_from_length(row.prompt_length)
            res = await send_one_request(
                session, server, row.row_id, t_rel, prompt,
                row.max_new_tokens, model)
            if record:
                results.append(res)

        # Periodic progress: print the elapsed timeline time every 4s.
        async def _progress():
            while not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), timeout=4.0)
                    return
                except asyncio.TimeoutError:
                    elapsed = min(time.monotonic() - t0, total)
                    print(f"[bench] {label} t={elapsed:.0f}s / {total:.0f}s "
                          f"({len(results)} recorded)", flush=True)

        prog = asyncio.create_task(_progress())
        try:
            await asyncio.gather(*[asyncio.create_task(_one(r)) for r in rows])
        finally:
            prog.cancel()
            try:
                await prog
            except (asyncio.CancelledError, Exception):
                pass
        print(f"[bench] {label} done", flush=True)
        results.sort(key=lambda x: x[0])
        return results, t_first_wall


# ----------------------------------------------------------------------
# Process teardown
# ----------------------------------------------------------------------
def terminate(proc: subprocess.Popen) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=30)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


async def main() -> None:
    gpu_default = detect_gpu_subdir()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timeline_csv", default=None,
                    help="Schedule CSV. Default: timelines/<gpu>/timeline_live.csv")
    ap.add_argument("--timeline-gpu", default=gpu_default, choices=["5090", "A100"],
                    help=f"timelines/<gpu>/ subdir. Default auto-detected ({gpu_default}).")
    ap.add_argument("--loose", action="store_true")
    ap.add_argument("--tight", action="store_true")
    ap.add_argument("--nutanix", action="store_true")
    ap.add_argument("--co", action="store_true",
                    help="Enable finetuning (co-serving). Off => inference only.")
    ap.add_argument("--warmup_count", type=int, default=1000)
    ap.add_argument("--warmup_duration_s", type=float, default=10.0)
    ap.add_argument("--warmup_rest_s", type=float, default=0.0)
    ap.add_argument("--startup-timeout", type=float, default=600.0)
    ap.add_argument("--api-server-count", type=int, default=None,
                    help="Number of frontend API server processes (shared "
                         "single EngineCore). >1 shards output processing. "
                         "Overrides server.api_server_count in the YAML.")
    args = ap.parse_args()

    if sum(bool(x) for x in (args.loose, args.tight, args.nutanix)) > 1:
        ap.error("--loose, --tight, --nutanix are mutually exclusive")

    timelines_dir = _HERE / "timelines" / args.timeline_gpu
    mode: Optional[str] = None
    if args.loose:
        mode, args.timeline_csv = "loose", str(timelines_dir / "timeline_loose.csv")
    elif args.tight:
        mode, args.timeline_csv = "tight", str(timelines_dir / "timeline_tight.csv")
    elif args.nutanix:
        mode, args.timeline_csv = "nutanix", str(timelines_dir / "timeline_nutanix.csv")
    elif args.timeline_csv is None:
        args.timeline_csv = str(timelines_dir / "timeline_live.csv")

    timeline_rows = load_timeline_csv(args.timeline_csv)
    print(f"[bench] loaded {len(timeline_rows)} rows from {args.timeline_csv}", flush=True)

    # Output suffix: <co?>_<mode>. Mirrors the plotter's {base}_{mode} scheme
    # (base = '_co' or '').
    base = "_co" if args.co else ""
    suffix = base + (f"_{mode}" if mode else "")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = str(OUTPUT_DIR / f"timeline_results{suffix}.csv")
    bwd_log = str(OUTPUT_DIR / f"bwd_log{suffix}.csv") if args.co else None
    if bwd_log and os.path.exists(bwd_log):
        os.remove(bwd_log)  # fresh log per run

    server = f"http://127.0.0.1:{_PORT}"
    server_log = str(OUTPUT_DIR / f"server{suffix}.log")
    cmd = build_server_cmd(args.co, bwd_log, args.api_server_count)
    print("[bench] launching:", " ".join(cmd), flush=True)
    print(f"[bench] results -> {out_csv}"
          + (f" | bwd_log -> {bwd_log}" if bwd_log else ""), flush=True)
    print(f"[bench] server log -> {server_log}  (tail -f to watch live)", flush=True)

    env = dict(os.environ)
    env.setdefault("HF_HOME", "/mnt/storage/huggingface")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    env["PYTHONSAFEPATH"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    # Capture all server stdout/stderr (scheduler / coordinator / backward
    # prints) to a file so the run can be inspected after the fact.
    logf = open(server_log, "w")
    proc = subprocess.Popen(cmd, env=env, cwd=tempfile.gettempdir(),
                            start_new_session=True,
                            stdout=logf, stderr=subprocess.STDOUT)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, stop.set)

    try:
        await wait_for_health(server, args.startup_timeout)
        if stop.is_set():
            return

        # Warmup row selection: min(count cap, duration cap) from row 0.
        warmup_n = 0
        if timeline_rows and args.warmup_count > 0 and args.warmup_duration_s > 0:
            base_ts = timeline_rows[0].timestamp_s
            by_dur = sum(1 for r in timeline_rows
                         if r.timestamp_s - base_ts <= args.warmup_duration_s)
            warmup_n = min(args.warmup_count, by_dur, len(timeline_rows))
        warmup_rows = timeline_rows[:warmup_n]
        print(f"[bench] warmup rows: {len(warmup_rows)}", flush=True)

        model = _INFER_LORA_NAME  # inference requests target the inference LoRA

        if warmup_rows:
            await _run_rows(server, warmup_rows, stop, model,
                            record=False, label="warmup")
            if stop.is_set():
                return
            if args.warmup_rest_s > 0:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=args.warmup_rest_s)
                    return
                except asyncio.TimeoutError:
                    pass

        # Start finetuning AFTER warmup (server launched with FT admission held
        # via start_on_launch=false). FT then co-serves the recorded timeline.
        if args.co:
            await start_finetuning(server)

        results, t_first_wall = await _run_rows(
            server, timeline_rows, stop, model, record=True, label="timeline")
        write_results_csv(out_csv, results)

        # Give the server a moment to flush any in-flight backward row, then
        # trim the bwd_log to the timeline window.
        if args.co and bwd_log and t_first_wall is not None:
            await asyncio.sleep(1.0)
            trim_bwd_log_before(bwd_log, t_first_wall)
    finally:
        stop.set()
        print("[bench] shutting down server", flush=True)
        terminate(proc)
        try:
            logf.close()
        except Exception:
            pass
        print(f"[bench] done | server log: {server_log}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
