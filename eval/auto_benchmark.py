#!/usr/bin/env python3
"""auto_benchmark.py — timeline benchmark for DeltaServe-on-vLLM.

Port of DeltaServe/eval/llama3/auto_benchmark.py adapted to the vLLM OpenAI
server:

- Launches `dserve-vllm serve` (Llama-3-8B + the inference LoRA adapter,
  finetuning enabled when --co) in its own process group; streams server logs
  live.
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

Usage (dserve-vllm env, CUDA env per README.md):
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
# Map ``--scheduler`` → the serving YAML that selects that scheduler at
# engine init (via ``slo.coserving_admission_phase``). Keep both YAMLs in
# sync EXCEPT for the phase + decode_only_ft_safety_margin knobs.
_SCHED_CONFIGS = {
    "prefill": _ROOT / "configs" / "serving_config_finetuning_llama3.yaml",
    "both": _ROOT / "configs" / "serving_config_finetuning_llama3_both.yaml",
}
_CONFIG_DEFAULT = _SCHED_CONFIGS["prefill"]

# Per-GPU base model + HF cache root. Both boxes resolve via the HF cache
# machinery (offline), so we pass HF repo ids (not direct paths). HF looks
# for `<HF_HUB_CACHE>/models--<org>--<model>/...`; the default cache root
# is `$HF_HOME/hub/`, but if the box's `models--*` dirs live somewhere
# without that `hub/` intermediate (e.g. directly under scratch), set
# `hf_hub_cache` explicitly to skip the indirection.
_GPU_ENV = {
    "5090": {
        "model": "meta-llama/Meta-Llama-3-8B",
        "hf_home": "/mnt/storage/huggingface",
        # 5090: standard layout, models live under $HF_HOME/hub/, default
        # resolver behavior works without HF_HUB_CACHE.
        "hf_hub_cache": None,
    },
    "A100": {
        "model": "meta-llama/Meta-Llama-3.1-8B",
        "hf_home": "/home/jiaxuan_chen/scratch",
        # A100: models are directly under /home/jiaxuan_chen/scratch/
        # (no `hub/` intermediate), so override HF_HUB_CACHE to point
        # there directly.
        "hf_hub_cache": "/home/jiaxuan_chen/scratch",
    },
}
_BASE_MODEL_DEFAULT = _GPU_ENV["5090"]["model"]
_HF_HOME_DEFAULT = _GPU_ENV["5090"]["hf_home"]
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
# Server launch (dserve-vllm serve, from the YAML — mirrors ft_experiment_llama3.py)
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


def _load_yaml_cfg(config_path: Optional[str] = None) -> dict:
    """Load the finetuning YAML once, with the repo root stripped from
    ``sys.path`` so ``vllm`` resolves to the installed package (not any
    source-tree copy). Same trick used by ``build_server_cmd``; factored out
    so ``main()`` can read knobs (e.g. the FT admission factor) for the
    output-file suffix without re-implementing the path dance.

    ``config_path`` overrides the default
    ``configs/serving_config_finetuning_llama3.yaml`` — used by the
    ``--config`` CLI arg to A/B different scheduler variants."""
    sys.path[:] = [p for p in sys.path
                   if os.path.abspath(p or ".") not in {str(_HERE), str(_ROOT)}]
    from vllm.deltaserve.config_loader import load_yaml_config
    return load_yaml_config(str(config_path or _CONFIG_DEFAULT))


def _phase_tag(cfg: dict) -> str:
    """Render ``slo.coserving_admission_phase`` as a filesystem-safe suffix
    tag. Default ``"prefill"`` (today's behaviour); ``"both"`` selects the
    BothPhaseFinetuneScheduler that admits FT on any step composition.
    Unrecognized values are stringified verbatim — the scheduler-selection
    layer is authoritative on what's actually valid."""
    phase = ((cfg.get("slo") or {})
             .get("coserving_admission_phase"))
    if phase is None:
        phase = ((cfg.get("finetune") or {})
                 .get("coserving_admission_phase"))
    return str(phase or "prefill")


def _ft_factor_tag(cfg: dict) -> str:
    """Render ``slo.ft_tokens_admission_constrain_factor`` as a filesystem-
    safe suffix tag. ``-1`` (the disabled sentinel) becomes ``off`` so a
    no-cap run reads as ``..._factor_off`` instead of ``..._factor_-1.0``.
    Integer values render without a decimal; floats keep ``%g`` precision."""
    factor = ((cfg.get("slo") or {})
              .get("ft_tokens_admission_constrain_factor"))
    if factor is None:
        # Belt + suspenders: some YAMLs may put it under finetune instead.
        factor = ((cfg.get("finetune") or {})
                  .get("ft_tokens_admission_constrain_factor"))
    if factor is None:
        factor = -1.0
    f = float(factor)
    if f == -1.0:
        return "off"
    if f.is_integer():
        return str(int(f))
    return f"{f:g}"


def build_server_cmd(co: bool, bwd_log_path: Optional[str],
                     api_server_count: Optional[int] = None,
                     base_model: str = _BASE_MODEL_DEFAULT,
                     config_path: Optional[str] = None) -> list[str]:
    """Build the `dserve-vllm serve` command. Imports config_loader with the
    repo root stripped from sys.path so `vllm` resolves to the installed
    package, not any source-tree copy.

    ``config_path`` overrides the default YAML — passed through from
    ``--config`` so the server gets the right finetune/slo knobs."""
    cfg = _load_yaml_cfg(config_path)
    from vllm.deltaserve.config_loader import split_config
    engine_kwargs, _, _ = split_config(cfg)
    engine_kwargs.pop("model", None)  # positional to `dserve-vllm serve`

    vllm_bin = str(Path(sys.executable).parent / "dserve-vllm")
    cmd = [vllm_bin, "serve", base_model]
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
                          poll_s: float = 1.0,
                          stop: Optional[asyncio.Event] = None) -> None:
    t0 = time.monotonic()
    async with aiohttp.ClientSession() as session:
        while True:
            # Honor Ctrl+C: if the caller's stop event fires, bail
            # immediately instead of polling out to max_wait_s.
            if stop is not None and stop.is_set():
                print("[bench] interrupted while waiting for server health", flush=True)
                return
            try:
                async with session.get(f"{server}/health", timeout=2) as r:
                    if r.status == 200:
                        print(f"[bench] server up at {server}", flush=True)
                        return
            except Exception:
                pass
            if time.monotonic() - t0 > max_wait_s:
                raise TimeoutError(f"server not healthy within {max_wait_s}s")
            # Sleep cooperatively so a stop.set() wakes us within poll_s.
            if stop is not None:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=poll_s)
                    continue  # loop will see stop.is_set() and return
                except asyncio.TimeoutError:
                    pass
            else:
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
    ap.add_argument("--scheduler", choices=sorted(_SCHED_CONFIGS.keys()),
                    default="prefill",
                    help="Which co-serving scheduler to load. 'prefill' "
                         "(default) uses the original FT-rides-prefill "
                         "scheduler; 'both' uses BothPhaseFinetuneScheduler "
                         "(FT admits to any step composition under SLO). Maps "
                         "to the corresponding serving YAML "
                         "(serving_config_finetuning_llama3{,_both}.yaml). "
                         "The loaded phase is appended to the output suffix "
                         "as _phase_<phase>.")
    ap.add_argument("--model", default=None,
                    help="Base model id (HF) or local path. Default is "
                         "per-GPU: " + ", ".join(
                             f"{g}={v['model']}" for g, v in _GPU_ENV.items()
                         ) + ". Use this to override (e.g. for a new local "
                         "checkpoint).")
    ap.add_argument("--hf-home", default=None,
                    help="HF cache root (sets HF_HOME for the server). "
                         "Default is per-GPU: " + ", ".join(
                             f"{g}={v['hf_home']}" for g, v in _GPU_ENV.items()
                         ) + ". Respected only if HF_HOME is not already in "
                         "your env.")
    ap.add_argument("--hf-hub-cache", default=None,
                    help="Override HF_HUB_CACHE (the dir directly containing "
                         "`models--<org>--<model>/`). Defaults to the per-GPU "
                         "value (only set when the box's cache layout skips "
                         "the standard $HF_HOME/hub/ intermediate; A100 box "
                         "needs this).")
    ap.add_argument("--f", "-f", dest="log_to_file", action="store_true",
                    help="Capture server stdout/stderr to "
                         "eval/output/server<suffix>.log instead of streaming "
                         "to this terminal. Off by default (logs print live so "
                         "you can see startup banners and any crash trace).")
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

    # Resolve the serving YAML path from --scheduler. The actual phase tag
    # in the output suffix is read from the LOADED YAML's
    # ``slo.coserving_admission_phase`` (not just args.scheduler) so a typo
    # in the YAML doesn't silently disagree with the output file name.
    config_path = str(_SCHED_CONFIGS[args.scheduler])
    print(f"[bench] --scheduler={args.scheduler} → "
          f"{Path(config_path).relative_to(_ROOT)}", flush=True)

    # Output suffix: <co?>_factor_<f>_phase_<p>_<mode>. Tags (only on --co
    # runs) let A/B comparisons across ``ft_tokens_admission_constrain_factor``
    # values AND scheduler ``coserving_admission_phase`` land in distinct
    # files. ``-1`` (disabled) renders as ``off``; default phase is
    # ``prefill``. Mirrors the plotter's {base}_{mode} scheme
    # (base = '_co_factor_X_phase_Y' or '').
    base = "_co" if args.co else ""
    if args.co:
        cfg = _load_yaml_cfg(config_path)
        factor_tag = _ft_factor_tag(cfg)
        phase_tag = _phase_tag(cfg)
        base = f"{base}_factor_{factor_tag}_phase_{phase_tag}"
        print(f"[bench] ft_tokens_admission_constrain_factor tag: "
              f"_factor_{factor_tag} | "
              f"coserving_admission_phase tag: _phase_{phase_tag}",
              flush=True)
    suffix = base + (f"_{mode}" if mode else "")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = str(OUTPUT_DIR / f"timeline_results{suffix}.csv")
    bwd_log = str(OUTPUT_DIR / f"bwd_log{suffix}.csv") if args.co else None
    if bwd_log and os.path.exists(bwd_log):
        os.remove(bwd_log)  # fresh log per run

    # Resolve the base model + HF cache + HF hub cache: explicit CLI flags
    # win, else look up by detected GPU, else fall back to the 5090 defaults.
    gpu_env = _GPU_ENV.get(args.timeline_gpu, _GPU_ENV["5090"])
    base_model = args.model or gpu_env["model"]
    hf_home = args.hf_home or gpu_env["hf_home"]
    hf_hub_cache = args.hf_hub_cache or gpu_env.get("hf_hub_cache")
    model_src = "--model" if args.model else f"auto (gpu={args.timeline_gpu})"
    hf_src = "--hf-home" if args.hf_home else f"auto (gpu={args.timeline_gpu})"
    print(f"[bench] base model:    {base_model}  [{model_src}]", flush=True)
    print(f"[bench] HF_HOME:       {hf_home}  [{hf_src}]", flush=True)
    if hf_hub_cache:
        hc_src = "--hf-hub-cache" if args.hf_hub_cache else f"auto (gpu={args.timeline_gpu})"
        print(f"[bench] HF_HUB_CACHE:  {hf_hub_cache}  [{hc_src}]", flush=True)

    server = f"http://127.0.0.1:{_PORT}"
    cmd = build_server_cmd(args.co, bwd_log, args.api_server_count,
                           base_model=base_model, config_path=config_path)
    print("[bench] launching:", " ".join(cmd), flush=True)
    print(f"[bench] results -> {out_csv}"
          + (f" | bwd_log -> {bwd_log}" if bwd_log else ""), flush=True)

    env = dict(os.environ)
    env.setdefault("HF_HOME", hf_home)
    if hf_hub_cache:
        env.setdefault("HF_HUB_CACHE", hf_hub_cache)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    env["PYTHONSAFEPATH"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    # --f → capture all server stdout/stderr (scheduler / coordinator /
    # backward prints) to a file so the run can be inspected after the fact.
    # Otherwise inherit the parent's stdio so they print live to the
    # terminal. The file path is reported either way so the user knows where
    # to look (or that nothing was written).
    logf = None
    server_log: Optional[str] = None
    if args.log_to_file:
        server_log = str(OUTPUT_DIR / f"server{suffix}.log")
        logf = open(server_log, "w")
        popen_kwargs = dict(stdout=logf, stderr=subprocess.STDOUT)
        print(f"[bench] server log -> {server_log}  (tail -f to watch live)",
              flush=True)
    else:
        popen_kwargs = dict()  # inherit parent stdio → terminal
        print("[bench] server log -> terminal (pass --f to capture to file)",
              flush=True)
    proc = subprocess.Popen(cmd, env=env, cwd=tempfile.gettempdir(),
                            start_new_session=True, **popen_kwargs)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    # SIGINT handling: first Ctrl+C sets the stop event so loops can wind
    # down cleanly. A second Ctrl+C escalates to SIGKILL on the server's
    # process group + raises KeyboardInterrupt, so the user is never
    # trapped if a long await (e.g. wait_for_health during model load, or
    # a hung _run_rows row) isn't checking stop fast enough.
    _ctrlc = {"count": 0}

    def _on_sigint() -> None:
        _ctrlc["count"] += 1
        if _ctrlc["count"] == 1:
            print("\n[bench] Ctrl+C — shutting down "
                  "(press again to force kill)", flush=True)
            stop.set()
        else:
            print("\n[bench] Ctrl+C (force) — SIGKILLing server now",
                  flush=True)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
            # Restore default SIGINT so a third Ctrl+C cleanly kills us too.
            loop.remove_signal_handler(signal.SIGINT)
            raise KeyboardInterrupt

    loop.add_signal_handler(signal.SIGINT, _on_sigint)

    try:
        await wait_for_health(server, args.startup_timeout, stop=stop)
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

        # Persist the benchmark wall-clock t0 so the plotter can anchor the
        # finetune log (which has wall-clock timestamps) to the same origin
        # as the inference log (which is monotonic-relative to this same t0).
        # Without this the FT series anchors at the first backward row's
        # wall clock, which lags benchmark t0 by however long it took the
        # first backward to fire — shifts the FT curve left and makes
        # inference/FT peaks appear in-phase when they're actually antiphase.
        if t_first_wall is not None:
            import json as _json
            meta_path = str(OUTPUT_DIR / f"bench_meta{suffix}.json")
            with open(meta_path, "w") as f:
                _json.dump({"t_first_wall_iso": t_first_wall.isoformat()}, f)

        # Give the server a moment to flush any in-flight backward row, then
        # trim the bwd_log to the timeline window.
        if args.co and bwd_log and t_first_wall is not None:
            await asyncio.sleep(1.0)
            trim_bwd_log_before(bwd_log, t_first_wall)
    finally:
        stop.set()
        print("[bench] shutting down server", flush=True)
        terminate(proc)
        if logf is not None:
            try:
                logf.close()
            except Exception:
                pass
        if server_log is not None:
            print(f"[bench] done | server log: {server_log}", flush=True)
        else:
            print("[bench] done", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
