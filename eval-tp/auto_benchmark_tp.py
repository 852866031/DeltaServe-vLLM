#!/usr/bin/env python3
"""auto_benchmark_tp.py — the timeline (real co-serving workload) benchmark
for tensor-parallel launches.

TP counterpart of eval/auto_benchmark.py. Same measurement: launch the server,
open FT admission, replay a request timeline (warmup rows unrecorded, then the
full timeline recorded) as streaming /v1/completions calls, write per-request
TTFT / latency / TBT to a results CSV, trim the backward-throughput log to the
timeline window, and persist the wall-clock t=0 anchor for the plotter.

What is TP-specific lives in eval-tp/launch_deltaserve.py (family presets,
YAML-derived base model + inference adapter, ``--tp`` override) and
ft_bench_tp.py (the stale-process pre-flight). The replay / CSV / trim
machinery is imported from eval/auto_benchmark.py — one implementation.

Timelines come from eval/timelines/5090/ (the board's request-rate envelope):

    --loose            timeline_loose.csv
    --tight            timeline_tight.csv
    --nutanix          timeline_nutanix.csv
    --nutanix-600-800  timeline_nutanix_original_600-800.csv (the 600-800 s slice
                       of the original Nutanix trace)
    --timeline_csv F   any CSV with timestamp_s,prompt_length,max_new_tokens

Usage (dserve-vllm env):

    python eval-tp/auto_benchmark_tp.py --family qwen3-14b --tp 2 --co --loose --kill-stale
    python eval-tp/auto_benchmark_tp.py --family llama3 --tp 2 --loose        # inference-only baseline
    python eval-tp/auto_benchmark_tp.py --family qwen3-14b --tp 2 --co --nutanix-600-800

Output (eval-tp/output/), tagged by family, TP size, co-serving knobs and mode:

    timeline_results_<family>_tp<N>[_co_factor_<f>_phase_<p>]_<mode>.csv
    bwd_log_<family>_tp<N>_co_factor_<f>_phase_<p>_<mode>.csv          (co only)
    bench_meta_<family>_tp<N>[...]_<mode>.json                          (t_first_wall anchor)
    server_<family>_tp<N>[...]_<mode>.log                               (both ranks' output)

``--publish`` drops the factor/phase tags (canonical ``_co_<mode>`` names, as
eval/auto_benchmark.py --publish does).
"""

import argparse
import asyncio
import datetime
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # eval-tp/
_ROOT = _HERE.parent
_EVAL = _ROOT / "eval"
sys.path.insert(0, str(_HERE))                   # launch_deltaserve, ft_bench_tp
sys.path.insert(0, str(_EVAL))                   # auto_benchmark (shared replay)

from auto_benchmark import (  # noqa: E402
    _ft_factor_tag,
    _phase_tag,
    _run_rows,
    load_timeline_csv,
    start_finetuning,
    trim_bwd_log_before,
    write_results_csv,
)
from ft_bench_tp import preflight_stale_check, wait_for_health  # noqa: E402
from launch_deltaserve import (  # noqa: E402
    _HF_HOME_DEFAULT,
    PRESETS,
    add_launch_args,
    resolve_launch,
    terminate,
)

OUTPUT_DIR = _HERE / "output"
TIMELINES_DIR = _EVAL / "timelines" / "5090"
MODES = {
    "loose": "timeline_loose.csv",
    "tight": "timeline_tight.csv",
    "nutanix": "timeline_nutanix.csv",
    "nutanix-600-800": "timeline_nutanix_original_600-800.csv",
}


def _load_cfg(path: str) -> dict:
    """The serving YAML (for the factor / phase output tags)."""
    sys.path[:] = [p for p in sys.path
                   if os.path.abspath(p or ".") not in {str(_HERE), str(_ROOT), str(_EVAL)}]
    from vllm.deltaserve.config_loader import load_yaml_config
    return load_yaml_config(path)


def _suffix(family: str, tp: int, co: bool, cfg: dict | None, publish: bool,
            mode: str | None) -> str:
    base = f"_{family}_tp{tp}"
    if co:
        base += "_co"
        if cfg is not None and not publish:
            base += f"_factor_{_ft_factor_tag(cfg)}_phase_{_phase_tag(cfg)}"
    return base + (f"_{mode}" if mode else "")


async def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_launch_args(ap)
    for m in MODES:
        ap.add_argument(f"--{m}", dest=f"mode_{m.replace('-', '_')}",
                        action="store_true", help=f"replay {MODES[m]}")
    ap.add_argument("--timeline_csv", default=None,
                    help="Explicit timeline CSV (overrides the mode flags).")
    ap.add_argument("--co", action="store_true",
                    help="Enable finetuning (co-serving). Off => inference-only baseline.")
    ap.add_argument("--warmup_count", type=int, default=1000)
    ap.add_argument("--warmup_duration_s", type=float, default=20.0)
    ap.add_argument("--warmup_rest_s", type=float, default=7.0)
    ap.add_argument("--startup-timeout", type=float, default=600.0)
    ap.add_argument("--api-server-count", type=int, default=None,
                    help="Frontend API server processes (overrides the YAML).")
    ap.add_argument("--hf-home", default=None)
    ap.add_argument("--publish", action="store_true",
                    help="Drop the _factor_<f>_phase_<p> tags (canonical names).")
    ap.add_argument("--real-timestamp", action="store_true",
                    help="Add a wall-clock 'timestamp' column to the results CSV.")
    ap.add_argument("--no-trim", action="store_true",
                    help="Keep every bwd_log row (skip the timeline-window trim).")
    ap.add_argument("--stream-log", action="store_true",
                    help="Stream the server's output to this terminal instead of "
                         "the per-run log file (both ranks print, so the file is "
                         "the default under TP).")
    ap.add_argument("--kill-stale", action="store_true",
                    help="Kill leftover GPU-resident processes from a previous run "
                         "instead of refusing to launch.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    chosen = [m for m in MODES if getattr(args, f"mode_{m.replace('-', '_')}")]
    if len(chosen) > 1:
        ap.error("pick at most one of " + ", ".join(f"--{m}" for m in MODES))
    mode = chosen[0] if chosen else None
    if args.timeline_csv is None:
        args.timeline_csv = str(TIMELINES_DIR / MODES[mode or "loose"])
        mode = mode or "loose"
    timeline_rows = load_timeline_csv(args.timeline_csv)

    # Resolve the launch once for the names, then rebuild with the bwd log.
    spec = resolve_launch(args, co=args.co, api_server_count=args.api_server_count)
    cfg_path = args.config or str(PRESETS[args.family])
    cfg = _load_cfg(cfg_path) if args.co else None
    suffix = _suffix(spec.family, spec.tp, args.co, cfg, args.publish, mode)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = str(OUTPUT_DIR / f"timeline_results{suffix}.csv")
    meta_path = str(OUTPUT_DIR / f"bench_meta{suffix}.json")
    server_log = str(OUTPUT_DIR / f"server{suffix}.log")
    bwd_log = str(OUTPUT_DIR / f"bwd_log{suffix}.csv") if args.co else None
    if bwd_log:
        open(bwd_log, "w").close()   # the server appends; start clean
        spec = resolve_launch(args, co=True, bwd_log_path=bwd_log,
                              api_server_count=args.api_server_count)
    cmd = spec.cmd

    print(f"[bench-tp] family          = {spec.family} (served as {spec.served_name})", flush=True)
    print(f"[bench-tp] config          = {cfg_path}", flush=True)
    print(f"[bench-tp] base model      = {spec.base_model}", flush=True)
    print(f"[bench-tp] inference LoRA  = {spec.infer_lora_name}", flush=True)
    print(f"[bench-tp] tensor_parallel = {spec.tp}", flush=True)
    print(f"[bench-tp] mode            = {mode or 'custom'}  ({len(timeline_rows)} rows "
          f"from {Path(args.timeline_csv).name})", flush=True)
    print(f"[bench-tp] co-serving      = {args.co}", flush=True)
    print(f"[bench-tp] results         = {out_csv}", flush=True)
    if bwd_log:
        print(f"[bench-tp] bwd log         = {bwd_log}", flush=True)
    print(f"[bench-tp] server log      = {'terminal' if args.stream_log else server_log}",
          flush=True)
    print(f"[bench-tp] server cmd      = {' '.join(cmd)}", flush=True)
    if args.dry_run:
        print("[bench-tp] --dry-run: not launching", flush=True)
        return 0
    if not preflight_stale_check(args.kill_stale):
        return 2

    env = dict(os.environ)
    env.setdefault("HF_HOME", args.hf_home or os.environ.get("HF_HOME") or _HF_HOME_DEFAULT)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    env["PYTHONSAFEPATH"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    logf = None
    popen_kwargs: dict = {}
    if not args.stream_log:
        logf = open(server_log, "w")
        popen_kwargs = dict(stdout=logf, stderr=subprocess.STDOUT)
    proc = subprocess.Popen(cmd, env=env, cwd="/tmp", start_new_session=True,
                            **popen_kwargs)
    server = f"http://127.0.0.1:{args.port}"
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    _ctrlc = {"count": 0}

    def _on_sigint() -> None:
        _ctrlc["count"] += 1
        if _ctrlc["count"] == 1:
            print("\n[bench-tp] Ctrl+C — shutting down (press again to force kill)",
                  flush=True)
            stop.set()
        else:
            print("\n[bench-tp] Ctrl+C (force) — SIGKILLing server now", flush=True)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
            loop.remove_signal_handler(signal.SIGINT)
            raise KeyboardInterrupt

    loop.add_signal_handler(signal.SIGINT, _on_sigint)

    rc = 1
    try:
        # ft_bench_tp's variant returns False as soon as the server process
        # exits, so a crashed launch fails fast instead of waiting out the
        # startup timeout.
        if not await wait_for_health(server, args.startup_timeout, proc):
            return 1
        if stop.is_set():
            return 1

        warmup_n = 0
        if timeline_rows and args.warmup_count > 0 and args.warmup_duration_s > 0:
            base_ts = timeline_rows[0].timestamp_s
            by_dur = sum(1 for r in timeline_rows
                         if r.timestamp_s - base_ts <= args.warmup_duration_s)
            warmup_n = min(args.warmup_count, by_dur, len(timeline_rows))
        warmup_rows = timeline_rows[:warmup_n]
        print(f"[bench-tp] warmup rows: {len(warmup_rows)}", flush=True)

        model = spec.infer_lora_name   # requests target the inference adapter
        if args.co:
            if not await start_finetuning(server):
                print("[bench-tp] could not open FT admission; aborting", flush=True)
                return 1
        if warmup_rows:
            await _run_rows(server, warmup_rows, stop, model, record=False, label="warmup")
            if stop.is_set():
                return 1
            if args.warmup_rest_s > 0:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=args.warmup_rest_s)
                    return 1
                except asyncio.TimeoutError:
                    pass

        results, t_first_wall = await _run_rows(
            server, timeline_rows, stop, model, record=True, label="timeline")
        write_results_csv(out_csv, results, t_first_wall=t_first_wall,
                          real_timestamp=args.real_timestamp)
        if t_first_wall is not None:
            with open(meta_path, "w") as f:
                json.dump({"t_first_wall_iso": t_first_wall.isoformat(),
                           "family": spec.family, "tp": spec.tp, "mode": mode,
                           "co": args.co, "timeline_csv": args.timeline_csv}, f)

        if args.co and bwd_log and t_first_wall is not None:
            await asyncio.sleep(1.0)
            if args.no_trim:
                print(f"[bench-tp] --no-trim: keeping bwd_log as-is ({bwd_log})", flush=True)
            else:
                base_ts = timeline_rows[0].timestamp_s if timeline_rows else 0.0
                cutoff = t_first_wall - datetime.timedelta(seconds=base_ts)
                trim_bwd_log_before(bwd_log, cutoff)

        ok = sum(1 for r in results if r[3] == "ok")
        ttfts = sorted(r[4] for r in results if r[4] is not None)
        print("=" * 62, flush=True)
        print(f"[bench-tp] {mode or 'custom'} | {spec.family} tp={spec.tp} "
              f"co={args.co}: {ok}/{len(results)} ok", flush=True)
        if ttfts:
            def pct(q):
                return ttfts[min(len(ttfts) - 1, int(q * len(ttfts)))]
            print(f"[bench-tp] TTFT p50={pct(0.5) * 1e3:.0f}ms "
                  f"p95={pct(0.95) * 1e3:.0f}ms p99={pct(0.99) * 1e3:.0f}ms", flush=True)
        print(f"[bench-tp] results -> {out_csv}", flush=True)
        print("=" * 62, flush=True)
        rc = 0
    finally:
        stop.set()
        print("[bench-tp] shutting down server", flush=True)
        terminate(proc)
        if logf is not None:
            logf.close()
            print(f"[bench-tp] server log: {server_log}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
