#!/usr/bin/env python
"""Launch a `dserve-vllm serve` HTTP server from a DeltaServe YAML — the minimal
"just start the server" harness for the TP work (Phase 7).

Unlike scripts/launch_deltaserve.py (which runs the OFFLINE `LLM` API and does
one generation), this starts the real HTTP server the same way eval/auto_benchmark.py
does (`dserve-vllm serve …` built from the YAML), streams its logs to this
terminal, waits for /health, and then idles until Ctrl-C. It exists so we can
watch the process-launch prints under tensor parallelism:

    [deltaserve] [executor] spawning VllmWorker-0 (local_rank=0) as NON-daemon …
    [deltaserve] [executor] spawning VllmWorker-1 (local_rank=1) as NON-daemon …
    [deltaserve] [worker] rank=0 local_rank=0 tp_size=2: spawning backward SFT child on cuda:0 …
    [deltaserve] [worker] rank=1 local_rank=1 tp_size=2: spawning backward SFT child on cuda:1 …
    [deltaserve] [backward] child ready pid=… …
    [deltaserve] [worker] rank=0 …: backward child pid=… ready
    [deltaserve] [worker] rank=1 …: backward child pid=… ready

Usage (dserve-vllm conda env, CUDA env per README.md):

    python eval-tp/launch_deltaserve.py                       # TP=2 config (default)
    python eval-tp/launch_deltaserve.py --config <yaml>       # any serving YAML
    python eval-tp/launch_deltaserve.py --tp 2                # override tensor_parallel_size
    python eval-tp/launch_deltaserve.py --start-finetuning    # also POST /start_finetuning once healthy
    python eval-tp/launch_deltaserve.py --dry-run             # print the server cmd, don't launch

Leave it running and hit Ctrl-C to shut down (SIGINT is forwarded to the whole
server process group).
"""

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # eval-tp/
_ROOT = _HERE.parent                             # repo root (DeltaServe-vLLM/)
_DEFAULT_CONFIG = _ROOT / "configs" / "serving_config_finetuning_llama3_tp2.yaml"
_INFER_LORA_DIR = _ROOT / "adapters" / "llama3-toy-lora"
_INFER_LORA_NAME = "llama3-toy-lora"
_SERVED_NAME = "llama3"
_BASE_MODEL_DEFAULT = "meta-llama/Meta-Llama-3-8B"
_HF_HOME_DEFAULT = "/mnt/storage/huggingface"


def _strip_repo_from_syspath() -> None:
    """Drop the repo root + this dir from sys.path so `import vllm` resolves to
    the installed editable package, not any source-tree shadow. Same guard
    eval/auto_benchmark.py and scripts/launch_deltaserve.py use."""
    sys.path[:] = [p for p in sys.path
                   if os.path.abspath(p or ".") not in {str(_HERE), str(_ROOT)}]


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


def build_server_cmd(config_path: str, port: int, base_model: str,
                     tp_override: int | None) -> tuple[list[str], dict]:
    """Build the `dserve-vllm serve` command from a DeltaServe YAML.

    Mirrors eval/auto_benchmark.build_server_cmd but always co-serving (the YAML
    decides enable_finetuning) and single-frontend. Returns (cmd, cfg)."""
    _strip_repo_from_syspath()
    from vllm.deltaserve.config_loader import load_yaml_config, split_config

    cfg = load_yaml_config(config_path)
    engine_kwargs, _, _ = split_config(cfg)
    engine_kwargs.pop("model", None)  # positional to `dserve-vllm serve`
    if tp_override is not None:
        engine_kwargs["tensor_parallel_size"] = int(tp_override)

    vllm_bin = str(Path(sys.executable).parent / "dserve-vllm")
    cmd = [vllm_bin, "serve", base_model]
    cmd += _engine_cli_args(engine_kwargs)
    cmd += ["--lora-modules", f"{_INFER_LORA_NAME}={_INFER_LORA_DIR}"]
    # Pass the finetune / debug / slo sections through as CLI flags.
    cmd += _finetune_cli_args(cfg.get("finetune") or {})
    cmd += _finetune_cli_args(cfg.get("debug") or {})
    cmd += _finetune_cli_args(cfg.get("slo") or {})
    cmd += ["--host", "127.0.0.1", "--port", str(port),
            "--served-model-name", _SERVED_NAME]
    api_server_count = int((cfg.get("server") or {}).get("api_server_count", 1) or 1)
    if api_server_count > 1:
        cmd += ["--api-server-count", str(api_server_count)]
    return cmd, cfg


def wait_for_health(server: str, max_wait_s: float, proc: subprocess.Popen) -> bool:
    """Poll /health until the server answers or the process dies. Returns True on
    healthy, False if the server exited first / timed out."""
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"[launch-tp] server process exited early (code {proc.returncode}) "
                  f"before /health — see the log above", flush=True)
            return False
        try:
            with urllib.request.urlopen(f"{server}/health", timeout=2) as r:
                if r.status == 200:
                    print(f"[launch-tp] server healthy at {server}", flush=True)
                    return True
        except Exception:
            pass
        time.sleep(1.0)
    print(f"[launch-tp] server not healthy within {max_wait_s}s", flush=True)
    return False


def post_start_finetuning(server: str) -> None:
    try:
        req = urllib.request.Request(f"{server}/start_finetuning", method="POST",
                                     data=b"")
        with urllib.request.urlopen(req, timeout=10) as r:
            print(f"[launch-tp] POST /start_finetuning → {r.status}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[launch-tp] POST /start_finetuning failed: {e}", flush=True)


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


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(_DEFAULT_CONFIG),
                    help="Serving YAML (default: configs/serving_config_finetuning_llama3_tp2.yaml)")
    ap.add_argument("--model", default=_BASE_MODEL_DEFAULT,
                    help="Base model id/path (default: %(default)s)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tp", type=int, default=None,
                    help="Override tensor_parallel_size from the YAML.")
    ap.add_argument("--hf-home", default=None,
                    help=f"HF cache root (sets HF_HOME; default {_HF_HOME_DEFAULT}).")
    ap.add_argument("--startup-timeout", type=float, default=600.0)
    ap.add_argument("--start-finetuning", action="store_true",
                    help="POST /start_finetuning once healthy (opens FT admission). "
                         "Leave OFF for an M1 launch-only test — the backward math "
                         "is not shard-correct until M2/M3.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the resolved server command and exit (no launch).")
    args = ap.parse_args()

    cmd, cfg = build_server_cmd(args.config, args.port, args.model, args.tp)
    tp = args.tp if args.tp is not None else int(
        (cfg.get("parallel") or {}).get("tensor_parallel_size", 1) or 1)

    print(f"[launch-tp] config          = {args.config}", flush=True)
    print(f"[launch-tp] tensor_parallel = {tp}", flush=True)
    print(f"[launch-tp] server cmd      = {' '.join(cmd)}", flush=True)

    if args.dry_run:
        print("[launch-tp] --dry-run: not launching", flush=True)
        return 0

    hf_home = args.hf_home or os.environ.get("HF_HOME") or _HF_HOME_DEFAULT
    env = dict(os.environ)
    env.setdefault("HF_HOME", hf_home)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    env["PYTHONUNBUFFERED"] = "1"
    print(f"[launch-tp] HF_HOME         = {env['HF_HOME']}", flush=True)

    # start_new_session=True → own process group so Ctrl-C reaches every worker,
    # frontend, and backward child. cwd=tmp so nothing imports the repo's vllm/.
    proc = subprocess.Popen(cmd, env=env, cwd=tempfile.gettempdir(),
                            start_new_session=True)

    server = f"http://127.0.0.1:{args.port}"
    try:
        healthy = wait_for_health(server, args.startup_timeout, proc)
        if not healthy:
            terminate(proc)
            return 1
        if args.start_finetuning:
            post_start_finetuning(server)
        print("[launch-tp] server is up. Press Ctrl-C to shut down.", flush=True)
        # Idle until the server dies or we're interrupted.
        while proc.poll() is None:
            time.sleep(1.0)
        print(f"[launch-tp] server exited (code {proc.returncode})", flush=True)
    except KeyboardInterrupt:
        print("\n[launch-tp] Ctrl-C — shutting the server down…", flush=True)
    finally:
        terminate(proc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
