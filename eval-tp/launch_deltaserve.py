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

    python eval-tp/launch_deltaserve.py                       # llama3 preset (TP=2 YAML)
    python eval-tp/launch_deltaserve.py --family qwen3-14b    # Qwen3-14B TP=2 preset
    python eval-tp/launch_deltaserve.py --family qwen3-0.6b   # Qwen3-0.6B single-GPU smoke
    python eval-tp/launch_deltaserve.py --config <yaml>       # any serving YAML
    python eval-tp/launch_deltaserve.py --tp 2                # override tensor_parallel_size
    python eval-tp/launch_deltaserve.py --start-finetuning    # also POST /start_finetuning once healthy
    python eval-tp/launch_deltaserve.py --dry-run             # print the server cmd, don't launch

Leave it running and hit Ctrl-C to shut down (SIGINT is forwarded to the whole
server process group).
"""

import argparse
import os
from dataclasses import dataclass
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # eval-tp/
_ROOT = _HERE.parent                             # repo root (DeltaServe-vLLM/)
_HF_HOME_DEFAULT = "/mnt/storage/huggingface"

# Family presets: the serving YAML each family launches with by default. The
# YAML is the single source of truth for the base model (``model.model``) and
# the inference adapter (``adapters.lora_path_0``); the preset name doubles as
# the served model name and the output-file tag.
PRESETS: dict[str, Path] = {
    "llama3": _ROOT / "configs" / "serving_config_finetuning_llama3_tp2.yaml",
    "qwen3-14b": _ROOT / "configs" / "serving_config_finetuning_qwen3_14b_tp2.yaml",
    "qwen3-0.6b": _ROOT / "configs" / "serving_config_finetuning_qwen3_0.6b.yaml",
}
DEFAULT_FAMILY = "llama3"


@dataclass
class LaunchSpec:
    """Everything a driver needs about one resolved launch."""

    cmd: list[str]
    cfg: dict
    tp: int
    family: str
    served_name: str
    base_model: str
    infer_lora_name: str
    infer_lora_dir: str


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


def build_server_cmd(config_path: str, port: int, base_model: str | None = None,
                     tp_override: int | None = None, *,
                     family: str = DEFAULT_FAMILY,
                     served_name: str | None = None,
                     co: bool = True,
                     bwd_log_path: str | None = None,
                     step_trace_path: str | None = None,
                     api_server_count: int | None = None) -> LaunchSpec:
    """Build the `dserve-vllm serve` command from a DeltaServe YAML.

    Mirrors eval/auto_benchmark.build_server_cmd. The base model defaults to
    the YAML's ``model.model`` and the inference adapter to the YAML's
    ``adapters.lora_path_0`` (its directory name is the adapter's served name);
    ``served_name`` defaults to the family.

    ``co=False`` launches the inference-only baseline: the finetune / debug /
    slo sections are NOT passed, so ``enable_finetuning`` stays at its default
    (False) and no backward child is spawned. ``bwd_log_path`` (co only) is the
    backward-throughput CSV the server appends to; ``step_trace_path`` (co only)
    the per-step predicted-vs-actual trace (``finetune.step_trace_path``, see
    eval-tp/analyze_step_trace.py). ``api_server_count``
    overrides the YAML's ``server.api_server_count``."""
    _strip_repo_from_syspath()
    from vllm.deltaserve.config_loader import load_yaml_config, split_config

    cfg = load_yaml_config(config_path)
    engine_kwargs, _, extras = split_config(cfg)
    yaml_model = engine_kwargs.pop("model", None)  # positional to `dserve-vllm serve`
    base_model = base_model or yaml_model
    if not base_model:
        raise SystemExit(f"[launch-tp] no base model: pass --model or set "
                         f"model.model in {config_path}")
    if tp_override is not None:
        engine_kwargs["tensor_parallel_size"] = int(tp_override)
    tp = int(engine_kwargs.get("tensor_parallel_size", 1) or 1)

    infer_lora_dir = (extras.get("adapters") or {}).get("lora_path_0")
    if not infer_lora_dir:
        raise SystemExit(f"[launch-tp] {config_path} has no adapters.lora_path_0 "
                         "(the inference adapter to serve)")
    infer_lora_name = Path(infer_lora_dir).name
    served_name = served_name or family

    vllm_bin = str(Path(sys.executable).parent / "dserve-vllm")
    cmd = [vllm_bin, "serve", base_model]
    cmd += _engine_cli_args(engine_kwargs)
    cmd += ["--lora-modules", f"{infer_lora_name}={infer_lora_dir}"]
    if co:
        # Pass the finetune / debug / slo sections through as CLI flags.
        cmd += _finetune_cli_args(cfg.get("finetune") or {})
        cmd += _finetune_cli_args(cfg.get("debug") or {})
        cmd += _finetune_cli_args(cfg.get("slo") or {})
        if bwd_log_path:
            cmd.append(f"--finetune-config.bwd_log_path={bwd_log_path}")
        if step_trace_path:
            cmd.append(f"--finetune-config.step_trace_path={step_trace_path}")
    cmd += ["--host", "127.0.0.1", "--port", str(port),
            "--served-model-name", served_name]
    if api_server_count is None:
        api_server_count = int((cfg.get("server") or {}).get("api_server_count", 1) or 1)
    if api_server_count > 1:
        cmd += ["--api-server-count", str(api_server_count)]
    return LaunchSpec(cmd=cmd, cfg=cfg, tp=tp, family=family,
                      served_name=served_name, base_model=base_model,
                      infer_lora_name=infer_lora_name,
                      infer_lora_dir=str(infer_lora_dir))


def add_launch_args(ap: argparse.ArgumentParser) -> None:
    """The family / config / model / port / tp flags shared by the eval-tp drivers."""
    ap.add_argument("--family", choices=sorted(PRESETS), default=DEFAULT_FAMILY,
                    help="Model family preset: picks the default YAML, the served "
                         "model name and the output-file tag (default: %(default)s)")
    ap.add_argument("--config", default=None,
                    help="Serving YAML (default: the family preset's YAML)")
    ap.add_argument("--model", default=None,
                    help="Base model id/path (default: the YAML's model.model)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tp", type=int, default=None,
                    help="Override tensor_parallel_size from the YAML.")


def resolve_launch(args: argparse.Namespace, **kwargs) -> LaunchSpec:
    """``build_server_cmd`` from the shared CLI flags (+ builder kwargs such as
    ``co`` / ``bwd_log_path`` / ``api_server_count``)."""
    config = args.config or str(PRESETS[args.family])
    return build_server_cmd(config, args.port, args.model, args.tp,
                            family=args.family, **kwargs)


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
    add_launch_args(ap)
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

    spec = resolve_launch(args)
    cmd = spec.cmd

    print(f"[launch-tp] family          = {spec.family} (served as {spec.served_name})", flush=True)
    print(f"[launch-tp] config          = {args.config or PRESETS[args.family]}", flush=True)
    print(f"[launch-tp] base model      = {spec.base_model}", flush=True)
    print(f"[launch-tp] inference LoRA  = {spec.infer_lora_name} ({spec.infer_lora_dir})", flush=True)
    print(f"[launch-tp] tensor_parallel = {spec.tp}", flush=True)
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
