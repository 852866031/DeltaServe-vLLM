#!/usr/bin/env python
"""Co-serving experiment harness (HTTP server + client).

1. Launches a real vLLM OpenAI server with finetuning enabled (config from YAML).
2. Every `--interval` seconds, fires one short completion (`--max-tokens`).
3. Repeats `--iterations` times, then shuts the server down.

The server's stdout is inherited, so the scheduler / coordinator / backward
decision prints stream live, e.g.:

  [ft-sched] inference batch -> injecting N FT samples ...
  [coord]    captured N FT tokens -> fill X/CAP ...
  [coord]    buffer FULL ... -> signal backward; FT admission CLOSED
  [backward] buffer-full signal ... cleaning + sleeping 2.0s ...
  [coord]    backward done ... -> FT admission OPEN
  [runner]   step mode: has_ft=False cudagraph_mode=...   (non-FT batch uses a graph)

Usage (inside the dserve-vllm conda env, with CUDA env from README.md):

    HF_HOME=/mnt/storage/huggingface HF_HUB_OFFLINE=1 \
    VLLM_USE_FLASHINFER_SAMPLER=0 python ft_experiment.py
    python ft_experiment.py --iterations 20 --interval 1.0 --max-tokens 10
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent  # repo root (this script lives in scripts/)
_DEFAULT_CONFIG = _ROOT / "configs" / "serving_config_finetuning_opt.yaml"
_SERVED_NAME = "opt125m"


def _engine_cli_args(engine_kwargs: dict) -> list[str]:
    """Translate EngineArgs kwargs to CLI flags (bool True -> --flag, else --k v)."""
    args: list[str] = []
    for key, value in engine_kwargs.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(flag)
        else:
            args += [flag, str(value)]
    return args


def _finetune_cli_args(finetune_section: dict) -> list[str]:
    """Translate the finetune YAML section to nested --finetune-config.k=v flags."""
    args: list[str] = []
    for key, value in finetune_section.items():
        if value is None:
            continue
        val = "true" if value is True else "false" if value is False else str(value)
        args.append(f"--finetune-config.{key}={val}")
    return args


def _wait_for_health(base_url: str, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(1.0)
    raise TimeoutError(f"server not healthy within {timeout}s")


def _completion(base_url: str, prompt: str, max_tokens: int) -> str:
    body = json.dumps({
        "model": _SERVED_NAME, "prompt": prompt,
        "max_tokens": max_tokens, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["choices"][0]["text"]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(_DEFAULT_CONFIG))
    ap.add_argument("--model", default="facebook/opt-125m")
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=10)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--startup-timeout", type=float, default=300.0)
    args = ap.parse_args()

    # Build the server command from the YAML (paths resolved to absolute).
    # Import here, after argparse, with the repo root removed from sys.path so
    # `import vllm` resolves to the installed package, not the ./dserve-vllm dir.
    _strip = {str(_HERE), str(_ROOT)}
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") not in _strip]
    from vllm.deltaserve.config_loader import load_yaml_config, split_config

    cfg = load_yaml_config(args.config)
    engine_kwargs, _, _ = split_config(cfg)
    engine_kwargs.pop("model", None)  # model is a positional arg to `dserve-vllm serve`

    # Use the `dserve-vllm` console script (`dserve-vllm serve <model>`), NOT
    # `python -m vllm.entrypoints.openai.api_server`: the latter triggers a
    # circular import (api_server -> arg_utils -> `from vllm import SamplingParams`
    # before vllm/__init__ finishes). The console script imports in the right
    # order, and its bin dir on sys.path[0] avoids any source-tree shadowing.
    vllm_bin = str(Path(sys.executable).parent / "dserve-vllm")
    cmd = [vllm_bin, "serve", args.model]
    cmd += _engine_cli_args(engine_kwargs)
    cmd += _finetune_cli_args(cfg.get("finetune") or {})
    cmd += _finetune_cli_args(cfg.get("debug") or {})
    cmd += _finetune_cli_args(cfg.get("slo") or {})
    cmd += ["--host", args.host, "--port", str(args.port),
            "--served-model-name", _SERVED_NAME]

    env = dict(os.environ)
    env.setdefault("HF_HOME", "/mnt/storage/huggingface")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    # `python -m ...` prepends the subprocess CWD to sys.path. Keep the path
    # clean of any source-tree shadowing of the installed `vllm` package by
    # running from a neutral CWD and disabling the unsafe-path prepend.
    env["PYTHONSAFEPATH"] = "1"
    server_cwd = tempfile.gettempdir()

    base_url = f"http://{args.host}:{args.port}"
    print(f"[experiment] launching server: {' '.join(cmd)}", flush=True)
    # New session => own process group, so we can kill the server + its
    # EngineCore/backward children together on shutdown.
    proc = subprocess.Popen(cmd, env=env, cwd=server_cwd, start_new_session=True)
    try:
        print(f"[experiment] waiting for {base_url}/health ...", flush=True)
        _wait_for_health(base_url, args.startup_timeout)
        print("[experiment] server ready; firing requests", flush=True)

        for i in range(args.iterations):
            print(f"\n[experiment] === request {i + 1}/{args.iterations} ===",
                  flush=True)
            try:
                text = _completion(base_url, args.prompt, args.max_tokens)
                print(f"[experiment] -> {text!r}", flush=True)
            except Exception as e:
                print(f"[experiment] request failed: {e}", flush=True)
            if i < args.iterations - 1:
                time.sleep(args.interval)
    finally:
        print("\n[experiment] shutting down server", flush=True)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            proc.wait(timeout=30)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        print("[experiment] done", flush=True)


if __name__ == "__main__":
    main()
