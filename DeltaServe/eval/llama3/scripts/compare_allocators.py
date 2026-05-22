#!/usr/bin/env python3
"""
compare_allocators.py — verify that PackedKVMemoryAllocator produces
identical inference output to UnifiedMemoryAllocator on the same prompts.

Spawns the api_server twice (sequentially, to avoid GPU contention), once
per YAML, sends N greedy prompts (do_sample=False), and diffs the
generated text + token count side-by-side.

Finetuning is not triggered (no /start_finetuning call), so only the
inference-side KV path exercises the allocator change.

Usage:
    python eval/llama3/compare_allocators.py
    python eval/llama3/compare_allocators.py --max_new_tokens 64

Exit code: 0 if all outputs match, 1 if any diverge.
"""
import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
# Script lives under llama3/scripts/; config, adapters, data, etc. are
# one level up.
PROJECT_DIR = SCRIPT_DIR.parent
CONFIG_DIR = PROJECT_DIR / "config"
ADAPTERS_DIR = PROJECT_DIR / "adapters"

PROMPTS = [
    "Once upon a time in a quiet village, there lived",
    "The capital of France is",
    "In machine learning, gradient descent",
]


def resolve_paths(yaml_path: Path) -> dict:
    """Same logic as launch_llama3.py.resolve_paths — turns relative paths
    inside the YAML into absolutes anchored at eval/llama3/."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f) or {}
    ft = cfg.get("finetune", {}) or {}
    lora = cfg.get("lora", {}) or {}
    project_root = yaml_path.parent.parent  # eval/llama3/
    data_name = Path(ft.get("data_path") or "").name
    ft_lora_name = Path(ft.get("lora_path") or "").name
    adapter_dirs = []
    for d in (lora.get("adapter_dirs") or []):
        p = Path(d)
        if p.is_absolute():
            adapter_dirs.append(d)
        elif (project_root / p).exists():
            adapter_dirs.append(str(project_root / p))
        else:
            adapter_dirs.append(d)
    return {
        "ft_data_path": str(project_root / "data" / data_name),
        "ft_lora_path": str(project_root / "adapters" / ft_lora_name),
        "adapter_dirs": adapter_dirs,
    }


def _yaml_lit(v) -> str:
    return yaml.safe_dump(v, default_flow_style=True).strip()


def build_command(config_path: Path, port: int,
                  allocator: Optional[str] = None) -> List[str]:
    """`allocator` is the value to pass to memory.allocator
    ("unified" or "packed_kv"). None leaves the YAML's default in place."""
    abs_paths = resolve_paths(config_path)
    overrides = [
        f"finetune.data_path={abs_paths['ft_data_path']}",
        f"finetune.lora_path={abs_paths['ft_lora_path']}",
        f"lora.adapter_dirs={_yaml_lit(abs_paths['adapter_dirs'])}",
        # All CUDA graphs on so both runs exercise the same fast path.
        # The point of the harness is to verify allocator equivalence,
        # which has to hold under the production-realistic config — not
        # the eager fallback. Both runs see identical graph flags.
        "cuda_graph.enable_decode_cuda_graph=true",
        "cuda_graph.enable_prefill_cuda_graph=true",
        "cuda_graph.enable_bwd_cuda_graph=true",
    ]
    if allocator is not None:
        overrides.append(f"memory.allocator={allocator}")
    parts = [
        sys.executable, "-u", "-m", "dserve.server.api_server",
        "--config", str(config_path),
        "--port", str(port),
        "--rank_id", "0",
    ]
    for o in overrides:
        parts += ["--override", o]
    return parts


def make_payload(prompt: str, base_model: str, lora_dir: str,
                 max_new_tokens: int) -> Dict:
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


async def wait_for_server(server: str, base_model: str, lora_dir: str,
                          max_wait_s: float = 300.0) -> bool:
    """Wait until the server's /generate endpoint accepts a 2-token probe.
    /health alone returns 200 before model load completes."""
    t0 = time.monotonic()
    probe = make_payload("ping", base_model, lora_dir, max_new_tokens=2)
    async with aiohttp.ClientSession() as s:
        while time.monotonic() - t0 < max_wait_s:
            try:
                async with s.post(f"{server}/generate", json=probe,
                                  timeout=aiohttp.ClientTimeout(total=30)) as r:
                    if r.status == 200:
                        return True
            except Exception:
                pass
            await asyncio.sleep(2.0)
    return False


async def send_prompt(server: str, base_model: str, lora_dir: str,
                      prompt: str, max_new_tokens: int) -> Dict:
    payload = make_payload(prompt, base_model, lora_dir, max_new_tokens)
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        async with s.post(f"{server}/generate", json=payload) as r:
            r.raise_for_status()
            # api_server returns JSON but without a JSON Content-Type header,
            # so we have to bypass aiohttp's mimetype check.
            return await r.json(content_type=None)


def terminate_server(p: Optional[subprocess.Popen]) -> None:
    if p is None or p.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGINT)
    except Exception:
        pass
    t0 = time.monotonic()
    while time.monotonic() - t0 < 2.0:
        if p.poll() is not None:
            return
        time.sleep(0.05)
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception:
        pass


async def run_one(config_path: Path, port: int, base_model: str,
                  lora_dir: str, max_new_tokens: int,
                  show_server_log: bool,
                  allocator: Optional[str] = None) -> List[Dict]:
    server = f"http://127.0.0.1:{port}"
    cmd = build_command(config_path, port, allocator=allocator)
    tag_suffix = f" (allocator={allocator})" if allocator else ""
    print(f"\n[{config_path.name}{tag_suffix}] launching api_server on port {port} ...")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    p = subprocess.Popen(
        cmd, preexec_fn=os.setsid,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )

    tag = config_path.stem
    def log_pump():
        for line in iter(p.stdout.readline, ''):
            if show_server_log:
                print(f"  [{tag}] {line.rstrip()}")
    threading.Thread(target=log_pump, daemon=True).start()

    try:
        ok = await wait_for_server(server, base_model, lora_dir)
        if not ok:
            raise RuntimeError(
                f"server with {config_path.name} did not become ready"
            )
        print(f"[{config_path.name}] ready, sending {len(PROMPTS)} prompts concurrently")

        # Fan out all prompts at once so the server sees them as a real
        # multi-request batch (exercises batched prefill + decode through
        # both allocators). asyncio.gather preserves call order in the
        # returned list, so the response slice still aligns with PROMPTS.
        responses = await asyncio.gather(*(
            send_prompt(server, base_model, lora_dir, p, max_new_tokens)
            for p in PROMPTS
        ))
        for i, (prompt, resp) in enumerate(zip(PROMPTS, responses)):
            text = resp.get("generated_text", [""])[0]
            ntok = resp.get("count_output_tokens")
            print(f"  prompt {i}: {prompt!r}")
            print(f"  output  : {text!r} (tokens={ntok})")
        return list(responses)
    finally:
        print(f"[{config_path.name}] shutting down ...")
        terminate_server(p)
        # Give CUDA / NCCL ports a moment to release before the next launch.
        await asyncio.sleep(3.0)


def diff_responses(a: List[Dict], b: List[Dict]) -> bool:
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    all_match = True
    for i, (ra, rb) in enumerate(zip(a, b)):
        ta = ra.get("generated_text", [""])[0]
        tb = rb.get("generated_text", [""])[0]
        na = ra.get("count_output_tokens")
        nb = rb.get("count_output_tokens")
        match = (ta == tb) and (na == nb)
        all_match = all_match and match
        print(f"\nPrompt {i}: {PROMPTS[i]!r}")
        print(f"  unified   : {ta!r} (tokens={na})")
        print(f"  packed_kv : {tb!r} (tokens={nb})")
        print(f"  match     : {'YES' if match else 'NO'}")
        if not match:
            for j, (ca, cb) in enumerate(zip(ta, tb)):
                if ca != cb:
                    print(f"  first char divergence at index {j}: "
                          f"{ca!r} vs {cb!r}")
                    print(f"  ...unified prefix:   {ta[max(0,j-20):j+20]!r}")
                    print(f"  ...packed_kv prefix: {tb[max(0,j-20):j+20]!r}")
                    break
            else:
                if len(ta) != len(tb):
                    short, long_ = (ta, tb) if len(ta) < len(tb) else (tb, ta)
                    print(f"  length differs: {len(ta)} vs {len(tb)} "
                          f"(divergence after char {len(short)})")
    print("\n" + "=" * 70)
    print(f"OVERALL: {'IDENTICAL' if all_match else 'DIVERGED'}")
    print("=" * 70)
    return all_match


async def main():
    ap = argparse.ArgumentParser()
    # Single YAML now — the previous `serving_config_finetuning_packed.yaml`
    # was folded into the default `serving_config_finetuning.yaml`. Both
    # allocators are exercised by overriding `memory.allocator` at launch.
    ap.add_argument("--cfg",
                    default=str(CONFIG_DIR / "serving_config_finetuning.yaml"),
                    help="Server YAML to launch under. memory.allocator is "
                         "overridden per run.")
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B")
    ap.add_argument("--lora_dir",
                    default=str(ADAPTERS_DIR / "llama3-toy-lora"))
    ap.add_argument("--port_a", type=int, default=9101)
    ap.add_argument("--port_b", type=int, default=9102)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--quiet_server", action="store_true",
                    help="hide server stdout (default: stream it)")
    args = ap.parse_args()

    cfg = Path(args.cfg).resolve()
    if not cfg.exists():
        print(f"missing config: {cfg}", file=sys.stderr)
        sys.exit(2)

    show_log = not args.quiet_server

    print("=" * 70)
    print(f"Run A (unified):  {cfg}  --override memory.allocator=unified")
    print("=" * 70)
    resp_a = await run_one(cfg, args.port_a, args.base_model,
                           args.lora_dir, args.max_new_tokens, show_log,
                           allocator="unified")

    print("\n" + "=" * 70)
    print(f"Run B (packed_kv): {cfg}  --override memory.allocator=packed_kv")
    print("=" * 70)
    resp_b = await run_one(cfg, args.port_b, args.base_model,
                           args.lora_dir, args.max_new_tokens, show_log,
                           allocator="packed_kv")

    ok = diff_responses(resp_a, resp_b)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        sys.exit(130)
