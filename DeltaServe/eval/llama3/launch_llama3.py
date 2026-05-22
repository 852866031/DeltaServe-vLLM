"""
launch_llama3.py — drives dserve.server.api_server via serving_config.yaml.

The server CLI is intentionally tiny (--config, --override, --port, --rank_id);
this wrapper just exposes the few knobs researchers vary per launch and
translates them into YAML overrides.

User flags → YAML overrides:
    --enable-finetuning      -> finetune.enabled
    --enable-cuda-graph      -> cuda_graph.enable_decode_cuda_graph
    --enable-bwd-cuda-graph  -> cuda_graph.enable_bwd_cuda_graph
    --ft_log_path            -> finetune.log_path
    --port                   -> server.port (passed through as a direct flag)
    --rank_id                -> server.rank_id (passed through as a direct flag)
"""
import argparse
import os
import shlex
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import requests
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
# Only two YAMLs exist now. The finetuning config bundles alpaca + packed_kv
# defaults; the no-finetuning config is its inference-only counterpart.
SERVING_CONFIG_FT = SCRIPT_DIR / "config" / "serving_config_finetuning.yaml"
SERVING_CONFIG_NOFT = SCRIPT_DIR / "config" / "serving_config_no_finetuning.yaml"

# Knobs that don't (yet) have YAML homes.
ENABLE_GPU_PROFILE = False


def internet_available(timeout: float = 2) -> bool:
    try:
        socket.gethostbyname("huggingface.co")
        requests.head("https://huggingface.co", timeout=timeout)
        return True
    except Exception:
        return False


def is_mps_running() -> bool:
    exe = shutil.which("nvidia-cuda-mps-control")
    if not exe:
        return False
    try:
        p = subprocess.Popen(
            [exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
        )
        p.communicate("get_server_list\nquit\n", timeout=2.0)
        return p.returncode == 0
    except Exception:
        return False


def resolve_paths(yaml_path: Path) -> dict:
    """
    Resolve the chosen YAML's relative paths to absolute ones, anchored at the
    YAML's parent-of-config directory. The api_server stores adapter paths in
    its lora_ranks dict using the exact strings it was given; benchmark
    clients call with absolute paths, so the YAML strings must be absolute
    too or the dict lookup misses (KeyError in mixed_req_queue).
    """
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f) or {}
    ft = cfg.get("finetune", {}) or {}
    lora = cfg.get("lora", {}) or {}
    project_root = yaml_path.parent.parent  # eval/llama3/
    data_name = Path(ft.get("data_path") or "").name
    ft_lora_name = Path(ft.get("lora_path") or "").name
    # Only resolve relative entries that actually exist as local paths under
    # project_root. HuggingFace IDs like "tloen/alpaca-lora-7b" are left as-is.
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


def _bool_lit(v: bool) -> str:
    return "true" if v else "false"


def _yaml_lit(v) -> str:
    """Inline-YAML-encode a value for use in --override KEY=VALUE."""
    return yaml.safe_dump(v, default_flow_style=True).strip()


if __name__ == "__main__":
    if not internet_available():
        print("⚠️  WARNING: Internet is not available. Exiting.")
        sys.exit(1)

    # if not is_mps_running():
    #     print("MPS control daemon is not running. Please start it with:")
    #     print("  sudo nvidia-cuda-mps-control -d")
    #     sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-finetuning", action="store_true")
    parser.add_argument("--enable-cuda-graph", action="store_true",
                        help="Enable CUDA graph capture for decode steps")
    parser.add_argument("--enable-prefill-cuda-graph", action="store_true",
                        help="Enable CUDA graph capture for prefill")
    parser.add_argument("--enable-bwd-cuda-graph", action="store_true",
                        help="Enable CUDA graph capture for backward steps")
    parser.add_argument("--rank_id", type=int, default=0)
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--ft_log_path", type=str,
                        default=str(SCRIPT_DIR / "bwd_log.csv"))
    parser.add_argument("--occupancy_log", type=str, default=None,
                        help="If set, the allocator samples (used pages / "
                             "total pages) once per second and writes a CSV "
                             "to this path. Override path becomes the value "
                             "of memory.unified_mem_manager_log_path.")
    parser.add_argument("--config", type=str, default=None,
                        help="Override the YAML config path. If omitted, the "
                             "default FT or no-FT config is picked from "
                             "--enable-finetuning.")
    args = parser.parse_args()

    if args.config:
        config_path = Path(args.config)
    else:
        config_path = SERVING_CONFIG_FT if args.enable_finetuning else SERVING_CONFIG_NOFT
    abs_paths = resolve_paths(config_path)

    overrides = [
        f"finetune.log_path={args.ft_log_path}",
        f"finetune.data_path={abs_paths['ft_data_path']}",
        f"finetune.lora_path={abs_paths['ft_lora_path']}",
        f"lora.adapter_dirs={_yaml_lit(abs_paths['adapter_dirs'])}",
        f"cuda_graph.enable_decode_cuda_graph={_bool_lit(args.enable_cuda_graph)}",
        f"cuda_graph.enable_prefill_cuda_graph={_bool_lit(args.enable_prefill_cuda_graph)}",
        f"cuda_graph.enable_bwd_cuda_graph={_bool_lit(args.enable_bwd_cuda_graph)}",
    ]
    if args.occupancy_log:
        overrides.append(
            f"memory.unified_mem_manager_log_path={args.occupancy_log}"
        )

    parts = ["python", "-m", "dserve.server.api_server",
             "--config", str(config_path),
             "--port", str(args.port),
             "--rank_id", str(args.rank_id)]
    for o in overrides:
        parts += ["--override", o]

    cmd = " ".join(shlex.quote(p) for p in parts)
    if ENABLE_GPU_PROFILE:
        cmd = ("nsys profile --cuda-memory-usage=true "
               "--trace-fork-before-exec=true --force-overwrite true "
               "-o trace " + cmd)

    print(cmd)
    os.system(cmd)
