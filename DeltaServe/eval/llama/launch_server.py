"""
launch_server.py — drives dserve.server.api_server (llama1) via
serving_config.yaml.

User flags → YAML overrides:
    --enable-finetuning  -> finetune.enabled
    --port               -> server.port (passed through directly)
    --rank_id            -> server.rank_id (passed through directly)

Preserves the original wrapper's offline/online HF-cache toggling and the
MPS daemon check. When offline, swaps the YAML's HuggingFace adapter dirs
for their on-disk cache snapshots via --override.
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
SERVING_CONFIG_FT = SCRIPT_DIR / "config" / "serving_config_finetuning.yaml"
SERVING_CONFIG_NOFT = SCRIPT_DIR / "config" / "serving_config_no_finetuning.yaml"

# Knobs that don't (yet) have YAML homes.
ENABLE_GPU_PROFILE = False

OFFLINE = {
    "base_model": "/projects/I20240005/jchen/hf_cache/models--huggyllama--llama-7b/snapshots/llama-7b",
    "adapter_dirs": [
        "/projects/I20240005/jchen/hf_cache/hub/models--tloen--alpaca-lora-7b/snapshots/12103d6baae1b320aa60631b38acb6ea094a0539",
        "/projects/I20240005/jchen/hf_cache/hub/models--MBZUAI--bactrian-x-llama-7b-lora/snapshots/73e293a50ce88d19581f76502aa7baef42bc228b",
    ],
    "hf_cache_dir": "/projects/I20240005/jchen/hf_cache",
}


def internet_available(timeout: float = 2) -> bool:
    try:
        socket.gethostbyname("huggingface.co")
        requests.head("https://huggingface.co", timeout=timeout)
        return True
    except Exception:
        return False


def enable_offline_mode():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = OFFLINE["hf_cache_dir"]
    os.environ["TRANSFORMERS_CACHE"] = OFFLINE["hf_cache_dir"]
    print("🔌 No internet detected. Running in OFFLINE mode.")
    print("   → HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1")
    print(f"   → HF_HOME={OFFLINE['hf_cache_dir']}\n")


def enable_online_mode():
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.pop(var, None)
    print("🌐 Internet detected. Running in ONLINE mode.\n")


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
    Resolve the chosen YAML's paths to absolutes, anchored at eval/llama/.
    Adapter entries that don't exist as local paths (HuggingFace IDs like
    'tloen/alpaca-lora-7b') are passed through unchanged.
    """
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f) or {}
    ft = cfg.get("finetune", {}) or {}
    lora = cfg.get("lora", {}) or {}
    project_root = yaml_path.parent.parent  # eval/llama/
    data_name = Path(ft.get("data_path") or "").name
    lora_name = Path(ft.get("lora_path") or "").name
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
        "ft_data_path": str(project_root / "config" / data_name),
        "ft_lora_path": str(project_root / "finetuning_adapter") if lora_name == "finetuning_adapter"
                        else str(project_root / lora_name),
        "adapter_dirs": adapter_dirs,
    }


def _yaml_lit(v) -> str:
    """Inline-YAML-encode a Python value for use in --override KEY=VALUE."""
    return yaml.safe_dump(v, default_flow_style=True).strip()


if __name__ == "__main__":
    online = internet_available()
    if online:
        enable_online_mode()
    else:
        os.system("nvidia-cuda-mps-control -d")
        enable_offline_mode()

    if not is_mps_running():
        print("MPS control daemon is not running. Please start it with:")
        print("  sudo nvidia-cuda-mps-control -d")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-finetuning", action="store_true")
    parser.add_argument("--rank_id", type=int, default=0)
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    config_path = SERVING_CONFIG_FT if args.enable_finetuning else SERVING_CONFIG_NOFT
    abs_paths = resolve_paths(config_path)

    overrides = [
        f"finetune.data_path={abs_paths['ft_data_path']}",
        f"finetune.lora_path={abs_paths['ft_lora_path']}",
        f"lora.adapter_dirs={_yaml_lit(abs_paths['adapter_dirs'])}",
    ]
    # Offline mode swaps the YAML's HF model/adapter strings for their on-disk
    # cache snapshot paths.
    if not online:
        overrides.append(f"model.dir={OFFLINE['base_model']}")
        overrides.append(f"lora.adapter_dirs={_yaml_lit(OFFLINE['adapter_dirs'])}")

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
