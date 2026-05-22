#!/usr/bin/env python
"""Launch vLLM (offline) from a DeltaServe YAML config so you can see the
[deltaserve] startup prints (worker flag + backward process spawn).

Analogue of DeltaServe's eval/llama3/launch_llama3.py, but it just feeds our
YAML through vllm.deltaserve.config_loader into vLLM's offline LLM and runs one
generation. Prints to watch for during startup (green in a TTY):

    [deltaserve] Worker rank=0 ... enable_finetuning=True
    [deltaserve] [backward] spawning child with CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=10 ...
    [deltaserve] [backward] stub started pid=... inherited CUDA_MPS_..=10 ...
    [deltaserve] [backward] child ready pid=...

Usage (inside the dserve-vllm conda env, with the CUDA env from
vllm_setup_5090.md exported):

    HF_HUB_OFFLINE=1 python launch_deltaserve.py
    HF_HUB_OFFLINE=1 python launch_deltaserve.py --model facebook/opt-125m   # fast
    python launch_deltaserve.py --dry-run                                    # CPU-only, no model load
"""

import argparse
import os
import sys
from pathlib import Path

# This script lives at the project root, which contains the ./vllm repo dir.
# That dir would shadow the *installed* `vllm` package (the editable install
# adds the inner package via a .pth). Drop the project root from sys.path so
# `import vllm` resolves to the real package, not the repo folder.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)  # repo root (this script lives in scripts/)
sys.path[:] = [p for p in sys.path
               if os.path.abspath(p or ".") not in (_HERE, _ROOT)]

_DEFAULT_CONFIG = Path(_ROOT) / "configs" / "serving_config_finetuning_llama3.yaml"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(_DEFAULT_CONFIG),
                    help="DeltaServe YAML config (default: configs/serving_config_finetuning_llama3.yaml)")
    ap.add_argument("--model", default=None,
                    help="override model dir/id from the YAML (e.g. facebook/opt-125m)")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true",
                    help="parse the config and print the mapped args, then exit "
                         "(no model load, no GPU)")
    args = ap.parse_args()

    from vllm.deltaserve.config_loader import load_yaml_config, split_config

    cfg = load_yaml_config(args.config)
    engine_kwargs, finetune_config, extras = split_config(cfg)
    if args.model is not None:
        engine_kwargs["model"] = args.model

    print(f"[deltaserve-launch] config         = {args.config}")
    print(f"[deltaserve-launch] engine_kwargs  = {engine_kwargs}")
    print(f"[deltaserve-launch] finetune       = {finetune_config}")
    print(f"[deltaserve-launch] adapters       = {extras.get('adapters')}")
    print(f"[deltaserve-launch] server (offline: ignored) = {extras.get('server')}")

    if args.dry_run:
        print("[deltaserve-launch] --dry-run: stopping before model load")
        return

    import re

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(**engine_kwargs, finetune_config=finetune_config)

    # Build a LoRARequest per inference adapter listed under `adapters`
    # (lora_path_0, lora_path_1, ...) — just like handing adapter dirs to vLLM.
    # The FT adapter is NOT here (it's only shared with the backward process).
    adapters = extras.get("adapters") or {}
    lora_requests = []
    for name, path in sorted(adapters.items()):
        if not isinstance(path, str):
            continue
        m = re.search(r"(\d+)$", name)
        lora_id = int(m.group(1)) + 1 if m else len(lora_requests) + 1
        lora_requests.append(
            LoRARequest(lora_name=name, lora_int_id=lora_id, lora_path=path))

    # Serve this prompt through the first inference adapter (if any).
    lora_request = lora_requests[0] if lora_requests else None
    if lora_request is not None:
        print(f"[deltaserve-launch] {len(lora_requests)} inference adapter(s) "
              f"available; generating WITH '{lora_request.lora_name}' "
              f"(id={lora_request.lora_int_id}) {lora_request.lora_path}")
    else:
        print("[deltaserve-launch] generating without a LoRA adapter")

    out = llm.generate(
        [args.prompt],
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
        lora_request=lora_request,
    )
    print("=== generation ===")
    print(out[0].outputs[0].text)


if __name__ == "__main__":
    main()
