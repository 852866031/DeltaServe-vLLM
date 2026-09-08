#!/usr/bin/env python
"""init_adapters_qwen3.py — train + save the toy Qwen3 LoRA adapters.

Qwen3 counterpart of ``init_adapters_llama3.py``. Trains a tiny Q/K/V/O LoRA
on a Qwen3 base model against the shared toy prompt set (``toy_adapters.py``)
and saves it to two adapter dirs under <repo>/adapters/:

    adapters/qwen3-<size>-toy-lora       ← the inference adapter
    adapters/qwen3-<size>-toy-lora-ft    ← the finetuning target

Two sizes are wired in:

    14b    Qwen/Qwen3-14B-Base   the TP=2 co-serving target (~30 GB bf16,
                                 needs both 5090s: device_map="auto")
    0.6b   Qwen/Qwen3-0.6B-Base  same architecture (q/k-norm, GQA) in a
                                 single-GPU footprint — for smoke tests and
                                 the q/k-norm backward gradcheck

Run (from repo root, inside the dserve-vllm conda env; the first run for a
size needs hub access to fetch the base weights):

    python scripts/init_adapters_qwen3.py                 # 14b
    python scripts/init_adapters_qwen3.py --size 0.6b
    python scripts/init_adapters_qwen3.py --skip-if-exists

Optional flags: --out-dir, --epochs, --model-id (override the HF id while
keeping the size's adapter name).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # scripts/
from toy_adapters import AdapterTarget, add_common_args, run  # noqa: E402

TARGETS = {
    "14b": AdapterTarget("Qwen/Qwen3-14B-Base", "qwen3-14b-toy-lora"),
    "0.6b": AdapterTarget("Qwen/Qwen3-0.6B-Base", "qwen3-0.6b-toy-lora"),
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", choices=sorted(TARGETS), default="14b",
                    help="Which Qwen3 size to build adapters for (default 14b)")
    ap.add_argument("--model-id", default=None,
                    help="Override the base-model HF id for the chosen size")
    add_common_args(ap)
    args = ap.parse_args()

    target = TARGETS[args.size]
    if args.model_id:
        target = AdapterTarget(args.model_id, target.adapter_name)
    run(target, args)


if __name__ == "__main__":
    sys.exit(main())
