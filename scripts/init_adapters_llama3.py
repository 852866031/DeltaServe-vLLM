#!/usr/bin/env python
"""init_adapters_llama3.py — train + save the toy Llama-3 LoRA adapters.

Port of DeltaServe/eval/llama3/init_adapters.py. Trains a tiny Q/K/V/O LoRA on
Llama-3-8B against the shared toy prompt set (``toy_adapters.py``), then saves
it to two adapter dirs under <repo>/adapters/:

    adapters/llama3-toy-lora       ← the inference adapter (served to /v1/...)
    adapters/llama3-toy-lora-ft    ← the finetuning target (identical copy at
                                     init time; the backward process trains
                                     this one further during co-serving)

The eval (eval/auto_benchmark.py) expects both directories to exist before it
launches the server.

Run (from repo root, inside the dserve-vllm conda env, with HF login for the
gated Meta-Llama-3-8B weights):

    huggingface-cli login                                     # one-time
    python scripts/init_adapters_llama3.py                    # single-GPU is fine
    accelerate launch --multi_gpu scripts/init_adapters_llama3.py   # 2+ GPUs

Optional flags:
    --out-dir DIR     parent dir for the two adapter folders (default:
                      <repo>/adapters)
    --epochs N        train epochs (default 2 — keeps wall time small)
    --skip-if-exists  do nothing if both target dirs already exist
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # scripts/
from toy_adapters import AdapterTarget, add_common_args, run  # noqa: E402

TARGET = AdapterTarget("meta-llama/Meta-Llama-3-8B", "llama3-toy-lora")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    run(TARGET, ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
