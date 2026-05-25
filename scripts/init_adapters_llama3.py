#!/usr/bin/env python
"""init_adapters_llama3.py — train + save the toy Llama-3 LoRA adapters.

Port of DeltaServe/eval/llama3/init_adapters.py. Trains a tiny Q/K/V/O LoRA on
Llama-3-8B against a hand-crafted prompt set, then saves it to two adapter
dirs under <repo>/adapters/:

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
import os
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # scripts/
_ROOT = _HERE.parent                              # repo root

MODEL_ID = "meta-llama/Meta-Llama-3-8B"

# Training knobs (kept tiny so it finishes quickly on one GPU)
MAX_LEN = 256
LR = 2e-4
PER_DEVICE_BS = 1
GRAD_ACCUM = 8
SAVE_STEPS = 200
LOG_STEPS = 10


def _build_toy_dataset():
    from datasets import Dataset

    texts = [
        "### Instruction:\nSay hello in one short sentence.\n### Response:\nHello! Nice to meet you.\n",
        "### Instruction:\nExplain what a GPU is in one sentence.\n### Response:\nA GPU is a processor specialized for fast parallel math, often used for graphics and ML.\n",
        "### Instruction:\nTranslate to French: 'Good morning'\n### Response:\nBonjour.\n",
        "### Instruction:\nList two prime numbers.\n### Response:\n2 and 3.\n",
        "### Instruction:\nWhat is 2+2?\n### Response:\n4.\n",
        "### Instruction:\nWrite a one-line definition of LoRA.\n### Response:\nLoRA fine-tunes a model by learning low-rank adapter matrices instead of updating all weights.\n",
    ]
    # Duplicate so Trainer gets enough steps even with small batch size.
    texts = texts * 20
    return Dataset.from_dict({"text": texts})


def _tokenize(ds, tokenizer, max_len):
    def _tok(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=max_len, padding="max_length",
        )

    return ds.map(_tok, batched=True, remove_columns=["text"])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(_ROOT / "adapters"),
                    help="Parent dir for adapter folders (default: <repo>/adapters)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--skip-if-exists", action="store_true",
                    help="No-op if both adapter dirs already exist (idempotent).")
    args = ap.parse_args()

    out_root = Path(args.out_dir).resolve()
    out_infer = out_root / "llama3-toy-lora"
    out_ft = out_root / "llama3-toy-lora-ft"

    if args.skip_if_exists and out_infer.exists() and out_ft.exists():
        print(f"[init_adapters] skip: both adapter dirs already exist under "
              f"{out_root}", flush=True)
        return

    # Heavy imports deferred so --help / --skip-if-exists don't touch torch.
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
        Trainer, TrainingArguments,
    )
    from peft import LoraConfig, TaskType, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        device_map="auto",
    )

    # LoRA on attention projections only — matches what the FT backward
    # service trains during co-serving.
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.enable_input_require_grads()

    ds = _build_toy_dataset()
    ds = _tokenize(ds, tokenizer, MAX_LEN)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    targs = TrainingArguments(
        output_dir=str(out_infer),
        per_device_train_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=args.epochs,
        learning_rate=LR,
        warmup_steps=10,
        logging_steps=LOG_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        report_to="none",
        bf16=use_bf16, fp16=False,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=collator)

    for d in (out_infer, out_ft):
        if d.exists():
            shutil.rmtree(d)

    print("\n>>> Starting training...\n", flush=True)
    trainer.train()

    out_infer.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_infer))
    tokenizer.save_pretrained(str(out_infer))
    print(f"Saved LoRA adapter to: {out_infer}")

    for entry in os.scandir(out_infer):
        if entry.is_dir() and entry.name.startswith("checkpoint-"):
            shutil.rmtree(entry.path)
    print(f"Deleted Trainer checkpoints from: {out_infer}")

    shutil.copytree(out_infer, out_ft)
    print(f"Copied adapter to: {out_ft}")


if __name__ == "__main__":
    sys.exit(main())
