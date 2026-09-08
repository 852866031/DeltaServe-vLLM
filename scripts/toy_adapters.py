"""toy_adapters.py — shared trainer for the toy LoRA adapter pairs.

Every model family DeltaServe-vLLM co-serves needs two sibling adapter dirs
under <repo>/adapters/:

    <name>       ← the inference adapter (served via vLLM's multi-LoRA path)
    <name>-ft    ← the finetuning target (identical copy at init time; the
                   backward process trains this one further during co-serving)

This module holds the model-agnostic part — toy dataset, PEFT config, the
Trainer run, and the save/copy step — so the per-family entry points
(``init_adapters_llama3.py``, ``init_adapters_qwen3.py``) only declare *which*
base model maps to *which* adapter name. It is not a CLI on its own.

Training is deliberately tiny (rank-16 Q/K/V/O LoRA, ~120 short samples,
2 epochs): the adapters only need to be well-formed and non-trivial, not good.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Models live under /mnt/storage on this box (see CLAUDE.md). Set before any
# HF import so the hub cache resolves there. Leave HF_HUB_OFFLINE to the
# caller: the first run for a new family has to fetch the base weights.
os.environ.setdefault("HF_HOME", "/mnt/storage/huggingface")

_ROOT = Path(__file__).resolve().parent.parent   # repo root

# LoRA on the attention projections only — the set the FT backward service
# trains during co-serving. Same names on Llama-3 and Qwen3.
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Training knobs (kept tiny so it finishes quickly on one or two GPUs).
MAX_LEN = 256
LR = 2e-4
PER_DEVICE_BS = 1
GRAD_ACCUM = 8
WARMUP_STEPS = 10
LOG_STEPS = 10
DATASET_REPEAT = 20   # duplicate the toy prompts so Trainer gets enough steps


@dataclass(frozen=True)
class AdapterTarget:
    """One (base model → adapter dir name) mapping.

    ``adapter_name`` is the inference adapter's directory name; the FT copy is
    ``adapter_name + "-ft"``.
    """

    model_id: str
    adapter_name: str

    @property
    def ft_name(self) -> str:
        return self.adapter_name + "-ft"


_TOY_TEXTS = [
    "### Instruction:\nSay hello in one short sentence.\n### Response:\nHello! Nice to meet you.\n",
    "### Instruction:\nExplain what a GPU is in one sentence.\n### Response:\nA GPU is a processor specialized for fast parallel math, often used for graphics and ML.\n",
    "### Instruction:\nTranslate to French: 'Good morning'\n### Response:\nBonjour.\n",
    "### Instruction:\nList two prime numbers.\n### Response:\n2 and 3.\n",
    "### Instruction:\nWhat is 2+2?\n### Response:\n4.\n",
    "### Instruction:\nWrite a one-line definition of LoRA.\n### Response:\nLoRA fine-tunes a model by learning low-rank adapter matrices instead of updating all weights.\n",
]


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """The CLI flags every per-family entry point shares."""
    parser.add_argument("--out-dir", default=str(_ROOT / "adapters"),
                        help="Parent dir for the adapter folders "
                             "(default: <repo>/adapters)")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Train epochs (default 2 — keeps wall time small)")
    parser.add_argument("--skip-if-exists", action="store_true",
                        help="No-op if both adapter dirs already exist.")


def run(target: AdapterTarget, args: argparse.Namespace) -> None:
    """Train ``target`` and save the inference + FT adapter pair."""
    out_root = Path(args.out_dir).resolve()
    out_infer = out_root / target.adapter_name
    out_ft = out_root / target.ft_name

    if args.skip_if_exists and out_infer.exists() and out_ft.exists():
        print(f"[init_adapters] skip: {out_infer.name} and {out_ft.name} "
              f"already exist under {out_root}", flush=True)
        return

    model, tokenizer = _load_peft_model(target.model_id)
    dataset = _ToyDataset(tokenizer)

    # Trainer's own output goes to a throwaway temp dir, so the adapter dirs
    # only ever hold the final save.
    with tempfile.TemporaryDirectory(prefix="toy_adapters_") as trainer_dir:
        _train(model, tokenizer, dataset, Path(trainer_dir), args.epochs)

    for d in (out_infer, out_ft):
        if d.exists():
            shutil.rmtree(d)
    out_infer.mkdir(parents=True)
    model.save_pretrained(str(out_infer))
    tokenizer.save_pretrained(str(out_infer))
    shutil.copytree(out_infer, out_ft)

    print(f"[init_adapters] saved inference adapter: {out_infer}")
    print(f"[init_adapters] saved finetuning target: {out_ft}")


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #

class _ToyDataset:
    """The toy prompts, tokenized to fixed length. A plain torch-style dataset
    (``__len__`` / ``__getitem__``) so we need no ``datasets`` dependency."""

    def __init__(self, tokenizer) -> None:
        texts = _TOY_TEXTS * DATASET_REPEAT
        enc = tokenizer(texts, truncation=True, max_length=MAX_LEN,
                        padding="max_length")
        self._rows = [
            {"input_ids": enc["input_ids"][i],
             "attention_mask": enc["attention_mask"][i]}
            for i in range(len(texts))
        ]

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, i: int) -> dict:
        return self._rows[i]


def _load_peft_model(model_id: str):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = torch.cuda.is_available()
    # device_map="auto" shards a model that does not fit one GPU across all
    # visible ones (naive pipeline) — enough for the 14B class on 2× 5090.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16 if use_bf16 else torch.float32,
        device_map="auto",
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none", target_modules=list(TARGET_MODULES),
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.enable_input_require_grads()   # needed with gradient checkpointing
    return model, tokenizer


def _train(model, tokenizer, dataset, trainer_dir: Path, epochs: int) -> None:
    import torch
    from transformers import (DataCollatorForLanguageModeling, Trainer,
                              TrainingArguments)

    targs = TrainingArguments(
        output_dir=str(trainer_dir),
        per_device_train_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=epochs,
        learning_rate=LR,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOG_STEPS,
        save_strategy="no",
        report_to="none",
        bf16=torch.cuda.is_available(), fp16=False,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(model=model, args=targs, train_dataset=dataset,
                      data_collator=collator)
    print("\n>>> Starting training...\n", flush=True)
    trainer.train()
