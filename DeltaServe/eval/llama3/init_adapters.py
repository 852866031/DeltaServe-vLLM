# train_llama3_qkvo_lora_ready.py
# Ready-to-run: trains a tiny LoRA adapter (Q/K/V/O only) on a toy dataset,
# prints LoRA parameter shapes, and saves the adapter.
#
# Run (2 GPUs):
#   pip install -U transformers datasets accelerate peft
#   huggingface-cli login   # required for Meta-Llama-3 weights
#   accelerate launch --multi_gpu train_llama3_qkvo_lora_ready.py
#
# Optional env vars (no prompts; safe defaults):
#   HF_HOME / TRANSFORMERS_CACHE as usual.

import os
import shutil
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
OUT_DIR = "adapters/llama3-toy-lora"
OUT_DIR_ft = "adapters/llama3-toy-lora-ft"

# Training knobs (kept tiny so it finishes quickly)
MAX_LEN = 256
EPOCHS = 2
LR = 2e-4
PER_DEVICE_BS = 1
GRAD_ACCUM = 8
SAVE_STEPS = 200
LOG_STEPS = 10


def build_toy_dataset() -> Dataset:
    texts = [
        "### Instruction:\nSay hello in one short sentence.\n### Response:\nHello! Nice to meet you.\n",
        "### Instruction:\nExplain what a GPU is in one sentence.\n### Response:\nA GPU is a processor specialized for fast parallel math, often used for graphics and ML.\n",
        "### Instruction:\nTranslate to French: 'Good morning'\n### Response:\nBonjour.\n",
        "### Instruction:\nList two prime numbers.\n### Response:\n2 and 3.\n",
        "### Instruction:\nWhat is 2+2?\n### Response:\n4.\n",
        "### Instruction:\nWrite a one-line definition of LoRA.\n### Response:\nLoRA fine-tunes a model by learning low-rank adapter matrices instead of updating all weights.\n",
    ]
    # Duplicate so Trainer gets enough steps even with small batch size
    texts = texts * 20
    return Dataset.from_dict({"text": texts})


def tokenize_dataset(ds: Dataset, tokenizer, max_len: int) -> Dataset:
    def _tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_len,
            padding="max_length",
        )

    return ds.map(_tok, batched=True, remove_columns=["text"])



def main():
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Base model
    use_bf16 = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        device_map="auto",
    )

    # LoRA ONLY on attention projections: q/k/v/o
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    # Print trainable parameters summary
    model.print_trainable_parameters()
    model.enable_input_require_grads()  # required for gradient checkpointing with PEFT


    # Data
    ds = build_toy_dataset()
    ds = tokenize_dataset(ds, tokenizer, MAX_LEN)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training args (Accelerate handles 2 GPUs when launched with --multi_gpu)
    args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        warmup_steps=10,
        logging_steps=LOG_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        report_to="none",
        bf16=use_bf16,
        fp16=False,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=data_collator,
    )

    # Clean up output dirs before training
    for d in (OUT_DIR, OUT_DIR_ft):
        if os.path.exists(d):
            shutil.rmtree(d)

    print("\n>>> Starting training...\n")
    trainer.train()

    # Save adapter only (PEFT)
    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"Saved LoRA adapter to: {OUT_DIR}")

    # Delete Trainer checkpoints (checkpoint-* subdirs inside OUT_DIR)
    for entry in os.scandir(OUT_DIR):
        if entry.is_dir() and entry.name.startswith("checkpoint-"):
            shutil.rmtree(entry.path)
    print(f"Deleted checkpoints from: {OUT_DIR}")

    # Copy everything in OUT_DIR to OUT_DIR_ft
    shutil.copytree(OUT_DIR, OUT_DIR_ft)
    print(f"Copied adapter to: {OUT_DIR_ft}")


if __name__ == "__main__":
    main()