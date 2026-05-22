#!/usr/bin/env python
"""Train tiny LoRA adapter(s) on facebook/opt-125m (attention + FFN).

Produces PEFT-format adapters that mirror the llama3 toy adapters in adapters/
(r=16, alpha=32, dropout=0.05, bias=none, CAUSAL_LM) but with OPT-125m module
names and FFN included:

    attention : q_proj, k_proj, v_proj, out_proj
    FFN       : fc1, fc2

It does a short training run on a small inline corpus (no `datasets` dep) just
to move the weights off their init and show the loss dropping, then saves the
adapter + tokenizer so vLLM's multi-LoRA path can load it later.

Usage (inside the dserve-vllm conda env):

    HF_HOME=/mnt/storage/huggingface HF_HUB_OFFLINE=1 python train_opt125m_lora.py
    python train_opt125m_lora.py --which lora            # only the inference adapter
    python train_opt125m_lora.py --steps 100 --lr 2e-4   # train harder

Outputs (default):
    adapters/opt125m-toy-lora      (inference adapter)
    adapters/opt125m-toy-lora-ft   (finetuning-target adapter)
"""

import argparse
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO = Path(__file__).resolve().parent.parent  # repo root (script lives in scripts/)

# OPT-125m attention + FFN projection module names.
_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]

# Tiny toy corpus — repetitive/simple so a few steps visibly drop the loss.
_CORPUS = [
    "DeltaServe co-serves inference and finetuning on the same GPU.",
    "The backward pass yields the GPU at every layer boundary.",
    "vLLM provides paged KV cache and multi-LoRA batching.",
    "LoRA adapts attention and feed-forward projections with low-rank updates.",
    "The capital of France is Paris.",
    "The capital of Japan is Tokyo.",
    "Co-serving keeps the GPU saturated without breaking inference SLOs.",
    "A finetuning sample is injected into an ordinary inference batch.",
]


def train_one(base_model, out_dir, target_modules, steps, lr, r, alpha,
              dropout, device):
    print(f"\n=== training adapter -> {out_dir} ===")
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32)
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.to(device)
    model.train()

    enc = tok(_CORPUS, return_tensors="pt", padding=True, truncation=True,
              max_length=64)
    input_ids = enc.input_ids.to(device)
    attn = enc.attention_mask.to(device)
    labels = input_ids.clone()
    labels[attn == 0] = -100  # don't train on padding

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr)

    for step in range(steps):
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        loss = out.loss
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step == 0 or (step + 1) % max(1, steps // 10) == 0:
            print(f"  step {step + 1:>4}/{steps}  loss={loss.item():.4f}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"  saved adapter + tokenizer to {out_dir}")
    print(f"  target_modules = {target_modules}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default="facebook/opt-125m")
    ap.add_argument("--out-root", default=str(_REPO / "adapters"))
    ap.add_argument("--which", choices=["lora", "ft", "both"], default="both",
                    help="which adapter(s) to produce (default: both)")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"base_model = {args.base_model}")
    print(f"device     = {args.device}")
    print(f"lora       = r={args.r} alpha={args.alpha} dropout={args.dropout} "
          f"bias=none task=CAUSAL_LM")

    out_root = Path(args.out_root)
    targets = {
        "lora": out_root / "opt125m-toy-lora",
        "ft": out_root / "opt125m-toy-lora-ft",
    }
    selected = ["lora", "ft"] if args.which == "both" else [args.which]
    for name in selected:
        train_one(
            base_model=args.base_model,
            out_dir=targets[name],
            target_modules=_TARGET_MODULES,
            steps=args.steps,
            lr=args.lr,
            r=args.r,
            alpha=args.alpha,
            dropout=args.dropout,
            device=args.device,
        )


if __name__ == "__main__":
    main()
