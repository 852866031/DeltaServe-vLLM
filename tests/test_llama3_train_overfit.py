#!/usr/bin/env python
"""End-to-end training proof for the llama3 manual LoRA backward (Phase 3).

The gradcheck (`test_llama3_backward.py`) proves the gradients are *correct at a
point*; this proves the full loop actually **learns**: the manual backward +
AdamW, run repeatedly on a fixed generated dataset where the forward uses the
*updated* master each step, must drive the loss down (overfit).

This is the real training path — it builds a `Llama3BackwardService` (tiny random
synthetic Llama) and calls `process_backward` (manual grads → clip → optimizer
step → master update) every iteration, recomputing the forward with the current
master to produce the captured activations. CPU, fp32, no GPU / no 8B.

    python tests/test_llama3_train_overfit.py
"""

import os
import sys

import torch

sys.path[:] = [p for p in sys.path
               if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]

from vllm.deltaserve.bwd_services import llama3 as L  # noqa: E402
from vllm.deltaserve.bwd_services.llama3 import Llama3BackwardService  # noqa: E402

# Small synthetic Llama with enough capacity to overfit a few fixed sequences.
D, NL, Hq, Hkv, Hd, INTER, VOCAB, R = 64, 2, 4, 2, 16, 128, 32, 8
THETA, EPS, SCALING = 10000.0, 1e-5, 2.0


def build_service(lr):
    torch.manual_seed(0)
    q_size, kv = Hq * Hd, Hkv * Hd

    def w(*s):
        return torch.randn(*s) * (1.0 / (s[-1] ** 0.5))

    base = {"model.embed_tokens.weight": torch.randn(VOCAB, D) * 0.1,
            "model.norm.weight": torch.randn(D).abs() + 0.5,
            "lm_head.weight": w(VOCAB, D)}
    for i in range(NL):
        p = f"model.layers.{i}."
        base[p + "self_attn.qkv_proj.weight"] = w(q_size + 2 * kv, D)
        base[p + "self_attn.o_proj.weight"] = w(D, D)
        base[p + "mlp.gate_up_proj.weight"] = w(2 * INTER, D)
        base[p + "mlp.down_proj.weight"] = w(D, INTER)
        base[p + "input_layernorm.weight"] = torch.randn(D).abs() + 0.5
        base[p + "post_attention_layernorm.weight"] = torch.randn(D).abs() + 0.5

    # PEFT-style LoRA init: A ~ small normal, B = 0 (initial delta = 0 → starts
    # as the frozen base model, so training can only help).
    ft = {}
    for i in range(NL):
        pre = f"base_model.model.model.layers.{i}.self_attn."
        for proj, out in [("q", q_size), ("k", kv), ("v", kv), ("o", D)]:
            ft[pre + f"{proj}_proj.lora_A.weight"] = torch.randn(R, D) * 0.02
            ft[pre + f"{proj}_proj.lora_B.weight"] = torch.zeros(out, R)

    meta = dict(hidden_size=D, num_hidden_layers=NL, num_attention_heads=Hq,
                num_key_value_heads=Hkv, head_dim=Hd, intermediate_size=INTER,
                rope_theta=THETA, rms_norm_eps=EPS, lora_scaling=SCALING,
                vocab_size=VOCAB, learning_rate=lr, weight_decay=0.0, gamma=1.0,
                backward_fp32=True, lm_head_key="lm_head.weight",
                norm_weight_key="model.norm.weight",
                embed_weight_key="model.embed_tokens.weight")
    svc = Llama3BackwardService(0)
    svc.shared = {"base": base, "ft": ft, "meta": meta}
    svc._build_state()
    svc._built = True
    return svc, base["model.embed_tokens.weight"]


def forward_capture(svc, embed_w, ids, seq_lens, b_start):
    """Full forward with the CURRENT master LoRA → the captured-activation dict
    that process_backward consumes (layer_in per layer + final_in + final_hidden)."""
    positions = torch.cat([torch.arange(s) for s in seq_lens])
    cos, sin = L.rope_cos_sin(positions, Hd, THETA)
    x = embed_w[ids]
    layer_in = []
    for i in range(NL):
        layer_in.append(x)
        lw = svc._layer_weights(i)
        x, _ = L.layer_forward(x, lw, SCALING, cos, sin, seq_lens, b_start,
                               svc.dims, EPS)
    final_in = x
    final_hidden = L.rmsnorm(final_in, svc.norm_w, EPS)
    return {"layer_in": layer_in, "final_in": final_in,
            "final_hidden": final_hidden, "concat_input_ids": ids}


def main():
    torch.manual_seed(1)
    # Fixed generated dataset: 3 short sequences (memorize → loss must drop).
    seq_lens = [5, 4, 6]
    n = sum(seq_lens)
    b_start, acc = [], 0
    for s in seq_lens:
        b_start.append(acc)
        acc += s
    ids = torch.randint(0, VOCAB, (n,))

    svc, embed_w = build_service(lr=5e-3)

    steps = 300
    losses = []
    for step in range(steps):
        acts = forward_capture(svc, embed_w, ids, seq_lens, b_start)
        loss, _ = svc.process_backward(acts, seq_lens, n, epoch=0)
        losses.append(loss)
        if step % 30 == 0 or step == steps - 1:
            print(f"  step {step:3d}  loss={loss:.4f}")

    initial = sum(losses[:3]) / 3
    final = sum(losses[-3:]) / 3
    drop = 1.0 - final / initial
    print(f"\n  initial≈{initial:.4f}  final≈{final:.4f}  drop={100 * drop:.1f}%")
    ok = final < 0.5 * initial and final < 1.0
    print(f"  [{'PASS' if ok else 'FAIL'}] training reduces loss (overfit) — "
          f"expect final < 0.5*initial and < 1.0")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
