#!/usr/bin/env python
"""Shared helpers for the backward-service tests (``tests/test_<family>_*.py``).

Family-parametrized builders so every family's gradcheck / TP / overfit test is
a thin script over the same harness: synthetic per-layer weights in the exact
``lw`` layout the family layer functions consume, TP sharding that mirrors vLLM
+ ``lora_shard_slice``, autograd references, and a synthetic trainer service.

Import this BEFORE the test's ``sys.path`` scrub (it lives in ``tests/``):

    import bwd_harness as H
    sys.path[:] = [...]
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from vllm.deltaserve.bwd_services.common.ops import rmsnorm, rope_cos_sin
from vllm.deltaserve.bwd_services.common.tp import lora_shard_slice

LORA_KEYS = ("qA", "qB", "kA", "kB", "vA", "vB", "oA", "oB")


class Checker:
    """PASS/FAIL bookkeeping with the usual max-relative-error criterion."""

    def __init__(self, tol: float) -> None:
        self.tol = tol
        self.passed = 0
        self.failed = 0

    def check(self, name: str, got: torch.Tensor, ref: torch.Tensor,
              tol: float | None = None) -> bool:
        diff = (got.float() - ref.float()).abs().max().item()
        scale = ref.float().abs().max().item() + 1e-8
        rel = diff / scale
        ok = rel < (self.tol if tol is None else tol)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:34s} max-rel-err={rel:.3e}")
        self.passed += ok
        self.failed += (not ok)
        return ok

    def ok(self, name: str, cond: bool, detail: str = "") -> bool:
        print(f"  [{'PASS' if cond else 'FAIL'}] {name:34s} {detail}")
        self.passed += bool(cond)
        self.failed += (not cond)
        return cond

    def finish(self) -> None:
        print(f"\n{self.passed} passed, {self.failed} failed")
        sys.exit(1 if self.failed else 0)


# --------------------------------------------------------------------------- #
# Per-layer weights
# --------------------------------------------------------------------------- #

def make_layer_weights(family, hidden, Hq, Hkv, Hd, inter, r, *,
                       with_grad=False, dtype=torch.float32, device="cpu",
                       gen=None, scale=0.1):
    """Synthetic frozen base weights + LoRA factors in the family's ``lw``
    layout (the dict ``LoraSftTrainerService._layer_weights`` produces).
    ``hidden`` (residual width) may differ from ``Hq*Hd`` (attention width).
    Families whose ``layer_weights`` name extra frozen tensors (Qwen3's
    ``q_norm``/``k_norm``, ``[Hd]``) get them too, with weights ≠ 1 so the
    norm path is actually exercised."""
    q_size, kv_size = Hq * Hd, Hkv * Hd

    def rn(*shape):
        return torch.randn(*shape, generator=gen, device=device)

    def w(*shape):
        return (rn(*shape) * scale).to(dtype)

    def norm(n):
        return (rn(n).abs() + 0.5).to(dtype)

    def p(*shape):
        t = (rn(*shape) * scale).float()
        return torch.nn.Parameter(t) if with_grad else t

    lw = {
        "q": w(q_size, hidden), "k": w(kv_size, hidden), "v": w(kv_size, hidden),
        "o": w(hidden, q_size),
        "gate": w(inter, hidden), "up": w(inter, hidden), "down": w(hidden, inter),
        "in_ln": norm(hidden), "post_ln": norm(hidden),
        "qA": p(r, hidden), "qB": p(q_size, r),
        "kA": p(r, hidden), "kB": p(kv_size, r),
        "vA": p(r, hidden), "vB": p(kv_size, r),
        "oA": p(r, q_size), "oB": p(hidden, r),
    }
    for key in family.layer_weights:
        if key in ("q_norm", "k_norm"):
            lw[key] = norm(Hd)
    return lw


def lora_params(lw):
    return [lw[k] for k in LORA_KEYS]


def detach_weights(lw):
    return {k: (v.detach() if torch.is_tensor(v) else v) for k, v in lw.items()}


def shard_layer_weights(lw, rank, tp, Hq, Hkv, Hd, inter):
    """Rank ``rank``'s shard of a full ``lw`` (vLLM's TP layout + the
    production ``lora_shard_slice``): column-parallel q/k/v/gate/up on output
    rows, row-parallel o/down on input cols, norms replicated."""
    ql, kl, il = (Hq // tp) * Hd, (Hkv // tp) * Hd, inter // tp
    s = {}
    s["q"] = lw["q"][rank * ql:(rank + 1) * ql]
    s["k"] = lw["k"][rank * kl:(rank + 1) * kl]
    s["v"] = lw["v"][rank * kl:(rank + 1) * kl]
    s["gate"] = lw["gate"][rank * il:(rank + 1) * il]
    s["up"] = lw["up"][rank * il:(rank + 1) * il]
    s["o"] = lw["o"][:, rank * ql:(rank + 1) * ql]
    s["down"] = lw["down"][:, rank * il:(rank + 1) * il]
    for key in ("in_ln", "post_ln", "q_norm", "k_norm"):
        if key in lw:
            s[key] = lw[key]
    for proj in ("q", "k", "v", "o"):
        for ab in ("A", "B"):
            s[proj + ab] = lora_shard_slice(proj, ab, lw[proj + ab], rank, tp, ql, kl)
    return s


# --------------------------------------------------------------------------- #
# Batch layout + references
# --------------------------------------------------------------------------- #

def seq_layout(seq_lens, Hd, theta, device="cpu"):
    """``(b_start, cos, sin)`` for a packed batch of ``seq_lens``."""
    b_start, acc = [], 0
    for s in seq_lens:
        b_start.append(acc)
        acc += s
    positions = torch.cat([torch.arange(s, device=device) for s in seq_lens])
    cos, sin = rope_cos_sin(positions, Hd, theta)
    return b_start, cos, sin


def layer_output(cache, lw):
    """Finish a layer from its ``layer_forward`` cache: the frozen down-proj +
    residual the family functions deliberately skip."""
    return cache["resid_mid"] + F.linear(F.silu(cache["gate"]) * cache["up"], lw["down"])


def ref_layer_output(family, x, lw, scaling, cos, sin, seq_lens, b_start, dims, eps):
    """Autograd-differentiable layer output through the family's own
    ``layer_forward`` (functional, so autograd is the reference)."""
    cache = family.layer_forward(x, lw, scaling, cos, sin, seq_lens, b_start, dims, eps)
    return layer_output(cache, lw)


def ref_head_loss(final_in, lm_w, norm_w, eps, ids, seq_lens, b_start, vocab):
    """Autograd reference for ``head_backward``: per-sample shift-by-one CE
    summed, divided by the number of predicted tokens."""
    normed = rmsnorm(final_in, norm_w, eps)
    total, n_valid = final_in.new_zeros(()), 0
    for st, ln in zip(b_start, seq_lens):
        if ln < 2:
            continue
        logits = normed[st:st + ln - 1] @ lm_w[:vocab].t()
        total = total + F.cross_entropy(logits, ids[st + 1:st + ln].long(),
                                        reduction="sum")
        n_valid += ln - 1
    return total / n_valid, n_valid


# --------------------------------------------------------------------------- #
# Synthetic trainer service
# --------------------------------------------------------------------------- #

def build_synthetic_service(service_cls, *, hidden, n_layers, Hq, Hkv, Hd, inter,
                            vocab, r, lr, theta=10000.0, eps=1e-5, scaling=2.0,
                            tied_lm_head=False, seed=0):
    """A ``LoraSftTrainerService`` subclass built on tiny random weights in the
    exact HF/PEFT naming the worker shares, with ``_build_state`` run. Returns
    ``(svc, embed_w)``. ``tied_lm_head=True`` omits ``lm_head.weight`` and
    points ``lm_head_key`` at the embedding, as tied-embedding models do."""
    torch.manual_seed(seed)
    fam = service_cls.family
    q_size, kv = Hq * Hd, Hkv * Hd

    def w(*s):
        return torch.randn(*s) * (1.0 / (s[-1] ** 0.5))

    def norm(n):
        return torch.randn(n).abs() + 0.5

    embed = torch.randn(vocab, hidden) * 0.1
    base = {"model.embed_tokens.weight": embed, "model.norm.weight": norm(hidden)}
    if tied_lm_head:
        lm_head_key = "model.embed_tokens.weight"
    else:
        base["lm_head.weight"] = w(vocab, hidden)
        lm_head_key = "lm_head.weight"
    shapes = {
        "qkv": (q_size + 2 * kv, hidden), "o": (hidden, q_size),
        "gate_up": (2 * inter, hidden), "down": (hidden, inter),
    }
    for i in range(n_layers):
        p = f"model.layers.{i}."
        for key, rel in fam.layer_weights.items():
            if key in shapes:
                base[p + rel] = w(*shapes[key])
            elif key in ("in_ln", "post_ln"):
                base[p + rel] = norm(hidden)
            else:  # per-head norms
                base[p + rel] = norm(Hd)

    # PEFT-style LoRA init: A ~ small normal, B = 0 (initial delta = 0 → starts
    # as the frozen base model, so training can only help).
    ft = {}
    for i in range(n_layers):
        pre = f"base_model.model.model.layers.{i}.self_attn."
        for proj, out, inp in [("q", q_size, hidden), ("k", kv, hidden),
                               ("v", kv, hidden), ("o", hidden, q_size)]:
            ft[pre + f"{proj}_proj.lora_A.weight"] = torch.randn(r, inp) * 0.02
            ft[pre + f"{proj}_proj.lora_B.weight"] = torch.zeros(out, r)

    meta = dict(hidden_size=hidden, num_hidden_layers=n_layers,
                num_attention_heads=Hq, num_key_value_heads=Hkv, head_dim=Hd,
                intermediate_size=inter, rope_theta=theta, rms_norm_eps=eps,
                lora_scaling=scaling, vocab_size=vocab, learning_rate=lr,
                weight_decay=0.0, gamma=1.0, backward_fp32=True,
                lm_head_key=lm_head_key, norm_weight_key="model.norm.weight",
                embed_weight_key="model.embed_tokens.weight")
    svc = service_cls(0)
    svc.shared = {"base": base, "ft": ft, "meta": meta}
    svc._build_state()
    svc._built = True
    return svc, embed


def forward_capture(svc, embed_w, ids, seq_lens, b_start):
    """Full forward with the CURRENT master LoRA → the captured-activation dict
    ``process_backward`` consumes (layer_in per layer + final_in + final_hidden)."""
    positions = torch.cat([torch.arange(s) for s in seq_lens])
    cos, sin = rope_cos_sin(positions, svc.Hd, svc.theta)
    x = embed_w[ids]
    layer_in = []
    for i in range(svc.L):
        layer_in.append(x)
        lw = svc._layer_weights(i)
        cache = svc._layer_forward(x, lw, svc.scaling, cos, sin, seq_lens,
                                   b_start, svc.dims, svc.eps)
        x = layer_output(cache, lw)
    final_in = x
    final_hidden = rmsnorm(final_in, svc.norm_w, svc.eps)
    return {"layer_in": layer_in, "final_in": final_in,
            "final_hidden": final_hidden, "concat_input_ids": ids}
