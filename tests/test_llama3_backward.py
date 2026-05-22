#!/usr/bin/env python
"""Correctness gate for the llama3 manual LoRA backward (Phase 3).

Validates the hand-derived gradients in `vllm.deltaserve.bwd_services.llama3`
against `torch.autograd` on tiny synthetic fp32 shapes (CPU, no GPU / no 8B):

  - head_backward      vs autograd of the LM-head + final-norm + per-sample CE loss
  - layer_backward     vs autograd of one full decoder layer's forward
                       (RMSNorm, q/k/v base+LoRA, RoPE, GQA causal attn, o base+LoRA,
                        SwiGLU MLP, residuals) w.r.t. the input + all 8 LoRA tensors

Both paths differentiate the SAME fp32 forward, so a correct manual backward matches
autograd to ~fp32 precision; a math bug shows up as O(1) relative error.

    python tests/test_llama3_backward.py
"""

import os
import sys

import torch

sys.path[:] = [p for p in sys.path
               if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]

from vllm.deltaserve.bwd_services import llama3 as L  # noqa: E402

_TOL = 1e-3
_passed = 0
_failed = 0


def _check(name, manual, ref):
    global _passed, _failed
    rel = (manual - ref).abs().max().item() / (ref.abs().max().item() + 1e-8)
    ok = rel < _TOL
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:18s} max-rel-err={rel:.3e}")
    _passed += ok
    _failed += (not ok)


def _ref_head_loss(final_in, lm_w, norm_w, eps, ids, seq_lens, b_start, vocab):
    """Autograd-friendly reference for head_backward's loss (per-sample shift CE)."""
    normed = L.rmsnorm(final_in, norm_w, eps)
    total = final_in.new_zeros(())
    n_valid = 0
    for st, ln in zip(b_start, seq_lens):
        if ln < 2:
            continue
        lg = normed[st:st + ln - 1] @ lm_w[:vocab].t()
        tgt = ids[st + 1:st + ln].long()
        total = total + torch.nn.functional.cross_entropy(lg, tgt, reduction="sum")
        n_valid += int(ln - 1)
    return total / max(n_valid, 1)


def test_head():
    print("test_head (LM head + final norm + CE):")
    torch.manual_seed(0)
    D, vocab = 8, 10
    seq_lens, b_start = [3, 2], [0, 3]
    n = sum(seq_lens)
    eps = 1e-5
    final_in = torch.randn(n, D, dtype=torch.float32, requires_grad=True)
    lm_w = torch.randn(vocab, D)
    norm_w = torch.randn(D).abs() + 0.5
    ids = torch.randint(0, vocab, (n,))

    loss_ref = _ref_head_loss(final_in, lm_w, norm_w, eps, ids, seq_lens, b_start, vocab)
    grad_ref = torch.autograd.grad(loss_ref, final_in)[0]

    loss_m, n_valid, grad_m = L.head_backward(
        final_in.detach(), lm_w, norm_w, eps, ids, seq_lens, b_start, vocab)

    print(f"  loss manual={loss_m:.6f} ref={loss_ref.item():.6f} "
          f"(n_valid={n_valid}, expect {sum(l - 1 for l in seq_lens)})")
    assert abs(loss_m - loss_ref.item()) < 1e-4, "loss mismatch"
    _check("grad_final_in", grad_m, grad_ref)


def _make_layer_weights(D, kv_size, inter, r, with_grad):
    def w(*shape):
        return torch.randn(*shape) * 0.1
    def p(*shape):
        t = torch.randn(*shape) * 0.1
        return torch.nn.Parameter(t, requires_grad=True) if with_grad else t
    lw = {
        "q": w(D, D), "k": w(kv_size, D), "v": w(kv_size, D), "o": w(D, D),
        "gate": w(inter, D), "up": w(inter, D), "down": w(D, inter),
        "in_ln": torch.randn(D).abs() + 0.5,
        "post_ln": torch.randn(D).abs() + 0.5,
        "qA": p(r, D), "qB": p(D, r),
        "kA": p(r, D), "kB": p(kv_size, r),
        "vA": p(r, D), "vB": p(kv_size, r),
        "oA": p(r, D), "oB": p(D, r),
    }
    return lw


def test_layer():
    print("test_layer (full decoder layer backward):")
    torch.manual_seed(1)
    Hq, Hkv, Hd = 2, 1, 4
    D, kv_size, inter, r = Hq * Hd, Hkv * Hd, 16, 2
    dims = (Hq, Hkv, Hd, kv_size)
    scaling, eps, theta = 2.0, 1e-5, 10000.0
    seq_lens, b_start = [3, 2], [0, 3]
    n = sum(seq_lens)

    positions = torch.cat([torch.arange(s) for s in seq_lens])
    cos, sin = L.rope_cos_sin(positions, Hd, theta)

    lw_grad = _make_layer_weights(D, kv_size, inter, r, with_grad=True)
    x_req = torch.randn(n, D, dtype=torch.float32, requires_grad=True)

    # Reference: full layer output (reconstruct out from the cache + frozen down).
    cache_ref = L.layer_forward(x_req, lw_grad, scaling, cos, sin, seq_lens, b_start, dims, eps)
    import torch.nn.functional as F
    out = cache_ref["resid_mid"] + F.linear(
        F.silu(cache_ref["gate"]) * cache_ref["up"], lw_grad["down"])
    g = torch.randn_like(out)
    lora_keys = ["qA", "qB", "kA", "kB", "vA", "vB", "oA", "oB"]
    params = [x_req] + [lw_grad[k] for k in lora_keys]
    ref = torch.autograd.grad((out * g).sum(), params)
    ref_x, ref_lora = ref[0], dict(zip(lora_keys, ref[1:]))

    # Manual: detached weights + cache from a no-grad remat (gate/up recomputed).
    lw_det = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in lw_grad.items()}
    with torch.no_grad():
        cache = L.layer_forward(x_req.detach(), lw_det, scaling, cos, sin,
                                seq_lens, b_start, dims, eps)
        grad_x_m, grads_m = L.layer_backward(g, cache, lw_det, scaling, cos, sin,
                                             seq_lens, b_start, dims, eps)
    _check("grad_x", grad_x_m, ref_x)
    for k in lora_keys:
        _check(f"grad_{k}", grads_m[k], ref_lora[k])

    # Saved gate||up path: passing the saved pre-activations must give identical grads
    # (skips the gate_up matmul; same math).
    with torch.no_grad():
        gu = torch.cat([cache["gate"], cache["up"]], dim=-1)
        cache_s = L.layer_forward(x_req.detach(), lw_det, scaling, cos, sin,
                                  seq_lens, b_start, dims, eps, saved_gate_up=gu)
        grad_x_s, grads_s = L.layer_backward(g, cache_s, lw_det, scaling, cos, sin,
                                             seq_lens, b_start, dims, eps)
    _check("grad_x (saved-gu)", grad_x_s, ref_x)
    _check("grad_qB (saved-gu)", grads_s["qB"], ref_lora["qB"])


def main():
    test_head()
    test_layer()
    print(f"\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
