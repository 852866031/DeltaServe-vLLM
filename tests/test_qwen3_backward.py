#!/usr/bin/env python
"""Gradcheck for the Qwen3 backward service (q/k-norm family).

Validates the hand-derived gradients in ``vllm.deltaserve.bwd_services.qwen3``
against autograd through the family's own functional ``layer_forward``:
head (final norm + LM head) and one decoder layer including the per-head
``q_norm``/``k_norm`` between the projection and RoPE. Uses a geometry where
the attention width differs from the residual width (Hq*Hd != hidden, as on
Qwen3-0.6B) so that conflation is caught too. fp32, CPU.

    python tests/test_qwen3_backward.py
"""

import os
import sys

import torch

# The harness lives next to this script; re-add tests/ first so a spawned
# worker (which inherits the scrubbed sys.path) can import it too.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bwd_harness as H  # noqa: E402

sys.path[:] = [p for p in sys.path
               if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]

from vllm.deltaserve.bwd_services import llama3 as LL  # noqa: E402
from vllm.deltaserve.bwd_services import qwen3 as Q  # noqa: E402
from vllm.deltaserve.bwd_services.common.head import head_backward  # noqa: E402

C = H.Checker(tol=1e-3)


def test_head():
    print("test_head (final norm + LM head):")
    torch.manual_seed(0)
    hidden, vocab, eps = 8, 10, 1e-6
    seq_lens = [3, 2]
    n = sum(seq_lens)
    b_start, _, _ = H.seq_layout(seq_lens, 4, 1e6)
    final_in = torch.randn(n, hidden, requires_grad=True)
    lm_w = torch.randn(vocab, hidden)
    norm_w = torch.randn(hidden).abs() + 0.5
    ids = torch.randint(0, vocab, (n,))
    loss_ref, _ = H.ref_head_loss(final_in, lm_w, norm_w, eps, ids, seq_lens, b_start, vocab)
    (grad_ref,) = torch.autograd.grad(loss_ref, final_in)
    loss_m, n_valid, grad_m = head_backward(
        final_in.detach(), lm_w, norm_w, eps, ids, seq_lens, b_start, vocab)
    C.check("loss", torch.tensor(loss_m), loss_ref.detach())
    C.check("grad_final_in", grad_m, grad_ref)


def test_layer():
    print("test_layer (q/k-norm decoder layer, Hq*Hd != hidden):")
    gen = torch.Generator().manual_seed(1)
    hidden, Hq, Hkv, Hd, inter, r = 8, 4, 2, 4, 16, 2      # q_size = 16 != 8
    kv_size = Hkv * Hd
    dims = (Hq, Hkv, Hd, kv_size)
    scaling, eps, theta = 2.0, 1e-6, 1e6
    seq_lens = [3, 2]
    n = sum(seq_lens)
    b_start, cos, sin = H.seq_layout(seq_lens, Hd, theta)

    lw_grad = H.make_layer_weights(Q.QWEN3, hidden, Hq, Hkv, Hd, inter, r,
                                   with_grad=True, gen=gen)
    x_req = torch.randn(n, hidden, generator=gen, requires_grad=True)

    # Sanity: the norm is live (Qwen3 layer differs from the Llama-3 layer on
    # the same weights).
    with torch.no_grad():
        out_q = H.ref_layer_output(Q.QWEN3, x_req, lw_grad, scaling, cos, sin,
                                   seq_lens, b_start, dims, eps)
        out_l = H.ref_layer_output(LL.LLAMA3, x_req, lw_grad, scaling, cos, sin,
                                   seq_lens, b_start, dims, eps)
    C.ok("q/k-norm changes the layer output",
         (out_q - out_l).abs().max().item() > 1e-3)

    out_ref = H.ref_layer_output(Q.QWEN3, x_req, lw_grad, scaling, cos, sin,
                                 seq_lens, b_start, dims, eps)
    g = torch.randn(out_ref.shape, generator=gen)
    ref = torch.autograd.grad((out_ref * g).sum(), [x_req] + H.lora_params(lw_grad))
    lw_det = H.detach_weights(lw_grad)

    with torch.no_grad():
        cache = Q.layer_forward(x_req.detach(), lw_det, scaling, cos, sin,
                                seq_lens, b_start, dims, eps)
        grad_x_m, grads_m = Q.layer_backward(g, cache, lw_det, scaling, cos, sin,
                                             seq_lens, b_start, dims, eps)
    C.check("grad_x", grad_x_m, ref[0])
    for k, r_ in zip(H.LORA_KEYS, ref[1:]):
        C.check(f"grad_{k}", grads_m[k], r_)

    # The saved-gate_up shortcut must be exact.
    with torch.no_grad():
        saved_gu = torch.cat([cache["gate"], cache["up"]], dim=-1)
        cache_s = Q.layer_forward(x_req.detach(), lw_det, scaling, cos, sin,
                                  seq_lens, b_start, dims, eps, saved_gate_up=saved_gu)
        grad_x_s, grads_s = Q.layer_backward(g, cache_s, lw_det, scaling, cos, sin,
                                             seq_lens, b_start, dims, eps)
    C.check("grad_x (saved-gu)", grad_x_s, ref[0])
    C.check("grad_qB (saved-gu)", grads_s["qB"], ref[2])

    # The saved-qkv shortcut for Qwen3 feeds the PRE-norm q/k (what the
    # q_norm/k_norm pre-hooks capture) + v: the forward re-applies norm + RoPE
    # and the backward through the norm is exact vs the autograd reference.
    with torch.no_grad():
        cache_q = Q.layer_forward(
            x_req.detach(), lw_det, scaling, cos, sin, seq_lens, b_start, dims, eps,
            saved_qh=cache["q_pre"].reshape(n, -1), saved_kh=cache["k_pre"].reshape(n, -1),
            saved_vh=cache["vh"].reshape(n, -1))
        grad_x_q, grads_q = Q.layer_backward(g, cache_q, lw_det, scaling, cos, sin,
                                             seq_lens, b_start, dims, eps)
    C.check("grad_x (saved pre-norm qkv)", grad_x_q, ref[0])
    C.check("grad_qB (saved pre-norm qkv)", grads_q["qB"], ref[2])
    C.check("cache qh (saved pre-norm qkv)", cache_q["qh"], cache["qh"])
    C.ok("family flags", Q.QWEN3.supports_saved_qkv and Q.QWEN3.saved_qkv_pre_transform
         and "q_norm" in Q.QWEN3.layer_weights)


if __name__ == "__main__":
    test_head()
    test_layer()
    C.finish()
