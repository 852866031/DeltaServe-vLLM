#!/usr/bin/env python
"""CUDA-graph parity for the Qwen3 backward (q/k-norm family).

The runner's three captured regions with the Qwen3 family: the forward graph
(``qwen3.graph_forward_core`` — projection → per-head q/k-norm → RoPE →
padded attention → O → residual, writing ``q_pre``/``k_pre`` too) must
reproduce the eager ``layer_forward`` cache, and the graphed backward
(Graph A → O-bwd → Graph B → tail with the norm backward) must reproduce the
eager ``layer_backward`` gradients. Also checks the ``save_attn_ctx`` forward
branch. Requires CUDA; geometry has Hq*Hd != hidden.

    python tests/test_qwen3_backward_graph.py
"""

import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bwd_harness as H  # noqa: E402

sys.path[:] = [p for p in sys.path
               if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]

from vllm.deltaserve.bwd_services import qwen3 as Q  # noqa: E402
from vllm.deltaserve.bwd_services.common.graph import GraphedBackward  # noqa: E402

if not torch.cuda.is_available():
    print("CUDA not available — skipping Qwen3 graphed-backward parity test")
    sys.exit(0)

DEVICE = torch.device("cuda")
MDT = torch.bfloat16
CDT = torch.float32
_FP32_TOL = 1e-5     # graph vs eager: same math, same dtypes
C = H.Checker(tol=_FP32_TOL)

CACHE_KEYS = ("x", "x_norm1", "q_pre", "k_pre", "qh", "kh", "vh",
              "ctx_flat", "resid_mid", "gate", "up")


def _mk_svc(hidden, L_, Hq, Hkv, Hd, inter, eps, lws, scaling, theta,
            save_attn_ctx=False, save_resid_mid=False, save_attn_qkv=False):
    """Stand-in for LoraSftTrainerService — what GraphedBackward reads."""
    svc = SimpleNamespace(
        device_index=0, family=Q.QWEN3,
        D=hidden, L=L_, Hq=Hq, Hkv=Hkv, Hd=Hd, inter=inter,
        kv_size=Hkv * Hd, eps=eps, base_dtype=MDT, bwd_dtype=CDT,
        scaling=scaling, theta=theta,
        save_attn_qkv=save_attn_qkv, save_attn_ctx=save_attn_ctx,
        save_resid_mid=save_resid_mid,
    )
    svc._layer_weights = lambda i: lws[i]
    return svc


def _weights(L_, hidden, Hq, Hkv, Hd, inter, r, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    return [H.make_layer_weights(Q.QWEN3, hidden, Hq, Hkv, Hd, inter, r,
                                 dtype=MDT, device=DEVICE, gen=gen)
            for _ in range(L_)]


def test_forward_and_backward_graph_parity():
    print("test_forward_and_backward_graph_parity (Qwen3, Hq*Hd != hidden):")
    hidden, Hq, Hkv, Hd, inter, r = 24, 4, 2, 8, 32, 4     # q_size 32 != 24
    L_, eps, scaling, theta = 2, 1e-6, 2.0, 1e6
    s_max, bn_max, l_max = 16, 4, 8
    dims = (Hq, Hkv, Hd, Hkv * Hd)
    lws = _weights(L_, hidden, Hq, Hkv, Hd, inter, r, seed=3)
    svc = _mk_svc(hidden, L_, Hq, Hkv, Hd, inter, eps, lws, scaling, theta)
    runner = GraphedBackward(svc, s_max=s_max, bn_max=bn_max, l_max=l_max)
    C.ok("forward graphs captured for every layer", not runner.fwd_failed)

    for batch_idx, seq_lens in enumerate([[5, 3], [8, 4]]):
        n = sum(seq_lens)
        b_start = [sum(seq_lens[:i]) for i in range(len(seq_lens))]
        runner.begin_backward(n, seq_lens, b_start)
        _, cos, sin = H.seq_layout(seq_lens, Hd, theta, device=DEVICE)
        g = torch.randn(n, hidden, device=DEVICE, dtype=CDT)
        for i in range(L_):
            lw = lws[i]
            layer_in = torch.randn(n, hidden, device=DEVICE, dtype=MDT)
            saved_gu = torch.randn(n, 2 * inter, device=DEVICE, dtype=MDT)
            ref = Q.layer_forward(layer_in, lw, scaling, cos, sin, seq_lens,
                                  b_start, dims, eps, saved_gate_up=saved_gu)
            out = runner.forward(i, lw, layer_in, saved_gu, n)
            for key in CACHE_KEYS:
                C.check(f"b{batch_idx} L{i} fwd {key}", out[key], ref[key])
            # Backward: graphed (on the graph cache) vs eager (on the eager cache).
            gx_ref, grads_ref = Q.layer_backward(
                g, ref, lw, scaling, cos, sin, seq_lens, b_start, dims, eps, cdt=CDT)
            gx, grads = Q.layer_backward_graphed(
                runner, lambda: None, i, g, out, lw, scaling, cos, sin,
                seq_lens, b_start, dims, eps, CDT)
            C.check(f"b{batch_idx} L{i} bwd grad_x", gx, gx_ref)
            for k in H.LORA_KEYS:
                C.check(f"b{batch_idx} L{i} bwd grad_{k}", grads[k], grads_ref[k])


def test_saved_ctx_forward_graph():
    print("test_saved_ctx_forward_graph (Qwen3, save_attn_ctx=True):")
    hidden, Hq, Hkv, Hd, inter, r = 24, 4, 2, 8, 32, 4
    L_, eps, scaling, theta = 1, 1e-6, 2.0, 1e6
    s_max, bn_max, l_max = 16, 4, 8
    dims = (Hq, Hkv, Hd, Hkv * Hd)
    lws = _weights(L_, hidden, Hq, Hkv, Hd, inter, r, seed=7)
    svc = _mk_svc(hidden, L_, Hq, Hkv, Hd, inter, eps, lws, scaling, theta,
                  save_attn_ctx=True)
    runner = GraphedBackward(svc, s_max=s_max, bn_max=bn_max, l_max=l_max)
    seq_lens = [6, 2]
    n = sum(seq_lens)
    b_start = [0, 6]
    runner.begin_backward(n, seq_lens, b_start)
    _, cos, sin = H.seq_layout(seq_lens, Hd, theta, device=DEVICE)
    lw = lws[0]
    layer_in = torch.randn(n, hidden, device=DEVICE, dtype=MDT)
    saved_gu = torch.randn(n, 2 * inter, device=DEVICE, dtype=MDT)
    full = Q.layer_forward(layer_in, lw, scaling, cos, sin, seq_lens, b_start,
                           dims, eps, saved_gate_up=saved_gu)
    saved_ctx = full["ctx_flat"].reshape(n, -1).contiguous()
    ref = Q.layer_forward(layer_in, lw, scaling, cos, sin, seq_lens, b_start,
                          dims, eps, saved_gate_up=saved_gu, saved_ctx=saved_ctx)
    out = runner.forward(0, lw, layer_in, saved_gu, n, saved_ctx=saved_ctx)
    for key in CACHE_KEYS:
        C.check(f"saved-ctx fwd {key}", out[key], ref[key])


def test_saved_resid_mid_forward_graph():
    """``save_resid_mid`` (+ ``save_attn_ctx``): the graphed forward skips
    the O projection and reads the staged residual; must match the full
    recompute, and the graphed backward on that cache must match eager."""
    print("test_saved_resid_mid_forward_graph (Qwen3, save_resid_mid + ctx):")
    hidden, Hq, Hkv, Hd, inter, r = 24, 4, 2, 8, 32, 4
    L_, eps, scaling, theta = 2, 1e-6, 2.0, 1e6
    s_max, bn_max, l_max = 16, 4, 8
    dims = (Hq, Hkv, Hd, Hkv * Hd)
    lws = _weights(L_, hidden, Hq, Hkv, Hd, inter, r, seed=9)
    svc = _mk_svc(hidden, L_, Hq, Hkv, Hd, inter, eps, lws, scaling, theta,
                  save_attn_ctx=True, save_resid_mid=True)
    runner = GraphedBackward(svc, s_max=s_max, bn_max=bn_max, l_max=l_max)
    C.ok("forward graphs captured (save_resid_mid)", not runner.fwd_failed)
    for batch_idx, seq_lens in enumerate([[5, 3], [8, 4]]):
        n = sum(seq_lens)
        b_start = [sum(seq_lens[:i]) for i in range(len(seq_lens))]
        runner.begin_backward(n, seq_lens, b_start)
        _, cos, sin = H.seq_layout(seq_lens, Hd, theta, device=DEVICE)
        g = torch.randn(n, hidden, device=DEVICE, dtype=CDT)
        for i in range(L_):
            lw = lws[i]
            layer_in = torch.randn(n, hidden, device=DEVICE, dtype=MDT)
            saved_gu = torch.randn(n, 2 * inter, device=DEVICE, dtype=MDT)
            full = Q.layer_forward(layer_in, lw, scaling, cos, sin, seq_lens,
                                   b_start, dims, eps, saved_gate_up=saved_gu)
            saved = dict(saved_ctx=full["ctx_flat"].reshape(n, -1).contiguous(),
                         saved_resid_mid=full["resid_mid"].contiguous())
            ref = Q.layer_forward(layer_in, lw, scaling, cos, sin, seq_lens,
                                  b_start, dims, eps, saved_gate_up=saved_gu, **saved)
            out = runner.forward(i, lw, layer_in, saved_gu, n, **saved)
            for key in CACHE_KEYS:
                C.check(f"b{batch_idx} L{i} eager-saved {key}", ref[key], full[key])
                C.check(f"b{batch_idx} L{i} graph {key}", out[key], full[key])
            gx_ref, grads_ref = Q.layer_backward(
                g, full, lw, scaling, cos, sin, seq_lens, b_start, dims, eps, cdt=CDT)
            gx, grads = Q.layer_backward_graphed(
                runner, lambda: None, i, g, out, lw, scaling, cos, sin,
                seq_lens, b_start, dims, eps, CDT)
            C.check(f"b{batch_idx} L{i} bwd grad_x", gx, gx_ref)
            for k in H.LORA_KEYS:
                C.check(f"b{batch_idx} L{i} bwd grad_{k}", grads[k], grads_ref[k])


def test_saved_qkv_pre_norm(all_saves):
    """``save_attn_qkv`` for Qwen3 = the PRE-norm q/k (q_norm/k_norm inputs)
    + v. Eager ``layer_forward(saved_qh/kh/vh=...)`` and the graphed forward
    (core skips the Q/K/V GEMM, re-applies norm + RoPE from the staged
    pre-norm q/k) must reproduce the full recompute — cache incl. q_pre /
    k_pre — and the graphed backward on that cache must match eager. With
    ``all_saves`` the ctx + residual are saved too (only in_ln recomputed)."""
    tag = "all-saves" if all_saves else "qkv-only"
    print(f"test_saved_qkv_pre_norm (Qwen3, {tag}):")
    hidden, Hq, Hkv, Hd, inter, r = 24, 4, 2, 8, 32, 4
    L_, eps, scaling, theta = 2, 1e-6, 2.0, 1e6
    s_max, bn_max, l_max = 16, 4, 8
    dims = (Hq, Hkv, Hd, Hkv * Hd)
    lws = _weights(L_, hidden, Hq, Hkv, Hd, inter, r, seed=11)
    svc = _mk_svc(hidden, L_, Hq, Hkv, Hd, inter, eps, lws, scaling, theta,
                  save_attn_qkv=True, save_attn_ctx=all_saves,
                  save_resid_mid=all_saves)
    runner = GraphedBackward(svc, s_max=s_max, bn_max=bn_max, l_max=l_max)
    C.ok(f"{tag}: forward graphs captured", not runner.fwd_failed
         and runner.save_attn_qkv)
    for batch_idx, seq_lens in enumerate([[5, 3], [8, 4]]):
        n = sum(seq_lens)
        b_start = [sum(seq_lens[:i]) for i in range(len(seq_lens))]
        runner.begin_backward(n, seq_lens, b_start)
        _, cos, sin = H.seq_layout(seq_lens, Hd, theta, device=DEVICE)
        g = torch.randn(n, hidden, device=DEVICE, dtype=CDT)
        for i in range(L_):
            lw = lws[i]
            layer_in = torch.randn(n, hidden, device=DEVICE, dtype=MDT)
            saved_gu = torch.randn(n, 2 * inter, device=DEVICE, dtype=MDT)
            full = Q.layer_forward(layer_in, lw, scaling, cos, sin, seq_lens,
                                   b_start, dims, eps, saved_gate_up=saved_gu)
            # What the accumulator captures on Qwen3: pre-norm q/k, v.
            saved = dict(saved_qh=full["q_pre"].reshape(n, -1).contiguous(),
                         saved_kh=full["k_pre"].reshape(n, -1).contiguous(),
                         saved_vh=full["vh"].reshape(n, -1).contiguous())
            if all_saves:
                saved.update(saved_ctx=full["ctx_flat"].contiguous(),
                             saved_resid_mid=full["resid_mid"].contiguous())
            ref = Q.layer_forward(layer_in, lw, scaling, cos, sin, seq_lens,
                                  b_start, dims, eps, saved_gate_up=saved_gu, **saved)
            out = runner.forward(i, lw, layer_in, saved_gu, n, **saved)
            for key in CACHE_KEYS:
                C.check(f"{tag} b{batch_idx} L{i} eager-saved {key}", ref[key], full[key])
                C.check(f"{tag} b{batch_idx} L{i} graph {key}", out[key], full[key])
            gx_ref, grads_ref = Q.layer_backward(
                g, full, lw, scaling, cos, sin, seq_lens, b_start, dims, eps, cdt=CDT)
            gx_s, grads_s = Q.layer_backward(
                g, ref, lw, scaling, cos, sin, seq_lens, b_start, dims, eps, cdt=CDT)
            gx, grads = Q.layer_backward_graphed(
                runner, lambda: None, i, g, out, lw, scaling, cos, sin,
                seq_lens, b_start, dims, eps, CDT)
            C.check(f"{tag} b{batch_idx} L{i} eager-saved bwd grad_x", gx_s, gx_ref)
            C.check(f"{tag} b{batch_idx} L{i} graph bwd grad_x", gx, gx_ref)
            for k in H.LORA_KEYS:
                C.check(f"{tag} b{batch_idx} L{i} graph bwd grad_{k}", grads[k], grads_ref[k])


if __name__ == "__main__":
    test_forward_and_backward_graph_parity()
    test_saved_ctx_forward_graph()
    test_saved_resid_mid_forward_graph()
    test_saved_qkv_pre_norm(all_saves=False)
    test_saved_qkv_pre_norm(all_saves=True)
    C.finish()
