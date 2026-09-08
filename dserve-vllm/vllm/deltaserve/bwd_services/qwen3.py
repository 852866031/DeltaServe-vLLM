# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Qwen3 family: the per-layer math of the manual LoRA-SFT backward.

Qwen3 (``Qwen3ForCausalLM``) is a Llama-shaped SwiGLU decoder with exactly one
extra op in attention: a per-head ``RMSNorm(head_dim)`` on q and k — ``q_norm``
/ ``k_norm`` — applied AFTER the qkv projection and BEFORE RoPE
(``vllm/model_executor/models/qwen3.py::Qwen3Attention.forward``). The norms
are frozen (LoRA targets stay q/k/v/o), so they only add a forward step in the
rematerialization and an ``rmsnorm_backward`` (fp32, per the precision
contract) between the RoPE backward and the q/k projection backward.

Everything else — module names, fused qkv/gate_up weights, GQA, SwiGLU, the
residual layout, the TP shard geometry and the 7 all-reduces per layer — is
identical to Llama-3, so this module composes the same ``common`` cores. The
norm acts within a head and heads never straddle TP ranks, so it adds no
collective.

CUDA-graph path: all three captured regions (forward remat, FFN-bwd,
padded-attention-bwd) — ``graph_forward_core`` mirrors Llama-3's with the
norm between the projection and RoPE and writes the pre-norm q/k into the
runner's ``static_q_pre`` / ``static_k_pre`` for the eager backward tail.

``save_attn_qkv`` for this family captures q/k BEFORE the norm — pre-hooks on
``self_attn.q_norm`` / ``k_norm`` see the raw projection outputs (see
``Family.saved_qkv_pre_transform``), which is exactly what ``rmsnorm_backward``
needs — plus v from the ``self_attn.attn`` pre-hook. The remat then skips the
Q/K/V GEMM and only re-applies the elementwise norm + RoPE, so with
``save_attn_ctx`` + ``save_resid_mid`` the per-layer forward recompute is the
in_ln RMSNorm alone, as for Llama-3.
"""

import torch

from vllm.deltaserve.bwd_services.common.attention import (
    attn_backward_core,
    attn_forward_core,
)
from vllm.deltaserve.bwd_services.common.family import BASE_LAYER_WEIGHTS, Family
from vllm.deltaserve.bwd_services.common.ffn import (
    ffn_backward_core,
    ffn_forward_tail,
)
from vllm.deltaserve.bwd_services.common.ops import (
    apply_rope,
    proj,
    proj_backward,
    rmsnorm,
    rmsnorm_backward,
    rope_backward,
)
from vllm.deltaserve.bwd_services.common.tp import reduce_partial
from vllm.deltaserve.bwd_services.common.trainer import LoraSftTrainerService

# Frozen per-layer base weights: the shared SwiGLU set plus the q/k norms.
LAYER_WEIGHTS = {
    **BASE_LAYER_WEIGHTS,
    "q_norm": "self_attn.q_norm.weight",
    "k_norm": "self_attn.k_norm.weight",
}


# --------------------------------------------------------------------------- #
# Eager layer forward (rematerialization) + backward
# --------------------------------------------------------------------------- #

def layer_forward(x, lw, scaling, cos, sin, seq_lens, b_start, dims, eps,
                  saved_gate_up=None, saved_qh=None, saved_kh=None,
                  saved_vh=None, saved_ctx=None, saved_resid_mid=None,
                  all_reduce=None):
    """Rematerialize one Qwen3 decoder layer forward, returning the ``cache``
    the manual backward needs. Same contract as the Llama-3 ``layer_forward``
    (see that docstring for the ``saved_*`` shortcuts and ``all_reduce``), plus:

    - q/k pass through the per-head ``q_norm`` / ``k_norm`` between the
      projection and RoPE;
    - the cache carries the PRE-norm ``q_pre`` [n, Hq, Hd] / ``k_pre``
      [n, Hkv, Hd] the norm backward needs.

    ``saved_qh/kh`` for THIS family are the PRE-norm q/k (the raw projection
    outputs, captured at the ``q_norm``/``k_norm`` inputs — flat
    ``[n, q_size]`` / ``[n, kv_size]``); ``saved_vh`` is v. The Q/K/V GEMM is
    skipped and only the elementwise norm + RoPE are re-applied, so the cache
    still carries ``q_pre``/``k_pre`` for the norm backward."""
    Hq, Hkv, Hd, kv_size = dims
    n = x.shape[0]
    q_size = Hq * Hd

    x_norm1 = rmsnorm(x, lw["in_ln"], eps)
    if (saved_qh is not None
            and saved_kh is not None
            and saved_vh is not None):
        q_pre = saved_qh.view(n, Hq, Hd)
        k_pre = saved_kh.view(n, Hkv, Hd)
        qh = apply_rope(rmsnorm(q_pre, lw["q_norm"], eps), cos, sin)
        kh = apply_rope(rmsnorm(k_pre, lw["k_norm"], eps), cos, sin)
        vh = saved_vh.view(n, Hkv, Hd)
    else:
        q = proj(x_norm1, lw["q"], lw["qA"], lw["qB"], scaling)        # [n, q_size]
        k = proj(x_norm1, lw["k"], lw["kA"], lw["kB"], scaling)        # [n, kv_size]
        v = proj(x_norm1, lw["v"], lw["vA"], lw["vB"], scaling)
        q_pre = q.view(n, Hq, Hd)
        k_pre = k.view(n, Hkv, Hd)
        # Qwen3: per-head RMSNorm on q and k, then RoPE.
        qh = apply_rope(rmsnorm(q_pre, lw["q_norm"], eps), cos, sin)
        kh = apply_rope(rmsnorm(k_pre, lw["k_norm"], eps), cos, sin)
        vh = v.view(n, Hkv, Hd)

    if saved_ctx is not None:
        ctx_flat = saved_ctx.reshape(n, q_size)
    else:
        ctx_flat = attn_forward_core(qh, kh, vh, seq_lens, b_start, dims, x.dtype)

    if saved_resid_mid is not None:
        # Skip the O projection (+ its TP all-reduce) — the post-attention
        # residual was captured in the FT forward (already reduced, full width).
        resid_mid = saved_resid_mid
    else:
        o = proj(ctx_flat, lw["o"], lw["oA"], lw["oB"], scaling)     # [n, hidden]
        if all_reduce is not None:
            o = all_reduce(o)
        resid_mid = x + o

    gate, up = ffn_forward_tail(resid_mid, lw, eps, saved_gate_up)

    return {"x": x, "x_norm1": x_norm1, "q_pre": q_pre, "k_pre": k_pre,
            "qh": qh, "kh": kh, "vh": vh,
            "ctx_flat": ctx_flat, "resid_mid": resid_mid,
            "gate": gate, "up": up}


def _qk_norm_backward(grad_qh, grad_kh, cache, lw, cos, sin, eps, cdt):
    """RoPE backward, then the per-head q/k-norm backward (fp32), for the
    attention-core output grads. Returns (grad_q, grad_k) as [n, H, Hd] in
    ``cdt`` — the grads w.r.t. the raw q/k projection outputs."""
    if cache.get("q_pre") is None:
        raise RuntimeError(
            "qwen3 layer_backward needs the pre-norm q/k in the cache")
    grad_q = rmsnorm_backward(cache["q_pre"], rope_backward(grad_qh, cos, sin),
                              lw["q_norm"], eps).to(cdt)
    grad_k = rmsnorm_backward(cache["k_pre"], rope_backward(grad_kh, cos, sin),
                              lw["k_norm"], eps).to(cdt)
    return grad_q, grad_k


def layer_backward(grad_out, cache, lw, scaling, cos, sin, seq_lens, b_start,
                   dims, eps, cdt=torch.float32,
                   *, grad_qh_buf=None, grad_kh_buf=None, grad_vh_buf=None,
                   all_reduce=None, reduce_factors=True):
    """Manual gradient of one Qwen3 layer — the Llama-3 ``layer_backward``
    with the q/k-norm backward inserted between the RoPE backward and the q/k
    projection backward. Returns (grad_x, grads in cdt). See the Llama-3
    docstring for the precision rules and the TP reduction set (unchanged)."""
    Hq, Hkv, Hd, kv_size = dims
    n = grad_out.shape[0]
    q_size = Hq * Hd

    # --- FFN backward (frozen MLP): grad w.r.t. resid_mid (incl. residual) ---
    grad_resid_mid = ffn_backward_core(grad_out, cache, lw, eps, cdt)
    if all_reduce is not None:
        grad_resid_mid = reduce_partial(all_reduce, grad_resid_mid, grad_out)

    # --- O backward (cdt) ---
    grad_ctx_flat, grad_oA, grad_oB = proj_backward(
        cache["ctx_flat"], grad_resid_mid, lw["o"], lw["oA"], lw["oB"], scaling, cdt)

    # --- attention backward (per-sample, GQA): core in fp32, results back to cdt ---
    grad_ctx = grad_ctx_flat.view(n, Hq, Hd)
    grad_qh, grad_kh, grad_vh = attn_backward_core(
        cache["qh"], cache["kh"], cache["vh"], grad_ctx,
        seq_lens, b_start, dims, cdt,
        grad_qh_buf=grad_qh_buf, grad_kh_buf=grad_kh_buf, grad_vh_buf=grad_vh_buf)

    # --- RoPE backward → q/k-norm backward (Qwen3) ---
    grad_q, grad_k = _qk_norm_backward(grad_qh, grad_kh, cache, lw, cos, sin,
                                       eps, cdt)
    grad_q = grad_q.reshape(n, q_size)
    grad_k = grad_k.reshape(n, kv_size)
    grad_v = grad_vh.reshape(n, kv_size)

    # --- q/k/v projection backward (input = x_norm1), cdt ---
    xn1 = cache["x_norm1"]
    gx_q, grad_qA, grad_qB = proj_backward(xn1, grad_q, lw["q"], lw["qA"], lw["qB"], scaling, cdt)
    gx_k, grad_kA, grad_kB = proj_backward(xn1, grad_k, lw["k"], lw["kA"], lw["kB"], scaling, cdt)
    gx_v, grad_vA, grad_vB = proj_backward(xn1, grad_v, lw["v"], lw["vA"], lw["vB"], scaling, cdt)
    grad_x_norm1 = gx_q + gx_k + gx_v

    # --- RMSNorm backward to x (fp32) + residual path ---
    grad_x = rmsnorm_backward(cache["x"], grad_x_norm1, lw["in_ln"], eps).to(cdt) + grad_resid_mid

    grads = {"qA": grad_qA, "qB": grad_qB, "kA": grad_kA, "kB": grad_kB,
             "vA": grad_vA, "vB": grad_vB, "oA": grad_oA, "oB": grad_oB}

    if all_reduce is not None:
        grad_x = reduce_partial(all_reduce, grad_x, grad_resid_mid)
        if reduce_factors:
            # Inline reduce of the replicated factors. The trainer passes
            # reduce_factors=False under TP and buckets them instead (M4.3:
            # one collective per group of layers on the comm stream).
            for _k in ("qA", "kA", "vA", "oB"):
                if grads[_k] is not None:
                    grads[_k] = all_reduce(grads[_k])
    return grad_x, grads


# --------------------------------------------------------------------------- #
# CUDA-graph path: forward remat + FFN-bwd + attention-bwd graphs
# --------------------------------------------------------------------------- #

def graph_forward_core(runner, lw: dict) -> None:
    """The captureable Qwen3 layer forward on the ``GraphedBackward``
    runner's static buffers — the Llama-3 core with the per-head q/k-norm
    between the projection and RoPE. Reads ``static_layer_in`` +
    ``static_cos/sin``; writes the same outputs as Llama-3's (ending at
    ``static_o``, the runner's ``forward_tail`` finishes the layer) plus
    ``static_q_pre`` / ``static_k_pre`` (the norm inputs the backward tail's
    ``rmsnorm_backward`` consumes). In ``save_attn_qkv`` mode the pre-norm
    q/k (+ v) are read from ``static_saved_qh/kh/vh`` and the Q/K/V GEMM is
    skipped; only the norm + RoPE are re-applied. Zero tail rows stay zero:
    ``rmsnorm(0) = 0`` and RoPE of zero is zero, so the padded scatter's
    ``accumulate=True`` contract holds."""
    s = runner.s_max
    Hq, Hkv, Hd = runner.Hq, runner.Hkv, runner.Hd
    scaling, eps = runner.scaling, runner.eps

    # 1) RMSNorm(in_ln) — the Q/K/V LoRA-A backward needs x_norm1.
    x_norm1 = rmsnorm(runner.static_layer_in, lw["in_ln"], eps)
    runner.static_x_norm1.copy_(x_norm1)

    if runner.save_attn_qkv:
        # 2') Saved pre-norm q/k (q_norm/k_norm inputs) + v, staged by
        #     ``stage_forward_inputs``. No projection GEMM.
        q_pre = runner.static_saved_qh.view(s, Hq, Hd)
        k_pre = runner.static_saved_kh.view(s, Hkv, Hd)
        v = runner.static_saved_vh
    else:
        # 2) Q/K/V projections (base + LoRA).
        q = proj(x_norm1, lw["q"], lw["qA"], lw["qB"], scaling)      # [s, q_size]
        k = proj(x_norm1, lw["k"], lw["kA"], lw["kB"], scaling)      # [s, kv]
        v = proj(x_norm1, lw["v"], lw["vA"], lw["vB"], scaling)
        q_pre = q.view(s, Hq, Hd)
        k_pre = k.view(s, Hkv, Hd)
    runner.static_q_pre.copy_(q_pre)
    runner.static_k_pre.copy_(k_pre)

    # 3) RoPE on the normed q, k; pack v.
    qh = apply_rope(rmsnorm(q_pre, lw["q_norm"], eps), runner.static_cos,
                    runner.static_sin)
    kh = apply_rope(rmsnorm(k_pre, lw["k_norm"], eps), runner.static_cos,
                    runner.static_sin)
    vh = v.view(s, Hkv, Hd)

    # Flat copies for the eager backward tail + padded scatter for Graph B.
    runner.static_qh_flat.copy_(qh)
    runner.static_kh_flat.copy_(kh)
    runner.static_vh_flat.copy_(vh)
    runner.scatter_qkv_padded(qh, kh, vh)

    # 4) Attention forward → static_ctx_flat (or the saved context).
    if runner.save_attn_ctx:
        runner.static_ctx_flat.copy_(runner.static_saved_ctx)
    else:
        runner.padded_attn_forward_core()

    # 5) O projection (base + LoRA) → static_o; the residual add + gate/up
    #    split are the runner's ``forward_tail`` (captured here at tp_size ==
    #    1, eager after the o all-reduce under TP — M5). Skipped entirely in
    #    save_resid_mid mode (resid_mid staged into static_resid_mid).
    if not runner.save_resid_mid:
        o = proj(runner.static_ctx_flat, lw["o"], lw["oA"], lw["oB"], scaling)
        runner.static_o.copy_(o)


@torch.no_grad()
def layer_backward_graphed(runner, maybe_pause, layer_id, g, cache, lw, scaling,
                           cos, sin, seq_lens, b_start, dims, eps, cdt,
                           *, all_reduce=None, reduce_factors=True):
    """Graphed per-layer backward: Graph A (FFN-bwd) → eager O-bwd → pause →
    Graph B (padded-attn-bwd) → eager tail (RoPE bwd → q/k-norm bwd → Q/K/V
    proj bwd → in_ln rmsnorm bwd). Same values as the eager ``layer_backward``,
    including its six TP reduces when ``all_reduce`` is given (all in this
    eager code, never inside a replay — M5). The cache (from the forward
    graph's ``cache_views`` or the eager ``layer_forward``) carries ``q_pre``
    / ``k_pre``."""
    Hq, _, Hd, kv_size = dims
    n = g.shape[0]
    q_size = Hq * Hd

    grad_resid_mid = runner.ffn_backward(layer_id, g, cache, lw)
    if all_reduce is not None:
        grad_resid_mid = reduce_partial(all_reduce, grad_resid_mid, g)

    grad_ctx_flat, grad_oA, grad_oB = proj_backward(
        cache["ctx_flat"], grad_resid_mid, lw["o"],
        lw["oA"], lw["oB"], scaling, cdt)

    maybe_pause()

    grad_ctx = grad_ctx_flat.view(n, Hq, Hd)
    grad_qh, grad_kh, grad_vh = runner.attn_backward(
        layer_id, cache["qh"], cache["kh"], cache["vh"], grad_ctx,
        seq_lens, b_start, dims)

    grad_q, grad_k = _qk_norm_backward(grad_qh, grad_kh, cache, lw, cos, sin,
                                       eps, cdt)
    grad_q = grad_q.reshape(n, q_size)
    grad_k = grad_k.reshape(n, kv_size)
    grad_v = grad_vh.reshape(n, kv_size)
    xn1 = cache["x_norm1"]
    gx_q, grad_qA, grad_qB = proj_backward(
        xn1, grad_q, lw["q"], lw["qA"], lw["qB"], scaling, cdt)
    gx_k, grad_kA, grad_kB = proj_backward(
        xn1, grad_k, lw["k"], lw["kA"], lw["kB"], scaling, cdt)
    gx_v, grad_vA, grad_vB = proj_backward(
        xn1, grad_v, lw["v"], lw["vA"], lw["vB"], scaling, cdt)
    grad_x_norm1 = gx_q + gx_k + gx_v
    grad_x = rmsnorm_backward(
        cache["x"], grad_x_norm1, lw["in_ln"], eps).to(cdt) + grad_resid_mid

    grads = {"qA": grad_qA, "qB": grad_qB, "kA": grad_kA, "kB": grad_kB,
             "vA": grad_vA, "vB": grad_vB, "oA": grad_oA, "oB": grad_oB}

    if all_reduce is not None:
        grad_x = reduce_partial(all_reduce, grad_x, grad_resid_mid)
        if reduce_factors:
            # Inline reduce of the replicated factors. The trainer passes
            # reduce_factors=False under TP and buckets them instead (M4.3:
            # one collective per group of layers on the comm stream).
            for _k in ("qA", "kA", "vA", "oB"):
                if grads[_k] is not None:
                    grads[_k] = all_reduce(grads[_k])
    return grad_x, grads


# --------------------------------------------------------------------------- #
# Family record + service
# --------------------------------------------------------------------------- #

QWEN3 = Family(
    name="qwen3",
    archs=("Qwen3ForCausalLM", "qwen3"),
    layer_forward=layer_forward,
    layer_backward=layer_backward,
    layer_weights=LAYER_WEIGHTS,
    graph_forward_core=graph_forward_core,
    layer_backward_graphed=layer_backward_graphed,
    saved_qkv_pre_transform=True,   # q/k captured at the q_norm/k_norm inputs
)


class Qwen3BackwardService(LoraSftTrainerService):
    """The Qwen3 backward child: the family-agnostic trainer driving the
    layer math above."""

    family = QWEN3
