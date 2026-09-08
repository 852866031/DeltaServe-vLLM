# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Llama-3 family: the per-layer math of the manual LoRA-SFT backward.

Trains the FT LoRA adapter (attention-only q/k/v/o) in the backward subprocess
from the captured residual-stream activations, while inference keeps serving.
Mirrors DeltaServe ``models/{llama,llama3}/SFT_service.py`` but adapted to vLLM:

- PEFT-separate LoRA (``lora_A [r,in]``, ``lora_B [out,r]``, delta =
  scaling·(x@Aᵀ)@Bᵀ) rather than DeltaServe's packed ``[2,4r,H,Hd]``; grads
  derived directly in PEFT layout.
- vLLM's **fused** base weights (``qkv_proj``/``gate_up_proj``) — sliced once at
  setup by the trainer.
- vLLM RoPE (NeoX rotate-half; cos/sin rebuilt from ``inv_freq=1/theta^(2i/d)``).
- No score clamp (vLLM doesn't clamp).

We capture only the per-layer residual-stream **input** (``layer_in[i]``) +
``final_in``, so each layer's forward is **rematerialized** from ``layer_in[i]``
to recover the intermediates the manual backward needs, then the gradients are
computed by hand.

This module holds only what is Llama-specific: the decoder-layer composition
(RMSNorm → Q/K/V (+LoRA) → RoPE → GQA attention → O (+LoRA) → residual →
SwiGLU FFN) in eager and CUDA-graph form. The building blocks live in
``bwd_services/common`` and are shared with the other families; the trainer
service that drives them is ``common/trainer.py``.

Precision (see ``deltaserve-backward-precision`` memory): scores/softmax/RMSNorm/
LM-head in fp32; bulk projections in the weights' dtype; fp32 LoRA master /
optimizer. MLP/embeddings/norms are frozen — only the 8 LoRA tensors per layer
get gradients.
"""

import torch

from vllm.deltaserve.bwd_services.common.attention import (
    attn_backward_core,
    attn_forward_core,
)
from vllm.deltaserve.bwd_services.common.family import Family
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


# --------------------------------------------------------------------------- #
# Eager layer forward (rematerialization) + backward
# --------------------------------------------------------------------------- #

def layer_forward(x, lw, scaling, cos, sin, seq_lens, b_start, dims, eps,
                  saved_gate_up=None, saved_qh=None, saved_kh=None,
                  saved_vh=None, saved_ctx=None, saved_resid_mid=None,
                  all_reduce=None):
    """Rematerialize one Llama decoder layer forward, returning the ``cache`` the
    manual backward needs (attention internals + MLP pre-activations). The frozen
    `down` matmul / layer output are NOT computed — the backward only needs the
    incoming gradient, not the layer output (we already saved the next layer's input).

    When ``saved_gate_up`` ([n, 2*intermediate] = gate||up, captured in the forward)
    is given, the MLP `gate_up` matmul is skipped entirely — the biggest recompute in
    the layer. Otherwise gate/up are recomputed (the gradcheck path).

    When ``saved_qh/kh/vh`` are given (each [n, q_size or kv_size] flat,
    captured by ``self_attn.attn`` forward_pre_hook), we ALSO skip Q/K/V proj
    + RoPE — the second-biggest recompute. ``x_norm1`` is still computed from
    ``x`` (cheap; the Q/K/V LoRA-A grad needs it as ``grad_A = grad_Z.t() @
    x_norm1``). ``cos/sin`` are unused on this fast path. If any of the three
    saved tensors is absent, the recompute path runs.

    When ``saved_ctx`` ([n, q_size] = the attention context / o_proj input,
    captured by a ``self_attn.attn`` forward_hook) is given, the attention
    forward (the per-sample scores/softmax/AV loop) is SKIPPED — ``ctx_flat``
    is read from the saved tensor. ``qh/kh/vh`` are still produced (saved or
    recomputed) because the attention BACKWARD consumes them. Composes with
    ``saved_qh/kh/vh``: with both, only RMSNorm in_ln + O-proj + residual are
    recomputed.

    When ``saved_resid_mid`` ([n, hidden] = the post-attention residual
    ``x + o``, captured by a ``post_attention_layernorm`` forward_pre_hook) is
    given, the O projection (base + LoRA GEMM) and the residual add are
    SKIPPED — and with it, under TP, the forward's only all-reduce. ``ctx_flat``
    is still produced (the O-proj BACKWARD reads it). Composes with the other
    two: with all three saved, only RMSNorm in_ln is recomputed.

    ``lw`` is a dict of base weights (q,k,v,o,gate,up,down,in_ln,post_ln) + LoRA
    (qA,qB,kA,kB,vA,vB,oA,oB). Functional (no in-place) so autograd can differentiate
    it for the gradcheck. ``dims`` = (Hq, Hkv, Hd, kv_size); the residual width is
    ``x.shape[-1]`` and may differ from the attention width ``Hq*Hd``.

    ``all_reduce`` (TP only): o_proj is row-parallel — each rank holds ctx for
    its own heads and o.weight sharded on the input dim, so ``o`` is a partial
    sum. Reducing it mirrors the real forward's post-attention reduce so
    ``resid_mid`` is the full residual on every rank. None for tp=1."""
    Hq, Hkv, Hd, kv_size = dims
    n = x.shape[0]
    q_size = Hq * Hd

    x_norm1 = rmsnorm(x, lw["in_ln"], eps)
    if (saved_qh is not None
            and saved_kh is not None
            and saved_vh is not None):
        # Skip Q/K/V proj + RoPE — use the saved post-RoPE flat tensors.
        qh = saved_qh.view(n, Hq, Hd)
        kh = saved_kh.view(n, Hkv, Hd)
        vh = saved_vh.view(n, Hkv, Hd)
    else:
        q = proj(x_norm1, lw["q"], lw["qA"], lw["qB"], scaling)        # [n, q_size]
        k = proj(x_norm1, lw["k"], lw["kA"], lw["kB"], scaling)        # [n, kv_size]
        v = proj(x_norm1, lw["v"], lw["vA"], lw["vB"], scaling)
        qh = apply_rope(q.view(n, Hq, Hd), cos, sin)
        kh = apply_rope(k.view(n, Hkv, Hd), cos, sin)
        vh = v.view(n, Hkv, Hd)

    if saved_ctx is not None:
        # Skip the attention forward (scores/softmax/AV) — use the saved ctx.
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

    return {"x": x, "x_norm1": x_norm1, "qh": qh, "kh": kh, "vh": vh,
            "ctx_flat": ctx_flat, "resid_mid": resid_mid,
            "gate": gate, "up": up}


def layer_backward(grad_out, cache, lw, scaling, cos, sin, seq_lens, b_start,
                   dims, eps, cdt=torch.float32,
                   *, grad_qh_buf=None, grad_kh_buf=None, grad_vh_buf=None,
                   all_reduce=None, reduce_factors=True):
    """Manual gradient of one layer. Bulk matmuls (FFN-bwd, LoRA-grad, rope/proj-bwd)
    run in ``cdt`` (bf16 in prod, fp32 for the gradcheck); the **attention core**
    (scores/softmax-bwd/dQ/dK/dV) and **RMSNorm backward** always run in fp32 (the
    load-bearing precision rules — DeltaServe). Returns (grad_x, grads in cdt).
    MLP/norms frozen. Composes ``ffn_backward_core`` + ``attn_backward_core`` so
    the eager and graphed paths share one definition of the math.

    ``grad_{qh,kh,vh}_buf`` are optional persistent buffers passed through to
    ``attn_backward_core`` — used by the service to avoid 96 zero-fill kernel
    launches per backward. Not provided by the gradcheck test.

    TP reductions (``all_reduce`` given; 7 per layer incl. the forward's o):
      - ``grad_resid_mid``: the FFN-path term is a per-rank partial while the
        residual passthrough (``grad_out``) is already full → reduce only the
        partial (``reduce_partial``), else the residual counts ``tp_size``×.
      - ``grad_x``: same shape of argument for the attention-path term vs the
        already-reduced ``grad_resid_mid``.
      - ``grad_{q,k,v}A`` / ``grad_oB``: replicated factors → grads sum across
        shards. ``grad_{q,k,v}B`` (output-sharded) and ``grad_oA``
        (input-sharded) are already this rank's shard — no reduce."""
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

    grad_q = rope_backward(grad_qh, cos, sin).reshape(n, q_size)
    grad_k = rope_backward(grad_kh, cos, sin).reshape(n, kv_size)
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
# CUDA-graph path (Phase 5): the captureable forward + the graphed backward
# --------------------------------------------------------------------------- #

def graph_forward_core(runner, lw: dict) -> None:
    """The captureable layer-forward body (no eager helpers, no python loop
    over samples), operating on the ``GraphedBackward`` runner's static
    buffers. Reads ``static_layer_in`` + ``static_cos/sin`` +
    ``static_saved_gate_up``; writes:

      - ``static_x_norm1`` (eager Q/K/V LoRA-A bwd input)
      - ``static_qh_flat / kh_flat / vh_flat`` (eager RoPE-bwd input)
      - ``static_qh_pad  / kh_pad  / vh_pad`` (Graph B input — scattered)
      - ``static_ctx_flat`` (eager O-proj bwd input)
      - ``static_o`` (the O-proj output; the runner's ``forward_tail`` turns
        it into ``static_resid_mid`` + ``static_gate / static_up`` — the
        Graph A inputs — after the TP all-reduce when there is one)

    Mirrors ``layer_forward`` with ``saved_gate_up != None`` up to the O
    projection; differs only in being shape-stable at the fixed ``s_max``
    slab and writing into static buffers in place."""
    s = runner.s_max
    Hq, Hkv, Hd = runner.Hq, runner.Hkv, runner.Hd
    scaling = runner.scaling

    # 1) RMSNorm(in_ln) on [s_max, D] — runs in BOTH modes (the Q/K/V
    #    LoRA-A backward needs x_norm1; cheap, ~few MFLOPs).
    x_norm1 = rmsnorm(runner.static_layer_in, lw["in_ln"], runner.eps)
    runner.static_x_norm1.copy_(x_norm1)

    if runner.save_attn_qkv:
        # Fast path: post-RoPE q/k/v were captured in the FT forward and
        # staged into static_saved_qh/kh/vh by ``stage_forward_inputs``.
        # Skip Q/K/V proj + RoPE entirely. Bandwidth-only — three model-
        # dtype reads of [s, q_size] / [s, kv_size].
        qh = runner.static_saved_qh.view(s, Hq, Hd)
        kh = runner.static_saved_kh.view(s, Hkv, Hd)
        vh = runner.static_saved_vh.view(s, Hkv, Hd)
    else:
        # 2) Q/K/V projections (base + LoRA) — LoRA `.data` refs are stable.
        q = proj(x_norm1, lw["q"], lw["qA"], lw["qB"], scaling)      # [s, q_size]
        k = proj(x_norm1, lw["k"], lw["kA"], lw["kB"], scaling)      # [s, kv]
        v = proj(x_norm1, lw["v"], lw["vA"], lw["vB"], scaling)

        # 3) RoPE on q, k; pack v.
        qh = apply_rope(q.view(s, Hq, Hd), runner.static_cos, runner.static_sin)
        kh = apply_rope(k.view(s, Hkv, Hd), runner.static_cos, runner.static_sin)
        vh = v.view(s, Hkv, Hd)

    # "Flat" writes (model dtype, shape [s, H, Hd]) — read by the eager
    # RoPE-bwd tail. In save_attn_qkv mode the eager tail still needs
    # these reshaped views; copy is cheap and keeps the cache_views
    # contract identical to the recompute path.
    runner.static_qh_flat.copy_(qh)
    runner.static_kh_flat.copy_(kh)
    runner.static_vh_flat.copy_(vh)
    # Scatter into padded layout (Graph B inputs). All s_max rows write,
    # so we MUST use accumulate=True: tail rows (k ≥ n) have
    # (bn_idx, pos_idx) = (0, 0), which coincides with the legit
    # (sample 0, position 0) slot — without accumulate they'd overwrite
    # it (last-write-wins). With accumulate=True, the legit row adds
    # qh[real_0] and tail rows add 0 (input tail is zero because
    # rmsnorm(0)·W = 0 and 0·W^T = 0; in save_attn_qkv mode the staged
    # static_saved_qh/kh/vh tails are zeroed in stage_forward_inputs).
    # Pre-zeroed in ``stage_forward_inputs`` so accumulate starts from a
    # clean slab.
    runner.scatter_qkv_padded(qh, kh, vh)

    # 4) Attention forward → static_ctx_flat. In save_attn_ctx mode the
    #    forward attention (scores/softmax/AV) is skipped — ctx was captured
    #    in the FT forward and staged into static_saved_ctx. The q/k/v
    #    padded scatter above still ran (Graph B / attn-bwd reads it).
    if runner.save_attn_ctx:
        runner.static_ctx_flat.copy_(runner.static_saved_ctx)
    else:
        runner.padded_attn_forward_core()

    # 5) O projection (base + LoRA) → static_o. The residual add + the saved
    #    gate||up split are the runner's ``forward_tail``: captured right after
    #    this at tp_size == 1, run eagerly after the o all-reduce under TP
    #    (o_proj is row-parallel, so ``o`` is a partial sum per rank — M5).
    #    In save_resid_mid mode the post-attention residual was staged straight
    #    into static_resid_mid (Graph A's input): no O projection, no reduce.
    if not runner.save_resid_mid:
        o = proj(runner.static_ctx_flat, lw["o"], lw["oA"], lw["oB"], scaling)
        runner.static_o.copy_(o)


@torch.no_grad()
def layer_backward_graphed(runner, maybe_pause, layer_id, g, cache, lw, scaling,
                           cos, sin, seq_lens, b_start, dims, eps, cdt,
                           *, all_reduce=None, reduce_factors=True):
    """Graphed per-layer backward: Graph A (FFN-bwd) → eager O-bwd → pause
    (yield GPU to inference) → Graph B (padded-attn-bwd) → eager tail
    (RoPE / Q-K-V proj / in_ln rmsnorm). Same gradient values as the eager
    ``layer_backward``; differs only in *when* host dispatch happens.

    Pause cadence: once per layer, between the two graphs — the eager
    path's once-per-layer pause-at-start is preserved, just relocated to
    the mid-layer point so each graph runs without yielding (the captured
    region can't host an mp.Event.wait anyway).

    ``all_reduce`` (TP, M5): the six backward-side reduces of ``layer_backward``
    at the same points — all in this eager code, never inside a replay."""
    Hq, _, Hd, kv_size = dims
    n = g.shape[0]
    q_size = Hq * Hd

    # --- Graph A: FFN-bwd (frozen MLP, with residual) ---
    grad_resid_mid = runner.ffn_backward(layer_id, g, cache, lw)
    if all_reduce is not None:
        # M5: same reduce set as the eager ``layer_backward`` — the FFN-path
        # partial only (the residual passthrough ``g`` is already full).
        grad_resid_mid = reduce_partial(all_reduce, grad_resid_mid, g)

    # --- Eager: O-projection backward (LoRA grad lives here) ---
    grad_ctx_flat, grad_oA, grad_oB = proj_backward(
        cache["ctx_flat"], grad_resid_mid, lw["o"],
        lw["oA"], lw["oB"], scaling, cdt)

    # --- Yield GPU to inference between graphs (preserves the load-bearing
    # per-layer pause cadence; mp.Event.wait can't run inside a graph). ---
    maybe_pause()

    # --- Graph B: padded-attention backward CORE ---
    grad_ctx = grad_ctx_flat.view(n, Hq, Hd)
    grad_qh, grad_kh, grad_vh = runner.attn_backward(
        layer_id, cache["qh"], cache["kh"], cache["vh"], grad_ctx,
        seq_lens, b_start, dims)

    # --- Eager tail: RoPE bwd → Q/K/V proj bwd → in_ln rmsnorm bwd ---
    grad_q = rope_backward(grad_qh, cos, sin).reshape(n, q_size)
    grad_k = rope_backward(grad_kh, cos, sin).reshape(n, kv_size)
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

LLAMA3 = Family(
    name="llama3",
    archs=("LlamaForCausalLM", "llama3", "llama"),
    layer_forward=layer_forward,
    layer_backward=layer_backward,
    graph_forward_core=graph_forward_core,
    layer_backward_graphed=layer_backward_graphed,
)


class Llama3BackwardService(LoraSftTrainerService):
    """The Llama-3 backward child: the family-agnostic trainer driving the
    layer math above."""

    family = LLAMA3
