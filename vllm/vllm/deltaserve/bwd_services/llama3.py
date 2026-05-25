# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Llama-3 backward service — manual LoRA SFT backward (Phase 3).

Trains the FT LoRA adapter (attention-only q/k/v/o) in the backward subprocess
from the captured residual-stream activations, while inference keeps serving.
Mirrors DeltaServe `models/{llama,llama3}/SFT_service.py` but adapted to vLLM:

- PEFT-separate LoRA (`lora_A [r,in]`, `lora_B [out,r]`, delta = scaling·(x@Aᵀ)@Bᵀ)
  rather than DeltaServe's packed `[2,4r,H,Hd]`; grads derived directly in PEFT layout.
- vLLM's **fused** base weights (`qkv_proj`/`gate_up_proj`) — sliced once at setup.
- vLLM RoPE (NeoX rotate-half; cos/sin rebuilt from `inv_freq=1/theta^(2i/d)`).
- No score clamp (vLLM doesn't clamp).

We capture only the per-layer residual-stream **input** (`layer_in[i]`) + `final_in`,
so each layer's forward is **rematerialized** from `layer_in[i]` to recover the
intermediates the manual backward needs, then the gradients are computed by hand.

Precision (see `deltaserve-backward-precision` memory): scores/softmax/RMSNorm/LM-head
in fp32; bulk projections in the weights' dtype; fp32 LoRA master / optimizer. The
manual backward runs in fp32. MLP/embeddings/norms are frozen — only the 8 LoRA
tensors per layer get gradients.
"""

import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.deltaserve import dprint
from vllm.deltaserve.bwd_services.base import BackwardService

_TOL = 5e-2  # bf16 round-trip tolerance for the capture/forward-fidelity checks
_VOCAB_CHUNK = 16384


# --------------------------------------------------------------------------- #
# Math helpers (module-level so the gradcheck test can call them directly).
# Everything here operates in the dtype of its inputs, except scores/softmax/
# RMSNorm internals which upcast to fp32. The gradcheck passes fp32 throughout.
# --------------------------------------------------------------------------- #

def rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float):
    """NeoX cos/sin for the given positions. Returns (cos, sin) [n, head_dim//2] fp32."""
    inv_freq = 1.0 / (theta ** (
        torch.arange(0, head_dim, 2, dtype=torch.float32, device=positions.device)
        / head_dim))
    freqs = torch.outer(positions.float(), inv_freq)   # [n, head_dim//2]
    return freqs.cos(), freqs.sin()


def apply_rope(xh: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """NeoX rotate-half on [n, H, head_dim]; cos/sin [n, head_dim//2]."""
    h = xh.shape[-1] // 2
    x1, x2 = xh[..., :h], xh[..., h:]
    c, s = cos[:, None, :].to(xh.dtype), sin[:, None, :].to(xh.dtype)
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


def rope_backward(g: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Inverse of apply_rope (transpose of the rotation)."""
    h = g.shape[-1] // 2
    g1, g2 = g[..., :h], g[..., h:]
    c, s = cos[:, None, :].to(g.dtype), sin[:, None, :].to(g.dtype)
    return torch.cat([g1 * c + g2 * s, -g1 * s + g2 * c], dim=-1)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float):
    """RMSNorm forward (fp32 internal), output cast back to x.dtype."""
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return ((xf * inv) * w.float()).to(x.dtype)


def rmsnorm_backward(x: torch.Tensor, grad_y: torch.Tensor, w: torch.Tensor,
                     eps: float):
    """Exact gradient of y = rmsnorm(x)·w w.r.t. x (fp32)."""
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)   # 1/rms
    xhat = xf * inv
    g_xhat = grad_y.float() * w.float()
    d = xf.shape[-1]
    dot = (g_xhat * xhat).sum(-1, keepdim=True)
    return (g_xhat - xhat * dot / d) * inv


def _logits_chunked(h: torch.Tensor, lm_w: torch.Tensor, vocab: int):
    """fp32 logits = h @ lm_w[:vocab].T, computed in vocab chunks (bounds the fp32
    LM-head temporary). h [m, D], lm_w [vocab_pad, D]. Returns [m, vocab] fp32."""
    hf = h.float()
    out = hf.new_empty((hf.shape[0], vocab))
    for c in range(0, vocab, _VOCAB_CHUNK):
        e = min(c + _VOCAB_CHUNK, vocab)
        out[:, c:e] = hf @ lm_w[c:e].float().t()
    return out


def _proj(xin, w_base, A, B, scaling):
    """y = xin @ w_base.T + scaling·(xin @ A.T) @ B.T. w_base [out,in], A [r,in], B [out,r].

    The LoRA master is fp32; cast its compute copy to xin's dtype (bf16 in prod,
    fp32 in the gradcheck) so the matmul dtypes match (DeltaServe fp32-master /
    low-precision-compute rule)."""
    y = F.linear(xin, w_base)
    if A is not None:
        y = y + scaling * F.linear(F.linear(xin, A.to(xin.dtype)), B.to(xin.dtype))
    return y


def _proj_backward(xin, gy, w_base, A, B, scaling, cdt=torch.float32):
    """Grad of _proj, computed in the bulk compute dtype ``cdt`` (bf16 in prod,
    fp32 in the gradcheck). Returns (grad_xin, grad_A, grad_B).

    Casts the weights inside the function on every call. This is intentional
    and matches DeltaServe's precision contract (bf16 compute for projection
    backward, fp32 LoRA master).

    Why we don't amortize the casts: DeltaServe packs all 8 LoRA tensors per
    layer into a single ``[2, 4r, H, Hd]`` buffer cast once at the layer
    boundary (SFT_service.py:424). We use separate qA/qB/… tensors so the
    cast count is the cardinality of distinct tensors (7 base + 8 LoRA = 15),
    and that's the same whether we cast inline or pre-cast. The real-cast
    count is also mode-symmetric:
      - bf16 cdt (default): 7 base ``.to(bf16)`` are no-ops, 8 LoRA fp32→bf16
        are real casts. Total: 8 real casts/layer.
      - fp32 cdt (``backward_fp32=True``, rare): 7 base bf16→fp32 are real
        casts (heavier — base tensors are big, e.g. ``down`` is [D, inter]),
        8 LoRA ``.to(fp32)`` are no-ops. Total: 7 real casts/layer.
    Amortizing wouldn't change either count. Keep this layout."""
    xin = xin.to(cdt)
    gy = gy.to(cdt)
    grad_xin = gy @ w_base.to(cdt)
    grad_A = grad_B = None
    if A is not None:
        Ac, Bc = A.to(cdt), B.to(cdt)
        Z = xin @ Ac.t()                       # [n, r]
        grad_Z = scaling * (gy @ Bc)           # [n, r]
        grad_A = grad_Z.t() @ xin              # [r, in]
        grad_B = scaling * (gy.t() @ Z)        # [out, r]
        grad_xin = grad_xin + grad_Z @ Ac
    return grad_xin, grad_A, grad_B


def layer_forward(x, lw, scaling, cos, sin, seq_lens, b_start, dims, eps,
                  saved_gate_up=None):
    """Rematerialize one Llama decoder layer forward, returning the ``cache`` the
    manual backward needs (attention internals + MLP pre-activations). The frozen
    `down` matmul / layer output are NOT computed — the backward only needs the
    incoming gradient, not the layer output (we already saved the next layer's input).

    When ``saved_gate_up`` ([n, 2*intermediate] = gate||up, captured in the forward)
    is given, the MLP `gate_up` matmul is skipped entirely — the biggest recompute in
    the layer. Otherwise gate/up are recomputed (the gradcheck path).

    ``lw`` is a dict of base weights (q,k,v,o,gate,up,in_ln,post_ln) + LoRA
    (qA,qB,kA,kB,vA,vB,oA,oB). Functional (no in-place) so autograd can differentiate
    it for the gradcheck. ``dims`` = (Hq, Hkv, Hd, kv_size)."""
    Hq, Hkv, Hd, kv_size = dims
    n = x.shape[0]
    D = Hq * Hd
    scale = 1.0 / math.sqrt(Hd)
    kv_repeat = Hq // Hkv

    x_norm1 = rmsnorm(x, lw["in_ln"], eps)
    q = _proj(x_norm1, lw["q"], lw["qA"], lw["qB"], scaling)        # [n, D]
    k = _proj(x_norm1, lw["k"], lw["kA"], lw["kB"], scaling)        # [n, kv_size]
    v = _proj(x_norm1, lw["v"], lw["vA"], lw["vB"], scaling)
    qh = apply_rope(q.view(n, Hq, Hd), cos, sin)
    kh = apply_rope(k.view(n, Hkv, Hd), cos, sin)
    vh = v.view(n, Hkv, Hd)

    ctx_blocks = []
    for st, ln in zip(b_start, seq_lens):
        q_blk = qh[st:st + ln].transpose(0, 1)      # [Hq, L, Hd]
        k_blk = kh[st:st + ln].transpose(0, 1)      # [Hkv, L, Hd]
        v_blk = vh[st:st + ln].transpose(0, 1)
        if kv_repeat != 1:
            k_rep = k_blk.repeat_interleave(kv_repeat, 0)
            v_rep = v_blk.repeat_interleave(kv_repeat, 0)
        else:
            k_rep, v_rep = k_blk, v_blk
        scores = (q_blk.float() @ k_rep.float().transpose(-1, -2)) * scale
        mask = torch.triu(torch.ones(ln, ln, dtype=torch.bool, device=x.device), 1)
        scores = scores.masked_fill(mask, -1e9)
        att = torch.softmax(scores, dim=-1)         # [Hq, L, L] fp32
        ctx_blk = (att @ v_rep.float()).to(x.dtype).transpose(0, 1)  # [L, Hq, Hd]
        ctx_blocks.append(ctx_blk)
    ctx_flat = torch.cat(ctx_blocks, 0).reshape(n, D)

    o = _proj(ctx_flat, lw["o"], lw["oA"], lw["oB"], scaling)       # [n, D]
    resid_mid = x + o

    if saved_gate_up is not None:
        # Skip the gate_up matmul — use the saved forward pre-activations.
        inter = saved_gate_up.shape[-1] // 2
        gate = saved_gate_up[:, :inter]
        up = saved_gate_up[:, inter:]
    else:
        x_norm2 = rmsnorm(resid_mid, lw["post_ln"], eps)
        gate = F.linear(x_norm2, lw["gate"])
        up = F.linear(x_norm2, lw["up"])

    return {"x": x, "x_norm1": x_norm1, "qh": qh, "kh": kh, "vh": vh,
            "ctx_flat": ctx_flat, "resid_mid": resid_mid,
            "gate": gate, "up": up}


def ffn_backward_core(grad_out, cache, lw, eps, cdt=torch.float32):
    """FFN-block backward (frozen MLP, with residual): returns grad_resid_mid.

    Captures the shape-stable bulk of the per-layer backward: cast-in, FFN
    silu/sigmoid + 3 GEMMs against frozen down/gate/up, rmsnorm-post_ln backward,
    plus the residual add. Inputs are all sized by the chained ``grad_out``
    (``[n, D]``) and ``cache`` slices captured at the same n. Pure function over
    its inputs so it can be wrapped 1:1 by a CUDA graph at fixed n=s_max."""
    gout = grad_out.to(cdt)
    gate = cache["gate"].to(cdt)
    up = cache["up"].to(cdt)
    sig = torch.sigmoid(gate)
    silu = gate * sig
    silu_grad = sig * (1.0 + gate * (1.0 - sig))
    grad_h_mid = gout @ lw["down"].to(cdt)           # [n, inter]
    grad_up = grad_h_mid * silu
    grad_gate = grad_h_mid * up * silu_grad
    grad_x_norm2 = grad_gate @ lw["gate"].to(cdt) + grad_up @ lw["up"].to(cdt)
    return rmsnorm_backward(cache["resid_mid"], grad_x_norm2,
                            lw["post_ln"], eps).to(cdt) + gout


def attn_backward_core(qh, kh, vh, grad_ctx, seq_lens, b_start, dims, cdt=torch.float32,
                       *, grad_qh_buf=None, grad_kh_buf=None, grad_vh_buf=None):
    """Per-sample GQA attention backward (eager). Returns (grad_qh, grad_kh,
    grad_vh) in flat ``[n, H_*, Hd]`` layout, in ``cdt``. Per-sample shapes
    (variable seq_lens) prevent CUDA-graph capture; the runner uses a padded
    variant for the graphed fast path and silently falls back to this when a
    batch overflows the padded bounds. Scores/softmax/dQ/dK/dV always run in
    fp32 (the load-bearing GQA precision rule).

    If ``grad_qh_buf/kh_buf/vh_buf`` are provided (persistent buffers sized at
    s_max), the function zeroes their first n rows in-place and returns views
    into them — saves L=32 fresh-allocation zero-fills per backward. Falls
    back to fresh allocation when not provided (preserves the gradcheck path,
    which has no service to own persistent buffers)."""
    Hq, Hkv, Hd, _ = dims
    scale = 1.0 / math.sqrt(Hd)
    kv_repeat = Hq // Hkv
    n = grad_ctx.shape[0]
    device = grad_ctx.device
    # Use the persistent buffers only when they're large enough for n. They're
    # sized at s_max for the production path (n ≤ s_max always); the runner's
    # eager-fallback inside Llama3GraphedBackward.attn_backward also passes
    # its own static_grad_* (also s_max). The undersized branch fires only in
    # contrived parity tests that exceed s_max — we silently alloc fresh.
    if grad_qh_buf is not None and grad_qh_buf.shape[0] >= n:
        grad_qh_buf[:n].zero_()
        grad_kh_buf[:n].zero_()
        grad_vh_buf[:n].zero_()
        grad_qh = grad_qh_buf[:n]
        grad_kh = grad_kh_buf[:n]
        grad_vh = grad_vh_buf[:n]
    else:
        grad_qh = torch.zeros((n, Hq, Hd), dtype=cdt, device=device)
        grad_kh = torch.zeros((n, Hkv, Hd), dtype=cdt, device=device)
        grad_vh = torch.zeros((n, Hkv, Hd), dtype=cdt, device=device)
    qh_f, kh_f, vh_f = qh.float(), kh.float(), vh.float()
    for st, ln in zip(b_start, seq_lens):
        q_blk = qh_f[st:st + ln].transpose(0, 1)     # [Hq, L, Hd] fp32
        k_blk = kh_f[st:st + ln].transpose(0, 1)     # [Hkv, L, Hd] fp32
        v_blk = vh_f[st:st + ln].transpose(0, 1)
        if kv_repeat != 1:
            k_rep = k_blk.repeat_interleave(kv_repeat, 0)
            v_rep = v_blk.repeat_interleave(kv_repeat, 0)
        else:
            k_rep, v_rep = k_blk, v_blk
        mask = torch.triu(torch.ones(ln, ln, dtype=torch.bool, device=device), 1)
        scores = (q_blk @ k_rep.transpose(-1, -2)) * scale   # fp32
        scores = scores.masked_fill(mask, -1e9)
        att = torch.softmax(scores, dim=-1)          # [Hq, L, L] fp32
        g = grad_ctx[st:st + ln].transpose(0, 1).float()     # fp32
        grad_att = g @ v_rep.transpose(-1, -2)       # [Hq, L, L]
        grad_v_rep = att.transpose(-1, -2) @ g       # [Hq, L, Hd]
        sm = (grad_att * att).sum(-1, keepdim=True)
        grad_scores = (att * (grad_att - sm)).masked_fill(mask, 0.0)
        grad_q_blk = (grad_scores @ k_rep) * scale
        grad_k_rep = (grad_scores.transpose(-1, -2) @ q_blk) * scale
        if kv_repeat != 1:
            grad_k_kv = grad_k_rep.view(Hkv, kv_repeat, ln, Hd).sum(1)
            grad_v_kv = grad_v_rep.view(Hkv, kv_repeat, ln, Hd).sum(1)
        else:
            grad_k_kv, grad_v_kv = grad_k_rep, grad_v_rep
        grad_qh[st:st + ln] = grad_q_blk.transpose(0, 1).to(cdt)
        grad_kh[st:st + ln] = grad_k_kv.transpose(0, 1).to(cdt)
        grad_vh[st:st + ln] = grad_v_kv.transpose(0, 1).to(cdt)
    return grad_qh, grad_kh, grad_vh


def layer_backward(grad_out, cache, lw, scaling, cos, sin, seq_lens, b_start,
                   dims, eps, cdt=torch.float32,
                   *, grad_qh_buf=None, grad_kh_buf=None, grad_vh_buf=None):
    """Manual gradient of one layer. Bulk matmuls (FFN-bwd, LoRA-grad, rope/proj-bwd)
    run in ``cdt`` (bf16 in prod, fp32 for the gradcheck); the **attention core**
    (scores/softmax-bwd/dQ/dK/dV) and **RMSNorm backward** always run in fp32 (the
    load-bearing precision rules — DeltaServe). Returns (grad_x, grads in cdt).
    MLP/norms frozen. Composes ``ffn_backward_core`` + ``attn_backward_core`` so
    the eager and graphed paths share one definition of the math.

    ``grad_{qh,kh,vh}_buf`` are optional persistent buffers passed through to
    ``attn_backward_core`` — used by the service to avoid 96 zero-fill kernel
    launches per backward. Not provided by the gradcheck test."""
    Hq, Hkv, Hd, kv_size = dims
    n = grad_out.shape[0]
    D = Hq * Hd

    # --- FFN backward (frozen MLP): grad w.r.t. resid_mid (incl. residual) ---
    grad_resid_mid = ffn_backward_core(grad_out, cache, lw, eps, cdt)

    # --- O backward (cdt) ---
    grad_ctx_flat, grad_oA, grad_oB = _proj_backward(
        cache["ctx_flat"], grad_resid_mid, lw["o"], lw["oA"], lw["oB"], scaling, cdt)

    # --- attention backward (per-sample, GQA): core in fp32, results back to cdt ---
    grad_ctx = grad_ctx_flat.view(n, Hq, Hd)
    grad_qh, grad_kh, grad_vh = attn_backward_core(
        cache["qh"], cache["kh"], cache["vh"], grad_ctx,
        seq_lens, b_start, dims, cdt,
        grad_qh_buf=grad_qh_buf, grad_kh_buf=grad_kh_buf, grad_vh_buf=grad_vh_buf)

    grad_q = rope_backward(grad_qh, cos, sin).reshape(n, D)
    grad_k = rope_backward(grad_kh, cos, sin).reshape(n, kv_size)
    grad_v = grad_vh.reshape(n, kv_size)

    # --- q/k/v projection backward (input = x_norm1), cdt ---
    xn1 = cache["x_norm1"]
    gx_q, grad_qA, grad_qB = _proj_backward(xn1, grad_q, lw["q"], lw["qA"], lw["qB"], scaling, cdt)
    gx_k, grad_kA, grad_kB = _proj_backward(xn1, grad_k, lw["k"], lw["kA"], lw["kB"], scaling, cdt)
    gx_v, grad_vA, grad_vB = _proj_backward(xn1, grad_v, lw["v"], lw["vA"], lw["vB"], scaling, cdt)
    grad_x_norm1 = gx_q + gx_k + gx_v

    # --- RMSNorm backward to x (fp32) + residual path ---
    grad_x = rmsnorm_backward(cache["x"], grad_x_norm1, lw["in_ln"], eps).to(cdt) + grad_resid_mid

    grads = {"qA": grad_qA, "qB": grad_qB, "kA": grad_kA, "kB": grad_kB,
             "vA": grad_vA, "vB": grad_vB, "oA": grad_oA, "oB": grad_oB}
    return grad_x, grads


def head_backward(final_in, lm_w, norm_w, eps, ids, seq_lens, b_start, vocab):
    """LM-head + final-norm: per-sample shift CE loss + grad w.r.t. final_in.
    Returns (loss: float, n_valid: int, grad_final_in [n,D] fp32)."""
    normed = rmsnorm(final_in, norm_w, eps)
    n = final_in.shape[0]
    logit_grad = normed.new_zeros((n, vocab), dtype=torch.float32)
    total_loss = normed.new_zeros((), dtype=torch.float32)
    n_valid = 0
    for st, ln in zip(b_start, seq_lens):
        if ln < 2:
            continue
        h = normed[st:st + ln - 1]                   # positions predict next
        lg = _logits_chunked(h, lm_w, vocab)         # [ln-1, vocab] fp32
        tgt = ids[st + 1:st + ln].long()
        total_loss = total_loss + F.cross_entropy(lg, tgt, reduction="sum")
        p = torch.softmax(lg, dim=-1)
        p[torch.arange(ln - 1, device=p.device), tgt] -= 1.0
        logit_grad[st:st + ln - 1] = p
        n_valid += int(ln - 1)
    if n_valid == 0:
        return 0.0, 0, final_in.new_zeros((n, final_in.shape[-1]), dtype=torch.float32)
    logit_grad /= n_valid
    # grad w.r.t. the post-final-norm hidden = logit_grad @ lm_w, chunked over vocab.
    grad_normed = logit_grad.new_zeros((n, final_in.shape[-1]))
    for c in range(0, vocab, _VOCAB_CHUNK):
        e = min(c + _VOCAB_CHUNK, vocab)
        grad_normed += logit_grad[:, c:e] @ lm_w[c:e].float()
    grad_final_in = rmsnorm_backward(final_in, grad_normed, norm_w, eps)
    return float(total_loss.item() / n_valid), n_valid, grad_final_in


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #

_PROJ = ("q", "k", "v", "o")


class Llama3BackwardService(BackwardService):

    def __init__(self, device_index: int) -> None:
        super().__init__(device_index)
        self.is_trainer = True
        self._built = False
        self.lora: dict = {}            # layer -> {proj -> {"A":Param,"B":Param}}
        self.base: dict = {}            # layer -> {q,k,v,o,gate,up,down,in_ln,post_ln}
        self.optimizer = None
        self.scheduler = None
        # CUDA-graph runner (Phase 5). Attached in _build_state when
        # meta["backward_cuda_graph"] is True; None means all-eager (today's
        # behaviour). See vllm/deltaserve/bwd_services/llama3_graph.py.
        self.graph_runner = None
        # Persistent grad_qh/kh/vh buffers (allocated in _build_state once
        # dims + cdt are known). Reused across layers and backwards by the
        # eager attn_backward_core path via its grad_*_buf kwargs.
        self._grad_qh_buf = None
        self._grad_kh_buf = None
        self._grad_vh_buf = None

    # -- build master params + optimizer once weights are received -----------

    def _handle_share_weights(self, conn, msg) -> None:
        super()._handle_share_weights(conn, msg)
        try:
            self._build_state()
            self._built = True
        except Exception as e:  # noqa: BLE001 — surface but keep the service alive
            import traceback
            dprint(f"[backward] llama3 state build failed: {e}")
            traceback.print_exc()

    def _build_state(self) -> None:
        meta = self.shared["meta"]
        base = self.shared["base"] or {}
        ft = self.shared["ft"] or {}
        self.D = int(meta["hidden_size"])
        self.L = int(meta["num_hidden_layers"])
        self.Hq = int(meta["num_attention_heads"])
        self.Hkv = int(meta["num_key_value_heads"])
        self.Hd = int(meta["head_dim"])
        self.kv_size = self.Hkv * self.Hd
        self.q_size = self.Hq * self.Hd
        self.inter = int(meta["intermediate_size"])
        self.theta = float(meta["rope_theta"])
        self.eps = float(meta["rms_norm_eps"])
        self.scaling = float(meta.get("lora_scaling", 1.0))
        self.vocab = int(meta["vocab_size"])
        self.dims = (self.Hq, self.Hkv, self.Hd, self.kv_size)

        # With enable_lora, vLLM wraps the projections, so the frozen base weight is
        # named e.g. "...qkv_proj.base_layer.weight". Normalize by stripping the
        # ".base_layer" infix so we can look weights up by their logical names
        # (norms/lm_head aren't wrapped, so they pass through unchanged).
        bn = {k.replace(".base_layer.", "."): v for k, v in base.items()}

        def bw(name):
            if name not in bn:
                raise KeyError(
                    f"base weight {name!r} not found (have e.g. "
                    f"{[k for k in list(bn)[:4]]} … {len(bn)} keys)")
            return bn[name]

        self.lm_w = bw("lm_head.weight")
        self.norm_w = bw("model.norm.weight")
        # Bulk backward compute dtype: model dtype (bf16) by default, fp32 if the
        # config flag is set. Attention core / RMSNorm-bwd / LM-head stay fp32.
        self.base_dtype = self.lm_w.dtype
        self.bwd_dtype = torch.float32 if meta.get("backward_fp32") else self.base_dtype

        # Slice the fused base weights ([out,in]) once into per-layer views.
        for i in range(self.L):
            p = f"model.layers.{i}."
            qkv = bw(p + "self_attn.qkv_proj.weight")
            gate_up = bw(p + "mlp.gate_up_proj.weight")
            self.base[i] = {
                "q": qkv[:self.q_size],
                "k": qkv[self.q_size:self.q_size + self.kv_size],
                "v": qkv[self.q_size + self.kv_size:],
                "o": bw(p + "self_attn.o_proj.weight"),
                "gate": gate_up[:self.inter],
                "up": gate_up[self.inter:],
                "down": bw(p + "mlp.down_proj.weight"),
                "in_ln": bw(p + "input_layernorm.weight"),
                "post_ln": bw(p + "post_attention_layernorm.weight"),
            }

        # FT adapter -> per-layer/proj fp32 master nn.Parameters (child-owned clone).
        key_re = re.compile(
            r"layers\.(\d+)\.self_attn\.([qkvo])_proj\.lora_([AB])\.weight")
        params: list[nn.Parameter] = []
        for key, t in ft.items():
            m = key_re.search(key)
            if m is None:
                continue
            layer, proj, ab = int(m.group(1)), m.group(2), m.group(3)
            param = nn.Parameter(t.detach().clone().float())  # fp32 master
            self.lora.setdefault(layer, {}).setdefault(proj, {})[ab] = param
            params.append(param)

        self.optimizer = torch.optim.AdamW(
            params, lr=float(meta["learning_rate"]), betas=(0.9, 0.999),
            weight_decay=float(meta["weight_decay"]))
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=float(meta["gamma"]))
        dprint(
            f"[backward] llama3 built: {self.L} layers, {len(params)} LoRA tensors, "
            f"scaling={self.scaling}, lr={meta['learning_rate']}, "
            f"wd={meta['weight_decay']}, gamma={meta['gamma']}"
        )

        # Persistent grad_qh/kh/vh buffers for the eager attn_backward_core path
        # (passed via attn_backward_core's optional grad_*_buf kwargs). Sized at
        # s_max so they fit any backward, sliced to [:n] per call. Saves 96
        # zero-fill kernel launches per backward (32 layers × 3 tensors). The
        # graphed path has its own static buffers in Llama3GraphedBackward.
        s_max = int(meta.get("max_saved_finetuning_tokens", 0))
        if s_max > 0:
            dev = self.lm_w.device
            cdt = self.bwd_dtype
            self._grad_qh_buf = torch.zeros((s_max, self.Hq, self.Hd),
                                            dtype=cdt, device=dev)
            self._grad_kh_buf = torch.zeros((s_max, self.Hkv, self.Hd),
                                            dtype=cdt, device=dev)
            self._grad_vh_buf = torch.zeros((s_max, self.Hkv, self.Hd),
                                            dtype=cdt, device=dev)
        else:
            self._grad_qh_buf = self._grad_kh_buf = self._grad_vh_buf = None

        # [Phase 5] Optional CUDA-graph runner. Per-layer FFN-bwd + padded-attn-bwd
        # graphs, captured lazily on first replay. Falls back per-layer on capture
        # or shape-fit failure (gradient values unchanged in either case).
        if meta.get("backward_cuda_graph"):
            from vllm.deltaserve.bwd_services.llama3_graph import (
                Llama3GraphedBackward,
            )
            s_max = int(meta.get("max_saved_finetuning_tokens", 0))
            bn_max = int(meta.get("backward_cuda_graph_attn_bn_max", 8))
            l_max = int(meta.get("backward_cuda_graph_attn_l_max", 64))
            if s_max <= 0:
                dprint(
                    "[backward] backward_cuda_graph requested but "
                    "max_saved_finetuning_tokens not provided; keeping eager")
            else:
                try:
                    self.graph_runner = Llama3GraphedBackward(
                        self, s_max=s_max, bn_max=bn_max, l_max=l_max)
                except Exception as e:  # noqa: BLE001
                    import traceback
                    dprint(f"[backward] graph runner init failed: {e}; "
                           f"falling back to eager backward")
                    traceback.print_exc()
                    self.graph_runner = None

    def _layer_weights(self, i: int) -> dict:
        """Base (compute dtype) + fp32 LoRA params for layer i, as one dict."""
        lw = dict(self.base[i])
        ld = self.lora.get(i, {})
        for proj in _PROJ:
            lw[proj + "A"] = ld.get(proj, {}).get("A")
            lw[proj + "B"] = ld.get(proj, {}).get("B")
        return lw

    # -- the real backward (overrides base.process_backward) ------------------

    def process_backward(self, activations, sample_lens, n, epoch):
        if not self._built:
            raise RuntimeError("llama3 backward state not built (weights not shared)")
        # Mode tag for the one-line cycle log (set in BackwardService dispatcher).
        self._last_mode = "graph" if self.graph_runner is not None else "eager"
        seq_lens = [int(s) for s in sample_lens]
        b_start, acc = [], 0
        for s in seq_lens:
            b_start.append(acc)
            acc += s
        device = activations["final_in"].device
        positions = torch.cat([
            torch.arange(s, device=device) for s in seq_lens]) if seq_lens else \
            torch.zeros(0, device=device)
        cos, sin = rope_cos_sin(positions, self.Hd, self.theta)
        ids = activations["concat_input_ids"][:n]

        self.optimizer.zero_grad(set_to_none=True)

        # Head: loss + grad w.r.t. final_in (pre-final-norm residual = layer_in[L]).
        loss, n_valid, g = head_backward(
            activations["final_in"][:n], self.lm_w, self.norm_w, self.eps,
            ids, seq_lens, b_start, self.vocab)

        # Saved MLP pre-activations (gate||up) per layer, if captured in the forward
        # — lets the remat skip the gate_up matmul (the layer's biggest recompute).
        saved_gu = activations.get("mlp_gate_up")

        # If the graphed runner is attached, build per-backward scatter indices
        # + key-pad mask once (re-used across all L layers' padded-attention
        # replays). Eager fallback for batches that overflow the padded budget.
        if self.graph_runner is not None:
            self.graph_runner.begin_backward(n, seq_lens, b_start)

        # Per-layer manual backward, chaining the input gradient down the stack.
        for i in reversed(range(self.L)):
            if self.graph_runner is None:
                # Eager path (today's behaviour). Pause at the layer boundary
                # before the layer's compute — original yield cadence.
                self._maybe_pause()
            lw = self._layer_weights(i)
            x = activations["layer_in"][i][:n]
            gu = saved_gu[i][:n] if saved_gu else None
            with torch.no_grad():
                cache = layer_forward(x, lw, self.scaling, cos, sin,
                                      seq_lens, b_start, self.dims, self.eps,
                                      saved_gate_up=gu)
                if self.graph_runner is None:
                    grad_x, grads = layer_backward(
                        g, cache, lw, self.scaling, cos, sin,
                        seq_lens, b_start, self.dims, self.eps,
                        cdt=self.bwd_dtype,
                        grad_qh_buf=self._grad_qh_buf,
                        grad_kh_buf=self._grad_kh_buf,
                        grad_vh_buf=self._grad_vh_buf)
                else:
                    grad_x, grads = self._layer_backward_graphed(
                        i, g, cache, lw, cos, sin, seq_lens, b_start)
            # Write grads to the fp32 master params (cast up from the bulk compute
            # dtype; PEFT layout matches), then per-layer clip to 1.0 (DeltaServe).
            ld = self.lora.get(i, {})
            layer_params = []
            for proj in _PROJ:
                pa, pb = ld.get(proj, {}).get("A"), ld.get(proj, {}).get("B")
                if pa is not None:
                    pa.grad = grads[proj + "A"].float()
                    pb.grad = grads[proj + "B"].float()
                    layer_params += [pa, pb]
            if layer_params:
                torch.nn.utils.clip_grad_norm_(layer_params, max_norm=1.0)
            g = grad_x

        self.optimizer.step()
        if epoch > self.current_epoch:
            self.scheduler.step()
            self.current_epoch = epoch

        # Publish the updated fp32 master into vLLM's served LoRA buffers (the exact
        # tensors inference reads). Safe with no locking: FT admission is closed for
        # the whole backward, so the adapter is idle until the done-reply reopens it.
        self._publish_to_served()

        return loss, n_valid

    @torch.no_grad()
    def _layer_backward_graphed(self, i, g, cache, lw, cos, sin,
                                seq_lens, b_start):
        """Graphed per-layer backward: Graph A (FFN-bwd) → eager O-bwd → pause
        (yield GPU to inference) → Graph B (padded-attn-bwd) → eager tail
        (RoPE / Q-K-V proj / in_ln rmsnorm). Same gradient values as the eager
        ``layer_backward``; differs only in *when* host dispatch happens.

        Pause cadence: once per layer, between the two graphs — the eager
        path's once-per-layer pause-at-start is preserved, just relocated to
        the mid-layer point so each graph runs without yielding (the captured
        region can't host an mp.Event.wait anyway)."""
        cdt = self.bwd_dtype
        Hq, _, Hd, kv_size = self.dims
        n = g.shape[0]
        D = Hq * Hd

        # --- Graph A: FFN-bwd (frozen MLP, with residual) ---
        grad_resid_mid = self.graph_runner.ffn_backward(i, g, cache, lw)

        # --- Eager: O-projection backward (LoRA grad lives here) ---
        grad_ctx_flat, grad_oA, grad_oB = _proj_backward(
            cache["ctx_flat"], grad_resid_mid, lw["o"],
            lw["oA"], lw["oB"], self.scaling, cdt)

        # --- Yield GPU to inference between graphs (preserves the load-bearing
        # per-layer pause cadence; mp.Event.wait can't run inside a graph). ---
        self._maybe_pause()

        # --- Graph B: padded-attention backward CORE ---
        grad_ctx = grad_ctx_flat.view(n, Hq, Hd)
        grad_qh, grad_kh, grad_vh = self.graph_runner.attn_backward(
            i, cache["qh"], cache["kh"], cache["vh"], grad_ctx,
            seq_lens, b_start, self.dims)

        # --- Eager tail: RoPE bwd → Q/K/V proj bwd → in_ln rmsnorm bwd ---
        grad_q = rope_backward(grad_qh, cos, sin).reshape(n, D)
        grad_k = rope_backward(grad_kh, cos, sin).reshape(n, kv_size)
        grad_v = grad_vh.reshape(n, kv_size)
        xn1 = cache["x_norm1"]
        gx_q, grad_qA, grad_qB = _proj_backward(
            xn1, grad_q, lw["q"], lw["qA"], lw["qB"], self.scaling, cdt)
        gx_k, grad_kA, grad_kB = _proj_backward(
            xn1, grad_k, lw["k"], lw["kA"], lw["kB"], self.scaling, cdt)
        gx_v, grad_vA, grad_vB = _proj_backward(
            xn1, grad_v, lw["v"], lw["vA"], lw["vB"], self.scaling, cdt)
        grad_x_norm1 = gx_q + gx_k + gx_v
        grad_x = rmsnorm_backward(
            cache["x"], grad_x_norm1, lw["in_ln"], self.eps).to(cdt) + grad_resid_mid

        grads = {"qA": grad_qA, "qB": grad_qB, "kA": grad_kA, "kB": grad_kB,
                 "vA": grad_vA, "vB": grad_vB, "oA": grad_oA, "oB": grad_oB}
        return grad_x, grads

    @torch.no_grad()
    def _publish_to_served(self) -> None:
        """Write the trained fp32 master into vLLM's served LoRA stacked buffers
        (DeltaServe's load-time refresh): per (layer, proj), clamp + cast to the
        served dtype, with α/r baked into B (vLLM applies scale=1). No transpose —
        PEFT A [r,in] / B [out,r] match vLLM's stacked layout.

        The scaling convention is load-bearing: vLLM's punica forward calls
        ``add_lora_linear(..., 1.0, ...)`` (see
        ``vllm/lora/layers/base_linear.py:226``) — the third positional arg
        ``scale`` is hardcoded to 1.0, NOT taken from the adapter's stored
        ``LoRALayerWeights.scaling`` (alpha/r). We multiply ``pb * self.scaling``
        here so the net inference effect is
        ``(x @ A) @ (B·s) · 1.0 = s · (x · A · B)`` — the correct PEFT LoRA
        forward with one scaling application. If vLLM ever switches that
        callsite to pass the per-adapter scaling instead of 1.0, this publish
        MUST drop the ``* self.scaling`` multiplication or inference will see
        ``scaling²`` and silently degrade output quality.

        Locking: relies on the FT adapter being pre-loaded into a dedicated
        served-LoRA slot at startup (see ``_maybe_share_ft_served_lora`` in
        ``v1/worker/gpu_worker.py``). No inference request can land on this
        slot, so the in-place write is race-free with concurrent inference.
        FT admission is closed for the entire backward cycle (coordinator
        keeps ``pending_backward=True`` until the child acks), so even FT
        requests can't read the slot mid-publish."""
        if not self.lora_buffers:
            return
        slot = int(self.lora_buffers["slot"])
        for i, projd in self.lora_buffers["layers"].items():
            ld = self.lora.get(int(i), {})
            for proj, buf in projd.items():
                pa = ld.get(proj, {}).get("A")
                pb = ld.get(proj, {}).get("B")
                if pa is None:
                    continue
                a_buf, b_buf = buf["a"], buf["b"]
                r, in_dim = pa.shape          # PEFT A [r, in]
                out_dim = pb.shape[0]          # PEFT B [out, r]
                a_buf[slot, 0, :r, :in_dim].copy_(
                    pa.clamp(-6.5e4, 6.5e4).to(a_buf.dtype))
                b_buf[slot, 0, :out_dim, :r].copy_(
                    (pb * self.scaling).clamp(-6.5e4, 6.5e4).to(b_buf.dtype))

    # -- capture correctness checks (debug-gated) -----------------------------

    def verify_activations(self, activations: dict, n: int) -> None:
        meta = self.shared["meta"]
        base = self.shared["base"] or {}
        ids = activations["concat_input_ids"][:n].long()
        layer_in = activations.get("layer_in") or []
        embed_w = base.get(meta.get("embed_weight_key"))
        if layer_in and embed_w is not None:
            d = (layer_in[0][:n].float() - embed_w[ids].float()).abs().max().item()
            scale = embed_w[ids].float().abs().max().item() + 1e-6
            dprint(f"[verify] layer_in[0] vs embed[ids]: max|Δ|={d:.3e} "
                   f"rel={d / scale:.3e} {'OK' if d / scale < _TOL else 'FAIL'}")
        final_in = activations.get("final_in")
        final_hidden = activations.get("final_hidden")
        norm_w = base.get(meta.get("norm_weight_key"))
        if final_in is not None and final_hidden is not None and norm_w is not None:
            h = rmsnorm(final_in[:n], norm_w, float(meta.get("rms_norm_eps", 1e-5)))
            fh = final_hidden[:n].float()
            d = (h.float() - fh).abs().max().item()
            scale = fh.abs().max().item() + 1e-6
            dprint(f"[verify] RMSNorm(final_in) vs final_hidden: max|Δ|={d:.3e} "
                   f"rel={d / scale:.3e} {'OK' if d / scale < _TOL else 'FAIL'}")
