# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Elementary ops of the manual LoRA-SFT backward: RoPE, RMSNorm, and the
LoRA-augmented linear projection, each with its hand-derived backward.

Everything operates in the dtype of its inputs, except the RMSNorm internals
which upcast to fp32 (the DeltaServe precision contract). The gradcheck tests
pass fp32 throughout. ``rmsnorm`` / ``rmsnorm_backward`` reduce over the last
dim only, so they apply unchanged to per-head ``[n, H, Hd]`` tensors (Qwen3's
q/k-norm) as well as to the ``[n, hidden]`` residual stream.
"""

import torch
import torch.nn.functional as F


import math

# RoPE variants the backward reproduces exactly (same frequencies as vLLM's
# served rotary embedding). Anything else is refused at startup
# (``ft_meta.rope_scaling_of``) instead of training with the wrong positions.
SUPPORTED_ROPE_TYPES = ("default", "linear", "llama3")


def rope_inv_freq(head_dim: int, theta: float, rope_scaling: dict | None = None,
                  device=None) -> torch.Tensor:
    """Inverse frequencies [head_dim//2] fp32, with the model's RoPE scaling.

    ``rope_scaling`` is the normalized dict from ``ft_meta.rope_scaling_of``
    (``None`` / ``rope_type == "default"`` → plain RoPE). ``llama3`` is the
    Llama-3.1/3.2/3.3 frequency-dependent rescale, written as vLLM's
    ``Llama3RotaryEmbedding._compute_inv_freq``; ``linear`` divides every
    frequency by ``factor`` (vLLM divides the positions — same angles)."""
    inv = 1.0 / (theta ** (
        torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        / head_dim))
    rtype = (rope_scaling or {}).get("rope_type", "default")
    if rtype in (None, "default"):
        return inv
    if rtype == "linear":
        return inv / float(rope_scaling["factor"])
    if rtype == "llama3":
        factor = float(rope_scaling["factor"])
        low = float(rope_scaling["low_freq_factor"])
        high = float(rope_scaling["high_freq_factor"])
        orig = float(rope_scaling["original_max_position_embeddings"])
        low_wavelen, high_wavelen = orig / low, orig / high
        wavelen = 2 * math.pi / inv
        smooth = ((orig / wavelen - low) / (high - low)) if low != high else 0
        return torch.where(
            wavelen < high_wavelen, inv,
            torch.where(wavelen > low_wavelen, inv / factor,
                        (1 - smooth) * inv / factor + smooth * inv))
    raise NotImplementedError(
        f"RoPE type {rtype!r} is not supported by the DeltaServe backward "
        f"(supported: {', '.join(SUPPORTED_ROPE_TYPES)})")


def rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float,
                 rope_scaling: dict | None = None):
    """NeoX cos/sin for the given positions. Returns (cos, sin) [n, head_dim//2] fp32."""
    inv_freq = rope_inv_freq(head_dim, theta, rope_scaling, device=positions.device)
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


def proj(xin, w_base, A, B, scaling):
    """y = xin @ w_base.T + scaling·(xin @ A.T) @ B.T. w_base [out,in], A [r,in], B [out,r].

    The LoRA master is fp32; cast its compute copy to xin's dtype (bf16 in prod,
    fp32 in the gradcheck) so the matmul dtypes match (DeltaServe fp32-master /
    low-precision-compute rule)."""
    y = F.linear(xin, w_base)
    if A is not None:
        y = y + scaling * F.linear(F.linear(xin, A.to(xin.dtype)), B.to(xin.dtype))
    return y


def proj_backward(xin, gy, w_base, A, B, scaling, cdt=torch.float32):
    """Grad of ``proj``, computed in the bulk compute dtype ``cdt`` (bf16 in prod,
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
