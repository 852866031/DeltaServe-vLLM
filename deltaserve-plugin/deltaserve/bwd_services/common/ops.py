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
