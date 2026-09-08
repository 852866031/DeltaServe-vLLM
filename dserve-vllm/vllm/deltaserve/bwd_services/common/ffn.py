# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SwiGLU FFN block (frozen MLP, post-attention RMSNorm, residual): the forward
tail the layer remat needs and the backward core.

The FFN is never trained (LoRA targets are q/k/v/o only), so the forward only
has to produce the ``gate`` / ``up`` pre-activations the backward reads — and
skips even that when the FT forward captured them (``saved_gate_up``).
"""

import torch
import torch.nn.functional as F

from vllm.deltaserve.bwd_services.common.ops import rmsnorm, rmsnorm_backward


def ffn_forward_tail(resid_mid, lw, eps, saved_gate_up=None):
    """``(gate, up)`` for the backward. Uses the captured ``saved_gate_up``
    ([n, 2*intermediate] = gate||up) when given — skipping the gate_up matmul,
    the biggest recompute in the layer — else recomputes from ``resid_mid``."""
    if saved_gate_up is not None:
        inter = saved_gate_up.shape[-1] // 2
        return saved_gate_up[:, :inter], saved_gate_up[:, inter:]
    x_norm2 = rmsnorm(resid_mid, lw["post_ln"], eps)
    return F.linear(x_norm2, lw["gate"]), F.linear(x_norm2, lw["up"])


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
