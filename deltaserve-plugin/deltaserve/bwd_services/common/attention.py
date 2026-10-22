# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-sample GQA causal attention: eager forward and backward cores.

Both loop over the samples of the packed batch (``seq_lens`` / ``b_start``),
so shapes vary per sample and neither is CUDA-graph-capturable; the graphed
runner (``common/graph.py``) has padded rewrites of the same math and falls
back to these when a batch overflows the padded bounds. Scores/softmax and
the dQ/dK/dV products always run in fp32 (the load-bearing GQA precision rule;
downgrading them plateaued Llama-3 loss in DeltaServe). ``dims`` is
``(Hq, Hkv, Hd, kv_size)``.
"""

import math

import torch


def attn_forward_core(qh, kh, vh, seq_lens, b_start, dims, out_dtype):
    """Causal GQA attention over the packed batch. ``qh`` [n, Hq, Hd],
    ``kh``/``vh`` [n, Hkv, Hd] (post-RoPE). Returns the flat context
    ``[n, Hq*Hd]`` in ``out_dtype`` — the o_proj input."""
    Hq, Hkv, Hd, _ = dims
    n = qh.shape[0]
    scale = 1.0 / math.sqrt(Hd)
    kv_repeat = Hq // Hkv
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
        mask = torch.triu(
            torch.ones(ln, ln, dtype=torch.bool, device=qh.device), 1)
        scores = scores.masked_fill(mask, -1e9)
        att = torch.softmax(scores, dim=-1)         # [Hq, L, L] fp32
        ctx_blk = (att @ v_rep.float()).to(out_dtype).transpose(0, 1)  # [L, Hq, Hd]
        ctx_blocks.append(ctx_blk)
    return torch.cat(ctx_blocks, 0).reshape(n, Hq * Hd)


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
    # eager-fallback inside GraphedBackward.attn_backward also passes its own
    # static_grad_* (also s_max). The undersized branch fires only in contrived
    # parity tests that exceed s_max — we silently alloc fresh.
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
