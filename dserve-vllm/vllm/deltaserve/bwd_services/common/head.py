# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Final RMSNorm + LM head: per-sample shifted cross-entropy loss and the
gradient w.r.t. the pre-final-norm residual (``final_in``).

The LM head runs in fp32 (the precision contract): fp32 inputs, fp32
accumulation, fp32 softmax / cross-entropy. The head weight is shared from
vLLM in the model dtype (bf16), so each vocab chunk is converted to fp32 —
ONCE per pass per cycle — right before its GEMM. All samples' predicting
rows are batched into one GEMM per chunk: the loss and gradient are row-wise
independent, so the per-sample structure (shift-by-one targets, ``n_valid``
normalisation) is pure row indexing and batching changes no arithmetic
beyond the GEMM tile order. Two passes over the head per cycle (logits, then
``logit_grad @ W``), ~2 × the head's bytes of conversion traffic.

Before this restructure the head was computed per sample: 8 conversions of
the whole head + 8 GEMMs with ~31 rows each, which on Qwen3-14B cost ~63 ms
of a ~165 ms uncontended cycle (profiled 2026-09-08). The head is frozen;
only ``grad_final_in`` flows back into the layer stack.
"""

import torch
import torch.nn.functional as F

from vllm.deltaserve.bwd_services.common.ops import rmsnorm, rmsnorm_backward

VOCAB_CHUNK = 16384


def logits_chunked(h: torch.Tensor, lm_w: torch.Tensor, vocab: int):
    """fp32 logits = h @ lm_w[:vocab].T, computed in vocab chunks (bounds the fp32
    LM-head temporary). h [m, D], lm_w [vocab_pad, D]. Returns [m, vocab] fp32."""
    hf = h.float()
    out = hf.new_empty((hf.shape[0], vocab))
    for c in range(0, vocab, VOCAB_CHUNK):
        e = min(c + VOCAB_CHUNK, vocab)
        out[:, c:e] = hf @ lm_w[c:e].float().t()
    return out


def head_backward(final_in, lm_w, norm_w, eps, ids, seq_lens, b_start, vocab,
                  pause_fn=None):
    """LM-head + final-norm: per-sample shift CE loss + grad w.r.t. final_in.
    Returns (loss: float, n_valid: int, grad_final_in [n,D] fp32).

    ``pause_fn`` (the service's ``_maybe_pause``) is called once per vocab
    chunk in both passes: the head is ~18 ms on Qwen3-14B and used to be one
    uninterruptible block at the very start of the cycle — exactly where the
    first inference prefill after the buffer-full trigger lands.

    Rows ``st .. st+ln-2`` of each sample predict ids ``st+1 .. st+ln-1``;
    samples with ``ln < 2`` contribute nothing. The loss is the SUM of the
    per-row cross-entropies divided by ``n_valid`` (the number of predicting
    rows), and the gradient is ``(softmax - onehot) / n_valid`` — identical to
    the original per-sample loop, computed for all rows at once."""
    # rmsnorm computes in fp32 but returns the input dtype (bf16 in prod);
    # the head contract is fp32 end to end, so upcast the rows once here.
    normed = rmsnorm(final_in, norm_w, eps)
    n, D = final_in.shape
    device = final_in.device
    rows, tgts = [], []
    for st, ln in zip(b_start, seq_lens):
        if ln < 2:
            continue
        rows.append(torch.arange(st, st + ln - 1, device=device))
        tgts.append(ids[st + 1:st + ln].long())
    n_valid = sum(int(r.numel()) for r in rows)
    if n_valid == 0:
        return 0.0, 0, final_in.new_zeros((n, D), dtype=torch.float32)
    rows = torch.cat(rows)
    tgt = torch.cat(tgts)
    h = normed[rows].float()                          # [m, D] fp32, m = n_valid

    # Pass 1: fp32 logits for every predicting row, one GEMM per vocab chunk
    # (the bf16 head chunk converted once here).
    logits = h.new_empty((n_valid, vocab))
    for c in range(0, vocab, VOCAB_CHUNK):
        if pause_fn is not None:
            pause_fn()
        e = min(c + VOCAB_CHUNK, vocab)
        logits[:, c:e] = h @ lm_w[c:e].float().t()
    total_loss = F.cross_entropy(logits, tgt, reduction="sum")
    p = torch.softmax(logits, dim=-1)
    del logits
    p[torch.arange(n_valid, device=device), tgt] -= 1.0
    p /= n_valid                                      # dL/dlogits, [m, vocab]

    # Pass 2: grad w.r.t. the normed rows = p @ W, chunked over vocab (the
    # head chunk converted once again); scatter back to the full [n, D].
    grad_rows = p.new_zeros((n_valid, D))
    for c in range(0, vocab, VOCAB_CHUNK):
        if pause_fn is not None:
            pause_fn()
        e = min(c + VOCAB_CHUNK, vocab)
        grad_rows += p[:, c:e] @ lm_w[c:e].float()
    grad_normed = grad_rows.new_zeros((n, D))         # fp32 [n, D]
    grad_normed[rows] = grad_rows
    grad_final_in = rmsnorm_backward(final_in, grad_normed, norm_w, eps)
    return float(total_loss.item() / n_valid), n_valid, grad_final_in
