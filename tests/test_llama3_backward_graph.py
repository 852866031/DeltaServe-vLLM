#!/usr/bin/env python
"""Parity gate for the CUDA-graph backward (Phase 5).

Confirms that ``Llama3GraphedBackward`` (per-layer FFN graph + padded-attention
graph) produces gradients bit-identical to the eager core helpers
(``ffn_backward_core`` + ``attn_backward_core``) over a range of seq_lens at a
fixed ``s_max`` (max_saved_finetuning_tokens).

Requires CUDA. Mock service object stands in for ``Llama3BackwardService`` —
the graph runner only reads dims/dtypes/device off it.

    python tests/test_llama3_backward_graph.py
"""

import os
import sys
from types import SimpleNamespace

import torch

sys.path[:] = [p for p in sys.path
               if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]

from vllm.deltaserve.bwd_services import llama3 as L  # noqa: E402
from vllm.deltaserve.bwd_services.llama3_graph import (  # noqa: E402
    Llama3GraphedBackward,
)

if not torch.cuda.is_available():
    print("CUDA not available — skipping graphed-backward parity test")
    sys.exit(0)

DEVICE = torch.device("cuda")
MDT = torch.bfloat16        # model dtype (matches Llama-3 forward)
_TOL = 5e-3                 # bf16 round-trip tolerance for the bulk-cdt paths
_FP32_TOL = 1e-5            # tighter tolerance when both paths run in fp32
_passed = 0
_failed = 0


def _check(name, manual, ref, tol):
    global _passed, _failed
    diff = (manual.float() - ref.float()).abs().max().item()
    scale = ref.float().abs().max().item() + 1e-8
    rel = diff / scale
    ok = rel < tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:34s} max-rel-err={rel:.3e}")
    _passed += ok
    _failed += (not ok)


def _make_layer_weights(D, kv_size, inter, r, dtype):
    """Synthetic frozen base weights + tiny LoRA params. Same naming as the
    real `_layer_weights` dict in Llama3BackwardService."""
    def w(*shape):
        return (torch.randn(*shape, device=DEVICE) * 0.1).to(dtype)
    def p(*shape):
        return torch.nn.Parameter((torch.randn(*shape, device=DEVICE) * 0.1).float())
    return {
        "q": w(D, D), "k": w(kv_size, D), "v": w(kv_size, D), "o": w(D, D),
        "gate": w(inter, D), "up": w(inter, D), "down": w(D, inter),
        "in_ln": (torch.randn(D, device=DEVICE).abs() + 0.5).to(dtype),
        "post_ln": (torch.randn(D, device=DEVICE).abs() + 0.5).to(dtype),
        "qA": p(r, D), "qB": p(D, r),
        "kA": p(r, D), "kB": p(kv_size, r),
        "vA": p(r, D), "vB": p(kv_size, r),
        "oA": p(r, D), "oB": p(D, r),
    }


def _mk_svc(D, L_, Hq, Hkv, Hd, inter, eps, mdt, cdt, lws=None):
    """Minimal stand-in for Llama3BackwardService — the runner reads dims/dtypes
    and calls ``_layer_weights(i)`` during startup graph capture. If ``lws`` is
    provided, ``_layer_weights`` returns ``lws[i]``; otherwise it raises so the
    runner's startup capture path can't accidentally succeed without weights."""
    svc = SimpleNamespace(
        device_index=0,
        D=D, L=L_, Hq=Hq, Hkv=Hkv, Hd=Hd, inter=inter,
        kv_size=Hkv * Hd, eps=eps,
        base_dtype=mdt, bwd_dtype=cdt,
    )
    if lws is None:
        def _missing(i):
            raise RuntimeError("test forgot to attach lws")
        svc._layer_weights = _missing
    else:
        svc._layer_weights = lambda i: lws[i]
    return svc


def _make_inputs(seq_lens, D, Hq, Hkv, Hd, inter, mdt, cdt, s_max):
    """Build (cache, g, grad_ctx) for one synthetic layer at the requested
    seq_lens. Pads nothing — `n = sum(seq_lens)`, can be < s_max."""
    n = sum(seq_lens)
    return (
        {
            "x": torch.randn(n, D, device=DEVICE, dtype=mdt),
            "x_norm1": torch.randn(n, D, device=DEVICE, dtype=mdt),
            "qh": torch.randn(n, Hq, Hd, device=DEVICE, dtype=mdt),
            "kh": torch.randn(n, Hkv, Hd, device=DEVICE, dtype=mdt),
            "vh": torch.randn(n, Hkv, Hd, device=DEVICE, dtype=mdt),
            "ctx_flat": torch.randn(n, Hq * Hd, device=DEVICE, dtype=mdt),
            "resid_mid": torch.randn(n, D, device=DEVICE, dtype=mdt),
            "gate": torch.randn(n, inter, device=DEVICE, dtype=mdt),
            "up": torch.randn(n, inter, device=DEVICE, dtype=mdt),
        },
        torch.randn(n, D, device=DEVICE, dtype=cdt),
        torch.randn(n, Hq, Hd, device=DEVICE, dtype=cdt),
    )


def test_ffn_graph_parity():
    """FFN-bwd graph replay must match the eager ffn_backward_core."""
    print("test_ffn_graph_parity:")
    torch.manual_seed(0)
    Hq, Hkv, Hd = 4, 2, 16
    D, kv_size, inter, r = Hq * Hd, Hkv * Hd, 32, 4
    L_, eps, cdt = 3, 1e-5, torch.float32  # fp32 so we can use tight tolerance
    s_max = 16

    # Per-layer weights (capture binds to these references; built before the
    # runner so its startup prepare() can find them via svc._layer_weights).
    lws = [_make_layer_weights(D, kv_size, inter, r, MDT) for _ in range(L_)]
    svc = _mk_svc(D, L_, Hq, Hkv, Hd, inter, eps, MDT, cdt, lws=lws)
    runner = Llama3GraphedBackward(svc, s_max=s_max, bn_max=4, l_max=8)

    # Run two batches with different n / seq_lens — same captured graphs,
    # different staged inputs. begin_backward sets indices; ffn graph is
    # independent of seq_lens so it must match for any n ≤ s_max.
    for batch_idx, seq_lens in enumerate([[5, 3], [4, 4, 4]]):
        n = sum(seq_lens)
        b_start = [sum(seq_lens[:i]) for i in range(len(seq_lens))]
        runner.begin_backward(n, seq_lens, b_start)
        for i in range(L_):
            lw = lws[i]
            cache, g, _ = _make_inputs(seq_lens, D, Hq, Hkv, Hd, inter,
                                       MDT, cdt, s_max)
            # Eager reference
            ref = L.ffn_backward_core(g, cache, lw, eps, cdt)
            # Graphed
            out = runner.ffn_backward(i, g, cache, lw)
            _check(f"batch{batch_idx} layer{i} grad_resid_mid",
                   out[:n], ref, _FP32_TOL)


def test_attn_graph_parity():
    """Padded-attention graph replay must match the eager attn_backward_core.
    Verified at seq_lens that FIT inside (bn_max, l_max)."""
    print("test_attn_graph_parity:")
    torch.manual_seed(1)
    Hq, Hkv, Hd = 4, 2, 16
    D, kv_size, inter, r = Hq * Hd, Hkv * Hd, 32, 4
    L_, eps, cdt = 2, 1e-5, torch.float32
    s_max, bn_max, l_max = 16, 4, 8
    dims = (Hq, Hkv, Hd, kv_size)

    lws = [_make_layer_weights(D, kv_size, inter, r, MDT) for _ in range(L_)]
    svc = _mk_svc(D, L_, Hq, Hkv, Hd, inter, eps, MDT, cdt, lws=lws)
    runner = Llama3GraphedBackward(svc, s_max=s_max, bn_max=bn_max, l_max=l_max)

    # Two fitting batches — re-uses the same per-layer attn graphs.
    for batch_idx, seq_lens in enumerate([[5, 3], [8, 8, 8, 8]]):
        n = sum(seq_lens)
        b_start = [sum(seq_lens[:i]) for i in range(len(seq_lens))]
        runner.begin_backward(n, seq_lens, b_start)
        for i in range(L_):
            cache, _, grad_ctx = _make_inputs(seq_lens, D, Hq, Hkv, Hd, inter,
                                              MDT, cdt, s_max)
            # Eager reference
            ref_qh, ref_kh, ref_vh = L.attn_backward_core(
                cache["qh"], cache["kh"], cache["vh"], grad_ctx,
                seq_lens, b_start, dims, cdt)
            # Graphed
            out_qh, out_kh, out_vh = runner.attn_backward(
                i, cache["qh"], cache["kh"], cache["vh"], grad_ctx,
                seq_lens, b_start, dims)
            _check(f"batch{batch_idx} layer{i} grad_qh",
                   out_qh[:n], ref_qh, _FP32_TOL)
            _check(f"batch{batch_idx} layer{i} grad_kh",
                   out_kh[:n], ref_kh, _FP32_TOL)
            _check(f"batch{batch_idx} layer{i} grad_vh",
                   out_vh[:n], ref_vh, _FP32_TOL)


def test_attn_overflow_fallback():
    """Batches that overflow (bn_max, l_max) must silently fall back to the
    eager core (and still produce the correct result)."""
    print("test_attn_overflow_fallback:")
    torch.manual_seed(2)
    Hq, Hkv, Hd = 4, 2, 16
    D, kv_size, inter, r = Hq * Hd, Hkv * Hd, 32, 4
    L_, eps, cdt = 1, 1e-5, torch.float32
    s_max, bn_max, l_max = 32, 2, 4
    dims = (Hq, Hkv, Hd, kv_size)

    lws = [_make_layer_weights(D, kv_size, inter, r, MDT) for _ in range(L_)]
    svc = _mk_svc(D, L_, Hq, Hkv, Hd, inter, eps, MDT, cdt, lws=lws)
    runner = Llama3GraphedBackward(svc, s_max=s_max, bn_max=bn_max, l_max=l_max)

    # Overflow on l: max sample length 6 > l_max=4 → fallback.
    seq_lens = [6, 4]
    n = sum(seq_lens)
    b_start = [0, 6]
    runner.begin_backward(n, seq_lens, b_start)
    cache, _, grad_ctx = _make_inputs(seq_lens, D, Hq, Hkv, Hd, inter,
                                      MDT, cdt, s_max)
    ref_qh, ref_kh, ref_vh = L.attn_backward_core(
        cache["qh"], cache["kh"], cache["vh"], grad_ctx,
        seq_lens, b_start, dims, cdt)
    out_qh, out_kh, out_vh = runner.attn_backward(
        0, cache["qh"], cache["kh"], cache["vh"], grad_ctx,
        seq_lens, b_start, dims)
    _check("overflow grad_qh", out_qh, ref_qh, _FP32_TOL)
    _check("overflow grad_kh", out_kh, ref_kh, _FP32_TOL)
    _check("overflow grad_vh", out_vh, ref_vh, _FP32_TOL)
    assert not runner._attn_fit, "expected _attn_fit=False for overflow batch"


def main():
    test_ffn_graph_parity()
    test_attn_graph_parity()
    test_attn_overflow_fallback()
    print(f"\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
