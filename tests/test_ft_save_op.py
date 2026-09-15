#!/usr/bin/env python
"""mixed-fwd-cuda-graph — the graph-capturable activation save op.

GPU (one device). Exercises ``torch.ops.vllm.dserve_save_rows`` the way the
has_ft piecewise graphs use it: capture once with the persistent index
tensors, then replay with DIFFERENT index contents and check the saved rows
follow the contents, not the capture. Also: a capture with ``ft_save=None``
records no work, interleaved FT rows, the residual sum, the scratch row, and
``FinetuneAccumulator.arm_op_step`` end to end on a fake vLLM-named model
that carries the call sites.

    python tests/test_ft_save_op.py
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bwd_harness as H  # noqa: E402

sys.path[:] = [p for p in sys.path
               if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]

from vllm.deltaserve.accumulate import FinetuneAccumulator  # noqa: E402
from vllm.deltaserve.ft_save_op import FtSaveState, maybe_save  # noqa: E402
from vllm.forward_context import ForwardContext, override_forward_context  # noqa: E402

C = H.Checker(tol=1e-6)
DEV = torch.device("cuda")
D, INTER, NL, S_MAX = 8, 12, 2, 16
HQ, HKV, HD = 2, 1, 4


def _ctx(ft_save):
    return ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={},
                          ft_save=ft_save)


def test_capture_follows_index_contents():
    print("test_capture_follows_index_contents:")
    n_tok = 32
    buf = torch.zeros(S_MAX + 1, D, device=DEV, dtype=torch.bfloat16)
    src = torch.zeros(S_MAX, device=DEV, dtype=torch.int64)
    dst = torch.full((S_MAX,), S_MAX, device=DEV, dtype=torch.int64)
    marker = torch.zeros(1, device=DEV, dtype=torch.int32)
    x = torch.randn(n_tok, D, device=DEV, dtype=torch.bfloat16)
    r = torch.randn(n_tok, D, device=DEV, dtype=torch.bfloat16)
    state = FtSaveState(src=src, dst=dst, bufs=[buf, None])

    def run(slot):
        torch.ops.vllm.dserve_save_rows(x, r, marker, slot)

    s = torch.cuda.Stream()
    with override_forward_context(_ctx(state)), torch.cuda.stream(s):
        for _ in range(2):
            run(0)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            run(0)
    torch.cuda.synchronize()

    # Replay 1: interleaved rows 3, 7, 20 → buffer rows 5, 6, 7.
    pos = [3, 7, 20]
    src.copy_(torch.tensor(pos + [0] * (S_MAX - 3), device=DEV))
    dst.copy_(torch.tensor([5, 6, 7] + [S_MAX] * (S_MAX - 3), device=DEV))
    buf.zero_()
    g.replay(); torch.cuda.synchronize()
    want = (x[pos].float() + r[pos].float()).to(torch.bfloat16)
    C.check("replay 1: rows 5..7 = x+r at 3,7,20", buf[5:8].float(), want.float())
    C.ok("replay 1: rows outside untouched", float(buf[:5].abs().sum() + buf[8:S_MAX].abs().sum()) == 0.0)

    # Replay 2: different positions AND offset, same graph.
    pos = [31, 0]
    src.copy_(torch.tensor(pos + [0] * (S_MAX - 2), device=DEV))
    dst.copy_(torch.tensor([S_MAX - 1, 12] + [S_MAX] * (S_MAX - 2), device=DEV))
    buf.zero_()
    g.replay(); torch.cuda.synchronize()
    C.check("replay 2: row 15 = x+r at 31", buf[S_MAX - 1].float(),
            (x[31].float() + r[31].float()).to(torch.bfloat16).float())
    C.check("replay 2: row 12 = x+r at 0", buf[12].float(),
            (x[0].float() + r[0].float()).to(torch.bfloat16).float())
    C.ok("replay 2: earlier rows 5..7 not rewritten", float(buf[5:8].abs().sum()) == 0.0)
    C.ok("scratch row absorbs the unused slots", True)

    # Disabled slot (None buffer) and a capture with ft_save=None: no work.
    buf.zero_()
    with override_forward_context(_ctx(state)):
        run(1)
    torch.cuda.synchronize()
    C.ok("None slot: nothing written", float(buf.abs().sum()) == 0.0)
    with override_forward_context(_ctx(None)), torch.cuda.stream(s):
        g0 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g0, stream=s):
            run(0)
    torch.cuda.synchronize()
    buf.zero_()
    with override_forward_context(_ctx(state)):
        g0.replay()
    torch.cuda.synchronize()
    C.ok("graph captured with ft_save=None replays to nothing", float(buf.abs().sum()) == 0.0)


class _AddNorm(nn.Module):
    def forward(self, x, residual=None):
        r = x if residual is None else x + residual
        return r * 0.5, r


class _AttnCore(nn.Module):
    """``self_attn.attn(q, k, v)`` — the module the reference hooks watch."""

    def forward(self, q, k, v):
        return q + 0.25


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = nn.Linear(D, HQ * HD + 2 * HKV * HD, bias=False)
        self.attn = _AttnCore()

    def forward(self, h):
        qkv = self.qkv_proj(h)
        q, k, v = qkv.split([HQ * HD, HKV * HD, HKV * HD], dim=-1)
        maybe_save(self, "q", q); maybe_save(self, "k", k); maybe_save(self, "v", v)
        ctx = self.attn(q, k, v)
        maybe_save(self, "ctx", ctx)
        return ctx


class _Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Linear(D, 2 * INTER, bias=False)
        self.down_proj = nn.Linear(INTER, D, bias=False)

    def forward(self, x):
        gu = self.gate_up_proj(x)
        maybe_save(self, "gate_up", gu)
        return self.down_proj(gu[:, :INTER] * gu[:, INTER:])


class _Layer(nn.Module):
    _dserve_save_points = True

    def __init__(self):
        super().__init__()
        self.input_layernorm = _AddNorm()
        self.self_attn = _Attn()
        self.post_attention_layernorm = _AddNorm()
        self.mlp = _Mlp()

    def forward(self, h, residual):
        maybe_save(self, "layer_in", h, residual)
        if residual is None:
            residual = h
            h, _ = self.input_layernorm(h)
        else:
            h, residual = self.input_layernorm(h, residual)
        h = self.self_attn(h)
        maybe_save(self, "resid_mid", h, residual)
        h, residual = self.post_attention_layernorm(h, residual)
        return self.mlp(h), residual


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer() for _ in range(NL)])
        self.norm = _AddNorm()

    def forward(self, h):
        residual = None
        for layer in self.layers:
            h, residual = layer(h, residual)
        maybe_save(self, "final_in", h, residual)
        out, _ = self.norm(h, residual)
        return out


class _Top(nn.Module):
    """vLLM naming: the decoder lives under ``model`` (so the final norm is
    ``model.norm`` and the layers ``model.layers.i``)."""

    def __init__(self):
        super().__init__()
        self.model = _Model()

    def forward(self, h):
        return self.model(h)


def test_accumulator_op_path_matches_hooks():
    print("test_accumulator_op_path_matches_hooks (fake model, graph replay vs hooks):")
    torch.manual_seed(0)
    model = _Top().to(DEV).to(torch.bfloat16)
    n_tok = 24
    x = torch.randn(n_tok, D, device=DEV, dtype=torch.bfloat16)

    # Reference: the hook path on a second accumulator (eager).
    ref = FinetuneAccumulator(model, S_MAX, D, DEV, torch.bfloat16,
                              intermediate_size=INTER, q_size=HQ * HD, kv_size=HKV * HD,
                              save_attn_qkv=True, save_attn_ctx=True, save_resid_mid=True,
                              attn_qkv_pre_transform=False)
    ref.register_hooks()
    mask = np.zeros(n_tok, dtype=bool)
    pos = [2, 9, 10, 17, 23]
    mask[pos] = True
    with override_forward_context(_ctx(None)):
        ref.begin_step(torch.from_numpy(mask).to(DEV), len(pos), offset=4, start=0,
                       contiguous=False)
        model(x)
        ref.end_step()
    for h in ref._handles:
        h.remove()
    torch.cuda.synchronize()

    # Op path: install, arm with the same positions/offset, capture, replay.
    acc = FinetuneAccumulator(model, S_MAX, D, DEV, torch.bfloat16,
                              intermediate_size=INTER, q_size=HQ * HD, kv_size=HKV * HD,
                              save_attn_qkv=True, save_attn_ctx=True, save_resid_mid=True,
                              attn_qkv_pre_transform=False)
    C.ok("install_save_points on a family with call sites", acc.install_save_points(model))
    C.ok("slot table has 7L+1 entries", len(acc.op_bufs) == 7 * NL + 1)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        with override_forward_context(_ctx(acc.capture_state())):
            for _ in range(2):
                model(x)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                model(x)
    torch.cuda.synchronize()
    C.ok("capture left the buffers untouched (scratch only)",
         all(float(b[:S_MAX].abs().sum()) == 0.0 for b in acc.op_bufs if b is not None))
    with torch.cuda.stream(s):
        st = acc.arm_op_step(np.array(pos), len(pos), 4)
        with override_forward_context(_ctx(st)):
            g.replay()
    torch.cuda.synchronize()
    for i in range(NL):
        C.check(f"layer_in[{i}]", acc.layer_in[i][:S_MAX].float(), ref.layer_in[i][:S_MAX].float())
        C.check(f"resid_mid[{i}]", acc.resid_mid[i][:S_MAX].float(), ref.resid_mid[i][:S_MAX].float())
        C.check(f"gate_up[{i}]", acc.mlp_gate_up[i][:S_MAX].float(), ref.mlp_gate_up[i][:S_MAX].float())
        C.check(f"q[{i}]", acc.attn_qh[i][:S_MAX].float(), ref.attn_qh[i][:S_MAX].float())
        C.check(f"k[{i}]", acc.attn_kh[i][:S_MAX].float(), ref.attn_kh[i][:S_MAX].float())
        C.check(f"v[{i}]", acc.attn_vh[i][:S_MAX].float(), ref.attn_vh[i][:S_MAX].float())
        C.check(f"ctx[{i}]", acc.attn_ctx[i][:S_MAX].float(), ref.attn_ctx[i][:S_MAX].float())
    C.check("final_in", acc.final_in[:S_MAX].float(), ref.final_in[:S_MAX].float())
    # Second step at another offset with other rows, same graph.
    pos2 = [0, 5]
    mask2 = np.zeros(n_tok, dtype=bool); mask2[pos2] = True
    with override_forward_context(_ctx(None)):
        ref.register_hooks()
        ref.begin_step(torch.from_numpy(mask2).to(DEV), 2, offset=9, contiguous=False)
        model(x)
        ref.end_step()
    with torch.cuda.stream(s):
        st = acc.arm_op_step(np.array(pos2), 2, 9)
        with override_forward_context(_ctx(st)):
            g.replay()
    torch.cuda.synchronize()
    C.check("second step, layer_in[1] rows 9..10", acc.layer_in[1][9:11].float(), ref.layer_in[1][9:11].float())
    C.check("second step keeps rows 4..8 from step 1", acc.layer_in[1][4:9].float(), ref.layer_in[1][4:9].float())


if __name__ == "__main__":
    test_capture_follows_index_contents()
    test_accumulator_op_path_matches_hooks()
    C.finish()
