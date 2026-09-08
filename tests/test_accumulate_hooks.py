#!/usr/bin/env python
"""Hook-level gate for ``FinetuneAccumulator`` (CPU, no vLLM model).

Builds a fake decoder module tree with vLLM's module names and fused add-norm
calling convention, registers the accumulator's hooks, runs a fake forward
with FT rows at a given offset, and checks every per-layer buffer holds
exactly the FT rows of the tensor it is meant to capture:

  - ``layer_in[i]``  = residual stream entering ``layers.i.input_layernorm``
  - ``resid_mid[i]`` = ``o + residual`` entering ``layers.i.post_attention_layernorm``
    (``save_resid_mid`` — the post-attention residual / FFN input)
  - ``mlp_gate_up[i]`` = ``layers.i.mlp.gate_up_proj`` output
  - ``final_in``     = residual entering ``model.norm``

on both the contiguous-slice fast path and the boolean-mask fallback, with a
non-zero write offset; and that ``save_resid_mid=False`` allocates nothing.

    python tests/test_accumulate_hooks.py
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bwd_harness as H  # noqa: E402

sys.path[:] = [p for p in sys.path
               if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]

from vllm.deltaserve.accumulate import FinetuneAccumulator  # noqa: E402

C = H.Checker(tol=1e-6)
D, INTER, NL = 8, 12, 3


class FakeAddNorm(nn.Module):
    """vLLM's fused add-norm signature: ``(x, residual=None) -> (normed, residual)``
    where the returned residual is ``x + residual``. The "norm" is a scale so
    the values stay distinguishable."""

    def forward(self, x, residual=None):
        r = x if residual is None else x + residual
        return r * 0.5, r


HQ, HKV, HD = 2, 1, 4       # q_size 8 (= D), kv_size 4


class FakeAttnCore(nn.Module):
    """``self_attn.attn(q, k, v)`` stand-in: returns something q-shaped."""

    def forward(self, q, k, v):
        return q + 0.25


class FakeHeadNorm(nn.Module):
    """Qwen3's per-head q/k RMSNorm stand-in (input is a [n, H, Hd] view)."""

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


class FakeLayer(nn.Module):
    def __init__(self, qk_norm=False):
        super().__init__()
        self.qk_norm = qk_norm
        self.input_layernorm = FakeAddNorm()
        self.self_attn = nn.Module()
        self.self_attn.qkv_proj = nn.Linear(D, HQ * HD + 2 * HKV * HD, bias=False)
        if qk_norm:
            self.self_attn.q_norm = FakeHeadNorm()
            self.self_attn.k_norm = FakeHeadNorm()
        self.self_attn.attn = FakeAttnCore()
        self.post_attention_layernorm = FakeAddNorm()
        self.mlp = nn.Module()
        self.mlp.gate_up_proj = nn.Linear(D, 2 * INTER, bias=False)
        self.mlp.down_proj = nn.Linear(INTER, D, bias=False)

    def attention(self, h_in, trace=None):
        """vLLM's attention call order: qkv_proj → (q/k norm) → attn(q, k, v).
        RoPE is omitted (an orthogonal map the hooks never see for Qwen3;
        for the Llama-style post-RoPE hook it would just be part of q/k)."""
        qkv = self.self_attn.qkv_proj(h_in)
        q, k, v = qkv.split([HQ * HD, HKV * HD, HKV * HD], dim=-1)
        if self.qk_norm:
            q_pre = q.view(-1, HQ, HD)
            k_pre = k.view(-1, HKV, HD)
            if trace is not None:
                trace["q_saved"].append(q_pre.reshape(-1, HQ * HD))
                trace["k_saved"].append(k_pre.reshape(-1, HKV * HD))
            q = self.self_attn.q_norm(q_pre).view(q.shape)
            k = self.self_attn.k_norm(k_pre).view(k.shape)
        elif trace is not None:
            trace["q_saved"].append(q)
            trace["k_saved"].append(k)
        if trace is not None:
            trace["v_saved"].append(v)
        return self.self_attn.attn(q, k, v)

    def forward(self, h, r):
        h, r = self.input_layernorm(h, r)
        o = self.attention(h) * 3.0 + 1.0       # attention + a stand-in o_proj
        h, r = self.post_attention_layernorm(o, r)
        gu = self.mlp.gate_up_proj(h)
        h = self.mlp.down_proj(gu[:, :INTER] * gu[:, INTER:])
        return h, r


class FakeModel(nn.Module):
    def __init__(self, qk_norm=False):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([FakeLayer(qk_norm) for _ in range(NL)])
        self.model.norm = FakeAddNorm()

    def forward(self, x):
        h, r = x, None
        trace = {"layer_in": [], "resid_mid": [], "gate_up": [],
                 "q_saved": [], "k_saved": [], "v_saved": []}
        for layer in self.model.layers:
            trace["layer_in"].append(h if r is None else h + r)
            h_in, r_in = layer.input_layernorm(h, r)
            o = layer.attention(h_in, trace) * 3.0 + 1.0
            trace["resid_mid"].append(o + r_in)
            h2, r2 = layer.post_attention_layernorm(o, r_in)
            gu = layer.mlp.gate_up_proj(h2)
            trace["gate_up"].append(gu)
            h = layer.mlp.down_proj(gu[:, :INTER] * gu[:, INTER:])
            r = r2
        trace["final_in"] = h + r
        self.model.norm(h, r)
        return trace


def run_case(name, contiguous, offset, qk_norm=False):
    torch.manual_seed(0)
    model = FakeModel(qk_norm=qk_norm)
    acc = FinetuneAccumulator(model, max_saved=16, hidden_size=D, device="cpu",
                              dtype=torch.float32, intermediate_size=INTER,
                              q_size=HQ * HD, kv_size=HKV * HD,
                              save_attn_qkv=True, attn_qkv_pre_transform=qk_norm,
                              save_attn_ctx=True, save_resid_mid=True)
    acc.register_hooks()
    C.ok(f"{name}: resid_mid buffers allocated", len(acc.resid_mid) == NL
         and "resid_mid" in acc.buffers)
    C.ok(f"{name}: attn q/k/v + ctx buffers allocated",
         len(acc.attn_qh) == NL and len(acc.attn_ctx) == NL and acc._save_attn_qkv)

    n_total, n_ft = 10, 4
    if contiguous:
        start = 6
        mask = torch.zeros(n_total, dtype=torch.bool)
        mask[start:start + n_ft] = True
    else:
        start = 0
        mask = torch.tensor([1, 0, 1, 0, 0, 1, 0, 0, 1, 0], dtype=torch.bool)
    x = torch.randn(n_total, D)
    acc.begin_step(mask, n_ft, offset=offset, start=start, contiguous=contiguous)
    trace = model(x)
    acc.end_step()

    for i in range(NL):
        C.check(f"{name}: layer_in[{i}]", acc.layer_in[i][offset:offset + n_ft],
                trace["layer_in"][i][mask])
        C.check(f"{name}: resid_mid[{i}]", acc.resid_mid[i][offset:offset + n_ft],
                trace["resid_mid"][i][mask])
        C.check(f"{name}: mlp_gate_up[{i}]",
                acc.mlp_gate_up[i][offset:offset + n_ft], trace["gate_up"][i][mask])
        # q/k at the stage the family's backward needs (pre-norm for the
        # qk_norm model, straight from the attn call otherwise); v; ctx.
        C.check(f"{name}: attn_qh[{i}]", acc.attn_qh[i][offset:offset + n_ft],
                trace["q_saved"][i][mask])
        C.check(f"{name}: attn_kh[{i}]", acc.attn_kh[i][offset:offset + n_ft],
                trace["k_saved"][i][mask])
        C.check(f"{name}: attn_vh[{i}]", acc.attn_vh[i][offset:offset + n_ft],
                trace["v_saved"][i][mask])
        C.check(f"{name}: attn_ctx[{i}]", acc.attn_ctx[i][offset:offset + n_ft],
                (trace["q_saved"][i] if not qk_norm else
                 model.model.layers[i].self_attn.q_norm(
                     trace["q_saved"][i].view(-1, HQ, HD)).reshape(-1, HQ * HD)
                 )[mask] + 0.25)
        # Rows outside the written span stay zero.
        C.ok(f"{name}: resid_mid[{i}] untouched outside span",
             acc.resid_mid[i][:offset].abs().sum().item() == 0.0
             and acc.resid_mid[i][offset + n_ft:].abs().sum().item() == 0.0)
    C.check(f"{name}: final_in", acc.final_in[offset:offset + n_ft],
            trace["final_in"][mask])

    # Hooks are inert off-step.
    before = acc.resid_mid[0].clone()
    model(torch.randn(n_total, D))
    C.ok(f"{name}: hooks inert when not armed",
         torch.equal(before, acc.resid_mid[0]))
    # Tier-C hygiene zeroes the new buffers too.
    acc.zero_offset_range(offset, n_ft)
    C.ok(f"{name}: zero_offset_range clears resid_mid",
         acc.resid_mid[0].abs().sum().item() == 0.0)


def test_disabled_allocates_nothing():
    acc = FinetuneAccumulator(FakeModel(), max_saved=16, hidden_size=D,
                              device="cpu", dtype=torch.float32,
                              intermediate_size=INTER)
    C.ok("save_resid_mid=False: no buffers", acc.resid_mid == []
         and "resid_mid" not in acc.buffers)
    # Pre-transform q/k requested on a model WITHOUT q/k norm modules: the
    # save must disable itself rather than capture the wrong stage.
    acc = FinetuneAccumulator(FakeModel(qk_norm=False), max_saved=16, hidden_size=D,
                              device="cpu", dtype=torch.float32,
                              intermediate_size=INTER, q_size=HQ * HD,
                              kv_size=HKV * HD, save_attn_qkv=True,
                              attn_qkv_pre_transform=True)
    C.ok("pre-transform save without norm modules → disabled",
         not acc._save_attn_qkv and acc.attn_qh == [])


if __name__ == "__main__":
    print("test_accumulate_hooks (fake vLLM-named decoder, CPU):")
    run_case("llama-style contiguous/off=0", contiguous=True, offset=0)
    run_case("llama-style contiguous/off=5", contiguous=True, offset=5)
    run_case("llama-style mask/off=3", contiguous=False, offset=3)
    run_case("qwen-style (pre-norm q/k) contiguous/off=2", contiguous=True,
             offset=2, qk_norm=True)
    run_case("qwen-style (pre-norm q/k) mask/off=0", contiguous=False,
             offset=0, qk_norm=True)
    test_disabled_allocates_nothing()
    C.finish()
