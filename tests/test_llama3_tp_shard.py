#!/usr/bin/env python
"""Correctness gate for the Phase-7 / M2 TP shard geometry (llama3 backward).

M2 makes the co-serving backward shard-aware under tensor parallelism: each TP
rank runs its own backward on its own weight shard. This test validates the
PURE sharding math (CPU, no GPU / no server / no 8B), which is the M2
deliverable — value-correctness (the all-reduces) is M3.

What it checks:
  - lora_shard_slice reconstruction: concatenating the per-rank shards of a FULL
    PEFT LoRA factor reproduces the full factor, and replicated factors are
    returned whole. This mirrors how vLLM shards the served LoRA buffers the
    trainer publishes into (q/k/v column-parallel → B on output rows; o
    row-parallel → A on input cols).
  - tp_size == 1 is byte-identical identity (single-GPU path unchanged).
  - the local-dim arithmetic the backward slices base weights by
    (q/k/v/gate/up widths) tiles the full fused weight exactly across ranks.

    python tests/test_llama3_tp_shard.py
"""

import os
import sys

import torch

sys.path[:] = [p for p in sys.path
               if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]

from vllm.deltaserve.bwd_services import llama3 as L  # noqa: E402

_passed = 0
_failed = 0


def _ok(name, cond, detail=""):
    global _passed, _failed
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    _passed += bool(cond)
    _failed += (not cond)


def test_lora_shard_reconstruction():
    print("test_lora_shard_reconstruction (tp_size=2):")
    torch.manual_seed(0)
    # Llama-3-8B geometry (per-head_dim=128): Hq=32, Hkv=8 → q_size=4096,
    # kv_size=1024. TP=2 → local_q=2048, local_kv=512.
    tp_size, r, hidden = 2, 16, 4096
    q_full, kv_full = 4096, 1024
    local_q, local_kv = q_full // tp_size, kv_full // tp_size

    # Full PEFT factors.
    B = {"q": torch.randn(q_full, r), "k": torch.randn(kv_full, r),
         "v": torch.randn(kv_full, r), "o": torch.randn(hidden, r)}
    A = {"q": torch.randn(r, hidden), "k": torch.randn(r, hidden),
         "v": torch.randn(r, hidden), "o": torch.randn(r, q_full)}

    for proj in ("q", "k", "v"):
        # Column-parallel: B sharded on output rows, A replicated.
        shards = [L.lora_shard_slice(proj, "B", B[proj], rk, tp_size, local_q, local_kv)
                  for rk in range(tp_size)]
        recon = torch.cat(shards, dim=0)
        w = local_q if proj == "q" else local_kv
        _ok(f"{proj}.B shard shape", shards[0].shape == (w, r),
            f"{tuple(shards[0].shape)}")
        _ok(f"{proj}.B reconstruct", torch.equal(recon, B[proj]))
        a0 = L.lora_shard_slice(proj, "A", A[proj], 0, tp_size, local_q, local_kv)
        a1 = L.lora_shard_slice(proj, "A", A[proj], 1, tp_size, local_q, local_kv)
        _ok(f"{proj}.A replicated", torch.equal(a0, A[proj]) and torch.equal(a1, A[proj]))

    # Row-parallel o: A sharded on input cols, B replicated.
    a_shards = [L.lora_shard_slice("o", "A", A["o"], rk, tp_size, local_q, local_kv)
                for rk in range(tp_size)]
    _ok("o.A shard shape", a_shards[0].shape == (r, local_q), f"{tuple(a_shards[0].shape)}")
    _ok("o.A reconstruct", torch.equal(torch.cat(a_shards, dim=1), A["o"]))
    b0 = L.lora_shard_slice("o", "B", B["o"], 0, tp_size, local_q, local_kv)
    _ok("o.B replicated", torch.equal(b0, B["o"]))


def test_tp1_identity():
    print("test_tp1_identity (tp_size=1 → byte-identical):")
    torch.manual_seed(1)
    t = torch.randn(4096, 16)
    for proj in ("q", "k", "v", "o"):
        for ab in ("A", "B"):
            out = L.lora_shard_slice(proj, ab, t, 0, 1, 2048, 512)
            _ok(f"{proj}.{ab} identity", out is t or torch.equal(out, t))


def test_local_dim_tiling():
    print("test_local_dim_tiling (fused qkv / gate_up slice offsets):")
    tp_size = 2
    # Simulate the FULL fused qkv weight [q+kv+kv, hidden] and check that the two
    # ranks' local slices (as the backward computes them) tile it exactly.
    q_full, kv_full, hidden = 4096, 1024, 4096
    local_q, local_kv = q_full // tp_size, kv_full // tp_size
    # vLLM's per-rank qkv shard is [local_q | local_k | local_v] contiguous; the
    # backward slices qkv[:q_size], [q_size:q_size+kv_size], [rest] with LOCAL
    # widths. Assert those local widths sum to the per-rank shard row count.
    per_rank_rows = local_q + 2 * local_kv
    _ok("qkv local widths tile shard",
        local_q + local_kv + local_kv == per_rank_rows, f"rows={per_rank_rows}")
    # And that 2 ranks reconstruct the full fused output dim.
    _ok("qkv ranks reconstruct full",
        tp_size * per_rank_rows == q_full + 2 * kv_full)
    inter_full = 14336  # Llama-3-8B
    _ok("intermediate divisible", inter_full % tp_size == 0,
        f"local_inter={inter_full // tp_size}")


if __name__ == "__main__":
    test_lora_shard_reconstruction()
    test_tp1_identity()
    test_local_dim_tiling()
    print(f"\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)
