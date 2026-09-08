#!/usr/bin/env python
"""TP shard geometry for Qwen3 (Phase 7 / M2 gate, second family).

Pure sharding math, no GPU: ``lora_shard_slice`` reconstruction + tp=1
identity + the trainer's fused qkv / gate_up local-dim tiling at the two real
geometries — Qwen3-0.6B (attention width 2048 != hidden 1024, tied head) and
Qwen3-14B (the TP=2 target).

    python tests/test_qwen3_tp_shard.py
"""

import os
import sys

import torch

# The harness lives next to this script; re-add tests/ first so a spawned
# worker (which inherits the scrubbed sys.path) can import it too.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bwd_harness as H  # noqa: E402

sys.path[:] = [p for p in sys.path
               if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]

from vllm.deltaserve.bwd_services.common.tp import lora_shard_slice  # noqa: E402

C = H.Checker(tol=0.0)

GEOMETRIES = {
    # name: (hidden, Hq, Hkv, Hd, inter)
    "qwen3-0.6b": (1024, 16, 8, 128, 3072),
    "qwen3-14b": (5120, 40, 8, 128, 17408),
}


def test_geometry(name, hidden, Hq, Hkv, Hd, inter, tp=2, r=16):
    print(f"test_shard[{name}] hidden={hidden} Hq={Hq} Hkv={Hkv} inter={inter} tp={tp}:")
    torch.manual_seed(0)
    q_full, kv_full = Hq * Hd, Hkv * Hd
    ql, kl, il = q_full // tp, kv_full // tp, inter // tp
    C.ok("heads/inter divisible", Hq % tp == 0 and Hkv % tp == 0 and inter % tp == 0)

    A = {p: torch.randn(r, hidden) for p in ("q", "k", "v")}
    A["o"] = torch.randn(r, q_full)
    B = {"q": torch.randn(q_full, r), "k": torch.randn(kv_full, r),
         "v": torch.randn(kv_full, r), "o": torch.randn(hidden, r)}

    for proj in ("q", "k", "v"):
        shards = [lora_shard_slice(proj, "B", B[proj], rk, tp, ql, kl) for rk in range(tp)]
        C.ok(f"{proj}B shards concat == full",
             torch.equal(torch.cat(shards, 0), B[proj]))
        a0 = lora_shard_slice(proj, "A", A[proj], 0, tp, ql, kl)
        a1 = lora_shard_slice(proj, "A", A[proj], 1, tp, ql, kl)
        C.ok(f"{proj}A replicated", torch.equal(a0, A[proj]) and torch.equal(a1, A[proj]))
    a_shards = [lora_shard_slice("o", "A", A["o"], rk, tp, ql, kl) for rk in range(tp)]
    C.ok("oA shards concat(dim=1) == full", torch.equal(torch.cat(a_shards, 1), A["o"]))
    C.ok("oA shard width == local q", a_shards[0].shape == (r, ql))
    b0 = lora_shard_slice("o", "B", B["o"], 0, tp, ql, kl)
    C.ok("oB replicated [hidden, r]", torch.equal(b0, B["o"]) and b0.shape == (hidden, r))

    for proj in ("q", "k", "v", "o"):
        for ab, t in (("A", A[proj]), ("B", B[proj])):
            C.ok(f"tp=1 identity {proj}{ab}",
                 lora_shard_slice(proj, ab, t, 0, 1, q_full, kv_full) is t)

    # Trainer tiling of the per-rank fused qkv / gate_up (as vLLM shards them:
    # each rank's fused tensor is cat(q_rk, k_rk, v_rk)).
    q, k, v = torch.randn(q_full, hidden), torch.randn(kv_full, hidden), torch.randn(kv_full, hidden)
    for rk in range(tp):
        fused = torch.cat([q[rk * ql:(rk + 1) * ql], k[rk * kl:(rk + 1) * kl],
                           v[rk * kl:(rk + 1) * kl]], 0)
        C.ok(f"rank{rk} fused qkv tiling",
             torch.equal(fused[:ql], q[rk * ql:(rk + 1) * ql])
             and torch.equal(fused[ql:ql + kl], k[rk * kl:(rk + 1) * kl])
             and torch.equal(fused[ql + kl:], v[rk * kl:(rk + 1) * kl]))
    C.ok("local widths", (ql, kl, il) == ((Hq // tp) * Hd, (Hkv // tp) * Hd, inter // tp),
         f"local q={ql} kv={kl} inter={il}")


if __name__ == "__main__":
    for name, geo in GEOMETRIES.items():
        test_geometry(name, *geo)
    C.finish()
