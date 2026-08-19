#!/usr/bin/env python
"""Correctness gate for the Phase-7 / M3 TP backward (llama3), the milestone gate.

Runs the shard-aware layer forward-remat + backward under a REAL 2-process
collective (gloo, CPU — no GPU / no server / no 8B) and asserts that the
all-reduced / reassembled per-rank gradients reproduce the single-GPU reference
to fp32 precision. This is what makes the M2 shapes VALUE-correct:

  - forward o_proj (row-parallel) all-reduce → correct full ``resid_mid``
  - grad_x all-reduce (column-parallel qkv + FFN partials)
  - grad_{q,k,v}A all-reduce (replicated column-parallel A)
  - grad_oB all-reduce (replicated row-parallel B)
  - grad_{q,k,v}B (output-sharded) / grad_oA (input-sharded): concat == reference

The reference is the tp_size=1 manual backward, itself gradchecked vs autograd in
tests/test_llama3_backward.py — so this test isolates the TP reductions.

    python tests/test_llama3_tp_backward_gloo.py
"""

import os
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path[:] = [p for p in sys.path
               if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]

from vllm.deltaserve.bwd_services import llama3 as L  # noqa: E402

# Small Llama-ish geometry. Hq=4/Hkv=2 GQA, Hd=8 → D=32; TP=2 → local 2/1 heads.
CFG = dict(Hq=4, Hkv=2, Hd=8, inter=64, r=4, eps=1e-5, scaling=2.0,
           theta=10000.0, seq_lens=[3, 2], tp=2, port=29688)
_TOL = 1e-4


def _build_full(cfg, seed=0):
    """Full (unsharded) weights + LoRA + inputs, all fp32."""
    torch.manual_seed(seed)
    Hq, Hkv, Hd, inter, r = cfg["Hq"], cfg["Hkv"], cfg["Hd"], cfg["inter"], cfg["r"]
    D = Hq * Hd
    q_size, kv_size = Hq * Hd, Hkv * Hd
    n = sum(cfg["seq_lens"])
    g = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=g, dtype=torch.float32)  # noqa: E731
    lw = {
        "q": rn(q_size, D), "k": rn(kv_size, D), "v": rn(kv_size, D),
        "o": rn(D, q_size), "gate": rn(inter, D), "up": rn(inter, D),
        "down": rn(D, inter),
        "in_ln": rn(D).abs() + 0.5, "post_ln": rn(D).abs() + 0.5,
        "qA": rn(r, D), "qB": rn(q_size, r), "kA": rn(r, D), "kB": rn(kv_size, r),
        "vA": rn(r, D), "vB": rn(kv_size, r), "oA": rn(r, q_size), "oB": rn(D, r),
    }
    x = rn(n, D)
    grad_out = rn(n, D)
    return {"lw": lw, "x": x, "grad_out": grad_out}


def _rope(cfg):
    b_start, acc, seq_lens = [], 0, cfg["seq_lens"]
    for s in seq_lens:
        b_start.append(acc)
        acc += s
    positions = torch.cat([torch.arange(s) for s in seq_lens]).float()
    cos, sin = L.rope_cos_sin(positions, cfg["Hd"], cfg["theta"])
    return cos, sin, b_start


def _shard_lw(lw, rk, cfg):
    """Slice the full weights + LoRA into rank rk's shard (matches vLLM +
    lora_shard_slice + the backward's base-weight slicing)."""
    tp, Hq, Hkv, Hd, inter = cfg["tp"], cfg["Hq"], cfg["Hkv"], cfg["Hd"], cfg["inter"]
    ql, kl, il = (Hq // tp) * Hd, (Hkv // tp) * Hd, inter // tp
    s = {}
    # column-parallel base: qkv/gate/up sharded on output rows.
    s["q"] = lw["q"][rk * ql:(rk + 1) * ql]
    s["k"] = lw["k"][rk * kl:(rk + 1) * kl]
    s["v"] = lw["v"][rk * kl:(rk + 1) * kl]
    s["gate"] = lw["gate"][rk * il:(rk + 1) * il]
    s["up"] = lw["up"][rk * il:(rk + 1) * il]
    # row-parallel base: o/down sharded on input cols.
    s["o"] = lw["o"][:, rk * ql:(rk + 1) * ql]
    s["down"] = lw["down"][:, rk * il:(rk + 1) * il]
    # replicated norms.
    s["in_ln"], s["post_ln"] = lw["in_ln"], lw["post_ln"]
    # LoRA via the production sharder.
    for proj in ("q", "k", "v", "o"):
        for ab in ("A", "B"):
            s[proj + ab] = L.lora_shard_slice(proj, ab, lw[proj + ab], rk, tp, ql, kl)
    return s


def _worker(rank, cfg, wpath, outdir):
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{cfg['port']}",
        rank=rank, world_size=cfg["tp"])

    def _ar(t):
        t = t.contiguous()
        dist.all_reduce(t)
        return t

    full = torch.load(wpath)
    lw = _shard_lw(full["lw"], rank, cfg)
    x, grad_out = full["x"], full["grad_out"]
    cos, sin, b_start = _rope(cfg)
    dims = (cfg["Hq"] // cfg["tp"], cfg["Hkv"] // cfg["tp"], cfg["Hd"],
            (cfg["Hkv"] // cfg["tp"]) * cfg["Hd"])

    cache = L.layer_forward(x, lw, cfg["scaling"], cos, sin, cfg["seq_lens"],
                            b_start, dims, cfg["eps"], all_reduce=_ar)
    grad_x, grads = L.layer_backward(
        grad_out, cache, lw, cfg["scaling"], cos, sin, cfg["seq_lens"],
        b_start, dims, cfg["eps"], cdt=torch.float32, all_reduce=_ar)

    torch.save({"grad_x": grad_x, **grads},
               os.path.join(outdir, f"rank{rank}.pt"))
    dist.destroy_process_group()


def _ref(full, cfg):
    """Single-GPU (tp=1) reference — no reduces."""
    cos, sin, b_start = _rope(cfg)
    dims = (cfg["Hq"], cfg["Hkv"], cfg["Hd"], cfg["Hkv"] * cfg["Hd"])
    cache = L.layer_forward(full["x"], full["lw"], cfg["scaling"], cos, sin,
                            cfg["seq_lens"], b_start, dims, cfg["eps"])
    grad_x, grads = L.layer_backward(
        full["grad_out"], cache, full["lw"], cfg["scaling"], cos, sin,
        cfg["seq_lens"], b_start, dims, cfg["eps"], cdt=torch.float32)
    return grad_x, grads


def main():
    cfg = CFG
    passed = failed = 0

    def check(name, a, b):
        nonlocal passed, failed
        rel = (a - b).abs().max().item() / (b.abs().max().item() + 1e-8)
        ok = rel < _TOL
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:16s} max-rel-err={rel:.3e}")
        passed += ok
        failed += (not ok)

    full = _build_full(cfg)
    ref_gx, ref = _ref(full, cfg)

    tmp = tempfile.mkdtemp(prefix="tp_bwd_")
    wpath = os.path.join(tmp, "weights.pt")
    torch.save(full, wpath)

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_worker, args=(rk, cfg, wpath, tmp))
             for rk in range(cfg["tp"])]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        if p.exitcode != 0:
            print(f"  worker exited with code {p.exitcode}")
            sys.exit(1)

    r0 = torch.load(os.path.join(tmp, "rank0.pt"))
    r1 = torch.load(os.path.join(tmp, "rank1.pt"))
    tp = cfg["tp"]
    Hq, Hkv, Hd, inter = cfg["Hq"], cfg["Hkv"], cfg["Hd"], cfg["inter"]
    ql, kl = (Hq // tp) * Hd, (Hkv // tp) * Hd

    print("test_tp_backward_gloo (tp=2, real gloo all-reduce):")
    # grad_x: all-reduced → both ranks hold the full correct grad.
    check("grad_x r0", r0["grad_x"], ref_gx)
    check("grad_x r0==r1", r0["grad_x"], r1["grad_x"])
    # Replicated factors: all-reduced → equal to reference on each rank.
    for k in ("qA", "kA", "vA", "oB"):
        check(f"{k} (reduced)", r0[k], ref[k])
    # Output-sharded q/k/v B: concat over ranks == reference.
    check("qB concat", torch.cat([r0["qB"], r1["qB"]], 0), ref["qB"])
    check("kB concat", torch.cat([r0["kB"], r1["kB"]], 0), ref["kB"])
    check("vB concat", torch.cat([r0["vB"], r1["vB"]], 0), ref["vB"])
    # Input-sharded o A: concat over ranks (dim=1) == reference.
    check("oA concat", torch.cat([r0["oA"], r1["oA"]], 1), ref["oA"])

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
