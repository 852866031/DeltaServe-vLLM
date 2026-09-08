#!/usr/bin/env python
"""Two-process TP backward for the Qwen3 family (Phase 7 / M3 gate, second
family): real gloo all-reduces injected into the shard-aware ``layer_forward``
/ ``layer_backward`` must reproduce the tp=1 reference. Geometry has
Hq*Hd != hidden. CPU, fp32, port 29689 (distinct from the Llama-3 test).

    python tests/test_qwen3_tp_backward_gloo.py
"""

import os
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# The harness lives next to this script; re-add tests/ first so a spawned
# worker (which inherits the scrubbed sys.path) can import it too.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bwd_harness as H  # noqa: E402

sys.path[:] = [p for p in sys.path
               if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]

from vllm.deltaserve.bwd_services import qwen3 as Q  # noqa: E402

# Hq=4/Hkv=2 GQA, Hd=8 → attention width 32; residual width 24 (≠ 32).
CFG = dict(hidden=24, Hq=4, Hkv=2, Hd=8, inter=64, r=4, eps=1e-6, scaling=2.0,
           theta=1e6, seq_lens=[3, 2], tp=2, port=29689)
_TOL = 1e-4


def _build_full(cfg, seed=0):
    gen = torch.Generator().manual_seed(seed)
    lw = H.make_layer_weights(Q.QWEN3, cfg["hidden"], cfg["Hq"], cfg["Hkv"],
                              cfg["Hd"], cfg["inter"], cfg["r"], gen=gen, scale=1.0)
    n = sum(cfg["seq_lens"])
    return {"lw": lw,
            "x": torch.randn(n, cfg["hidden"], generator=gen),
            "grad_out": torch.randn(n, cfg["hidden"], generator=gen)}


def _worker(rank, cfg, wpath, outdir):
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{cfg['port']}",
        rank=rank, world_size=cfg["tp"])

    def _ar(t):
        t = t.contiguous()
        dist.all_reduce(t)
        return t

    full = torch.load(wpath)
    lw = H.shard_layer_weights(full["lw"], rank, cfg["tp"], cfg["Hq"], cfg["Hkv"],
                               cfg["Hd"], cfg["inter"])
    b_start, cos, sin = H.seq_layout(cfg["seq_lens"], cfg["Hd"], cfg["theta"])
    tp = cfg["tp"]
    dims = (cfg["Hq"] // tp, cfg["Hkv"] // tp, cfg["Hd"], (cfg["Hkv"] // tp) * cfg["Hd"])
    cache = Q.layer_forward(full["x"], lw, cfg["scaling"], cos, sin, cfg["seq_lens"],
                            b_start, dims, cfg["eps"], all_reduce=_ar)
    grad_x, grads = Q.layer_backward(
        full["grad_out"], cache, lw, cfg["scaling"], cos, sin, cfg["seq_lens"],
        b_start, dims, cfg["eps"], cdt=torch.float32, all_reduce=_ar)
    torch.save({"grad_x": grad_x, **grads}, os.path.join(outdir, f"rank{rank}.pt"))
    dist.destroy_process_group()


def _ref(full, cfg):
    b_start, cos, sin = H.seq_layout(cfg["seq_lens"], cfg["Hd"], cfg["theta"])
    dims = (cfg["Hq"], cfg["Hkv"], cfg["Hd"], cfg["Hkv"] * cfg["Hd"])
    cache = Q.layer_forward(full["x"], full["lw"], cfg["scaling"], cos, sin,
                            cfg["seq_lens"], b_start, dims, cfg["eps"])
    return Q.layer_backward(full["grad_out"], cache, full["lw"], cfg["scaling"], cos,
                            sin, cfg["seq_lens"], b_start, dims, cfg["eps"],
                            cdt=torch.float32)


def main():
    cfg = CFG
    C = H.Checker(tol=_TOL)
    full = _build_full(cfg)
    ref_gx, ref = _ref(full, cfg)

    tmp = tempfile.mkdtemp(prefix="tp_bwd_qwen3_")
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

    print("test_qwen3_tp_backward_gloo (tp=2, real gloo all-reduce):")
    C.check("grad_x r0", r0["grad_x"], ref_gx)
    C.check("grad_x r0==r1", r0["grad_x"], r1["grad_x"])
    for k in ("qA", "kA", "vA", "oB"):
        C.check(f"{k} (reduced)", r0[k], ref[k])
    C.check("qB concat", torch.cat([r0["qB"], r1["qB"]], 0), ref["qB"])
    C.check("kB concat", torch.cat([r0["kB"], r1["kB"]], 0), ref["kB"])
    C.check("vB concat", torch.cat([r0["vB"], r1["vB"]], 0), ref["vB"])
    C.check("oA concat", torch.cat([r0["oA"], r1["oA"]], 1), ref["oA"])
    C.finish()


if __name__ == "__main__":
    main()
