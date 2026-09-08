#!/usr/bin/env python
"""M4.3 gate on CPU: gradient bucketing + rank-symmetric clip through the REAL
trainer under a 2-process gloo group (no GPU / no server).

Runs ``LoraSftTrainerService.process_backward`` (eager, fp32) at tp=2 for a
few steps — ``FactorBucket`` groups of 2 layers over 3 layers (2 buckets per
cycle), the ``CommQueue`` in its synchronous CPU mode, ``clip_layers_symmetric_``
with the clip FIRING (per-layer norms exceed 1.0 on this model) — and checks:

  - TP2 masters == the tp=1 masters (replicated exactly, sharded reassembled),
    and the loss trajectories agree, with the clip on: the rank-symmetric clip
    closes the M4.3 trap (before it, TP2 drifted from tp=1 whenever a layer's
    norm exceeded 1.0);
  - rank 0 == rank 1 on the replicated masters (no desync);
  - collectives per cycle == 3 per layer (o + 2 residual) + buckets + 1 clip.

    python tests/test_tp_bucket_gloo.py
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bwd_harness as H  # noqa: E402
import test_tp_trainer_graph_nccl as T  # noqa: E402  (helpers; CPU-safe import)

sys.path[:] = [p for p in sys.path
               if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]

PORT = 29720
TOL_REF = 1e-4
TOL_SAME = 1e-6


def _worker(rank, fam_name, wpath, outdir):
    try:
        full = torch.load(wpath)
        meta = T.make_meta(fam_name, tp_size=T.TP, tp_rank=rank, port=PORT, graph=False)
        counter = T.ReduceCounter()
        svc = T.make_service(fam_name, T.shard_base(full["base"], rank, T.TP, fam_name),
                             full["ft"], meta, torch.device("cpu"))
        assert svc._bucket is not None and svc._comm is not None
        assert svc._comm.stream is None, "CPU CommQueue must be synchronous"
        losses, masters, counts = T.train(svc, full["embed"], full["ids"], counter)
        torch.save({"losses": losses, "masters": masters, "counts": counts,
                    "n_groups": svc._bucket.num_groups},
                   os.path.join(outdir, f"rank{rank}.pt"))
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)


def run_family(fam_name, C):
    print(f"test_tp_bucket_gloo ({fam_name}, tp=2, gloo, NL={T.NL}, "
          f"bucket_layers={T.BUCKET_LAYERS}, {T.STEPS} steps):")
    full = T.build_full(fam_name, seed=31 if fam_name == "llama3" else 37)
    tmp = tempfile.mkdtemp(prefix="tp_bucket_")
    wpath = os.path.join(tmp, "full.pt")
    torch.save(full, wpath)
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_worker, args=(rk, fam_name, wpath, tmp))
             for rk in range(T.TP)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    if any(p.exitcode != 0 for p in procs):
        C.ok(f"{fam_name} workers exited cleanly", False)
        return
    r = [torch.load(os.path.join(tmp, f"rank{rk}.pt")) for rk in range(T.TP)]

    ref_svc = T.make_service(fam_name, full["base"], full["ft"],
                             T.make_meta(fam_name, tp_size=1, tp_rank=0, port=0,
                                         graph=False), torch.device("cpu"))
    ref_losses, ref_masters, _ = T.train(ref_svc, full["embed"], full["ids"])
    from vllm.deltaserve.bwd_services.common.tp import clip_layers_symmetric_
    n = sum(T.SEQ_LENS); b_start = [0, T.SEQ_LENS[0]]
    acts = T.remat_activations(ref_svc, full["embed"], full["ids"], T.SEQ_LENS, b_start)
    ref_svc.process_backward(acts, T.SEQ_LENS, n, epoch=0)
    # process_backward already clipped these grads: a layer sitting exactly at
    # max_norm=1.0 is the evidence that the clip fired on this model.
    norms = clip_layers_symmetric_(ref_svc.lora, ref_svc.L, None, max_norm=float("inf"))
    C.ok(f"{fam_name} the clip fires on this model (a layer norm == 1.0 post-clip)",
         bool((norms > 0.999).any()), f"norms {[f'{v:.2f}' for v in norms.tolist()]}")
    print(f"    losses tp2 {['%.5f' % v for v in r[0]['losses']]}")
    print(f"    losses tp1 {['%.5f' % v for v in ref_losses]}")

    for s in range(T.STEPS):
        C.check(f"{fam_name} step{s} loss r0 == r1", torch.tensor(r[0]["losses"][s]),
                torch.tensor(r[1]["losses"][s]), tol=TOL_SAME)
        C.check(f"{fam_name} step{s} loss tp2 == tp1", torch.tensor(r[0]["losses"][s]),
                torch.tensor(ref_losses[s]), tol=TOL_REF)
    for k in ref_masters:
        if k.endswith(("qA", "kA", "vA", "oB")):
            C.check(f"{fam_name} master {k} r0 == r1", r[0]["masters"][k],
                    r[1]["masters"][k], tol=TOL_SAME)
            C.check(f"{fam_name} master {k} tp2 == tp1", r[0]["masters"][k],
                    ref_masters[k], tol=TOL_REF)
    for i in range(T.NL):
        for proj in ("q", "k", "v"):
            k = f"L{i}.{proj}B"
            C.check(f"{fam_name} master {k} tp2 concat == tp1",
                    torch.cat([r[0]["masters"][k], r[1]["masters"][k]], 0),
                    ref_masters[k], tol=TOL_REF)
        k = f"L{i}.oA"
        C.check(f"{fam_name} master {k} tp2 concat == tp1",
                torch.cat([r[0]["masters"][k], r[1]["masters"][k]], 1),
                ref_masters[k], tol=TOL_REF)
    n_groups = math.ceil(T.NL / T.BUCKET_LAYERS)
    want = T.NL * 3 + n_groups + 1
    C.ok(f"{fam_name} {want} collectives per cycle ({n_groups} buckets)",
         r[0]["n_groups"] == n_groups
         and all(c == want for rk in range(T.TP) for c in r[rk]["counts"]),
         f"got {r[0]['counts']}")


if __name__ == "__main__":
    C = H.Checker(tol=TOL_SAME)
    for fam in ("llama3", "qwen3"):
        run_family(fam, C)
    C.finish()
