#!/usr/bin/env python
"""M5 gate: the CUDA-graph backward under TP=2 on two real GPUs (NCCL).

Phase 7 / M5 re-enables ``finetune.backward_cuda_graph`` under tensor
parallelism by keeping every collective OUT of the captured regions: the
forward graph is split at the O-proj output (``static_o``, all-reduced
eagerly before the runner's ``forward_tail``), and the family's
``layer_backward_graphed`` issues the six backward-side reduces from its eager
code. This test runs that path as production does — one process per GPU, the
backward children's own NCCL group, ``GraphedBackward`` capturing on each rank
— and checks, per family:

  1. graph-TP == eager-TP on every rank (same math, replayed: cache entries,
     grad_x and all eight LoRA grads at fp32 tolerance), including the
     padded-attention OVERFLOW batch (eager forward fallback inside the runner,
     which must still reduce ``o``);
  2. the reduced tensors agree across ranks (lock-step, no divergence);
  3. graph-TP == the tp=1 single-GPU reference (sharded factors reassembled),
     at fp32 tolerance for fp32 model dtype and a bf16-appropriate tolerance for
     bf16 (the o reduce then sums two bf16 partials);
  4. the collective COUNT per layer: 7 all-reduces on the graph path as on the
     eager path (1 forward + 6 backward), and 6 with ``save_resid_mid`` on —
     the forward remat then reads the captured (already reduced) post-attention
     residual and issues no collective at all. The saved-residual run also
     checks graph == eager-with-saved == the full recompute; the ``all`` run
     adds ``save_attn_qkv`` (post-RoPE q/k for Llama-3, pre-norm q/k for
     Qwen3) + ``save_attn_ctx`` — the "only in_ln is recomputed" mode.

Requires 2 CUDA devices; skips otherwise.

    python tests/test_tp_backward_graph_nccl.py [--family llama3|qwen3|all]
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import traceback
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bwd_harness as H  # noqa: E402

sys.path[:] = [p for p in sys.path
               if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]

from vllm.deltaserve.bwd_services import llama3 as L  # noqa: E402
from vllm.deltaserve.bwd_services import qwen3 as Q  # noqa: E402

FAMILIES = {"llama3": L, "qwen3": Q}
TP = 2
BASE_PORT = 29690

# Small GQA geometry; Hq*Hd != hidden for qwen3 (as on the real model).
GEOM = {
    "llama3": dict(hidden=32, Hq=4, Hkv=2, Hd=8, inter=64, r=4, eps=1e-5,
                   scaling=2.0, theta=500000.0),
    "qwen3": dict(hidden=24, Hq=4, Hkv=2, Hd=8, inter=64, r=4, eps=1e-6,
                  scaling=2.0, theta=1e6),
}
L_ = 2
S_MAX, BN_MAX, L_MAX = 16, 4, 8
# Two fitting batches + one that overflows l_max (→ eager forward/attn fallback).
BATCHES = [[5, 3], [8, 4], [9, 2]]
CACHE_KEYS = ("x_norm1", "qh", "kh", "vh", "ctx_flat", "resid_mid", "gate", "up")
TOL_SAME_MATH = 1e-5     # graph vs eager on the same rank / rank vs rank
TOL_REF = {torch.float32: 1e-4, torch.bfloat16: 3e-2}


def _to(lw, device, mdt):
    """Move a full CPU fp32 ``lw`` to ``device``: base in model dtype, LoRA fp32
    (the trainer's masters are fp32; ``proj`` casts them per matmul)."""
    out = {}
    for k, v in lw.items():
        v = v.to(device)
        out[k] = v if k in H.LORA_KEYS else v.to(mdt)
    return out


def _worker(rank, fam_name, mdt, wpath, outdir, port, saves="none"):
    try:
        _worker_body(rank, fam_name, mdt, wpath, outdir, port, saves)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)


def _worker_body(rank, fam_name, mdt, wpath, outdir, port, saves):
    save_rm = saves in ("rm", "all")
    save_all = saves == "all"
    from vllm.deltaserve.bwd_services.common.graph import GraphedBackward

    fam_mod = FAMILIES[fam_name]
    FAM = fam_mod.LLAMA3 if fam_name == "llama3" else fam_mod.QWEN3
    g = GEOM[fam_name]
    hidden, Hq, Hkv, Hd, inter = g["hidden"], g["Hq"], g["Hkv"], g["Hd"], g["inter"]
    eps, scaling, theta = g["eps"], g["scaling"], g["theta"]

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=TP)

    n_reduce = [0]

    def _ar(t):
        n_reduce[0] += 1
        t = t.contiguous()
        dist.all_reduce(t)
        return t

    full = torch.load(wpath)
    shards = [_to(H.shard_layer_weights(lw, rank, TP, Hq, Hkv, Hd, inter),
                  device, mdt) for lw in full["lws"]]
    Hq_l, Hkv_l, inter_l = Hq // TP, Hkv // TP, inter // TP
    dims = (Hq_l, Hkv_l, Hd, Hkv_l * Hd)

    svc = SimpleNamespace(
        device_index=rank, family=FAM,
        D=hidden, L=L_, Hq=Hq_l, Hkv=Hkv_l, Hd=Hd, inter=inter_l,
        kv_size=Hkv_l * Hd, eps=eps, base_dtype=mdt, bwd_dtype=torch.float32,
        scaling=scaling, theta=theta,
        save_attn_qkv=save_all, save_attn_ctx=save_all, save_resid_mid=save_rm,
        _all_reduce=_ar,                      # what the trainer exposes (M3)
    )
    svc._layer_weights = lambda i: shards[i]
    runner = GraphedBackward(svc, s_max=S_MAX, bn_max=BN_MAX, l_max=L_MAX)

    results = {"fwd_failed": sorted(runner.fwd_failed),
               "ffn_failed": sorted(runner.ffn_failed),
               "attn_failed": runner.attn_failed, "batches": []}
    for bi, seq_lens in enumerate(BATCHES):
        n = sum(seq_lens)
        b_start, cos, sin = H.seq_layout(seq_lens, Hd, theta, device=device)
        runner.begin_backward(n, seq_lens, b_start)
        per_layer = []
        for i in range(L_):
            lw = shards[i]
            inp = full["inputs"][bi][i]
            x = inp["layer_in"].to(device).to(mdt)
            # Sharded gate||up: this rank's local gate rows ‖ local up rows.
            gu_full = inp["saved_gu"].to(device).to(mdt)
            gu = torch.cat([gu_full[:, rank * inter_l:(rank + 1) * inter_l],
                            gu_full[:, inter + rank * inter_l:
                                    inter + (rank + 1) * inter_l]], dim=1)
            gy = inp["g"].to(device)

            # Eager TP path (the M3-validated reference on this rank): the
            # full recompute, 1 forward reduce + 6 backward reduces.
            c0 = n_reduce[0]
            cache_e = FAM.layer_forward(x, lw, scaling, cos, sin, seq_lens,
                                        b_start, dims, eps, saved_gate_up=gu,
                                        all_reduce=_ar)
            gx_e, grads_e = FAM.layer_backward(
                gy, cache_e, lw, scaling, cos, sin, seq_lens, b_start, dims,
                eps, cdt=torch.float32, all_reduce=_ar)
            n_eager = n_reduce[0] - c0
            # save_resid_mid: feed the (already reduced, full) residual the
            # real forward would have captured to the graph path.
            saved = dict(saved_resid_mid=cache_e["resid_mid"].contiguous()) \
                if save_rm else {}
            if save_all:
                # What the accumulator captures: post-RoPE q/k for Llama-3,
                # the PRE-norm q/k (q_norm/k_norm inputs) for Qwen3; v; ctx.
                qk = ("q_pre", "k_pre") if FAM.saved_qkv_pre_transform else ("qh", "kh")
                saved.update(
                    saved_qh=cache_e[qk[0]].reshape(n, -1).contiguous(),
                    saved_kh=cache_e[qk[1]].reshape(n, -1).contiguous(),
                    saved_vh=cache_e["vh"].reshape(n, -1).contiguous(),
                    saved_ctx=cache_e["ctx_flat"].contiguous())
            # Graph TP path (M5): runner forward (+ eager o reduce + tail
            # unless the residual is saved), then the family's graphed
            # backward with the six eager reduces.
            c0 = n_reduce[0]
            cache_g = runner.forward(i, lw, x, gu, n, **saved)
            n_graph_fwd = n_reduce[0] - c0
            gx_g, grads_g = fam_mod.layer_backward_graphed(
                runner, lambda: None, i, gy, cache_g, lw, scaling, cos, sin,
                seq_lens, b_start, dims, eps, torch.float32, all_reduce=_ar)
            n_graph = n_reduce[0] - c0

            def _cpu(d, keys):
                return {k: d[k].detach().float().cpu().clone() for k in keys}

            per_layer.append({
                "eager": {**_cpu(cache_e, CACHE_KEYS), "grad_x": gx_e.float().cpu(),
                          **_cpu(grads_e, H.LORA_KEYS)},
                "graph": {**_cpu(cache_g, CACHE_KEYS), "grad_x": gx_g.float().cpu(),
                          **_cpu(grads_g, H.LORA_KEYS)},
                "attn_fit": runner._attn_fit,
                "n_eager": n_eager, "n_graph": n_graph,
                "n_graph_fwd": n_graph_fwd,
            })
        results["batches"].append(per_layer)

    torch.cuda.synchronize()
    torch.save(results, os.path.join(outdir, f"rank{rank}.pt"))
    dist.destroy_process_group()


def _reference(fam_name, mdt, full):
    """tp=1 eager reference on cuda:0 with the FULL weights."""
    fam_mod = FAMILIES[fam_name]
    FAM = fam_mod.LLAMA3 if fam_name == "llama3" else fam_mod.QWEN3
    g = GEOM[fam_name]
    Hq, Hkv, Hd = g["Hq"], g["Hkv"], g["Hd"]
    device = torch.device("cuda:0")
    dims = (Hq, Hkv, Hd, Hkv * Hd)
    lws = [_to(lw, device, mdt) for lw in full["lws"]]
    out = []
    for bi, seq_lens in enumerate(BATCHES):
        b_start, cos, sin = H.seq_layout(seq_lens, Hd, g["theta"], device=device)
        per_layer = []
        for i in range(L_):
            inp = full["inputs"][bi][i]
            x = inp["layer_in"].to(device).to(mdt)
            gu = inp["saved_gu"].to(device).to(mdt)
            gy = inp["g"].to(device)
            cache = FAM.layer_forward(x, lws[i], g["scaling"], cos, sin, seq_lens,
                                      b_start, dims, g["eps"], saved_gate_up=gu)
            gx, grads = FAM.layer_backward(
                gy, cache, lws[i], g["scaling"], cos, sin, seq_lens, b_start,
                dims, g["eps"], cdt=torch.float32)
            per_layer.append({"grad_x": gx.float().cpu(),
                              **{k: grads[k].float().cpu() for k in H.LORA_KEYS}})
        out.append(per_layer)
    return out


def run_family(fam_name, mdt, port, C, saves="none"):
    save_rm = saves in ("rm", "all")
    g = GEOM[fam_name]
    FAM = FAMILIES[fam_name].LLAMA3 if fam_name == "llama3" else FAMILIES[fam_name].QWEN3
    Hq, Hkv, Hd, inter, hidden = g["Hq"], g["Hkv"], g["Hd"], g["inter"], g["hidden"]
    tag = f"{fam_name}/{str(mdt).split('.')[-1]}" + ("" if saves == "none" else f"/save_{saves}")
    print(f"test_tp_backward_graph_nccl ({tag}, tp={TP}, real NCCL, "
          f"batches={BATCHES}):")

    gen = torch.Generator().manual_seed(11 if fam_name == "llama3" else 13)
    lws = [H.make_layer_weights(FAM, hidden, Hq, Hkv, Hd, inter, g["r"],
                                dtype=torch.float32, device="cpu", gen=gen)
           for _ in range(L_)]
    inputs = []
    for seq_lens in BATCHES:
        n = sum(seq_lens)
        inputs.append([{
            "layer_in": torch.randn(n, hidden, generator=gen),
            "saved_gu": torch.randn(n, 2 * inter, generator=gen),
            "g": torch.randn(n, hidden, generator=gen),
        } for _ in range(L_)])
    full = {"lws": lws, "inputs": inputs}

    tmp = tempfile.mkdtemp(prefix="tp_graph_")
    wpath = os.path.join(tmp, "weights.pt")
    torch.save(full, wpath)

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_worker,
                         args=(rk, fam_name, mdt, wpath, tmp, port, saves))
             for rk in range(TP)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    if any(p.exitcode != 0 for p in procs):
        C.ok(f"{tag} workers exited cleanly", False,
             f"exit codes {[p.exitcode for p in procs]}")
        return

    r = [torch.load(os.path.join(tmp, f"rank{rk}.pt")) for rk in range(TP)]
    for rk in range(TP):
        C.ok(f"{tag} r{rk} every region captured",
             not r[rk]["fwd_failed"] and not r[rk]["ffn_failed"]
             and not r[rk]["attn_failed"],
             f"fwd_failed={r[rk]['fwd_failed']} ffn_failed={r[rk]['ffn_failed']} "
             f"attn_failed={r[rk]['attn_failed']}")
    ref = _reference(fam_name, mdt, full)
    tol_ref = TOL_REF[mdt]
    ql, kl = (Hq // TP) * Hd, (Hkv // TP) * Hd

    for bi, seq_lens in enumerate(BATCHES):
        fits = max(seq_lens) <= L_MAX and len(seq_lens) <= BN_MAX
        C.ok(f"{tag} b{bi} attn_fit={fits} as expected",
             all(r[rk]["batches"][bi][0]["attn_fit"] == fits for rk in range(TP)))
        for i in range(L_):
            lay = [r[rk]["batches"][bi][i] for rk in range(TP)]
            # (4) collective count per layer: eager 1 + 6; graph 6 + (0 if the
            # residual is saved else 1). Same on both ranks (lock-step).
            for rk in range(TP):
                C.ok(f"{tag} b{bi} L{i} r{rk} eager reduces == 7",
                     lay[rk]["n_eager"] == 7, f"got {lay[rk]['n_eager']}")
                want_fwd = 0 if save_rm else 1
                C.ok(f"{tag} b{bi} L{i} r{rk} graph fwd reduces == {want_fwd}",
                     lay[rk]["n_graph_fwd"] == want_fwd,
                     f"got {lay[rk]['n_graph_fwd']}")
                C.ok(f"{tag} b{bi} L{i} r{rk} graph reduces == {6 + want_fwd}",
                     lay[rk]["n_graph"] == 6 + want_fwd, f"got {lay[rk]['n_graph']}")
            # (1) graph == eager on each rank.
            for rk in range(TP):
                for k in CACHE_KEYS + ("grad_x",) + H.LORA_KEYS:
                    C.check(f"{tag} b{bi} L{i} r{rk} graph==eager {k}",
                            lay[rk]["graph"][k], lay[rk]["eager"][k],
                            tol=TOL_SAME_MATH)
            # (2) reduced tensors identical across ranks.
            for k in ("grad_x", "qA", "kA", "vA", "oB", "resid_mid"):
                C.check(f"{tag} b{bi} L{i} graph r0==r1 {k}",
                        lay[0]["graph"][k], lay[1]["graph"][k], tol=TOL_SAME_MATH)
            # (3) graph-TP vs the tp=1 reference.
            rf = ref[bi][i]
            C.check(f"{tag} b{bi} L{i} graph vs tp1 grad_x",
                    lay[0]["graph"]["grad_x"], rf["grad_x"], tol=tol_ref)
            for k in ("qA", "kA", "vA", "oB"):
                C.check(f"{tag} b{bi} L{i} graph vs tp1 {k}",
                        lay[0]["graph"][k], rf[k], tol=tol_ref)
            for k in ("qB", "kB", "vB"):
                C.check(f"{tag} b{bi} L{i} graph vs tp1 {k} (concat)",
                        torch.cat([lay[0]["graph"][k], lay[1]["graph"][k]], 0),
                        rf[k], tol=tol_ref)
            C.check(f"{tag} b{bi} L{i} graph vs tp1 oA (concat)",
                    torch.cat([lay[0]["graph"]["oA"], lay[1]["graph"]["oA"]], 1),
                    rf["oA"], tol=tol_ref)
    del ql, kl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["llama3", "qwen3", "all"], default="all")
    ap.add_argument("--quiet", action="store_true",
                    help="print only failures and the summary")
    args = ap.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() < TP:
        print(f"need {TP} CUDA devices — skipping TP graph parity test")
        sys.exit(0)

    C = H.Checker(tol=TOL_SAME_MATH)
    if args.quiet:
        _orig = C.check

        def _check(name, got, ref, tol=None):
            diff = (got.float() - ref.float()).abs().max().item()
            scale = ref.float().abs().max().item() + 1e-8
            rel = diff / scale
            ok = rel < (C.tol if tol is None else tol)
            if not ok:
                print(f"  [FAIL] {name:60s} max-rel-err={rel:.3e}")
            C.passed += ok
            C.failed += (not ok)
            return ok
        C.check = _check
        del _orig

    fams = ["llama3", "qwen3"] if args.family == "all" else [args.family]
    port = BASE_PORT
    for fam in fams:
        for mdt in (torch.float32, torch.bfloat16):
            for saves in ("none", "rm", "all"):
                run_family(fam, mdt, port, C, saves=saves)
                port += 1
    C.finish()


if __name__ == "__main__":
    main()
