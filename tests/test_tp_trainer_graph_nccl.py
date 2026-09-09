#!/usr/bin/env python
"""TP=2 trainer gate on two real GPUs (NCCL): M5 graphs + M4.3 bucketing.

Runs the real ``LoraSftTrainerService.process_backward`` — ``_build_state``
standing up the backward NCCL group, the graph runner, the M4.3
``CommQueue`` + ``FactorBucket`` and the rank-symmetric clip, fused AdamW —
for a few training steps on a tiny synthetic model, in five configurations:

  graph          backward_cuda_graph on, no activation saves
  eager          backward_cuda_graph off
  graph+save     graph + save_resid_mid (no forward collective)
  graph+sync     graph with DSERVE_BWD_COMM_SYNC=1 (wait after every submit)
  graph+delay    graph with DSERVE_BWD_COMM_DELAY (comm stream spins before
                 each collective, so a missing wait reads stale data reliably)

and checks:

  1. graph == eager == graph+save: same loss trajectory and masters (1e-5);
  2. graph == graph+sync == graph+delay BIT FOR BIT (the async comm path
     must not change a single value — that is the stream-ordering gate);
  3. both ranks hold identical replicated masters, and TP2 == the tp=1 eager
     reference, masters and losses, WITH the per-layer clip firing (norms
     exceed 1.0 on this model): the rank-symmetric clip closes the M4.3 trap;
  4. the collective count per cycle: 2 residual reduces per layer (+1 forward
     ``o`` reduce per layer unless the residual is saved) + 1 bucket per group
     of layers + 1 for the clip norms — no inline factor-grad reduces.

fp32 throughout so the tolerances are tight. Requires 2 CUDA devices.

    python tests/test_tp_trainer_graph_nccl.py [--family llama3|qwen3|all]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bwd_harness as H  # noqa: E402

sys.path[:] = [p for p in sys.path
               if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]

from vllm.deltaserve.bwd_services.common.ops import rmsnorm, rope_cos_sin  # noqa: E402
from vllm.deltaserve.bwd_services.llama3 import Llama3BackwardService  # noqa: E402
from vllm.deltaserve.bwd_services.qwen3 import Qwen3BackwardService  # noqa: E402

SERVICES = {"llama3": Llama3BackwardService, "qwen3": Qwen3BackwardService}
TP = 2
BASE_PORT = 29700

GEOM = {
    "llama3": dict(hidden=32, Hq=4, Hkv=2, Hd=8, inter=64, vocab=64, r=4,
                   eps=1e-5, theta=500000.0),
    "qwen3": dict(hidden=24, Hq=4, Hkv=2, Hd=8, inter=64, vocab=64, r=4,
                  eps=1e-6, theta=1e6),
}
NL, SCALING, LR = 3, 2.0, 1e-2
BUCKET_LAYERS = 2            # NL=3 → 2 buckets per cycle (exercises the grouping)
S_MAX, BN_MAX, L_MAX = 16, 4, 8
SEQ_LENS = [6, 4]
STEPS = 4
TOL_SAME_MATH = 1e-5
TOL_REF = 1e-4
MODES = ["graph", "eager", "graph+save", "graph+sync", "graph+delay"]


def build_full(fam_name, seed):
    """Full (unsharded) fp32 base + FT adapter in the HF/PEFT naming the worker
    shares, plus the embedding and a fixed token batch."""
    torch.manual_seed(seed)
    g = GEOM[fam_name]
    hidden, Hq, Hkv, Hd, inter, vocab, r = (g["hidden"], g["Hq"], g["Hkv"], g["Hd"],
                                            g["inter"], g["vocab"], g["r"])
    fam = SERVICES[fam_name].family
    q_size, kv = Hq * Hd, Hkv * Hd

    def w(*s):
        return torch.randn(*s) * (1.0 / (s[-1] ** 0.5))

    def norm(n):
        return torch.randn(n).abs() + 0.5

    embed = torch.randn(vocab, hidden) * 0.1
    base = {"model.embed_tokens.weight": embed,
            "model.norm.weight": norm(hidden), "lm_head.weight": w(vocab, hidden)}
    shapes = {"qkv": (q_size + 2 * kv, hidden), "o": (hidden, q_size),
              "gate_up": (2 * inter, hidden), "down": (hidden, inter)}
    for i in range(NL):
        p = f"model.layers.{i}."
        for key, rel in fam.layer_weights.items():
            if key in shapes:
                base[p + rel] = w(*shapes[key])
            elif key in ("in_ln", "post_ln"):
                base[p + rel] = norm(hidden)
            else:
                base[p + rel] = norm(Hd)
    ft = {}
    for i in range(NL):
        pre = f"base_model.model.model.layers.{i}.self_attn."
        for proj, out, inp in [("q", q_size, hidden), ("k", kv, hidden),
                               ("v", kv, hidden), ("o", hidden, q_size)]:
            ft[pre + f"{proj}_proj.lora_A.weight"] = torch.randn(r, inp) * 0.02
            # Non-zero B so the very first step already has non-trivial grads
            # on every factor (PEFT's B=0 init would zero the A grads).
            ft[pre + f"{proj}_proj.lora_B.weight"] = torch.randn(out, r) * 0.02
    ids = torch.randint(0, vocab, (sum(SEQ_LENS),))
    return {"base": base, "ft": ft, "embed": embed, "ids": ids}


def shard_base(base, rank, tp, fam_name):
    """This rank's shard of the full base dict, laid out as vLLM shares it:
    qkv/gate_up column-parallel (output rows, per-projection contiguous),
    o/down row-parallel (input cols); everything else replicated."""
    g = GEOM[fam_name]
    Hq, Hkv, Hd, inter = g["Hq"], g["Hkv"], g["Hd"], g["inter"]
    q_size, kv = Hq * Hd, Hkv * Hd
    ql, kl, il = q_size // tp, kv // tp, inter // tp
    out = {}
    for k, t in base.items():
        if k.endswith("self_attn.qkv_proj.weight"):
            q, kk, v = t[:q_size], t[q_size:q_size + kv], t[q_size + kv:]
            out[k] = torch.cat([q[rank * ql:(rank + 1) * ql],
                                kk[rank * kl:(rank + 1) * kl],
                                v[rank * kl:(rank + 1) * kl]], 0)
        elif k.endswith("self_attn.o_proj.weight"):
            out[k] = t[:, rank * ql:(rank + 1) * ql]
        elif k.endswith("mlp.gate_up_proj.weight"):
            gate, up = t[:inter], t[inter:]
            out[k] = torch.cat([gate[rank * il:(rank + 1) * il],
                                up[rank * il:(rank + 1) * il]], 0)
        elif k.endswith("mlp.down_proj.weight"):
            out[k] = t[:, rank * il:(rank + 1) * il]
        else:
            out[k] = t
    return {k: v.contiguous() for k, v in out.items()}


def make_meta(fam_name, *, tp_size, tp_rank, port, graph, save_rm=False):
    g = GEOM[fam_name]
    return dict(hidden_size=g["hidden"], num_hidden_layers=NL,
                num_attention_heads=g["Hq"], num_key_value_heads=g["Hkv"],
                head_dim=g["Hd"], intermediate_size=g["inter"],
                rope_theta=g["theta"], rms_norm_eps=g["eps"], lora_scaling=SCALING,
                vocab_size=g["vocab"], learning_rate=LR, weight_decay=0.0,
                gamma=1.0, backward_fp32=True, lm_head_key="lm_head.weight",
                norm_weight_key="model.norm.weight",
                embed_weight_key="model.embed_tokens.weight",
                tp_size=tp_size, tp_rank=tp_rank, backward_nccl_port=port,
                backward_cuda_graph=graph, backward_cuda_graph_attn_bn_max=BN_MAX,
                # exercise the run-ahead-bounded _maybe_pause ring (event
                # record + synchronize per boundary) on the real trainer
                backward_run_ahead_boundaries=2,
                backward_cuda_graph_attn_l_max=L_MAX,
                max_saved_finetuning_tokens=S_MAX,
                save_attn_qkv=False, save_attn_ctx=False,
                save_resid_mid=save_rm, bucket_layers=BUCKET_LAYERS)


def make_service(fam_name, base, ft, meta, device):
    svc = SERVICES[fam_name](device.index if device.type == "cuda" else 0)
    svc.shared = {"base": {k: v.to(device) for k, v in base.items()},
                  "ft": {k: v.to(device) for k, v in ft.items()},
                  "meta": meta}
    svc._build_state()
    svc._built = True
    # A grant that is always SET: exercises the rank-symmetric pause's
    # per-boundary gloo agreement (tp>1) without ever blocking.
    import multiprocessing as _mp
    svc._gpu_grant = _mp.Event()
    svc._gpu_grant.set()
    return svc


@torch.no_grad()
def remat_activations(svc, embed, ids, seq_lens, b_start):
    """Full forward with the CURRENT masters → the activation dict
    ``process_backward`` consumes, incl. the per-layer (local) gate||up so the
    forward-graph path runs, and the post-attention residual (full width,
    already reduced under TP) for the save_resid_mid mode."""
    positions = torch.cat([torch.arange(s, device=embed.device) for s in seq_lens])
    cos, sin = rope_cos_sin(positions, svc.Hd, svc.theta)
    x = embed[ids]
    layer_in, gate_up, resid_mid = [], [], []
    for i in range(svc.L):
        layer_in.append(x)
        lw = svc._layer_weights(i)
        cache = svc._layer_forward(x, lw, svc.scaling, cos, sin, seq_lens, b_start,
                                   svc.dims, svc.eps, all_reduce=svc._all_reduce)
        gate_up.append(torch.cat([cache["gate"], cache["up"]], 1).contiguous())
        resid_mid.append(cache["resid_mid"].contiguous())
        ffn = F.linear(F.silu(cache["gate"]) * cache["up"], lw["down"])
        if svc._all_reduce is not None:
            ffn = svc._all_reduce(ffn)
        x = cache["resid_mid"] + ffn
    return {"layer_in": layer_in, "final_in": x,
            "final_hidden": rmsnorm(x, svc.norm_w, svc.eps),
            "concat_input_ids": ids, "mlp_gate_up": gate_up,
            "resid_mid": resid_mid}


class ReduceCounter:
    """Counts ``dist.all_reduce`` calls (the trainer calls it through the
    module attribute, so patching the attribute sees every collective)."""

    def __init__(self):
        self.n = 0
        self._real = dist.all_reduce
        dist.all_reduce = self._wrapped

    def _wrapped(self, *a, **k):
        # Count only the data-path collectives (NCCL, default group). The
        # rank-symmetric pause agrees over the children's gloo CPU group
        # (``group=`` set) — not a GPU collective, not counted.
        if k.get("group") is None:
            self.n += 1
        return self._real(*a, **k)


def train(svc, embed, ids, counter=None):
    n = sum(SEQ_LENS)
    b_start = [sum(SEQ_LENS[:i]) for i in range(len(SEQ_LENS))]
    losses, counts = [], []
    for _ in range(STEPS):
        acts = remat_activations(svc, embed, ids, SEQ_LENS, b_start)
        c0 = counter.n if counter else 0
        loss, _ = svc.process_backward(acts, SEQ_LENS, n, epoch=0)
        counts.append((counter.n - c0) if counter else 0)
        losses.append(float(loss))
    masters = {f"L{i}.{proj}{ab}": svc.lora[i][proj][ab].detach().float().cpu().clone()
               for i in range(svc.L) for proj in ("q", "k", "v", "o")
               for ab in ("A", "B")}
    return losses, masters, counts


def _worker(rank, fam_name, mode, wpath, outdir, port):
    try:
        if mode == "graph+sync":
            os.environ["DSERVE_BWD_COMM_SYNC"] = "1"
        if mode == "graph+delay":
            os.environ["DSERVE_BWD_COMM_DELAY"] = str(20_000_000)   # ~10 ms spin
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        full = torch.load(wpath)
        graph = mode != "eager"
        save_rm = mode == "graph+save"
        meta = make_meta(fam_name, tp_size=TP, tp_rank=rank, port=port, graph=graph,
                         save_rm=save_rm)
        counter = ReduceCounter()
        svc = make_service(fam_name, shard_base(full["base"], rank, TP, fam_name),
                           full["ft"], meta, device)
        assert (svc.graph_runner is not None) == graph
        assert svc._bucket is not None and svc._comm is not None, "M4.3 not armed"
        if mode == "graph+sync":
            assert svc._comm.sync_mode
        if mode == "graph+delay":
            assert svc._comm.delay_cycles > 0
        if graph:
            r = svc.graph_runner
            assert not r.fwd_failed and not r.ffn_failed and not r.attn_failed
        losses, masters, counts = train(svc, full["embed"].to(device),
                                        full["ids"].to(device), counter)
        torch.cuda.synchronize()
        torch.save({"losses": losses, "masters": masters, "mode": svc._last_mode,
                    "counts": counts, "n_groups": svc._bucket.num_groups},
                   os.path.join(outdir, f"rank{rank}.pt"))
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)


def run_tp(fam_name, mode, full, port):
    tmp = tempfile.mkdtemp(prefix="tp_trainer_")
    wpath = os.path.join(tmp, "full.pt")
    torch.save(full, wpath)
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_worker, args=(rk, fam_name, mode, wpath, tmp, port))
             for rk in range(TP)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    if any(p.exitcode != 0 for p in procs):
        return None
    return [torch.load(os.path.join(tmp, f"rank{rk}.pt")) for rk in range(TP)]


def run_family(fam_name, port, C):
    g = GEOM[fam_name]
    print(f"test_tp_trainer_graph_nccl ({fam_name}, tp={TP}, {STEPS} steps, "
          f"seq_lens={SEQ_LENS}, NL={NL}, bucket_layers={BUCKET_LAYERS}):")
    full = build_full(fam_name, seed=21 if fam_name == "llama3" else 23)

    res = {}
    for k, mode in enumerate(MODES):
        res[mode] = run_tp(fam_name, mode, full, port + k)
        if res[mode] is None:
            C.ok(f"{fam_name} {mode}: workers exited cleanly", False)
            return
    C.ok(f"{fam_name} modes took the intended path",
         all(res[m][0]["mode"] == ("eager" if m == "eager" else "graph") for m in MODES))

    # tp=1 eager reference on cuda:0 (after the children are done).
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    ref_svc = make_service(fam_name, full["base"], full["ft"],
                           make_meta(fam_name, tp_size=1, tp_rank=0, port=0,
                                     graph=False), device)
    ref_losses, ref_masters, _ = train(ref_svc, full["embed"].to(device),
                                       full["ids"].to(device))
    # The clip must actually fire on this model for the symmetric-clip check
    # to mean anything: peek at the per-layer norm of one more step.
    from vllm.deltaserve.bwd_services.common.tp import clip_layers_symmetric_
    n = sum(SEQ_LENS); b_start = [0, 6]
    acts = remat_activations(ref_svc, full["embed"].to(device), full["ids"].to(device),
                             SEQ_LENS, b_start)
    ref_svc.optimizer.zero_grad(set_to_none=True)
    ref_svc.process_backward(acts, SEQ_LENS, n, epoch=0)   # leaves .grad populated
    norms = clip_layers_symmetric_(ref_svc.lora, ref_svc.L, None, max_norm=float("inf"))
    print(f"    per-layer grad norms (tp1, post-clip; 1.00 = the clip fired): "
          f"{[f'{v:.2f}' for v in norms.tolist()]}")
    C.ok(f"{fam_name} the clip fires on this model", bool((norms > 0.999).any()))

    G = res["graph"]
    print(f"    losses graph-tp2 {['%.5f' % v for v in G[0]['losses']]}")
    print(f"    losses tp1-ref   {['%.5f' % v for v in ref_losses]}")
    print(f"    collectives per cycle: graph {G[0]['counts']}, "
          f"graph+save {res['graph+save'][0]['counts']}, eager {res['eager'][0]['counts']}")

    # (1) same math across graph / eager / graph+save.
    for m in ("eager", "graph+save"):
        for s in range(STEPS):
            C.check(f"{fam_name} step{s} loss {m} == graph",
                    torch.tensor(res[m][0]["losses"][s]),
                    torch.tensor(G[0]["losses"][s]), tol=TOL_SAME_MATH)
        for k in G[0]["masters"]:
            C.check(f"{fam_name} master {k} {m} == graph",
                    res[m][0]["masters"][k], G[0]["masters"][k], tol=TOL_SAME_MATH)
    # (2) async comm path bit-identical to sync mode and to the delayed mode.
    for m in ("graph+sync", "graph+delay"):
        C.ok(f"{fam_name} {m}: losses bit-identical to graph",
             res[m][0]["losses"] == G[0]["losses"])
        C.ok(f"{fam_name} {m}: masters bit-identical to graph (both ranks)",
             all(torch.equal(res[m][rk]["masters"][k], G[rk]["masters"][k])
                 for rk in range(TP) for k in G[0]["masters"]))
    # (3) lock-step + tp=1 parity WITH the clip on.
    for s in range(STEPS):
        C.check(f"{fam_name} step{s} loss r0 == r1", torch.tensor(G[0]["losses"][s]),
                torch.tensor(G[1]["losses"][s]), tol=TOL_SAME_MATH)
        C.check(f"{fam_name} step{s} loss tp2 == tp1", torch.tensor(G[0]["losses"][s]),
                torch.tensor(ref_losses[s]), tol=TOL_REF)
    C.ok(f"{fam_name} loss decreases", G[0]["losses"][-1] < G[0]["losses"][0],
         f"{G[0]['losses'][0]:.5f} → {G[0]['losses'][-1]:.5f}")
    for k in ref_masters:
        if k.endswith(("qA", "kA", "vA", "oB")):
            C.check(f"{fam_name} master {k} r0 == r1", G[0]["masters"][k],
                    G[1]["masters"][k], tol=TOL_SAME_MATH)
            C.check(f"{fam_name} master {k} tp2 == tp1", G[0]["masters"][k],
                    ref_masters[k], tol=TOL_REF)
    for i in range(NL):
        for proj in ("q", "k", "v"):
            k = f"L{i}.{proj}B"
            C.check(f"{fam_name} master {k} tp2 concat == tp1",
                    torch.cat([G[0]["masters"][k], G[1]["masters"][k]], 0),
                    ref_masters[k], tol=TOL_REF)
        k = f"L{i}.oA"
        C.check(f"{fam_name} master {k} tp2 concat == tp1",
                torch.cat([G[0]["masters"][k], G[1]["masters"][k]], 1),
                ref_masters[k], tol=TOL_REF)
    # (4) collective count per cycle.
    n_groups = math.ceil(NL / BUCKET_LAYERS)
    C.ok(f"{fam_name} bucket groups == {n_groups}", G[0]["n_groups"] == n_groups)
    for m, fwd in (("graph", 1), ("eager", 1), ("graph+save", 0)):
        want = NL * (2 + fwd) + n_groups + 1
        C.ok(f"{fam_name} {m}: {want} collectives per cycle on both ranks",
             all(c == want for rk in range(TP) for c in res[m][rk]["counts"]),
             f"got {res[m][0]['counts']}")
    del g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["llama3", "qwen3", "all"], default="all")
    args = ap.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < TP:
        print(f"need {TP} CUDA devices — skipping TP trainer graph test")
        sys.exit(0)
    C = H.Checker(tol=TOL_SAME_MATH)
    fams = ["llama3", "qwen3"] if args.family == "all" else [args.family]
    port = BASE_PORT
    for fam in fams:
        run_family(fam, port, C)
        port += len(MODES)
    C.finish()


if __name__ == "__main__":
    main()
