#!/usr/bin/env python
"""Trained-adapter checkpoints (POST /save_adapter + periodic saves), on CPU.

  A. ``adapter_io``: ``merge_tp_shards`` is the exact inverse of the production
     ``lora_shard_slice`` (tp 2 and 4); result files / pruning.
  B. trainer, tp=1: six backward cycles through the real cycle handler
     (``_handle_process_activations``, a fake pipe) with
     ``adapter_save_every_backwards=2``, ``keep_last=2`` and
     ``adapter_save_every_epoch`` over epochs 0,0,0,1,1,2: the right
     checkpoints exist, each holds bit-exactly the masters of its moment
     (``epoch-E`` = after the epoch's last backward), the ack goes out before
     the write, keys are the source adapter's, the source config is copied.
     A manual save (the ``save_adapter`` command) holds the current masters.
  C. loadable: the checkpoint loads with PEFT onto a transformers Llama built
     from the same base weights (llama3 RoPE scaling on), and its logits match
     our forward with the trained masters; the untrained source adapter's do
     not (control).
  D. failures are reported, not lost: save before the state is built, an
     unwritable target, a service that trains nothing → ``save_error.json``
     that ``read_save_result`` returns.
  E. TP=2 over gloo (two processes): each rank holds its shard; a manual save
     and a periodic save write ONE merged adapter equal to the full masters;
     a rank that fails still joins the gather and rank 0 reports its error;
     the next save works (no hang).

    python tests/test_adapter_save.py
"""

import json
import os
import socket
import sys
import tempfile

import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # spawned TP ranks
import bwd_harness as H  # noqa: E402

sys.path[:] = [p for p in sys.path
               if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]

from deltaserve.adapter_io import (  # noqa: E402
    ERROR_FILE,
    STATE_FILE,
    WEIGHTS_FILE,
    merge_tp_shards,
    prune_checkpoints,
    read_save_result,
    write_peft_adapter,
    write_save_error,
)
from deltaserve.bwd_services.base import BackwardService  # noqa: E402
from deltaserve.bwd_services.common.ops import rmsnorm, rope_cos_sin  # noqa: E402
from deltaserve.bwd_services.common.tp import lora_shard_slice  # noqa: E402
from deltaserve.bwd_services.llama3 import Llama3BackwardService  # noqa: E402

C = H.Checker(tol=1e-5)
D, NL, Hq, Hkv, Hd, INTER, VOCAB, R = 64, 2, 4, 2, 16, 96, 64, 4
THETA, EPS, SCALING = 5e5, 1e-5, 2.0
ROPE = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
        "high_freq_factor": 4.0, "original_max_position_embeddings": 64}


def load_st(path):
    from safetensors.torch import load_file
    return load_file(os.path.join(path, WEIGHTS_FILE))


def same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def masters(svc) -> dict:
    return {svc._ft_keys[k]: svc.lora[k[0]][k[1]][k[2]].detach().clone()
            for k in svc._ft_keys}


def make_source_adapter(root, ft):
    """A PEFT adapter dir for the synthetic model (what finetuning_lora_path
    points at)."""
    src = os.path.join(root, "source-adapter")
    os.makedirs(src)
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in ft.items()},
              os.path.join(src, WEIGHTS_FILE))
    with open(os.path.join(src, "adapter_config.json"), "w") as f:
        json.dump({"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": R,
                   "lora_alpha": SCALING * R, "lora_dropout": 0.0,
                   "bias": "none", "base_model_name_or_path": "synthetic-llama",
                   "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}, f)
    return src


def build(root, **policy):
    svc, embed = H.build_synthetic_service(
        Llama3BackwardService, hidden=D, n_layers=NL, Hq=Hq, Hkv=Hkv, Hd=Hd,
        inter=INTER, vocab=VOCAB, r=R, lr=3e-2, theta=THETA, eps=EPS,
        scaling=SCALING, rope_scaling=ROPE)
    svc.shared["meta"].update(finetuning_lora_path=make_source_adapter(
        root, svc.shared["ft"]), **policy)
    return svc, embed


def capture(svc, embed, ids, seq):
    """Forward with the current masters → the activations the worker saves."""
    b_start = [sum(seq[:i]) for i in range(len(seq))]
    pos = torch.cat([torch.arange(s) for s in seq])
    cos, sin = rope_cos_sin(pos, svc.Hd, svc.theta, svc.rope_scaling)
    x = embed[ids]
    layer_in = []
    for i in range(svc.L):
        layer_in.append(x.clone())
        lw = svc._layer_weights(i)
        x = H.layer_output(svc._layer_forward(x, lw, svc.scaling, cos, sin, seq,
                                              b_start, svc.dims, svc.eps), lw)
    return {"layer_in": layer_in, "final_in": x.clone(),
            "concat_input_ids": ids.clone()}


class Pipe:
    def __init__(self, on_send=None):
        self.sent, self.on_send = [], on_send

    def send(self, m):
        self.sent.append(m)
        if self.on_send:
            self.on_send(m)


def main_cpu():
    # --------------------------------------------------------------------------- #
    print("A. adapter_io")
    g = torch.Generator().manual_seed(0)
    full = {}
    for i in range(2):
        for proj, out, inp in [("q", 8 * 4, 24), ("k", 4 * 4, 24), ("v", 4 * 4, 24),
                               ("o", 24, 8 * 4)]:
            full[(i, proj, "A")] = torch.randn(R, inp, generator=g)
            full[(i, proj, "B")] = torch.randn(out, R, generator=g)
    for tp in (2, 4):
        shards = [{k: lora_shard_slice(k[1], k[2], t, rk, tp, (8 // tp) * 4, (4 // tp) * 4)
                   .clone() for k, t in full.items()} for rk in range(tp)]
        C.ok(f"merge_tp_shards inverts lora_shard_slice (tp={tp})",
             same(merge_tp_shards(shards), full))
    with tempfile.TemporaryDirectory() as root:
        out = os.path.join(root, "a")
        write_peft_adapter(out, {"x.q_proj.lora_A.weight": torch.ones(R, 3),
                                 "x.q_proj.lora_B.weight": torch.ones(3, R)},
                           source_dir=None, scaling=2.0, state={"save_id": "s1"})
        cfg = json.load(open(os.path.join(out, "adapter_config.json")))
        C.ok("minimal adapter_config without a source",
             cfg["r"] == R and cfg["lora_alpha"] == 2.0 * R
             and cfg["target_modules"] == ["q_proj"], str(cfg)[:70])
        C.ok("read_save_result: saved", (read_save_result(out, "s1") or {}).get("saved") is True)
        C.ok("read_save_result: other save_id → None", read_save_result(out, "s2") is None)
        write_save_error(out, "s2", "boom")
        r = read_save_result(out, "s2") or {}
        C.ok("read_save_result: error", r.get("saved") is False and r.get("error") == "boom")
        write_peft_adapter(out, load_st(out), source_dir=None, scaling=2.0,
                           state={"save_id": "s3"})
        C.ok("a later save clears the stale error file",
             not os.path.exists(os.path.join(out, ERROR_FILE)))
        C.ok("no temp files left", not [f for f in os.listdir(out) if ".tmp-" in f],
             str(os.listdir(out)))
        for n in (1, 2, 10, 3):
            os.makedirs(os.path.join(root, f"backward-{n:06d}"))
        os.makedirs(os.path.join(root, "epoch-0"))
        prune_checkpoints(root, "backward-", 2)
        C.ok("prune keeps the newest 2, leaves others",
             sorted(os.listdir(root)) == ["a", "backward-000003", "backward-000010",
                                          "epoch-0"], str(sorted(os.listdir(root))))


    # --------------------------------------------------------------------------- #
    print("\nB. trainer, tp=1: periodic + epoch + manual saves")
    EPOCHS = [0, 0, 0, 1, 1, 2]
    with tempfile.TemporaryDirectory() as root:
        ck = os.path.join(root, "ckpt")
        svc, embed = build(root, adapter_save_path=ck, adapter_save_every_backwards=2,
                           adapter_save_every_epoch=True, adapter_save_keep_last=2)
        source = {k: v.clone() for k, v in svc.shared["ft"].items()}
        after = {}           # cycle -> masters after it
        acked_before_write = []

        def on_send(m, cyc=[0]):
            cyc[0] += 1
            acked_before_write.append(
                not os.path.exists(os.path.join(ck, f"backward-{cyc[0]:06d}")))

        pipe = Pipe(on_send)
        # One fixed batch every cycle, so the loss must fall (overfit).
        seq = [40, 25]
        ids = torch.randint(0, VOCAB, (sum(seq),),
                            generator=torch.Generator().manual_seed(3))
        for c, ep in enumerate(EPOCHS, start=1):
            svc.activations = capture(svc, embed, ids, seq)
            svc._handle_process_activations(pipe, {"n": sum(seq), "sample_lens": seq,
                                                   "epoch": ep})
            after[c] = masters(svc)
        losses = [m["loss"] for m in pipe.sent]
        C.ok("six cycles acked, loss finite + falling",
             len(losses) == 6 and all(l is not None for l in losses)
             and all(b < a for a, b in zip(losses, losses[1:])),
             f"{[round(l, 3) for l in losses]}")
        C.ok("ack sent before the periodic write", all(acked_before_write))
        C.ok("trained masters moved away from the source",
             not same(after[6], source))
        names = sorted(os.listdir(ck))
        C.ok("checkpoints: backward-4/6 (pruned 2), epoch-0/1",
             names == ["backward-000004", "backward-000006", "epoch-0", "epoch-1"],
             str(names))
        for name, cyc in [("backward-000004", 4), ("backward-000006", 6),
                          ("epoch-0", 3), ("epoch-1", 5)]:
            d = os.path.join(ck, name)
            C.ok(f"{name} == masters after cycle {cyc}", same(load_st(d), after[cyc]))
        st = json.load(open(os.path.join(ck, "epoch-0", STATE_FILE)))
        C.ok("epoch-0 trainer_state", st["reason"] == "epoch" and st["epoch"] == 0
             and st["save_id"] == "epoch-0" and st["backward_cycles"] == 3
             and abs(st["last_loss"] - losses[2]) < 1e-9
             and st["tokens_trained"] == 3 * 65, str(st)[:90])
        st = json.load(open(os.path.join(ck, "backward-000006", STATE_FILE)))
        C.ok("backward-6 trainer_state", st["backward_cycles"] == 6
             and st["tokens_trained"] == 6 * 65
             and abs(st["last_loss"] - losses[-1]) < 1e-6, str(st)[:90])
        C.ok("keys are the source adapter's",
             load_st(os.path.join(ck, "epoch-0")).keys() == source.keys())
        C.ok("adapter_config copied from the source",
             json.load(open(os.path.join(ck, "epoch-0", "adapter_config.json")))
             == json.load(open(os.path.join(svc.shared["meta"]["finetuning_lora_path"],
                                            "adapter_config.json"))))
        # Manual save through the child's command handler.
        man = os.path.join(root, "manual")
        svc._handle_save_adapter({"cmd": "save_adapter", "path": man, "save_id": "m1"})
        res = read_save_result(man, "m1") or {}
        C.ok("manual save: saved + current masters",
             res.get("saved") is True and same(load_st(man), after[6]), str(res)[:80])
        C.ok("manual save sends nothing on the pipe", len(pipe.sent) == 6)
        trained_dir, trained_svc, trained_embed = man, svc, embed

        # ---------------------------------------------------------------------- #
        print("\nC. the checkpoint loads with PEFT and serves the trained model")
        from peft import PeftModel
        from transformers import LlamaConfig, LlamaForCausalLM

        base = trained_svc.shared["base"]
        cfg = LlamaConfig(hidden_size=D, num_hidden_layers=NL, num_attention_heads=Hq,
                          num_key_value_heads=Hkv, head_dim=Hd, intermediate_size=INTER,
                          vocab_size=VOCAB, rms_norm_eps=EPS, max_position_embeddings=512,
                          rope_parameters={"rope_theta": THETA, **ROPE},
                          tie_word_embeddings=False, attention_bias=False,
                          mlp_bias=False)
        cfg._attn_implementation = "eager"
        hf = LlamaForCausalLM(cfg).eval()
        q, kv = Hq * Hd, Hkv * Hd
        sd = {"model.embed_tokens.weight": base["model.embed_tokens.weight"],
              "model.norm.weight": base["model.norm.weight"],
              "lm_head.weight": base["lm_head.weight"]}
        for i in range(NL):
            p = f"model.layers.{i}."
            qkv = base[p + "self_attn.qkv_proj.weight"]
            gu = base[p + "mlp.gate_up_proj.weight"]
            sd.update({p + "self_attn.q_proj.weight": qkv[:q],
                       p + "self_attn.k_proj.weight": qkv[q:q + kv],
                       p + "self_attn.v_proj.weight": qkv[q + kv:],
                       p + "self_attn.o_proj.weight": base[p + "self_attn.o_proj.weight"],
                       p + "mlp.gate_proj.weight": gu[:INTER],
                       p + "mlp.up_proj.weight": gu[INTER:],
                       p + "mlp.down_proj.weight": base[p + "mlp.down_proj.weight"],
                       p + "input_layernorm.weight": base[p + "input_layernorm.weight"],
                       p + "post_attention_layernorm.weight":
                           base[p + "post_attention_layernorm.weight"]})
        missing, unexpected = hf.load_state_dict(sd, strict=False)
        C.ok("HF Llama built from the shared base weights",
             not unexpected and all("rotary" in m for m in missing),
             f"missing={missing} unexpected={unexpected}")
        ids = torch.randint(0, VOCAB, (1, 120), generator=torch.Generator().manual_seed(9))
        acts = capture(trained_svc, trained_embed, ids[0], [120])
        ours = rmsnorm(acts["final_in"], trained_svc.norm_w, EPS) @ trained_svc.lm_w.t()
        with torch.no_grad():
            got = PeftModel.from_pretrained(hf, trained_dir).eval()(ids).logits[0]
            hf2 = LlamaForCausalLM(cfg).eval()
            hf2.load_state_dict(sd, strict=False)
            src_logits = PeftModel.from_pretrained(
                hf2, trained_svc.shared["meta"]["finetuning_lora_path"]).eval()(ids).logits[0]
        C.check("PEFT(trained ckpt) logits == ours", got, ours, tol=1e-4)
        d = ((src_logits - ours).abs().max() / ours.abs().max()).item()
        C.ok("control: source adapter's logits differ", d > 1e-3, f"rel={d:.2e}")


    # --------------------------------------------------------------------------- #
    print("\nD. failures land in save_error.json")
    with tempfile.TemporaryDirectory() as root:
        svc, _ = build(root)
        svc._built = False
        svc.save_adapter(os.path.join(root, "x"), "e1")
        r = read_save_result(os.path.join(root, "x"), "e1") or {}
        C.ok("state not built", r.get("saved") is False and "not built" in r.get("error", ""),
             str(r.get("error")))
        svc._built = True
        bad = os.path.join(root, "bad")
        os.makedirs(os.path.join(bad, WEIGHTS_FILE))       # a dir where the file goes
        svc.save_adapter(bad, "e2")
        r = read_save_result(bad, "e2") or {}
        C.ok("unwritable weights file", r.get("saved") is False and r.get("error"),
             str(r.get("error"))[:70])
        C.ok("failed write leaves no temp file",
             not [f for f in os.listdir(bad) if ".tmp-" in f], str(os.listdir(bad)))
        ipc = BackwardService(0)
        ipc._handle_save_adapter({"path": os.path.join(root, "ipc"), "save_id": "e3"})
        r = read_save_result(os.path.join(root, "ipc"), "e3") or {}
        C.ok("protocol-only service reports it", r.get("saved") is False, str(r.get("error")))




# --------------------------------------------------------------------------- #


def shard_base(base, rank, tp):
    ql, kl, il = (Hq // tp) * Hd, (Hkv // tp) * Hd, INTER // tp
    q, kv = Hq * Hd, Hkv * Hd
    out = {}
    for k, t in base.items():
        if k.endswith("qkv_proj.weight"):
            qq, kk, vv = t[:q], t[q:q + kv], t[q + kv:]
            out[k] = torch.cat([qq[rank * ql:(rank + 1) * ql],
                                kk[rank * kl:(rank + 1) * kl],
                                vv[rank * kl:(rank + 1) * kl]])
        elif k.endswith("o_proj.weight"):
            out[k] = t[:, rank * ql:(rank + 1) * ql]
        elif k.endswith("gate_up_proj.weight"):
            out[k] = torch.cat([t[:INTER][rank * il:(rank + 1) * il],
                                t[INTER:][rank * il:(rank + 1) * il]])
        elif k.endswith("down_proj.weight"):
            out[k] = t[:, rank * il:(rank + 1) * il]
        else:
            out[k] = t
    return out


def tp_worker(rank, tp, port, root, base, ft, meta, target, q):
    try:
        svc = Llama3BackwardService(0)
        m = dict(meta, tp_size=tp, tp_rank=rank, backward_nccl_port=port,
                 adapter_save_path=os.path.join(root, "tp"),
                 adapter_save_every_backwards=2)
        svc.shared = {"base": shard_base(base, rank, tp), "ft": ft, "meta": m}
        svc._build_state()
        svc._built = True
        # "Trained" masters: this rank's shard of a known full adapter.
        with torch.no_grad():
            for (i, proj, ab), key in svc._ft_keys.items():
                svc.lora[i][proj][ab].copy_(lora_shard_slice(
                    proj, ab, target[key], rank, tp, svc.q_size, svc.kv_size))
        svc._handle_save_adapter({"path": os.path.join(root, "m"), "save_id": "t1"})
        svc._after_cycle(0, True)
        svc._after_cycle(0, True)            # 2nd cycle → backward-000002
        broken = svc.lora[1].pop("k") if rank == 1 else None
        svc.save_adapter(os.path.join(root, "err"), "t2")
        if broken is not None:
            svc.lora[1]["k"] = broken
        svc.save_adapter(os.path.join(root, "after"), "t3")
        q.put((rank, "ok"))
    except Exception as e:  # noqa: BLE001
        import traceback
        q.put((rank, f"{e}\n{traceback.format_exc()}"))


if __name__ == "__main__":
    main_cpu()
    print("\nE. TP=2 over gloo")
    with tempfile.TemporaryDirectory() as root:
        svc1, _ = build(root)
        target = {k: torch.randn(v.shape, generator=torch.Generator().manual_seed(7))
                  for k, v in svc1.shared["ft"].items()}
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        procs = [ctx.Process(target=tp_worker, args=(
            rk, 2, port, root, svc1.shared["base"], svc1.shared["ft"],
            svc1.shared["meta"], target, q)) for rk in range(2)]
        for p in procs:
            p.start()
        res = dict(q.get(timeout=300) for _ in procs)
        for p in procs:
            p.join(60)
        C.ok("both ranks finished (no hang)", res == {0: "ok", 1: "ok"}, str(res)[:300])
        for name, sid in [("m", "t1"), ("tp/backward-000002", "backward-000002"),
                          ("after", "t3")]:
            d = os.path.join(root, name)
            r = read_save_result(d, sid) or {}
            C.ok(f"{name}: one merged adapter == full masters",
                 r.get("saved") is True and same(load_st(d), target)
                 and r.get("tp_size") == 2, str(r)[:60])
        r = read_save_result(os.path.join(root, "err"), "t2") or {}
        C.ok("rank-1 failure reported by rank 0",
             r.get("saved") is False and "rank 1" in r.get("error", ""),
             str(r.get("error"))[:70])
    C.finish()
