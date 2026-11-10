#!/usr/bin/env python
"""RoPE scaling in the backward matches the served forward (Llama-3.1/3.2/3.3).

Llama-3.1/3.2/3.3 share ``LlamaForCausalLM`` with Llama-3 but rescale the RoPE
frequencies (``rope_type == "llama3"``). The backward rematerializes q/k with
its own RoPE, so it must use exactly the frequencies vLLM serves with — before
this fix it read only ``rope_theta`` and trained those models with the wrong
positions, silently.

  A. frequencies: ``rope_inv_freq`` / ``rope_cos_sin`` / ``apply_rope`` vs
     vLLM's own rotary embedding (``get_rope``: the served forward) and
     transformers' reference init, for llama3 (3.1-8B and 3.2-1B parameters),
     linear and default RoPE; plain RoPE vs llama3 differs (negative control).
  B. config: ``rope_scaling_of`` on the real local configs (Llama-3.2-1B,
     Llama-3.1-8B, Meta-Llama-3-8B, Qwen3-0.6B) and synthetic ones; every
     unsupported variant (yarn, dynamic, longrope, partial rotary, mrope)
     raises instead of training silently wrong. ``build_backward_meta``
     carries it.
  C. trainer: ``process_backward`` on a synthetic Llama with llama3 scaling,
     activations captured with vLLM's served RoPE, gives the autograd
     gradients (after the same per-layer clip); the same service with the
     scaling dropped (the old behaviour) does not.

CPU only.   python tests/test_rope_scaling.py
"""

import glob
import os
import sys

import torch

import bwd_harness as H

sys.path[:] = [p for p in sys.path
               if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]

from deltaserve.bwd_services.common.ops import (  # noqa: E402
    apply_rope,
    rope_cos_sin,
    rope_inv_freq,
)
from deltaserve.bwd_services.llama3 import Llama3BackwardService  # noqa: E402
from deltaserve.ft_meta import build_backward_meta, rope_scaling_of  # noqa: E402

C = H.Checker(tol=1e-5)
HF = os.environ.get("HF_HOME", "/mnt/storage/huggingface")

LLAMA32 = {"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0,
           "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}
LLAMA31 = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
           "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}
LINEAR = {"rope_type": "linear", "factor": 4.0}


def vllm_rope(head_dim, theta, scaling, max_pos=131072):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import get_rope

    rp = {"rope_theta": theta, **(scaling or {"rope_type": "default"})}
    with set_current_vllm_config(VllmConfig()):
        return get_rope(head_dim, max_pos, True, rp, torch.float32)


def vllm_cos_sin(rope, positions):
    cache = rope.cos_sin_cache[positions]           # [n, Hd] = cos || sin
    return cache.chunk(2, dim=-1)


# --------------------------------------------------------------------------- #
print("A. frequencies vs vLLM's served rotary embedding / transformers")
POS = torch.tensor([0, 1, 7, 100, 1000, 4095, 8191, 8192, 20000, 65535, 131071])
for name, Hd, theta, sc in [("llama3 (3.2-1B)", 64, 5e5, LLAMA32),
                            ("llama3 (3.1-8B)", 128, 5e5, LLAMA31),
                            ("linear x4", 128, 1e4, LINEAR),
                            ("default (Qwen3)", 128, 1e6, None)]:
    rope = vllm_rope(Hd, theta, sc)
    C.check(f"{name}: inv_freq", rope_inv_freq(Hd, theta, sc),
            rope._compute_inv_freq(theta) if sc is None or sc["rope_type"] != "linear"
            else rope._compute_inv_freq(theta) / sc["factor"])
    cos, sin = rope_cos_sin(POS, Hd, theta, sc)
    vc, vs = vllm_cos_sin(rope, POS)
    C.check(f"{name}: cos", cos, vc, tol=2e-4)
    C.check(f"{name}: sin", sin, vs, tol=2e-4)
    # The rotation itself on q/k, as the served attention applies it.
    g = torch.Generator().manual_seed(0)
    q = torch.randn(len(POS), 4 * Hd, generator=g)
    k = torch.randn(len(POS), 2 * Hd, generator=g)
    vq, vk = rope.forward_native(POS, q.clone(), k.clone())
    C.check(f"{name}: rotated q", apply_rope(q.view(len(POS), 4, Hd), cos, sin),
            vq.view(len(POS), 4, Hd), tol=2e-4)
    C.check(f"{name}: rotated k", apply_rope(k.view(len(POS), 2, Hd), cos, sin),
            vk.view(len(POS), 2, Hd), tol=2e-4)

from transformers import LlamaConfig  # noqa: E402
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS  # noqa: E402

for name, sc in [("llama3 (3.2-1B)", LLAMA32), ("llama3 (3.1-8B)", LLAMA31)]:
    cfg = LlamaConfig(hidden_size=2048, num_attention_heads=32, head_dim=64,
                      max_position_embeddings=131072,
                      rope_parameters={"rope_theta": 5e5, **sc})
    ref, _ = ROPE_INIT_FUNCTIONS["llama3"](cfg, "cpu")
    C.check(f"{name}: vs transformers", rope_inv_freq(64, 5e5, sc), ref)

plain_c, _ = rope_cos_sin(POS, 64, 5e5)
v32c, _ = vllm_cos_sin(vllm_rope(64, 5e5, LLAMA32), POS)
d = (plain_c - v32c).abs().max().item()
C.ok("negative control: plain != llama3", d > 0.5, f"max|Δcos|={d:.3f}")
try:
    rope_inv_freq(64, 5e5, {"rope_type": "yarn", "factor": 4.0})
    C.ok("rope_inv_freq rejects yarn", False)
except NotImplementedError:
    C.ok("rope_inv_freq rejects yarn", True)


# --------------------------------------------------------------------------- #
print("\nB. rope_scaling_of on real + synthetic configs")


def local_config(repo):
    snaps = glob.glob(f"{HF}/hub/models--{repo.replace('/', '--')}/snapshots/*")
    if not snaps:
        return None
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(snaps[0])


for repo, want in [("meta-llama/Llama-3.2-1B", LLAMA32),
                   ("meta-llama/Llama-3.1-8B", LLAMA31),
                   ("meta-llama/Meta-Llama-3-8B", None),
                   ("Qwen/Qwen3-0.6B-Base", None)]:
    cfg = local_config(repo)
    if cfg is None:
        print(f"  [SKIP] {repo}: not in {HF}")
        continue
    got = rope_scaling_of(cfg)
    C.ok(f"{repo}", got == want, f"→ {got}")


class Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


C.ok("legacy rope_scaling with 'type'",
     rope_scaling_of(Cfg(rope_scaling={"type": "linear", "factor": 2})) ==
     {"rope_type": "linear", "factor": 2.0})
C.ok("no rope config at all → None", rope_scaling_of(Cfg()) is None)
C.ok("rope_parameters default → None",
     rope_scaling_of(Cfg(rope_parameters={"rope_type": "default",
                                          "rope_theta": 1e6})) is None)
for name, cfg in [
        ("yarn", Cfg(rope_parameters={"rope_type": "yarn", "factor": 4.0,
                                      "rope_theta": 1e6})),
        ("dynamic", Cfg(rope_scaling={"type": "dynamic", "factor": 2.0})),
        ("longrope (su alias)", Cfg(rope_scaling={"type": "su"})),
        ("partial rotary", Cfg(rope_parameters={"rope_type": "default"},
                               partial_rotary_factor=0.5)),
        ("mrope", Cfg(rope_parameters={"rope_type": "default",
                                       "mrope_section": [16, 24, 24]})),
        ("llama3 missing keys", Cfg(rope_parameters={"rope_type": "llama3",
                                                     "factor": 8.0}))]:
    try:
        rope_scaling_of(cfg)
        C.ok(f"refuses {name}", False, "no error raised")
    except ValueError as e:
        C.ok(f"refuses {name}", True, str(e)[:60])


class FT:   # the FinetuneConfig fields build_backward_meta reads
    learning_rate, weight_decay, gamma = 1e-4, 0.0, 1.0
    backward_fp32 = backward_cuda_graph = False
    backward_cuda_graph_attn_bn_max, backward_cuda_graph_attn_l_max = 8, 64
    max_saved_finetuning_tokens = 1024
    save_attn_qkv = save_attn_ctx = save_resid_mid = False


cfg32 = local_config("meta-llama/Llama-3.2-1B") or Cfg(
    num_attention_heads=32, hidden_size=2048, head_dim=64, vocab_size=128256,
    num_hidden_layers=16, num_key_value_heads=8, intermediate_size=8192,
    rope_parameters={"rope_theta": 5e5, **LLAMA32})
meta = build_backward_meta(cfg32, FT, lm_head_key="x", embed_weight_key="x",
                           has_final_norm=True, lora_scaling=2.0, tp_size=1,
                           tp_rank=0)
C.ok("build_backward_meta carries llama3", meta["rope_scaling"] == LLAMA32
     and meta["rope_theta"] == 5e5, f"{meta['rope_scaling']}")
try:
    build_backward_meta(Cfg(**{**cfg32.__dict__, "rope_parameters": {
        "rope_type": "yarn", "factor": 4.0, "rope_theta": 1e6}}), FT,
        lm_head_key="x", embed_weight_key="x", has_final_norm=True,
        lora_scaling=2.0, tp_size=1, tp_rank=0)
    C.ok("build_backward_meta refuses yarn", False)
except ValueError:
    C.ok("build_backward_meta refuses yarn", True)


# --------------------------------------------------------------------------- #
print("\nC. trainer gradients with llama3 RoPE (served RoPE = vLLM's)")
D, NL, Hq, Hkv, Hd, INTER, VOCAB, R = 64, 2, 4, 2, 32, 96, 48, 4
THETA, EPS, SCALING = 5e5, 1e-5, 2.0
SEQ = [300, 170]
# Small original context so the rescaled band covers most frequencies at
# these short positions (the real models rescale beyond 2048-token waves).
SC = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
      "high_freq_factor": 4.0, "original_max_position_embeddings": 256}


def reference(svc, served_rope):
    """Autograd loss + LoRA grads (per-layer clipped like the trainer), with
    the activations a vLLM forward with ``served_rope`` would capture."""
    fam = svc.family
    ids = torch.randint(0, VOCAB, (sum(SEQ),), generator=torch.Generator().manual_seed(1))
    b_start = [0, SEQ[0]]
    pos = torch.cat([torch.arange(s) for s in SEQ])
    cos, sin = vllm_cos_sin(served_rope, pos)
    x = svc.lm_w.new_tensor(0)  # placeholder for type
    embed = svc.shared["base"]["model.embed_tokens.weight"]
    x = embed[ids]
    layer_in, lws = [], []
    for i in range(svc.L):
        lw = dict(svc._layer_weights(i))
        for k in H.LORA_KEYS:
            lw[k] = lw[k].detach().clone().requires_grad_(True)
        lws.append(lw)
        layer_in.append(x.detach())
        x = H.ref_layer_output(fam, x, lw, svc.scaling, cos, sin, SEQ, b_start,
                               svc.dims, svc.eps)
    loss, _ = H.ref_head_loss(x, svc.lm_w, svc.norm_w, svc.eps, ids, SEQ,
                              b_start, svc.vocab)
    loss.backward()
    grads = []
    for lw in lws:
        ps = [lw[k] for k in H.LORA_KEYS]
        torch.nn.utils.clip_grad_norm_(ps, max_norm=1.0)
        grads.append({k: lw[k].grad for k in H.LORA_KEYS})
    acts = {"layer_in": layer_in, "final_in": x.detach(), "concat_input_ids": ids}
    return loss.item(), grads, acts


def trainer_grads(rope_scaling):
    svc, _ = H.build_synthetic_service(
        Llama3BackwardService, hidden=D, n_layers=NL, Hq=Hq, Hkv=Hkv, Hd=Hd,
        inter=INTER, vocab=VOCAB, r=R, lr=0.0, theta=THETA, eps=EPS,
        scaling=SCALING, rope_scaling=rope_scaling)
    # Non-zero B so every LoRA factor gets a gradient.
    with torch.no_grad():
        for ld in svc.lora.values():
            for pd in ld.values():
                pd["B"].normal_(0, 0.05)
    return svc


served = vllm_rope(Hd, THETA, SC, max_pos=1024)
for label, sc, expect_match in [("with llama3 scaling (fixed)", SC, True),
                                ("scaling dropped (old behaviour)", None, False)]:
    svc = trainer_grads(sc)
    ref_loss, ref_grads, acts = reference(svc, served)
    loss, _ = svc.process_backward(acts, SEQ, sum(SEQ), epoch=0)
    worst = 0.0
    for i in range(NL):
        for k in H.LORA_KEYS:
            got = svc.lora[i][k[0]][k[1]].grad
            ref = ref_grads[i][k]
            worst = max(worst, ((got - ref).abs().max() /
                                (ref.abs().max() + 1e-12)).item())
    dl = abs(float(loss) - ref_loss)
    if expect_match:
        C.ok(f"{label}: loss", dl < 1e-5, f"|Δloss|={dl:.2e}")
        C.ok(f"{label}: grads", worst < 1e-4, f"max-rel-err={worst:.2e}")
    else:
        C.ok(f"{label}: grads differ (control)", worst > 1e-2,
             f"max-rel-err={worst:.2e}")

C.finish()
