#!/usr/bin/env python
"""The backward attention graph's budget chosen from the dataset
(``backward_cuda_graph_attn_bn_max`` / ``_l_max`` = ``auto``), the launch note
for a configured one, the eager-cycle accounting, and the switch that starts
the backward child without torch's ``_native`` overrides. CPU.

    python tests/test_bwd_budget.py
"""

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bwd_harness as H  # noqa: E402

sys.path[:] = [p for p in sys.path
               if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]

from deltaserve import bwd_budget as B  # noqa: E402
from deltaserve.config import FinetuneConfig  # noqa: E402
from deltaserve.config_loader import FINETUNE_SECTIONS  # noqa: E402

C = H.Checker(tol=1e-9)
NS = types.SimpleNamespace

print("A. the epoch's backward batches")
# admission takes the smallest sample first; a batch is whole samples
bt = B.simulate_batches([5, 5, 5, 5, 20, 20, 30, 64, 100], 40)
C.ok("smallest first, whole samples, up to the batch size; too-long dropped",
     bt == [(5, 20), (1, 20), (1, 30)], str(bt))
C.ok("nothing to pack", B.simulate_batches([], 40) == []
     and B.simulate_batches([100], 40) == [])

print("B. the choice")
# 800 samples of 18-22 tokens and 200 of 40-60: early batches hold ~13 short
# samples, late ones ~4 long ones.
lengths = [18 + i % 5 for i in range(800)] + [40 + i % 21 for i in range(200)]
info = B.choose_budget(lengths, 256, local_heads=20)
bt = B.simulate_batches(lengths, 256)
C.ok("the chosen shape holds at least 99 % of the batches",
     info["coverage"] >= 0.99
     and B.coverage_of(lengths, 256, info["bn_max"], info["l_max"]) >= 0.99,
     f"{info['bn_max']} x {info['l_max']}: {info['coverage']:.3f}")
C.ok("... with room for the sample count (margin) and no more length than the data has",
     info["bn_max"] > max(b for b, _ in bt) and info["l_max"] <= 60,
     f"batches hold up to {max(b for b, _ in bt)} samples")
C.ok("the old default held almost none of them",
     B.coverage_of(lengths, 256, 8, 64) < 0.5,
     f"8 x 64: {B.coverage_of(lengths, 256, 8, 64):.2f}")
big = B.choose_budget(lengths, 512, local_heads=20)
C.ok("a larger backward batch gets a larger sample budget",
     big["bn_max"] > info["bn_max"], f"{info['bn_max']} -> {big['bn_max']}")
tight = B.choose_budget(lengths, 256, local_heads=20, max_scores_mb=0.5)
C.ok("the scores bound caps the shape (best coverage within it)",
     tight["scores_mb"] <= 0.5 and tight["coverage"] < info["coverage"],
     f"{tight['bn_max']} x {tight['l_max']}: {tight['scores_mb']:.2f} MB, "
     f"{tight['coverage']:.2f}")
uniform = B.choose_budget([20] * 500, 256)
C.ok("uniform samples: exactly what a batch holds, plus the margin",
     uniform["l_max"] == 20 and uniform["bn_max"] == 14, str(uniform["bn_max"]))

print("C. from the config")
cfg = NS(max_saved_finetuning_tokens=256, backward_cuda_graph_attn_bn_max="auto",
         backward_cuda_graph_attn_l_max="auto")
bn, l, note = B.resolve(cfg, lengths=lengths, local_heads=20)
C.ok("auto on both keys: the dataset's choice",
     (bn, l) == (info["bn_max"], info["l_max"]) and "auto" in B.describe(bn, l, note))
cfg = NS(max_saved_finetuning_tokens=256, backward_cuda_graph_attn_bn_max=8,
         backward_cuda_graph_attn_l_max=64)
bn, l, note = B.resolve(cfg, lengths=lengths, local_heads=20)
text = B.describe(bn, l, note)
C.ok("numbers are taken as they are",
     (bn, l) == (8, 64))
C.ok("... and the launch note says they do not hold the data, with a suggestion",
     "EAGER" in text and f"{info['bn_max']} x {info['l_max']}" in text, text)
cfg = NS(max_saved_finetuning_tokens=256, backward_cuda_graph_attn_bn_max=40,
         backward_cuda_graph_attn_l_max=64)
bn, l, note = B.resolve(cfg, lengths=lengths, local_heads=20)
C.ok("a configured budget that holds everything: no warning",
     "EAGER" not in B.describe(bn, l, note) and "100 %" in B.describe(bn, l, note))
cfg = NS(max_saved_finetuning_tokens=256, backward_cuda_graph_attn_bn_max="auto",
         backward_cuda_graph_attn_l_max=48)
bn, l, note = B.resolve(cfg, lengths=lengths, local_heads=20)
C.ok("auto on one key only keeps the other", l == 48 and bn == info["bn_max"])
cfg = NS(max_saved_finetuning_tokens=256, backward_cuda_graph_attn_bn_max="auto",
         backward_cuda_graph_attn_l_max="auto", data_path=None)
bn, l, note = B.resolve(cfg, tokenizer_name=None)
C.ok("auto without a dataset: the built-in values, said so",
     (bn, l) == (B.DEFAULT_BN_MAX, B.DEFAULT_L_MAX)
     and "built-in" in B.describe(bn, l, note))
cfg = NS(max_saved_finetuning_tokens=1024, bwd_tokens=256,
         backward_cuda_graph_attn_bn_max="auto", backward_cuda_graph_attn_l_max="auto")
bn2, l2, _ = B.resolve(cfg, lengths=lengths, local_heads=20)
C.ok("the batch is bwd_batch_size where the config has one",
     (bn2, l2) == (info["bn_max"], info["l_max"]))

d = FinetuneConfig()
C.ok("auto is the default; the child starts without torch/_native by default",
     d.backward_cuda_graph_attn_bn_max == "auto"
     and d.backward_cuda_graph_attn_l_max == "auto"
     and d.backward_disable_torch_native is True
     and "backward_disable_torch_native" in FINETUNE_SECTIONS["finetune_features"])
n = FinetuneConfig(backward_cuda_graph_attn_bn_max=16, backward_cuda_graph_attn_l_max=48)
C.ok("numbers still accepted", n.backward_cuda_graph_attn_bn_max == 16)

print("D. the child's meta")
from deltaserve.ft_meta import build_backward_meta  # noqa: E402

hf = NS(hidden_size=64, num_hidden_layers=2, num_attention_heads=8,
        num_key_value_heads=4, intermediate_size=128, vocab_size=100,
        rms_norm_eps=1e-6, rope_theta=1e6, head_dim=8, architectures=["Qwen3ForCausalLM"])
kw = dict(lm_head_key="lm_head.weight", embed_weight_key="e", has_final_norm=True,
          lora_scaling=2.0, tp_size=2, tp_rank=0)
on = FinetuneConfig(enable_finetuning=True, backward_cuda_graph=True,
                    backward_cuda_graph_attn_bn_max=12,
                    backward_cuda_graph_attn_l_max=40)
meta = build_backward_meta(hf, on, **kw)
C.ok("configured numbers reach the child as ints",
     (meta["backward_cuda_graph_attn_bn_max"], meta["backward_cuda_graph_attn_l_max"])
     == (12, 40) and "12 samples x 40" in meta["attn_budget_note"])
auto = FinetuneConfig(enable_finetuning=True, backward_cuda_graph=True)
meta = build_backward_meta(hf, auto, **kw)      # no data_path: built-in values
C.ok("auto with no corpus: ints, never the string",
     isinstance(meta["backward_cuda_graph_attn_bn_max"], int)
     and isinstance(meta["backward_cuda_graph_attn_l_max"], int))
off = FinetuneConfig(enable_finetuning=True, backward_cuda_graph=False)
meta = build_backward_meta(hf, off, **kw)
C.ok("backward graphs off: the corpus is not looked at",
     meta["attn_budget_note"] is None
     and isinstance(meta["backward_cuda_graph_attn_bn_max"], int))

print("E. the child's environment")
import deltaserve.backward_process as BP  # noqa: E402

seen = {}


class FakeProc:
    def __init__(self, *a, **k):
        pass

    def start(self):
        seen["env"] = os.environ.get("TORCH_DISABLE_NATIVE_JIT")

    def is_alive(self):
        return True


def spawn(flag):
    bp = BP.BackwardProcess(mps_percentage=0, device_index=0,
                            service_name="Qwen3ForCausalLM",
                            disable_torch_native=flag)
    bp._ctx = NS(Process=FakeProc, Pipe=bp._ctx.Pipe, Event=bp._ctx.Event)
    try:
        bp.start()
    except Exception:  # noqa: BLE001 — the handshake after the spawn is not under test
        pass
    return seen.pop("env", "not started")


os.environ.pop("TORCH_DISABLE_NATIVE_JIT", None)
C.ok("on: the child is spawned with TORCH_DISABLE_NATIVE_JIT=1", spawn(True) == "1")
C.ok("... and the serving process's environment is left as it was",
     "TORCH_DISABLE_NATIVE_JIT" not in os.environ)
C.ok("off: torch's default for the child", spawn(False) is None)

print("F. eager cycles are counted")
from deltaserve.coordinator import FinetuneCoordinator  # noqa: E402

c = FinetuneCoordinator(64)
for fit in (True, False, False, None):
    c.note_backward_done({"event": "activations_processed", "n": 10,
                          "attn_graph": fit})
st = c.status()
C.ok("status: 2 of 4 backwards ran their attention eager",
     st["backwards_done"] == 4 and st["backwards_attn_eager"] == 2, str(st))

C.finish()
