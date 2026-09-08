#!/usr/bin/env python
"""End-to-end training check for the Qwen3 backward service.

Builds a tiny synthetic ``Qwen3BackwardService`` (q/k-norm layers, TIED LM
head so the ``meta["lm_head_key"]`` path is exercised as on Qwen3-0.6B),
captures activations with the family's own forward, and runs the real
``process_backward`` (manual grads → clip → fused AdamW → master update) 300
times on a fixed batch. The loss must collapse.

    python tests/test_qwen3_train_overfit.py
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

from vllm.deltaserve.bwd_services.qwen3 import Qwen3BackwardService  # noqa: E402


def main():
    torch.manual_seed(1)
    seq_lens = [5, 4, 6]
    n = sum(seq_lens)
    b_start, _, _ = H.seq_layout(seq_lens, 16, 1e6)
    vocab = 32
    ids = torch.randint(0, vocab, (n,))

    svc, embed_w = H.build_synthetic_service(
        Qwen3BackwardService, hidden=48, n_layers=2, Hq=4, Hkv=2, Hd=16,
        inter=128, vocab=vocab, r=8, lr=5e-3, theta=1e6, eps=1e-6,
        tied_lm_head=True)
    assert svc.lm_w is embed_w, "tied head must resolve to the embedding"
    assert "q_norm" in svc.base[0] and svc.base[0]["q_norm"].shape == (16,)

    steps = 300
    losses = []
    for step in range(steps):
        acts = H.forward_capture(svc, embed_w, ids, seq_lens, b_start)
        loss, _ = svc.process_backward(acts, seq_lens, n, epoch=0)
        losses.append(loss)
        if step % 30 == 0 or step == steps - 1:
            print(f"  step {step:3d}  loss={loss:.4f}")

    initial = sum(losses[:3]) / 3
    final = sum(losses[-3:]) / 3
    print(f"\n  initial≈{initial:.4f}  final≈{final:.4f}  "
          f"drop={100 * (1 - final / initial):.1f}%")
    ok = final < 0.5 * initial and final < 1.0
    print(f"  [{'PASS' if ok else 'FAIL'}] training reduces loss (overfit) — "
          f"expect final < 0.5*initial and < 1.0")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
