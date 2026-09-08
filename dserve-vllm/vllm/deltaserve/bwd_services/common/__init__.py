# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Family-agnostic building blocks of the LoRA-SFT backward services.

Everything here is shared by every trained model family (Llama-3, Qwen3, …):

- ``ops``        — RoPE, RMSNorm, LoRA-augmented projections (+ their backwards)
- ``attention``  — per-sample GQA attention forward / backward cores
- ``ffn``        — SwiGLU FFN forward tail / backward core
- ``head``       — final-norm + LM-head loss and gradient
- ``tp``         — tensor-parallel shard slicing, backward NCCL group, reduces
- ``family``     — the ``Family`` record a per-family module fills in
- ``trainer``    — ``LoraSftTrainerService``: build state, run the backward, publish
- ``graph``      — ``GraphedBackward``: CUDA-graph capture/replay of the backward

A family module (``bwd_services/llama3.py``, ``bwd_services/qwen3.py``) composes
these into its own ``layer_forward`` / ``layer_backward`` and declares a
``Family`` + a one-line service subclass. Nothing in this package branches on
the model family.
"""
