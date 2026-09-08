# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The record a model family hands to the family-agnostic trainer.

A family module (``bwd_services/llama3.py``, ``bwd_services/qwen3.py``) owns
the layer math — its own ``layer_forward`` / ``layer_backward`` composed from
the ``common`` cores — and describes, through a ``Family``, which frozen base
weights each layer needs and which functions drive the (optional) CUDA-graph
path. ``LoraSftTrainerService`` reads only this record; it never branches on
the family name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# The LoRA-trained projections, in the order the trainer iterates them.
LORA_PROJS = ("q", "k", "v", "o")

# Per-layer frozen base weights every SwiGLU decoder family shares, keyed by
# the ``lw`` name the layer math uses -> parameter name under
# ``model.layers.{i}.``. Fused tensors carry the key of the fused module; the
# trainer slices ``qkv`` into q/k/v and ``gate_up`` into gate/up by this
# rank's local widths.
BASE_LAYER_WEIGHTS: dict[str, str] = {
    "qkv": "self_attn.qkv_proj.weight",
    "o": "self_attn.o_proj.weight",
    "gate_up": "mlp.gate_up_proj.weight",
    "down": "mlp.down_proj.weight",
    "in_ln": "input_layernorm.weight",
    "post_ln": "post_attention_layernorm.weight",
}


@dataclass(frozen=True)
class Family:
    """What the trainer needs to know about one model family."""

    name: str
    """Short tag used in logs ("llama3", "qwen3")."""

    archs: tuple[str, ...]
    """HF architecture strings + aliases that select this family."""

    layer_forward: Callable
    """Rematerialize one decoder layer; returns the ``cache`` its backward reads."""

    layer_backward: Callable
    """Manual per-layer backward; returns ``(grad_x, lora_grads)``."""

    layer_weights: dict[str, str] = field(default_factory=lambda: dict(BASE_LAYER_WEIGHTS))
    """Frozen base weights per layer (see ``BASE_LAYER_WEIGHTS``); families
    with extra frozen tensors (Qwen3's q/k-norm) extend the base map."""

    graph_forward_core: Callable | None = None
    """``(runner, lw) -> None``: the captureable layer forward writing into the
    runner's static buffers, ending at the O-proj output ``static_o`` (the
    runner's ``forward_tail`` adds the residual and splits gate||up — after
    the TP all-reduce of ``o`` when there is one). ``None`` → the graph runner
    keeps the FFN-bwd and attention-bwd graphs but runs this family's forward
    eager."""

    layer_backward_graphed: Callable | None = None
    """The per-layer backward orchestrating Graph A / pause / Graph B / eager
    tail on the runner; takes ``all_reduce=`` and issues the same TP reduces as
    ``layer_backward`` from its eager code. ``None`` → the family runs fully
    eager even when ``backward_cuda_graph`` is requested."""

    supports_saved_qkv: bool = True
    """Whether the ``save_attn_qkv`` shortcut is exact for this family (the
    trainer forces the recompute path when False)."""

    saved_qkv_pre_transform: bool = False
    """Where the accumulator captures q/k for ``save_attn_qkv``. False: the
    ``self_attn.attn`` pre-hook, i.e. post-RoPE q/k ready for the attention
    core (Llama-3). True: the inputs of the family's per-head transform
    modules (``self_attn.q_norm`` / ``k_norm`` pre-hooks on Qwen3), i.e. the
    raw projection outputs — the backward needs those for the transform's
    backward, and the remat re-applies the elementwise transform + RoPE. v is
    always the ``self_attn.attn`` pre-hook's third arg."""
