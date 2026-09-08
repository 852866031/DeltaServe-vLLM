# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Model-derived constants the backward child needs (the ``meta`` dict).

Kept outside the worker so every read of ``hf_config`` / the adapter config
that the finetuning path depends on lives in one deltaserve-owned place, with
no model-family branches. ``build_backward_meta`` is the single producer of the
dict that ``BackwardProcess.share_weights`` ships to the child and that the
trainer's ``_build_state`` consumes.
"""

from __future__ import annotations

import json
import os
from typing import Any

from vllm.deltaserve import dprint

_DEFAULT_ROPE_THETA = 10000.0


def rope_theta_of(hf_config: Any) -> float:
    """RoPE base frequency for the backward's RoPE rematerialization.

    transformers >= 5 (and vLLM's config normalization) move the value into
    ``hf_config.rope_parameters["rope_theta"]`` and drop the top-level
    ``rope_theta`` attribute. Reading only the attribute silently yields the
    10000 default for every modern model (Llama-3 uses 5e5, Qwen3 1e6), which
    desynchronizes the remat from the served forward — so check the nested
    dict first and warn loudly when neither carries a value."""
    rp = getattr(hf_config, "rope_parameters", None)
    if isinstance(rp, dict) and rp.get("rope_theta") is not None:
        return float(rp["rope_theta"])
    theta = getattr(hf_config, "rope_theta", None)
    if theta is not None:
        return float(theta)
    dprint("[deltaserve] WARNING: hf_config carries no rope_theta "
           f"(rope_parameters={rp!r}); backward RoPE falls back to "
           f"{_DEFAULT_ROPE_THETA}. The remat will NOT match the served model "
           "if it uses a different base frequency.")
    return _DEFAULT_ROPE_THETA


def read_lora_scaling(adapter_path: str | None) -> float:
    """``lora_alpha / r`` from the adapter's ``adapter_config.json`` (1.0 if
    the file is absent). The backward bakes this into B at publish time."""
    if not adapter_path:
        return 1.0
    cfg_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(cfg_path):
        return 1.0
    with open(cfg_path) as f:
        acfg = json.load(f)
    r = float(acfg.get("r", 1) or 1)
    alpha = float(acfg.get("lora_alpha", r))
    return alpha / r


def effective_save_attn_qkv(arch: str, ft_cfg: Any) -> bool:
    """``finetune.save_attn_qkv`` as the accumulator should apply it: off when
    the family's backward cannot consume saved q/k at all, so the worker does
    not allocate and fill buffers the child will ignore."""
    if not ft_cfg.save_attn_qkv:
        return False
    from vllm.deltaserve.bwd_services.registry import get_family

    fam = get_family(arch)
    if fam is not None and not fam.supports_saved_qkv:
        dprint(f"[deltaserve] save_attn_qkv is not exact for {fam.name}; "
               "the backward recomputes Q/K/V — not allocating attn_qkv buffers")
        return False
    return True


def saved_qkv_pre_transform(arch: str) -> bool:
    """Whether the accumulator must capture q/k at the INPUT of the family's
    per-head transform modules (``self_attn.q_norm`` / ``k_norm`` — Qwen3)
    instead of post-RoPE at ``self_attn.attn`` (Llama-3). See
    ``Family.saved_qkv_pre_transform``."""
    from vllm.deltaserve.bwd_services.registry import get_family

    fam = get_family(arch)
    return bool(fam is not None and fam.saved_qkv_pre_transform)


def build_backward_meta(hf_config: Any, ft_cfg: Any, *, lm_head_key: str | None,
                        embed_weight_key: str | None, has_final_norm: bool,
                        lora_scaling: float, tp_size: int, tp_rank: int) -> dict:
    """The ``meta`` dict shipped to the backward child with the shared weights.

    Model dims (for the per-layer forward rematerialization), the LM-head /
    norm / embedding weight names the worker resolved, optimizer + backward
    feature flags from ``FinetuneConfig``, and this rank's TP geometry. Every
    read is a generic ``hf_config`` attribute; nothing here is family-specific.
    ``head_dim`` falls back to ``hidden_size // num_attention_heads`` for
    configs that omit it (Llama-3); Qwen3 states it explicitly (and its
    ``head_dim * num_heads`` may differ from ``hidden_size``)."""
    num_heads = int(hf_config.num_attention_heads)
    head_dim = int(getattr(hf_config, "head_dim", None)
                   or hf_config.hidden_size // num_heads)
    return {
        "lm_head_key": lm_head_key,
        "vocab_size": int(hf_config.vocab_size),
        "logit_scale": float(getattr(hf_config, "logit_scale", 1.0) or 1.0),
        "rms_norm_eps": float(getattr(hf_config, "rms_norm_eps", 1e-5) or 1e-5),
        "norm_weight_key": "model.norm.weight" if has_final_norm else None,
        "embed_weight_key": embed_weight_key,
        # Model dims for the backward's per-layer forward rematerialization.
        "hidden_size": int(hf_config.hidden_size),
        "num_hidden_layers": int(hf_config.num_hidden_layers),
        "num_attention_heads": num_heads,
        "num_key_value_heads": int(
            getattr(hf_config, "num_key_value_heads", num_heads)),
        "head_dim": head_dim,
        "intermediate_size": int(hf_config.intermediate_size),
        "rope_theta": rope_theta_of(hf_config),
        "lora_scaling": float(lora_scaling),
        # Optimizer hyperparameters (Phase 3 LoRA backward).
        "learning_rate": float(ft_cfg.learning_rate),
        "weight_decay": float(ft_cfg.weight_decay),
        "gamma": float(ft_cfg.gamma),
        "backward_fp32": bool(ft_cfg.backward_fp32),
        # CUDA-graph backward (Phase 5): bn_max/l_max bound the padded-attention
        # region; s_max comes from max_saved_finetuning_tokens (the same value
        # sizes the activation buffers, so graphs and the pool are aligned).
        "backward_cuda_graph": bool(ft_cfg.backward_cuda_graph),
        "backward_cuda_graph_attn_bn_max":
            int(ft_cfg.backward_cuda_graph_attn_bn_max),
        "backward_cuda_graph_attn_l_max":
            int(ft_cfg.backward_cuda_graph_attn_l_max),
        "max_saved_finetuning_tokens":
            int(ft_cfg.max_saved_finetuning_tokens),
        # Forward-side activation saves the backward may short-circuit on:
        # post-RoPE q/k/v per layer (``attn_qh/kh/vh``) and the attention
        # context (``attn_ctx``). The child re-checks ``save_attn_qkv`` against
        # its family (see ``effective_save_attn_qkv``).
        "save_attn_qkv": bool(ft_cfg.save_attn_qkv),
        "save_attn_ctx": bool(ft_cfg.save_attn_ctx),
        "save_resid_mid": bool(ft_cfg.save_resid_mid),
        # [DeltaServe] Phase 7: TP shard geometry. The backward divides the
        # head counts / intermediate size by tp_size to slice its LOCAL shards,
        # and uses tp_rank to slice the (full, disk-loaded) FT-adapter master
        # into this rank's LoRA shard. tp_size=1 → local == full.
        "tp_size": int(tp_size),
        "tp_rank": int(tp_rank),
        # Rendezvous port for the backward-only NCCL group (the backward
        # children are NOT in vLLM's inference group). Override via env if
        # the default clashes on the box.
        "backward_nccl_port": int(
            os.environ.get("DSERVE_BACKWARD_NCCL_PORT", "29677")),
    }
