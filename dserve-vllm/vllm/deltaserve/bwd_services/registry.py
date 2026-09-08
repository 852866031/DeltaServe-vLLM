# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Model architecture → backward service.

The worker names the child's service by the model's HF architecture string
(``hf_config.architectures[0]``); short aliases are accepted for tests and
tooling. Imports are lazy so a child only loads its own family's module.
"""

from __future__ import annotations

import importlib

_SERVICES: dict[str, str] = {
    # Llama-3 (trainer)
    "LlamaForCausalLM": "vllm.deltaserve.bwd_services.llama3:Llama3BackwardService",
    "llama3": "vllm.deltaserve.bwd_services.llama3:Llama3BackwardService",
    "llama": "vllm.deltaserve.bwd_services.llama3:Llama3BackwardService",
    # Qwen3 (trainer)
    "Qwen3ForCausalLM": "vllm.deltaserve.bwd_services.qwen3:Qwen3BackwardService",
    "qwen3": "vllm.deltaserve.bwd_services.qwen3:Qwen3BackwardService",
    # OPT (loss-only reference path)
    "OPTForCausalLM": "vllm.deltaserve.bwd_services.opt:OPTBackwardService",
    "opt": "vllm.deltaserve.bwd_services.opt:OPTBackwardService",
}


def supported_names() -> list[str]:
    return sorted(_SERVICES)


def get_service(name: str):
    """Return the ``BackwardService`` subclass for an architecture string.
    Raises ``NotImplementedError`` (listing what is supported) otherwise."""
    target = _SERVICES.get(name)
    if target is None:
        raise NotImplementedError(
            f"[deltaserve] no backward service for {name!r}; supported: "
            f"{', '.join(supported_names())}")
    module_name, _, class_name = target.partition(":")
    return getattr(importlib.import_module(module_name), class_name)


def is_trainer(name: str) -> bool:
    """True if the family actually trains (publishes LoRA weights back into
    the served slot); False for loss-only services and unknown names."""
    try:
        return bool(getattr(get_service(name), "is_trainer", False))
    except NotImplementedError:
        return False


def get_family(name: str):
    """The ``Family`` record of a trainer service, or None."""
    try:
        return getattr(get_service(name), "family", None)
    except NotImplementedError:
        return None
