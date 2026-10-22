# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-model DeltaServe backward (SFT) services.

The child backward process runs one of these services, selected by the model's
HF architecture string (``registry.py``). The model-agnostic recv/dispatch loop
lives in ``base.BackwardService``; the shared LoRA-SFT trainer stack (layer
cores, TP reduces, CUDA graphs, the trainer service) in ``common/``; and each
model family's own layer math in its module (``llama3.py``, ``qwen3.py``).
Mirrors DeltaServe's
``models/{llama,llama3}/SFT_service.py`` split.
"""
from deltaserve.bwd_services.base import BackwardService, service_main
from deltaserve.bwd_services.registry import (
    get_family,
    get_service,
    supported_names,
)

__all__ = ["BackwardService", "get_family", "get_service", "service_main",
           "supported_names"]
