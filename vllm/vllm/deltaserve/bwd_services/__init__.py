# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-model DeltaServe backward (SFT) services.

The child backward process runs one of these services, selected by the model's
HF architecture string. The model-agnostic recv/dispatch loop lives in
``base.BackwardService``; per-model SFT math (logit reconstruction, loss, and —
in later slices — the LoRA backward) lives in the per-model modules
(``opt.py``, and later ``llama.py`` / ``llama3.py``). Mirrors DeltaServe's
``models/{llama,llama3}/SFT_service.py`` split.
"""

from vllm.deltaserve.bwd_services.base import BackwardService, get_service, service_main

__all__ = ["BackwardService", "get_service", "service_main"]
