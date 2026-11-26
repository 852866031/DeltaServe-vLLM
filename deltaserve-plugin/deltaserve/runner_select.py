# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Which vLLM model runner hosts the co-serving hooks
(``finetune_features.model_runner``): v1 (``gpu_model_runner.py``, in vLLM up
to v0.33) or v2 (``vllm/v1/worker/gpu``, the only one from v0.34 on)."""

import importlib.util
import os

CHOICES = ("auto", "v1", "v2")
# Overrides the config key when set (tests and A/B runs: one config, both runners).
ENV = "DSERVE_MODEL_RUNNER"


def v1_runner_available() -> bool:
    return importlib.util.find_spec("vllm.v1.worker.gpu_model_runner") is not None


def wants_v2(ft_cfg) -> bool:
    """The runner a co-serving launch asked for (True = v2). Raises on a
    choice this vLLM cannot serve."""
    choice = str(os.environ.get(ENV) or getattr(ft_cfg, "model_runner", "auto")
                 or "auto").lower()
    if choice not in CHOICES:
        raise ValueError(
            f"finetune_features.model_runner is one of {', '.join(CHOICES)}; "
            f"got {choice!r}")
    has_v1 = v1_runner_available()
    if choice == "v1" and not has_v1:
        raise ValueError(
            "finetune_features.model_runner: v1 — this vLLM has no v1 model "
            "runner (removed after v0.33); use v2 or auto")
    return choice == "v2" or not has_v1


def pin_vllm_runner(env=None) -> None:
    """Make FT-off launches (test baselines) use the runner the FT launches
    get: the two differ in bf16 numerics. On a vLLM with both runners that is
    ``VLLM_USE_V2_MODEL_RUNNER`` (v2 only when ``DSERVE_MODEL_RUNNER=v2``); a
    vLLM with only v2 rejects the variable, so it is removed."""
    env = os.environ if env is None else env
    if v1_runner_available():
        env["VLLM_USE_V2_MODEL_RUNNER"] = (
            "1" if str(env.get(ENV, "")).lower() == "v2" else "0")
    else:
        env.pop("VLLM_USE_V2_MODEL_RUNNER", None)
