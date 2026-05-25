# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""YAML config loader for DeltaServe co-serving — a thin layer over vLLM.

This is purely additive: it reads a DeltaServe-style sectioned YAML and maps it
onto vLLM's existing ``EngineArgs`` + our ``FinetuneConfig``. It does NOT modify
any upstream vLLM behaviour; it just produces the objects vLLM already consumes.

YAML shape (see configs/serving_config_finetuning.yaml)::

    finetune:                 # → FinetuneConfig (our sub-config)
      enable_finetuning: true
      backward_mps_percentage: 10

    server:                   # → returned separately (host/port/rank for serving)
      host: 127.0.0.1
      port: 8000
      rank_id: 0

    model:                    # ┐
      model: meta-llama/...   # │ every other section is a flat dict of
      dtype: auto             # │ EngineArgs field names; they are merged
    engine:                   # │ together and passed straight to EngineArgs.
      gpu_memory_utilization: 0.85
      tensor_parallel_size: 1

Special sections: ``finetune`` and ``server``. Any other top-level section is
treated as a bag of ``EngineArgs`` kwargs (so unknown vLLM knobs need no loader
change — just add them under any non-special section). Unknown finetune keys are
rejected by ``FinetuneConfig`` (pydantic extra=forbid); unknown EngineArgs keys
raise a clear ``TypeError`` from ``EngineArgs``.
"""

from pathlib import Path
from typing import Any

import yaml

from vllm.config import FinetuneConfig
from vllm.deltaserve import dprint
from vllm.engine.arg_utils import EngineArgs

_FINETUNE_SECTION = "finetune"
# `debug` and `slo` are folded INTO the FinetuneConfig kwargs (their keys are
# FinetuneConfig fields, e.g. print_weight_hash, ttft_slo); they just group the
# observability / SLO knobs to mirror DeltaServe's sectioned YAML.
_DEBUG_SECTION = "debug"
_SLO_SECTION = "slo"
# Sections that are NOT EngineArgs kwargs and are returned to the caller verbatim
# as `extras[section]`: `server` (host/port/rank for the API server) and
# `adapters` (inference LoRA adapter paths, applied per-request by the caller).
_PASSTHROUGH_SECTIONS = ("server", "adapters")
_SPECIAL_SECTIONS = (_FINETUNE_SECTION, _DEBUG_SECTION, _SLO_SECTION,
                     *_PASSTHROUGH_SECTIONS)


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Read the YAML file into a dict of sections. Validates top-level shape.

    Relative path values (any key containing ``path``, e.g. ``finetuning_lora_path``
    or ``lora_path_0``) inside the ``finetune`` and ``adapters`` sections are
    resolved to absolute against the YAML file's directory, so they work
    regardless of CWD (the worker process may run elsewhere). Mirrors
    DeltaServe's launcher behaviour.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"DeltaServe config not found: {path}")
    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path}: top level must be a mapping of sections, got {type(raw).__name__}"
        )
    for section, body in raw.items():
        if body is not None and not isinstance(body, dict):
            raise ValueError(
                f"{path}: section '{section}' must be a mapping (got "
                f"{type(body).__name__})"
            )

    base_dir = path.resolve().parent
    for section in (_FINETUNE_SECTION, "adapters"):
        body = raw.get(section) or {}
        for key, val in body.items():
            if "path" in key.lower() and isinstance(val, str) \
                    and not Path(val).is_absolute():
                body[key] = str((base_dir / val).resolve())

    print_loaded_config(raw, source=str(path))
    return raw


def print_loaded_config(config: dict[str, Any], source: str | None = None) -> None:
    """Print every loaded section and its key/values (one line per section)."""
    header = "loaded config" + (f" ({source})" if source else "")
    dprint(header + ":")
    for section, body in config.items():
        body = body or {}
        kvs = "  ".join(f"{k}={v}" for k, v in body.items()) or "(empty)"
        dprint(f"  [{section}] {kvs}")


def split_config(config: dict[str, Any]) -> tuple[dict, FinetuneConfig, dict]:
    """Split a loaded config into (engine_kwargs, FinetuneConfig, extras).

    `extras` is a dict of the passthrough sections (e.g. {"server": {...},
    "adapters": {...}}). Every non-special section is merged into one
    engine_kwargs dict; a key appearing in two sections is a config error.
    """
    # `debug` and `slo` keys are folded into the FinetuneConfig kwargs.
    finetune_body = {**(config.get(_FINETUNE_SECTION) or {}),
                     **(config.get(_DEBUG_SECTION) or {}),
                     **(config.get(_SLO_SECTION) or {})}
    extras = {name: dict(config.get(name) or {}) for name in _PASSTHROUGH_SECTIONS}

    engine_kwargs: dict[str, Any] = {}
    for section, body in config.items():
        if section in _SPECIAL_SECTIONS:
            continue
        for key, value in (body or {}).items():
            if key in engine_kwargs:
                raise ValueError(
                    f"duplicate EngineArgs key '{key}' (appears in section "
                    f"'{section}' and another section)"
                )
            engine_kwargs[key] = value

    finetune_config = FinetuneConfig(**finetune_body)
    return engine_kwargs, finetune_config, extras


def build_engine_args(config: dict[str, Any]) -> tuple[EngineArgs, dict]:
    """Build an ``EngineArgs`` (with finetune_config attached) + extras dict."""
    engine_kwargs, finetune_config, extras = split_config(config)
    engine_args = EngineArgs(**engine_kwargs, finetune_config=finetune_config)
    return engine_args, extras


def engine_args_from_yaml(path: str | Path) -> tuple[EngineArgs, dict]:
    """Convenience: load a YAML path → (EngineArgs, extras)."""
    return build_engine_args(load_yaml_config(path))
