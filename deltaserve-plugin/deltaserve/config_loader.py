# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""YAML config loader for DeltaServe co-serving — a thin layer over vLLM.

It reads a DeltaServe-style sectioned YAML and maps it onto vLLM's existing
``EngineArgs`` + our ``FinetuneConfig``. It does NOT modify any upstream vLLM
behaviour; it just produces the objects vLLM already consumes.

YAML shape (see configs/serving_config_finetuning_qwen3_14b_tp2.yaml)::

    finetune:     # training workload + co-serving behaviour switches
      enable_finetuning: true
      graph_ft_batches: true
    throttles:    # how much FT is admitted next to inference
      ft_tokens_admission_constrain_factor: -1
    slo:          # SLO targets + the execution-time estimator
      ttft_slo: 0.4
    debug:        # prints + trace / log paths
      print_scheduler_add: true

    server:       # → returned separately (host/port/rank for serving)
      port: 8000
    adapters:     # → returned separately (inference LoRA adapter paths)
      lora_path_0: ../adapters/qwen3-14b-toy-lora

    model:        # ┐ every other section is a flat dict of EngineArgs field
      model: ...  # │ names; they are merged together and passed straight to
    engine:       # │ EngineArgs.
      gpu_memory_utilization: 0.80

The four finetune sections all fold into ONE flat ``FinetuneConfig``; the
section a key belongs to is ``FINETUNE_SECTIONS``. A key found in another of the
four sections (the pre-2026-10 layout kept the throttles under ``finetune`` /
``slo``) still loads, with a one-line warning naming its section. Unknown
finetune keys are rejected by ``FinetuneConfig`` (extra=forbid); unknown
EngineArgs keys raise a clear ``TypeError`` from ``EngineArgs``.
"""

from pathlib import Path
from typing import Any

import yaml

from deltaserve.config import FinetuneConfig
from deltaserve import dprint
from vllm.engine.arg_utils import EngineArgs

# Section → the FinetuneConfig fields that belong to it.
FINETUNE_SECTIONS: dict[str, tuple[str, ...]] = {
    "finetune": (
        "enable_finetuning", "start_on_launch", "finetuning_lora_path",
        "data_path", "max_prepare", "num_epochs", "learning_rate",
        "weight_decay", "gamma", "max_saved_finetuning_tokens",
        "backward_mps_percentage", "backward_fp32", "backward_cuda_graph",
        "backward_cuda_graph_attn_bn_max", "backward_cuda_graph_attn_l_max",
        "backward_run_ahead_boundaries", "pause_on_arrival", "save_activations",
        "save_attn_qkv",
        "save_attn_ctx", "save_resid_mid", "graph_ft_batches",
        "graph_ft_max_tokens", "graph_ft_only_batches", "forward_interruptible",
        "ft_only_admission_grace_ms", "pause_prefill_always",
        "pause_until_prefill_done",
    ),
    "throttles": (
        "coserving_admission_phase", "ft_tokens_admission_constrain_factor",
        "match_prefill_workload_factor", "fwd_token_throttle_enable",
        "fwd_token_throttle", "rps_throttle_enable", "rps_throttle_close_rps",
        "rps_throttle_open_rps", "rps_throttle_window_s",
        "rps_throttle_close_time",
    ),
    "slo": (
        "ttft_slo", "avg_tbt_slo", "max_tbt_slo", "decode_only_ft_safety_margin",
        "profile_on_launch", "profile_num_repeats", "validate_estimator",
        "estimator_validation_path", "estimator_state_save_path",
        "estimator_state_load_path",
    ),
    "debug": (
        "print_weight_hash", "print_activation_hash", "print_step_mode",
        "print_scheduler_add", "print_engine_batch_exec",
        "print_engine_batch_done", "print_engine_req_recv", "step_trace_path",
        "bwd_log_path", "batch_prediction_stats_path",
    ),
}
SECTION_OF: dict[str, str] = {k: s for s, keys in FINETUNE_SECTIONS.items()
                              for k in keys}
# Keys old YAMLs may still carry that no longer do anything: dropped with a warning.
RETIRED_KEYS: dict[str, str] = {
    "backward_sleep_seconds": "only the removed OPT loss-only service slept",
}
# Sections that are NOT EngineArgs kwargs and are returned to the caller verbatim
# as `extras[section]`: `server` (host/port/rank for the API server) and
# `adapters` (inference LoRA adapter paths, applied per-request by the caller).
_PASSTHROUGH_SECTIONS = ("server", "adapters")
_SPECIAL_SECTIONS = (*FINETUNE_SECTIONS, *_PASSTHROUGH_SECTIONS)


def load_yaml_config(path: str | Path, verbose: bool = True) -> dict[str, Any]:
    """Read the YAML file into a dict of sections. Validates top-level shape.

    Relative path values (any key containing ``path``, e.g. ``finetuning_lora_path``,
    ``bwd_log_path`` or ``lora_path_0``) inside the finetune sections and
    ``adapters`` are resolved to absolute against the YAML file's directory, so
    they work regardless of CWD (the worker process may run elsewhere). Mirrors
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
    for section in (*FINETUNE_SECTIONS, "adapters"):
        body = raw.get(section) or {}
        for key, val in body.items():
            if "path" in key.lower() and isinstance(val, str) \
                    and not Path(val).is_absolute():
                body[key] = str((base_dir / val).resolve())

    if verbose:
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


def finetune_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """The four finetune sections merged into flat ``FinetuneConfig`` kwargs.

    A key outside its own section still counts (with a warning naming the
    section it belongs in); a retired key is dropped with a warning; a key that
    appears twice is an error."""
    out: dict[str, Any] = {}
    for section in FINETUNE_SECTIONS:
        for key, value in (config.get(section) or {}).items():
            if key in RETIRED_KEYS:
                dprint(f"[config] WARNING: '{section}.{key}' is ignored "
                       f"({RETIRED_KEYS[key]}); remove it")
                continue
            home = SECTION_OF.get(key)
            if home is not None and home != section:
                dprint(f"[config] WARNING: '{section}.{key}' belongs in "
                       f"section '{home}' (loaded anyway)")
            if key in out:
                raise ValueError(f"finetune key '{key}' appears in two sections")
            out[key] = value
    return out


def finetune_value(config: dict[str, Any], key: str, default: Any = None) -> Any:
    """One finetune setting from a loaded config, whichever section holds it."""
    for section in FINETUNE_SECTIONS:
        body = config.get(section) or {}
        if key in body:
            return body[key]
    return default


def additional_config(config: dict[str, Any], **overrides: Any) -> dict:
    """vLLM ``additional_config`` carrying the finetune sections (+ overrides,
    e.g. per-run log paths); None values are dropped."""
    kw = {**finetune_kwargs(config), **overrides}
    return {"deltaserve": {k: v for k, v in kw.items() if v is not None}}


def finetune_cli_args(config: dict[str, Any], **overrides: Any) -> list[str]:
    """The finetune sections as a ``vllm serve`` argument
    (``--additional-config '{"deltaserve": {...}}'``)."""
    import json

    return ["--additional-config",
            json.dumps(additional_config(config, **overrides))]


def split_config(config: dict[str, Any]) -> tuple[dict, FinetuneConfig, dict]:
    """Split a loaded config into (engine_kwargs, FinetuneConfig, extras).

    `extras` is a dict of the passthrough sections (e.g. {"server": {...},
    "adapters": {...}}). Every non-special section is merged into one
    engine_kwargs dict; a key appearing in two sections is a config error.
    """
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

    finetune_config = FinetuneConfig(**finetune_kwargs(config))
    return engine_kwargs, finetune_config, extras


def build_engine_args(config: dict[str, Any]) -> tuple[EngineArgs, dict]:
    """Build an ``EngineArgs`` (finetune sections in additional_config) + extras."""
    engine_kwargs, _, extras = split_config(config)
    engine_args = EngineArgs(**engine_kwargs,
                             additional_config=additional_config(config))
    return engine_args, extras


def engine_args_from_yaml(path: str | Path) -> tuple[EngineArgs, dict]:
    """Convenience: load a YAML path → (EngineArgs, extras)."""
    return build_engine_args(load_yaml_config(path))
