# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Trained-adapter checkpoints in PEFT format.

The backward child owns the fp32 LoRA masters; ``trainer.save_adapter`` turns
them into a checkpoint with the helpers here:

- ``merge_tp_shards``: under TP each rank holds a shard of the column-parallel
  q/k/v B (output rows) and of the row-parallel o A (input columns); the other
  factors are replicated. Inverse of ``tp.lora_shard_slice``.
- ``write_peft_adapter``: ``adapter_model.safetensors`` (fp32 masters, keys of
  the source adapter, B unscaled — PEFT applies ``lora_alpha / r`` itself),
  the source ``adapter_config.json``, and ``trainer_state.json`` last. Every
  file is written to a temp name and renamed, so a reader never sees a torn
  file; ``trainer_state.json`` (with the request's ``save_id``) is the
  completion marker POST /save_adapter waits for.
- ``write_save_error`` / ``read_save_result``: the other half of that
  handshake (the child replies through the file system — its pipe carries only
  backward acks).

A checkpoint loads anywhere a PEFT LoRA does: vLLM ``--lora-modules``, PEFT
``PeftModel.from_pretrained``, or as ``finetune_param.finetuning_lora_path`` to keep
training from it (the optimizer state starts fresh).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time

import torch

STATE_FILE = "trainer_state.json"
ERROR_FILE = "save_error.json"
WEIGHTS_FILE = "adapter_model.safetensors"
CONFIG_FILE = "adapter_config.json"


def merge_tp_shards(shards: list[dict]) -> dict:
    """Merge per-rank ``{(layer, proj, ab): tensor}`` dicts (rank order) into
    full factors: q/k/v B concatenated on dim 0, o A on dim 1, every other
    factor replicated (rank 0's)."""
    if len(shards) == 1:
        return dict(shards[0])
    out = {}
    for key, t0 in shards[0].items():
        _, proj, ab = key
        if proj in ("q", "k", "v") and ab == "B":
            out[key] = torch.cat([s[key] for s in shards], dim=0)
        elif proj == "o" and ab == "A":
            out[key] = torch.cat([s[key] for s in shards], dim=1)
        else:
            out[key] = t0
    return out


def _atomic_write_bytes(path: str, write) -> None:
    tmp = f"{path}.tmp-{os.getpid()}"
    try:
        write(tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _json_dump(path: str, obj) -> None:
    def w(tmp):
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
    _atomic_write_bytes(path, w)


def _adapter_config(source_dir: str | None, tensors: dict, scaling: float) -> dict:
    """The source adapter's config (rank / alpha / targets / base model) — the
    trained factors keep its shapes — or a minimal LoRA config."""
    if source_dir:
        src = os.path.join(source_dir, CONFIG_FILE)
        if os.path.isfile(src):
            with open(src) as f:
                return json.load(f)
    r = next((t.shape[0] for k, t in tensors.items() if "lora_A" in k), 1)
    targets = sorted({m.group(1) for k in tensors
                      if (m := re.search(r"\.(\w+_proj)\.lora_", k))})
    return {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": int(r),
            "lora_alpha": float(scaling) * int(r), "target_modules": targets,
            "lora_dropout": 0.0, "bias": "none"}


def write_peft_adapter(out_dir: str, tensors: dict, *, source_dir: str | None,
                       scaling: float, state: dict) -> str:
    """Write a PEFT LoRA checkpoint to ``out_dir`` (created; files replaced
    atomically, ``trainer_state.json`` last). ``tensors`` maps PEFT key →
    tensor. Returns ``out_dir``."""
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    cpu = {k: v.detach().to("cpu", torch.float32).contiguous()
           for k, v in tensors.items()}
    _atomic_write_bytes(os.path.join(out_dir, WEIGHTS_FILE),
                        lambda tmp: save_file(cpu, tmp, metadata={"format": "pt"}))
    _json_dump(os.path.join(out_dir, CONFIG_FILE),
               _adapter_config(source_dir, cpu, scaling))
    err = os.path.join(out_dir, ERROR_FILE)
    if os.path.exists(err):
        os.remove(err)
    _json_dump(os.path.join(out_dir, STATE_FILE),
               {**state, "saved_unix_time": time.time(),
                "num_tensors": len(cpu)})
    return out_dir


def write_save_error(out_dir: str, save_id: str, message: str) -> None:
    """Record a failed save where POST /save_adapter looks for it."""
    try:
        os.makedirs(out_dir, exist_ok=True)
        _json_dump(os.path.join(out_dir, ERROR_FILE),
                   {"save_id": save_id, "error": message,
                    "unix_time": time.time()})
    except OSError:
        pass


def read_save_result(out_dir: str, save_id: str) -> dict | None:
    """The outcome of save ``save_id`` if it is on disk yet:
    ``{"saved": True, **trainer_state}`` / ``{"saved": False, "error": …}``,
    else None."""
    for name, ok in ((STATE_FILE, True), (ERROR_FILE, False)):
        p = os.path.join(out_dir, name)
        try:
            with open(p) as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        if d.get("save_id") == save_id:
            return {"saved": ok, **d}
    return None


def prune_checkpoints(root: str, prefix: str, keep_last: int) -> list[str]:
    """Delete all but the newest ``keep_last`` ``<root>/<prefix><N>`` dirs
    (by N). Returns the removed paths. ``keep_last <= 0`` keeps everything."""
    if keep_last <= 0 or not os.path.isdir(root):
        return []
    found = []
    for name in os.listdir(root):
        m = re.fullmatch(re.escape(prefix) + r"(\d+)", name)
        if m and os.path.isdir(os.path.join(root, name)):
            found.append((int(m.group(1)), os.path.join(root, name)))
    found.sort()
    removed = [p for _, p in found[:-keep_last]]
    for p in removed:
        shutil.rmtree(p, ignore_errors=True)
    return removed
