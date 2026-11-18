# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Save a finetuning job's trained LoRA adapter as a normal PEFT adapter.

The trained weights live in the backward child as fp32 masters. Under tensor
parallelism every rank's child holds only its SHARD of them (the partition of
``bwd_services.common.tp.lora_shard_slice``):

    q/k/v  B [out, r]  split on output rows      A [r, hidden] the same on all ranks
    o      A [r, in]   split on input columns    B [hidden, r] the same on all ranks

``merge_rank_shards`` puts the ranks back together (the inverse of the slice)
and ``write_adapter`` writes a directory in the layout of the adapter the job
started from: the same tensor names and shapes, the same ``adapter_config.json``
and side files — so vLLM (or PEFT) loads it like any other adapter. The masters
are the plain PEFT factors (alpha/r is applied only when publishing into the
served slot), so nothing has to be un-scaled.

``export_trained_adapter`` is the engine-side call that collects the shards from
every rank's worker and writes the adapter.

torch / safetensors only; no vLLM import.
"""

import os
import re
import shutil

import torch

WEIGHTS_FILE = "adapter_model.safetensors"
LORA_KEY_RE = re.compile(
    r"layers\.(\d+)\.self_attn\.([qkvo])_proj\.lora_([AB])\.weight")
# (proj, factor) → the dim the ranks' shards are concatenated on.
_SHARD_DIM = {("q", "B"): 0, ("k", "B"): 0, ("v", "B"): 0, ("o", "A"): 1}


def merge_rank_shards(per_rank: list[dict]) -> dict:
    """``per_rank[r]`` is rank r's ``{PEFT key: tensor}``. Returns the full
    adapter state. A factor that is replicated across ranks must agree on all
    of them (its gradient is all-reduced); a disagreement is an error."""
    if not per_rank:
        raise ValueError("no rank state to merge")
    keys = set(per_rank[0])
    for r, state in enumerate(per_rank):
        if set(state) != keys:
            raise ValueError(f"rank {r} exported a different set of tensors")
    if len(per_rank) == 1:
        return {k: v.detach().float().cpu() for k, v in per_rank[0].items()}
    out = {}
    for key in sorted(keys):
        m = LORA_KEY_RE.search(key)
        if m is None:
            raise ValueError(f"not a LoRA attention factor: {key}")
        parts = [state[key].detach().float().cpu() for state in per_rank]
        dim = _SHARD_DIM.get((m.group(2), m.group(3)))
        if dim is not None:
            out[key] = torch.cat(parts, dim=dim).contiguous()
            continue
        for r, p in enumerate(parts[1:], 1):
            if p.shape != parts[0].shape or not torch.allclose(
                    p, parts[0], rtol=1e-4, atol=1e-6):
                raise ValueError(
                    f"replicated factor {key} differs between rank 0 and rank {r}")
        out[key] = parts[0].contiguous()
    return out


def write_adapter(src_dir: str, dst_dir: str, state: dict,
                  overwrite: bool = False) -> str:
    """Write ``state`` as an adapter directory shaped like ``src_dir`` (the
    adapter the job was initialised from). Every trained tensor must have the
    name and shape of one in the source; source tensors that were not trained
    are carried over. Written to a temp directory and renamed, so a reader
    never sees a half-written adapter."""
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    src_weights = os.path.join(src_dir, WEIGHTS_FILE)
    if not os.path.isfile(src_weights):
        raise FileNotFoundError(f"source adapter has no {WEIGHTS_FILE}: {src_dir}")
    with safe_open(src_weights, "pt") as f:
        shapes = {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}
        metadata = f.metadata() or {}
    unknown = sorted(set(state) - set(shapes))
    if unknown:
        raise ValueError(f"trained tensors not in the source adapter: {unknown[:3]}")
    for k, t in state.items():
        if tuple(t.shape) != shapes[k]:
            raise ValueError(f"{k}: trained shape {tuple(t.shape)} != source "
                             f"{shapes[k]} (TP shards not merged?)")
    if not state:
        raise ValueError("no trained tensor to save")
    out = {k: v.detach().float().cpu().contiguous() for k, v in state.items()}
    missing = set(shapes) - set(out)
    if missing:
        src = load_file(src_weights)
        out.update({k: src[k] for k in missing})

    dst_dir = os.path.abspath(dst_dir)
    if os.path.exists(dst_dir) and not overwrite:
        raise FileExistsError(f"adapter directory already exists: {dst_dir}")
    tmp = f"{dst_dir}.tmp-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    try:
        for name in os.listdir(src_dir):
            p = os.path.join(src_dir, name)
            if name != WEIGHTS_FILE and os.path.isfile(p):
                shutil.copy2(p, os.path.join(tmp, name))
        save_file(out, os.path.join(tmp, WEIGHTS_FILE),
                  metadata={**metadata, "format": "pt"})
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        os.replace(tmp, dst_dir)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return dst_dir


def save_job_adapter(job, engine, routing=None) -> str | None:
    """What 'ready' does for a finetuning job: write its trained adapter to
    ``ft_job.save_path`` (or, with adapter routing, the pool's adapter
    directory) and publish it to the pool when the job is auto-routable.
    Returns the saved path, or None when there is nowhere to save."""
    from deltaserve import dprint

    spec = job.spec
    dst = spec.save_path or (routing.adapter_dir(job.name) if routing else None)
    if dst is None and job.saved_path is not None:
        return job.saved_path
    if dst is None:
        dprint(f"[ft-job] {job.name} is ready but NOT saved: set ft_job.save_path "
               f"(or enable adapter routing)")
        return None
    if job.saved_path is None:           # the admission stage may have saved it
        if engine is None or not spec.adapter_path:
            raise RuntimeError(
                f"{job.name}: no engine / source adapter to export from")
        import time

        t0 = time.monotonic()
        job.saved_path = export_trained_adapter(engine, spec.adapter_path, dst,
                                                overwrite=True)
        # Runs on the engine loop: no step is scheduled for this long.
        dprint(f"[ft-job] {job.name} trained adapter saved to {job.saved_path} "
               f"in {(time.monotonic() - t0) * 1e3:.0f} ms (engine loop blocked)")
    if routing is not None and spec.auto_routing and job.admission is None:
        routing.publish(job)
    return job.saved_path


def export_trained_adapter(engine, src_dir: str, dst_dir: str,
                           overwrite: bool = False) -> str:
    """Collect the trained LoRA shards from every rank's worker, merge them and
    write the adapter. Call only when no backward is running (the job's
    training stage is done): the export is a synchronous round trip on each
    worker's pipe to its backward child."""
    results = engine.model_executor.collective_rpc("deltaserve_export_ft_adapter")
    results = [r for r in results if r is not None]
    errors = [r["error"] for r in results if isinstance(r, dict)]
    if errors:
        raise RuntimeError(f"adapter export failed on a worker: {errors[0]}")
    if not results:
        raise RuntimeError("no worker has a trained adapter to export")
    results.sort(key=lambda r: r[0])
    ranks = [r[0] for r in results]
    if ranks != list(range(len(ranks))):
        raise RuntimeError(f"adapter export is missing a rank: got {ranks}")
    return write_adapter(src_dir, dst_dir,
                         merge_rank_shards([r[1] for r in results]),
                         overwrite=overwrite)
