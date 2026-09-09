# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tensor-parallel support for the backward children (Phase 7).

Backward-per-rank: each TP rank trains its own weight shard in its own child.
The only cross-rank traffic is a small set of all-reduces on a dedicated NCCL
group across the children (never vLLM's inference group). ``tp_size == 1``
leaves every helper here inert.
"""

from __future__ import annotations

import os
from typing import Callable

import torch

from vllm.deltaserve import dprint

DEFAULT_BACKWARD_NCCL_PORT = 29677

# The LoRA factors whose gradients are REPLICATED across ranks (summed by an
# all-reduce) vs already this rank's SHARD (no reduce). q/k/v are
# column-parallel (B sharded on output rows, A replicated); o is row-parallel
# (A sharded on input cols, B replicated). Same partition as lora_shard_slice.
REPLICATED_FACTORS = ("qA", "kA", "vA", "oB")
SHARDED_FACTORS = ("qB", "kB", "vB", "oA")

# [M4.3] Layers per gradient bucket: the replicated factor grads of this many
# consecutive layers (in backward order) are reduced with ONE collective on the
# comm stream, overlapping the next group's compute. 8 → 4 buckets on Llama-3-8B,
# 5 on Qwen3-14B, instead of 128 / 160 inline reduces.
BUCKET_LAYERS = 8


def lora_shard_slice(proj: str, ab: str, t: torch.Tensor, tp_rank: int,
                     tp_size: int, local_q: int, local_kv: int) -> torch.Tensor:
    """Slice a FULL PEFT LoRA factor into this TP rank's shard (Phase 7 / M2).

    Matches how vLLM shards the served LoRA buffers we publish into (M4):
      q/k/v (column-parallel): B [out_full, r] sharded on output rows;
                               A [r, hidden]   replicated (returned whole).
      o     (row-parallel):    A [r, in_full]  sharded on input cols;
                               B [hidden, r]   replicated (returned whole).
    The head partition is contiguous per rank, so each shard is one contiguous
    slice at offset ``tp_rank * local_width``. ``tp_size == 1`` → identity, so
    the single-GPU masters are byte-identical to before.
    """
    if tp_size == 1:
        return t
    if proj in ("q", "k", "v") and ab == "B":
        w = local_q if proj == "q" else local_kv
        return t[tp_rank * w:(tp_rank + 1) * w, :]
    if proj == "o" and ab == "A":
        return t[:, tp_rank * local_q:(tp_rank + 1) * local_q]
    return t  # replicated factor (q/k/v A, o B)


def init_backward_tp_group(tp_size: int, tp_rank: int,
                           port: int = DEFAULT_BACKWARD_NCCL_PORT
                           ) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """Init (once) the process group across the backward children and return
    an in-place SUM all-reduce callable. ``None`` for ``tp_size <= 1``.

    Both children reach this ~concurrently (both workers share weights during
    load_model), so the TCP rendezvous completes. NCCL on CUDA, gloo otherwise
    (the CPU gradcheck tests)."""
    if tp_size <= 1:
        return None
    import torch.distributed as dist

    global _CPU_GROUP
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            init_method=f"tcp://127.0.0.1:{port}",
            rank=tp_rank, world_size=tp_size)
        dprint(f"[backward] TP group up: rank={tp_rank}/{tp_size} "
               f"backend={backend} port={port}")
    if _CPU_GROUP is None:
        # A gloo group over the same rendezvous for the rank-symmetric pause
        # (BackwardService._maybe_pause): CPU collectives only, never a GPU
        # kernel, so agreeing to pause cannot itself occupy the GPU.
        _CPU_GROUP = dist.new_group(backend="gloo")

    def _all_reduce(t: torch.Tensor) -> torch.Tensor:
        # NCCL/gloo need contiguous input; reduce is in-place SUM.
        t = t.contiguous()
        dist.all_reduce(t)
        return t

    return _all_reduce


_CPU_GROUP = None


def backward_cpu_group():
    """The gloo group across the backward children (None at tp=1 / before
    ``init_backward_tp_group``)."""
    return _CPU_GROUP


def reduce_partial(all_reduce, total: torch.Tensor,
                   full_part: torch.Tensor) -> torch.Tensor:
    """All-reduce only the per-rank PARTIAL of ``total = partial + full_part``.

    Used for the residual-stream gradients: ``full_part`` (the residual
    passthrough) is already identical on every rank, so summing ``total``
    wholesale would count it ``tp_size`` times. Subtract it, reduce, add it
    back once."""
    fp = full_part.to(total.dtype)
    return all_reduce(total - fp) + fp


# --------------------------------------------------------------------------- #
# [Phase 7 / M4.3] gradient bucketing on a comm stream + rank-symmetric clip
# --------------------------------------------------------------------------- #

class CommQueue:
    """All-reduces on a dedicated CUDA stream, with the ordering kept in ONE
    place. Three hazards, three guarantees:

      1. reduce-after-produce: ``submit`` records an event on the compute
         stream (the caller's current stream) and makes the comm stream wait on
         it before the collective, so the reduce never reads a half-written
         tensor;
      2. consume-after-reduce: ``wait`` makes the compute stream wait on every
         pending completion event; callers read reduced tensors only after it;
      3. no overwrite in flight: callers only submit PERSISTENT buffers (the
         ``FactorBucket`` slabs) that are rewritten no earlier than the next
         cycle, which starts after ``wait`` + the optimizer step.

    Without CUDA (the gloo CPU tests) every submit runs synchronously.
    Debug switches (env, read once): ``DSERVE_BWD_COMM_SYNC=1`` waits right
    after each submit (the async path must be bit-identical to this);
    ``DSERVE_BWD_COMM_DELAY=<cycles>`` spins the comm stream before each
    collective so a consumer that forgot to ``wait`` reads stale data every
    time instead of by chance."""

    def __init__(self, all_reduce: Callable, device) -> None:
        self._reduce = all_reduce
        dev = torch.device(device)
        self._cuda = torch.cuda.is_available() and dev.type == "cuda"
        self.stream = torch.cuda.Stream(device=dev) if self._cuda else None
        self._pending: list = []
        self.sync_mode = os.environ.get("DSERVE_BWD_COMM_SYNC", "0") == "1"
        self.delay_cycles = int(os.environ.get("DSERVE_BWD_COMM_DELAY", "0") or 0)
        self.submitted = 0

    def submit(self, t: torch.Tensor) -> None:
        """Enqueue an in-place SUM all-reduce of ``t`` on the comm stream."""
        self.submitted += 1
        if not self._cuda:
            self._reduce(t)
            return
        cur = torch.cuda.current_stream(t.device)
        ready = torch.cuda.Event()
        ready.record(cur)
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(ready)
            if self.delay_cycles:
                torch.cuda._sleep(self.delay_cycles)
            # dist.all_reduce (sync op) makes the CURRENT stream — the comm
            # stream here — wait for NCCL's internal stream, so ``done`` below
            # fires only after the collective has completed.
            self._reduce(t)
            done = torch.cuda.Event()
            done.record(self.stream)
        self._pending.append(done)
        if self.sync_mode:
            self.wait()

    def wait(self) -> None:
        """Make the compute stream wait for every submitted reduce."""
        if not self._pending:
            return
        cur = torch.cuda.current_stream(self.stream.device)
        for ev in self._pending:
            cur.wait_event(ev)
        self._pending.clear()


class FactorBucket:
    """One persistent flat buffer holding every layer's REPLICATED LoRA-factor
    gradients (``qA``/``kA``/``vA``/``oB``), laid out in backward order
    (layer L-1 first) so the layers of one group are one contiguous slice.
    The trainer copies each layer's four grads into its slab as they are
    produced, submits a group's slice to the ``CommQueue`` once its last layer
    is done, and reads the reduced grads back after ``wait``. Allocated once,
    outside any CUDA-graph pool; rewritten only in the next cycle."""

    def __init__(self, lora: dict, n_layers: int, dtype, device,
                 group_layers: int = BUCKET_LAYERS) -> None:
        self.order = [i for i in reversed(range(n_layers)) if i in lora]
        self.group_layers = max(1, int(group_layers))
        offsets: dict = {}
        total = 0
        for i in self.order:
            for key in REPLICATED_FACTORS:
                p = lora[i][key[0]][key[1]]
                offsets[(i, key)] = (total, tuple(p.shape))
                total += p.numel()
        self.flat = torch.zeros(max(total, 1), dtype=dtype, device=device)
        self._views = {k: self.flat[o:o + _numel(shape)].view(shape)
                       for k, (o, shape) in offsets.items()}
        # Group g = order[g*gl:(g+1)*gl]; its slice spans the first layer's
        # first factor to the last layer's last factor.
        self.groups: list[tuple[int, int, int]] = []   # (last_layer, start, end)
        for g0 in range(0, len(self.order), self.group_layers):
            layers = self.order[g0:g0 + self.group_layers]
            start = offsets[(layers[0], REPLICATED_FACTORS[0])][0]
            o_last, shape_last = offsets[(layers[-1], REPLICATED_FACTORS[-1])]
            self.groups.append((layers[-1], start, o_last + _numel(shape_last)))
        self._submit_after = {last: (s, e) for last, s, e in self.groups}

    @property
    def num_groups(self) -> int:
        return len(self.groups)

    def view(self, layer: int, key: str) -> torch.Tensor:
        return self._views[(layer, key)]

    def stash(self, layer: int, grads: dict) -> None:
        """Copy this layer's four replicated factor grads into their slabs."""
        for key in REPLICATED_FACTORS:
            g = grads.get(key)
            if g is not None:
                self._views[(layer, key)].copy_(g)

    def group_done(self, layer: int) -> torch.Tensor | None:
        """The flat slice to reduce if ``layer`` closes a group, else None."""
        se = self._submit_after.get(layer)
        return None if se is None else self.flat[se[0]:se[1]]


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


@torch.no_grad()
def clip_layers_symmetric_(lora: dict, n_layers: int, all_reduce, max_norm: float = 1.0
                           ) -> torch.Tensor:
    """Per-layer gradient clipping with the SAME semantics as calling
    ``torch.nn.utils.clip_grad_norm_(layer_params, max_norm)`` per layer —
    total = sqrt(sum of squared 2-norms over the layer's 8 LoRA grads),
    coef = min(1, max_norm / (total + 1e-6)), grads scaled in place — but with
    the norm taken over the FULL parameter set under TP: the replicated
    factors' grads are identical on every rank; the sharded factors' squared
    norms are summed across ranks with ONE all-reduce of an ``[n_layers]``
    vector. Every rank therefore derives the same coefficient (the rank-local
    norm each rank used before was missing the other ranks' shards — the M4.3
    clip trap). Returns the per-layer total norms."""
    dev = next(iter(lora.values()))["q"]["A"].device if lora else "cpu"
    sq_shard = torch.zeros(n_layers, dtype=torch.float32, device=dev)
    sq_rep = torch.zeros(n_layers, dtype=torch.float32, device=dev)
    for i, ld in lora.items():
        for proj, ab in ld.items():
            for k, prm in ab.items():
                g = prm.grad
                if g is None:
                    continue
                key = proj + k
                if key in REPLICATED_FACTORS:
                    sq_rep[i] += g.float().pow(2).sum()
                else:
                    sq_shard[i] += g.float().pow(2).sum()
    if all_reduce is not None:
        sq_shard = all_reduce(sq_shard)
    total = torch.sqrt(sq_shard + sq_rep)
    coef = torch.clamp(max_norm / (total + 1e-6), max=1.0)
    for i, ld in lora.items():
        c = coef[i]
        for ab in ld.values():
            for prm in ab.values():
                if prm.grad is not None:
                    prm.grad.mul_(c)
    return total
