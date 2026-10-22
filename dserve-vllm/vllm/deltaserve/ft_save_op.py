# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""[DeltaServe] Graph-capturable activation save (``mixed-fwd-cuda-graph``).

The activation hooks in ``accumulate.py`` read per-step Python state (row
positions, the reserved write offset) and therefore cannot live inside a
CUDA graph — which is why every step carrying finetuning tokens used to run
eager. This module is the graph-native replacement for **mixed** batches
(inference requests + FT samples):

* ``torch.ops.vllm.dserve_save_rows(x, r, marker, slot)`` gathers a FIXED
  number of rows (``max_saved``) of ``x`` (+ ``r`` when given — the fused
  add-norm's residual sum) through a persistent *source-index* tensor and
  writes them into the accumulator buffer for ``slot`` through a persistent
  *destination-index* tensor. Both index tensors, and the buffer table, are
  read from the step's ``FtSaveState`` (``active_state()``, set by the model
  runner around the forward / the has_ft captures), never from the graph: the
  FT rows' positions and the reserved offset are the CONTENTS of the index
  tensors, refreshed before each step. Unused slots gather row 0 into a
  scratch row past ``max_saved``.
* Inside vLLM's compiled forward the op is opaque to Dynamo, so its Python
  branch on the active state runs at *capture* time: the plain graphs
  (captured with no active state) contain no kernels for it, the ``has_ft`` graphs
  contain the gathers with the persistent addresses baked in, and the
  compiled-but-uncaptured path evaluates the branch every step.
* ``marker`` is a 1-element buffer on the calling module declared as
  mutated. Without a declared mutation the op has no outputs and would be
  dead-code-eliminated by Dynamo / Inductor; the marker costs one tiny
  in-place kernel per call and nothing else.

The op is issued by the accumulator's save-point hooks (``accumulate.py``) in
their inactive branch — the branch Dynamo traces into the compiled forward —
so the model code needs no call sites.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch

from vllm.utils.torch_utils import direct_register_custom_op


@dataclass
class FtSaveState:
    """What the op reads on a mixed FT step (or while capturing a has_ft graph)."""

    src: torch.Tensor           # int64 [max_saved]: rows of x to gather
    dst: torch.Tensor           # int64 [max_saved]: rows of the buffer to write
    bufs: list                  # slot -> buffer [max_saved + 1, width] or None


# The state of the forward in flight; None outside a mixed FT forward / has_ft
# capture. The model runs on one thread per worker process.
_ACTIVE: FtSaveState | None = None


@contextmanager
def active_state(state: FtSaveState | None):
    """Make ``state`` what the op reads for the duration of the block."""
    global _ACTIVE
    prev, _ACTIVE = _ACTIVE, state
    try:
        yield
    finally:
        _ACTIVE = prev


def _dserve_save_rows(x: torch.Tensor, r: torch.Tensor | None,
                      marker: torch.Tensor, slot: int) -> None:
    fs = _ACTIVE
    if fs is None:
        return
    buf = fs.bufs[slot] if slot < len(fs.bufs) else None
    if buf is None:
        return
    rows = x.index_select(0, fs.src)
    if r is not None:
        rows = rows + r.index_select(0, fs.src)
    if rows.dtype != buf.dtype:
        rows = rows.to(buf.dtype)
    buf.index_copy_(0, fs.dst, rows)


def _dserve_save_rows_fake(x: torch.Tensor, r: torch.Tensor | None,
                           marker: torch.Tensor, slot: int) -> None:
    return None


direct_register_custom_op(
    op_name="dserve_save_rows",
    op_func=_dserve_save_rows,
    mutates_args=["marker"],
    fake_impl=_dserve_save_rows_fake,
)


# The 1-element buffer the accumulator registers on every hooked module; the op
# declares it mutated so Dynamo / Inductor keep the call.
MARKER_ATTR = "_dserve_marker"
