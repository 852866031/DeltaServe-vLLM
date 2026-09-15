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
  read from the forward context (``ForwardContext.ft_save``), never from the
  graph: the FT rows' positions and the reserved offset are the CONTENTS of
  the index tensors, refreshed before each step. Unused slots gather row 0
  into a scratch row past ``max_saved``.
* Inside vLLM's compiled forward the op is opaque to Dynamo, so its Python
  branch on ``ft_save`` runs at *capture* time: the plain graphs (captured
  with ``ft_save = None``) contain no kernels for it, the ``has_ft`` graphs
  contain the gathers with the persistent addresses baked in, and the
  compiled-but-uncaptured path evaluates the branch every step.
* ``marker`` is a 1-element buffer on the calling module declared as
  mutated. Without a declared mutation the op has no outputs and would be
  dead-code-eliminated by Dynamo / Inductor; the marker costs one tiny
  in-place kernel per call and nothing else.

Call sites (``maybe_save``) sit in the model code at the seven points the
hooks used to watch. They are no-ops when the accumulator has not installed
a slot on the module — i.e. in every non-finetuning deployment.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.utils.torch_utils import direct_register_custom_op


@dataclass
class FtSaveState:
    """What the op reads from the forward context on a mixed FT step."""

    src: torch.Tensor           # int64 [max_saved]: rows of x to gather
    dst: torch.Tensor           # int64 [max_saved]: rows of the buffer to write
    bufs: list                  # slot -> buffer [max_saved + 1, width] or None


def _dserve_save_rows(x: torch.Tensor, r: torch.Tensor | None,
                      marker: torch.Tensor, slot: int) -> None:
    from vllm.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )

    if not is_forward_context_available():
        return
    fs = get_forward_context().ft_save
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


# Attribute names the accumulator installs on a module: the slot int and the
# shared marker buffer. Read with getattr so modules without them (no
# finetuning, or a family without call sites) trace to nothing.
SLOT_ATTR = "_dserve_slot_"
MARKER_ATTR = "_dserve_marker"


def maybe_save(module: torch.nn.Module, name: str, x: torch.Tensor,
               r: torch.Tensor | None = None) -> None:
    """Model-side call site: save the FT rows of ``x`` (+ ``r``) into the
    accumulator slot the accumulator registered on ``module`` under
    ``name``, if any. Traces to a single opaque op (or nothing)."""
    slot = getattr(module, SLOT_ATTR + name, None)
    if slot is None:
        return
    torch.ops.vllm.dserve_save_rows(x, r, getattr(module, MARKER_ATTR), slot)
