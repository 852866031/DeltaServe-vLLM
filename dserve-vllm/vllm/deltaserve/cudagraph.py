# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CUDA-graph dispatch for batches that carry finetuning samples.

* An FT-only step and a mixed step without the graph-capturable save path
  run eager: the activation hooks read per-step Python state a replay would
  bake in.
* A mixed step on the save-op path (``finetune.graph_ft_batches``) replays a
  graph from a second PIECEWISE set — captured with the save op live — keyed
  by ``FtBatchDescriptor``. Being its own class, it never compares equal to the
  plain ``BatchDescriptor`` of the same shape, so the two sets cannot collide
  in the dispatcher's key sets or the piecewise backend's graph tables.

``FtCudagraphDispatcher`` takes the step's FT mode either per call
(``dispatch(has_ft=...)``: the scheduler's graph-eligibility query) or from
``ft_mode``, which the model runner sets around its own dispatch (the runner's
calls into ``_determine_batch_execution_and_padding`` carry no FT argument).
"""

import threading
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from itertools import product

from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.logger import init_logger
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

logger = init_logger(__name__)


@dataclass(frozen=True)
class FtBatchDescriptor(BatchDescriptor):
    """A batch with FT samples next to inference tokens, saved by the
    graph-capturable op (its own capture set)."""

    has_ft: bool = True


class FtCudagraphDispatcher(CudagraphDispatcher):
    # Per-step FT mode of the runner's own dispatches: None (no FT tokens),
    # "eager" (FT-only, or mixed without the save op) or "graph".
    ft_mode: str | None = None

    def _ft_local(self):
        tl = self.__dict__.get("_ft_tl")
        if tl is None:
            tl = self.__dict__.setdefault("_ft_tl", threading.local())
        return tl

    def _create_padded_batch_descriptor(self, *args, **kwargs) -> BatchDescriptor:
        desc = super()._create_padded_batch_descriptor(*args, **kwargs)
        if getattr(self._ft_local(), "graph", False):
            desc = FtBatchDescriptor(
                **{f.name: getattr(desc, f.name) for f in fields(BatchDescriptor)})
        return desc

    def dispatch(self, num_tokens, uniform_decode=False, has_lora=False,
                 num_active_loras=0, valid_modes=None, invalid_modes=None,
                 has_ft: bool | None = None):
        mode = self.ft_mode if has_ft is None else ("graph" if has_ft else None)
        if mode == "eager":
            valid_modes = {CUDAGraphMode.NONE}
        elif mode == "graph":
            # mixed FT batches only ever match the has_ft PIECEWISE set
            invalid_modes = set(invalid_modes or ()) | {CUDAGraphMode.FULL}
        tl = self._ft_local()
        tl.graph = mode == "graph"
        try:
            return super().dispatch(
                num_tokens, uniform_decode=uniform_decode, has_lora=has_lora,
                num_active_loras=num_active_loras, valid_modes=valid_modes,
                invalid_modes=invalid_modes)
        finally:
            tl.graph = False

    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode,
                                  uniform_decode_query_len: int = 1):
        super().initialize_cudagraph_keys(cudagraph_mode, uniform_decode_query_len)
        ft_cfg = self.vllm_config.finetune_config
        if not (ft_cfg.enable_finetuning and ft_cfg.graph_ft_batches
                and cudagraph_mode.mixed_mode() == CUDAGraphMode.PIECEWISE):
            return
        # Same sizes up to graph_ft_max_tokens, LoRA cases with an adapter
        # active (an FT batch always carries the FT adapter).
        ft_max = int(ft_cfg.graph_ft_max_tokens)
        tl = self._ft_local()
        tl.graph = True
        n = 0
        try:
            for bs, num_active_loras in product(
                    self.compilation_config.cudagraph_capture_sizes,
                    self._get_lora_cases()):
                if bs > ft_max or num_active_loras <= 0:
                    continue
                desc = self._create_padded_batch_descriptor(
                    bs, False, True, num_active_loras)
                self.add_cudagraph_key(
                    CUDAGraphMode.PIECEWISE,
                    replace(desc, num_reqs=None, uniform=False))
                n += 1
        finally:
            tl.graph = False
        logger.info("[deltaserve] has_ft piecewise cudagraph keys: %d "
                    "(sizes <= %d)", n, ft_max)


# ─── between-segment poll for abortable FT-only graph steps ─────────────────
# A piecewise CUDA graph forward replays one segment per layer from Python
# (CUDAGraphWrapper.__call__), with attention in between. While an FT-only
# step runs from those graphs, ``_SEGMENT_POLL`` is the tier-C abort poll: it
# runs before every segment and raises FTAborted to leave the forward.
_SEGMENT_POLL = None


def install_segment_poll() -> None:
    """Wrap CUDAGraphWrapper.__call__ once per process. Outside an FT-only
    graph step the wrapper costs one global read per segment."""
    from vllm.compilation.cuda_graph import CUDAGraphWrapper

    if getattr(CUDAGraphWrapper, "_dserve_segment_poll", False):
        return
    call = CUDAGraphWrapper.__call__

    def __call__(self, *args, **kwargs):
        poll = _SEGMENT_POLL
        if poll is not None:
            poll()
        return call(self, *args, **kwargs)

    CUDAGraphWrapper.__call__ = __call__
    CUDAGraphWrapper._dserve_segment_poll = True


@contextmanager
def segment_poll(poll):
    global _SEGMENT_POLL
    prev, _SEGMENT_POLL = _SEGMENT_POLL, poll
    try:
        yield
    finally:
        _SEGMENT_POLL = prev
