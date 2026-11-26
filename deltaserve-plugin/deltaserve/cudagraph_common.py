# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CUDA-graph pieces shared by both runners' FT dispatch (``cudagraph.py`` for
the v1 runner's ``CudagraphDispatcher``, ``cudagraph_v2.py`` for the v2
runner's ``CudaGraphManager``): the ``has_ft`` batch descriptor that keys the
second piecewise graph set, and the abort poll between piecewise segments.
"""

from contextlib import contextmanager
from dataclasses import dataclass

from vllm.forward_context import BatchDescriptor


@dataclass(frozen=True)
class FtBatchDescriptor(BatchDescriptor):
    """A batch with FT samples next to inference tokens, saved by the
    graph-capturable op (its own capture set)."""

    has_ft: bool = True


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


def polled_segment(segment):
    """One replayed segment of a breakable CUDA graph capture, with the
    FT-only abort poll in front of it (``segment_poll`` sets the poll for the
    duration of an abortable step; one global read otherwise)."""
    def run():
        poll = _SEGMENT_POLL
        if poll is not None:
            poll()
        return segment()

    return run
