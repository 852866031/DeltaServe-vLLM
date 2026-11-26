# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CUDA-graph dispatch for batches that carry finetuning samples, on the v2
model runner (``vllm/v1/worker/gpu``; the counterpart of ``cudagraph.py``).

The v2 runner owns a ``ModelCudaGraphManager``: it dispatches a batch to a
``BatchExecutionDescriptor`` *before* the inputs are prepared, and a piecewise
step then runs the compiled model, whose per-segment graphs are keyed by the
forward context's ``BatchDescriptor``. ``FtCudaGraphManager`` (adopted by class
swap right after the runner builds its manager) adds the FT rules on top:

* ``ft_mode == "eager"`` — an FT-only step, or a mixed step without the
  graph-capturable save path: the eager descriptor, whatever was captured.
* ``ft_mode == "graph"`` — a mixed step on the save-op path: the PIECEWISE
  descriptor of its size if a ``has_ft`` graph was captured for it, else eager
  (the compiled forward without a graph). ``run_pw_graph`` swaps the forward
  context's descriptor for an ``FtBatchDescriptor``, so the compiled model
  replays the second graph set and never the plain one of the same shape.
* ``capture`` runs upstream's capture, then a second pass over the ``has_ft``
  sizes with the save op live.
* The breakable CUDA graph (``VLLM_USE_BREAKABLE_CUDAGRAPH``; no
  ``torch.compile``) takes the same path: its wrapper keys its captures by the
  forward context's descriptor too, the capture pass executes the hooks' save
  op, and the abort poll is put between the replayed segments.

The runner sets ``ft_mode`` per step; the scheduler's graph-eligibility query
(``will_use_graph``) passes the mode explicitly and touches no shared state.
"""

from dataclasses import fields, replace

from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import init_logger
from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager

from deltaserve.cudagraph_common import FtBatchDescriptor, polled_segment
from deltaserve.ft_save_op import active_state

logger = init_logger(__name__)

_PASS_NAME = "Capturing has_ft CUDA graphs"


class FtCudaGraphManager(ModelCudaGraphManager):
    # The runner's per-step FT mode: None (no FT tokens), "eager" or "graph".
    ft_mode: str | None = None
    # The descriptor of the runner's last dispatch (the step's real mode).
    last_desc = None
    _ft_capturing = False
    _ft_descs: tuple = ()
    _ft_keys: frozenset = frozenset()
    _ft_accumulator = None

    @classmethod
    def adopt(cls, manager: ModelCudaGraphManager,
              accumulator=None) -> "FtCudaGraphManager":
        """Turn the manager the stock runner built into this subclass and pick
        the ``has_ft`` capture set out of its PIECEWISE descriptors.
        ``accumulator`` is a callable returning the FinetuneAccumulator (it is
        created after the runner)."""
        manager.__class__ = cls
        manager._ft_accumulator = accumulator
        manager._ft_init()
        return manager

    def _ft_init(self) -> None:
        self.ft_mode = None
        self.last_desc = None
        self._ft_capturing = False
        self._ft_descs = ()
        self._ft_keys = frozenset()
        ft_cfg = self.vllm_config.finetune_config
        if not (ft_cfg.enable_finetuning and ft_cfg.graph_ft_batches):
            return
        if self.cudagraph_mode.mixed_mode() != CUDAGraphMode.PIECEWISE:
            return
        # With the breakable CUDA graph (VLLM_USE_BREAKABLE_CUDAGRAPH) the
        # forward is recorded as it runs, hooks included: the save op the
        # hooks issue is executed during the has_ft capture pass and its
        # gathers land in the recorded segments — the same second graph set,
        # keyed by the same FtBatchDescriptor, without Dynamo.
        # Same sizes up to graph_ft_max_tokens, LoRA cases with an adapter
        # active (an FT batch always carries the FT adapter).
        ft_max = int(ft_cfg.graph_ft_max_tokens)
        descs = tuple(
            d for d in self._capture_descs.get(CUDAGraphMode.PIECEWISE, ())
            if d.num_active_loras > 0 and d.num_tokens <= ft_max
            and d.uniform_token_count is None and d.max_query_len is None
            and d.num_ubatches == 1)
        self._ft_descs = descs
        self._ft_keys = frozenset((d.num_tokens, d.num_active_loras)
                                  for d in descs)
        logger.info("[deltaserve] has_ft piecewise cudagraph keys: %d "
                    "(sizes <= %d)", len(descs), ft_max)

    # -- dispatch -----------------------------------------------------------

    def dispatch(self, num_reqs, num_tokens, uniform_token_count,
                 num_active_loras, max_query_len=None, num_ubatches=1):
        desc = self._ft_dispatch(self.ft_mode, num_reqs, num_tokens,
                                 uniform_token_count, num_active_loras,
                                 max_query_len, num_ubatches)
        self.last_desc = desc
        return desc

    def _ft_dispatch(self, mode, num_reqs, num_tokens, uniform_token_count,
                     num_active_loras, max_query_len=None, num_ubatches=1):
        if mode is None:
            return super().dispatch(num_reqs, num_tokens, uniform_token_count,
                                    num_active_loras, max_query_len, num_ubatches)
        if mode == "graph":
            # A mixed FT batch only ever matches the has_ft PIECEWISE set: ask
            # for a token-keyed graph (never a decode one) and keep it only if
            # its has_ft twin was captured.
            desc = super().dispatch(num_reqs, num_tokens, None,
                                    num_active_loras, None, num_ubatches)
            if (desc.cg_mode == CUDAGraphMode.PIECEWISE
                    and (desc.num_tokens, desc.num_active_loras) in self._ft_keys):
                return desc
        # Eager: upstream's own no-graph descriptor (what it returns for a
        # batch with no tokens), at this batch's size.
        none = super().dispatch(num_reqs, 0, None, num_active_loras, None,
                                num_ubatches)
        return replace(none, num_tokens=num_tokens)

    def will_use_graph(self, num_tokens: int, num_reqs: int,
                       uniform_decode: bool, num_active_loras: int,
                       has_ft: bool | None) -> bool:
        """The scheduler's query: would a step of this shape replay a graph?
        ``has_ft``: None = no FT tokens, True = a mixed step on the save-op
        path, False = an FT step that runs eager."""
        mode = None if has_ft is None else ("graph" if has_ft else "eager")
        q = self.decode_query_len if uniform_decode else None
        desc = self._ft_dispatch(mode, num_reqs, num_tokens, q,
                                 num_active_loras, q)
        return desc.cg_mode != CUDAGraphMode.NONE

    # -- forward ------------------------------------------------------------

    def run_pw_graph(self, model, model_inputs):
        if self._ft_capturing or self.ft_mode == "graph":
            ctx = get_forward_context()
            desc = ctx.batch_descriptor
            if desc is not None and not isinstance(desc, FtBatchDescriptor):
                ctx.batch_descriptor = FtBatchDescriptor(
                    **{f.name: getattr(desc, f.name)
                       for f in fields(BatchDescriptor)})
        return super().run_pw_graph(model, model_inputs)

    # -- capture ------------------------------------------------------------

    def capture(self, *args, **kwargs):
        super().capture(*args, **kwargs)
        if not self._ft_descs:
            return
        # The has_ft set: the same sizes again, dispatched into their own
        # graph table (run_pw_graph) and captured with the save op live (the
        # index tensors point at the scratch row).
        acc = self._ft_accumulator() if self._ft_accumulator is not None else None
        saved = self._capture_descs
        self._capture_descs = {CUDAGraphMode.PIECEWISE: list(self._ft_descs)}
        self._ft_capturing = True
        try:
            kwargs["progress_bar_desc"] = _PASS_NAME
            with active_state(acc.capture_state() if acc is not None else None):
                super().capture(*args, **kwargs)
        finally:
            self._ft_capturing = False
            self._capture_descs = saved
        self._ft_poll_breakable_segments()

    def _ft_poll_breakable_segments(self) -> None:
        """[breakable CUDA graph] The tier-C abort poll of an FT-only graph
        step runs between segments. A breakable capture replays a list of
        callables (graph segment, eager attention, graph segment, ...): each
        callable of a has_ft capture is given the poll in front of it."""
        runner = getattr(self, "breakable_cg_runner", None)
        if not self.use_breakable_cg or runner is None:
            return
        n = 0
        for entry in runner.entries.values():
            cap = entry.capture
            if (cap is None or getattr(cap, "_dserve_polled", False)
                    or not isinstance(entry.batch_descriptor, FtBatchDescriptor)):
                continue
            cap.segments = [polled_segment(seg) for seg in cap.segments]
            cap._dserve_polled = True
            n += 1
        logger.info("[deltaserve] has_ft breakable CUDA graphs: %d captures, "
                    "abort poll between their segments", n)
