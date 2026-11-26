# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""``DeltaServeModelRunnerV2``: vLLM's v2 ``GPUModelRunner``
(``vllm/v1/worker/gpu/model_runner.py`` — the only runner from v0.34 on) plus
the per-step co-serving work. The counterpart of ``model_runner.py``; the work
itself is ``runner_common.FtRunnerMixin`` in both.

``DeltaServeWorker._make_model_runner`` builds it (upstream's override point),
so the stock runner carries no DeltaServe code. The v2 runner has no
``_model_forward``: its forward is inline in ``execute_model``, behind a
dispatch that happens before the inputs are prepared. The hooks therefore sit
on what it does factor out:

- ``finish_requests``: drop the state of FT requests the scheduler retired.
- ``gather_batch_req_state`` (the step's batch order, before the dispatch):
  build the ``finetune_mask`` and tell the graph manager whether this step runs
  eager or from the ``has_ft`` graph set.
- ``step_timing.forward_start()`` (upstream calls it right before the forward
  on every path): ``_ft_forward_begin`` — arm the activation save, the
  GPU-timing ring, the backward pause, the tier-C abort.
- a forward pre-hook on the model: ``skip_compiled`` for a step whose hooks
  must run (it has to be set inside the forward context).
- ``execute_model``: the per-step bookkeeping around the stock method, and
  ``_ft_forward_end`` once it returns or raises — timing end, the FT input ids
  and the commit / relay of the saved rows; an aborted FT-only forward becomes
  the engine's roll-back sentinel.
- ``sample_tokens``: the TP relay fields on every output.
- ``initialize_kv_cache``: adopt the graph manager the runner just built
  (``cudagraph_v2.FtCudaGraphManager``: dispatch + the ``has_ft`` capture pass).
"""

import time
from contextlib import ExitStack
from copy import copy

from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import StepTimingCollector
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from deltaserve.cudagraph_common import install_segment_poll, segment_poll
from deltaserve.cudagraph_v2 import FtCudaGraphManager
from deltaserve.ft_save_op import active_state
from deltaserve.runner_common import FtRunnerMixin, _FtStepAborted


class _FtStepTiming(StepTimingCollector):
    """The runner's step-timing collector, announcing each forward to the
    runner first (``forward_start`` is the one call upstream makes right
    before the forward on the FULL, PIECEWISE and eager paths alike)."""

    _ds_runner = None

    def forward_start(self) -> None:
        runner = self._ds_runner
        if runner is not None:
            runner._ft_forward_begin()
        super().forward_start()


class _Forward:
    """One forward in flight: what ``_ft_forward_begin`` armed and
    ``_ft_forward_end`` has to settle."""

    __slots__ = ("sched", "coord", "accumulator", "save_acts", "slot",
                 "cudagraph_mode", "t_wall0", "t_host0", "pause_bwd",
                 "pause_throttle", "ft_only", "poller", "stack")


class DeltaServeModelRunnerV2(FtRunnerMixin, GPUModelRunner):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._ds_init()
        self._ds_fwd: _Forward | None = None
        self._ds_skip_compiled = False
        self._ds_hooked_model = None
        timing = self.step_timing
        timing.__class__ = _FtStepTiming
        timing._ds_runner = self
        if self.vllm_config.finetune_config.graph_ft_only_batches:
            install_segment_poll()

    # The object the scheduler queries for graph eligibility (tp=1).
    @property
    def cudagraph_dispatcher(self):
        return self.cudagraph_manager

    def _ft_step_input_ids(self, total: int):
        return self.execute_model_state.input_batch.input_ids[:total]

    def initialize_kv_cache(self, *args, **kwargs):
        out = super().initialize_kv_cache(*args, **kwargs)
        # Built anew on every call (once for graph-memory profiling, once for
        # real): adopt it before anything is captured.
        if self.cudagraph_manager is not None:
            FtCudaGraphManager.adopt(
                self.cudagraph_manager, lambda: self._finetune_accumulator)
        return out

    def profile_run(self, *args, **kwargs):
        # The stock profile run selects max_loras dummy adapters and ends by
        # removing every adapter, the pinned FT one included (it is loaded
        # again by the first FT request — on the v1 runner too, whose profile
        # run selects none). With the FT adapter still holding a slot the last
        # dummy has no room: remove it before instead of after.
        if self.lora_config is not None and self.lora_manager is not None:
            self.lora_manager.remove_all_adapters()
        return super().profile_run(*args, **kwargs)

    # -- per-step hooks into the stock execute path ------------------------

    def execute_model(self, scheduler_output, intermediate_tensors=None,
                      dummy_run: bool = False, **kwargs):
        if dummy_run:
            # Warmup / profiling / capture forwards: nothing of ours.
            return super().execute_model(scheduler_output, intermediate_tensors,
                                         dummy_run=True, **kwargs)
        coord = self._ft_coordinator
        self._ft_relay_saved = None
        if coord is not None:
            self._ft_before_step(scheduler_output, coord)
        self._ds_sched = scheduler_output
        self._ds_fwd = None
        manager = self.cudagraph_manager
        if manager is not None:
            manager.last_desc = None
        try:
            try:
                out = super().execute_model(scheduler_output,
                                            intermediate_tensors, **kwargs)
            except BaseException as e:
                where = self._ft_forward_end(e)
                if where is None:
                    raise
                return self._ft_on_abort(where, coord)
            self._ft_forward_end(None)
        finally:
            self._ds_sched = None
            self._ds_fwd = None
            self._ds_skip_compiled = False
            if manager is not None:
                manager.ft_mode = None
        if (out is EMPTY_MODEL_RUNNER_OUTPUT and coord is not None
                and coord.relay_mode):
            # [M4.2] Idle (0-token) steps are how the engine keeps stepping
            # while a backward is outstanding, and this early return is the
            # output the scheduler sees for them (sample_tokens is not
            # reached). Relay the ack + timing here too. All ranks take this
            # path in lock-step, so the ack's all-reduce is safe. A private
            # copy: the shared EMPTY sentinel must not be mutated.
            out = copy(EMPTY_MODEL_RUNNER_OUTPUT)
            self._ft_fill_relay_fields(out, coord)
        return out

    def finish_requests(self, scheduler_output) -> None:
        # FT requests retire scheduler-side without entering finished_req_ids
        # (the frontend must never see them): free their slot here. Sorted,
        # like the stock finished path, so every TP rank frees in one order.
        for req_id in sorted(getattr(scheduler_output,
                                     "finetune_retired_req_ids", None) or ()):
            self._remove_request(req_id)
        return super().finish_requests(scheduler_output)

    def gather_batch_req_state(self, scheduler_output, dummy_run):
        state, uniform = super().gather_batch_req_state(scheduler_output,
                                                        dummy_run)
        if state is not None and self._ds_sched is not None:
            # ``state.req_ids`` is the order of the rows in the flat batch
            # (the v2 runner sorts it each step).
            self._build_finetune_mask(scheduler_output, state.req_ids,
                                      state.num_scheduled_tokens)
            manager = self.cudagraph_manager
            if manager is not None:
                # A step with FT tokens runs eager (the hooks read per-step
                # Python state) unless it is on the graph-capturable save
                # path, which dispatches into the has_ft piecewise set.
                manager.ft_mode = (
                    None if not self._ft_has
                    else "graph" if self._ft_graph_step else "eager")
        return state, uniform

    def _ds_hook_model(self) -> None:
        model = self.model
        if self._ds_hooked_model is model:
            return
        self._ds_hooked_model = model

        def pre(module, args):
            # Inside the forward context, before the compiled submodule is
            # called: an FT step on the hook path bypasses the compiled graph
            # so the submodule hooks reliably fire.
            if self._ds_skip_compiled:
                get_forward_context().skip_compiled = True

        model.register_forward_pre_hook(pre)

    def _ft_forward_begin(self) -> None:
        scheduler_output = self._ds_sched
        coord = self._ft_coordinator
        if scheduler_output is None or coord is None:
            return
        manager = self.cudagraph_manager
        desc = manager.last_desc if manager is not None else None
        cudagraph_mode = desc.cg_mode if desc is not None else CUDAGraphMode.NONE
        if self._ft_has and not self._ft_graph_step:
            assert cudagraph_mode == CUDAGraphMode.NONE, (
                f"[deltaserve] FT step must be eager, got {cudagraph_mode}")
            self._ds_hook_model()
            self._ds_skip_compiled = True
        # Readable per-batch log: shape plus whether THIS step actually ran a
        # CUDA graph (the real dispatch decision, not inferred from has_ft).
        self._log_finetuning_batch(scheduler_output, cudagraph_mode)

        # Arm the activation save. The write offset is the disjoint per-step
        # offset the scheduler reserved for this step (async-safe); fall back
        # to the coordinator's current offset (profiling / no reservation).
        accumulator = self._finetune_accumulator
        save_acts = self.vllm_config.finetune_config.save_activations
        stashed = getattr(scheduler_output, "_ft_write_offset", None)
        self._ft_offset = (int(stashed) if stashed is not None
                           else coord.current_offset())
        save_state = None
        if self._ft_has and accumulator is not None and save_acts:
            if self._ft_graph_step:
                # Graph path: refresh the persistent index tensors; the hooks
                # stay disarmed (they are inert inside the compiled forward).
                save_state = accumulator.arm_op_step(
                    self._ft_positions_np, self._ft_num, self._ft_offset)
            else:
                accumulator.begin_step(
                    self._ft_mask_gpu, self._ft_num, self._ft_offset,
                    start=self._ft_start, contiguous=self._ft_contiguous)

        fwd = _Forward()
        fwd.sched, fwd.coord = scheduler_output, coord
        fwd.accumulator, fwd.save_acts = accumulator, save_acts
        fwd.cudagraph_mode = cudagraph_mode
        fwd.ft_only, fwd.poller = False, None
        fwd.stack = ExitStack()
        fwd.slot, start_evt, fwd.pause_bwd, fwd.pause_throttle = (
            self._ft_timing_begin(scheduler_output, coord))
        fwd.t_wall0 = time.time()
        fwd.t_host0 = time.perf_counter()
        self._ds_fwd = fwd
        start_evt.record()

        # [forward_interruptible / tier C] Arm the per-layer abort path for a
        # pure FT-only forward (a mid-forward abort would cost any inference
        # tokens their step, so co-serve batches are never aborted).
        ft_only = bool(self._ft_has
                       and self.vllm_config.finetune_config.forward_interruptible
                       and self._ft_num == scheduler_output.total_num_scheduled_tokens)
        # A previous abort's pending marker for sample_tokens is consumed by any
        # new forward.
        self._ft_pending_abort = False
        poller = coord.ft_abort_poller
        fwd.ft_only, fwd.poller = ft_only, poller
        if poller is not None and not ft_only:
            # [tier C under TP] An inference-bearing batch: every arrival so
            # far has been scheduled — stop counting them as pending.
            poller.note_served()
        if ft_only:
            # Pipeline-depth-2 contamination: a prior FT-only batch tripped the
            # abort flag, but THIS batch was scheduled before it was set. Bail
            # at entry — no kernels queued. Under TP the entry decision is the
            # poller's rank-symmetric one (every rank bails or none does).
            if (poller.entry() if poller is not None
                    else coord.ft_abort_event.is_set()):
                raise _FtStepAborted("entry")
            # Mark the live FT-only forward so the input thread starts
            # signalling.
            coord.ft_only_in_flight = True

        poll = None
        if ft_only and self._ft_graph_step and poller is not None:
            # Graph step: the hooks are inactive inside the compiled forward,
            # so the abort poll runs between graph segments.
            from deltaserve.coordinator import FTAborted

            def poll():
                if poller.hook_check(True):
                    raise FTAborted()
        fwd.stack.enter_context(active_state(save_state))
        fwd.stack.enter_context(segment_poll(poll))

    def _ft_forward_end(self, error: BaseException | None) -> str | None:
        """Settle the forward ``_ft_forward_begin`` armed. ``error`` is what
        the stock ``execute_model`` raised, if anything; returns where an
        FT-only forward was aborted (tier C), else None."""
        fwd, self._ds_fwd = self._ds_fwd, None
        self._ds_skip_compiled = False
        if fwd is None:
            # No forward was begun: a 0-token step, or a failure before it.
            return None
        fwd.stack.close()
        from deltaserve.coordinator import FTAborted

        coord, accumulator, poller = fwd.coord, fwd.accumulator, fwd.poller
        aborted = None
        if isinstance(error, _FtStepAborted):
            aborted = error.where
        elif isinstance(error, FTAborted):
            # Raised by an activation hook (or the segment poll) on a late
            # arrival.
            aborted = (f"layer {poller.layer_calls}"
                       if poller is not None else "mid-forward")
        ok = error is None
        self._ft_timing_end(fwd.sched, coord, fwd.slot, ok, fwd.cudagraph_mode,
                            fwd.t_wall0, fwd.t_host0, fwd.pause_bwd,
                            fwd.pause_throttle)
        # [forward_interruptible / tier C] Always clear the live-FT-only
        # marker — even on aborted / failed forwards — so the input thread
        # stops signalling; end_step() disarms the hooks of the bailed step.
        if fwd.ft_only:
            coord.ft_only_in_flight = False
            if poller is not None:
                poller.finish()
        if aborted is not None and accumulator is not None:
            accumulator.end_step()
        if not ok:
            return aborted
        state = self.execute_model_state
        if (self._ft_has and accumulator is not None and state is not None
                and state.hidden_states is not None
                and not self.is_pooling_model):
            self._ft_after_forward(state.hidden_states, coord, accumulator,
                                   fwd.save_acts)
        return None

    def sample_tokens(self, grammar_output):
        coord = self._ft_coordinator
        if coord is not None:
            self._ft_maybe_resume_backward(coord)
        if self.execute_model_state is None and self._ft_pending_abort:
            # [tier C under TP] execute_model returned the abort sentinel
            # without leaving sampling state; the engine still calls
            # sample_tokens (its execute future was not resolved at dispatch
            # time), so answer with the sentinel rather than None.
            self._ft_pending_abort = False
            return self._ft_abort_sentinel(coord)
        out = super().sample_tokens(grammar_output)
        if coord is not None and coord.relay_mode and out is not None:
            # [Phase 7 / M4.1] Relay this rank's backward ack + saved counts to
            # the EngineCore scheduler. The async output wraps the
            # ModelRunnerOutput the scheduler eventually receives — set the
            # fields on that one.
            if out is EMPTY_MODEL_RUNNER_OUTPUT:
                out = copy(EMPTY_MODEL_RUNNER_OUTPUT)
            target = getattr(out, "model_runner_output", out)
            if isinstance(target, ModelRunnerOutput):
                self._ft_fill_relay_fields(target, coord)
        return out
