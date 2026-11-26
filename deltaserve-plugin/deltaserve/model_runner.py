# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""``DeltaServeModelRunner``: vLLM's v1 ``GPUModelRunner`` plus the per-step
co-serving work.

``DeltaServeWorker.init_device`` adopts the runner the stock worker built
(``adopt``: the instance's class becomes this subclass and its CUDA-graph
dispatcher an ``FtCudagraphDispatcher``), so ``gpu_model_runner.py`` carries no
DeltaServe code. Everything hangs off methods upstream already factors out:

- ``_update_states``: drop the cached state of FT requests the scheduler retired.
- ``_prepare_inputs``: build the step's ``finetune_mask`` (which flat-batch rows
  are FT tokens) and tell the dispatcher whether this step runs eager (FT-only,
  or mixed without the save op) or from the ``has_ft`` graph set.
- ``_model_forward`` (upstream's documented override point): arm the
  activation save, the GPU-timing ring for the SLO estimator, the backward
  pause around inference prefills, and the tier-C mid-forward abort; after the
  forward, save the FT input ids and commit (tp=1) or relay (TP) the rows.
- ``execute_model``: per-step bookkeeping before the stock method (deferred
  backward resume, throttle release, the relayed backward trigger) and the
  idle-step relay output; an aborted FT-only forward becomes the engine's
  roll-back sentinel.
- ``sample_tokens``: the TP relay fields on every output.
- ``_warmup_and_capture``: capture the ``has_ft`` graphs with the save op live.
"""

import time
from copy import copy

from vllm.config import CUDAGraphMode
from deltaserve.cudagraph import (
    FtBatchDescriptor,
    FtCudagraphDispatcher,
    install_segment_poll,
    segment_poll,
)
from deltaserve.ft_save_op import active_state
from deltaserve.runner_common import FtRunnerMixin, _FtStepAborted
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class DeltaServeModelRunner(FtRunnerMixin, GPUModelRunner):
    @classmethod
    def adopt(cls, runner: GPUModelRunner) -> "DeltaServeModelRunner":
        """Turn the stock runner (constructed by the stock worker) into this
        subclass, before any model / KV-cache / graph setup."""
        runner.__class__ = cls
        runner.cudagraph_dispatcher.__class__ = FtCudagraphDispatcher
        runner._ds_init()
        if runner.vllm_config.finetune_config.graph_ft_only_batches:
            install_segment_poll()
        return runner

    # -- per-step hooks into the stock execute path ------------------------

    def execute_model(self, scheduler_output, intermediate_tensors=None):
        coord = self._ft_coordinator
        self._ft_relay_saved = None
        if coord is not None:
            self._ft_before_step(scheduler_output, coord)
        self._ds_sched = scheduler_output
        try:
            out = super().execute_model(scheduler_output, intermediate_tensors)
        except _FtStepAborted as e:
            return self._ft_on_abort(e.where, coord)
        finally:
            self._ds_sched = None
            self.cudagraph_dispatcher.ft_mode = None
        if (out is EMPTY_MODEL_RUNNER_OUTPUT and coord is not None
                and coord.relay_mode):
            # [M4.2] Idle (0-token) steps are how the engine keeps stepping
            # while a backward is outstanding (FinetuneScheduler.has_requests),
            # and this early return is the output the scheduler sees for them
            # (sample_tokens is not reached). Relay the ack + timing here too —
            # otherwise the ack only arrives with the next inference batch and
            # FT stalls through every idle valley. All ranks take this path in
            # lock-step, so the ack's all-reduce is safe. A private copy: the
            # shared EMPTY sentinel must not be mutated.
            out = copy(EMPTY_MODEL_RUNNER_OUTPUT)
            self._ft_fill_relay_fields(out, coord)
        return out

    def _update_states(self, scheduler_output):
        # FT requests retire scheduler-side without entering finished_req_ids
        # (the frontend must never see them); drop their cached state so
        # ``self.requests`` does not grow by one entry per FT injection. Same
        # removals as the stock finished path, so its condense() covers them.
        for req_id in (getattr(scheduler_output, "finetune_retired_req_ids", None)
                       or ()):
            req_state = self.requests.pop(req_id, None)
            if req_state is not None:
                self.num_prompt_logprobs.pop(req_id, None)
                # vLLM >= 0.33: the stock finished path calls this hook too
                # (platform runners clean request-scoped side caches there)
                hook = getattr(self, "_on_request_state_removed", None)
                if hook is not None:
                    hook(req_id, req_state)
            if req_id in self.input_batch.req_id_to_index:
                self.input_batch.remove_request(req_id)
        return super()._update_states(scheduler_output)

    def _prepare_inputs(self, scheduler_output, num_scheduled_tokens_np, *args,
                        **kwargs):
        self._build_finetune_mask(scheduler_output, self.input_batch.req_ids,
                                  num_scheduled_tokens_np)
        # A step with FT tokens runs eager (the hooks read per-step Python
        # state) unless it is a MIXED batch on the graph-capturable save path,
        # which dispatches into the has_ft piecewise set.
        self.cudagraph_dispatcher.ft_mode = (
            None if not self._ft_has
            else "graph" if self._ft_graph_step else "eager")
        return super()._prepare_inputs(scheduler_output, num_scheduled_tokens_np,
                                       *args, **kwargs)

    def _model_forward(self, *args, **kwargs):
        scheduler_output = self._ds_sched
        coord = self._ft_coordinator
        if scheduler_output is None or coord is None:
            return super()._model_forward(*args, **kwargs)
        fwd_ctx = get_forward_context()
        cudagraph_mode = fwd_ctx.cudagraph_runtime_mode
        if self._ft_has and not self._ft_graph_step:
            assert cudagraph_mode == CUDAGraphMode.NONE, (
                f"[deltaserve] FT step must be eager, got {cudagraph_mode}")
            # bypass the compiled graph so the submodule hooks reliably fire;
            # mixed graph steps run compiled with the save op armed.
            fwd_ctx.skip_compiled = True
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

        slot, start_evt, pause_bwd, pause_throttle = self._ft_timing_begin(
            scheduler_output, coord)
        t_wall0 = time.time()
        t_host0 = time.perf_counter()
        start_evt.record()

        # [forward_interruptible / tier C] Arm the per-layer abort path for a
        # pure FT-only forward (a mid-forward abort would cost any inference
        # tokens their step, so co-serve batches are never aborted).
        ft_only = bool(self._ft_has
                       and self.vllm_config.finetune_config.forward_interruptible
                       and self._ft_num == scheduler_output.total_num_scheduled_tokens)
        aborted = None
        # A previous abort's pending marker for sample_tokens is consumed by any
        # new forward (tp=1 routes the sentinel around sample_tokens entirely,
        # so the marker must not leak to a later idle step).
        self._ft_pending_abort = False
        poller = coord.ft_abort_poller
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
                aborted = "entry"
            else:
                # Mark the live FT-only forward so the input thread starts
                # signalling.
                coord.ft_only_in_flight = True

        model_output = None
        try:
            if aborted is None:
                from deltaserve.coordinator import FTAborted
                poll = None
                if ft_only and self._ft_graph_step and poller is not None:
                    # Graph step: the hooks are inactive inside the compiled
                    # forward, so the abort poll runs between graph segments.
                    def poll():
                        if poller.hook_check(True):
                            raise FTAborted()
                try:
                    with active_state(save_state), segment_poll(poll):
                        model_output = super()._model_forward(*args, **kwargs)
                except FTAborted:
                    # Raised by an activation hook on a late arrival.
                    aborted = (f"layer {poller.layer_calls}"
                               if poller is not None else "mid-forward")
        finally:
            ok = aborted is None and model_output is not None
            self._ft_timing_end(scheduler_output, coord, slot, ok,
                                cudagraph_mode, t_wall0, t_host0, pause_bwd,
                                pause_throttle)
            # [forward_interruptible / tier C] Always clear the live-FT-only
            # marker — even on aborted / failed forwards — so the input thread
            # stops signalling; end_step() disarms the hooks of the bailed step.
            if ft_only:
                coord.ft_only_in_flight = False
                if poller is not None:
                    poller.finish()
            if aborted is not None and accumulator is not None:
                accumulator.end_step()
        if aborted is not None:
            raise _FtStepAborted(aborted)

        if (self._ft_has and accumulator is not None
                and get_pp_group().is_last_rank and not self.is_pooling_model):
            hidden = (model_output[0] if self.use_aux_hidden_state_outputs
                      else model_output)
            self._ft_after_forward(hidden, coord, accumulator, save_acts)
        return model_output

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
        if coord is not None and coord.relay_mode:
            # [Phase 7 / M4.1] Relay this rank's backward ack + saved counts to
            # the EngineCore scheduler. Async outputs wrap the ModelRunnerOutput
            # the scheduler eventually receives — set the fields on that one.
            if out is EMPTY_MODEL_RUNNER_OUTPUT:
                # vLLM >= 0.33 returns the shared EMPTY sentinel (formerly
                # None) when there is nothing to sample: fill a private copy,
                # never the singleton every later empty step would share.
                out = copy(EMPTY_MODEL_RUNNER_OUTPUT)
            target = getattr(out, "_model_runner_output", out)
            if isinstance(target, ModelRunnerOutput):
                self._ft_fill_relay_fields(target, coord)
        return out

    def _warmup_and_capture(self, desc, *args, **kwargs):
        if not isinstance(desc, FtBatchDescriptor):
            return super()._warmup_and_capture(desc, *args, **kwargs)
        # A has_ft graph: dispatch into its set and capture with the save op
        # live (the index tensors point at the scratch row).
        acc = self._finetune_accumulator
        self.cudagraph_dispatcher.ft_mode = "graph"
        try:
            with active_state(acc.capture_state() if acc is not None else None):
                return super()._warmup_and_capture(desc, *args, **kwargs)
        finally:
            self.cudagraph_dispatcher.ft_mode = None

    def _ft_step_input_ids(self, total: int):
        return self.input_ids.gpu[:total]
