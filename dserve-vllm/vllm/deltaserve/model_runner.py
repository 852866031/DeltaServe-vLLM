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

import numpy as np
import torch

from vllm.config import CUDAGraphMode
from vllm.deltaserve import dprint
from vllm.deltaserve.cudagraph import FtBatchDescriptor, FtCudagraphDispatcher
from vllm.deltaserve.ft_save_op import active_state
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

_RING = 4  # timing-event ring; > batch-queue depth (2) so a reused slot is settled


class _FtStepAborted(Exception):
    """An FT-only forward was abandoned (tier C) — unwinds execute_model."""

    def __init__(self, where: str) -> None:
        super().__init__(where)
        self.where = where


class DeltaServeModelRunner(GPUModelRunner):
    @classmethod
    def adopt(cls, runner: GPUModelRunner) -> "DeltaServeModelRunner":
        """Turn the stock runner (constructed by the stock worker) into this
        subclass, before any model / KV-cache / graph setup."""
        runner.__class__ = cls
        runner.cudagraph_dispatcher.__class__ = FtCudagraphDispatcher
        runner._ds_init()
        return runner

    def _ds_init(self) -> None:
        # Wired by DeltaServeWorker after the weights load.
        self._finetune_accumulator = None
        self._backward_process = None
        self._ft_coordinator = None
        self._ft_accum_verified = False
        # Per-step FT layout (_build_finetune_mask).
        self._ft_has = False
        self._ft_mask_gpu = None
        self._ft_num = 0
        self._ft_sample_lens: list[int] = []
        self._ft_start = 0
        self._ft_contiguous = False
        self._ft_positions_np = None
        self._ft_graph_step = False
        self._ft_offset = 0
        self._ds_sched = None
        # Cross-step state.
        self._ft_relay_saved = None
        self._ft_pending_abort = False
        self._throttle_held = False
        self._ft_resume_evt = None
        self._ft_capture_evt = None
        self._ft_timing_events: list | None = None
        self._ft_timing_owner: list = []
        self._ft_timing_pos = 0

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

    def _ft_before_step(self, scheduler_output, coord) -> None:
        # [pause_until_prefill_done] Poll the deferred re-grant on EVERY call,
        # idle steps included: after the last prefill of a burst the engine
        # may only issue 0-token steps, and a rank whose "prefill done" event
        # had not fired yet would otherwise keep its grant cleared for good.
        self._ft_maybe_resume_backward(coord)
        cfg = self.vllm_config.finetune_config
        if self._throttle_held:
            # [fwd_token_throttle] Idle-release safety net: the stock method
            # returns early on a 0-token step, which is exactly the state after
            # a burst ends with the throttle still holding the backward paused;
            # without a release here the cross-step release in
            # ``_ft_pause_decision`` never runs and FT deadlocks. Release
            # whenever the throttle would not (re-)engage on this step — an
            # empty / feature-less step has nothing inference-side to protect.
            feats = getattr(scheduler_output, "_ft_step_features", None)
            release = True
            if feats is not None and cfg.fwd_token_throttle_enable:
                thr = int(cfg.fwd_token_throttle)
                if thr > 0:
                    release = (int(feats.t_in) - int(feats.t_ft)
                               + int(feats.b_d)) < thr
            if release:
                coord.gpu_resume_backward()
                self._throttle_held = False
                coord.fwd_throttle_active = False
        if coord.relay_mode:
            # [Phase 7 / M4.1] Fire a relayed backward trigger even on an idle
            # (0-token) step so a buffer-full / epoch-flush trigger the
            # scheduler emitted is not dropped at the stock early return. Only
            # the IPC send happens here; the ack is polled on later steps.
            cmd = getattr(scheduler_output, "finetune_backward_trigger", None)
            if cmd:
                # The trigger rides on the NEXT step's SchedulerOutput. If that
                # step carries inference work the pause rules protect (a
                # prefill, or a throttle HIT), clear the grant BEFORE the child
                # is signalled: otherwise the child passes its initial
                # ``_maybe_pause`` with the grant still set and runs its first
                # segment against this step (measured at TP=4: a burst's first
                # prefill took 0.55-0.72 s, 18-29 requests queued behind it).
                # The regular pause decision owns the resume bookkeeping.
                feats = getattr(scheduler_output, "_ft_step_features", None)
                if feats is not None:
                    inf_prefill = int(feats.t_in) - int(feats.t_ft)
                    thr = int(cfg.fwd_token_throttle)
                    if inf_prefill > 0 or (
                            cfg.fwd_token_throttle_enable and thr > 0
                            and inf_prefill + int(feats.b_d) >= thr):
                        coord.gpu_pause_backward()
                coord.execute_trigger(cmd)

    def _update_states(self, scheduler_output):
        # FT requests retire scheduler-side without entering finished_req_ids
        # (the frontend must never see them); drop their cached state so
        # ``self.requests`` does not grow by one entry per FT injection. Same
        # removals as the stock finished path, so its condense() covers them.
        for req_id in (getattr(scheduler_output, "finetune_retired_req_ids", None)
                       or ()):
            if self.requests.pop(req_id, None) is not None:
                self.num_prompt_logprobs.pop(req_id, None)
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
                from vllm.deltaserve.coordinator import FTAborted
                try:
                    with active_state(save_state):
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

    def _ft_timing_begin(self, scheduler_output, coord):
        """[Phase 4] GPU-accurate per-step timing for the SLO estimator via a
        ring of CUDA-event pairs, so the hot path never synchronizes: each step
        records into a slot, and the slot's PREVIOUS occupant (RING steps old,
        hence complete) is drained to the coordinator queue the scheduler feeds
        its tracker from. Also decides this step's backward pause."""
        if self._ft_timing_events is None:
            self._ft_timing_events = [
                (torch.cuda.Event(enable_timing=True),
                 torch.cuda.Event(enable_timing=True)) for _ in range(_RING)]
            self._ft_timing_owner = [None] * _RING
            self._ft_timing_pos = 0
        # Share vLLM's real cudagraph dispatcher with the scheduler (same
        # process on single-GPU) so it can query graph eligibility directly.
        if coord.cudagraph_dispatcher is None:
            coord.cudagraph_dispatcher = self.cudagraph_dispatcher
        slot = self._ft_timing_pos
        start_evt, end_evt = self._ft_timing_events[slot]
        prev = self._ft_timing_owner[slot]
        if prev is not None and end_evt.query():
            prev_sched, prev_rec, prev_graph, prev_extra = prev
            prev_feats = getattr(prev_sched, "_ft_step_features", None)
            if prev_rec and prev_feats is not None:
                # was_graph = the mode the runner actually used (the scheduler
                # cannot see the dispatcher under TP).
                coord.push_sample(
                    prev_feats, start_evt.elapsed_time(end_evt) / 1000.0,
                    prev_graph, getattr(prev_sched, "_ft_step_predicted", None),
                    prev_extra)
        self._ft_timing_owner[slot] = None
        pause_bwd, pause_throttle = self._ft_pause_decision(scheduler_output, coord)
        return slot, start_evt, pause_bwd, pause_throttle

    def _ft_pause_decision(self, scheduler_output, coord):
        """[Phase 5] Yield the GPU to this forward: pause the backward child
        for an INFERENCE prefill (TTFT-critical), and — with the fwd-token
        throttle — while the step's inference token count reaches the
        threshold. Decode-only steps do not pause it. Gated on a pending
        backward unless ``pause_prefill_always``. Returns
        (paused, paused_by_throttle); the throttle pause is sticky across
        consecutive heavy steps and released here once a step drops below."""
        cfg = self.vllm_config.finetune_config
        feats = getattr(scheduler_output, "_ft_step_features", None)
        pause_bwd = pause_throttle = False
        if feats is not None and (coord.pending_backward
                                  or cfg.pause_prefill_always):
            # ``t_in`` includes the FT subset: an FT-only step has no inference
            # work to protect (and pausing on it left ranks disagreeing on the
            # grant once the engine went idle — see pause_prefill_always).
            inf_prefill = int(feats.t_in) - int(feats.t_ft)
            if cfg.fwd_token_throttle_enable:
                thr = int(cfg.fwd_token_throttle)
                if thr > 0:
                    # ``>=``: with a threshold of 1 the throttle must also cover
                    # the single-token decode step after a burst's first prefill.
                    pause_throttle = inf_prefill + int(feats.b_d) >= thr
            pause_bwd = inf_prefill > 0 or pause_throttle
        if self._throttle_held and not pause_throttle:
            coord.gpu_resume_backward()
            self._throttle_held = False
            coord.fwd_throttle_active = False
        if pause_bwd:
            coord.gpu_pause_backward()
            # Remember whether the THROTTLE holds the pause (sticky), and mirror
            # it for the scheduler's pre-pause check in ``_trigger_backward``.
            self._throttle_held = pause_throttle
            coord.fwd_throttle_active = pause_throttle
        return pause_bwd, pause_throttle

    def _ft_timing_end(self, scheduler_output, coord, slot, ok, cudagraph_mode,
                       t_wall0, t_host0, pause_bwd, pause_throttle) -> None:
        if ok:
            # Record the end event (read RING steps later) and own the slot with
            # whether to record it — sampled now, so the warmup/recorded gate is
            # right. [M4.2] The gate rides SchedulerOutput (the worker's own
            # flag is never toggled under TP).
            _, end_evt = self._ft_timing_events[slot]
            end_evt.record()
            self._ft_timing_owner[slot] = (
                scheduler_output,
                bool(getattr(scheduler_output, "finetune_record_timing",
                             coord.record_timing)),
                bool(cudagraph_mode is not None
                     and cudagraph_mode != CUDAGraphMode.NONE),
                # [step_trace] (seq, t_exec wall, host dispatch s, paused)
                (getattr(scheduler_output, "_ft_step_seq", None),
                 t_wall0, time.perf_counter() - t_host0, pause_bwd))
            self._ft_timing_pos = (slot + 1) % len(self._ft_timing_events)
        # [Phase 5] Return the GPU to the backward child after enqueuing the
        # prefill forward — fire-and-forget, never a host synchronize; in the
        # finally path so a failed forward cannot leave the backward stalled.
        # The sticky throttle pause is released by _ft_pause_decision instead.
        if pause_bwd and not pause_throttle:
            if ok and self.vllm_config.finetune_config.pause_until_prefill_done:
                # Keep the child paused until this prefill has COMPLETED on the
                # GPU: resume when an event recorded behind it has fired
                # (``_ft_maybe_resume_backward``; the child's 5 s cap bounds a
                # missed resume).
                ev = torch.cuda.Event()
                ev.record()
                self._ft_resume_evt = ev
            else:
                coord.gpu_resume_backward()

    def _ft_after_forward(self, hidden_states, coord, accumulator,
                          save_acts) -> None:
        """Save the FT tokens' input ids, mark the step's activation writes
        done, then commit the rows (tp=1) or relay their count (TP)."""
        if save_acts:
            total = self._ft_mask_gpu.numel()
            accumulator.accumulate_final(
                hidden_states, self.input_ids.gpu[:total], self._ft_mask_gpu,
                self._ft_num, self._ft_offset, start=self._ft_start,
                contiguous=self._ft_contiguous)
            accumulator.end_step()
            self._maybe_verify_ft_accumulation()
            # The backward trigger waits on this event (not a device-wide
            # synchronize): it only needs the save copies visible to the child.
            evt = self._ft_capture_evt
            if evt is None:
                evt = self._ft_capture_evt = torch.cuda.Event()
            evt.record()
            coord.capture_done_evt = evt
        # Advance the coordinator either way, so the admission / backward cycle
        # is identical for the save-on/off A/B.
        if coord.relay_mode:
            # [Phase 7 / M4.1] The EngineCore scheduler owns the trigger
            # decision under TP: relay this step's saved counts to it.
            self._ft_relay_saved = (int(self._ft_num), list(self._ft_sample_lens))
        else:
            coord.record_capture(self._ft_num, self._ft_sample_lens)

    def _ft_on_abort(self, where: str, coord):
        # [forward_interruptible / tier C] The post-forward path is invalid
        # without a model output: hand the engine a sentinel to roll back, and
        # zero the partial writes at this batch's offset (the backward would not
        # read them — fill_count guards — but the buffer stays easy to reason
        # about).
        if self._finetune_accumulator is not None:
            self._finetune_accumulator.zero_offset_range(self._ft_offset,
                                                         self._ft_num)
        dprint(f"[ft-abort] tier C: FT-only forward ({self._ft_num} tokens) "
               f"aborted at {where} for a late inference arrival")
        # sample_tokens (called by the engine under TP) returns the same
        # sentinel from its no-state path when this is set.
        self._ft_pending_abort = True
        return self._ft_abort_sentinel(coord)

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

    # -- helpers ------------------------------------------------------------

    def _build_finetune_mask(
        self,
        scheduler_output,
        req_ids: list[str],
        num_scheduled_tokens_np: np.ndarray,
    ) -> None:
        """[DeltaServe] Build the per-token finetune_mask in flat-batch order.

        Sets self._ft_has (bool, no device sync) and self._ft_mask_gpu (bool
        tensor [total_num_scheduled_tokens] or None). Token spans come from the
        InputBatch request order (req_ids) + the per-request scheduled-token
        counts — NOT scheduler-dict order.

        Also computes (self._ft_start, self._ft_contiguous) for the slice-
        based fast path in the accumulator: if all FT-True positions form a
        single uninterrupted span, hooks can use ``val[start:start+n]``
        instead of ``val[mask]`` (saves the index_select kernel + gather
        allocation per layer). Contiguity isn't guaranteed by vLLM (an FT
        request can fill a freed inference slot mid-batch), so the mask is
        kept as the fallback path.
        """
        self._ft_has = False
        self._ft_mask_gpu = None
        self._ft_num = 0
        self._ft_sample_lens = []
        self._ft_start = 0
        self._ft_contiguous = False
        self._ft_positions_np = None
        self._ft_graph_step = False
        ft_ids = getattr(scheduler_output, "finetune_req_ids", None)
        if not ft_ids:
            return
        total = int(num_scheduled_tokens_np.sum())
        cu = np.concatenate(([0], np.cumsum(num_scheduled_tokens_np))).astype(np.int64)
        mask = np.zeros(total, dtype=bool)
        sample_lens: list[int] = []
        ft_first = -1
        ft_last = -1
        for i, req_id in enumerate(req_ids):
            if req_id in ft_ids:
                mask[cu[i]:cu[i + 1]] = True
                # req_ids order == the order rows appear in hidden_states[mask],
                # so these per-sample lengths line up with the buffer writes.
                sample_lens.append(int(cu[i + 1] - cu[i]))
                if ft_first < 0:
                    ft_first = int(cu[i])
                ft_last = int(cu[i + 1])  # exclusive end
        n = int(mask.sum())
        if n == 0:
            return
        self._ft_has = True
        self._ft_num = n
        self._ft_sample_lens = sample_lens
        # Slice-path metadata: span is [ft_first, ft_last). Contiguous iff the
        # span's length equals the total FT token count (no inference rows
        # interleaved within the span).
        self._ft_start = ft_first
        self._ft_contiguous = (ft_last - ft_first) == n
        self._ft_mask_gpu = torch.from_numpy(mask).to(
            self.device, non_blocking=True
        )
        # [mixed-fwd-cuda-graph] A MIXED batch (inference tokens present) may
        # run the compiled forward + has_ft graphs with the save op; FT-only
        # batches keep the eager hook path (mid-forward abort). Decided here,
        # before dispatch, from the config + the accumulator's readiness.
        _acc = getattr(self, "_finetune_accumulator", None)
        if (n < total and _acc is not None and getattr(_acc, "op_ready", False)
                and self.vllm_config.finetune_config.graph_ft_batches
                and self.vllm_config.finetune_config.save_activations):
            self._ft_positions_np = np.flatnonzero(mask)
            self._ft_graph_step = True

    def _log_finetuning_batch(self, scheduler_output: "SchedulerOutput",
                              cudagraph_mode) -> None:
        """[DeltaServe] Print this batch's shape — prefill/ft token counts and the
        per-sample KV-cache size (context length) of each decoding request — plus
        whether it actually ran a CUDA graph (from the real dispatch decision).
        Gated on ``print_engine_batch_exec`` (or the ``print_step_mode`` master
        switch); skipped for decode-only batches."""
        _ftc = self.vllm_config.finetune_config
        # Hot-path early-out: one attribute load + short-circuit OR before any
        # per-batch accounting below.
        if not (_ftc.print_engine_batch_exec or _ftc.print_step_mode):
            return
        _coord = getattr(self, "_ft_coordinator", None)
        if _coord is not None and getattr(_coord, "profiling", False):
            return  # offline profiling: a progress bar is shown instead
        ft_ids = getattr(scheduler_output, "finetune_req_ids", set())
        new_ids = {r.req_id for r in scheduler_output.scheduled_new_reqs}
        cached = scheduler_output.scheduled_cached_reqs
        computed = dict(zip(cached.req_ids, cached.num_computed_tokens))
        prefill = ft = 0
        decode_kv: list[int] = []
        for req_id, nt in scheduler_output.num_scheduled_tokens.items():
            if req_id in ft_ids:
                ft += nt
            elif req_id in new_ids or nt > 1:  # new / chunked prefill
                prefill += nt
            else:  # single-token running req = decode; KV size = context length
                decode_kv.append(computed.get(req_id, 0) + nt)
        if prefill == 0 and ft == 0:
            return  # decode-only batch: not interesting, skip
        from vllm.deltaserve import dprint

        eager = cudagraph_mode == CUDAGraphMode.NONE
        mode = "eager" if eager else f"graph({getattr(cudagraph_mode, 'name', cudagraph_mode)})"
        total = prefill + ft + len(decode_kv)
        # Scheduler occupancy when this batch was built (stamped by the
        # FinetuneScheduler): running (incl. in-flight FT awaiting retire),
        # running_inf (inference-only subset), and waiting.
        run = getattr(scheduler_output, "_ft_running", -1)
        run_inf = getattr(scheduler_output, "_ft_running_inf", -1)
        wait = getattr(scheduler_output, "_ft_waiting", -1)
        # FT admission state from the coordinator: OPEN/CLOSED, whether a
        # backward is in flight, and the activation-buffer fill (committed +
        # reserved-in-flight) vs capacity.
        coord = getattr(self, "_ft_coordinator", None)
        _st = getattr(scheduler_output, "finetune_admit_state", None)
        if _st is None and coord is not None:
            # No stamp (non-FinetuneScheduler step): read the local coordinator.
            # Under TP that mirror carries no buffer accounting — the stamped
            # scheduler-side state above is the truthful source.
            _st = (coord.admission_open, coord.pending_backward,
                   coord.fill_count + coord.reserved_fill, coord.capacity)
        if _st is not None:
            _open, _pending, _fill, _cap = _st
            admit = "CLOSED" if (not _open or _pending) else "OPEN"
            bwd = " bwd" if _pending else ""
            admit_str = f" | FT {admit}{bwd} buf={_fill}/{_cap}"
        else:
            admit_str = ""
        import time as _time
        ts = _time.strftime("%H:%M:%S") + f".{int((_time.time() % 1) * 1000):03d}"
        # Elapsed since FT admission opened (POST /start_finetuning); t=0 until set.
        ft_t0 = getattr(coord, "ft_start_time", None) if coord is not None else None
        elapsed = (_time.monotonic() - ft_t0) if ft_t0 is not None else 0.0
        # Surface the fwd-token-throttle state in the batch log when the
        # feature is enabled: makes "bwd looks stalled" easy to diagnose.
        # The HIT decision uses INFERENCE-ONLY tokens (prefill + decode,
        # excluding FT) — matching the actual throttle in execute_model
        # — because an FT-only batch has no inference work to protect
        # the bwd from, so the throttle shouldn't fire for it. The
        # displayed ``(total=N)`` field above still shows the full
        # batch count for context; the throttle line shows the
        # inf-only total it compared against.
        throttle_part = ""
        if _ftc.fwd_token_throttle_enable:
            _thr = int(_ftc.fwd_token_throttle)
            _inf_total = prefill + len(decode_kv)
            _hit = _inf_total >= _thr if _thr > 0 else False
            throttle_part = (f" | fwd_throttle={_thr} "
                             f"inf_total={_inf_total} "
                             f"({'HIT' if _hit else 'ok'})")
        dprint(
            f"[batch {ts} t={elapsed:+.3f}s] prefill={prefill} ft={ft} "
            f"decode={decode_kv} (total={total}) | {mode} | running={run} "
            f"(inf={run_inf}) waiting={wait}{admit_str}{throttle_part}"
        )

    def _maybe_verify_ft_accumulation(self) -> None:
        """[DeltaServe] One-shot: hash the accumulated FT activation buffers on
        the parent and compare to the backward process's mapped view (zero-copy).
        """
        if getattr(self, "_ft_accum_verified", True):
            return
        if not self.vllm_config.finetune_config.print_activation_hash:
            self._ft_accum_verified = True
            return
        backward = getattr(self, "_backward_process", None)
        accumulator = getattr(self, "_finetune_accumulator", None)
        if backward is None or accumulator is None:
            return
        from vllm.deltaserve import dprint
        from vllm.deltaserve.backward_process import (
            activation_hash_report,
            print_hash_report,
        )

        torch.cuda.synchronize()  # ensure writes are visible cross-process
        n = self._ft_num
        parent = activation_hash_report(accumulator.buffers, n)
        print_hash_report(parent, "parent")
        child = backward.hash_activations(n)
        dprint(f"[accumulate] activation hashes parent==child: {parent == child} "
               f"(n={n} FT tokens)")
        self._ft_accum_verified = True

    def _ft_relay_backward_done(self, coord) -> dict | None:
        """[DeltaServe] Phase 7 / M4.2: the backward ack to relay this step, or
        None. Only ``output_rank``'s output reaches the scheduler, so an ack is
        handed over only once EVERY rank's child is done: each rank polls its
        own child, the ranks MIN-all-reduce a "mine is done" flag over the TP
        group's CPU (gloo) group — no GPU sync, and only while a backward is
        outstanding — and all take their ack on the same step."""
        if not coord.relay_backward_outstanding():
            return None
        mine = coord.poll_own_backward_ack()
        from vllm.distributed.parallel_state import get_tp_group

        tp = get_tp_group()
        if tp.world_size > 1:
            flag = torch.tensor([1 if mine else 0], dtype=torch.int32)
            torch.distributed.all_reduce(
                flag, op=torch.distributed.ReduceOp.MIN, group=tp.cpu_group)
            all_done = bool(flag.item())
        else:
            all_done = mine
        return coord.take_relay_ack() if all_done else None

    def _ft_fill_relay_fields(self, output, coord) -> None:
        """[DeltaServe] Phase 7: set the worker → scheduler relay fields on
        ``output`` (called on every step's output under TP — from
        ``sample_tokens`` for real batches and from ``execute_model``'s idle
        early return for 0-token steps)."""
        output.finetune_backward_done = self._ft_relay_backward_done(coord)
        output.finetune_saved = getattr(self, "_ft_relay_saved", None)
        self._ft_relay_saved = None
        # Mirror ft_started (set on THIS worker coord by the
        # deltaserve_start_finetuning collective_rpc) so the EngineCore
        # scheduler coord — a separate process under TP — opens admission.
        output.finetune_ft_started = bool(coord.ft_started)
        # [M4.2] Relay the timing samples the CUDA-event ring completed
        # (pushed to THIS worker coord in execute_model). Drained every step,
        # so the worker-side queue stays bounded.
        _samples = coord.drain_completed_samples()
        output.finetune_timing = _samples or None

    def _ft_maybe_resume_backward(self, coord) -> None:
        """[pause_until_prefill_done] Resume the backward child once the
        prefill that paused it has completed on the GPU (non-blocking event
        query). Called on every execute_model (incl. idle steps) and
        sample_tokens; a newer pause simply replaces the pending event."""
        ev = getattr(self, "_ft_resume_evt", None)
        if ev is not None and ev.query():
            self._ft_resume_evt = None
            coord.gpu_resume_backward()

    def _ft_abort_sentinel(self, coord):
        """[forward_interruptible / tier C] The empty output that tells the
        engine to roll the FT-only step back. Carries the TP relay fields so
        an outstanding backward ack is not delayed by the aborted step."""
        from vllm.v1.outputs import ModelRunnerOutput
        out = ModelRunnerOutput(req_ids=[], req_id_to_index={})
        out._ft_aborted = True
        out.finetune_aborted = True
        if coord is not None and getattr(coord, "relay_mode", False):
            self._ft_fill_relay_fields(out, coord)
        return out
