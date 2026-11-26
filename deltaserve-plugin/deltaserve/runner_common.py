# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""``FtRunnerMixin``: the per-step co-serving logic that does not depend on
which vLLM model runner hosts it.

Two runners use it: ``model_runner.DeltaServeModelRunner`` (vLLM's v1
``GPUModelRunner``, removed upstream after v0.33) and
``model_runner_v2.DeltaServeModelRunnerV2`` (``vllm/v1/worker/gpu``, the only
runner from v0.34 on). Each hosts the hooks on the methods its runner offers and
calls into here for the work itself: the FT mask, the backward pause / resume
decisions, the GPU-timing ring for the SLO estimator, the post-forward save,
the tier-C abort outcome and the TP relay fields.

A host provides ``cudagraph_dispatcher`` (the object the scheduler queries for
graph eligibility) and ``_ft_step_input_ids(total)`` (this step's flat input
ids on the GPU).
"""

import time

import numpy as np
import torch

from vllm.config import CUDAGraphMode
from deltaserve import dprint
from vllm.v1.outputs import ModelRunnerOutput

_RING = 4  # timing-event ring; > batch-queue depth (2) so a reused slot is settled


class _FtStepAborted(Exception):
    """An FT-only forward was abandoned (tier C) — unwinds execute_model."""

    def __init__(self, where: str) -> None:
        super().__init__(where)
        self.where = where


def _empty_output() -> ModelRunnerOutput:
    """A fresh empty ModelRunnerOutput (vLLM <= 0.33 requires
    ``req_id_to_index``; later versions dropped the field)."""
    if "req_id_to_index" in ModelRunnerOutput.__dataclass_fields__:
        return ModelRunnerOutput(req_ids=[], req_id_to_index={})
    return ModelRunnerOutput(req_ids=[])


class FtRunnerMixin:

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
        self._ft_prefill_evt = None
        self._ft_capture_evt = None
        self._ft_timing_events: list | None = None
        self._ft_timing_owner: list = []
        self._ft_timing_pos = 0

    def _ft_before_step(self, scheduler_output, coord) -> None:
        # [pause_until_prefill_done] Poll the deferred re-grant on EVERY call,
        # idle steps included: after the last prefill of a burst the engine
        # may only issue 0-token steps, and a rank whose "prefill done" event
        # had not fired yet would otherwise keep its grant cleared for good.
        self._ft_maybe_resume_backward(coord)
        cfg = self.vllm_config.finetune_config
        if getattr(self, "_ds_route_hold", False):
            # [adapter routing] The encode step before this one cleared the
            # grant and left it cleared: an encode and the prefill it leads to
            # are one stretch of TTFT-critical work. This step decides. With an
            # inference prefill the pause simply continues
            # (``_ft_pause_decision`` takes it over and owns the resume);
            # without one — the routed request is held for a cold adapter, was
            # cancelled, did not fit — the child gets the GPU back now.
            feats = getattr(scheduler_output, "_ft_step_features", None)
            if feats is None or int(feats.t_in) - int(feats.t_ft) <= 0:
                self._ds_route_hold = False
                stats = getattr(self, "_ds_route_stats", None)
                if stats is not None:
                    stats["pause_released_no_prefill"] += 1
                if not self._throttle_held:
                    coord.gpu_resume_backward()
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
            if getattr(scheduler_output, "finetune_pause_backward", False):
                release = False     # [avg_tbt_pause_backward] still held
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
                    if inf_prefill > 0 or self._ft_pauses_backward(feats, coord) or (
                            cfg.fwd_token_throttle_enable and thr > 0
                            and inf_prefill + int(feats.b_d) >= thr
                    ) or getattr(scheduler_output, "finetune_pause_backward", False):
                        coord.gpu_pause_backward()
                # Under async scheduling the PREVIOUS step may still be on the
                # GPU. If it carried an inference prefill, that prefill decided
                # "nothing to pause" (no backward was pending yet) and would run
                # against the fresh child: measured 1.5-4x slower prefills
                # (114-156 ms for ~120 tokens) and a burst's TTFT tail past the
                # SLO. Start the child paused and resume it when that prefill
                # has completed (polled like pause_until_prefill_done).
                evt = self._ft_prefill_evt
                if evt is not None and not evt.query():
                    coord.gpu_pause_backward()
                    self._ft_resume_evt = evt
                coord.execute_trigger(cmd)

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
            # [avg_tbt_pause_backward] The scheduler found that this decode
            # step, sharing the GPU with the backward, would take the average
            # time between tokens past its target. Held like the throttle's
            # pause: across consecutive such steps, without a resume in the
            # gap between two of them (the child would run a segment into the
            # next step), and released by the first step that does not ask.
            if getattr(scheduler_output, "finetune_pause_backward", False):
                pause_throttle = True
            pause_bwd = (inf_prefill > 0 or pause_throttle
                         or self._ft_pauses_backward(feats, coord))
        if getattr(self, "_ds_route_hold", False):
            # [adapter routing] The encode step paused the child for the
            # prefill in this step (``_ft_before_step`` kept the pause): this
            # step owns it now, whatever ``pending_backward`` says by now.
            self._ds_route_hold = False
            pause_bwd = True
            stats = getattr(self, "_ds_route_stats", None)
            if stats is not None:
                stats["pause_kept_for_prefill"] += 1
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

    @staticmethod
    def _ft_pauses_backward(feats, coord) -> bool:
        """[bwd_batch_size] With several backward batches per optimizer step
        the forward keeps collecting FT activations while a batch runs. Such a
        step pauses the child exactly like an inference prefill does, so a
        forward that carries FT tokens never shares the GPU with a backward —
        the SLO estimator's FT regimes stay what they were fitted on. Without
        the feature FT is never admitted during a backward: always False."""
        return bool(getattr(coord, "multi_batch", False)
                    and int(feats.t_ft) > 0)

    def _ft_timing_end(self, scheduler_output, coord, slot, ok, cudagraph_mode,
                       t_wall0, t_host0, pause_bwd, pause_throttle) -> None:
        feats = getattr(scheduler_output, "_ft_step_features", None)
        if (ok and coord.relay_mode and feats is not None
                and (int(feats.t_in) > int(feats.t_ft)
                     or self._ft_pauses_backward(feats, coord))):
            # Completion of this step's inference prefill, for a backward
            # trigger that arrives while it is still running (_ft_before_step).
            evt = torch.cuda.Event()
            evt.record()
            self._ft_prefill_evt = evt
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
                hidden_states, self._ft_step_input_ids(total), self._ft_mask_gpu,
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
        # sentinel from its no-state path when this is set — that one carries
        # the TP relay fields: the engine reads model output from the sample
        # future, so relay fields here would be consumed (an ack taken) and
        # then dropped. Without a sample_tokens call they ride the next output.
        self._ft_pending_abort = True
        return self._ft_abort_sentinel(coord, relay=False)

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
        _cfg = self.vllm_config.finetune_config
        if ((n < total or _cfg.graph_ft_only_batches)
                and _acc is not None and getattr(_acc, "op_ready", False)
                and _cfg.graph_ft_batches and _cfg.save_activations
                # the save op moves a fixed number of rows per step: a step
                # with more FT rows than that keeps the hook path
                and n <= getattr(_acc, "op_rows", n)):
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
        from deltaserve import dprint

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
            _st = (coord.admission_open,
                   getattr(coord, "backward_blocks_admission",
                           coord.pending_backward),
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
        from deltaserve import dprint
        from deltaserve.backward_process import (
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
        ev = self._ft_resume_evt
        if ev is not None and ev.query():
            self._ft_resume_evt = None
            # A sticky throttle pause is released by _ft_pause_decision only.
            if not self._throttle_held:
                coord.gpu_resume_backward()

    def _ft_abort_sentinel(self, coord, relay: bool = True):
        """[forward_interruptible / tier C] The empty output that tells the
        scheduler to roll the FT-only step back. With ``relay`` it carries the
        TP relay fields so an outstanding backward ack is not delayed by the
        aborted step."""
        out = _empty_output()
        out.finetune_aborted = True
        if relay and coord is not None and coord.relay_mode:
            self._ft_fill_relay_fields(out, coord)
        return out
