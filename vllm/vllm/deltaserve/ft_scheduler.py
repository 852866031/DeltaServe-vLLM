# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Scheduler subclass that injects finetuning samples into inference batches.

FT samples are injected directly into the scheduler queues (NOT via add_request,
which has frontend/streaming/connector side effects), ride along a real step as
max_tokens=1 prefill-only requests, and are scrubbed + KV-freed in
update_from_output BEFORE the base loop runs — so they never produce an
EngineCoreOutput and are invisible to the frontend.

Selected via scheduler_config.scheduler_cls when finetune.enable_finetuning is
set (wired in VllmConfig.__post_init__).
"""

import time

from vllm.config import CUDAGraphMode
from vllm.deltaserve import dprint
from vllm.deltaserve.coordinator import get_coordinator
from vllm.deltaserve.estimator import (
    MergedExecutionEstimator,
    StepExecutionTracker,
    StepFeatures,
)
from vllm.deltaserve.ft_injector import FinetuneInjector
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus


class FinetuneScheduler(AsyncScheduler):
    """FT-injecting scheduler. Inherits AsyncScheduler so it carries the
    output-placeholder bookkeeping vLLM's pipelined (async) scheduling needs;
    behaves like the base Scheduler when async is off."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        ft_cfg = self.vllm_config.finetune_config
        capacity = max(1, int(ft_cfg.max_saved_finetuning_tokens))

        # Mirror EngineCore's block hasher so FT requests behave like real ones
        # under prefix caching.
        block_hasher = None
        if self.cache_config.enable_prefix_caching:
            hash_fn = get_hash_fn_by_name(self.cache_config.prefix_caching_hash_algo)
            init_none_hash(hash_fn)
            hbs = kwargs.get("hash_block_size") or self.block_size
            block_hasher = get_request_block_hasher(hbs, hash_fn)

        tokenize = self._build_tokenize()
        self._ft_injector = FinetuneInjector(ft_cfg, tokenize, block_hasher)
        self._block_hasher = block_hasher

        # [Phase 4] Offline-profiling control. When `_profiling_mode` is set the
        # launch driver supplies the exact batch composition, so the automatic
        # FT injection is suppressed. (The warmup/recorded gate lives on the
        # coordinator as `record_timing`, sampled by the runner's deferred timer.)
        self._profiling_mode = False
        self._prof_counter = 0

        # Co-serving coordinator: per-step budget = full capacity
        # (max_saved_finetuning_tokens), so an idle/FT-only step can fill the
        # buffer in a single eager forward instead of fragmenting across several.
        # During co-serving the SLO gate (_slo_ft_budget) caps FT tokens by the
        # predicted step time vs TTFT/TBT once the estimator is warm; this is the
        # cold-start/idle fallback. Tell it the smallest sample length so it
        # knows when no more samples can fit (full).
        self._coord = get_coordinator(capacity=capacity,
                                      per_step_budget=capacity)
        lengths = self._ft_injector.store.sorted_lengths
        if lengths:
            self._coord.min_sample_len = lengths[0]
        # [eval] finetune-throughput log path (written per completed backward).
        self._coord.bwd_log_path = getattr(ft_cfg, "bwd_log_path", None)
        # FT admission master switch: when start_on_launch is False, FT stays
        # closed until POST /start_finetuning flips it (the launch profiler
        # bypasses this, so profiling is unaffected).
        self._coord.ft_started = bool(getattr(ft_cfg, "start_on_launch", True))
        # [forward_interruptible] Register the backward-done hook: when the
        # coordinator's poll_backward observes the backward acked completion,
        # the FT samples that contributed to that backward get their
        # ``trained=True`` bit set (only then — replaces the previous
        # behaviour of marking trained at admit time, which was incorrect
        # under rollback because a sample admitted then rolled back would
        # have been silently marked trained without any backward work).
        self._coord.on_backward_done = (
            lambda samples: self._ft_injector.store.commit_claimed(samples))

        # [Phase 4] SLO execution-time estimator + per-step tracker. The launch
        # profiler seeds these before serving; every served step is recorded and
        # the model is refit on a cadence. The admission gate (Part D) consumes
        # `self._estimator`; until then admission stays the fixed coordinator
        # budget but the estimator is still trained online.
        self._estimator = MergedExecutionEstimator()
        self._tracker = StepExecutionTracker()
        self._stats_csv_path = getattr(
            ft_cfg, "batch_prediction_stats_path", None)
        # SLO targets (seconds) for the admission gate.
        self._ttft_slo = float(ft_cfg.ttft_slo)
        self._max_tbt_slo = float(ft_cfg.max_tbt_slo)
        self._avg_tbt_slo = float(ft_cfg.avg_tbt_slo)
        # [async] Under pipelined (async) scheduling, schedule(N+1) runs while
        # forward(N) is still on the GPU, so a request admitted now isn't
        # prefilled until the in-flight step(s) finish. We add that queue wait to
        # the TTFT prediction, estimated (pessimistically) as the previous
        # scheduled step's predicted time. Zero under sync (the prior forward has
        # already completed by the time schedule() runs).
        self._async = bool(self.scheduler_config.async_scheduling)
        self._last_step_pred_s = 0.0

        dprint(
            f"[ft-sched] FinetuneScheduler active | capacity={capacity} "
            f"per_step_budget={self._coord.per_step_budget} "
            f"min_sample_len={self._coord.min_sample_len} | "
            f"corpus={len(self._ft_injector.store)} samples"
        )

    def _build_tokenize(self):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.vllm_config.model_config.tokenizer)
        return lambda text: tok(text)["input_ids"]

    # ─── SLO-aware FT admission budget ──────────────────────────────────
    def _current_step_features(self) -> tuple[StepFeatures, float | None]:
        """Estimate the inference composition of the step about to run (before
        FT is added): decode part from `self.running` (exact), prefill part from
        the `self.waiting` head within the token budget. Returns (features,
        earliest_waiting_arrival)."""
        b_d = 0
        k = 0.0
        for req in self.running:
            if getattr(req, "is_finetuning", False):
                continue
            b_d += 1
            k += req.num_computed_tokens
        budget = self.max_num_scheduled_tokens
        remaining = max(0, budget - b_d)  # decode consumes ~1 token/req
        t_in = 0.0
        prefill_lens: list[int] = []
        earliest_arrival: float | None = None
        for req in self.waiting:
            if earliest_arrival is None:
                earliest_arrival = req.arrival_time
            if remaining <= 0:
                break
            n = min(req.num_tokens - req.num_computed_tokens, remaining)
            if n <= 0:
                continue
            prefill_lens.append(int(n))
            t_in += n
            remaining -= n
        feats = StepFeatures(t_in=t_in, p=len(prefill_lens), t_ft=0.0,
                             b_d=b_d, k=k, prefill_lens=prefill_lens or None)
        return feats, earliest_arrival

    def _slo_ft_budget(self, feats: StepFeatures,
                       earliest_arrival: float | None) -> int:
        """FT token budget for this step: the min of the SLO-derived headroom
        and the coordinator's buffer-space cap. Falls back to the fixed
        coordinator budget while the estimator is still cold. `feats` is the
        upcoming step's inference composition (before FT is added)."""
        coord_budget = self._coord.next_ft_budget()
        if coord_budget <= 0 or not self._estimator.is_ready:
            return coord_budget
        # Baseline step time WITHOUT the new FT (eager: adding FT forces eager).
        t_current = self._estimator.predict(feats, will_use_graph=False)
        now = time.time()
        # Under async, the step we're building queues behind the in-flight
        # forward, so the waiting request can't start prefilling until that
        # finishes. Charge that wait against the TTFT budget (pessimistic: the
        # full predicted time of the previously scheduled step).
        queue_wait = self._last_step_pred_s if self._async else 0.0
        budgets_s: list[float] = []
        if earliest_arrival is not None and feats.t_in > 0:
            # TTFT: now + (in-flight queue wait) + (this step) must beat the
            # deadline for the earliest waiting request.
            budgets_s.append(
                earliest_arrival + 0.9 * self._ttft_slo
                - now - queue_wait - t_current)
        if feats.b_d > 0:
            # max-TBT: this step's time is the inter-token interval for decodes.
            budgets_s.append(self._max_tbt_slo - t_current)
        if not budgets_s:
            # No inference constraint this step (idle but for FT) → buffer cap.
            return coord_budget
        slo_x = min(self._estimator.max_next_ft_tokens(b) for b in budgets_s)
        return max(0, min(coord_budget, slo_x))

    # ─── SLO estimator: feature extraction + graph predicate ────────────
    def _features_from_output(
        self, output: SchedulerOutput
    ) -> tuple[StepFeatures, set[int]]:
        """Compute the merged-estimator features for the batch this step will
        run, from the realized SchedulerOutput. Returns (features, lora_ids).

        Call from `schedule()` (after `super().schedule()`) while every request
        is still in `self.requests` — scheduled FT reqs are scrubbed later in
        `update_from_output`.
        """
        t_in = 0.0
        prefill_lens: list[int] = []
        t_ft = 0.0
        b_d = 0
        k = 0.0
        lora_ids: set[int] = set()
        for req_id, n in output.num_scheduled_tokens.items():
            req = self.requests.get(req_id)
            if req is None:
                continue
            if req.lora_request and req.lora_request.lora_int_id > 0:
                lora_ids.add(req.lora_request.lora_int_id)
            # Prefill iff this step is still consuming prompt tokens; otherwise
            # the request is generating (decode). FT reqs are prefill-only.
            if req.num_computed_tokens < req.num_prompt_tokens:
                prefill_lens.append(int(n))
                t_in += n
                if getattr(req, "is_finetuning", False):
                    t_ft += n
            else:
                b_d += 1
                k += req.num_computed_tokens
        feats = StepFeatures(
            t_in=t_in, p=len(prefill_lens), t_ft=t_ft, b_d=b_d, k=k,
            prefill_lens=prefill_lens or None)
        return feats, lora_ids

    def _will_use_graph(self, feats: StepFeatures, lora_ids: set[int],
                        total_tokens: int) -> bool:
        """Query vLLM's real CUDAGraph dispatcher (shared via the coordinator on
        single-GPU). Co-serve steps force eager; no dispatcher ⇒ eager."""
        if feats.has_ft:
            return False
        disp = self._coord.cudagraph_dispatcher
        if disp is None:
            return False
        has_lora = self.lora_config is not None and len(lora_ids) > 0
        uniform_decode = feats.t_in == 0 and feats.b_d > 0
        mode, _ = disp.dispatch(
            num_tokens=int(total_tokens),
            uniform_decode=uniform_decode,
            has_lora=has_lora,
            num_active_loras=len(lora_ids))
        return mode != CUDAGraphMode.NONE

    def has_requests(self) -> bool:
        """Keep the engine stepping while finetuning is live, even with no
        inference in flight.

        vLLM's busy loop blocks on the input queue when `has_requests()` is
        False (`EngineCore.has_work`), so without this the co-serving loop would
        stall whenever inference goes idle: FT-only batches would stop and the
        backward-done signal wouldn't be polled (`poll_backward` runs in
        `schedule()`) until the next client request arrived. Returning True while
        FT has work keeps FT-only batches flowing during idle and reopens
        admission promptly after a backward. Empty steps (admission closed during
        a backward) are cheap — `execute_model` early-returns on 0 tokens."""
        if super().has_requests():
            return True
        if self._profiling_mode:
            return False  # profiler drives steps explicitly
        if not self.vllm_config.finetune_config.enable_finetuning:
            return False
        coord = self._coord
        if not coord.ft_started:
            return False  # FT held closed (awaiting POST /start_finetuning)
        if coord.pending_backward:
            return True  # poll the in-flight backward to completion
        if coord.epoch_flush_pending and (
                coord.fill_count + coord.reserved_fill) > 0:
            return True  # flush flag set at inject → step so the backward fires  # epoch tail awaiting flush
        # Otherwise step for FT only if a sample can ACTUALLY be admitted this
        # step (buffer space ≥ one sample). A partially-full buffer that can't
        # grow and isn't flush-pending is not work — returning True there would
        # spin the engine at 100% CPU (starving request intake) until inference
        # idles enough to let FT resume. `next_ft_budget` is buffer-space-based
        # (the SLO gate may still admit 0 under load, but then inference itself
        # keeps the engine busy via super().has_requests()).
        if coord.next_ft_budget() >= max(1, coord.min_sample_len):
            store = self._ft_injector.store
            return store.has_next() or store.current_epoch < store.total_epochs
        return False

    def would_step_be_ft_only(self) -> bool:
        """[forward_interruptible / tier A] True iff the next schedule() call
        would produce an FT-only batch — no inference requests in
        ``self.running`` (excluding in-flight FT requests awaiting retire) and
        none in ``self.waiting`` (FT samples aren't injected until inside
        ``schedule()``, so the waiting queue at this point only holds
        inference). Engine main loop consults this before its grace poll to
        decide whether to spend the wait window.

        Cheap: walks ``self.running`` once with an attribute check. Called at
        most once per busy-loop iteration."""
        if self.waiting:
            return False
        for req in self.running:
            if not getattr(req, "is_finetuning", False):
                return False
        return True

    def write_estimator_stats(self) -> None:
        """Dump predicted-vs-actual step stats (called on FT exit)."""
        self._tracker.write_prediction_stats_csv(self._stats_csv_path)

    def shutdown(self) -> None:
        # [Phase 4] Dump the predicted-vs-actual estimator stats before teardown.
        try:
            if self._stats_csv_path:
                self.write_estimator_stats()
        except Exception as e:
            dprint(f"[ft-sched] estimator stats dump failed: {e}")
        super().shutdown()

    # ─── Offline-profiling helpers (used by EngineCore.profile_execution_model) ──
    def new_profiling_request(self, prompt_ids, max_tokens: int, is_ft: bool):
        """Build a synthetic profiling Request (not enqueued)."""
        from vllm.sampling_params import SamplingParams
        from vllm.v1.request import Request
        self._prof_counter += 1
        tag = "ft" if is_ft else "inf"
        req = Request(
            request_id=f"__prof_{tag}_{self._prof_counter}",
            prompt_token_ids=list(prompt_ids),
            sampling_params=SamplingParams(max_tokens=max_tokens,
                                           temperature=0.0),
            pooling_params=None,
            lora_request=self._ft_injector.ft_lora if is_ft else None,
            block_hasher=self._block_hasher,
        )
        if is_ft:
            req.is_finetuning = True
        return req

    def enqueue_profiling_request(self, req) -> None:
        self._enqueue_waiting_request(req)
        self.requests[req.request_id] = req

    def purge_profiling_requests(self) -> None:
        """Drop any leftover synthetic profiling requests + free their KV, so the
        scheduler is clean before live serving begins."""
        for req_id in [r for r in self.requests if r.startswith("__prof_")]:
            req = self.requests.get(req_id)
            if req is None:
                continue
            req.status = RequestStatus.FINISHED_LENGTH_CAPPED
            if req in self.running:
                self.running.remove(req)
            self._free_blocks(req)
        # Drop any still queued in waiting.
        leftover = [r for r in self.waiting
                    if r.request_id.startswith("__prof_")]
        if leftover:
            self.waiting.remove_requests(leftover)

    def schedule(self) -> SchedulerOutput:
        # [Phase 4] Drain step durations the runner finished timing (deferred
        # CUDA-event ring) into the tracker. Pairs each step's features (stamped
        # at schedule) with its measured duration (read a few steps later).
        for feats, dur, was_graph, predicted in (
                self._coord.drain_completed_samples()):
            if dur > 0:
                self._tracker.add(feats, dur, predicted=predicted,
                                  was_graph=was_graph)

        # Live refit of the SLO estimator every REFIT_EVERY steps. Skipped during
        # offline profiling (one fit at the end of the pass).
        if not self._profiling_mode and self._tracker.check_refit():
            self._estimator.data_fit(self._tracker)

        # Reopen FT admission if the backward process finished consuming a buffer.
        self._coord.poll_backward()
        # Terminal/partial epoch flush fallback (guarded by reserved==0 inside,
        # so it can't race in-flight saves): fires when no further FT forward
        # will carry the flush.
        self._coord.try_epoch_flush()

        # [forward_interruptible] Snapshot admission state before any FT
        # injection / note_injection / reserve mutates it. Stashed on the
        # output so _rollback_ft_step can restore it verbatim if this batch
        # turns out to be FT-only and a pre-emption fires before dispatch.
        _admit_snap = self._coord.snapshot_admission()

        # [Phase 4] FT rides along PREFILL steps only. Skip FT on decode-only
        # steps (decode tokens present, no prefill tokens): adding FT prefill
        # there would spike the decoders' TBT with no prefill to amortize
        # against. Computed from the upcoming step's inference composition.
        # SLO-aware admission then caps the FT tokens so the predicted step time
        # keeps inference TTFT/TBT within target (fixed coordinator budget while
        # the estimator is cold).
        feats, earliest_arrival = self._current_step_features()
        decode_only = feats.b_d > 0 and feats.t_in == 0
        budget = 0 if decode_only else self._slo_ft_budget(
            feats, earliest_arrival)
        # Optional cap: on prefill-carrying steps, limit FT tokens to
        # `prefill_tokens * factor` (config ft_tokens_admission_constrain_factor)
        # so FT prefill can't dwarf the inference prefill it rides with. factor
        # == -1 disables it; prefill-free (idle/FT-only) steps are unaffected.
        _factor = self.vllm_config.finetune_config.\
            ft_tokens_admission_constrain_factor
        if _factor != -1 and feats.t_in > 0:
            budget = min(budget, int(feats.t_in * _factor))
        admitted_now = 0
        if not self._profiling_mode and budget > 0:
            reqs = self._ft_injector.next_ft_requests(budget)
            for req in reqs:
                self._enqueue_waiting_request(req)
                self.requests[req.request_id] = req
            admitted_now = sum(r.num_prompt_tokens for r in reqs)

        # After injecting, hand the coordinator the smallest sample that could
        # still be added (peek-next, no mark). It raises the flush flag when the
        # buffer can't grow — epoch drained (None) or the next sample won't fit
        # the free space (would overflow) — so the partial buffer gets trained
        # instead of wedging at a near-full level during idle (e.g. stuck at
        # 208/256). The backward trigger consumes the flag. ``admitted_now``
        # makes the close check use POST-admit free space (the reserve() call
        # below hasn't run yet, so space_remaining alone reads the prior step's
        # reservations only). Under async this closes admission at the exact
        # schedule where the buffer becomes "full enough that the next sample
        # won't fit" rather than one schedule call later.
        _nxt = self._ft_injector.store.pop_next()
        self._coord.note_injection(
            _nxt.input_len if _nxt is not None else None,
            admitted_now=admitted_now)

        # Track the FT corpus epoch so the backward can StepLR per epoch
        # (next_ft_requests advances it via the store's advance_epoch).
        self._coord.current_epoch = self._ft_injector.store.current_epoch

        output = super().schedule()

        # Partition FT requests. Under async scheduling, schedule(N+1) runs
        # before update_from_output(N), so PRIOR-step FT requests are still in
        # self.running/self.requests here (skipped by the base async early-skip
        # because they hold an output placeholder) awaiting their own step's
        # retire. We must NOT touch those — only the FT requests freshly injected
        # THIS step (placeholders == 0) are partitioned: scheduled ones go to
        # finetune_req_ids (retired in this step's update_from_output); ones that
        # didn't fit are cleaned up so they don't accumulate. (Deleting an
        # in-flight prior-step req here would orphan it in self.running — its own
        # update_from_output could no longer find it to free → leak → the running
        # batch fills to max_num_seqs and the engine stalls.)
        scheduled_ft: set[str] = set()
        dropped_ft_samples: list = []   # released back to store below
        for req_id, req in list(self.requests.items()):
            if not req.is_finetuning:
                continue
            if req_id in output.num_scheduled_tokens:
                scheduled_ft.add(req_id)
            elif req.num_output_placeholders == 0:
                # Fresh inject that didn't schedule this step — safe to drop.
                self.waiting.remove_requests([req])
                del self.requests[req_id]
                _s = getattr(req, "_ft_sample", None)
                if _s is not None:
                    dropped_ft_samples.append(_s)
            # else: in-flight from a prior step — leave for its own retire.
        output.finetune_req_ids = scheduled_ft
        # [forward_interruptible] Release samples we claimed at admit but
        # didn't actually schedule — return them to the selectable pool so
        # the next step can pick them up. (Today this path is rare; before
        # the store-API split these samples were silently lost.)
        if dropped_ft_samples:
            self._ft_injector.store.release_claimed(dropped_ft_samples)

        # [async] Reserve the activation-buffer rows for the FT samples that
        # ACTUALLY scheduled this step, and stash their disjoint write offset for
        # the runner. Reserved at schedule (not committed at forward) so that
        # under pipelining two in-flight steps never overlap their writes or
        # overflow the buffer. record_capture commits + reconciles post-forward.
        # The samples are appended to the coordinator's ``buffer_samples`` so
        # the backward-done hook can commit-train exactly the right samples.
        if scheduled_ft:
            ft_tokens = sum(output.num_scheduled_tokens.get(rid, 0)
                            for rid in scheduled_ft)
            ft_samples = [getattr(self.requests[rid], "_ft_sample", None)
                          for rid in scheduled_ft]
            ft_samples = [s for s in ft_samples if s is not None]
            if ft_tokens > 0:
                output._ft_write_offset = self._coord.reserve(
                    ft_tokens, samples=ft_samples)
            output._ft_samples = ft_samples
            output._ft_tokens_reserved = ft_tokens
        # Stash the admission snapshot on the output so _rollback_ft_step
        # can restore it cleanly if this batch is pre-empted before dispatch.
        output._ft_admit_snapshot = _admit_snap

        # [Phase 4] Stamp the regime + predicted duration for THIS step now,
        # while all requests are still present and the dispatcher reflects the
        # state the runner will execute in. Paired with the measured duration in
        # `update_from_output` to feed the tracker (which can't reclassify the
        # sample later — was_graph is frozen here).
        if output.total_num_scheduled_tokens > 0:
            feats, lora_ids = self._features_from_output(output)
            was_graph = self._will_use_graph(
                feats, lora_ids, output.total_num_scheduled_tokens)
            predicted = (self._estimator.predict(feats, was_graph)
                         if self._estimator.is_ready else None)
            output._ft_step_features = feats
            output._ft_step_was_graph = was_graph
            output._ft_step_predicted = predicted
            # Remember this step's predicted time as the in-flight queue-wait
            # estimate for the next schedule() (async TTFT term).
            if predicted is not None:
                self._last_step_pred_s = predicted
        # Diagnostics: how full the scheduler is when this batch was built.
        # `running` includes in-flight FT reqs awaiting retire; `running_inf`
        # is the inference-only subset (FT reqs are scrubbed each step).
        output._ft_running = len(self.running)
        output._ft_running_inf = sum(
            1 for r in self.running if not getattr(r, "is_finetuning", False))
        output._ft_waiting = len(self.waiting)
        return output

    def _rollback_ft_step(self, scheduler_output: SchedulerOutput) -> int:
        """[forward_interruptible / tiers B + C] Undo all FT-side state
        changes for the FT requests in ``scheduler_output``. Returns the
        number of FT requests that were rolled back.

        Used in two places:
          - **Tier B** (engine, post-schedule pre-execute): a late inference
            arrival landed between ``schedule()`` returning and
            ``execute_model()`` being called; we abandon the FT-only batch
            and re-schedule with the inference request now in waiting.
          - **Tier C** (runner, mid-forward abort): the activation hook
            raised on a late ADD; the runner caught it and calls this from
            ``execute_model``'s exception handler.

        Steps mirror the FT retire path in ``update_from_output`` but also
        return samples to the store, undo the coordinator's reservation,
        and restore admission flags so the next ``schedule()`` sees the
        same state as if this step never happened. Idempotent: calling
        twice on the same scheduler_output is a no-op the second time.
        """
        ft_ids = getattr(scheduler_output, "finetune_req_ids", set())
        if not ft_ids:
            return 0
        # Free KV + drop from scheduler state — mirrors the FT retire path
        # in update_from_output. Note: _free_blocks itself removes the
        # request from self.requests (scheduler.py:1863), so we do NOT
        # del self.requests[req_id] again afterwards (would KeyError).
        for req_id in list(ft_ids):
            req = self.requests.get(req_id)
            if req is None:
                continue
            req.status = RequestStatus.FINISHED_LENGTH_CAPPED
            if req in self.running:
                self.running.remove(req)
            self._free_blocks(req)
        # Restore the activation-buffer reservation and the in-flight
        # samples list on the coordinator.
        samples = getattr(scheduler_output, "_ft_samples", None) or []
        n_reserved = getattr(scheduler_output, "_ft_tokens_reserved", 0)
        if n_reserved or samples:
            self._coord.release_reserve(n_reserved, samples=samples)
        # Return claimed samples to the selectable pool so the next schedule
        # can re-admit them (the engine's re-schedule call after rollback,
        # or a later step after admission has been re-evaluated).
        if samples:
            self._ft_injector.store.release_claimed(samples)
        # Restore admission flags so note_injection's effect from this
        # schedule() is undone (admission_open / epoch_flush_pending).
        snap = getattr(scheduler_output, "_ft_admit_snapshot", None)
        if snap is not None:
            self._coord.restore_admission(snap)
        # Clear the tier-C abort event so the next FT-only batch starts with
        # a clean signal — otherwise a stale "set" from this cycle would
        # immediately bail every subsequent FT-only batch's entry check
        # (pipeline-depth contamination). Safe to clear unconditionally:
        # the late ADD that originally tripped this event is already on
        # input_queue and will get a fresh trigger on the NEXT FT-only batch
        # via the input thread's per-ADD set.
        self._coord.ft_abort_event.clear()
        # Mark the scheduler_output as rolled back so any double-call
        # (e.g. C then accidental B) is a no-op.
        scheduler_output.finetune_req_ids = set()
        scheduler_output._ft_samples = []
        scheduler_output._ft_tokens_reserved = 0
        return len(ft_ids)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict:
        # Retire FT requests BEFORE the base loop: free their KV the same step
        # WITHOUT touching finished_req_ids (so the frontend never sees them).
        # Once removed from self.requests, the base loop's `request is None`
        # check skips them — no EngineCoreOutput is built.
        for req_id in scheduler_output.finetune_req_ids:
            req = self.requests.get(req_id)
            if req is None:
                continue
            req.status = RequestStatus.FINISHED_LENGTH_CAPPED
            if req in self.running:
                self.running.remove(req)
            self._free_blocks(req)

        # NOTE: estimator samples are recorded in schedule() by draining the
        # coordinator's completed-sample queue — the runner times each step with
        # a deferred CUDA-event ring, so the duration isn't known here (it lands
        # a few steps later, off the hot path).
        return super().update_from_output(scheduler_output, model_runner_output)
