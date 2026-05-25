# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Co-serving coordinator: the FT activation-buffer fill state + admission gate.

Single source of truth (one instance per EngineCore process, shared by the
FinetuneScheduler and the GPUModelRunner since single-GPU runs both in-process):

  - ``fill_count`` is the activation-buffer write offset / fill level; the
    buffer holds ``capacity = max_saved_finetuning_tokens`` token rows.
  - The scheduler asks ``next_ft_budget()`` for how many FT tokens it may inject
    this step (up to ``per_step_budget = capacity``, capped by free space,
    and 0 while a backward pass is pending).
  - The runner writes captured FT activations at ``current_offset()`` and calls
    ``record_capture(n)``; when the buffer fills, that signals the backward
    process and CLOSES admission.
  - ``poll_backward()`` (called each step) reopens admission once the backward
    process reports it has consumed + cleaned the buffer.

All decisions are printed via ``dprint`` (green, main process).
"""

import csv
import datetime
import os
import threading
import time

import torch

from vllm.deltaserve import dprint


class FTAborted(Exception):
    """[forward_interruptible / tier C] Sentinel raised from an FT activation
    hook when ``FinetuneCoordinator.ft_abort_event`` is set mid-forward
    (signal: a late inference ADD arrived while an FT-only batch was on the
    GPU). Caught at ``execute_model``'s outermost boundary, which then runs
    ``_rollback_ft_step`` and returns an empty ModelRunnerOutput. Narrowly
    typed so nothing else in PyTorch swallows it."""
    pass

_COORDINATOR: "FinetuneCoordinator | None" = None


def get_coordinator(capacity: int | None = None,
                    per_step_budget: int | None = None) -> "FinetuneCoordinator | None":
    """Process-wide singleton. First call (from the scheduler) creates it."""
    global _COORDINATOR
    if _COORDINATOR is None and capacity is not None:
        _COORDINATOR = FinetuneCoordinator(capacity, per_step_budget)
    return _COORDINATOR


class FinetuneCoordinator:
    def __init__(self, capacity: int, per_step_budget: int | None = None) -> None:
        self.capacity = int(capacity)
        self.per_step_budget = int(
            per_step_budget if per_step_budget is not None else capacity
        )
        # COMMITTED FT rows — activations actually saved into the buffer
        # (incremented post-forward in record_capture).
        self.fill_count = 0
        # [async] RESERVED-but-not-yet-saved FT rows — admitted into a scheduled
        # step whose forward hasn't run yet. Under async scheduling, schedule(N+1)
        # runs before record_capture(N), so admission must account for in-flight
        # reservations (committed + reserved ≤ capacity) to avoid overflow, and
        # each in-flight step gets a disjoint write offset (committed+reserved).
        # In sync mode this is reserved at schedule and reconciled to 0 in the
        # same step's record_capture, so it behaves exactly as before.
        self.reserved_fill = 0
        # Per-FT-sample token counts accumulated alongside fill_count (in
        # buffer-write order); forwarded to the backward process so it can split
        # the flat buffers into samples and shift-by-1 for next-token targets.
        self.sample_lens: list[int] = []
        # Parallel to ``sample_lens``: the source FinetuningSample objects in
        # buffer-write order. Used to call ``store.commit_claimed`` when the
        # backward acks completion (via ``on_backward_done`` below), so the
        # "trained=True" bit is only set after a backward has actually
        # processed the sample's activations (fixes the prior behaviour of
        # marking samples trained at admit time). [forward_interruptible]
        self.buffer_samples: list = []
        # Hook invoked from poll_backward on success with the FinetuningSample
        # list that was in the buffer. Set by the scheduler at init; defaults
        # to no-op so non-FT paths / tests are unaffected.
        self.on_backward_done = None
        self.admission_open = True
        self.pending_backward = False
        # [async] An epoch boundary was reached (set by the scheduler when the
        # corpus epoch drains). The flush fires from record_capture once the
        # buffer's in-flight saves settle (reserved_fill == 0), so it can't race
        # in-flight accumulations; a scheduler-side fallback covers the case
        # where no further FT forward follows (terminal partial buffer).
        self.epoch_flush_pending = False
        self.min_sample_len = 1            # set by the scheduler from the store
        self.backward_process = None       # set by the worker
        self.backward_sleep_s = 2.0        # set by the worker from config
        # Current FT corpus epoch (set each step by the scheduler from the store);
        # forwarded to the backward process so it can StepLR per epoch.
        self.current_epoch = 0
        self._cycle = 0
        # [Phase 4] Wall-clock (GPU-synchronized) duration of the most recently
        # executed step, in seconds. Written by the model runner after the
        # forward (CUDA-event timed); read by the scheduler to feed the
        # SLO execution-time estimator. None until the first timed step.
        self.last_step_s: float | None = None
        # [Phase 4] vLLM's real CUDAGraph dispatcher (set by the model runner
        # when it shares a process with the scheduler, i.e. single-GPU). The
        # scheduler queries it to know whether a prospective batch will run as
        # a graph — we connect to vLLM's own decision rather than mirroring it.
        # None when the runner is in a different process (TP>1); the scheduler
        # then falls back to the eager regime.
        self.cudagraph_dispatcher = None
        # [Phase 4] True while the launch-time profiler is driving synthetic
        # batches. Silences per-step batch logging (a progress bar is shown
        # instead) and suppresses online refit.
        self.profiling = False
        # [Phase 4] Completed (features, duration, was_graph, predicted) samples
        # for the SLO estimator. The runner times each step with a deferred CUDA
        # event ring (read RING steps later, off the hot path) and PUSHES the
        # finished sample here; the scheduler DRAINS into its tracker. Decoupled
        # because the duration isn't known at update_from_output time anymore.
        # `record_timing` mirrors the scheduler's record gate (off during the
        # profiler's warmup pass) and is sampled by the runner at forward time.
        self._completed_samples: list = []
        self.record_timing = True
        # [eval] Optional finetune-throughput log: one row per completed
        # backward (set from finetune.bwd_log_path by the scheduler). The eval
        # harness reads it for the FT-throughput plot.
        self.bwd_log_path: str | None = None
        self._bwd_idx = 0
        self._bwd_total_tokens = 0
        # FT admission master switch (separate from the buffer/backward gates).
        # When False, no FT is injected regardless of buffer space — flipped on
        # by POST /start_finetuning. Set from finetune.start_on_launch by the
        # scheduler. The launch profiler bypasses this (it injects FT directly),
        # so profiling is unaffected.
        self.ft_started = True
        # Wall-clock (monotonic) instant FT admission was opened by HTTP, used
        # as t=0 for the [batch] log's elapsed-since-start timer. None until
        # start_finetuning() fires.
        self.ft_start_time: float | None = None
        # [diag] count of real inference requests admitted to the engine (bumped
        # by EngineCore on each ADD). Once it passes a threshold the runner logs
        # EVERY batch (incl. decode-only) so the stall window is fully visible.
        self.inf_req_count = 0
        # CUDA event recorded by the runner right after the latest step's FT
        # activation writes. _trigger_backward waits on it (instead of a
        # full-device synchronize) to guarantee the capture copies are visible
        # to the backward process before it reads the shared buffer.
        self.capture_done_evt = None
        # [forward_interruptible / tier C] Set by the input-socket thread on
        # every ADD that lands while ``ft_only_in_flight`` is True. Checked
        # by the activation hooks at every layer boundary; when set, the
        # hook raises ``FTAborted`` and ``execute_model`` rolls back.
        # threading.Event so the cross-thread set/check is GIL-safe and the
        # cost when not set is one C-level bool load.
        self.ft_abort_event = threading.Event()
        # [forward_interruptible / tier C] True only while an FT-only forward
        # is on the GPU. Cheap gate for the input thread so the abort-event
        # set is skipped entirely when no FT-only batch is running. Set
        # before kernel dispatch in execute_model and cleared in its
        # finally block.
        self.ft_only_in_flight = False

    def push_sample(self, features, duration, was_graph, predicted) -> None:
        self._completed_samples.append((features, duration, was_graph, predicted))

    def drain_completed_samples(self) -> list:
        out = self._completed_samples
        self._completed_samples = []
        return out

    def space_remaining(self) -> int:
        # Free rows = capacity minus committed AND reserved-in-flight.
        return max(0, self.capacity - self.fill_count - self.reserved_fill)

    def next_ft_budget(self) -> int:
        """How many FT tokens the scheduler may admit this step."""
        if not self.ft_started or not self.admission_open or self.pending_backward:
            return 0
        return min(self.per_step_budget, self.space_remaining())

    def start_finetuning(self) -> None:
        """Open FT admission (POST /start_finetuning). Idempotent."""
        if self.ft_start_time is None:
            self.ft_start_time = time.monotonic()
        if not self.ft_started:
            self.ft_started = True
            dprint("[coord] finetuning STARTED (admission opened by HTTP)")

    def current_offset(self) -> int:
        # Next free write position = committed + reserved-in-flight. (Fallback;
        # the scheduler reserves a per-step offset via reserve().)
        return self.fill_count + self.reserved_fill

    def reserve(self, n: int, samples: list | None = None) -> int:
        """Reserve n FT rows for a scheduled step and return the buffer write
        offset they own. Called by the scheduler AFTER the FT partition (only the
        rows that actually scheduled). Committed by record_capture post-forward.

        ``samples`` is the parallel list of FinetuningSamples (one per
        scheduled FT request) — appended to ``buffer_samples`` so the
        backward-done hook can later commit-train exactly the samples whose
        activations contributed to that backward."""
        offset = self.fill_count + self.reserved_fill
        self.reserved_fill += int(n)
        if samples:
            self.buffer_samples.extend(samples)
        return offset

    def release_reserve(self, n: int, samples: list | None = None) -> None:
        """[forward_interruptible] Symmetric counterpart to ``reserve``: undo
        the bump when the FT scheduling for a step is rolled back (Phase B
        post-schedule pre-empt) or its forward is aborted (Phase C mid-
        forward abort). Clamps at 0 so a stray over-release can't drive
        ``reserved_fill`` negative. Also drops the matching samples from
        ``buffer_samples`` (by request_id) so the next backward-commit
        doesn't try to commit-train samples that never had their activations
        actually saved."""
        self.reserved_fill = max(0, self.reserved_fill - int(n))
        if samples and self.buffer_samples:
            drop = {s.request_id for s in samples}
            self.buffer_samples = [
                s for s in self.buffer_samples if s.request_id not in drop]

    def snapshot_admission(self) -> tuple:
        """[forward_interruptible] Capture the flags ``note_injection`` may
        flip during this ``schedule()`` call so ``_rollback_ft_step`` can
        restore them. Notably does NOT include ``reserved_fill`` — that's
        undone via ``release_reserve(n)``, and restoring a snapshotted
        ``reserved_fill`` would clobber any intervening commit (e.g. a prior
        in-flight FT batch's ``record_capture`` between the snapshot of N+1
        and the rollback of N+1 under pipelined stepping). Only the
        scheduler-only state goes here."""
        return (self.admission_open, self.epoch_flush_pending)

    def restore_admission(self, snap: tuple) -> None:
        """[forward_interruptible] Inverse of ``snapshot_admission``. Only
        restores ``admission_open`` + ``epoch_flush_pending``; reserved_fill
        is the responsibility of ``release_reserve`` (see snapshot docstring)."""
        self.admission_open, self.epoch_flush_pending = snap

    def request_epoch_flush(self) -> None:
        """Scheduler signals the corpus epoch has drained; the flush fires once
        in-flight saves settle (reserved_fill == 0)."""
        self.epoch_flush_pending = True

    def note_injection(self, next_sample_len: int | None,
                       admitted_now: int = 0) -> None:
        """Called by the scheduler right after each FT injection, with the
        smallest sample that could still be added (peek-next, no mark; None if
        the epoch is drained). Raises the flush flag (``epoch_flush_pending``)
        when the activation buffer can't grow any further — the epoch finished
        OR the next sample won't fit the free space (would overflow) — so the
        partial buffer gets trained instead of wedging at a near-full level
        during idle. The backward trigger (``record_capture`` / ``try_epoch_flush``)
        consumes the flag and ``_trigger_backward`` unsets it. No-op on an empty
        buffer (nothing to train) or while a backward is already pending.

        ``admitted_now`` is the FT token count just admitted in this same
        schedule call, BEFORE ``reserve()`` runs. Under async scheduling we want
        the close check to use POST-admit free space (``space_remaining`` -
        ``admitted_now``); otherwise the admission_open flag is one schedule()
        call late under pipelining (the next schedule sees correct budget via
        ``reserved_fill`` so this isn't a correctness bug — just lets the engine
        skip an empty FT-only step and reach idle/inference sooner)."""
        if self.pending_backward:
            return
        if (self.fill_count + self.reserved_fill + admitted_now) <= 0:
            return
        effective_space = self.space_remaining() - admitted_now
        if next_sample_len is None or next_sample_len > effective_space:
            self.epoch_flush_pending = True
            self.admission_open = False

    def try_epoch_flush(self) -> None:
        """Scheduler-side fallback (top of schedule, guarded by reserved==0 so it
        can't race in-flight saves): fire the epoch flush when no further FT
        forward will carry it (terminal partial buffer).

        Note: does NOT require admission_open. The epoch-boundary path closes
        admission ("hold the next epoch until the tail flushes"), and this is the
        trigger that flushes that tail — gating on admission_open would deadlock
        (admission closed waiting for a flush that can't fire because admission is
        closed). `not pending_backward` already prevents double-firing."""
        if (self.epoch_flush_pending and self.reserved_fill == 0
                and self.fill_count > 0 and not self.pending_backward):
            self._trigger_backward(reason="epoch-end flush")

    def record_capture(self, n: int, sample_lens: list[int] | None = None) -> None:
        """Runner reports n FT token rows just SAVED at this step's reserved
        offset. Commits them (and reconciles the reservation). Both backward
        triggers fire HERE, post-forward — after the activations are saved and
        once no in-flight saves remain (reserved_fill == 0) — so neither can race
        an in-flight accumulation:
          * buffer full → train the slice;
          * epoch boundary on a non-full buffer → flush the epoch's trailing
            samples (so they get trained and StepLR steps at the boundary).
        """
        if n <= 0:
            return
        self.fill_count += n
        self.reserved_fill = max(0, self.reserved_fill - n)
        if sample_lens:
            self.sample_lens.extend(int(x) for x in sample_lens)
        if self.pending_backward or self.reserved_fill > 0:
            return  # wait until all in-flight saves for this buffer are committed
        if self.space_remaining() < self.min_sample_len:
            self._trigger_backward()
        elif self.epoch_flush_pending and self.fill_count > 0:
            self._trigger_backward(reason="epoch-end flush")

    def _trigger_backward(self, reason: str = "buffer FULL") -> None:
        # During offline profiling the backward is detached and the buffer is
        # reset between shapes; a full-cap co-serve shape would otherwise fire
        # (and print) a spurious backward signal. Profiling only times forwards.
        if self.profiling:
            return
        self.admission_open = False
        self.pending_backward = True
        self.epoch_flush_pending = False
        self._cycle += 1
        # dprint(
        #     f"[coord] {reason} ({self.fill_count}/{self.capacity}) -> "
        #     f"signal backward (cycle {self._cycle}); FT admission CLOSED"
        # )
        if self.backward_process is not None:
            # Make sure the capture writes are visible to the backward process
            # before it reads/cleans the shared buffer. Wait only on the FT
            # capture-completion event (recorded by the runner right after the
            # activation copies) rather than a full-device synchronize, so the
            # engine loop isn't stalled on unrelated in-flight GPU work. Fall
            # back to a device sync if no event has been recorded yet (e.g. an
            # epoch flush before any capture, or save_activations disabled).
            if self.capture_done_evt is not None:
                self.capture_done_evt.synchronize()
            else:
                torch.cuda.synchronize()
            self.backward_process.notify_buffer_full(
                self.fill_count, self.backward_sleep_s, list(self.sample_lens),
                epoch=self.current_epoch)

    def gpu_pause_backward(self) -> None:
        """[Phase 5] Ask the backward child to yield the GPU (it pauses at its
        next layer boundary). Called by the model runner before an inference
        prefill forward. No-op when no backward child (e.g. profiling)."""
        if self.backward_process is not None:
            self.backward_process.set_pause(True)

    def gpu_resume_backward(self) -> None:
        """[Phase 5] Return the GPU to the backward child after the inference
        prefill forward completes."""
        if self.backward_process is not None:
            self.backward_process.set_pause(False)

    def poll_backward(self) -> None:
        """Non-blocking: reopen admission once the backward pass has finished.

        On completion, invokes ``on_backward_done(buffer_samples)`` (set by
        the scheduler at init) so the store can call ``commit_claimed`` on
        exactly the samples whose activations were just trained on. Done
        BEFORE clearing ``buffer_samples`` so the hook sees the full list.
        """
        if not self.pending_backward or self.backward_process is None:
            return
        resp = self.backward_process.poll_response()
        if resp is not None:
            if (self.bwd_log_path and isinstance(resp, dict)
                    and resp.get("event") == "activations_processed"):
                self._write_bwd_log_row(
                    int(resp.get("n", 0)), resp.get("loss"))
            # Commit the in-buffer claimed samples → trained=True (one-shot
            # snapshot so the hook can mutate state freely).
            if self.on_backward_done is not None and self.buffer_samples:
                trained_now = list(self.buffer_samples)
                try:
                    self.on_backward_done(trained_now)
                except Exception as e:
                    dprint(f"[coord] on_backward_done hook failed: {e}")
            self.fill_count = 0
            self.reserved_fill = 0
            self.sample_lens = []
            self.buffer_samples = []
            self.pending_backward = False
            self.admission_open = True

    def _write_bwd_log_row(self, n: int, loss) -> None:
        """Append one finetune-throughput row to bwd_log_path. Timestamp is
        ISO wall-clock (milliseconds) when the backward reported done — fine
        enough for ~80ms backward cadence, so multiple backwards in the same
        second don't collapse into one bucket on the eval plotter's
        time-aligned throughput band."""
        try:
            self._bwd_idx += 1
            self._bwd_total_tokens += int(n)
            parent = os.path.dirname(self.bwd_log_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            new_file = not os.path.exists(self.bwd_log_path)
            with open(self.bwd_log_path, "a", newline="") as f:
                w = csv.writer(f)
                if new_file:
                    w.writerow(["timestamp", "epoch", "batch_idx",
                                "batch_tokens", "batch_loss",
                                "total_processed_tokens"])
                w.writerow([
                    datetime.datetime.now().isoformat(timespec="milliseconds"),
                    self.current_epoch, self._bwd_idx, int(n),
                    "" if loss is None else loss, self._bwd_total_tokens,
                ])
        except Exception as e:
            dprint(f"[coord] bwd_log write failed: {e}")
