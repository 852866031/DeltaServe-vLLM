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

import collections
import csv
import datetime
import os
import threading
import time

import torch

from vllm.deltaserve import dprint


class RpsTracker:
    """Sliding-window inference-arrival rate tracker for the
    ``finetune.rps_throttle_*`` knobs.

    Hooked from ``v1/engine/core.py``'s input-queue drain — every
    request that lands in the engine's input_queue calls
    ``note_arrival(time.monotonic())``. ``rps(now)`` returns the count
    of arrivals in the last ``window_s`` seconds, divided by
    ``window_s``. Old timestamps are pruned lazily on access. Single-
    threaded inside EngineCore so no lock is needed."""

    __slots__ = ("_window_s", "_arrivals")

    def __init__(self, window_s: float) -> None:
        self._window_s = float(window_s)
        self._arrivals: "collections.deque[float]" = collections.deque()

    def note_arrival(self, ts: float) -> None:
        self._arrivals.append(ts)

    def rps(self, now: float) -> float:
        cutoff = now - self._window_s
        while self._arrivals and self._arrivals[0] < cutoff:
            self._arrivals.popleft()
        if self._window_s <= 0:
            return 0.0
        return len(self._arrivals) / self._window_s


class FtArrivalSignal:
    """[forward_interruptible / tier C under TP] A monotonic inference-arrival
    counter in POSIX shared memory, written by the EngineCore input thread on
    every ADD and read by every TP worker. Under TP the workers are separate
    processes from the engine, so the tp=1 ``threading.Event`` never reaches
    them; a counter (rather than a flag) lets each worker compare "arrivals
    now" against "arrivals when this FT-only forward started". Single writer,
    8-byte aligned store — a torn read can only make an abort spurious or one
    poll late, never wrong."""

    ENV = "DSERVE_FT_ARRIVAL_SHM"

    def __init__(self, shm, owner: bool) -> None:
        self._shm = shm
        self._owner = owner
        # No persistent buffer export (a live memoryview/ctypes view would make
        # SharedMemory.close() raise BufferError at interpreter exit); struct
        # packs/unpacks through a transient view.
        import struct
        self._pack = struct.Struct("<Q")

    @classmethod
    def create(cls) -> "FtArrivalSignal":
        from multiprocessing import shared_memory
        name = f"dserve_ft_arrivals_{os.getpid()}"
        shm = shared_memory.SharedMemory(name=name, create=True, size=8)
        sig = cls(shm, owner=True)
        sig._pack.pack_into(shm.buf, 0, 0)
        return sig

    @classmethod
    def attach(cls, name: str) -> "FtArrivalSignal":
        from multiprocessing import shared_memory
        return cls(shared_memory.SharedMemory(name=name, create=False), owner=False)

    @property
    def name(self) -> str:
        return self._shm.name

    def bump(self) -> None:
        self._pack.pack_into(self._shm.buf, 0, self.read() + 1)

    def read(self) -> int:
        return self._pack.unpack_from(self._shm.buf, 0)[0]

    def close(self) -> None:
        try:
            self._shm.close()
            if self._owner:
                self._shm.unlink()
        except Exception:  # noqa: BLE001
            pass


# [forward_interruptible / tier C] How many layers the CPU launch may run
# ahead of the GPU inside an FT-only forward. The abort check runs in a
# forward hook at CPU launch time; unbounded, the CPU finishes launching all
# layers long before the GPU executes them, so an arrival early in the
# forward's wall time is seen at a late layer index (measured: aborts
# clustered at layers 31-36 of 40) and an arrival after the last launch is
# not seen at all. Waiting on the event of the layer launched two boundaries
# ago keeps the GPU fed while making the check track GPU progress.
_LAUNCH_AHEAD_LAYERS = 2


class _LaunchAheadGate:
    def __init__(self) -> None:
        import collections
        self._events: collections.deque = collections.deque()
        self._cuda = torch.cuda.is_available()

    def reset(self) -> None:
        self._events.clear()

    def step(self) -> None:
        if not self._cuda:
            return
        ev = torch.cuda.Event()
        ev.record()
        self._events.append(ev)
        if len(self._events) > _LAUNCH_AHEAD_LAYERS:
            self._events.popleft().synchronize()


class LocalAbortPoller:
    """[forward_interruptible / tier C at tp=1] Same interface as
    ``FtAbortPoller`` over the in-process ``threading.Event`` (set by the
    engine's input thread on an ADD while an FT-only forward is in flight),
    with the same launch-ahead bound so an abort tracks GPU progress."""

    def __init__(self, event) -> None:
        self.event = event
        self.active = False
        self.layer_calls = 0
        self._gate = _LaunchAheadGate()

    def note_served(self) -> None:
        pass

    def entry(self) -> bool:
        self.layer_calls = 0
        self._gate.reset()
        aborted = self.event.is_set()
        self.active = not aborted
        return aborted

    def layer(self) -> bool:
        self.layer_calls += 1
        self._gate.step()
        return self.event.is_set()

    def hook_check(self, boundary: bool) -> bool:
        return self.layer() if (boundary and self.active) else False

    def finish(self) -> None:
        self.active = False


class FtAbortPoller:
    """[forward_interruptible / tier C under TP] The rank-symmetric abort
    decision. Every TP rank must leave an FT-only forward at the SAME layer
    (or not at all) or the next TP collective deadlocks, so a rank's local
    observation ("arrivals changed since this forward started") is MAX-
    all-reduced over the TP group's gloo ``cpu_group`` before it is acted on:
    one tiny CPU collective at entry and one per layer boundary, no GPU sync.
    Both ranks call ``entry`` / ``layer`` at identical points (same
    SchedulerOutput, same hooks), so the collective counts always match."""

    def __init__(self, signal: FtArrivalSignal, cpu_group) -> None:
        self.signal = signal
        self.group = cpu_group
        self.seen = signal.read()       # arrivals accounted for (served / re-scheduled)
        self._entry = self.seen
        self._flag = torch.zeros(1, dtype=torch.int64)
        self.layer_calls = 0
        self.reduces = 0
        self._gate = _LaunchAheadGate()
        # True only between ``entry()`` and ``finish()`` — i.e. while an
        # FT-ONLY forward is on the GPU. The accumulator hooks are armed for
        # every FT-bearing batch, including co-serving batches that carry
        # inference tokens; those must never be aborted (their inference
        # requests would lose the step), so ``hook_check`` is inert when
        # inactive and, being a plain attribute read, runs no collective.
        self.active = False

    def _symmetric(self, local: bool) -> bool:
        import torch.distributed as dist
        self._flag.fill_(1 if local else 0)
        dist.all_reduce(self._flag, op=dist.ReduceOp.MAX, group=self.group)
        self.reduces += 1
        return bool(self._flag.item())

    def note_served(self) -> None:
        """An inference-bearing batch is being executed: whatever arrived so
        far has been scheduled, so it no longer counts as pending."""
        self.seen = self.signal.read()

    def entry(self) -> bool:
        """At the start of an FT-only forward: abort before any kernel if an
        arrival is pending (landed after the last inference batch)."""
        cur = self.signal.read()
        self._entry = cur
        self.layer_calls = 0
        self._gate.reset()
        aborted = self._symmetric(cur != self.seen)
        self.active = not aborted
        return aborted

    def layer(self) -> bool:
        """At a layer boundary of the FT-only forward. Bounds the CPU's
        launch-ahead first so the decision reflects GPU progress."""
        self.layer_calls += 1
        self._gate.step()
        return self._symmetric(self.signal.read() != self._entry)

    def hook_check(self, boundary: bool) -> bool:
        """What the accumulator hooks call: only the layer-boundary hook
        (``input_layernorm`` pre-hook) runs the collective; every other hook
        in the layer returns False without one, keeping the per-rank
        collective count identical and the cost to one reduce per layer."""
        return self.layer() if (boundary and self.active) else False

    def finish(self) -> None:
        """After the FT-only forward (completed or aborted): the arrivals seen
        so far will be scheduled next step (rollback re-schedules; a completed
        step frees the engine), so they are accounted for."""
        self.active = False
        self.seen = self.signal.read()


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
        # [DeltaServe] Phase 7 / M4.1: TP relay mode. Under TP>1 (multiproc) the
        # scheduler (EngineCore) and runner (worker) hold SEPARATE coordinator
        # singletons: the scheduler decides the trigger but the backward_process
        # IPC handle lives on the worker's coordinator. When relay_mode is set:
        #   - scheduler coord (backward_process is None): _trigger_backward stashes
        #     the trigger params into _pending_trigger_cmd instead of no-op'ing;
        #     the scheduler drains it onto SchedulerOutput. It applies the worker's
        #     relayed saved-counts (apply_relayed_saved) and backward-done ack
        #     (apply_relayed_done).
        #   - worker coord (backward_process set): executes the relayed trigger
        #     (execute_trigger → notify_buffer_full) and polls the child ack
        #     (poll_backward_relay), reporting both back via ModelRunnerOutput.
        # relay_mode stays False for tp=1 → every path below is inert and the
        # single-GPU coordinator behaves exactly as before.
        self.relay_mode = False
        self._pending_trigger_cmd: dict | None = None
        # Corpus total tokens per epoch (set by the scheduler after
        # FinetuningStore.load). On tp=1 the scheduler sends it straight to the
        # child; under relay mode the scheduler coordinator has no child, so
        # the value rides on every relayed trigger command and the worker
        # coordinator forwards it to its own child ONCE (the child's per-epoch
        # progress meter printed ``N/?`` forever before this).
        self.corpus_total_tokens: int = 0
        self._corpus_meta_relayed = False
        # [rps_throttle] Sliding-window arrival-rate tracker. Created
        # eagerly with a 1.0s default window so ``note_arrival`` from
        # the engine input-queue drain is always safe to call; the
        # scheduler resizes the window at startup once it has read
        # the FinetuneConfig knob ``rps_throttle_window_s``.
        self.rps_tracker = RpsTracker(window_s=1.0)
        # Hysteresis latch for the rps_throttle gate. True once the
        # close threshold has been crossed and admission was closed
        # by the throttle; flipped back to False once RPS drops below
        # the open threshold and admission is re-opened by the
        # throttle. ``admission_open`` is what's actually consulted by
        # ``next_ft_budget`` — this flag only exists so the throttle
        # knows whether IT was the reason the gate is closed (we don't
        # want to spuriously re-open over a buffer-full close).
        self.rps_throttle_active = False
        # Wall-clock instant (monotonic seconds) at which the throttle
        # most recently engaged. Used by ``check_rps_throttle`` to
        # enforce ``rps_throttle_close_time`` — the temporal-hysteresis
        # minimum hold. ``None`` while disengaged.
        self.rps_throttle_engaged_at: float | None = None
        # Mirror of the runner's ``_throttle_held`` flag — True while
        # the fwd_token_throttle is keeping the backward paused. The
        # runner writes it from its per-step pause-decision block
        # (gpu_model_runner.py). Read by ``_trigger_backward`` so the
        # scheduler can pre-pause the bwd when the last step's
        # throttle was HIT, avoiding wasted setup work in the bwd
        # subprocess before it blocks at its first ``_maybe_pause``.
        self.fwd_throttle_active = False
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
        # [forward_interruptible / tier C under TP] Worker side: the
        # rank-symmetric poller (None at tp=1 → the threading.Event path).
        self.ft_abort_poller = None
        # [diag] Wall-clock instant the current pending backward was kicked
        # off (set by _trigger_backward, cleared in poll_backward on ack).
        # poll_backward warns once if the gap to "now" exceeds the threshold
        # below — a backward that takes >> the normal ~100ms almost always
        # means the child crashed silently (e.g. CUDA OOM, OS-killed) and
        # the parent will spin forever waiting for an ack that never comes.
        self._pending_backward_t0: float | None = None
        self._pending_backward_warned = False
        self.pending_backward_warn_s = 5.0
        # [M4.2] Worker side: this rank's backward ack, held until every rank
        # reports done (see relay_backward_outstanding / take_relay_ack).
        self._relay_ack: dict | None = None

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

    def check_rps_throttle(self, now: float, close_rps: float,
                           open_rps: float, window_s: float,
                           close_time: float = 0.0) -> None:
        """[rps_throttle] Per-step decision: close ``admission_open``
        when the inference arrival rate crosses ``close_rps``; reopen
        when it drops below ``open_rps``. The gap between the two
        thresholds is the spatial hysteresis band — RPS sitting
        inside it leaves the gate in its current state, preventing
        flapping at the boundary.

        ``close_time`` adds TEMPORAL hysteresis on top: once the
        throttle engages, the release check is deferred until at
        least ``close_time`` seconds have elapsed since the engage
        moment. A transient sub-second dip below ``open_rps`` no
        longer flaps the gate — useful when bursts are short but
        you want the FT pause to be sticky for a while.

        Option B (overload ``admission_open``): the throttle uses the
        existing admission flag rather than a separate gate, so
        ``next_ft_budget`` doesn't need a new AND clause. The
        coordinator's other writers of ``admission_open`` (buffer
        fill in ``note_injection``, backward start in
        ``_trigger_backward``, backward done in ``poll_backward``)
        still operate; the throttle interleaves with them under the
        single-threaded EngineCore. ``space_remaining()`` /
        ``pending_backward`` checks in ``next_ft_budget`` protect
        against any spurious "open" the throttle issues over a
        buffer-full state.

        ``self.rps_throttle_active`` is the throttle's internal
        latch so it knows whether to flip on the next transition;
        it's not consulted by anything else.
        ``self.rps_throttle_engaged_at`` is the engage timestamp
        used to enforce ``close_time``."""
        if close_rps <= 0:
            return
        # Allow the scheduler to resize the window at startup; cheap
        # if it didn't change.
        if abs(self.rps_tracker._window_s - float(window_s)) > 1e-9:
            self.rps_tracker._window_s = float(window_s)
        cur_rps = self.rps_tracker.rps(now)
        if self.rps_throttle_active:
            # Engaged → release when load drops back below open_rps
            # AND the minimum-hold ``close_time`` has elapsed since
            # the engage moment. The held_for / remaining bookkeeping
            # in the dprint makes it obvious from the log why a
            # release was deferred when one would otherwise fire.
            _engaged_at = self.rps_throttle_engaged_at or now
            _held_for = max(0.0, now - _engaged_at)
            if cur_rps < open_rps:
                # Idle bypass: a fully-zero RPS reading means
                # ``rps_tracker`` saw NO arrivals in the last
                # ``window_s`` seconds — there's no inference work
                # to protect from, so skip the close_time hold and
                # release immediately. ``cur_rps`` is an exact
                # ``count / window_s`` so ``== 0.0`` is reliable
                # (no float fuzz); ``count == 0`` ⇔ ``rps == 0``.
                _idle_bypass = (cur_rps == 0.0)
                if (close_time > 0 and _held_for < close_time
                        and not _idle_bypass):
                    _remaining = close_time - _held_for
                    # Quiet log — fires every step the release is
                    # deferred; only print at >=1% remaining so a
                    # near-zero residual doesn't spam the trace.
                    if _remaining > 0.01 * close_time:
                        dprint(
                            f"[rps_throttle] release deferred: "
                            f"rps={cur_rps:.1f} < "
                            f"open_rps={open_rps:g} but held only "
                            f"{_held_for:.2f}s / {close_time:g}s "
                            f"(remaining {_remaining:.2f}s)")
                else:
                    self.rps_throttle_active = False
                    self.rps_throttle_engaged_at = None
                    self.admission_open = True
                    _reason = ("idle-bypass: rps=0 for the last "
                               f"{self.rps_tracker._window_s:g}s"
                               if _idle_bypass and close_time > 0
                                   and _held_for < close_time
                               else f"rps={cur_rps:.1f} < "
                                    f"open_rps={open_rps:g}")
                    dprint(
                        f"[rps_throttle] release: {_reason} → "
                        f"admission_open=True "
                        f"(held for {_held_for:.2f}s)")
        else:
            # Idle → engage when load crosses close_rps.
            if cur_rps > close_rps:
                self.rps_throttle_active = True
                self.rps_throttle_engaged_at = now
                self.admission_open = False
                _ct = (f" min_hold={close_time:g}s"
                       if close_time > 0 else "")
                dprint(
                    f"[rps_throttle] engage: rps={cur_rps:.1f} > "
                    f"close_rps={close_rps:g} → "
                    f"admission_open=False{_ct}")

    def start_finetuning(self) -> None:
        """Open FT admission (POST /start_finetuning). Idempotent."""
        if self.ft_start_time is None:
            self.ft_start_time = time.monotonic()
        if not self.ft_started:
            self.ft_started = True
            dprint("[coord] finetuning STARTED (admission opened by HTTP)")

    def stop_finetuning(self) -> None:
        """Close FT admission (POST /stop_finetuning). Idempotent.

        Flips ``ft_started`` back to False — the master gate in
        ``next_ft_budget`` that keeps FT samples out of the batch.
        Other coordinator state (``admission_open``, ``fill_count``,
        ``pending_backward``, throttle latches, …) is intentionally
        left alone:

          * Any in-flight backward continues to completion — stopping
            doesn't abort a cycle, it just denies NEW admissions.
          * Buffer space + counts are preserved, so a later
            ``start_finetuning`` resumes from exactly where we left
            off (the next FT step will find the same partially-filled
            buffer, the same per-epoch processed-token count, etc.).
          * ``ft_start_time`` is preserved on purpose — it anchors the
            runner's ``[batch ...] t=+Ns`` log line and would jump
            backwards on a re-start if we reset it.

        Cycle: ``start`` → run for a while → ``stop`` → idle (no new
        FT, in-flight bwd drains) → ``start`` → resume."""
        if self.ft_started:
            self.ft_started = False
            dprint(
                "[coord] finetuning STOPPED (admission closed by HTTP) "
                f"| fill_count={self.fill_count} "
                f"reserved_fill={self.reserved_fill} "
                f"pending_backward={self.pending_backward}")

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
        # [diag] start a timer so poll_backward can warn if the child never
        # acks within `pending_backward_warn_s` (dead-child symptom).
        self._pending_backward_t0 = time.monotonic()
        self._pending_backward_warned = False
        # dprint(
        #     f"[coord] {reason} ({self.fill_count}/{self.capacity}) -> "
        #     f"signal backward (cycle {self._cycle}); FT admission CLOSED"
        # )
        if self.backward_process is not None:
            # [fwd_token_throttle] If the last step's throttle decision
            # was HIT, the inference side is still busy — proactively
            # pause the backward BEFORE sending the work signal so it
            # lands in a known-paused state. Without this, the
            # backward subprocess receives the signal, sets up its
            # cycle, and only blocks at the first ``_maybe_pause``
            # layer boundary — wasted setup work the throttle would
            # have otherwise prevented. ``fwd_throttle_active`` is
            # mirrored from the runner's ``_throttle_held`` flag every
            # time the per-step pause decision flips, so this read is
            # the same answer as "was the most recent batch a HIT?".
            if self.fwd_throttle_active:
                self.gpu_pause_backward()
                dprint(f"[coord] {reason} but fwd_throttle is HIT → "
                       f"pre-paused backward (cycle {self._cycle})")
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
        elif self.relay_mode:
            # [M4.1] Scheduler-side coordinator under TP: no local handle. Stash
            # the trigger params for the scheduler to broadcast on SchedulerOutput
            # (→ every worker fires its own backward child, lock-step). The
            # capture-completion sync happens worker-side in execute_trigger.
            self._pending_trigger_cmd = {
                "n": int(self.fill_count),
                "sample_lens": list(self.sample_lens),
                "epoch": int(self.current_epoch),
                "sleep_s": float(self.backward_sleep_s),
                # One int per trigger; the worker forwards it to its child once.
                "total_tokens_per_epoch": int(self.corpus_total_tokens),
            }

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

    # A backward that has been outstanding this long (vs ~0.1-0.5 s normally)
    # is treated as dead even when the child's liveness is unknown (relay mode).
    backward_dead_after_s = 60.0

    def _check_backward_stuck(self, alive) -> None:
        """[diag] Warn once when a backward has been outstanding longer than
        ``pending_backward_warn_s``; if the child is known dead (``alive`` is
        False) or the wait exceeds ``backward_dead_after_s``, DISABLE
        finetuning (``ft_started=False``) with an error instead of leaving FT
        admission silently closed forever (the C7 wedge)."""
        t0 = self._pending_backward_t0
        if t0 is None:
            return
        waited = time.monotonic() - t0
        if (not self._pending_backward_warned
                and waited > self.pending_backward_warn_s):
            exitcode = getattr(getattr(self.backward_process, "_proc", None),
                               "exitcode", None)
            dprint(
                f"[coord] !!! backward STUCK: no ack in {waited:.1f}s "
                f"(threshold {self.pending_backward_warn_s:.1f}s) | "
                f"child alive={alive} exitcode={exitcode} | FT admission stays "
                f"CLOSED until the ack arrives — likely the backward subprocess "
                f"died (CUDA OOM / segfault / OS kill). Check stderr above for "
                f"[backward] traces."
            )
            self._pending_backward_warned = True
        # Only after the warning threshold: server teardown kills the child
        # before the coordinator stops polling, which would otherwise print a
        # spurious "DEAD after 0s" at every shutdown.
        if (waited > self.pending_backward_warn_s
                and (alive is False or waited > self.backward_dead_after_s)
                and self.ft_started):
            dprint(
                f"[coord] !!! backward child {'DEAD' if alive is False else 'unresponsive'} "
                f"after {waited:.0f}s — FINETUNING DISABLED for the rest of this run "
                f"(inference continues; restart the server to resume finetuning)."
            )
            self.ft_started = False
            self.admission_open = False

    def poll_backward(self) -> None:
        """Non-blocking: reopen admission once the backward pass has finished.

        On completion, invokes ``on_backward_done(buffer_samples)`` (set by
        the scheduler at init) so the store can call ``commit_claimed`` on
        exactly the samples whose activations were just trained on. Done
        BEFORE clearing ``buffer_samples`` so the hook sees the full list.
        """
        if not self.pending_backward:
            return
        if self.backward_process is None:
            # Relay mode (scheduler-side coordinator under TP): the ack comes
            # through apply_relayed_done; only the stuck-detector runs here.
            self._check_backward_stuck(alive=None)
            return
        resp = self.backward_process.poll_response()
        if resp is None:
            alive = getattr(getattr(self.backward_process, "_proc", None),
                            "is_alive", lambda: None)()
            self._check_backward_stuck(alive=alive)
            return
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
        self._pending_backward_t0 = None
        self._pending_backward_warned = False

    # -- [M4.1] TP relay bridge -------------------------------------------
    # Scheduler-side (EngineCore coordinator) helpers: consume what the worker
    # relays and produce the trigger command for SchedulerOutput.

    def take_trigger_cmd(self) -> dict | None:
        """Scheduler side: pop the pending trigger command (set by
        _trigger_backward under relay_mode) to broadcast on SchedulerOutput.
        None when there's nothing to fire this step."""
        cmd = self._pending_trigger_cmd
        self._pending_trigger_cmd = None
        return cmd

    def apply_relayed_saved(self, n: int, sample_lens: list[int] | None) -> None:
        """Scheduler side: apply the worker's relayed per-step saved-token count
        (its record_capture result). Mirrors record_capture so the scheduler's
        buffer accounting advances and its buffer-full / epoch-flush trigger can
        fire (→ _pending_trigger_cmd)."""
        if not self.relay_mode:
            return
        self.record_capture(int(n), list(sample_lens) if sample_lens else None)

    def apply_relayed_done(self, resp: dict | None) -> None:
        """Scheduler side: apply the worker's relayed backward-done ack. Mirrors
        poll_backward's post-ack path: write the bwd_log row, commit the trained
        samples, reset the buffer, and reopen FT admission."""
        if not self.relay_mode or not resp:
            return
        if (self.bwd_log_path and isinstance(resp, dict)
                and resp.get("event") == "activations_processed"):
            self._write_bwd_log_row(int(resp.get("n", 0)), resp.get("loss"))
        if self.on_backward_done is not None and self.buffer_samples:
            try:
                self.on_backward_done(list(self.buffer_samples))
            except Exception as e:  # noqa: BLE001
                dprint(f"[coord] on_backward_done (relay) failed: {e}")
        self.fill_count = 0
        self.reserved_fill = 0
        self.sample_lens = []
        self.buffer_samples = []
        self.pending_backward = False
        self.admission_open = True
        self._pending_backward_t0 = None
        self._pending_backward_warned = False

    # Worker-side (per-rank coordinator that owns the backward_process handle).

    def execute_trigger(self, cmd: dict) -> None:
        """Worker side: fire this rank's backward child for a relayed trigger
        command. Syncs on the capture-completion event (so the child sees the
        saved activations) then sends the work signal over the child's pipe."""
        if self.backward_process is None or not cmd:
            return
        # Forward the corpus total to this rank's child the first time it
        # arrives (pipe sends are ordered, so it lands before the work signal
        # below and the very first cycle line already shows ``N/total``).
        total = int(cmd.get("total_tokens_per_epoch", 0) or 0)
        if total > 0 and not self._corpus_meta_relayed:
            self.backward_process.set_corpus_meta(total)
            self._corpus_meta_relayed = True
        if self.fwd_throttle_active:
            self.gpu_pause_backward()
        if self.capture_done_evt is not None:
            self.capture_done_evt.synchronize()
        else:
            torch.cuda.synchronize()
        self.pending_backward = True
        self.backward_process.notify_buffer_full(
            int(cmd.get("n", 0)), self.backward_sleep_s,
            list(cmd.get("sample_lens", [])), epoch=int(cmd.get("epoch", 0)))

    def poll_backward_relay(self) -> dict | None:
        """Worker side: non-blocking poll of this rank's backward child. Returns
        the ack payload (to relay to the scheduler via ModelRunnerOutput) when
        the backward finished, else None. Does NOT run on_backward_done / commit
        (that is scheduler-side). Single-rank semantics — the runner uses the
        three-call sequence below instead so the ack is only relayed once EVERY
        rank's child is done (M4.2 ack-race fix)."""
        if self.poll_own_backward_ack():
            return self.take_relay_ack()
        return None

    # [DeltaServe] Phase 7 / M4.2: the backward ack race. Only ``output_rank``'s
    # ModelRunnerOutput reaches the scheduler, so relaying this rank's own ack
    # would reopen admission while the other rank's child may still be
    # publishing / zeroing buffers. The runner therefore (1) polls its own
    # child into ``_relay_ack``, (2) all-reduces a "mine is done" flag with MIN
    # across the TP ranks, and (3) takes + relays the ack only when the reduced
    # flag says every rank is done — on the same step on every rank, so the
    # outstanding state clears in lock-step.

    def relay_backward_outstanding(self) -> bool:
        """Worker side: True while a relayed backward has been fired on this
        rank and its ack has not yet been handed to the scheduler."""
        return self.pending_backward or self._relay_ack is not None

    def poll_own_backward_ack(self) -> bool:
        """Worker side: poll this rank's child once (non-blocking) and stash
        its ack. Returns True once the ack is held (this rank's backward is
        done), whether it arrived now or earlier."""
        if (self._relay_ack is None and self.pending_backward
                and self.backward_process is not None):
            resp = self.backward_process.poll_response()
            if resp is not None:
                self.pending_backward = False
                self._relay_ack = resp
        return self._relay_ack is not None

    def take_relay_ack(self) -> dict | None:
        """Worker side: hand over the stashed ack (after every rank agreed it
        is done) and clear the outstanding state."""
        resp = self._relay_ack
        self._relay_ack = None
        return resp

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
            # Header on a new OR empty file: the eval drivers truncate the
            # log at launch (the server appends), which leaves an existing
            # zero-byte file — without the header the plotter / trimmer
            # cannot read the rows.
            new_file = (not os.path.exists(self.bwd_log_path)
                        or os.path.getsize(self.bwd_log_path) == 0)
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
