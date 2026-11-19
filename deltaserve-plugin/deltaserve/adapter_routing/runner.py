# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Drives one indexing job across executors, chunk by chunk.

Indexing runs on its own: ``IndexingRunner`` is a thread that takes a chunk,
has it encoded, stores the progress, takes the next. The scheduler does not
issue its work — it only says where indexing may run, and only when that
changes (``IndexingPlacement``):

- **gpu** while the engine has no inference request (and has had none for a
  quiet period) and a GPU executor exists — the indexing process that shares
  the route encoder's weights (``index_process``);
- **cpu** otherwise, with the thread count of the current level.

``IndexingRunner.step()`` does one thing per call: collect the chunk in flight,
or hand the next one to the chosen executor. Moving between devices costs
nothing because the job's state is the indexer's checkpoint:

- cpu → gpu: the CPU chunk in flight is allowed to finish, the next one goes to
  the GPU;
- gpu → cpu: the GPU chunk in flight is dropped (the indexing process stops at
  its next encoder layer once the grant is cleared) and its samples are handed
  out again on the CPU.

An executor is anything with ``submit(texts, threads)``, ``poll(timeout)`` →
embeddings | None, ``cancel()``, ``loaded``, ``unload()`` and ``stop()``.
"""

import threading
import time

from deltaserve import dprint
from deltaserve.adapter_routing.index_process import Preempted


IDLE, BUSY = "idle", "busy"


class IndexingPlacement:
    """What the scheduler tells indexing, and only when it changes:

    - **busy** — an inference request is in the engine: indexing stays off the
      GPU (the grant is cleared at once) and on the CPU uses at most
      ``busy_threads`` threads;
    - **idle** — none has been for ``gpu_after_quiet_s`` seconds: the GPU may
      be used (grant set), the CPU with ``idle_threads``.

    ``note_inference()`` is the scheduler's side: called when a request
    arrives and by every ``schedule()`` that still has one. The quiet period
    keeps the short gaps between requests of a steady load from flipping the
    level (a GPU chunk started in such a gap would be dropped by the next
    arrival). ``grant`` is the ``IndexingGrant`` the indexing process obeys."""

    def __init__(self, gpu_after_quiet_s: float = 3.0, clock=time.monotonic,
                 grant=None, idle_threads: int = 8, busy_threads: int = 2) -> None:
        self.gpu_after_quiet_s = float(gpu_after_quiet_s)
        self._clock = clock
        self._last_inference: float | None = None
        self.grant = grant
        self.idle_threads, self.busy_threads = int(idle_threads), int(busy_threads)
        self.level = IDLE
        self.changes = 0
        if grant is not None:
            grant.set(False)          # nothing is granted before the first look

    def _set(self, level: str) -> None:
        if level != self.level:
            self.level = level
            self.changes += 1
        if self.grant is not None and self.grant.is_set() != (level == IDLE):
            self.grant.set(level == IDLE)

    def note_inference(self) -> None:
        """Inference work is present (an arrival, a request still running)."""
        self._last_inference = self._clock()
        self._set(BUSY)

    def refresh(self) -> str:
        """The level now (the quiet period may have run out)."""
        last = self._last_inference
        quiet = last is None or self._clock() - last >= self.gpu_after_quiet_s
        self._set(IDLE if quiet else BUSY)
        return self.level

    def cpu_threads(self) -> int:
        return self.idle_threads if self.refresh() == IDLE else self.busy_threads

    def choose(self, gpu_available: bool) -> str:
        return "gpu" if self.refresh() == IDLE and gpu_available else "cpu"


class IndexingRunner:
    def __init__(self, indexer, executors: dict, cpu_threads=None, placement=None,
                 chunk_samples: int | dict = 16, recheck_s: float = 0.5,
                 poll_s: float = 0.05, name: str = "indexing",
                 stop_executors: bool = True, on_exit=None) -> None:
        self.indexer, self.executors = indexer, dict(executors)
        # ``cpu_threads() -> int``: threads for the next CPU chunk (0 = wait).
        # Default: the placement's level.
        if cpu_threads is None and placement is not None:
            cpu_threads = placement.cpu_threads
        self.cpu_threads, self.placement = cpu_threads, placement
        self.chunk = (chunk_samples if isinstance(chunk_samples, dict)
                      else {d: int(chunk_samples) for d in self.executors})
        self.recheck_s, self.poll_s, self.name = recheck_s, poll_s, name
        self._inflight: tuple[str, int] | None = None
        self._paused_since: float | None = None
        self.state = "pending"            # pending / cpu / gpu / paused / done
        self.pause_reason = ""
        self.chunks = {d: 0 for d in self.executors}
        self.discarded = 0
        self.preempted = 0            # GPU chunks inference interrupted
        self.error: str | None = None
        self._gpu_off_until = 0.0     # after a GPU executor failure: CPU only
        # Shared executors (one encoder process for every job) are not ours to stop.
        self.stop_executors, self.on_exit = stop_executors, on_exit
        self._thread = None
        self._stop = threading.Event()

    # ------------------------------------------------------------- one step
    def _device(self) -> str:
        if self.placement is None:
            return "cpu" if "cpu" in self.executors else next(iter(self.executors))
        gpu = "gpu" in self.executors and time.monotonic() >= self._gpu_off_until \
            and getattr(self.executors["gpu"], "ready", True)
        dev = self.placement.choose(gpu)
        return dev if dev in self.executors else next(iter(self.executors))

    def step(self) -> float:
        """One action. Returns how long the caller may sleep before the next."""
        idx = self.indexer
        if idx.done:
            self.state = "done"
            return 0.0
        dev = self._device()
        if self._inflight is not None:
            cur, start = self._inflight
            if cur == "gpu" and dev != "gpu":
                # Inference is back: drop the GPU chunk now, redo it on the CPU.
                self.executors[cur].cancel()
                self._inflight = None
                self.discarded += 1
                return 0.0
            try:
                emb = self.executors[cur].poll(self.poll_s)
            except Exception as e:  # noqa: BLE001
                if cur != "gpu" or "cpu" not in self.executors:
                    raise
                self._inflight = None
                self.discarded += 1
                if isinstance(e, Preempted):
                    # Inference took the GPU back mid-chunk: the scheduler's
                    # level says where the samples go next.
                    self.preempted += 1
                    return 0.0
                # The indexing process could not encode it: redo it on the
                # CPU and leave the GPU alone for a while.
                dprint(f"[indexing] {self.name}: GPU chunk failed ({e}); "
                       f"continuing on the CPU")
                self._gpu_off_until = time.monotonic() + 30.0
                return 0.0
            if emb is None:
                return 0.0
            self._inflight = None
            if idx.commit(start, emb):
                self.chunks[cur] += 1
            else:
                self.discarded += 1
            return 0.0

        threads = 0
        if dev == "cpu" and self.cpu_threads is not None:
            threads = int(self.cpu_threads())
            if threads < 1:
                if self.state != "paused":
                    dprint(f"[indexing] {self.name} paused by the scheduler")
                    self._paused_since = time.monotonic()
                self.state, self.pause_reason = "paused", "scheduler"
                return self.recheck_s
        if self.state != dev:
            done, total = idx.progress
            dprint(f"[indexing] {self.name} running on {dev} ({done}/{total})")
        self.state, self.pause_reason = dev, ""
        start, texts = idx.next_chunk(self.chunk[dev])
        self.executors[dev].submit(texts, threads)
        self._inflight = (dev, start)
        return 0.0

    # --------------------------------------------------------------- thread
    @property
    def done(self) -> bool:
        return self.indexer.done and self._inflight is None

    def _loop(self) -> None:
        try:
            while not self._stop.is_set() and not self.done:
                wait = self.step()
                if wait > 0:
                    self._stop.wait(wait)
            if self.done:
                self.state = "done"
        except Exception as e:  # noqa: BLE001 — surfaced through the stage
            self.error = f"{type(e).__name__}: {e}"
        finally:
            if self.stop_executors:
                for ex in self.executors.values():
                    try:
                        ex.stop()
                    except Exception:  # noqa: BLE001
                        pass
            if self.on_exit is not None:
                self.on_exit()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name=f"deltaserve-{self.name}")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def describe(self) -> dict:
        done, total = self.indexer.progress
        return {"state": self.state, "encoded": done, "total": total,
                "chunks": dict(self.chunks), "discarded": self.discarded,
                "preempted": self.preempted,
                "pause_reason": self.pause_reason, "error": self.error}
