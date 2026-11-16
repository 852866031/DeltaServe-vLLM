# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Drives one indexing job across executors, chunk by chunk.

``IndexingPlacement`` says where the next chunk goes:

- **cpu** while inference work is present (or was, in the last
  ``gpu_after_quiet_s`` seconds) — throttled by the ``CpuGovernor``;
- **gpu** when only finetuning batches are running and a GPU executor exists.

``IndexingRunner.step()`` does one thing per call: collect the chunk in flight,
or hand the next one to the chosen executor. Moving between devices costs
nothing because the job's state is the indexer's checkpoint:

- cpu → gpu: the CPU chunk in flight is allowed to finish, the next one goes to
  the GPU;
- gpu → cpu: the GPU chunk in flight is cancelled at once (the quick kill when
  inference arrives) and handed out again on the CPU.

``start()`` runs the steps on a daemon thread so indexing progresses whether or
not the engine is stepping. An executor is anything with ``submit(texts,
threads)``, ``poll(timeout)`` → embeddings | None, ``cancel()``, ``loaded``,
``unload()`` and ``stop()``.
"""

import threading
import time

from deltaserve import dprint


class IndexingPlacement:
    def __init__(self, gpu_after_quiet_s: float = 3.0, clock=time.monotonic) -> None:
        self.gpu_after_quiet_s = float(gpu_after_quiet_s)
        self._clock = clock
        self._last_inference: float | None = None

    def note_inference(self) -> None:
        """Call whenever inference work is seen (an arrival, a running request)."""
        self._last_inference = self._clock()

    def choose(self, gpu_available: bool) -> str:
        if not gpu_available:
            return "cpu"
        last = self._last_inference
        if last is not None and self._clock() - last < self.gpu_after_quiet_s:
            return "cpu"
        return "gpu"


class RouterEncodeExecutor:
    """The GPU executor: indexing chunks are encoded by the router service's
    resident encoder (indexing *borrows* it — nothing else is loaded on the
    GPU). The service runs a chunk only when no route request is waiting.
    ``cancel`` stops waiting for the chunk in flight: its result is dropped,
    and the runner hands the same samples to the CPU."""

    device, loaded = "gpu", True

    def __init__(self, socket: str, timeout_s: float = 30.0) -> None:
        from deltaserve.adapter_routing.service import RouterClient

        self._client = RouterClient(socket, timeout_s=timeout_s)
        self._timeout = timeout_s
        self._seq = 0
        self._job = None            # [seq, Event, embeddings, error]

    def submit(self, texts: list, threads: int = 0) -> None:
        self._seq += 1
        job = [self._seq, threading.Event(), None, None]
        self._job = job

        def run():
            try:
                job[2] = self._client.encode(texts, self._timeout)
            except Exception as e:  # noqa: BLE001 — surfaced by poll()
                job[3] = e
            job[1].set()

        threading.Thread(target=run, daemon=True, name="deltaserve-gpu-chunk").start()

    def poll(self, timeout: float = 0.0):
        job = self._job
        if job is None or not job[1].wait(timeout):
            return None
        self._job = None
        if job[3] is not None:
            raise job[3]
        return job[2]

    def cancel(self) -> None:
        self._job = None

    def unload(self) -> None:
        """Nothing to free: the encoder is the router service's."""

    def stop(self) -> None:
        self._job = None
        self._client.close()


class IndexingRunner:
    def __init__(self, indexer, executors: dict, governor=None, placement=None,
                 chunk_samples: int | dict = 16, recheck_s: float = 0.5,
                 poll_s: float = 0.05, name: str = "indexing",
                 stop_executors: bool = True, on_exit=None) -> None:
        self.indexer, self.executors = indexer, dict(executors)
        self.governor, self.placement = governor, placement
        self.chunk = (chunk_samples if isinstance(chunk_samples, dict)
                      else {d: int(chunk_samples) for d in self.executors})
        self.recheck_s, self.poll_s, self.name = recheck_s, poll_s, name
        self._inflight: tuple[str, int] | None = None
        self._paused_since: float | None = None
        self.state = "pending"            # pending / cpu / gpu / paused / done
        self.pause_reason = ""
        self.chunks = {d: 0 for d in self.executors}
        self.discarded = 0
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
        gpu = "gpu" in self.executors and time.monotonic() >= self._gpu_off_until
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
                # The router service could not encode it: redo it on the CPU
                # and leave the GPU alone for a while.
                dprint(f"[indexing] {self.name}: GPU chunk failed ({e}); "
                       f"continuing on the CPU")
                self._gpu_off_until = time.monotonic() + 30.0
                self._inflight = None
                self.discarded += 1
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
        if dev == "cpu" and self.governor is not None:
            ex = self.executors[dev]
            d = self.governor.decide(loaded=bool(ex.loaded))
            if not d.run:
                now = time.monotonic()
                if self.state != "paused" or d.reason != self.pause_reason:
                    dprint(f"[indexing] {self.name} paused: {d.reason}")
                    self._paused_since = now
                self.state, self.pause_reason = "paused", d.reason
                if ex.loaded and "memory" in d.reason:
                    ex.unload()           # give the memory back while waiting
                return self.recheck_s
            threads = d.threads
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
                "pause_reason": self.pause_reason, "error": self.error}
