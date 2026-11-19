#!/usr/bin/env python
"""Adapter indexing: centroid index, resumable indexer, the scheduler's
placement + GPU grant, the runner, the GPU indexing process's server and its
executor, and the ``indexing`` job stage (deltaserve.adapter_routing).

CPU-only. The encoder is replaced by a deterministic fake, so no model is
loaded — except in the last, opt-in test. The indexing process itself (CUDA
IPC, ``IndexEncoderProcess``) is not started: its server loop and chunk
function run here in a thread with a fake tokenizer / model.

    python tests/test_adapter_indexing.py
    DSERVE_TEST_REAL_ENCODER=1 python tests/test_adapter_indexing.py   # + real e5 on CPU

Checks
  * the centroid is the re-normalised mean of the normalised embeddings, however
    the job was chunked, interrupted, resumed or moved between executors;
  * a checkpoint resumes the same job and is ignored for a different one; a
    late result of a dropped chunk is rejected;
  * the centroid index routes by cosine top-1 and falls back below a threshold;
  * config: ``indexing_cpu_busy_threads`` (default 2, >= 1); the governor's and
    the router service's settings no longer exist; cpu-list parsing;
  * ``IndexingGrant``: created cleared, attached by name, set / cleared seen
    through both handles, closed (the owner removes the segment);
  * ``IndexingPlacement`` (injected clock, a grant): grant cleared at start;
    idle → grant set, idle thread count, "gpu" when a GPU executor exists;
    ``note_inference()`` → busy and grant cleared at once, busy thread count,
    "cpu"; busy until the quiet period has passed again; ``changes`` counts
    level changes only;
  * the runner: ``cpu_threads() == 0`` pauses ("scheduler") and resumes; the
    CPU executor's thread count follows the placement's level; cpu → gpu when
    inference is quiet, the GPU chunk dropped the moment inference returns; a
    GPU chunk answered ``Preempted`` is counted and handed out again — on the
    GPU if the level is still idle, on the CPU once it is busy — with no GPU
    back-off, while any other GPU failure backs off to the CPU; a GPU executor
    that is not ready is not used;
  * ``index_process._encode``: no grant / grant cleared → ``Preempted`` before
    the forward; otherwise prefix, truncation, masked mean pooling, unit length;
  * ``index_process._serve`` + ``SharedEncoderExecutor`` over a unix socket:
    encode → embeddings; grant cleared → "preempted" → ``poll`` raises
    ``Preempted``; an encoder exception → "error" → RuntimeError; ping → the
    chunks / preempted / errors counts; cancel, stop + reconnect, a process
    that is gone or does not answer; and the runner driving that executor
    through a mid-chunk preemption;
  * the ``indexing`` stage is refused with adapter routing off, runs next to
    the other stages with it on, and publishes the centroid.
"""

import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_passed = 0
_failed = 0
_TTY = sys.stdout.isatty()
_GREEN = "\033[92m" if _TTY else ""
_RED = "\033[91m" if _TTY else ""
_RESET = "\033[0m" if _TTY else ""
DIM = 16


def green(msg):
    print(f"{_GREEN}{msg}{_RESET}")


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  {_GREEN}[PASS]{_RESET} {name}")
    else:
        _failed += 1
        print(f"  {_RED}[FAIL]{_RESET} {name}")


def error_of(fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        return str(e)
    return None


def fake_embed(texts):
    """Deterministic, un-normalised 'embedding' per text."""
    out = []
    for t in texts:
        rng = np.random.default_rng(abs(hash(("emb", t))) % (2 ** 32))
        out.append(rng.normal(size=DIM) * (1 + len(t) % 3))
    return np.array(out, dtype=np.float32).reshape(len(texts), DIM)


def reference_centroid(texts):
    e = fake_embed(texts).astype(np.float64)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    m = e.mean(0)
    return (m / np.linalg.norm(m)).astype(np.float32)


class FakeExec:
    """An executor that answers after ``delay`` polls."""

    def __init__(self, device, delay=0):
        self.device, self.delay = device, delay
        self.loaded = False
        self.job = None
        self.submitted, self.cancelled, self.unloads, self.stopped = 0, 0, 0, False
        self.threads_seen, self.texts_seen = [], []

    def submit(self, texts, threads):
        assert self.job is None, "chunk already in flight"
        self.job = [list(texts), self.delay]
        self.submitted += 1
        self.threads_seen.append(threads)
        self.texts_seen.append(list(texts))

    def poll(self, timeout=0.0):
        if self.job is None:
            return None
        if self.job[1] > 0:
            self.job[1] -= 1
            return None
        texts, self.job = self.job[0], None
        self.loaded = True
        return fake_embed(texts)

    def cancel(self):
        self.job = None
        self.cancelled += 1

    def unload(self):
        self.loaded = False
        self.unloads += 1

    def stop(self):
        self.stopped = True


def corpus(n, d, name="corpus.txt"):
    p = Path(d) / name
    p.write_text("".join(f"sample number {i} " + "x" * (i % 7) + "\n\n"
                         for i in range(n)))
    return str(p)


def test_index_and_checkpoint():
    green("indexer + checkpoint:")
    from deltaserve.adapter_routing.index import (
        Indexer,
        IndexingState,
        fingerprint,
        select_samples,
    )

    with tempfile.TemporaryDirectory() as d:
        path = corpus(53, d)
        texts = select_samples(path, 1000)
        check("one sample per non-empty line", len(texts) == 53)
        capped = select_samples(path, 20)
        check("capped selection is deterministic and without repeats",
              len(capped) == 20 == len(set(capped))
              and capped == select_samples(path, 20))
        fp = fingerprint(texts, "enc", "query: ", 512)
        check("fingerprint depends on data and encoder settings",
              fp != fingerprint(texts[:-1], "enc", "query: ", 512)
              and fp != fingerprint(texts, "enc2", "query: ", 512)
              and fp != fingerprint(texts, "enc", "query: ", 256))
        ref = reference_centroid(texts)

        ix = Indexer(texts, fp)
        while not ix.done:
            s, chunk = ix.next_chunk(7)
            ix.commit(s, fake_embed(chunk))
        check("centroid = normalised mean of normalised embeddings",
              np.allclose(ix.centroid(), ref, atol=1e-6))
        ix2 = Indexer(texts, fp)
        while not ix2.done:
            s, chunk = ix2.next_chunk(1 + ix2.state.cursor % 5)
            ix2.commit(s, fake_embed(chunk))
        check("chunking does not change the centroid",
              np.allclose(ix2.centroid(), ref, atol=1e-6))

        ck = Path(d) / "ck" / "job.npz"
        a = Indexer(texts, fp, ck)
        s0, c0 = a.next_chunk(10)
        check("next_chunk does not advance", a.next_chunk(10)[0] == s0 == 0)
        a.commit(s0, fake_embed(c0))
        s1, c1 = a.next_chunk(10)
        check("a late result of a dropped chunk is rejected",
              a.commit(s0, fake_embed(c0)) is False and a.state.cursor == 10)
        check("a result longer than the dataset is rejected",
              a.commit(s1, fake_embed(texts)) is False)
        check("unfinished job has no centroid", error_of(a.centroid) is not None)
        a.commit(s1, fake_embed(c1))
        check("checkpoint written after every chunk",
              IndexingState.load(ck).cursor == 20)

        b = Indexer(texts, fp, ck)                       # "restart"
        check("a restart resumes at the checkpoint",
              b.resumed_from == 20 and b.progress == (20, 53))
        while not b.done:
            s, chunk = b.next_chunk(9)
            b.commit(s, fake_embed(chunk))
        check("resumed job gives the same centroid",
              np.allclose(b.centroid(), ref, atol=1e-6))
        c = Indexer(texts[:-1], fingerprint(texts[:-1], "enc", "query: ", 512), ck)
        check("a checkpoint of another dataset is not resumed",
              c.resumed_from == 0 and c.state.cursor == 0)
        Path(ck).write_bytes(b"garbage")
        check("a corrupt checkpoint is ignored",
              Indexer(texts, fp, ck).state.cursor == 0)


def test_centroid_index():
    green("centroid index:")
    from deltaserve.adapter_routing.index import CentroidIndex

    idx = CentroidIndex("enc", "query: ", 512)
    check("empty index routes to nothing", idx.route(np.ones(DIM))[0] is None)
    e = np.eye(DIM, dtype=np.float32)
    idx.add("a", 3 * e[0])
    idx.add("b", e[1])
    idx.add("c", e[2])
    name, score = idx.route(0.9 * e[1] + 0.1 * e[0])
    check("cosine top-1", name == "b" and 0.99 < score <= 1.0)
    check("below the threshold → fall back (None), score still reported",
          idx.route(e[1] + e[0], threshold=0.9) == (None, idx.route(e[1] + e[0])[1]))
    check("at or above the threshold → routed",
          idx.route(e[1], threshold=0.9)[0] == "b")
    idx.add("a", e[3])
    check("adding an existing name replaces its centroid",
          len(idx) == 3 and idx.route(e[3])[0] == "a")
    check("dimension mismatch rejected",
          error_of(lambda: idx.add("z", np.ones(DIM + 1))) is not None)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "sub" / "centroids.npz"
        idx.save(p)
        back = CentroidIndex.load(p)
        check("save / load round-trip (names, vectors, encoder)",
              back.names == idx.names and back.encoder == "enc"
              and back.max_tokens == 512
              and np.allclose(back.scores(e[2]), idx.scores(e[2])))
    check("remove", idx.remove("b") and "b" not in idx
          and idx.route(e[1])[0] != "b" and not idx.remove("b"))


def test_cpu_helpers_and_config():
    green("CPU list + config:")
    from deltaserve.adapter_routing import encoder as enc
    from deltaserve.config import AdapterRoutingConfig

    check("cpu list parsing (now in adapter_routing.encoder)",
          enc.parse_cpu_list("8-11, 24") == {8, 9, 10, 11, 24}
          and enc.parse_cpu_list(None) is None and enc.parse_cpu_list("") is None)
    c = AdapterRoutingConfig()
    check("defaults: 8 CPU threads while idle, 2 while inference is present, "
          "GPU indexing on after 3 s of quiet",
          (c.indexing_cpu_max_threads, c.indexing_cpu_busy_threads,
           c.indexing_on_gpu, c.indexing_gpu_after_quiet_s) == (8, 2, True, 3.0))
    check("indexing_cpu_busy_threads must be >= 1",
          "indexing_cpu_busy_threads" in (error_of(
              lambda: AdapterRoutingConfig(indexing_cpu_busy_threads=0)) or ""))
    gone = ("indexing_cpu_reserve_cores", "indexing_min_free_memory_gb",
            "arrival_on_route", "route_batch_max", "route_batch_wait_ms",
            "route_timeout_ms")
    check("the governor's and the router service's settings are gone",
          all(not hasattr(c, n) and error_of(
              lambda n=n: AdapterRoutingConfig(**{n: 1})) is not None
              for n in gone))


def test_grant():
    green("indexing grant (one byte of shared memory):")
    from deltaserve.adapter_routing.index_process import GRANT_ENV, IndexingGrant

    g = IndexingGrant.create()
    try:
        check("created cleared, named after the creating process",
              not g.is_set() and str(os.getpid()) in g.name)
        check("the name travels to the indexing process in an env var",
              isinstance(GRANT_ENV, str) and GRANT_ENV)
        h = IndexingGrant.attach(g.name)
        check("attached by name: same segment, reads cleared",
              h.name == g.name and not h.is_set())
        g.set(True)
        check("set by the engine → seen by the other handle", h.is_set())
        g.set(False)
        check("cleared by the engine → seen by the other handle", not h.is_set())
        h.set(True)
        check("it is the same byte in both directions", g.is_set())
        h.close()
        check("closing a reader leaves the segment in place",
              g.is_set() and IndexingGrant.attach(g.name) is not None)
    finally:
        name = g.name
        g.close()
    check("closing the owner removes the segment",
          error_of(lambda: IndexingGrant.attach(name)) is not None)
    check("close is idempotent", error_of(g.close) is None)
    g2 = IndexingGrant.create()
    check("a new grant after the old one was closed starts cleared",
          not g2.is_set())
    g2.close()


def test_placement():
    green("placement: the scheduler's level, the grant, the thread count:")
    from deltaserve.adapter_routing.index_process import IndexingGrant
    from deltaserve.adapter_routing.runner import BUSY, IDLE, IndexingPlacement

    now = {"t": 100.0}
    grant = IndexingGrant.create()
    try:
        grant.set(True)                       # left over from somewhere
        p = IndexingPlacement(gpu_after_quiet_s=3.0, clock=lambda: now["t"],
                              grant=grant, idle_threads=8, busy_threads=2)
        check("nothing is granted before the first look",
              not grant.is_set() and p.grant is grant and p.changes == 0)
        check("never saw inference → idle, grant set",
              p.refresh() == IDLE and p.level == IDLE and grant.is_set())
        check("idle: the CPU indexer's full thread count", p.cpu_threads() == 8)
        check("idle: gpu when a GPU executor exists, cpu otherwise",
              p.choose(True) == "gpu" and p.choose(False) == "cpu")
        check("looking is not a change", p.changes == 0)

        p.note_inference()
        check("note_inference() → busy at once, grant cleared at once",
              p.level == BUSY and not grant.is_set() and p.changes == 1)
        check("busy: the reduced thread count, cpu even with a GPU executor",
              p.cpu_threads() == 2 and p.choose(True) == "cpu")
        now["t"] += 2.9
        check("quiet for less than the quiet period → still busy, not granted",
              p.refresh() == BUSY and p.choose(True) == "cpu"
              and not grant.is_set())
        p.note_inference()                    # a request still running
        now["t"] += 2.9
        check("inference seen again restarts the quiet period",
              p.refresh() == BUSY and not grant.is_set() and p.changes == 1)
        now["t"] += 0.2
        check("quiet long enough → idle again, grant set",
              p.choose(True) == "gpu" and p.level == IDLE and grant.is_set()
              and p.cpu_threads() == 8)
        for _ in range(5):
            p.refresh()
            p.cpu_threads()
            p.choose(True)
        check("changes counts level changes only (busy, idle = 2)",
              p.changes == 2)
        p.note_inference()
        p.note_inference()
        check("repeated note_inference() while busy is one change",
              p.changes == 3 and not grant.is_set())
        grant.set(True)                       # someone else wrote the byte
        p.note_inference()
        check("a grant that disagrees with the level is corrected without "
              "counting a change", not grant.is_set() and p.changes == 3)
    finally:
        grant.close()

    q = IndexingPlacement(3.0, clock=lambda: now["t"])
    q.note_inference()
    check("without a grant the level and thread counts work the same "
          "(defaults 8 / 2)",
          q.grant is None and q.level == BUSY and q.cpu_threads() == 2
          and q.choose(True) == "cpu")
    now["t"] += 3.0
    check("... and it returns to idle after the quiet period",
          q.cpu_threads() == 8 and q.choose(True) == "gpu" and q.changes == 2)


def test_runner():
    green("runner: pause, resume, thread count, migrate, quick kill:")
    from deltaserve.adapter_routing.index import Indexer, fingerprint, select_samples
    from deltaserve.adapter_routing.index_process import IndexingGrant, Preempted
    from deltaserve.adapter_routing.runner import BUSY, IndexingPlacement, IndexingRunner

    with tempfile.TemporaryDirectory() as d:
        texts = select_samples(corpus(40, d), 1000)
        fp = fingerprint(texts, "enc", "query: ", 512)
        ref = reference_centroid(texts)

        # ---- cpu_threads: the thread count of the next CPU chunk, 0 = wait
        cpu, threads = FakeExec("cpu", delay=1), {"n": 0}
        r = IndexingRunner(Indexer(texts, fp), {"cpu": cpu},
                           cpu_threads=lambda: threads["n"],
                           chunk_samples=8, recheck_s=0.25)
        check("cpu_threads() = 0 → no chunk, state 'paused' by the scheduler, "
              "come back later",
              r.step() == 0.25 and r.state == "paused"
              and r.pause_reason == "scheduler" and cpu.submitted == 0)
        check("still 0 → still paused", r.step() == 0.25 and cpu.submitted == 0)
        threads["n"] = 4
        r.step()
        check("threads again → chunk submitted with that many threads",
              r.state == "cpu" and r.pause_reason == ""
              and cpu.threads_seen == [4])
        r.step()
        check("chunk not ready yet → nothing committed", r.indexer.progress[0] == 0)
        threads["n"] = 0
        r.step()
        check("a chunk in flight is collected even after threads went to 0",
              r.indexer.progress[0] == 8)
        r.step()
        check("... and the next one waits",
              r.state == "paused" and cpu.submitted == 1
              and r.describe()["pause_reason"] == "scheduler")
        threads["n"] = 3
        for _ in range(100):
            if r.done:
                break
            r.step()
        check("CPU-only job completes with the reference centroid",
              r.done and cpu.threads_seen == [4] + [3] * 4
              and np.allclose(r.indexer.centroid(), ref, atol=1e-6))

        # ---- the default cpu_threads is the placement's level
        now = {"t": 100.0}
        cpu = FakeExec("cpu")
        place = IndexingPlacement(3.0, clock=lambda: now["t"],
                                  idle_threads=6, busy_threads=2)
        r = IndexingRunner(Indexer(texts, fp), {"cpu": cpu}, placement=place,
                           chunk_samples=8)
        check("cpu_threads defaults to the placement's", r.cpu_threads == place.cpu_threads)
        r.step(); r.step()                         # idle: submit, collect
        place.note_inference()
        r.step(); r.step()                         # busy
        r.step(); r.step()                         # still busy
        now["t"] += 10
        r.step(); r.step()                         # quiet again
        check("threads handed to the CPU executor follow the level "
              "(idle 6, busy 2, busy 2, idle 6)",
              cpu.threads_seen == [6, 2, 2, 6])

        # ---- cpu ↔ gpu with the grant
        grant = IndexingGrant.create()
        try:
            cpu, gpu = FakeExec("cpu", delay=1), FakeExec("gpu", delay=2)
            place = IndexingPlacement(3.0, clock=lambda: now["t"], grant=grant)
            r = IndexingRunner(Indexer(texts, fp, Path(d) / "m.npz"),
                               {"cpu": cpu, "gpu": gpu},
                               placement=place, chunk_samples={"cpu": 4, "gpu": 8})
            place.note_inference()
            r.step()                                   # cpu chunk submitted
            check("inference active → starts on the CPU with the busy thread "
                  "count, GPU not granted",
                  r.state == "cpu" and cpu.threads_seen == [2]
                  and not grant.is_set())
            now["t"] += 10                             # inference went quiet
            r.step()
            r.step()
            check("the CPU chunk in flight finishes after going quiet",
                  r.indexer.progress[0] == 4 and cpu.cancelled == 0)
            r.step()
            check("next chunk goes to the GPU, which is granted",
                  r.state == "gpu" and gpu.submitted == 1 and grant.is_set()
                  and gpu.threads_seen == [0])
            place.note_inference()                     # a request arrives
            check("request arrives → grant cleared before the runner even looks",
                  not grant.is_set())
            r.step()
            check("... and the GPU chunk is dropped at once, nothing committed",
                  gpu.cancelled == 1 and r.discarded == 1 and r.preempted == 0
                  and r.indexer.progress[0] == 4)
            r.step()
            check("the dropped chunk is redone on the CPU",
                  r.state == "cpu" and cpu.submitted == 2
                  and cpu.texts_seen[-1] == gpu.texts_seen[-1][:4])
            for i in range(400):
                if r.done:
                    break
                if i == 6:
                    now["t"] += 10                     # quiet again → back to GPU
                r.step()
            check("job that moved cpu → gpu → cpu → gpu still gives the reference "
                  f"centroid (chunks {r.chunks})",
                  r.done and r.chunks["gpu"] > 0 and r.chunks["cpu"] > 0
                  and np.allclose(r.indexer.centroid(), ref, atol=1e-6))

            # ---- the indexing process answers "preempted"
            class Preempting(FakeExec):
                """The chunk is preempted ``times`` times; ``arrival`` = the
                request that cleared the grant is noted during the poll (the
                runner had not seen it when it decided to wait for the GPU)."""

                def __init__(self, times, arrival=None):
                    super().__init__("gpu")
                    self.times, self.arrival = times, arrival

                def poll(self, timeout=0.0):
                    if self.job is not None and self.times > 0:
                        self.times -= 1
                        self.job = None
                        if self.arrival is not None:
                            self.arrival()
                        raise Preempted()
                    return super().poll(timeout)

            now["t"] += 10
            cpu, gpu = FakeExec("cpu"), Preempting(1)
            place = IndexingPlacement(3.0, clock=lambda: now["t"], grant=grant)
            r = IndexingRunner(Indexer(texts, fp), {"cpu": cpu, "gpu": gpu},
                               placement=place, chunk_samples={"cpu": 4, "gpu": 8})
            r.step()
            wait = r.step()
            check("a preempted GPU chunk: counted, dropped, nothing committed, "
                  "no error",
                  (r.preempted, r.discarded, r.indexer.progress[0]) == (1, 1, 0)
                  and wait == 0.0 and r.error is None)
            check("no GPU back-off after a preemption", r._gpu_off_until == 0.0)
            r.step()
            check("level still idle → the same samples go to the GPU again at once",
                  r.state == "gpu" and gpu.submitted == 2 and cpu.submitted == 0
                  and gpu.texts_seen[0] == gpu.texts_seen[1])
            r.step()
            check("... and are committed there", r.indexer.progress[0] == 8
                  and r.chunks == {"cpu": 0, "gpu": 1})

            cpu = FakeExec("cpu")
            place = IndexingPlacement(3.0, clock=lambda: now["t"], grant=grant)
            gpu = Preempting(1, arrival=place.note_inference)
            r = IndexingRunner(Indexer(texts, fp), {"cpu": cpu, "gpu": gpu},
                               placement=place, chunk_samples={"cpu": 4, "gpu": 8})
            r.step()
            r.step()                                   # preempted by an arrival
            r.step()
            check("preempted by an arrival (level busy) → the same samples are "
                  "handed out again on the CPU, with the busy thread count",
                  r.preempted == 1 and place.level == BUSY and r.state == "cpu"
                  and cpu.texts_seen == [gpu.texts_seen[0][:4]]
                  and cpu.threads_seen == [2] and gpu.submitted == 1)
            r.step()
            now["t"] += 10                             # quiet again
            r.step()
            check("quiet again → straight back to the GPU (no 30 s back-off)",
                  r.state == "gpu" and gpu.submitted == 2
                  and r._gpu_off_until == 0.0)
            for _ in range(400):
                if r.done:
                    break
                r.step()
            check("the preempted job gives the reference centroid; describe() "
                  "reports the preemption",
                  r.done and np.allclose(r.indexer.centroid(), ref, atol=1e-6)
                  and r.describe()["preempted"] == 1
                  and r.describe()["discarded"] == 1)

            gpu = Preempting(1)
            r = IndexingRunner(Indexer(texts, fp), {"gpu": gpu},
                               placement=IndexingPlacement(
                                   3.0, clock=lambda: now["t"]),
                               chunk_samples=8)
            r.step()
            check("without a CPU executor a GPU failure is not absorbed",
                  error_of(r.step) is not None)

            # ---- any other GPU failure still backs off to the CPU
            class Flaky(FakeExec):
                def poll(self, timeout=0.0):
                    self.job = None
                    raise RuntimeError("the indexing process did not answer")

            cpu, gpu = FakeExec("cpu"), Flaky("gpu")
            r = IndexingRunner(Indexer(texts, fp), {"cpu": cpu, "gpu": gpu},
                               placement=IndexingPlacement(
                                   3.0, clock=lambda: now["t"], grant=grant),
                               chunk_samples={"cpu": 4, "gpu": 8})
            r.step()
            r.step()
            check("a GPU chunk that fails otherwise: dropped, not a preemption, "
                  "GPU left alone for a while",
                  r.discarded == 1 and r.preempted == 0
                  and r._gpu_off_until > time.monotonic() + 20)
            r.step()
            check("... its samples go to the CPU although the level is idle",
                  r.state == "cpu" and grant.is_set()
                  and cpu.texts_seen == [gpu.texts_seen[0][:4]])
            for _ in range(400):
                if r.done:
                    break
                r.step()
            check("a failing GPU executor: the job finishes on the CPU",
                  r.done and r.error is None and gpu.submitted == 1
                  and r.chunks["gpu"] == 0 and r.chunks["cpu"] == 10
                  and np.allclose(r.indexer.centroid(), ref, atol=1e-6))

            class NotUp(FakeExec):
                ready = False

            cpu, gpu = FakeExec("cpu"), NotUp("gpu")
            r = IndexingRunner(Indexer(texts, fp), {"cpu": cpu, "gpu": gpu},
                               placement=IndexingPlacement(
                                   3.0, clock=lambda: now["t"], grant=grant),
                               chunk_samples={"cpu": 4, "gpu": 8})
            r.step(); r.step()
            gpu.ready = True
            r.step()
            check("a GPU executor that is not ready yet is not used; it is "
                  "once it is",
                  cpu.submitted == 1 and gpu.submitted == 1
                  and cpu.threads_seen == [8])
        finally:
            grant.close()

        cpu = FakeExec("cpu")
        r = IndexingRunner(Indexer(texts, fp), {"cpu": cpu}, chunk_samples=8)
        r.start()
        t0 = time.time()
        while not r.done and time.time() - t0 < 10:
            time.sleep(0.01)
        r.stop()
        check("thread mode runs to completion and stops its executors "
              "(no placement, no cpu_threads: threads = 0, the executor's default)",
              r.done and r.state == "done" and cpu.stopped and r.error is None
              and set(cpu.threads_seen) == {0})

        class Bad(FakeExec):
            def poll(self, timeout=0.0):
                raise RuntimeError("encoder died")

        r = IndexingRunner(Indexer(texts, fp), {"cpu": Bad("cpu")})
        r.start()
        t0 = time.time()
        while r.error is None and time.time() - t0 < 10:
            time.sleep(0.01)
        check("an executor failure is reported, not swallowed",
              r.error is not None and "encoder died" in r.error)


# ------------------------------------------------- the indexing process, no GPU
class FakeBatch(dict):
    def to(self, device):
        self.device = device
        return self


class FakeTok:
    """Character-level 'tokenizer': ids 1..7, 0 = padding."""

    def __init__(self):
        self.seen, self.kwargs = [], None

    def __call__(self, texts, **kwargs):
        import torch

        self.seen.append(list(texts))
        self.kwargs = kwargs
        rows = [[ord(c) % 7 + 1 for c in t][:kwargs["max_length"]] for t in texts]
        width = max(len(r) for r in rows)
        ids = torch.tensor([r + [0] * (width - len(r)) for r in rows])
        return FakeBatch(input_ids=ids, attention_mask=(ids != 0).long())


def _table():
    import torch

    return torch.randn(8, DIM, generator=torch.Generator().manual_seed(7))


class FakeModel:
    """'Encoder': a fixed vector per token id (padding included, so pooling
    must use the mask). ``before`` runs at the start of a forward — where the
    real process's per-layer hook looks at the grant."""

    def __init__(self):
        self.calls, self.before, self.delay = 0, None, 0.0

    def __call__(self, input_ids, attention_mask):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.before is not None:
            self.before()
        return SimpleNamespace(last_hidden_state=_table()[input_ids])


def model_embed(texts, prefix="q: ", max_tokens=32):
    """What FakeTok + FakeModel + mean pooling + normalisation must give."""
    table = _table().numpy().astype(np.float64)
    out = []
    for t in texts:
        ids = [ord(c) % 7 + 1 for c in prefix + t][:max_tokens]
        v = table[ids].mean(0)
        out.append(v / np.linalg.norm(v))
    return np.array(out, dtype=np.float32)


def test_index_process_encode():
    green("indexing process: one chunk (index_process._encode, on the CPU):")
    from deltaserve.adapter_routing import index_process as ip

    texts = ["short", "a much longer sample text", "mid length"]
    grant = ip.IndexingGrant.create()
    try:
        tok, model = FakeTok(), FakeModel()
        raised = False
        try:
            ip._encode(model, tok, None, "q: ", 32, "cpu", texts)
        except ip.Preempted:
            raised = True
        check("no grant from the engine → Preempted before the forward",
              raised and model.calls == 0)
        raised = False
        try:
            ip._encode(model, tok, grant, "q: ", 32, "cpu", texts)
        except ip.Preempted:
            raised = True
        check("grant cleared → Preempted before the forward",
              raised and model.calls == 0)
        grant.set(True)
        emb = ip._encode(model, tok, grant, "q: ", 32, "cpu", texts)
        check("grant set → one forward, [n, dim] float32 numpy",
              model.calls == 1 and isinstance(emb, np.ndarray)
              and emb.shape == (3, DIM) and emb.dtype == np.float32)
        check("the prefix is prepended; padded, truncated to max_tokens",
              tok.seen[-1] == ["q: " + t for t in texts]
              and tok.kwargs["max_length"] == 32 and tok.kwargs["padding"]
              and tok.kwargs["truncation"])
        check("masked mean pooling, unit length (padding does not count)",
              np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)
              and np.allclose(emb, model_embed(texts), atol=1e-5))
        short = ip._encode(model, tok, grant, "q: ", 6, "cpu", texts)
        check("max_tokens reaches the tokenizer",
              np.allclose(short, model_embed(texts, max_tokens=6), atol=1e-5))

        def clear_then_stop():                # the per-layer hook's behaviour
            grant.set(False)
            raise ip.Preempted()

        model.before = clear_then_stop
        raised = False
        try:
            ip._encode(model, tok, grant, "q: ", 32, "cpu", texts)
        except ip.Preempted:
            raised = True
        check("a grant cleared during the forward aborts the chunk", raised)
    finally:
        grant.close()


def test_index_process_server():
    green("indexing process: server + SharedEncoderExecutor over a unix socket "
          "(fake encoder, no GPU):")
    from deltaserve.adapter_routing import index_process as ip

    grant = ip.IndexingGrant.create()
    try:
        _server_checks(ip, grant)
    finally:
        grant.close()


def _server_checks(ip, grant):
    import threading
    from multiprocessing import Pipe
    from multiprocessing.connection import Listener

    from deltaserve.adapter_routing.index import Indexer, fingerprint, select_samples
    from deltaserve.adapter_routing.runner import BUSY, IndexingPlacement, IndexingRunner

    texts = ["first sample", "second, longer sample", "third"]
    tok, model = FakeTok(), FakeModel()
    stats = {"chunks": 0, "preempted": 0, "errors": 0}

    # ---- the protocol itself, on a pipe
    here, there = Pipe()
    th = threading.Thread(target=ip._serve, daemon=True, args=(
        there, model, tok, grant, "q: ", 32, "cpu", stats))
    th.start()

    def ask(msg):
        here.send(msg)
        return here.recv() if here.poll(10) else None

    check("ping → ('pong', stats)",
          ask(("ping",)) == ("pong", {"chunks": 0, "preempted": 0, "errors": 0}))
    reply = ask(("encode", texts))
    check("encode with the grant cleared → ('preempted',), no forward",
          reply == ("preempted",) and model.calls == 0)
    grant.set(True)
    reply = ask(("encode", texts))
    check("encode with the grant set → ('ok', embeddings)",
          reply[0] == "ok" and np.allclose(reply[1], model_embed(texts), atol=1e-5))

    def boom():
        raise RuntimeError("CUDA out of memory")

    model.before = boom
    reply = ask(("encode", texts))
    check("an exception in the encoder → ('error', 'Type: message'), "
          "the server keeps serving",
          reply == ("error", "RuntimeError: CUDA out of memory"))
    model.before = None
    reply = ask(("flush",))
    check("an unknown request → ('error', ...)",
          reply[0] == "error" and "flush" in reply[1])
    check("stats count chunks / preempted / errors",
          ask(("ping",)) == ("pong", {"chunks": 1, "preempted": 1, "errors": 1}))
    here.close()
    th.join(5)
    check("the engine hanging up ends that connection's server thread",
          not th.is_alive())

    # ---- the executor, against a listener like the process's
    with tempfile.TemporaryDirectory() as d:
        path = ip.socket_path(d)
        check("socket path: keyed by the index directory, short enough for a "
              "unix socket",
              path == ip.socket_path(d + "/")
              and path != ip.socket_path(d + "-other") and len(path) < 100)
        try:
            os.unlink(path)
        except OSError:
            pass
        ex = ip.SharedEncoderExecutor(path, timeout_s=10.0)
        check("not ready before the process listens; ping gives None",
              not ex.ready and ex.ping() is None)
        check("the executor is the GPU one and never holds the weights itself",
              ex.device == "gpu" and ex.loaded is True and ex.unload() is None)
        check("nothing submitted → poll gives None", ex.poll(0.0) is None)
        listener = Listener(path, family="AF_UNIX", authkey=ip._authkey(path))
        stats = {"chunks": 0, "preempted": 0, "errors": 0}

        def accept():
            while True:
                try:
                    conn = listener.accept()
                except Exception:  # noqa: BLE001 — listener closed
                    return
                threading.Thread(target=ip._serve, daemon=True, args=(
                    conn, model, tok, grant, "q: ", 32, "cpu", stats)).start()

        threading.Thread(target=accept, daemon=True).start()

        def collect(timeout=10.0):
            t0 = time.time()
            while time.time() - t0 < timeout:
                out = ex.poll(0.05)
                if out is not None:
                    return out
            return None

        try:
            check("ready once the socket exists; ping returns the stats",
                  ex.ready and ex.ping() == stats == {
                      "chunks": 0, "preempted": 0, "errors": 0})
            grant.set(True)
            check("encode() → the embeddings",
                  np.allclose(ex.encode(texts), model_embed(texts), atol=1e-5))
            ex.submit(texts, 0)
            emb = collect()
            check("submit / poll → the embeddings, then nothing more",
                  emb is not None and np.allclose(emb, model_embed(texts), atol=1e-5)
                  and ex.poll(0.0) is None)

            grant.set(False)
            calls = model.calls
            ex.submit(texts, 0)
            got = None
            try:
                collect()
            except ip.Preempted as e:
                got = e
            check("grant cleared → poll raises Preempted, no forward ran",
                  got is not None and model.calls == calls)
            check("encode() raises Preempted too",
                  type(_raised(lambda: ex.encode(texts))) is ip.Preempted)
            grant.set(True)

            model.before = boom
            ex.submit(texts, 0)
            err = _raised(collect)
            check("an encoder failure → RuntimeError carrying the process's message",
                  type(err) is RuntimeError
                  and str(err) == "RuntimeError: CUDA out of memory")
            model.before = None
            check("ping: chunks / preempted / errors as the process counted them",
                  ex.ping() == {"chunks": 2, "preempted": 2, "errors": 1})

            model.delay = 0.3
            ex.submit(texts, 0)
            ex.cancel()
            check("cancel → the chunk in flight is no longer waited for",
                  ex.poll(0.5) is None)
            model.delay = 0.0
            ex.submit(texts[:1], 0)
            emb = collect()
            check("the next chunk after a cancel gets its own answer",
                  emb is not None and emb.shape == (1, DIM)
                  and np.allclose(emb, model_embed(texts[:1]), atol=1e-5))

            # The runner on this executor: GPU while idle, preempted mid-chunk
            # by an arrival, the same samples on the CPU.
            class CpuExec(FakeExec):
                def poll(self, timeout=0.0):
                    if self.job is None:
                        return None
                    job, self.job = self.job[0], None
                    return model_embed(job)

            data = select_samples(corpus(24, d), 1000)
            fp = fingerprint(data, "enc", "q: ", 32)
            now = {"t": 0.0}
            place = IndexingPlacement(3.0, clock=lambda: now["t"], grant=grant)
            cpu = CpuExec("cpu")
            r = IndexingRunner(Indexer(data, fp), {"cpu": cpu, "gpu": ex},
                               placement=place, poll_s=5.0,
                               chunk_samples={"cpu": 4, "gpu": 8})
            r.step(); r.step()
            check("runner + executor: idle → a chunk encoded by the indexing "
                  "process", r.chunks == {"cpu": 0, "gpu": 1} and grant.is_set())

            def arrival():                    # a request arrives mid-forward
                place.note_inference()
                if not grant.is_set():
                    raise ip.Preempted()

            # The runner is already waiting for the chunk when it happens.
            model.before, model.delay = arrival, 0.2
            r.step(); r.step()
            model.before, model.delay = None, 0.0
            check("an arrival mid-chunk: the process answers preempted, the "
                  "runner counts it and has committed nothing of it",
                  r.preempted == 1 and r.indexer.progress[0] == 8
                  and place.level == BUSY and not grant.is_set()
                  and r._gpu_off_until == 0.0)
            r.step(); r.step()
            check("... and the same samples are encoded on the CPU",
                  r.chunks == {"cpu": 1, "gpu": 1}
                  and cpu.texts_seen[0] == data[8:12])
            now["t"] += 10
            for _ in range(200):
                if r.done:
                    break
                r.step()
            ref = model_embed(data).astype(np.float64).mean(0)
            check("the job finishes on the GPU after the quiet period, with "
                  f"the centroid of all samples (chunks {r.chunks})",
                  r.done and r.chunks["gpu"] >= 2 and r.chunks["cpu"] == 1
                  and np.allclose(r.indexer.centroid(),
                                  ref / np.linalg.norm(ref), atol=1e-5))

            ex.stop()
            check("stop() hangs up; the executor reconnects when used again",
                  ex._conn is None and ex.ping() is not None)
            ex.stop()
        finally:
            listener.close()
            try:
                os.unlink(path)
            except OSError:
                pass
        check("process gone: not ready, ping None, a chunk fails with an "
              "ordinary error (not Preempted: the runner backs off to the CPU)",
              not ex.ready and ex.ping() is None
              and type(_raised(lambda: ex.encode(texts))).__name__ != "Preempted"
              and _raised(lambda: ex.encode(texts)) is not None)

        # A process that accepts but never answers.
        mute = Listener(path, family="AF_UNIX", authkey=ip._authkey(path))
        conns = []
        threading.Thread(target=lambda: conns.append(mute.accept()),
                         daemon=True).start()
        slow = ip.SharedEncoderExecutor(path, timeout_s=0.3)
        try:
            t0 = time.time()
            err = _raised(lambda: slow.encode(texts))
            check("a process that does not answer → TimeoutError after "
                  "timeout_s, connection dropped",
                  type(err) is TimeoutError and 0.25 < time.time() - t0 < 5
                  and slow._conn is None)
        finally:
            slow.stop()
            mute.close()
            try:
                os.unlink(path)
            except OSError:
                pass


def _raised(fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        return e
    return None


def test_stage():
    green("the 'indexing' job stage:")
    from deltaserve.config import AdapterRoutingConfig
    from deltaserve.ft_job import FinetuneJob, FinetuneJobSpec

    spec0 = FinetuneJobSpec(stages=("init", "indexing"))
    err = error_of(spec0.validate)
    check("adapter routing off → 'indexing' refused, names the switch",
          err is not None and "enable_adapter_routing" in err)
    check("adapter routing on → accepted",
          error_of(lambda: spec0.validate(True)) is None)
    check("still refused with routing off after the stage was registered",
          error_of(spec0.validate) is not None)

    from deltaserve.adapter_routing import stages as st
    from deltaserve.adapter_routing.index import CentroidIndex, select_samples

    with tempfile.TemporaryDirectory() as d:
        data = corpus(30, d)
        cfg = AdapterRoutingConfig(
            enable_adapter_routing=True, index_path=str(Path(d) / "index"),
            indexing_chunk_samples=8, indexing_max_samples=25)
        fakes = []

        def executors(_cfg):
            fakes.append(FakeExec("cpu"))
            return {"cpu": fakes[-1]}

        keep = st.IndexingStage.make_executors
        st.IndexingStage.make_executors = staticmethod(executors)
        try:
            store = type("S", (), {"data_path": data, "current_epoch": 0,
                                   "total_epochs": 1,
                                   "has_next": lambda s: True,
                                   "has_claimed": lambda s: False,
                                   "__len__": lambda s: 30})()
            ctx = SimpleNamespace(
                store=store, adapter_routing_config=cfg,
                coordinator=SimpleNamespace(pending_backward=False,
                                            fill_count=0, reserved_fill=0))
            spec = FinetuneJobSpec(
                stages=("init", "training", "indexing", "ready"), data_path=data)
            spec.validate(True)
            job = FinetuneJob(spec=spec, name="dserve-routing-test")
            job.tick(ctx)
            check("training and indexing run side by side",
                  job.stage == "training+indexing")
            t0 = time.time()
            while job.stage_state["indexing"] != "done" and time.time() - t0 < 20:
                job.tick(ctx)
                time.sleep(0.02)
            check("indexing finishes while training is still running",
                  job.stage_state["indexing"] == "done" and job.stage == "training")
            path = Path(cfg.index_path) / st.INDEX_FILE
            idx = CentroidIndex.load(path)
            want = reference_centroid(select_samples(data, 25))
            check("centroid published under the job's name, capped to max_samples",
                  idx.names == ["dserve-routing-test"]
                  and np.allclose(idx._matrix[0], want, atol=1e-6))
            check("the index records the encoder that built it",
                  idx.encoder == cfg.encoder and idx.max_tokens == 512)
            check("job status shows indexing progress",
                  job.describe()["detail"]["indexing"]["encoded"] == 25)
            check("'ready' still waits for training",
                  job.stage_state["ready"] == "pending")

            cfg2 = AdapterRoutingConfig(
                enable_adapter_routing=True, index_path=cfg.index_path,
                encoder="other/encoder")
            ctx2 = SimpleNamespace(store=store, adapter_routing_config=cfg2,
                                   coordinator=ctx.coordinator)
            j2 = FinetuneJob(spec=FinetuneJobSpec(
                stages=("init", "indexing"), data_path=data), name="second")
            t0 = time.time()
            while not j2.finished and time.time() - t0 < 20:
                j2.tick(ctx2)
                time.sleep(0.02)
            check("mixing encoders in one index fails the job",
                  j2.failed and "different encoders" in (j2.error or ""))
        finally:
            st.IndexingStage.make_executors = keep


def test_real_encoder_on_cpu():
    if os.environ.get("DSERVE_TEST_REAL_ENCODER") != "1":
        green("real encoder on CPU: skipped (set DSERVE_TEST_REAL_ENCODER=1)")
        return
    green("real encoder in the CPU indexer process:")
    from deltaserve.adapter_routing.encoder import CpuIndexerProcess

    proc = CpuIndexerProcess("intfloat/e5-large-v2", "query: ", 512, cpus=None)
    try:
        texts = ["Translate to English: Guten Morgen", "What is 2 + 2?",
                 "Translate to English: Bonjour le monde"]
        proc.submit(texts, threads=4)
        t0, emb = time.time(), None
        while emb is None and time.time() - t0 < 300:
            emb = proc.poll(1.0)
        check("child answered", emb is not None)
        check("child runs at idle priority", proc.info is not None
              and proc.info[0] in ("SCHED_IDLE", "nice 19"))
        check("embeddings are [n, 1024] and unit length",
              emb.shape == (3, 1024)
              and np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4))
        sim = emb @ emb.T
        check("the two translation prompts are closest to each other",
              sim[0, 2] > sim[0, 1] and sim[0, 2] > sim[1, 2])
        proc.unload()
        proc.submit(texts[:1], threads=2)
        t0, again = time.time(), None
        while again is None and time.time() - t0 < 300:
            again = proc.poll(1.0)
        check("reloads after unload and reproduces the embedding",
              again is not None and np.allclose(again[0], emb[0], atol=1e-4))
    finally:
        proc.stop()


def main():
    green("=== adapter indexing ===")
    test_index_and_checkpoint()
    test_centroid_index()
    test_cpu_helpers_and_config()
    test_grant()
    test_placement()
    test_runner()
    test_index_process_encode()
    test_index_process_server()
    test_stage()
    test_real_encoder_on_cpu()
    print()
    if _failed:
        print(f"{_RED}{_failed} failed{_RESET}, {_passed} passed")
        sys.exit(1)
    green(f"all {_passed} checks passed")


if __name__ == "__main__":
    main()
