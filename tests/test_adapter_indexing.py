#!/usr/bin/env python
"""Adapter indexing: centroid index, resumable indexer, CPU governor, runner,
and the ``indexing`` job stage (deltaserve.adapter_routing).

CPU-only. The encoder is replaced by a deterministic fake, so no model is
loaded — except in the last, opt-in test:

    python tests/test_adapter_indexing.py
    DSERVE_TEST_REAL_ENCODER=1 python tests/test_adapter_indexing.py   # + real e5 on CPU

Checks
  * the centroid is the re-normalised mean of the normalised embeddings, however
    the job was chunked, interrupted, resumed or moved between executors;
  * a checkpoint resumes the same job and is ignored for a different one; a
    late result of a dropped chunk is rejected;
  * the centroid index routes by cosine top-1 and falls back below a threshold;
  * the CPU governor pauses on low memory / no idle cores and caps threads;
  * the runner pauses and resumes, moves cpu → gpu when inference is quiet and
    drops the GPU chunk the moment inference returns;
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
        self.threads_seen = []

    def submit(self, texts, threads):
        assert self.job is None, "chunk already in flight"
        self.job = [list(texts), self.delay]
        self.submitted += 1
        self.threads_seen.append(threads)

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


def test_governor():
    green("CPU governor:")
    from deltaserve.adapter_routing import governor as g

    check("cpu list parsing",
          g.parse_cpu_list("8-11, 24") == {8, 9, 10, 11, 24}
          and g.parse_cpu_list(None) is None and g.parse_cpu_list("") is None)

    GB = 1 << 30
    mem = {"v": 64 * GB}
    ticks = {"t": 0, "busy": set()}

    def cpu_probe():
        ticks["t"] += 100
        return {c: (ticks["t"], 0 if c in ticks["busy"] else ticks["t"])
                for c in range(16)}

    def gov(**kw):
        ticks["t"] = 0
        base = dict(max_threads=8, reserve_cores=2, min_free_memory_bytes=4 * GB,
                    need_memory_bytes=3 * GB, mem_probe=lambda: mem["v"],
                    cpu_probe=cpu_probe)
        base.update(kw)
        return g.CpuGovernor(**base)

    gv = gov()
    check("first call only measures", not gv.decide().run)
    ticks["busy"] = set()
    check("idle host → capped at max_threads", gv.decide().threads == 8)
    ticks["busy"] = set(range(11))
    d = gv.decide()
    check("5 idle, 2 reserved → 3 threads", d.threads == 3)
    ticks["busy"] = set(range(14))
    d = gv.decide()
    check("2 idle, 2 reserved → pause", not d.run and "idle" in d.reason)
    ticks["busy"] = set()
    pinned = gov(cpus={8, 9, 10, 11, 12})
    pinned.decide()
    check("only the allowed cores count", pinned.decide().threads == 3)

    gm = gov()
    gm.decide()
    mem["v"] = 6 * GB
    d = gm.decide(loaded=False)
    check("not loaded: needs reserve + encoder → pause on 6 GiB",
          not d.run and "memory" in d.reason)
    check("already loaded: only the reserve → runs on 6 GiB",
          gm.decide(loaded=True).run)
    mem["v"] = 3 * GB
    check("below the reserve → pause even when loaded",
          not gm.decide(loaded=True).run)
    mem["v"] = 64 * GB

    check("real /proc readers return sane values",
          g.read_mem_available_bytes() > 0 and len(g.read_cpu_times()) >= 1)


def test_runner():
    green("runner: pause, resume, migrate, quick kill:")
    from deltaserve.adapter_routing.index import Indexer, fingerprint, select_samples
    from deltaserve.adapter_routing.runner import IndexingPlacement, IndexingRunner

    class Gov:
        def __init__(self):
            self.script = []          # decisions to return, then "run 4 threads"

        def decide(self, loaded=False):
            self.loaded_seen = loaded
            if self.script:
                return self.script.pop(0)
            return SimpleNamespace(run=True, threads=4, reason="ok")

    def pause(reason):
        return SimpleNamespace(run=False, threads=0, reason=reason)

    with tempfile.TemporaryDirectory() as d:
        texts = select_samples(corpus(40, d), 1000)
        fp = fingerprint(texts, "enc", "query: ", 512)
        ref = reference_centroid(texts)

        cpu, gov = FakeExec("cpu", delay=1), Gov()
        r = IndexingRunner(Indexer(texts, fp), {"cpu": cpu}, governor=gov,
                           chunk_samples=8, recheck_s=0.25)
        gov.script = [pause("no idle cores: 1 idle, 2 reserved")]
        check("governor says pause → no chunk, come back later",
              r.step() == 0.25 and r.state == "paused" and cpu.submitted == 0)
        r.step()
        check("resources back → chunk submitted with the governor's threads",
              r.state == "cpu" and cpu.threads_seen == [4])
        r.step()
        check("chunk not ready yet → nothing committed", r.indexer.progress[0] == 0)
        r.step()
        check("chunk collected and committed", r.indexer.progress[0] == 8)
        cpu.loaded = True
        gov.script = [pause("low memory: 100 MiB available, need 4096 MiB")]
        r.step()
        check("memory pause unloads the encoder", r.state == "paused"
              and cpu.unloads == 1)
        for _ in range(100):
            if r.done:
                break
            r.step()
        check("CPU-only job completes with the reference centroid",
              r.done and np.allclose(r.indexer.centroid(), ref, atol=1e-6))

        now = {"t": 100.0}
        place = IndexingPlacement(gpu_after_quiet_s=3.0, clock=lambda: now["t"])
        check("no GPU executor → cpu", place.choose(False) == "cpu")
        check("never saw inference → gpu", place.choose(True) == "gpu")
        place.note_inference()
        check("inference just seen → cpu", place.choose(True) == "cpu")
        now["t"] += 2.9
        check("quiet for less than the threshold → still cpu",
              place.choose(True) == "cpu")
        now["t"] += 0.2
        check("quiet long enough → gpu", place.choose(True) == "gpu")

        cpu, gpu = FakeExec("cpu", delay=1), FakeExec("gpu", delay=2)
        place = IndexingPlacement(3.0, clock=lambda: now["t"])
        r = IndexingRunner(Indexer(texts, fp, Path(d) / "m.npz"),
                           {"cpu": cpu, "gpu": gpu}, governor=Gov(),
                           placement=place, chunk_samples={"cpu": 4, "gpu": 8})
        place.note_inference()
        r.step()                                   # cpu chunk submitted
        check("inference active → starts on the CPU", r.state == "cpu")
        now["t"] += 10                             # inference went quiet
        r.step()
        r.step()
        check("the CPU chunk in flight finishes after going quiet",
              r.indexer.progress[0] == 4 and cpu.cancelled == 0)
        r.step()
        check("next chunk goes to the GPU", r.state == "gpu" and gpu.submitted == 1)
        place.note_inference()                     # a request arrives
        r.step()
        check("request arrives → GPU chunk dropped at once, nothing committed",
              gpu.cancelled == 1 and r.discarded == 1
              and r.indexer.progress[0] == 4)
        r.step()
        check("the dropped chunk is redone on the CPU",
              r.state == "cpu" and cpu.submitted == 2)
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

        cpu = FakeExec("cpu")
        r = IndexingRunner(Indexer(texts, fp), {"cpu": cpu}, governor=Gov(),
                           chunk_samples=8)
        r.start()
        t0 = time.time()
        while not r.done and time.time() - t0 < 10:
            time.sleep(0.01)
        r.stop()
        check("thread mode runs to completion and stops its executors",
              r.done and r.state == "done" and cpu.stopped and r.error is None)

        class Bad(FakeExec):
            def poll(self, timeout=0.0):
                raise RuntimeError("encoder died")

        r = IndexingRunner(Indexer(texts, fp), {"cpu": Bad("cpu")}, governor=Gov())
        r.start()
        t0 = time.time()
        while r.error is None and time.time() - t0 < 10:
            time.sleep(0.01)
        check("an executor failure is reported, not swallowed",
              r.error is not None and "encoder died" in r.error)


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
            indexing_chunk_samples=8, indexing_cpu_reserve_cores=0,
            indexing_min_free_memory_gb=0.0, indexing_max_samples=25)
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
                encoder="other/encoder", indexing_cpu_reserve_cores=0,
                indexing_min_free_memory_gb=0.0)
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
    test_governor()
    test_runner()
    test_stage()
    test_real_encoder_on_cpu()
    print()
    if _failed:
        print(f"{_RED}{_failed} failed{_RESET}, {_passed} passed")
        sys.exit(1)
    green(f"all {_passed} checks passed")


if __name__ == "__main__":
    main()
