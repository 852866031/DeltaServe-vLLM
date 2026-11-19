#!/usr/bin/env python
"""Adapter handover: what happens when a finetuning job ends, without holding
the engine loop (deltaserve.adapter_save.JobSave, the backward child's shard
writer, the worker's adapter pre-read, the job stages' polling).

CPU-only, no model:

    python tests/test_adapter_handover.py

Checks
  * the backward child's shard writer: the file appears under its name only
    when complete; a failure leaves ``<path>.error`` with the reason;
  * ``JobSave``: ``start()`` only triggers (one worker call) and returns while
    the shards are still being written; the thread then merges the ranks,
    writes an adapter identical to a synchronous export, publishes it to the
    pool when the job is auto-routable, removes the shard directory;
  * a worker error at the trigger raises at once; a rank that cannot write its
    shard, or shards that never appear, fail the job through ``done()``;
  * ``start_job_save``: nothing to save / already saved are handled at once;
  * the job stages: 'ready' and 'admission' stay running until the handover
    is done and fail with its error;
  * the worker's switch: the adapter file is read once in the background
    (``deltaserve_prepare_ft_adapter``), the check and the switch share one
    copy of it, and a job that was never pre-read still switches.
"""

import os
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

import torch

_passed = 0
_failed = 0
_TTY = sys.stdout.isatty()
_GREEN = "\033[92m" if _TTY else ""
_RED = "\033[91m" if _TTY else ""
_RESET = "\033[0m" if _TTY else ""


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  {_GREEN}[PASS]{_RESET} {name}")
    else:
        _failed += 1
        print(f"  {_RED}[FAIL]{_RESET} {name}")
    return cond


def key(layer, proj, factor):
    return (f"base_model.model.model.layers.{layer}.self_attn.{proj}_proj."
            f"lora_{factor}.weight")


def make_adapter(directory, seed=0, layers=2, r=4, hidden=8):
    """A tiny PEFT adapter directory; returns its full state."""
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(seed)
    state = {}
    for i in range(layers):
        for proj in "qkvo":
            state[key(i, proj, "A")] = torch.randn(r, hidden, generator=g)
            state[key(i, proj, "B")] = torch.randn(hidden, r, generator=g)
    os.makedirs(directory, exist_ok=True)
    save_file(state, os.path.join(directory, "adapter_model.safetensors"))
    with open(os.path.join(directory, "adapter_config.json"), "w") as f:
        f.write('{"r": 4}')
    return state


def rank_shards(state, tp=2):
    """The per-rank shards of ``state`` (the inverse of merge_rank_shards)."""
    from deltaserve.adapter_save import _SHARD_DIM, LORA_KEY_RE

    ranks = [{} for _ in range(tp)]
    for k, v in state.items():
        m = LORA_KEY_RE.search(k)
        dim = _SHARD_DIM.get((m.group(2), m.group(3)))
        for r in range(tp):
            ranks[r][k] = v.chunk(tp, dim=dim)[r].clone() if dim is not None \
                else v.clone()
    return ranks


class FakeExecutor:
    """Answers the save trigger as two TP ranks would: each rank's 'backward
    child' writes its shard through the real writer."""

    def __init__(self, shards, delay=0.0, fail_rank=None, mute=False,
                 error=None):
        self.shards, self.delay = shards, delay
        self.fail_rank, self.mute, self.error = fail_rank, mute, error
        self.calls = []

    def collective_rpc(self, method, args=()):
        from deltaserve.bwd_services.base import _write_shard_async

        self.calls.append(method)
        if self.error:
            return [{"error": self.error}, (1, "x")]
        directory = args[0]
        out = []
        for rank, shard in enumerate(self.shards):
            path = os.path.join(directory, f"shard-rank{rank}.safetensors")
            if not self.mute:
                target = "/proc/nope/shard" if rank == self.fail_rank else path
                if rank == self.fail_rank:
                    # the writer reports under the path the engine watches
                    def bad(path=path):
                        with open(path + ".error", "w") as f:
                            f.write("OSError: disk full")
                    threading.Timer(self.delay, bad).start()
                else:
                    threading.Timer(self.delay, _write_shard_async,
                                    (shard, target)).start()
            out.append((rank, path))
        return list(reversed(out))          # replies in any order


def job(src, name="job-a", auto=False, save_path=None):
    return SimpleNamespace(
        name=name, saved_path=None, admission=None,
        spec=SimpleNamespace(adapter_path=src, save_path=save_path,
                             auto_routing=auto))


def wait(save, timeout=10.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            if save.done():
                return None
        except RuntimeError as e:
            return str(e)
        time.sleep(0.01)
    return "timeout"


def test_shard_writer():
    print("The backward child's shard writer")
    from safetensors.torch import load_file

    from deltaserve.bwd_services.base import _write_shard_async

    with tempfile.TemporaryDirectory() as d:
        state = {"a": torch.arange(6.0).reshape(2, 3)}
        path = os.path.join(d, "sub", "shard.safetensors")
        _write_shard_async(state, path)
        t0 = time.monotonic()
        while not os.path.exists(path) and time.monotonic() - t0 < 5:
            time.sleep(0.01)
        check("the shard is written (directory created), fp32, same values",
              os.path.exists(path) and not os.path.exists(path + ".tmp")
              and torch.equal(load_file(path)["a"], state["a"]))
        bad = os.path.join("/proc", "nope", "shard.safetensors")
        _write_shard_async(state, bad)
        time.sleep(0.3)
        check("a shard that cannot be written never appears under its name",
              not os.path.exists(bad))


def test_job_save():
    print("JobSave: trigger on the engine loop, the rest on a thread")
    from safetensors.torch import load_file

    from deltaserve.adapter_save import JobSave, merge_rank_shards, start_job_save

    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "src")
        trained = {k: v + 1.0 for k, v in make_adapter(src).items()}
        shards = rank_shards(trained)
        published = []
        routing = SimpleNamespace(
            adapter_dir=lambda name: os.path.join(d, "pool", name),
            publish=lambda j: published.append((j.name, j.saved_path)))

        ex = FakeExecutor(shards, delay=0.3)
        j = job(src, auto=True)
        t0 = time.monotonic()
        save = start_job_save(j, SimpleNamespace(model_executor=ex), routing)
        held = time.monotonic() - t0
        check("start() returns while the shards are still being written "
              f"({held * 1e3:.0f} ms; they take 300 ms)",
              isinstance(save, JobSave) and held < 0.2 and not save.finished
              and j.saved_path is None)
        check("it made exactly one worker call",
              ex.calls == ["deltaserve_save_ft_adapter"])
        check("not done yet", save.done() is False)
        err = wait(save)
        dst = os.path.join(d, "pool", "job-a")
        check("then the job is saved where the pool keeps its adapters",
              err is None and j.saved_path == dst and os.path.isdir(dst))
        saved = load_file(os.path.join(dst, "adapter_model.safetensors"))
        check("the saved adapter is the merged ranks — the trained weights, "
              "every tensor",
              set(saved) == set(trained)
              and all(torch.equal(saved[k], trained[k]) for k in trained)
              and all(torch.equal(saved[k], merge_rank_shards(shards)[k])
                      for k in trained))
        check("the source adapter's side files are carried over",
              os.path.exists(os.path.join(dst, "adapter_config.json")))
        check("an auto-routable job is published to the pool, after the save",
              published == [("job-a", dst)])
        check("the shard directory is gone",
              not os.path.exists(save.shard_dir))
        check("trigger and total times are recorded",
              save.trigger_ms is not None and save.total_ms >= save.trigger_ms)

        published.clear()
        again = start_job_save(j, SimpleNamespace(model_executor=ex), routing)
        check("already saved: nothing to do, and published once only when the "
              "pool has not admitted it", again is None and published
              == [("job-a", dst)])
        j.admission = {"level": "active"}
        published.clear()
        check("admitted already: not published again",
              start_job_save(j, None, routing) is None and not published)

        j2 = job(src, name="job-b", save_path=os.path.join(d, "out"))
        save = start_job_save(j2, SimpleNamespace(
            model_executor=FakeExecutor(shards)), None)
        check("no adapter routing: saved to ft_job.save_path, not published",
              wait(save) is None and j2.saved_path == os.path.join(d, "out"))
        j3 = job(src, name="job-c")
        check("nowhere to save: reported, nothing started",
              start_job_save(j3, SimpleNamespace(model_executor=ex), None) is None
              and j3.saved_path is None)

        j4 = job(src, name="job-d", save_path=os.path.join(d, "out4"))
        try:
            start_job_save(j4, SimpleNamespace(model_executor=FakeExecutor(
                shards, error="RuntimeError: no backward child")), None)
            raised = ""
        except RuntimeError as e:
            raised = str(e)
        check("a worker error at the trigger fails the job at once",
              "no backward child" in raised)

        j5 = job(src, name="job-e", save_path=os.path.join(d, "out5"))
        save = start_job_save(j5, SimpleNamespace(model_executor=FakeExecutor(
            shards, delay=0.05, fail_rank=1)), None)
        err = wait(save)
        check("a rank that cannot write its shard fails the job with the reason",
              err is not None and "disk full" in err and j5.saved_path is None
              and not os.path.exists(os.path.join(d, "out5")))

        j6 = job(src, name="job-f", save_path=os.path.join(d, "out6"))
        JobSave.SHARD_WAIT_S, old = 0.2, JobSave.SHARD_WAIT_S
        try:
            save = start_job_save(j6, SimpleNamespace(
                model_executor=FakeExecutor(shards, mute=True)), None)
            err = wait(save)
        finally:
            JobSave.SHARD_WAIT_S = old
        check("shards that never appear fail the job instead of hanging it",
              err is not None and "did not appear" in err)


def test_stages():
    print("Job stages wait for the handover")
    from deltaserve.ft_job import ReadyStage

    state = {"done": False, "error": None, "started": 0}

    def on_ready(j):
        state["started"] += 1

    def handover_done(j):
        if state["error"]:
            raise RuntimeError(state["error"])
        return state["done"]

    ctx = SimpleNamespace(on_job_ready=on_ready, job_handover_done=handover_done)
    stage = ReadyStage()
    stage.start(None, ctx)
    check("'ready' starts the handover once and stays running",
          state["started"] == 1 and stage.poll(None, ctx) is False)
    state["done"] = True
    check("and is done when the handover is", stage.poll(None, ctx) is True)
    state["error"] = "RuntimeError: disk full"
    try:
        stage.poll(None, ctx)
        raised = False
    except RuntimeError:
        raised = True
    check("a failed handover fails the stage", raised)
    check("a server without the hook: 'ready' is done at once",
          ReadyStage().poll(None, SimpleNamespace()) is True)
    try:
        from deltaserve.adapter_routing.stages import AdmissionStage

        state.update(done=False, error=None)
        adm = AdmissionStage()
        ctx2 = SimpleNamespace(admit_job=on_ready, job_handover_done=handover_done)
        adm.start(None, ctx2)
        first = adm.poll(None, ctx2)
        state["done"] = True
        check("'admission' waits for the handover in the same way",
              first is False and adm.poll(None, ctx2) is True)
        check("a job that did not train (no hook) is admitted at once",
              AdmissionStage().poll(None, SimpleNamespace()) is True)
    except ImportError as e:      # pragma: no cover
        check(f"admission stage importable ({e})", False)


def test_worker_switch():
    print("Worker: the next job's adapter is read once, in the background")
    import safetensors.torch as st

    from deltaserve.worker import DeltaServeWorker

    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "next")
        state = make_adapter(src, seed=3)
        with open(os.path.join(src, "adapter_config.json"), "w") as f:
            f.write('{"r": 4, "lora_alpha": 8}')
        reads, real = [], st.load_file

        def counting(path, *a, **k):
            reads.append(threading.current_thread().name)
            return real(path, *a, **k)

        st.load_file = counting
        try:
            sent = []

            def switch_adapter(ft_state, hparams):
                sent.append((ft_state, dict(hparams)))
                return {"event": "adapter_switched"}

            w = SimpleNamespace(device=torch.device("cpu"), backward_process=
                                SimpleNamespace(switch_adapter=switch_adapter))
            prep = DeltaServeWorker.deltaserve_prepare_ft_adapter
            switch = DeltaServeWorker.deltaserve_switch_ft_adapter
            check("prepare returns at once", prep(w, src) is True)
            prep(w, src)
            time.sleep(0.2)
            check("the file is read once, on a background thread",
                  reads == ["deltaserve-ft-adapter-read"])
            r1 = switch(w, src, {"dry_run": True, "learning_rate": 1e-4})
            r2 = switch(w, src, {"learning_rate": 1e-4})
            check("check + switch succeed without reading the file again",
                  r1 == {"event": "adapter_switched"} == r2 and len(reads) == 1)
            check("both calls hand the child the SAME tensors (one copy on the "
                  "device), fp32, with the adapter's scaling",
                  all(sent[0][0][k] is sent[1][0][k] for k in state)
                  and all(torch.equal(sent[1][0][k], state[k]) for k in state)
                  and sent[0][1]["dry_run"] is True
                  and "dry_run" not in sent[1][1]
                  and "lora_scaling" in sent[1][1])
            check("nothing is kept after the switch",
                  w._ft_switch_state is None and src not in w._ft_adapter_reads)
            reads.clear()
            w2 = SimpleNamespace(device=torch.device("cpu"), backward_process=
                                 SimpleNamespace(switch_adapter=switch_adapter))
            ok = switch(w2, src, {"dry_run": True})
            switch(w2, src, {})
            check("a job that was never pre-read still switches (one read, by "
                  "the call itself)", ok == {"event": "adapter_switched"}
                  and len(reads) == 1 and reads[0] != "deltaserve-ft-adapter-read")
            bad = switch(w2, os.path.join(d, "missing"), {"dry_run": True})
            check("a missing adapter is reported, never raised",
                  isinstance(bad, dict) and "error" in bad
                  and w2._ft_switch_state is None)
            w3 = SimpleNamespace(device=torch.device("cpu"))
            check("no backward child on the worker: reported",
                  "error" in switch(w3, src, {}))
        finally:
            st.load_file = real


def main():
    test_shard_writer()
    test_job_save()
    test_stages()
    test_worker_switch()
    print()
    if _failed:
        print(f"{_RED}{_failed} failed{_RESET}, {_passed} passed")
        sys.exit(1)
    print(f"{_GREEN}all {_passed} checks passed{_RESET}")


if __name__ == "__main__":
    main()
