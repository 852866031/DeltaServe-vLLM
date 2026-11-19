#!/usr/bin/env python
"""Finetuning jobs and stages (deltaserve.ft_job).

CPU-only, no model: the job layer is pure Python, driven here by the real
FinetuningStore and a stand-in for the coordinator fields the stages read.

    python tests/test_ft_job.py

Checks
  * spec parsing / validation: default stages, unknown and gated stages,
    missing requirements, adapter_init, auto_routing without adapter routing;
  * stages run when their requirements are done, side by side when they only
    share a prerequisite, and a failing stage fails the job without raising;
  * the training stage turns done exactly when the real store has trained its
    last epoch and the buffer / backward are settled;
  * one job at a time, the rest queue; adapter names;
  * the ``ft_job`` YAML section: flat keys set the FinetuneConfig fields the
    engine already reads, the rest becomes metadata, and a YAML without the
    section produces the launch arguments it always did.
"""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parent.parent
_CONFIG = _REPO / "configs" / "preset" / "serving_config_finetuning_qwen3_0.6b.yaml"

_passed = 0
_failed = 0
_TTY = sys.stdout.isatty()
_GREEN = "\033[92m" if _TTY else ""
_RED = "\033[91m" if _TTY else ""
_RESET = "\033[0m" if _TTY else ""


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


def ft_cfg(**kw):
    base = dict(finetuning_lora_path="/a", data_path="/d", num_epochs=2,
                learning_rate=1e-4, weight_decay=0.0, gamma=1.0, ft_job=None)
    base.update(kw)
    return SimpleNamespace(**base)


def coord(**kw):
    base = dict(pending_backward=False, fill_count=0, reserved_fill=0)
    base.update(kw)
    return SimpleNamespace(**base)


def test_spec():
    green("FinetuneJobSpec:")
    from deltaserve.ft_job import FinetuneJobSpec

    spec = FinetuneJobSpec.from_finetune_config(ft_cfg())
    check("no ft_job → implicit job, stages [init, training]",
          spec.stages == ("init", "training") and spec.name is None
          and not spec.auto_routing)
    check("flat fields carried into the spec",
          spec.adapter_path == "/a" and spec.data_path == "/d"
          and spec.num_epochs == 2)
    check("implicit job is valid", error_of(spec.validate) is None)

    def spec_of(**job):
        return FinetuneJobSpec.from_finetune_config(ft_cfg(ft_job=job))

    check("stages from the job config",
          spec_of(stages=["init", "training", "ready"]).stages
          == ("init", "training", "ready"))
    check("unknown ft_job key rejected",
          "unknown ft_job key" in (error_of(lambda: spec_of(bogus=1)) or ""))
    check("stages must be a list",
          error_of(lambda: spec_of(stages="init")) is not None)
    err = error_of(spec_of(stages=["init", "trainng"]).validate)
    check("unknown stage rejected, lists the available ones",
          err is not None and "trainng" in err and "training" in err)
    err = error_of(spec_of(stages=["init", "indexing"]).validate)
    check("'indexing' with adapter routing off names the switch",
          err is not None and "enable_adapter_routing" in err)
    check("missing 'init' rejected",
          error_of(spec_of(stages=["training"]).validate) is not None)
    check("duplicate stage rejected",
          error_of(spec_of(stages=["init", "init"]).validate) is not None)
    check("unsupported adapter_init rejected",
          "adapter_init" in (error_of(spec_of(adapter_init="random").validate) or ""))
    err = error_of(spec_of(auto_routing=True).validate)
    check("auto_routing with adapter routing off names the switch",
          err is not None and "enable_adapter_routing" in err)
    check("auto_routing accepted when adapter routing is on",
          error_of(lambda: spec_of(auto_routing=True).validate(True)) is None)


def test_stage_graph():
    green("stage scheduling:")
    from deltaserve import ft_job as fj

    log = []

    class _Rec(fj.Stage):
        done_after = 1

        def requirements(self, names):
            return ("init",)

        def start(self, job, ctx):
            log.append(("start", self.name))
            self.polls = 0

        def poll(self, job, ctx):
            self.polls += 1
            return self.polls >= self.done_after

    class A(_Rec):
        name = "t_a"
        done_after = 3

    class B(_Rec):
        name = "t_b"
        done_after = 1

    class Boom(_Rec):
        name = "t_boom"

        def poll(self, job, ctx):
            raise RuntimeError("kaput")

    for c in (A, B, Boom):
        fj.register_stage(c)
    try:
        ctx = SimpleNamespace(store=SimpleNamespace(data_path=None),
                              coordinator=coord())
        spec = fj.FinetuneJobSpec(stages=("init", "t_a", "t_b", "ready"))
        spec.validate()
        job = fj.FinetuneJob(spec=spec, name="j")
        check("new job is pending", job.stage == "pending")
        job.tick(ctx)
        check("init done, both dependants started in the same tick",
              job.stage_state["init"] == "done"
              and [n for _, n in log] == ["t_a", "t_b"])
        check("stages sharing a prerequisite overlap; the short one is done",
              job.stage_state["t_b"] == "done" and job.stage == "t_a")
        check("'ready' waits for every other stage",
              job.stage_state["ready"] == "pending")
        job.tick(ctx)
        check("still running after 2 polls", job.stage == "t_a")
        job.tick(ctx)
        check("all done → job finished, stage 'done'",
              job.finished and job.stage == "done"
              and job.stage_state["ready"] == "done")
        check("finished job ticks are no-ops", job.tick(ctx) is False)

        bad = fj.FinetuneJob(spec=fj.FinetuneJobSpec(
            stages=("init", "t_boom", "ready")), name="b")
        raised = error_of(lambda: bad.tick(ctx))
        check("a failing stage fails the job, does not raise",
              raised is None and bad.failed and bad.stage == "failed"
              and "kaput" in bad.error)
        check("stages after the failure never start",
              bad.stage_state["ready"] == "pending")
    finally:
        for c in (A, B, Boom):
            fj._STAGES.pop(c.name, None)


def test_training_stage_with_real_store():
    green("training stage against the real FinetuningStore:")
    from deltaserve.finetuning_store import FinetuningStore
    from deltaserve.ft_job import FinetuneJob, FinetuneJobSpec

    with tempfile.TemporaryDirectory() as d:
        data = Path(d) / "corpus.txt"
        data.write_text("a b c\nd e\nf g h i\n")
        store = FinetuningStore(str(data), tokenize=lambda t: t.split(),
                                total_epochs=2, max_saved_finetuning_tokens=16)
        store.load()
        c = coord()
        ctx = SimpleNamespace(store=store, coordinator=c)
        job = FinetuneJob(spec=FinetuneJobSpec(
            stages=("init", "training", "ready")), name="j")
        job.tick(ctx)
        check("init done, training running", job.stage == "training")

        def run_epoch(last: bool):
            """Admit every sample of the epoch, then 'train' them."""
            taken = []
            while store.has_next():
                s = store.pop_best_under(16, exclude=taken)
                taken.append(s)
                store.claim([s])
            c.fill_count, c.pending_backward = 9, True
            job.tick(ctx)
            ok = job.stage == "training"      # samples in flight → not done
            c.fill_count, c.pending_backward = 0, False
            if last:
                job.tick(ctx)
                ok = ok and job.stage == "training"   # still claimed
            store.commit_claimed(taken)
            return ok

        check("epoch 1 in flight → still training", run_epoch(last=False))
        job.tick(ctx)
        check("epoch 1 trained, one epoch left → still training",
              job.stage == "training")
        check("store moves to the last epoch", store.advance_epoch()
              and store.current_epoch == store.total_epochs - 1)
        check("last epoch claimed but not committed → still training",
              run_epoch(last=True))
        job.tick(ctx)
        check("last epoch committed, buffer empty, no backward → job done",
              job.finished and job.stage == "done")

        empty = Path(d) / "empty.txt"
        empty.write_text("\n")
        store2 = FinetuningStore(str(empty), tokenize=lambda t: t.split())
        store2.load()
        j2 = FinetuneJob(spec=FinetuneJobSpec(), name="e")
        j2.tick(SimpleNamespace(store=store2, coordinator=coord()))
        check("empty dataset fails the init stage",
              j2.failed and "no usable sample" in j2.error)


def test_manager_and_names():
    green("FtJobManager + adapter names:")
    import re

    from deltaserve.ft_job import FinetuneJobSpec, FtJobManager, generate_adapter_name

    check("random name, not auto-routable: dserve-<8 hex>",
          re.fullmatch(r"dserve-[0-9a-f]{8}", generate_adapter_name(False)))
    check("random name, auto-routable: dserve-routing-<8 hex>",
          re.fullmatch(r"dserve-routing-[0-9a-f]{8}", generate_adapter_name(True)))

    ctx = SimpleNamespace(store=SimpleNamespace(data_path=None),
                          coordinator=coord())
    mgr = FtJobManager()
    a = mgr.submit(FinetuneJobSpec(name="mine"))
    b = mgr.submit(FinetuneJobSpec())
    check("user-given name kept as is", a.name == "mine")
    check("unnamed job gets a generated name", b.name.startswith("dserve-"))
    check("first job runs, second queues",
          mgr.current is a and mgr.queue == [b])
    check("duplicate job name rejected",
          error_of(lambda: mgr.submit(FinetuneJobSpec(name="mine"))) is not None)
    check("invalid job rejected at submit",
          error_of(lambda: mgr.submit(FinetuneJobSpec(stages=("training",))))
          is not None)
    mgr.tick(ctx)
    check("finished job hands over to the queued one",
          mgr.current is b and mgr.finished == [a] and not mgr.queue)
    mgr.tick(ctx)
    st = mgr.status()
    check("status lists every job with its stages",
          st["current"] is None and [j["name"] for j in st["finished"]]
          == ["mine", b.name]
          and st["finished"][0]["stages"] == {"init": "done", "training": "done"})
    check("tick with no job is a no-op", error_of(lambda: mgr.tick(ctx)) is None)
    routed = FtJobManager(adapter_routing=True).submit(
        FinetuneJobSpec(auto_routing=True))
    check("auto-routable job gets a dserve-routing- name",
          routed.name.startswith("dserve-routing-"))


def test_online_jobs():
    green("jobs submitted while serving, run one after another:")
    import os
    import time

    from deltaserve.finetuning_store import FinetuningStore
    from deltaserve.ft_job import FinetuneJobSpec, FtJobManager
    from deltaserve.ft_scheduler import FinetuneScheduler as FS

    with tempfile.TemporaryDirectory() as d:
        def adapter(name):
            p = Path(d) / name
            p.mkdir()
            (p / "adapter_model.safetensors").write_bytes(b"x")
            return str(p)

        def corpus(name, n):
            p = Path(d) / name
            p.write_text("".join(f"w{i} a b\n" for i in range(n)))
            return str(p)

        launch_data = corpus("launch.txt", 3)
        store0 = FinetuningStore(launch_data, tokenize=lambda t: t.split(),
                                 total_epochs=1, max_saved_finetuning_tokens=16)
        store0.load()
        rpc = []           # the switch calls
        prepared = []      # [adapter handover] the background pre-read of a
        #                    queued job's adapter file (not a switch)

        class Sched:
            """What the job code needs of a FinetuneScheduler."""
            submit_ft_job = FS.submit_ft_job
            begin_job = FS.begin_job
            job_ready_to_train = FS.job_ready_to_train
            # This engine has no direct line to the backward children, so the
            # switch goes through the workers (tests/test_adapter_handover.py
            # covers the direct line).
            _handover_line = staticmethod(FS._handover_line)
            _adopt_job = FS._adopt_job

            def __init__(self):
                self.vllm_config = SimpleNamespace(finetune_config=SimpleNamespace(
                    finetuning_lora_path=adapter("launch-adapter"), num_epochs=1,
                    learning_rate=1e-4, weight_decay=0.0, gamma=1.0,
                    max_saved_finetuning_tokens=16, max_prepare=None))
                self._ft_jobs = FtJobManager()
                self._ft_injector = SimpleNamespace(store=store0)
                self._coord = coord(current_epoch=5, corpus_total_tokens=0,
                                    min_sample_len=1, admission_open=False,
                                    epoch_flush_pending=True)
                self._job_switch = None
                self._ft_exhausted_logged = True
                self._ds_engine = SimpleNamespace(model_executor=SimpleNamespace(
                    collective_rpc=lambda m, args=(), **k: (
                        prepared if m == "deltaserve_prepare_ft_adapter"
                        else rpc).append((m, args))))
                self._launch_job = self._ft_jobs.submit(FinetuneJobSpec(
                    name="launch", adapter_path="/a", data_path=launch_data)).name

            def _build_tokenize(self):
                return lambda t: t.split()

            store = property(lambda self: self._ft_injector.store)
            coordinator = property(lambda self: self._coord)

        s = Sched()
        mgr = s._ft_jobs
        mgr.tick(s)
        check("the launch job starts without a switch", mgr.current.stage == "training"
              and rpc == [])

        a2, d2 = adapter("job2"), corpus("two.txt", 5)
        j2 = s.submit_ft_job({"adapter_path": a2, "data_path": d2, "name": "second",
                              "num_epochs": 2, "learning_rate": 3e-4})
        check("POST /ft_jobs queues a job behind the running one",
              j2["name"] == "second" and [j.name for j in mgr.queue] == ["second"]
              and j2["num_epochs"] == 2 and mgr.busy)
        check("unknown keys, missing weights, missing data, no training stage are "
              "refused at submission",
              all(error_of(lambda spec=spec: s.submit_ft_job(spec)) is not None
                  for spec in ({"adapter_path": a2, "data_path": d2, "bogus": 1},
                               {"adapter_path": str(Path(d) / "nope"), "data_path": d2},
                               {"adapter_path": a2, "data_path": str(Path(d) / "no.txt")},
                               {"adapter_path": a2, "data_path": d2,
                                "stages": ["init", "ready"]})))

        mgr.tick(s)
        check("the queued job waits while the first is training",
              mgr.current.name == "launch" and rpc == [])
        taken = []
        while store0.has_next():                       # train the launch job out
            x = store0.pop_best_under(16, exclude=taken)
            taken.append(x)
            store0.claim([x])
        store0.commit_claimed(taken)
        mgr.tick(s)
        check("first job done → the second becomes current",
              mgr.current is not None and mgr.current.name == "second"
              and [j.name for j in mgr.finished] == ["launch"])

        s._coord.pending_backward = True               # previous job still settling
        mgr.tick(s)                                    # init starts: prepare
        t0 = time.time()
        while s._job_switch["store"] is None and not s._job_switch["error"] \
                and time.time() - t0 < 5:
            time.sleep(0.01)
        mgr.tick(s)
        check("its dataset is tokenized on a thread; no switch while a backward "
              "of the previous job is pending",
              s._job_switch is not None and s._job_switch["store"] is not None
              and rpc == [] and mgr.current.stage == "init"
              and s._ft_injector.store is store0)
        s._coord.pending_backward = False
        mgr.tick(s)
        check("the job's adapter file was pre-read in the background when it "
              "was prepared (one call, its path), so the switch does not wait "
              "for the disk",
              [c[1] for c in prepared] == [(a2,)])
        check("then every rank first CHECKS the adapter, then takes it, with the "
              "job's hyper-parameters",
              len(rpc) == 2 and rpc[0][0] == rpc[1][0] == "deltaserve_switch_ft_adapter"
              and rpc[0][1][1].get("dry_run") is True
              and not rpc[1][1][1].get("dry_run")
              and rpc[1][1][0] == a2 and rpc[1][1][1]["learning_rate"] == 3e-4
              and rpc[1][1][1]["total_tokens_per_epoch"] == 15)
        new = s._ft_injector.store
        check("the scheduler takes the new dataset and resets its counters",
              new is not store0 and len(new) == 5 and new.total_epochs == 2
              and s._coord.current_epoch == 0 and s._coord.admission_open
              and not s._coord.epoch_flush_pending
              and s._coord.corpus_total_tokens == 15
              and s._ft_exhausted_logged is False)
        check("the second job is now training", mgr.current.stage == "training")

        bad = s.submit_ft_job({"adapter_path": adapter("job3"),
                               "data_path": corpus("empty.txt", 0), "name": "third"})
        for epoch in range(2):                          # two epochs of job 2
            taken = []
            while new.has_next():
                x = new.pop_best_under(16, exclude=taken)
                taken.append(x)
                new.claim([x])
            new.commit_claimed(taken)
            if epoch == 0:
                new.advance_epoch()
        t0 = time.time()
        while mgr.busy and time.time() - t0 < 5:
            mgr.tick(s)
            time.sleep(0.01)
        fin = {j.name: j for j in mgr.finished}
        check("a job with an unusable dataset fails at init; the queue moves on "
              "and the server keeps its current adapter",
              fin["second"].stage == "done" and fin["third"].failed
              and "no usable sample" in fin["third"].error and len(rpc) == 2
              and not mgr.busy and bad["name"] == "third")

        calls = []

        def refuse(m, args=(), **k):       # rank 0 accepts, rank 1 refuses
            if m == "deltaserve_prepare_ft_adapter":
                return [True, True]
            calls.append(args[1].get("dry_run"))
            return [{"tensors": 8, "dry_run": True},
                    {"error": "ValueError: rank differs"}]

        s._ds_engine.model_executor.collective_rpc = refuse
        s.submit_ft_job({"adapter_path": adapter("job4"),
                         "data_path": corpus("four.txt", 2), "name": "fourth"})
        keep = s._ft_injector.store
        t0 = time.time()
        while mgr.busy and time.time() - t0 < 5:
            mgr.tick(s)
            time.sleep(0.01)
        check("an adapter one rank refuses fails its job in the CHECK phase: no "
              "rank is told to take it, the dataset is not switched",
              mgr.finished[-1].failed and "refused" in mgr.finished[-1].error
              and "rank differs" in mgr.finished[-1].error
              and calls == [True] and s._ft_injector.store is keep)

        s2 = Sched.__new__(Sched)
        s2.vllm_config = SimpleNamespace(finetune_config=SimpleNamespace(
            finetuning_lora_path=None))
        check("a server launched without a finetuning adapter refuses jobs",
              "finetuning slot" in (error_of(lambda: s2.submit_ft_job(
                  {"adapter_path": a2, "data_path": d2})) or ""))


def test_yaml_section():
    green("ft_job YAML section:")
    import yaml

    from deltaserve.config_loader import (
        additional_config,
        finetune_kwargs,
        load_yaml_config,
        split_config,
    )

    raw = load_yaml_config(_CONFIG, verbose=False)
    check("preset has no ft_job section → no ft_job launch argument",
          "ft_job" not in raw and "ft_job" not in additional_config(raw)["deltaserve"])
    _, ftc, _ = split_config(raw)
    check("preset still loads; FinetuneConfig.ft_job is None", ftc.ft_job is None)

    body = yaml.safe_load(_CONFIG.read_text())
    flat = {k: body["finetune_param"].pop(k) for k in ("data_path", "num_epochs")}
    lora = body["finetune_param"].pop("finetuning_lora_path")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cfg.yaml"

        def load(job):
            b = dict(body)
            b["ft_job"] = job
            p.write_text(yaml.safe_dump(b))
            return load_yaml_config(p, verbose=False)

        cfg = load({"name": "alpaca", "stages": ["init", "training", "ready"],
                    "adapter_path": "ad/x", "data_path": "ds.txt",
                    "num_epochs": flat["num_epochs"]})
        kw = finetune_kwargs(cfg)
        check("adapter_path / data_path set the flat fields, resolved to absolute",
              kw["finetuning_lora_path"] == str((Path(d) / "ad/x").resolve())
              and kw["data_path"] == str((Path(d) / "ds.txt").resolve()))
        check("training key set through the job", kw["num_epochs"] == flat["num_epochs"])
        check("the rest is ft_job metadata",
              kw["ft_job"] == {"name": "alpaca",
                               "stages": ["init", "training", "ready"]})
        _, ftc, _ = split_config(cfg)
        check("FinetuneConfig carries the metadata", ftc.ft_job["name"] == "alpaca")
        check("metadata reaches the launch arguments",
              additional_config(cfg)["deltaserve"]["ft_job"]["name"] == "alpaca")

        check("unknown ft_job key rejected by the loader",
              error_of(lambda: finetune_kwargs(load({"bogus": 1}))) is not None)
        check("bad stage rejected at load time (split_config)",
              "indexing" in (error_of(lambda: split_config(
                  load({"stages": ["init", "indexing"], "adapter_path": lora,
                        "data_path": "ds.txt"}))) or ""))

        b2 = yaml.safe_load(_CONFIG.read_text())
        b2["ft_job"] = {"num_epochs": 3}
        p.write_text(yaml.safe_dump(b2))
        err = error_of(lambda: finetune_kwargs(load_yaml_config(p, verbose=False)))
        check("a key set in both finetune and ft_job is an error",
              err is not None and "num_epochs" in err)


def main():
    green("=== finetuning jobs + stages ===")
    test_spec()
    test_stage_graph()
    test_training_stage_with_real_store()
    test_manager_and_names()
    test_online_jobs()
    test_yaml_section()
    print()
    if _failed:
        print(f"{_RED}{_failed} failed{_RESET}, {_passed} passed")
        sys.exit(1)
    green(f"all {_passed} checks passed")


if __name__ == "__main__":
    main()
