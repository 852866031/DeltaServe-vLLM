#!/usr/bin/env python
"""Adapter routing inside the engine (deltaserve.adapter_routing.engine): the
whole loop on CPU.

A finetuning job with stages [init, training, indexing, ready] runs against a
stand-in engine (fake LoRA calls, fake worker export, fake encoder). When it
reaches 'ready' its adapter must be saved, in the pool, resolvable by the API
servers and selected for a matching ``model: "auto"`` request.

    python tests/test_adapter_engine.py
    DSERVE_TEST_REAL_ENCODER=1 python tests/test_adapter_engine.py  # + real router service

Checks
  * job → indexing → 'ready' → saved adapter (TP shards merged) → pool →
    registry + routable index → resolver → router → middleware;
  * an adapter that is indexed but not ready is NOT routable;
  * hit counting, a restart restoring the same pool with the same LoRA ids,
    a job without auto_routing, GPU-slot count, LoRA serving required.
"""

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

_passed = 0
_failed = 0
_TTY = sys.stdout.isatty()
_GREEN = "\033[92m" if _TTY else ""
_RED = "\033[91m" if _TTY else ""
_RESET = "\033[0m" if _TTY else ""
DIM = 8
TOPICS = {"translate": 0, "sql": 1}
HID, Q, KV, R = 16, 16, 8, 2


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
        return f"{type(e).__name__}: {e}"
    return None


def topic_embed(texts):
    out = []
    for t in texts:
        v = np.full(DIM, 0.01, dtype=np.float32)
        for word, axis in TOPICS.items():
            if word in t.lower():
                v[axis] = 1.0
                break
        else:
            v[DIM - 1] = 1.0
        out.append(v)
    return np.array(out)


class FakeEncoder:
    def encode(self, texts):
        return topic_embed(texts)


class FakeExec:
    device, loaded = "cpu", True

    def __init__(self):
        self.job = None

    def submit(self, texts, threads):
        self.job = list(texts)

    def poll(self, timeout=0.0):
        texts, self.job = self.job, None
        return None if texts is None else topic_embed(texts)

    def cancel(self):
        self.job = None

    def unload(self):
        pass

    def stop(self):
        pass


def lora_state(seed):
    g = torch.Generator().manual_seed(seed)
    st = {}
    pre = "base_model.model.model.layers.0.self_attn."
    for proj, out, inp in [("q", Q, HID), ("k", KV, HID), ("v", KV, HID),
                           ("o", HID, Q)]:
        st[pre + f"{proj}_proj.lora_A.weight"] = torch.randn(R, inp, generator=g)
        st[pre + f"{proj}_proj.lora_B.weight"] = torch.randn(out, R, generator=g)
    return st


def rank_shards(state, tp=2):
    from deltaserve.adapter_save import LORA_KEY_RE
    from deltaserve.bwd_services.common.tp import lora_shard_slice

    out = []
    for r in range(tp):
        out.append((r, {k: lora_shard_slice(
            LORA_KEY_RE.search(k).group(2), LORA_KEY_RE.search(k).group(3), t, r,
            tp, Q // tp, KV // tp).clone() for k, t in state.items()}))
    return out


class FakeEngineCore:
    """What the routing engine and the adapter export need of EngineCore."""

    def __init__(self, trained):
        self.added, self.removed = [], []
        self.model_executor = SimpleNamespace(
            collective_rpc=lambda method, *a, **k: rank_shards(trained))

    def add_lora(self, req):
        self.added.append((req.lora_name, req.lora_int_id, req.lora_path))
        return True

    def remove_lora(self, lora_id):
        self.removed.append(lora_id)
        return True


def vllm_config(d, finetuning=True, max_loras=4, lora=True):
    from deltaserve.config import AdapterRoutingConfig

    cfg = AdapterRoutingConfig(
        enable_adapter_routing=True, index_path=str(Path(d) / "index"),
        fallback_score_threshold=0.5, indexing_chunk_samples=4,
        indexing_cpu_reserve_cores=0, indexing_min_free_memory_gb=0.0)
    return SimpleNamespace(
        adapter_routing_config=cfg,
        finetune_config=SimpleNamespace(enable_finetuning=finetuning),
        lora_config=SimpleNamespace(max_loras=max_loras) if lora else None)


def source_adapter(d):
    from safetensors.torch import save_file

    src = Path(d) / "init-adapter"
    src.mkdir(exist_ok=True)
    save_file(lora_state(1), str(src / "adapter_model.safetensors"))
    (src / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}))
    return str(src)


class SchedulerCtx:
    """The scheduler as the job stages and 'ready' see it."""

    def __init__(self, vc, engine, routing, data):
        self.adapter_routing_config = vc.adapter_routing_config
        self._ds_engine, self._adapter_routing = engine, routing
        self.training_done = False
        outer = self
        self.store = type("Store", (), {
            "data_path": data, "current_epoch": 0, "total_epochs": 1,
            "has_next": lambda s: not outer.training_done,
            "has_claimed": lambda s: False, "__len__": lambda s: 12})()
        self.coordinator = SimpleNamespace(pending_backward=False, fill_count=0,
                                           reserved_fill=0)

    def on_job_ready(self, job):
        from deltaserve.adapter_save import save_job_adapter

        save_job_adapter(job, self._ds_engine, self._adapter_routing)


async def auto_request(mw, text):
    seen, out = {}, {}

    async def app(scope, receive, send):
        seen["body"] = (await receive())["body"]
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    mw.app = app
    body = json.dumps({"model": "auto",
                       "messages": [{"role": "user", "content": text}]}).encode()
    sent = [{"type": "http.request", "body": body, "more_body": False}]

    async def receive():
        return sent.pop(0) if sent else {"type": "http.disconnect"}

    async def send(m):
        if m["type"] == "http.response.start":
            out.update({k.decode(): v.decode() for k, v in m["headers"]})

    await mw({"type": "http", "method": "POST", "path": "/v1/chat/completions",
              "headers": []}, receive, send)
    return json.loads(seen["body"])["model"], out


def test_closed_loop():
    green("closed loop: job → index → save → pool → resolver → route:")
    from safetensors.torch import load_file

    from deltaserve.adapter_routing import stages as st
    from deltaserve.adapter_routing.engine import AdapterRoutingEngine
    from deltaserve.adapter_routing.pool import (
        LORA_ID_BASE,
        ROUTABLE_FILE,
        read_registry,
    )
    from deltaserve.adapter_routing.resolver import PoolLoRAResolver
    from deltaserve.adapter_routing.service import Router
    from deltaserve.ft_job import FinetuneJobSpec, FtJobManager
    from deltaserve.route_middleware import AdapterRouteMiddleware

    keep = st.IndexingStage.make_executors
    st.IndexingStage.make_executors = staticmethod(lambda cfg: {"cpu": FakeExec()})
    try:
        with tempfile.TemporaryDirectory() as d:
            data = Path(d) / "corpus.txt"
            data.write_text("".join(f"translate sentence {i} to english\n"
                                    for i in range(12)))
            trained = lora_state(42)
            vc = vllm_config(d)
            engine = FakeEngineCore(trained)
            routing = AdapterRoutingEngine(engine, vc, start_service=False)
            check("pool starts empty; 4 LoRA slots minus the FT slot → 3 for the pool",
                  len(routing.pool) == 0 and routing.pool.gpu_slots == 3)
            ctx = SchedulerCtx(vc, engine, routing, str(data))

            mgr = FtJobManager(adapter_routing=True)
            job = mgr.submit(FinetuneJobSpec(
                stages=("init", "training", "indexing", "ready"),
                auto_routing=True, adapter_path=source_adapter(d),
                data_path=str(data)))
            check("unnamed auto-routable job is called dserve-routing-…",
                  job.name.startswith("dserve-routing-"))

            router = Router(FakeEncoder(), str(Path(routing.dir) / ROUTABLE_FILE),
                            threshold=0.5, reload_s=0.0)
            t0 = time.time()
            while job.stage_state["indexing"] != "done" and time.time() - t0 < 20:
                mgr.tick(ctx)
                time.sleep(0.02)
            check("indexing done while training is still running",
                  job.stage == "training" and job.saved_path is None)
            check("indexed but not ready → NOT routable yet (falls back)",
                  router.route("translate this").adapter is None
                  and len(routing.pool) == 0 and engine.added == [])

            ctx.training_done = True
            mgr.tick(ctx)
            check("training done → job ready and finished",
                  job.finished and not job.failed and mgr.current is None)

            saved = load_file(os.path.join(job.saved_path, "adapter_model.safetensors"))
            check("trained adapter saved under the pool's adapter directory, "
                  "TP shards merged",
                  job.saved_path == routing.adapter_dir(job.name)
                  and all(torch.equal(saved[k], trained[k]) for k in trained))
            check("adapter is in the pool with a pool-range LoRA id",
                  job.name in routing.pool
                  and routing.pool.entries[job.name].lora_int_id == LORA_ID_BASE + 1)
            check("the engine was asked to load it (name, id, saved path)",
                  engine.added == [(job.name, LORA_ID_BASE + 1, job.saved_path)])
            reg = read_registry(routing.dir)
            check("registry written for the API servers",
                  [(a["name"], a["path"]) for a in reg["adapters"]]
                  == [(job.name, job.saved_path)])

            req = PoolLoRAResolver(routing.dir).lookup(job.name)
            check("an API server's resolver finds it, with the same id and path",
                  req is not None and req.lora_int_id == LORA_ID_BASE + 1
                  and req.lora_path == job.saved_path)
            dec = router.route("please translate this sentence")
            check("the router service now selects it for a matching request",
                  dec.adapter == job.name and dec.score > 0.9)
            check("...and still falls back for an unrelated one",
                  router.route("what is the weather").adapter is None)

            mw = AdapterRouteMiddleware(None, router.route, "auto", "org/base")
            model, headers = asyncio.run(auto_request(mw, "translate: guten tag"))
            check("model='auto' request leaves the API server naming the new adapter",
                  model == job.name
                  and headers.get("x-deltaserve-routing") == "routed"
                  and headers.get("x-deltaserve-adapter") == job.name)
            model, headers = asyncio.run(auto_request(mw, "tell me a joke"))
            check("an unmatched request goes to the base model with the reason",
                  model == "org/base"
                  and headers.get("x-deltaserve-routing-reason") == "below-threshold")

            lr = SimpleNamespace(lora_name=job.name)
            routing.note_request(SimpleNamespace(lora_request=lr))
            routing.note_request(SimpleNamespace(lora_request=lr))
            routing.note_request(SimpleNamespace(lora_request=None))
            routing.note_request(SimpleNamespace(
                lora_request=SimpleNamespace(lora_name="some-static-adapter")))
            check("requests using the adapter are counted; others are ignored",
                  routing.pool.entries[job.name].hits == 2)
            status = routing.status()
            check("status reports the pool",
                  status["adapters"][0]["name"] == job.name
                  and status["router_service_alive"] is False)

            engine2 = FakeEngineCore(trained)
            routing2 = AdapterRoutingEngine(engine2, vc, start_service=False)
            check("server restart: the pool comes back with the same adapter and id, "
                  "and the engine reloads it",
                  list(routing2.pool.entries) == [job.name]
                  and engine2.added == [(job.name, LORA_ID_BASE + 1, job.saved_path)]
                  and routing2.pool.routable_names() == [job.name])

            job2 = mgr.submit(FinetuneJobSpec(
                name="plain-job", stages=("init", "training", "ready"),
                adapter_path=source_adapter(d), data_path=str(data),
                save_path=str(Path(d) / "plain-saved")))
            ctx2 = SchedulerCtx(vc, engine, routing, str(data))
            ctx2.training_done = True
            mgr.tick(ctx2)
            check("a job without auto_routing is saved to its save_path, not pooled",
                  job2.finished and not job2.failed
                  and job2.saved_path == str(Path(d) / "plain-saved")
                  and "plain-job" not in routing.pool)

            job3 = mgr.submit(FinetuneJobSpec(
                name="no-index", stages=("init", "training", "ready"),
                auto_routing=True, adapter_path=source_adapter(d),
                data_path=str(data)))
            mgr.tick(ctx2)
            check("auto_routing without an indexing stage: servable by name, "
                  "not selectable",
                  job3.finished and "no-index" in routing.pool
                  and "no-index" not in routing.pool.routable_names())
    finally:
        st.IndexingStage.make_executors = keep


class CountingExec(FakeExec):
    """Counts what an indexing job costs."""

    made = []

    def __init__(self):
        super().__init__()
        self.loaded = False
        self.submits, self.samples, self.unloads, self.stopped = 0, 0, 0, False
        CountingExec.made.append(self)

    def submit(self, texts, threads):
        self.submits += 1
        self.samples += len(texts)
        self.loaded = True
        super().submit(texts, threads)

    def unload(self):
        self.unloads += 1
        self.loaded = False

    def stop(self):
        self.stopped = True


def run_jobs(routing, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        routing.tick_jobs()
        if routing.jobs.current is None and not routing.jobs.queue:
            routing.tick_jobs()
            return True
        time.sleep(0.01)
    return False


def test_external_adapters():
    green("existing adapters, no training: jobs [init, indexing, ready]:")
    from deltaserve.adapter_routing import stages as st
    from deltaserve.adapter_routing.engine import AdapterRoutingEngine
    from deltaserve.adapter_routing.pool import ROUTABLE_FILE
    from deltaserve.adapter_routing.resolver import PoolLoRAResolver
    from deltaserve.adapter_routing.service import Router
    from deltaserve.adapter_routing.stages import SharedIndexer

    keep = st.IndexingStage.make_executors
    st.IndexingStage.make_executors = staticmethod(lambda cfg: {"cpu": CountingExec()})
    CountingExec.made.clear()
    try:
        with tempfile.TemporaryDirectory() as d:
            def corpus(name, word, n=10):
                p = Path(d) / f"{name}.txt"
                p.write_text("".join(f"{word} example number {i} " + "x" * (i * 3)
                                     + "\n" for i in range(n)))
                return str(p)

            src = source_adapter(d)
            rpc_calls = []
            engine = FakeEngineCore(lora_state(0))
            engine.model_executor.collective_rpc = (
                lambda method, *a, **k: rpc_calls.append(method) or [])
            vc = vllm_config(d, finetuning=False)
            routing = AdapterRoutingEngine(engine, vc, start_service=False,
                                           start_driver=False)

            j1 = routing.submit_adapter(src, name="translate-adapter",
                                        data_path=corpus("tr", "translate"))
            j2 = routing.submit_adapter(src, data_path=corpus("sq", "sql"))
            np.save(Path(d) / "c.npy", np.eye(DIM, dtype=np.float32)[5])
            j3 = routing.submit_adapter(src, name="with-centroid",
                                        centroid_path=str(Path(d) / "c.npy"))
            j4 = routing.submit_adapter(src, name="name-only")
            check("with a dataset → [init, indexing, ready]; with a centroid or "
                  "nothing → [init, ready]",
                  list(j1["stages"]) == ["init", "indexing", "ready"]
                  and list(j3["stages"]) == ["init", "ready"]
                  and list(j4["stages"]) == ["init", "ready"])
            check("a given name is kept; an unnamed adapter is dserve-routing-…",
                  j1["name"] == "translate-adapter"
                  and j2["name"].startswith("dserve-routing-"))
            check("works with finetuning off (no scheduler, no backward child)",
                  run_jobs(routing))
            fin = {j["name"]: j for j in routing.status()["jobs"]["finished"]}
            check("all four jobs finished without error",
                  len(fin) == 4 and all(j["error"] is None and j["stage"] == "done"
                                        for j in fin.values()))

            check("no export from the workers, nothing asked of the engine "
                  "(weights load on first use)",
                  rpc_calls == [] and engine.added == [])
            check("no copy: the pool points at the adapter where it already is",
                  all(e.path == src for e in routing.pool.entries.values())
                  and fin["translate-adapter"]["saved_path"] == src
                  and not os.path.exists(routing.adapter_dir("translate-adapter")))
            check("all on the disk tier until used",
                  all(e.tier == "disk" for e in routing.pool.entries.values()))
            check("ONE encoder process served every job, and was unloaded "
                  f"afterwards (created {len(CountingExec.made)})",
                  len(CountingExec.made) == 1 and CountingExec.made[0].unloads >= 1
                  and not CountingExec.made[0].stopped)
            check("exactly the 20 dataset samples were encoded",
                  CountingExec.made[0].samples == 20)
            check("the name-only adapter is in the pool but not selectable",
                  "name-only" in routing.pool
                  and sorted(routing.pool.routable_names())
                  == sorted([j1["name"], j2["name"], "with-centroid"]))

            router = Router(FakeEncoder(), str(Path(routing.dir) / ROUTABLE_FILE),
                            threshold=0.5, reload_s=0.0)
            check("requests are routed between the two indexed adapters",
                  router.route("translate this please").adapter == "translate-adapter"
                  and router.route("write a sql query").adapter == j2["name"])
            res = PoolLoRAResolver(routing.dir)
            ids = {n: res.lookup(n).lora_int_id for n in routing.pool.entries}
            check("every one resolves for the API servers, each with its own id",
                  len(set(ids.values())) == 4)
            routing.note_request(SimpleNamespace(
                lora_request=SimpleNamespace(lora_name="translate-adapter")))
            check("the first request for it loads it into the engine",
                  [a[0] for a in engine.added] == ["translate-adapter"]
                  and routing.pool.entries["translate-adapter"].tier == "gpu")

            n = CountingExec.made[0].samples
            vc2 = vllm_config(d, finetuning=False)
            vc2.adapter_routing_config.adapters = [
                {"path": src, "name": "translate-adapter",
                 "data_path": str(Path(d) / "tr.txt")},
                {"path": str(Path(d) / "missing"), "name": "broken",
                 "data_path": str(Path(d) / "tr.txt")},
                {"path": src, "name": "late", "data_path": corpus("po", "poem", 6)}]
            routing2 = AdapterRoutingEngine(FakeEngineCore(lora_state(0)), vc2,
                                            start_service=False, start_driver=False)
            check("restart with the adapters listed in the config",
                  run_jobs(routing2))
            fin2 = {j["name"]: j for j in routing2.status()["jobs"]["finished"]}
            check("an adapter whose centroid is already built is NOT re-encoded; "
                  "only the new one is",
                  sum(x.samples for x in CountingExec.made) == n + 6
                  and fin2["translate-adapter"]["error"] is None)
            check("a bad entry fails its own job with a clear error; the others "
                  "still go through",
                  fin2["broken"]["stage"] == "failed"
                  and "no adapter weights" in fin2["broken"]["error"]
                  and "late" in routing2.pool.routable_names()
                  and "broken" not in routing2.pool)
            check("the earlier pool is still there after the restart",
                  {"translate-adapter", j2["name"], "with-centroid", "name-only",
                   "late"} <= set(routing2.pool.entries))

            Path(d, "tr.txt").write_text("translate something new\n" * 3)
            routing2.submit_adapter(src, name="translate-adapter",
                                    data_path=str(Path(d) / "tr.txt"))
            run_jobs(routing2)
            check("a changed dataset is indexed again",
                  sum(x.samples for x in CountingExec.made) == n + 6 + 3)
            err = error_of(lambda: routing2.submit_adapter(
                str(Path(d) / "elsewhere"), name="late"))
            check("a name already used for another path is refused",
                  err is not None and "another path" in err)
            routing2.submit_adapter(src, name="warm", prewarm=True)
            eng2 = routing2.engine
            before = len(eng2.added)
            run_jobs(routing2)
            check("prewarm: true loads the adapter at registration",
                  len(eng2.added) == before + 1 and eng2.added[-1][0] == "warm")
            routing.shutdown()
            routing2.shutdown()
            check("shutdown stops the shared encoder process",
                  all(x.stopped for x in CountingExec.made))

            sh = SharedIndexer(lambda: {"cpu": CountingExec()})
            a = sh.acquire()
            check("the shared encoder is used by one job at a time",
                  a is not None and sh.acquire() is None)
            sh.release()
            check("...and handed to the next job without creating another",
                  sh.acquire() is a and sh.created == 1)
    finally:
        st.IndexingStage.make_executors = keep


def test_adapters_config():
    green("adapter_routing.adapters config:")
    import yaml

    from deltaserve.config import AdapterRoutingConfig
    from deltaserve.config_loader import adapter_routing_config, load_yaml_config

    ok = AdapterRoutingConfig(adapters=[{"path": "/a", "name": "x",
                                         "data_path": "/d", "prewarm": True}])
    check("a list of {path, name?, data_path?, centroid_path?, prewarm?}",
          ok.adapters[0]["name"] == "x")
    check("an entry without a path is rejected",
          error_of(lambda: AdapterRoutingConfig(adapters=[{"name": "x"}])) is not None)
    check("an unknown key is rejected",
          "unknown" in (error_of(lambda: AdapterRoutingConfig(
              adapters=[{"path": "/a", "dataset": "/d"}])) or ""))
    check("duplicate names are rejected",
          "duplicate" in (error_of(lambda: AdapterRoutingConfig(
              adapters=[{"path": "/a", "name": "x"},
                        {"path": "/b", "name": "x"}])) or ""))
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cfg.yaml"
        p.write_text(yaml.safe_dump({"adapter_routing": {
            "enable_adapter_routing": True,
            "adapters": [{"name": "x", "path": "ad/x", "data_path": "data/x.txt",
                          "centroid_path": "/abs/c.npy"}]}}))
        cfg = adapter_routing_config(load_yaml_config(p, verbose=False))
        a = cfg.adapters[0]
        check("relative paths of listed adapters are resolved against the YAML",
              a["path"] == str((Path(d) / "ad/x").resolve())
              and a["data_path"] == str((Path(d) / "data/x.txt").resolve())
              and a["centroid_path"] == "/abs/c.npy" and a["name"] == "x")


def test_engine_config():
    green("engine object:")
    from deltaserve.adapter_routing.engine import AdapterRoutingEngine, VllmLoRABackend

    with tempfile.TemporaryDirectory() as d:
        eng = FakeEngineCore(lora_state(0))
        r = AdapterRoutingEngine(eng, vllm_config(d, finetuning=False, max_loras=4),
                                 start_service=False)
        check("without finetuning every LoRA slot is the pool's",
              r.pool.gpu_slots == 4)
        check("default backend is the vLLM one",
              isinstance(r.pool.backend, VllmLoRABackend))
        check("adapter routing without LoRA serving is refused",
              "enable_lora" in (error_of(lambda: AdapterRoutingEngine(
                  eng, vllm_config(d, lora=False), start_service=False)) or ""))
        w = Path(d) / "w"
        w.mkdir()
        e = r.pool.register("a", str(w))
        r.pool.offload("a")
        check("offload reaches the engine as remove_lora(id)",
              eng.removed == [e.lora_int_id])
        r.shutdown()


def test_real_service():
    if os.environ.get("DSERVE_TEST_REAL_ENCODER") != "1":
        green("real router service process: skipped (set DSERVE_TEST_REAL_ENCODER=1)")
        return
    green("real router service process (e5 on CPU):")
    from deltaserve.adapter_routing.engine import AdapterRoutingEngine
    from deltaserve.adapter_routing.encoder import SentenceEncoder
    from deltaserve.adapter_routing.service import RouterClient, socket_path

    with tempfile.TemporaryDirectory() as d:
        vc = vllm_config(d)
        vc.adapter_routing_config.fallback_score_threshold = 0.0
        routing = AdapterRoutingEngine(FakeEngineCore(lora_state(0)), vc,
                                       start_service=True)
        try:
            client = RouterClient(socket_path(routing.dir), timeout_s=30.0)
            t0, pong = time.time(), None
            while pong is None and time.time() - t0 < 180:
                pong = client.ping()
                time.sleep(0.5)
            check("the engine-spawned router service answers on its socket",
                  pong is not None and routing.status()["router_service_alive"])
            check("empty pool → fallback 'no-adapters'",
                  client.route("translate this").reason == "no-adapters")
            enc = SentenceEncoder("intfloat/e5-large-v2", "query: ", 512, "cpu")
            w = Path(d) / "weights"
            w.mkdir()
            routing.pool.register(
                "dserve-routing-translate", str(w), centroid=enc.encode(
                    ["Translate to English: Guten Morgen",
                     "Translate to English: Wo ist der Bahnhof?"]).mean(0))
            t0, dec = time.time(), None
            while time.time() - t0 < 30:
                dec = client.route("Translate to English: Ich habe Hunger.")
                if dec.adapter:
                    break
                time.sleep(0.5)
            check("an adapter registered in the pool becomes routable in the "
                  f"running service (score {dec.score})",
                  dec.adapter == "dserve-routing-translate")
        finally:
            routing.shutdown()
        time.sleep(0.5)
        check("shutdown stops the service",
              not routing.status()["router_service_alive"])


def main():
    green("=== adapter routing: engine wiring ===")
    test_closed_loop()
    test_external_adapters()
    test_adapters_config()
    test_engine_config()
    test_real_service()
    print()
    if _failed:
        print(f"{_RED}{_failed} failed{_RESET}, {_passed} passed")
        sys.exit(1)
    green(f"all {_passed} checks passed")


if __name__ == "__main__":
    main()
