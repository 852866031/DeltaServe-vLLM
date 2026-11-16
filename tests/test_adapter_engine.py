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
    test_engine_config()
    test_real_service()
    print()
    if _failed:
        print(f"{_RED}{_failed} failed{_RESET}, {_passed} passed")
        sys.exit(1)
    green(f"all {_passed} checks passed")


if __name__ == "__main__":
    main()
