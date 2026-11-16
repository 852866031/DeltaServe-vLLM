#!/usr/bin/env python
"""Pool adapters through vLLM's LoRA resolver hook
(deltaserve.adapter_routing.resolver).

CPU-only, no model. Uses the real ``OpenAIServingModels`` resolve path of the
pinned vLLM with a stand-in engine client, so the test exercises what an API
server actually does with our resolver.

    python tests/test_adapter_pool_resolver.py

Checks
  * a pool adapter is resolved by name with the pool's id and path; unknown
    names, adapters without weights on disk and a missing registry are not;
  * the registry is re-read when it changes (an adapter joins after launch);
  * the pool's id survives vLLM overwriting ``lora_int_id`` with its own
    per-process counter, and the request is a plain LoRARequest on the wire;
  * two API server processes (two model registries, different counters) hand
    the engine the SAME id for the same adapter — the reason ids are pinned;
  * installing registers the resolver and turns on vLLM's resolve path.
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

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


class NullBackend:
    def load(self, e): pass
    def activate(self, e): pass
    def deactivate(self, e): pass
    def unload(self, e): pass


def make_pool(d, names):
    from deltaserve.adapter_routing.pool import AdapterPool

    pool = AdapterPool(NullBackend(), gpu_slots=2, directory=d)
    for n in names:
        path = Path(d) / "weights" / n
        path.mkdir(parents=True)
        pool.register(n, str(path))
    return pool


def test_resolver():
    green("PoolLoRAResolver:")
    import msgspec

    from deltaserve.adapter_routing.pool import LORA_ID_BASE
    from deltaserve.adapter_routing.resolver import PoolLoRAResolver
    from vllm.lora.request import LoRARequest

    with tempfile.TemporaryDirectory() as d:
        r = PoolLoRAResolver(d)
        check("no registry yet → nothing resolves", r.lookup("x") is None)
        pool = make_pool(d, ["dserve-routing-aa", "dserve-routing-bb"])
        req = asyncio.run(r.resolve_lora("org/base", "dserve-routing-bb"))
        check("registry appears → pool adapter resolved with its id and path",
              isinstance(req, LoRARequest) and req.lora_name == "dserve-routing-bb"
              and req.lora_int_id == LORA_ID_BASE + 2
              and req.lora_path == pool.entries["dserve-routing-bb"].path)
        check("unknown name → None (vLLM answers 'model does not exist')",
              r.lookup("not-in-pool") is None)

        req.lora_int_id = 7          # what OpenAIServingModels.resolve_lora does
        check("the pool's id survives vLLM's reassignment",
              req.lora_int_id == LORA_ID_BASE + 2)
        req.lora_path = req.lora_path     # other fields stay assignable
        wire = msgspec.msgpack.decode(msgspec.msgpack.encode(req), type=LoRARequest)
        check("on the wire it is a plain LoRARequest with the pinned id",
              type(wire) is LoRARequest and wire.lora_int_id == LORA_ID_BASE + 2
              and wire.lora_name == req.lora_name)

        late = Path(d) / "weights" / "dserve-routing-cc"
        late.mkdir()
        os.utime(Path(d) / "pool.json", ns=(1, 1))
        pool.register("dserve-routing-cc", str(late))
        check("an adapter that joins the pool later is resolved without a restart",
              r.lookup("dserve-routing-cc").lora_int_id == LORA_ID_BASE + 3)
        pool.register("no-weights", str(Path(d) / "weights" / "missing"))
        check("an adapter whose weights are not on disk is not resolved",
              r.lookup("no-weights") is None)
        os.utime(Path(d) / "pool.json", ns=(2, 2))
        pool.remove("dserve-routing-aa")
        check("an adapter removed from the pool stops resolving",
              r.lookup("dserve-routing-aa") is None)


def make_engine():
    """The engine as an API server sees it: anything vLLM's registry reads at
    construction is a mock; ``add_lora`` records what the engine is asked to load."""
    from unittest.mock import MagicMock

    engine = MagicMock()
    engine.model_config = SimpleNamespace(model="org/base")
    engine.added = []

    async def add_lora(req):
        engine.added.append((req.lora_name, req.lora_int_id, req.lora_path))

    engine.add_lora = add_lora
    return engine


def serving_models(engine, resolver, burn=0):
    """A real OpenAIServingModels (one API server's model registry) with the
    given resolver; ``burn`` advances its private id counter."""
    from vllm.entrypoints.openai.models.serving import OpenAIServingModels
    from vllm.lora.resolver import LoRAResolverRegistry

    LoRAResolverRegistry.resolvers.clear()
    LoRAResolverRegistry.register_resolver("pool", resolver)
    m = OpenAIServingModels.__new__(OpenAIServingModels)
    OpenAIServingModels.__init__(
        m, engine_client=engine, base_model_paths=[], lora_modules=None)
    m.model_config = SimpleNamespace(model="org/base")
    m.engine_client = engine
    for _ in range(burn):
        m.lora_id_counter.inc(1)
    return m


def test_through_vllm():
    green("through vLLM's own resolve path (two API servers):")
    from deltaserve.adapter_routing.pool import LORA_ID_BASE
    from deltaserve.adapter_routing.resolver import PoolLoRAResolver
    from vllm.entrypoints.openai.models.serving import OpenAIServingModels  # noqa: F401
    from vllm.lora.request import LoRARequest

    with tempfile.TemporaryDirectory() as d:
        make_pool(d, ["dserve-routing-aa", "dserve-routing-bb"])
        engine = make_engine()
        try:
            server1 = serving_models(engine, PoolLoRAResolver(d), burn=0)
            server2 = serving_models(engine, PoolLoRAResolver(d), burn=5)
        except Exception as e:  # noqa: BLE001
            check(f"could build vLLM's model registry ({type(e).__name__}: {e})",
                  False)
            return

        r1 = asyncio.run(server1.resolve_lora("dserve-routing-bb"))
        check("API server 1 resolves the pool adapter and loads it into the engine",
              isinstance(r1, LoRARequest) and engine.added[-1][0] == "dserve-routing-bb")
        check("...and caches it in its adapter table",
              "dserve-routing-bb" in server1.lora_requests)
        r2 = asyncio.run(server2.resolve_lora("dserve-routing-bb"))
        ids = [i for n, i, _ in engine.added if n == "dserve-routing-bb"]
        check("API server 2 (different private counter) gives the engine the "
              f"SAME id {ids}",
              len(ids) == 2 and ids[0] == ids[1] == LORA_ID_BASE + 2
              and isinstance(r2, LoRARequest))
        asyncio.run(server2.resolve_lora("dserve-routing-aa"))
        asyncio.run(server1.resolve_lora("dserve-routing-aa"))
        by_name = {}
        for n, i, _ in engine.added:
            by_name.setdefault(n, set()).add(i)
        check("resolved in a different order on each server: still one id per "
              "adapter, and no id shared by two adapters",
              all(len(v) == 1 for v in by_name.values())
              and len({next(iter(v)) for v in by_name.values()}) == len(by_name))
        n = len(engine.added)
        again = asyncio.run(server1.resolve_lora("dserve-routing-bb"))
        check("a second request for the same adapter is served from the table",
              again is r1 and len(engine.added) == n)
        miss = asyncio.run(server1.resolve_lora("nope"))
        check("an unknown name is still a 404 from vLLM",
              not isinstance(miss, LoRARequest)
              and "nope" not in server1.lora_requests)


def test_invalidation():
    green("API-server cache invalidation:")
    from deltaserve.adapter_routing.resolver import (
        PoolLoRAResolver,
        make_invalidator,
    )
    from vllm.lora.request import LoRARequest

    with tempfile.TemporaryDirectory() as d:
        pool = make_pool(d, ["a", "b", "c"])
        resolver = PoolLoRAResolver(d)
        table = {n: resolver.lookup(n) for n in ("a", "b", "c")}
        table["static"] = LoRARequest(lora_name="static", lora_int_id=3,
                                      lora_path=str(Path(d)))
        app = SimpleNamespace(state=SimpleNamespace(
            openai_serving_models=SimpleNamespace(lora_requests=table)))
        before = make_invalidator(app, resolver)

        for n in ("a", "b", "c", "static", "unknown"):
            before(n)
        check("unchanged adapters stay cached", set(table) == {"a", "b", "c", "static"})

        os.utime(Path(d) / "pool.json", ns=(1, 1))
        pool.remove("a")
        before("a")
        check("an adapter removed from the pool is dropped from the table "
              "(vLLM then answers 'model does not exist')",
              "a" not in table and resolver.lookup("a") is None)

        new = Path(d) / "weights" / "b-v2"
        new.mkdir()
        old_id = table["b"].lora_int_id
        os.utime(Path(d) / "pool.json", ns=(2, 2))
        pool.admit("b", str(new))
        before("b")
        fresh = resolver.lookup("b")
        check("a new version (other path, other id) is dropped, so the next "
              "request resolves the new one",
              "b" not in table and fresh.lora_path == str(new)
              and fresh.lora_int_id != old_id)
        os.utime(Path(d) / "pool.json", ns=(3, 3))
        pool.retire("c")
        before("c")
        check("a retired adapter stays cached (it is still servable by name)",
              "c" in table)
        before("static")
        check("adapters that are not the pool's are never touched",
              "static" in table)
        app.state.openai_serving_models = None
        ok = True
        try:
            before("b")
        except Exception:  # noqa: BLE001
            ok = False
        check("before the server finished starting the hook does nothing", ok)


def test_install():
    green("install_resolver:")
    from deltaserve.adapter_routing.resolver import RESOLVER_NAME, install_resolver
    from deltaserve.config import AdapterRoutingConfig
    from vllm.lora.resolver import LoRAResolverRegistry

    keep = os.environ.pop("VLLM_ALLOW_RUNTIME_LORA_UPDATING", None)
    LoRAResolverRegistry.resolvers.clear()
    try:
        with tempfile.TemporaryDirectory() as d:
            cfg = AdapterRoutingConfig(enable_adapter_routing=True, index_path=d)
            r = install_resolver(cfg)
            check("resolver registered with vLLM under its name",
                  LoRAResolverRegistry.get_resolver(RESOLVER_NAME) is r
                  and r.directory == d)
            check("vLLM's resolve path switched on for this process",
                  os.environ.get("VLLM_ALLOW_RUNTIME_LORA_UPDATING") == "1")
    finally:
        LoRAResolverRegistry.resolvers.clear()
        if keep is None:
            os.environ.pop("VLLM_ALLOW_RUNTIME_LORA_UPDATING", None)
        else:
            os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = keep


def main():
    green("=== pool adapters via vLLM's LoRA resolver ===")
    test_resolver()
    test_through_vllm()
    test_invalidation()
    test_install()
    print()
    if _failed:
        print(f"{_RED}{_failed} failed{_RESET}, {_passed} passed")
        sys.exit(1)
    green(f"all {_passed} checks passed")


if __name__ == "__main__":
    main()
