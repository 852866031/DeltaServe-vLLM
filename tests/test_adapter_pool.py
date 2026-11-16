#!/usr/bin/env python
"""The adapter pool (deltaserve.adapter_routing.pool).

CPU-only, no model: the pool drives a fake LoRA backend that records what
would be loaded / activated / unloaded.

    python tests/test_adapter_pool.py

Checks
  * registration: capacity, duplicates, prewarm into a free GPU slot only,
    rollback when the backend fails;
  * GPU slots are kept by decayed frequency: the victim is the least-used
    adapter that is neither in the current batch nor protected, and a hit that
    cannot get a slot stays on the CPU tier;
  * offload (still routable, cold-loaded on the next hit), disable / enable
    (out of / back in the centroid index), remove;
  * routing through the pool with the base-model fallback;
  * the registry survives a save / restore.
"""

import sys
import tempfile

import numpy as np

_passed = 0
_failed = 0
_TTY = sys.stdout.isatty()
_GREEN = "\033[92m" if _TTY else ""
_RED = "\033[91m" if _TTY else ""
_RESET = "\033[0m" if _TTY else ""
DIM = 8
E = np.eye(DIM, dtype=np.float32)


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


class FakeBackend:
    """Stands in for vLLM's LoRA manager; enforces the same invariants."""

    def __init__(self, gpu_slots):
        self.gpu_slots = gpu_slots
        self.cpu, self.gpu = set(), set()
        self.log = []
        self.fail_load = False
        self._next_id = 0

    def load(self, e):
        if self.fail_load:
            raise OSError(f"cannot read {e.path}")
        assert e.name not in self.cpu
        self.cpu.add(e.name)
        self.log.append(("load", e.name))
        self._next_id += 1
        return self._next_id

    def activate(self, e):
        assert e.name in self.cpu, "activate before load"
        assert len(self.gpu) < self.gpu_slots, "no free GPU slot"
        self.gpu.add(e.name)
        self.log.append(("activate", e.name))

    def deactivate(self, e):
        assert e.name in self.gpu
        self.gpu.discard(e.name)
        self.log.append(("deactivate", e.name))

    def unload(self, e):
        assert e.name in self.cpu and e.name not in self.gpu
        self.cpu.discard(e.name)
        self.log.append(("unload", e.name))


def make(gpu_slots=2, protected=0, capacity=32, half_life=100.0):
    from deltaserve.adapter_routing.index import CentroidIndex
    from deltaserve.adapter_routing.pool import AdapterPool

    now = {"t": 1000.0}
    be = FakeBackend(gpu_slots)
    pool = AdapterPool(be, CentroidIndex("enc", "query: ", 512),
                       capacity=capacity, gpu_slots=gpu_slots,
                       protected=protected, half_life_s=half_life,
                       clock=lambda: now["t"])
    return pool, be, now


def consistent(pool, be):
    """The pool's view of the tiers matches what the backend holds."""
    gpu = {n for n, e in pool.entries.items() if e.tier == "gpu"}
    cpu = {n for n, e in pool.entries.items() if e.tier in ("gpu", "cpu")}
    return gpu == be.gpu and cpu == be.cpu and len(be.gpu) <= be.gpu_slots


def test_registration():
    green("registration:")
    from deltaserve.adapter_routing.pool import PoolFull

    pool, be, _ = make(gpu_slots=2, capacity=3)
    a = pool.register("a", "/p/a", centroid=E[0], job="job-a")
    check("first adapter is loaded and prewarmed into a free slot",
          a.tier == "gpu" and be.log == [("load", "a"), ("activate", "a")])
    pool.register("b", "/p/b", centroid=E[1])
    c = pool.register("c", "/p/c", centroid=E[2])
    check("no free slot → stays on the CPU tier, nobody is evicted",
          c.tier == "cpu" and pool.evictions == 0 and be.gpu == {"a", "b"})
    check("pool and backend agree", consistent(pool, be))
    check("duplicate name rejected",
          "already in the pool" in (error_of(lambda: pool.register("a", "/x")) or ""))
    check("capacity enforced",
          (error_of(lambda: pool.register("d", "/p/d")) or "").startswith(PoolFull.__name__))
    check("all three are routable", sorted(pool.routable_names()) == ["a", "b", "c"])

    pool, be, _ = make()
    be.fail_load = True
    err = error_of(lambda: pool.register("x", "/missing", centroid=E[0]))
    check("backend failure → not registered, not routable",
          err is not None and "x" not in pool and pool.routable_names() == [])
    pool, be, _ = make()
    e = pool.register("noindex", "/p/n")
    check("an adapter without a centroid is in the pool but not routable",
          "noindex" in pool and pool.routable_names() == []
          and e.describe(0, 1)["indexed"] is False)
    check("prewarm=False keeps it off the GPU",
          pool.register("cold", "/p/c", prewarm=False).tier == "cpu")


def test_frequency_eviction():
    green("GPU slots by decayed frequency:")
    pool, be, now = make(gpu_slots=2, protected=0, half_life=100.0)
    for i, n in enumerate("abcd"):
        pool.register(n, f"/p/{n}", centroid=E[i])
    for _ in range(5):
        pool.note_hit("a")
    pool.note_hit("b")
    check("hits on adapters already in slots evict nothing",
          pool.evictions == 0 and be.gpu == {"a", "b"})
    pool.note_hit("c")
    check("a new adapter evicts the least-used one (b, 1 hit), not a (5 hits)",
          be.gpu == {"a", "c"} and pool.entries["b"].tier == "cpu"
          and pool.evictions == 1)
    check("pool and backend agree after eviction", consistent(pool, be))

    now["t"] += 1000          # 10 half-lives: a's 5 hits decay to ~0.005
    pool.note_hit("c")
    pool.note_hit("d")
    check("frequency decays: the once-hot adapter is now the victim",
          be.gpu == {"c", "d"} and pool.entries["a"].tier == "cpu")

    pool, be, now = make(gpu_slots=2, protected=0)
    for i, n in enumerate("abc"):
        pool.register(n, f"/p/{n}", centroid=E[i])
    pool.note_hit("a")
    pool.note_hit("a")
    pool.note_hit("b")
    check("the victim is never an adapter of the batch being built",
          pool.choose_victim(in_use={"b"}) == "a")
    e = pool.note_hit("c", in_use={"a", "b"})
    check("every slot in use → the hit stays on the CPU tier, nothing evicted",
          e.tier == "cpu" and be.gpu == {"a", "b"} and pool.evictions == 0
          and e.hits == 1)

    pool, be, now = make(gpu_slots=2, protected=1)
    for i, n in enumerate("abc"):
        pool.register(n, f"/p/{n}", centroid=E[i])
    for _ in range(3):
        pool.note_hit("a")
    pool.note_hit("b")
    check("the top adapter is protected", pool.protected_names() == {"a"})
    check("protected + in use → no victim",
          pool.choose_victim(in_use={"b"}) is None)
    pool.note_hit("c")
    check("the unprotected slot is the one that turns over",
          be.gpu == {"a", "c"})
    pool2, _, _ = make(gpu_slots=2, protected=9)
    check("protected is capped at the slot count", pool2.protected == 2)

    pool, be, now = make(gpu_slots=1, half_life=0.0)
    pool.register("a", "/p/a")
    pool.note_hit("a")
    now["t"] += 1e6
    check("half-life 0 = plain hit counts (no decay)",
          pool.status()["adapters"][0]["score"] == 1.0)


def test_offload_disable_remove():
    green("offload / disable / remove:")
    pool, be, _ = make(gpu_slots=2)
    for i, n in enumerate("abc"):
        pool.register(n, f"/p/{n}", centroid=E[i])
    pool.offload("a")
    check("offload drops GPU slot and RAM copy",
          pool.entries["a"].tier == "disk" and "a" not in be.cpu
          and be.log[-2:] == [("deactivate", "a"), ("unload", "a")])
    check("an offloaded adapter is still routable",
          "a" in pool.routable_names() and pool.route(E[0])[0] == "a")
    n0 = len(be.log)
    pool.note_hit("a")
    check("next hit cold-loads it and gives it a slot",
          pool.entries["a"].tier == "gpu" and pool.cold_loads == 1
          and be.log[n0:] == [("load", "a"), ("activate", "a")])
    check("offloading twice is harmless", error_of(lambda: (pool.offload("c"),
                                                            pool.offload("c"))) is None)

    pool.retire("b")
    check("retired → out of the index, no GPU slot, no memory held",
          "b" not in pool.routable_names() and pool.entries["b"].tier == "disk"
          and pool.entries["b"].level == "retired"
          and "b" not in be.cpu and pool.route(E[1])[0] != "b")
    e = pool.note_hit("b")
    check("a retired adapter is still servable by name: the hit loads and counts it",
          e.hits == 1 and "b" in be.cpu and "b" not in pool.routable_names())
    rep = pool.reactivate("b")
    check("reactivate goes back through admission and into the index",
          rep["level"] == "active" and pool.route(E[1])[0] == "b"
          and pool.entries["b"].level == "active")

    pool.remove("a")
    check("remove takes it out of the pool, the index and the backend",
          "a" not in pool and "a" not in pool.routable_names()
          and "a" not in be.cpu and "a" not in be.gpu)
    check("unknown adapter → clear error",
          "not in the pool" in (error_of(lambda: pool.offload("zzz")) or ""))
    check("pool and backend agree at the end", consistent(pool, be))


def test_route_and_fallback():
    green("routing through the pool:")
    pool, _, _ = make()
    check("empty pool → fallback", pool.route(E[0])[0] is None)
    pool.register("a", "/p/a", centroid=E[0])
    pool.register("b", "/p/b", centroid=E[1])
    name, score = pool.route(0.8 * E[1] + 0.2 * E[0], threshold=0.5)
    check("best match above the threshold is selected", name == "b" and score > 0.9)
    name, score = pool.route(E[5], threshold=0.5)
    check("no adapter close enough → fallback to base, score reported",
          name is None and abs(score) < 1e-6)
    hits = pool.entries["b"].hits
    pool.route(E[1])
    check("routing alone is not a hit", pool.entries["b"].hits == hits)


def test_ids_and_autosave():
    green("stable LoRA ids + registry file:")
    import json
    from pathlib import Path

    from deltaserve.adapter_routing.index import CentroidIndex
    from deltaserve.adapter_routing.pool import (
        LORA_ID_BASE,
        REGISTRY_FILE,
        AdapterPool,
        read_registry,
    )

    with tempfile.TemporaryDirectory() as d:
        pool = AdapterPool(FakeBackend(2), CentroidIndex("enc", "", 512),
                           gpu_slots=2, directory=d)
        check("no registry before the first adapter", read_registry(d) is None)
        a = pool.register("a", "/p/a", centroid=E[0])
        b = pool.register("b", "/p/b", centroid=E[1])
        check("ids come from the pool's own range, one per adapter",
              a.lora_int_id == LORA_ID_BASE + 1 and b.lora_int_id == LORA_ID_BASE + 2)
        reg = read_registry(d)
        check("registry written on registration, with ids and paths",
              [(r["name"], r["lora_int_id"], r["path"]) for r in reg["adapters"]]
              == [("a", a.lora_int_id, "/p/a"), ("b", b.lora_int_id, "/p/b")])
        pool.offload("a")
        pool.note_hit("a")
        check("offload + cold load keep the adapter's id",
              pool.entries["a"].lora_int_id == LORA_ID_BASE + 1)
        pool.retire("b")
        check("registry follows retire / offload",
              {r["name"]: r["level"] for r in read_registry(d)["adapters"]}
              == {"a": "active", "b": "retired"})
        pool.remove("a")
        c = pool.register("c", "/p/c")
        check("a removed adapter's id is not handed out again",
              c.lora_int_id == LORA_ID_BASE + 3
              and [r["name"] for r in read_registry(d)["adapters"]] == ["b", "c"])
        Path(d, REGISTRY_FILE).write_text("{ not json")
        check("a corrupt registry reads as none", read_registry(d) is None)
        Path(d, REGISTRY_FILE).write_text(json.dumps([1, 2]))
        check("an old-format registry reads as none", read_registry(d) is None)


def test_levels_and_admission():
    green("active / retired, admission, auto-retire:")
    import os
    from pathlib import Path

    from deltaserve.adapter_routing.index import CentroidIndex
    from deltaserve.adapter_routing.pool import (
        AdapterPool,
        PoolFull,
        read_registry,
        routable_path,
    )

    rng = np.random.default_rng(0)
    common = rng.normal(size=DIM) * 3

    def topic(i, n=40, noise=0.15):
        base = np.zeros(DIM)
        base[i % DIM] = 1.0
        v = common + base + noise * rng.normal(size=(n, DIM))
        return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)

    def make(cap, d=None, **kw):
        now = {"t": 1000.0}
        be = FakeBackend(2)
        pool = AdapterPool(be, CentroidIndex("enc", "", 512), capacity=cap,
                           gpu_slots=2, directory=d, clock=lambda: now["t"],
                           grace_s=600.0, **kw)
        return pool, be, now

    def admit(pool, name, i, **kw):
        t = topic(i)
        return pool.admit(name, f"/p/{name}", centroid=t.mean(0), refs=t[:24], **kw)

    with tempfile.TemporaryDirectory() as d:
        pool, be, now = make(3, d)
        r0 = admit(pool, "a", 0)
        check("first adapter: admitted active, nothing to compare with",
              r0["level"] == "active" and r0["pool_before"] == 0)
        r1 = admit(pool, "b", 1)
        check("admission report: self-recall, nearest adapter, what it takes",
              r1["level"] == "active" and r1["self_recall"] >= 0.9
              and r1["nearest"] == "a" and r1["stolen_max"] <= 0.1)
        check("admission does not load weights unless asked (lazy)",
              be.log == [] and pool.entries["b"].tier == "disk")
        v1 = pool.version
        reg = read_registry(d)
        check("one published state: the registry names the routable index",
              reg["version"] == v1 and os.path.isfile(routable_path(d))
              and Path(routable_path(d)).name == reg["routable"])
        admit(pool, "c", 2)
        check("each admission publishes a new version",
              pool.version == v1 + 1 and read_registry(d)["version"] == v1 + 1)
        pool.note_hit("a")
        pool.save(d)
        check("a usage-only save does not make a new routable version",
              pool.version == v1 + 1)

        tw = topic(0, noise=0.05)                 # a near-duplicate of 'a'
        rt = pool.admit("a-twin", "/p/a-twin", centroid=tw.mean(0), refs=tw[:24])
        check("a full active set within the grace period: nobody can be retired, "
              "the newcomer is admitted retired",
              rt["level"] == "retired" and "grace" in rt["reason"]
              and rt["retired"] is None and len(pool.active_names()) == 3)
        check("...and the report still says how similar it is to 'a'",
              rt["nearest"] == "a" and rt["stolen_max"] > 0.2)

        now["t"] += 700                           # past the grace period
        for _ in range(3):
            pool.note_hit("b")
        pool.note_hit("c")
        pool.pin("a")                             # least used, but pinned
        v = pool.version
        r4 = admit(pool, "d", 3)
        check("full + past grace: the least-used unpinned adapter is retired, "
              "the newcomer is active",
              r4["level"] == "active" and r4["retired"] == "c"
              and sorted(pool.active_names()) == ["a", "b", "d"]
              and pool.entries["c"].level == "retired")
        check("newcomer in and victim out are ONE published step",
              pool.version == v + 1
              and sorted(CentroidIndex.load(routable_path(d)).names)
              == ["a", "b", "d"])
        check("the retired adapter keeps its centroid and reference samples",
              pool.entries["c"].centroid is not None
              and pool.entries["c"].refs is not None)
        rr = pool.reactivate("c")
        check("reactivating it needs no encoding and is again subject to the cap "
              "(newcomer 'd' is in its grace period → 'b'… or stays retired)",
              rr["level"] in ("active", "retired")
              and len(pool.active_names()) == 3)
        kept = sorted(f for f in os.listdir(d) if f.startswith("routable-"))
        check("old routable versions are cleaned up (current + previous kept)",
              len(kept) <= 2 and Path(routable_path(d)).name in kept)

        p2, _, _ = make(1, when_full="admit_retired")
        admit(p2, "x", 0)
        check("when_full: admit_retired → the newcomer waits in retired",
              admit(p2, "y", 1)["level"] == "retired" and p2.active_names() == ["x"])
        p3, _, _ = make(1, when_full="reject")
        admit(p3, "x", 0)
        err = error_of(lambda: admit(p3, "y", 1))
        check("when_full: reject → refused, the pool unchanged",
              err is not None and err.startswith(PoolFull.__name__)
              and list(p3.entries) == ["x"])

        p4, _, _ = make(8, max_stolen=0.2)
        admit(p4, "a", 0)
        admit(p4, "b", 1)
        r = p4.admit("a-twin", "/p/t", centroid=tw.mean(0), refs=tw[:24])
        check("optional rule admission_max_stolen: a candidate that takes too "
              "much of another adapter is admitted retired",
              r["level"] == "retired" and "takes" in r["reason"])
        p5, _, _ = make(8)
        admit(p5, "a", 0)
        admit(p5, "b", 1)
        r = p5.admit("a-twin", "/p/t", centroid=tw.mean(0), refs=tw[:24])
        check("default: adapters that are merely similar are NOT turned away",
              r["level"] == "active" and r["stolen_max"] > 0.2)

        p6, be6, _ = make(8)
        e0 = p6.register("v", "/p/v1", centroid=topic(4).mean(0))
        id0 = e0.lora_int_id
        r = p6.admit("v", "/p/v2", centroid=topic(4).mean(0), refs=topic(4)[:8])
        check("a new version under the same name replaces the old one in place, "
              "with a new LoRA id for the new weights",
              r["level"] == "active" and p6.entries["v"].path == "/p/v2"
              and p6.entries["v"].lora_int_id != id0 and len(p6.entries) == 1)
        r = p6.admit("nocentroid", "/p/n")
        check("no centroid → retired level (servable by name only)",
              r["level"] == "retired" and "nocentroid" not in p6.routable_names())


def test_save_restore():
    green("registry save / restore:")
    pool, _, _ = make(gpu_slots=1)
    for i, n in enumerate("abc"):
        pool.register(n, f"/p/{n}", centroid=E[i], job=f"job-{n}")
    pool.offload("b")
    pool.retire("c")
    pool.pin("a")
    with tempfile.TemporaryDirectory() as d:
        pool.save(d)
        from deltaserve.adapter_routing.index import CentroidIndex

        saved = CentroidIndex("enc", "query: ", 512)
        for i, n in enumerate("abc"):
            saved.add(n, E[i])
        new, be, _ = make(gpu_slots=1)
        check("nothing to restore from an empty directory",
              make()[0].restore(d + "/nope") == 0)
        n = new.restore(d, centroids=saved)
        check("all adapters restored", n == 3 and sorted(new.entries) == list("abc"))
        check("offloaded stays offloaded, others are loaded",
              new.entries["b"].tier == "disk" and new.entries["a"].tier == "gpu"
              and "b" not in be.cpu)
        check("retired stays retired and unroutable; a pin survives too",
              new.entries["c"].level == "retired" and new.entries["a"].pinned
              and sorted(new.routable_names()) == ["a", "b"])
        check("job names carried over", new.entries["a"].job == "job-a")
        check("LoRA ids survive the restart",
              {n: e.lora_int_id for n, e in new.entries.items()}
              == {n: e.lora_int_id for n, e in pool.entries.items()})
        later = new.register("d", "/p/d")
        check("an id is never reused after a restart",
              later.lora_int_id > max(e.lora_int_id for e in pool.entries.values()))
        check("restored pool is consistent with its backend", consistent(new, be))


def main():
    green("=== adapter pool ===")
    test_registration()
    test_frequency_eviction()
    test_offload_disable_remove()
    test_route_and_fallback()
    test_ids_and_autosave()
    test_levels_and_admission()
    test_save_restore()
    print()
    if _failed:
        print(f"{_RED}{_failed} failed{_RESET}, {_passed} passed")
        sys.exit(1)
    green(f"all {_passed} checks passed")


if __name__ == "__main__":
    main()
