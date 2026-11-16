# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The adapter pool: every auto-routable adapter and where it currently lives.

An adapter is in one of three tiers:

- ``gpu``  — in a LoRA GPU slot, usable in a batch right now;
- ``cpu``  — weights in host RAM (the default home of every adapter);
- ``disk`` — offloaded on request; still routable, reloaded on its next hit.

plus an ``enabled`` flag: a disabled adapter is taken out of the centroid index
and is never selected.

The pool is bookkeeping and policy. What actually moves weights is a
**backend** with four calls — ``load`` (disk → cpu), ``activate`` (cpu → gpu),
``deactivate`` (gpu → cpu), ``unload`` (→ disk); in the server that is a thin
adapter over vLLM's LoRA manager, in tests a fake.

GPU slots are kept by **frequency**: every hit bumps the adapter's score, which
decays with a half-life. When a slot is needed the victim is the lowest-scoring
adapter in a slot that is neither in use by the current batch nor one of the
``protected`` highest-scoring ones. If no slot can be freed the hit stays on
the CPU tier (the request waits for a slot, as with vLLM's own ``max_loras``).

Pure Python + numpy (through the centroid index); no torch / vLLM import.
"""

import functools
import json
import os
import threading
import time
from dataclasses import dataclass

from deltaserve import dprint
from deltaserve.adapter_routing.index import CentroidIndex

GPU, CPU, DISK = "gpu", "cpu", "disk"
REGISTRY_FILE = "pool.json"
# The centroids of the adapters that may be selected right now (in the pool and
# enabled) — what the router service reads. The indexing stage's own file holds
# every centroid ever built, including jobs that are not ready yet.
ROUTABLE_FILE = "routable.npz"
# Pool adapters get their LoRA int id from the pool, in a range of their own:
# vLLM keys everything by that id and each API server process would otherwise
# number runtime-loaded adapters with its own counter (see resolver.py).
LORA_ID_BASE = 100_000


@dataclass
class PoolEntry:
    name: str
    path: str
    job: str | None = None
    tier: str = DISK
    enabled: bool = True
    lora_int_id: int | None = None
    hits: int = 0
    score: float = 0.0            # decayed hit count as of ``score_at``
    score_at: float = 0.0
    centroid: object = None       # kept so a disabled adapter can be re-enabled
    refs: object = None           # its reference-sample embeddings (may be None)

    def describe(self, now: float, half_life: float) -> dict:
        return {"name": self.name, "path": self.path, "job": self.job,
                "lora_int_id": self.lora_int_id,
                "tier": self.tier, "enabled": self.enabled, "hits": self.hits,
                "score": round(decayed(self.score, now - self.score_at,
                                       half_life), 4),
                "indexed": self.centroid is not None}


def decayed(score: float, dt: float, half_life: float) -> float:
    if half_life <= 0 or dt <= 0:
        return score
    return score * 0.5 ** (dt / half_life)


class PoolFull(Exception):
    pass


def _locked(fn):
    """The pool is used from the engine loop and from the job driver thread."""
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return fn(self, *args, **kwargs)
    return wrapper


class AdapterPool:
    def __init__(self, backend, index: CentroidIndex | None = None,
                 capacity: int = 32, gpu_slots: int = 1, protected: int = 0,
                 half_life_s: float = 300.0, clock=time.monotonic,
                 directory: str | None = None) -> None:
        self.backend = backend
        self.index = index if index is not None else CentroidIndex()
        self.capacity, self.gpu_slots = int(capacity), int(gpu_slots)
        self.protected = max(0, min(int(protected), self.gpu_slots))
        self.half_life_s = float(half_life_s)
        self._clock = clock
        self.entries: dict[str, PoolEntry] = {}
        self.evictions = 0
        self.cold_loads = 0
        self._lock = threading.RLock()
        # Where the registry is kept (the API servers' resolver reads it);
        # written after every change when set.
        self.directory = directory
        self._next_id = LORA_ID_BASE + 1      # never reused, also after a restart

    # ---------------------------------------------------------- membership
    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, name: str) -> bool:
        return name in self.entries

    def _get(self, name: str) -> PoolEntry:
        try:
            return self.entries[name]
        except KeyError:
            raise KeyError(f"adapter '{name}' is not in the pool") from None

    def _autosave(self) -> None:
        if self.directory:
            self.save(self.directory)

    @_locked
    def register(self, name: str, path: str, centroid=None,
                 job: str | None = None, prewarm: bool = True,
                 lora_int_id: int | None = None, load: bool = True,
                 refs=None) -> PoolEntry:
        """Add an adapter (a job that reached ``ready``). With ``load`` it is
        loaded to the CPU tier and, with ``prewarm``, into a GPU slot if one is
        free — never by evicting. ``load=False`` only registers it: the weights
        stay on disk until the first request for it (lowest cost; nothing is
        asked of the engine)."""
        if name in self.entries:
            raise ValueError(f"adapter '{name}' is already in the pool")
        if len(self.entries) >= self.capacity:
            raise PoolFull(f"adapter pool is full ({self.capacity})")
        if lora_int_id is None:
            lora_int_id = self._next_id
        self._next_id = max(self._next_id, lora_int_id + 1)
        e = PoolEntry(name=name, path=path, job=job, score_at=self._clock(),
                      centroid=centroid, lora_int_id=lora_int_id, refs=refs)
        self.entries[name] = e
        try:
            if load:
                self._to_cpu(e)
            if centroid is not None:
                self.index.add(name, centroid, refs=refs)
            if load and prewarm and self._in_tier(GPU) < self.gpu_slots:
                self._activate(e)
        except Exception:
            self.entries.pop(name, None)
            self.index.remove(name)
            raise
        dprint(f"[adapter-pool] + {name} ({e.tier}) | {len(self)}/{self.capacity}")
        self._autosave()
        return e

    @_locked
    def remove(self, name: str) -> None:
        """Take an adapter out of the pool entirely."""
        e = self._get(name)
        self._to_disk(e)
        self.index.remove(name)
        del self.entries[name]
        self._autosave()

    # --------------------------------------------------------------- tiers
    def _in_tier(self, tier: str) -> int:
        return sum(1 for e in self.entries.values() if e.tier == tier)

    def _to_cpu(self, e: PoolEntry) -> None:
        if e.tier == DISK:
            self.backend.load(e)
            e.tier = CPU
        elif e.tier == GPU:
            self.backend.deactivate(e)
            e.tier = CPU

    def _activate(self, e: PoolEntry) -> None:
        if e.tier == DISK:
            self._to_cpu(e)
        if e.tier != GPU:
            self.backend.activate(e)
            e.tier = GPU

    def _to_disk(self, e: PoolEntry) -> None:
        if e.tier == GPU:
            self.backend.deactivate(e)
            e.tier = CPU
        if e.tier == CPU:
            self.backend.unload(e)
            e.tier = DISK

    def _score(self, e: PoolEntry, now: float) -> float:
        return decayed(e.score, now - e.score_at, self.half_life_s)

    def protected_names(self, now: float | None = None) -> set[str]:
        """The ``protected`` highest-scoring adapters among those in GPU slots."""
        now = self._clock() if now is None else now
        in_gpu = sorted((e for e in self.entries.values() if e.tier == GPU),
                        key=lambda e: (-self._score(e, now), e.name))
        return {e.name for e in in_gpu[:self.protected]}

    def choose_victim(self, in_use=(), exclude: str | None = None) -> str | None:
        """Which GPU-slot adapter to evict for a new one: the lowest decayed
        score, not in ``in_use`` (the batch being built), not protected. None
        when every slot is in use or protected."""
        now = self._clock()
        keep = set(in_use) | self.protected_names(now)
        if exclude:
            keep.add(exclude)
        cands = [e for e in self.entries.values()
                 if e.tier == GPU and e.name not in keep]
        if not cands:
            return None
        return min(cands, key=lambda e: (self._score(e, now), e.name)).name

    @_locked
    def ensure_gpu(self, name: str, in_use=()) -> bool:
        """Bring an adapter into a GPU slot, evicting by frequency if needed.
        False if no slot can be freed right now."""
        e = self._get(name)
        if e.tier == GPU:
            return True
        if self._in_tier(GPU) >= self.gpu_slots:
            victim = self.choose_victim(in_use=in_use, exclude=name)
            if victim is None:
                if e.tier == DISK:
                    self._to_cpu(e)       # at least get it off the disk
                return False
            self._to_cpu(self.entries[victim])
            self.evictions += 1
        self._activate(e)
        return True

    # ---------------------------------------------------------------- hits
    @_locked
    def note_hit(self, name: str, in_use=()) -> PoolEntry:
        """A request uses this adapter (routed or named): bump its score and
        make it usable — cold-load it if it was offloaded, give it a GPU slot
        if one can be had."""
        e = self._get(name)
        if not e.enabled:
            raise ValueError(f"adapter '{name}' is disabled")
        now = self._clock()
        e.score = self._score(e, now) + 1.0
        e.score_at = now
        e.hits += 1
        if e.tier == DISK:
            self.cold_loads += 1
        self.ensure_gpu(name, in_use=in_use)
        return e

    def route(self, embedding, threshold: float = -1.0) -> tuple[str | None, float]:
        """ARIADNE top-1 over the enabled, indexed adapters; (None, score) is
        the fallback to the base model. Does not count as a hit."""
        return self.index.route(embedding, threshold)

    # ----------------------------------------------------------- operator
    @_locked
    def offload(self, name: str) -> PoolEntry:
        """Drop the adapter's weights from GPU and RAM. It stays routable and
        is loaded back from disk on its next hit."""
        e = self._get(name)
        self._to_disk(e)
        dprint(f"[adapter-pool] offloaded {name}")
        self._autosave()
        return e

    @_locked
    def disable(self, name: str) -> PoolEntry:
        """Stop selecting this adapter: out of the centroid index, out of its
        GPU slot. Its weights stay in RAM unless it is also offloaded."""
        e = self._get(name)
        e.enabled = False
        self.index.remove(name)
        if e.tier == GPU:
            self._to_cpu(e)
        self._autosave()
        return e

    @_locked
    def enable(self, name: str) -> PoolEntry:
        e = self._get(name)
        e.enabled = True
        if e.centroid is not None:
            self.index.add(name, e.centroid, refs=e.refs)
        self._autosave()
        return e

    def routable_names(self) -> list[str]:
        return list(self.index.names)

    @_locked
    def status(self) -> dict:
        now = self._clock()
        return {
            "capacity": self.capacity, "gpu_slots": self.gpu_slots,
            "protected": sorted(self.protected_names(now)),
            "evictions": self.evictions, "cold_loads": self.cold_loads,
            "adapters": [e.describe(now, self.half_life_s)
                         for e in sorted(self.entries.values(),
                                         key=lambda e: e.name)],
        }

    # --------------------------------------------------------- persistence
    @_locked
    def save(self, directory: str) -> None:
        """Write the registry (names, paths, LoRA ids, flags) next to the
        centroid index: a restart finds the same pool, and the API servers'
        resolver learns from it which adapters exist and where."""
        os.makedirs(directory, exist_ok=True)
        reg = {"next_id": self._next_id, "adapters": [
            {"name": e.name, "path": e.path, "job": e.job,
             "lora_int_id": e.lora_int_id, "enabled": e.enabled,
             "offloaded": e.tier == DISK, "tier": e.tier,
             "score": round(self._score(e, self._clock()), 4)}
            for e in self.entries.values()]}
        tmp = os.path.join(directory, REGISTRY_FILE + ".tmp")
        with open(tmp, "w") as f:
            json.dump(reg, f, indent=1)
        os.replace(tmp, os.path.join(directory, REGISTRY_FILE))
        self.index.save(os.path.join(directory, ROUTABLE_FILE))

    @_locked
    def restore(self, directory: str, centroids: CentroidIndex | None = None) -> int:
        """Re-register the adapters of a saved registry, with the LoRA ids
        they had (centroids from the given index). Returns how many."""
        reg = read_registry(directory)
        if reg is None:
            return 0
        keep, self.directory = self.directory, None     # no rewrite per adapter
        try:
            for r in reg["adapters"]:
                c = refs = None
                if centroids is not None and r["name"] in centroids:
                    c = centroids.centroid_of(r["name"])
                    refs = centroids.refs.get(r["name"])
                self.register(r["name"], r["path"], centroid=c, job=r.get("job"),
                              prewarm=not r.get("offloaded"),
                              lora_int_id=r.get("lora_int_id"), refs=refs)
                if r.get("offloaded"):
                    self.offload(r["name"])
                if not r.get("enabled", True):
                    self.disable(r["name"])
            self._next_id = max(self._next_id, int(reg.get("next_id", 0)))
        finally:
            self.directory = keep
        self._autosave()
        return len(reg["adapters"])


def read_registry(directory: str) -> dict | None:
    """The saved registry of ``directory``, or None when there is none."""
    path = os.path.join(directory, REGISTRY_FILE)
    try:
        with open(path) as f:
            reg = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(reg, dict) or not isinstance(reg.get("adapters"), list):
        return None
    return reg
