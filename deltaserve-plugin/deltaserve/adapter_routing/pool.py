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
import re
import threading
import time
from dataclasses import dataclass

from deltaserve import dprint
from deltaserve.adapter_routing.index import CentroidIndex

GPU, CPU, DISK = "gpu", "cpu", "disk"
# Level: ACTIVE adapters are the ones auto-routing chooses among (they define
# the pool's centroid mean, statistics and threshold). RETIRED ones are never
# selected and hold no memory, but stay servable by name; their centroid and
# reference samples are kept, so re-activating one costs no encoding.
ACTIVE, RETIRED = "active", "retired"
RETIRE_LEAST_USED, ADMIT_RETIRED, REJECT = "retire_least_used", "admit_retired", "reject"
REGISTRY_FILE = "pool.json"
# The centroids of the adapters that may be selected right now (in the pool and
# enabled) — what the engine's router reads. The indexing stage's own file holds
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
    level: str = ACTIVE
    pinned: bool = False          # never auto-retired
    admitted_at: float = 0.0
    lora_int_id: int | None = None
    hits: int = 0
    score: float = 0.0            # decayed hit count as of ``score_at``
    score_at: float = 0.0
    centroid: object = None       # kept so a disabled adapter can be re-enabled
    refs: object = None           # its reference-sample embeddings (may be None)

    @property
    def enabled(self) -> bool:
        return self.level == ACTIVE

    def describe(self, now: float, half_life: float) -> dict:
        return {"name": self.name, "path": self.path, "job": self.job,
                "lora_int_id": self.lora_int_id, "level": self.level,
                "pinned": self.pinned,
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
                 directory: str | None = None,
                 when_full: str = RETIRE_LEAST_USED, grace_s: float = 600.0,
                 min_pool: int = 4, min_self_recall: float = 0.0,
                 max_stolen: float = 1.0) -> None:
        self.backend = backend
        # The engine's replay of vLLM's adapter caches. With it the tiers are
        # facts read from there (``sync_tiers``); without it (tests, no
        # engine) they are the pool's own LRU-by-frequency bookkeeping.
        self.mirror = None
        self.index = index if index is not None else CentroidIndex()
        # ``capacity`` bounds the ACTIVE adapters only (a quality knob: the more
        # adapters compete, the more often routing picks a neighbour).
        self.capacity, self.gpu_slots = int(capacity), int(gpu_slots)
        self.when_full, self.grace_s = when_full, float(grace_s)
        self.min_pool = int(min_pool)
        self.min_self_recall, self.max_stolen = float(min_self_recall), float(max_stolen)
        self.version = 0              # of the published routable index
        self._index_dirty = True
        self._defer = 0
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
        if self.directory and not self._defer:
            self.save(self.directory)

    def _index_add(self, e: "PoolEntry") -> None:
        self.index.add(e.name, e.centroid, refs=e.refs)
        self._index_dirty = True

    def _index_remove(self, name: str) -> None:
        if self.index.remove(name):
            self._index_dirty = True

    def active_names(self) -> list[str]:
        return [n for n, e in self.entries.items() if e.level == ACTIVE]

    def retired_names(self) -> list[str]:
        return [n for n, e in self.entries.items() if e.level == RETIRED]

    @_locked
    def register(self, name: str, path: str, centroid=None,
                 job: str | None = None, prewarm: bool = True,
                 lora_int_id: int | None = None, load: bool = True,
                 refs=None, level: str = ACTIVE, pinned: bool = False) -> PoolEntry:
        """Add an adapter (a job that reached ``ready``). With ``load`` it is
        loaded to the CPU tier and, with ``prewarm``, into a GPU slot if one is
        free — never by evicting. ``load=False`` only registers it: the weights
        stay on disk until the first request for it (lowest cost; nothing is
        asked of the engine)."""
        if name in self.entries:
            raise ValueError(f"adapter '{name}' is already in the pool")
        if level == ACTIVE and len(self.active_names()) >= self.capacity:
            raise PoolFull(f"adapter pool is full ({self.capacity} active)")
        if lora_int_id is None:
            lora_int_id = self._next_id
        self._next_id = max(self._next_id, lora_int_id + 1)
        e = PoolEntry(name=name, path=path, job=job, score_at=self._clock(),
                      centroid=centroid, lora_int_id=lora_int_id, refs=refs,
                      level=level, pinned=pinned, admitted_at=self._clock())
        self.entries[name] = e
        try:
            if level == ACTIVE:
                if load:
                    self._to_cpu(e)
                if centroid is not None:
                    self._index_add(e)
                if load and prewarm and self._in_tier(GPU) < self.gpu_slots:
                    self._activate(e)
        except Exception:
            self.entries.pop(name, None)
            self._index_remove(name)
            raise
        dprint(f"[adapter-pool] + {name} ({e.level}, {e.tier}) | "
               f"{len(self.active_names())}/{self.capacity} active, "
               f"{len(self.retired_names())} retired")
        self._autosave()
        return e

    @_locked
    def remove(self, name: str) -> None:
        """Take an adapter out of the pool entirely."""
        e = self._get(name)
        self._to_disk(e)
        self._index_remove(name)
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
        # Where it really is decides whether vLLM has to drop it.
        self.sync_tiers()
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
        e = self._get(name)      # a retired adapter is still servable by name
        now = self._clock()
        e.score = self._score(e, now) + 1.0
        e.score_at = now
        e.hits += 1
        if e.tier == DISK:
            self.cold_loads += 1
        if self.mirror is None:
            self.ensure_gpu(name, in_use=in_use)
        return e

    @_locked
    def sync_tiers(self, mirror=None) -> None:
        """Set every adapter's tier from where vLLM really has it (the engine's
        ``LoraCacheMirror``): the pool does not move adapters between tiers
        itself when one is attached."""
        mirror = mirror if mirror is not None else self.mirror
        if mirror is None:
            return
        for e in self.entries.values():
            i = e.lora_int_id
            e.tier = GPU if mirror.in_slot(i) else CPU if mirror.in_memory(i) else DISK

    def route(self, embedding, threshold: float = -1.0) -> tuple[str | None, float]:
        """ARIADNE top-1 over the enabled, indexed adapters; (None, score) is
        the fallback to the base model. Does not count as a hit."""
        return self.index.route(embedding, threshold)

    # ----------------------------------------------------------- admission
    def _retire_victim(self, exclude: str | None = None) -> str | None:
        """Who leaves the active set to make room: the least used adapter that
        is not pinned and past its grace period; among equally used ones, the
        most redundant (closest to another active centroid)."""
        now = self._clock()
        cands = [e for e in self.entries.values()
                 if e.level == ACTIVE and not e.pinned and e.name != exclude
                 and now - e.admitted_at >= self.grace_s]
        if not cands:
            return None

        def redundancy(e):
            if e.centroid is None or len(self.index) < 2 or e.name not in self.index:
                return 0.0
            s = self.index.scores(e.centroid)
            s[self.index.names.index(e.name)] = -1.0
            return float(s.max())

        return min(cands, key=lambda e: (round(self._score(e, now), 6),
                                         -redundancy(e), e.name)).name

    @_locked
    def admit(self, name: str, path: str, centroid=None, refs=None,
              job: str | None = None, load: bool = False, prewarm: bool = False,
              pinned: bool = False) -> dict:
        """Bring an adapter into the pool — the one way in, also for a retired
        adapter being re-activated or a new version replacing an old one.

        The candidate is compared with the active adapters on reference samples
        (``decision.admission_report``); nothing about the serving pool changes
        until the new state is complete, then it is published in ONE step
        (``save``), together with any adapter retired to make room. Returns the
        report, including the ``level`` it ended up at and why."""
        from deltaserve.adapter_routing.decision import admission_report

        old = self.entries.get(name)
        if centroid is None and old is not None:
            centroid, refs = old.centroid, old.refs
        report = {"name": name, "level": ACTIVE, "reason": "", "retired": None}
        if centroid is None:
            report.update(level=RETIRED, reason="no centroid: servable by name only")
        else:
            report.update(admission_report(self.index, name, centroid, refs,
                                           self.min_pool))
            sr = report.get("self_recall")
            if sr is not None and sr < self.min_self_recall:
                report.update(level=RETIRED, reason=(
                    f"self-recall {sr} < {self.min_self_recall}"))
            elif report.get("stolen_max", 0.0) > self.max_stolen:
                report.update(level=RETIRED, reason=(
                    f"takes {report['stolen_max']} of {report['stolen_from']}"))
        was_active = old is not None and old.level == ACTIVE
        victim = None
        if report["level"] == ACTIVE and not was_active \
                and len(self.active_names()) >= self.capacity:
            if self.when_full == REJECT:
                raise PoolFull(f"adapter pool is full ({self.capacity} active)")
            if self.when_full == RETIRE_LEAST_USED:
                victim = self._retire_victim(exclude=name)
            if victim is None:
                report.update(level=RETIRED, reason=(
                    "active set is full" if self.when_full == ADMIT_RETIRED
                    else "active set is full and nobody can be retired "
                         "(pinned or within the grace period)"))
        self._defer += 1
        try:
            if victim is not None:
                self.retire(victim)
                report["retired"] = victim
            if old is None:
                self.register(name, path, centroid=centroid, refs=refs, job=job,
                              load=load, prewarm=prewarm, level=report["level"],
                              pinned=pinned)
            else:
                if path != old.path:        # a new version: new weights, new id
                    self._to_disk(old)
                    old.path, old.lora_int_id = path, self._next_id
                    self._next_id += 1
                old.centroid, old.refs, old.job = centroid, refs, job or old.job
                old.pinned = old.pinned or pinned
                if report["level"] == ACTIVE:
                    old.level, old.admitted_at = ACTIVE, self._clock()
                    self._index_add(old)
                elif old.level == ACTIVE:
                    self.retire(name)
        finally:
            self._defer -= 1
        self._autosave()
        dprint(f"[adapter-pool] admission of {name}: {report['level']}"
               + (f" ({report['reason']})" if report["reason"] else "")
               + (f", retired {victim}" if victim else "")
               + f" | self-recall {report.get('self_recall')} nearest "
                 f"{report.get('nearest')} ({report.get('nearest_cosine')}) "
                 f"takes up to {report.get('stolen_max')}")
        return report

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
    def retire(self, name: str) -> PoolEntry:
        """Stop selecting this adapter and free its memory. It stays servable
        by name (loaded from disk when used); its centroid and reference
        samples are kept for a later ``reactivate``."""
        e = self._get(name)
        e.level = RETIRED
        self._index_remove(name)
        self._to_disk(e)
        self._autosave()
        return e

    @_locked
    def reactivate(self, name: str) -> dict:
        """Bring a retired adapter back through admission (no encoding)."""
        e = self._get(name)
        return self.admit(name, e.path)

    disable, enable = retire, reactivate

    @_locked
    def pin(self, name: str, pinned: bool = True) -> PoolEntry:
        """A pinned adapter is never retired automatically."""
        e = self._get(name)
        e.pinned = bool(pinned)
        self._autosave()
        return e

    def routable_names(self) -> list[str]:
        return list(self.index.names)

    @_locked
    def status(self) -> dict:
        now = self._clock()
        return {
            "capacity": self.capacity, "gpu_slots": self.gpu_slots,
            "active": len(self.active_names()),
            "retired": len(self.retired_names()), "version": self.version,
            "protected": sorted(self.protected_names(now)),
            "evictions": self.evictions, "cold_loads": self.cold_loads,
            "adapters": [e.describe(now, self.half_life_s)
                         for e in sorted(self.entries.values(),
                                         key=lambda e: e.name)],
        }

    # --------------------------------------------------------- persistence
    @_locked
    def save(self, directory: str) -> None:
        """Publish the pool: one atomic step. The routable index (the active
        adapters' centroids + reference samples) goes to a NEW versioned file
        first; the registry — names, paths, LoRA ids, levels, and which
        routable file is current — is then swapped in with a single rename.
        Readers (the engine's router, the API servers' resolver) follow the
        registry, so they see the old pool or the new one, never a mix; the
        pool serves throughout."""
        os.makedirs(directory, exist_ok=True)
        if self._index_dirty or not self.version:
            self.version += 1
            self.index.save(os.path.join(directory, routable_name(self.version)))
            self._index_dirty = False
        now = self._clock()
        reg = {"next_id": self._next_id, "version": self.version,
               "routable": routable_name(self.version), "adapters": [
                   {"name": e.name, "path": e.path, "job": e.job,
                    "lora_int_id": e.lora_int_id, "level": e.level,
                    "pinned": e.pinned, "enabled": e.enabled,
                    "offloaded": e.tier == DISK, "tier": e.tier,
                    "hits": e.hits, "score": round(self._score(e, now), 4)}
                   for e in self.entries.values()]}
        tmp = os.path.join(directory, REGISTRY_FILE + ".tmp")
        with open(tmp, "w") as f:
            json.dump(reg, f, indent=1)
        os.replace(tmp, os.path.join(directory, REGISTRY_FILE))
        for f in os.listdir(directory):          # keep the current and previous
            m = _ROUTABLE_RE.fullmatch(f)
            if m and int(m.group(1)) < self.version - 1:
                try:
                    os.unlink(os.path.join(directory, f))
                except OSError:
                    pass

    @_locked
    def restore(self, directory: str, centroids: CentroidIndex | None = None) -> int:
        """Re-register the adapters of a saved registry with the LoRA ids,
        levels and pins they had (centroids and reference samples from the
        given index). Returns how many."""
        reg = read_registry(directory)
        if reg is None:
            return 0
        self._defer += 1
        try:
            self.version = max(self.version, int(reg.get("version", 0)))
            for r in reg["adapters"]:
                c = refs = None
                if centroids is not None and r["name"] in centroids:
                    c = centroids.centroid_of(r["name"])
                    refs = centroids.refs.get(r["name"])
                level = r.get("level") or (ACTIVE if r.get("enabled", True) else RETIRED)
                offloaded = bool(r.get("offloaded"))
                self.register(r["name"], r["path"], centroid=c, job=r.get("job"),
                              prewarm=not offloaded, load=not offloaded,
                              lora_int_id=r.get("lora_int_id"), refs=refs,
                              level=level, pinned=bool(r.get("pinned")))
            self._next_id = max(self._next_id, int(reg.get("next_id", 0)))
        finally:
            self._defer -= 1
        self._index_dirty = True
        self._autosave()
        return len(reg["adapters"])


_ROUTABLE_RE = re.compile(r"routable-(\d+)\.npz")


def routable_name(version: int) -> str:
    return f"routable-{int(version)}.npz"


def routable_path(directory: str) -> str | None:
    """The routable index the registry of ``directory`` currently points at."""
    reg = read_registry(directory)
    if reg is None or not reg.get("routable"):
        return None
    return os.path.join(directory, reg["routable"])


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
