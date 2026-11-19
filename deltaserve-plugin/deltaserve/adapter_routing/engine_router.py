# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Adapter routing inside the engine: the scheduler chooses the adapter.

A routed request reaches the engine as a base-model request carrying the route
encoder's token ids (``route_middleware.ROUTE_HEADER`` in its
``trace_headers``). ``EngineRouter`` keeps it out of the scheduler until its
adapter is known:

    add_request ──► hold ──► [encode step] ──► decide ──► attach the adapter
                                                          ──► scheduler queue

``route_pending`` runs at the start of every ``schedule()``: one encode step
for the held requests (``deltaserve_route_encode`` on the workers, a batch of
at most ``route_encode_max_tokens`` padded tokens — the rest waits for the next
step), a decision per embedding against the pool's routable index, then each
request enters the scheduler exactly as if it had named that adapter. Its
prefill is scheduled by the same ``schedule()`` call.

Why here and not in the API server: the encode needs the GPU, and anything on
the GPU next to a model step adds to that step. As a step of its own, issued
by the one component that orders everything else on the card, it costs its own
time and nothing else — and the request's ``arrival_time`` is its real arrival.

**Adapter load.** A request — routed or naming its adapter — needs that
adapter in a GPU slot. vLLM loads lazily, at the start of the first step that
uses it, so where the adapter is decides what the request costs:

    GPU slot                      nothing
    host memory                   the step copies it into a slot (about 4 ms)
    disk, other work this step    COLD START: the request is held for ONE step;
                                  that step carries a preload hint, the workers
                                  read the adapter on a background thread while
                                  they run it; the next step takes the request
                                  (about 6 ms: register + copy into a slot)
    disk, nothing else to run     scheduled at once; the step reads and loads
                                  it itself (about 18 ms) — holding it would
                                  only add a step of waiting

so a cold adapter delays its own request and nobody else's. ``LoraCacheMirror``
is how the engine knows where an adapter is without asking the workers: it
replays vLLM's two LRU caches (host memory: ``max_cpu_loras``; GPU slots:
``max_loras``) from the adapters of every step it issues. A mirror that is off
costs latency only — the worker's load is correct either way (a read that is
not finished is waited for; an adapter that was not preloaded is read by the
step).

**Invariants** (``tests/test_engine_router.py``):

- *The prefix-cache hashes are recomputed when an adapter is attached.* vLLM
  hashes a request's token blocks, adapter name included, when the request
  object is built — here, before its adapter is known. Left alone, a routed
  request would reuse KV blocks the BASE model computed for the same prompt:
  wrong output, no error.
- *A held request that is aborted is dropped here.* It is not in the scheduler
  yet, so vLLM's abort does not find it.
- *A request is never stuck.* No encoder, a failed encode, an empty pool: the
  base model, with the reason in the decision.
- *Requests enter the scheduler in arrival order.*

Every adapter is assumed to use the base model's tokenizer and chat template:
the prompt is tokenized before its adapter is chosen.
"""

import time

from deltaserve import dprint
from deltaserve.route_middleware import (
    DECISION_HEADER,
    ROUTE_HEADER,
    UNAVAILABLE,
    RouteDecision,
    decode_route,
    encode_decision,
)


class _Held:
    __slots__ = ("request", "args", "kwargs", "key", "ids")

    def __init__(self, request, args, kwargs, key=None, ids=None) -> None:
        self.request, self.args, self.kwargs = request, args, kwargs
        self.key, self.ids = key, ids


class LoraCacheMirror:
    """vLLM's adapter caches as the engine can know them: which adapters are
    in host memory (LRU, ``cpu_capacity``) and which hold a GPU slot (LRU,
    ``gpu_slots``). Pinned adapters (the finetuning one) never leave."""

    def __init__(self, cpu_capacity: int, gpu_slots: int, pinned=()) -> None:
        from collections import OrderedDict

        self.cpu_capacity = max(1, int(cpu_capacity))
        self.gpu_slots = max(1, int(gpu_slots))
        self.pinned = set(pinned)
        self.cpu, self.gpu = OrderedDict(), OrderedDict()
        for i in self.pinned:
            self.cpu[i] = self.gpu[i] = None
        self.reads = self.copies = 0

    def in_memory(self, lora_id) -> bool:
        return lora_id in self.cpu

    def in_slot(self, lora_id) -> bool:
        return lora_id in self.gpu

    def _evict(self, cache) -> object:
        for k in cache:
            if k not in self.pinned:
                del cache[k]
                return k
        return None

    def note_read(self, lora_id) -> None:
        """The adapter is (being) read into host memory."""
        if lora_id in self.cpu:
            self.cpu.move_to_end(lora_id)
            return
        if len(self.cpu) >= self.cpu_capacity:
            gone = self._evict(self.cpu)
            self.gpu.pop(gone, None)      # leaving memory frees its slot too
        self.cpu[lora_id] = None

    def load_cost(self, lora_ids) -> tuple[int, int]:
        """(copies into a slot, reads from disk) a step with these adapters
        would do — without changing anything."""
        ids = set(lora_ids)
        return (sum(1 for i in ids if i not in self.gpu),
                sum(1 for i in ids if i not in self.cpu))

    def note_step(self, lora_ids) -> tuple[int, int]:
        """A step with these adapters is issued. Returns ``load_cost`` of it."""
        ids = list(dict.fromkeys(lora_ids))
        cost = self.load_cost(ids)
        for i in ids:
            if i not in self.cpu:
                self.reads += 1
            self.note_read(i)
        for i in ids:
            if i in self.gpu:
                self.gpu.move_to_end(i)
                continue
            self.copies += 1
            while len(self.gpu) >= self.gpu_slots:
                # never one of this step's own adapters
                victim = next((k for k in self.gpu
                               if k not in self.pinned and k not in ids), None)
                if victim is None:
                    break
                del self.gpu[victim]
            self.gpu[i] = None
        return cost

    def forget(self, lora_id) -> None:
        """Removed from vLLM (``remove_lora``)."""
        self.cpu.pop(lora_id, None)
        self.gpu.pop(lora_id, None)

    def status(self) -> dict:
        return {"in_memory": list(self.cpu), "in_slots": list(self.gpu),
                "disk_reads": self.reads, "slot_copies": self.copies}


def take_batch(held: list, max_tokens: int) -> int:
    """How many of ``held`` (in order) one encode step takes: as many as fit
    ``max_tokens`` once padded to the longest of them, at least one."""
    longest = n = 0
    for h in held:
        longest = max(longest, len(h.ids))
        if n and (n + 1) * longest > max_tokens:
            break
        n += 1
    return n


class EngineRouter:
    """``encode(batch) -> [n, dim] array | None``: the encode step.
    ``decide(embeddings) -> [RouteDecision]``. ``lora_for(name) -> LoRARequest
    | None``: the adapter as the pool knows it. ``add(request, *args, **kw)``:
    vLLM's own ``EngineCore.add_request`` for a request whose adapter is set."""

    def __init__(self, encode, decide, lora_for, add,
                 max_tokens: int = 1024, on_routed=None, mirror=None,
                 busy=None) -> None:
        self._encode, self._decide = encode, decide
        self._lora_for, self._add = lora_for, add
        self.max_tokens = int(max_tokens)
        self._on_routed = on_routed
        self.held: list[_Held] = []
        self.encode_steps = self.routed = self.fallbacks = self.dropped = 0
        # Cold start. ``busy() -> bool``: other inference work exists, so a
        # step runs whether or not the cold request is in it.
        self.mirror = mirror
        self.busy = busy
        self.on_busy = None           # inference work is present (indexing)
        self.cold: list[_Held] = []
        self._preload: dict = {}          # lora id → LoRARequest, for the next step
        self.cold_holds = 0
        # Seconds of encode steps issued by the schedule() call in progress
        # (the scheduler's hook resets it): the gap they add between two
        # decode steps, for the TBT check.
        self.encode_s = 0.0
        self.encode_wall_s = 0.0      # cumulative, engine side (status only)
        # What happened, for GET /adapters: encode steps by what led to them
        # ("schedule": the start of a schedule() call; "arrival": a request
        # that landed during the FT-only grace wait or after a rollback), how
        # many requests they carried, and how often the token cap split a
        # batch over two steps.
        self.encode_by = {"schedule": 0, "arrival": 0}
        self.encoded_requests = self.split_batches = 0

    # ----------------------------------------------------------------- hold
    @staticmethod
    def wants(request) -> bool:
        headers = getattr(request, "trace_headers", None)
        return bool(headers) and ROUTE_HEADER in headers

    def hold(self, request, args=(), kwargs=None) -> None:
        """A routed request arrived: keep it until ``route_pending``."""
        headers = dict(request.trace_headers)
        value = headers.pop(ROUTE_HEADER)
        request.trace_headers = headers or None
        try:
            key, ids = decode_route(value)
        except (ValueError, KeyError, TypeError):
            # Not ours to understand: serve it as it is (the base model).
            self._add(request, *args, **(kwargs or {}))
            return
        self.held.append(_Held(request, args, kwargs or {}, key, ids))

    @property
    def pending(self) -> bool:
        return bool(self.held or self.cold)

    def abort(self, request_ids) -> None:
        """Requests vLLM is aborting: a held one simply goes."""
        if not (self.held or self.cold) or request_ids is None:
            return
        gone = set(request_ids)
        for name in ("held", "cold"):
            before = getattr(self, name)
            kept = [h for h in before if h.request.request_id not in gone]
            self.dropped += len(before) - len(kept)
            setattr(self, name, kept)

    # ----------------------------------------------------------- cold start
    def hold_cold(self, request, args=(), kwargs=None) -> bool:
        """A request whose adapter is known is about to enter the scheduler.
        True = it was held instead: its adapter is on disk and there is other
        work for this step, so the step runs without it and reads the adapter
        in the background."""
        lora = getattr(request, "lora_request", None)
        if lora is None or self.mirror is None:
            return False
        if self.mirror.in_memory(lora.lora_int_id):
            return False
        if self.busy is None or not self.busy():
            return False
        self.cold.append(_Held(request, args, kwargs or {}))
        self._preload[lora.lora_int_id] = lora
        self.mirror.note_read(lora.lora_int_id)
        self.cold_holds += 1
        return True

    def request_preload(self, lora) -> None:
        """Have the workers read this adapter in the background, with the next
        step issued (a new adapter being published, ``prewarm``). Thread-safe
        enough: one dict store, taken by the engine thread."""
        if self.mirror is None or self.mirror.in_memory(lora.lora_int_id):
            return
        self._preload[lora.lora_int_id] = lora
        self.mirror.note_read(lora.lora_int_id)

    def release_cold(self) -> int:
        """Start of the next ``schedule()``: the held requests go in (first —
        they arrived before anything this call routes)."""
        if not self.cold:
            return 0
        cold, self.cold = self.cold, []
        for h in cold:
            self._add(h.request, *h.args, **h.kwargs)
        return len(cold)

    def step_issued(self, lora_ids) -> tuple:
        """The step being issued uses these adapters. Returns ``(preload,
        copies, reads)``: the adapters its workers should read in the
        background (cold-start holds made while it was built), and what the
        step itself loads."""
        copies = reads = 0
        if self.mirror is not None:
            copies, reads = self.mirror.note_step(lora_ids)
        preload, self._preload = list(self._preload.values()), {}
        return preload, copies, reads

    # ---------------------------------------------------------------- route
    def route_pending(self, why: str = "schedule") -> int:
        """Encode, decide and release held requests (one encode step's worth).
        Returns how many entered the scheduler."""
        if not self.held:
            return 0
        n = take_batch(self.held, self.max_tokens)
        batch, self.held = self.held[:n], self.held[n:]
        self.encode_by[why] = self.encode_by.get(why, 0) + 1
        self.encoded_requests += n
        if self.held:
            self.split_batches += 1
        embs = spent = None
        t0 = time.perf_counter()
        try:
            embs = self._encode([h.ids for h in batch])
            if isinstance(embs, tuple):
                # The worker's own time for the encode: the engine's wall
                # time also holds the wait for the step still in flight,
                # which the next model step pays with or without an encode.
                embs, spent = embs
            self.encode_steps += 1
        except Exception as e:  # noqa: BLE001 — a request is never stuck
            dprint(f"[adapter-routing] encode step failed ({type(e).__name__}: "
                   f"{e}); {len(batch)} request(s) served by the base model")
        wall = time.perf_counter() - t0
        self.encode_wall_s += wall
        self.encode_s += wall if spent is None else min(float(spent), wall)
        if embs is None or len(embs) != len(batch):
            decisions = [RouteDecision(None, None, UNAVAILABLE) for _ in batch]
        else:
            try:
                decisions = self._decide(list(embs))
            except Exception as e:  # noqa: BLE001
                dprint(f"[adapter-routing] decision failed ({type(e).__name__}: "
                       f"{e}); served by the base model")
                decisions = [RouteDecision(None, None, UNAVAILABLE) for _ in batch]
        for h, decision in zip(batch, decisions):
            self._release(h, decision)
        return len(batch)

    def _release(self, h: _Held, decision: RouteDecision) -> None:
        request = h.request
        if decision.adapter:
            lora = None
            try:
                lora = self._lora_for(decision.adapter)
            except Exception:  # noqa: BLE001
                pass
            if lora is None:
                # Chosen from the index but gone from the pool since.
                decision = RouteDecision(None, decision.score, UNAVAILABLE,
                                         decision.confidence, decision.candidates)
            else:
                attach_adapter(request, lora)
        if decision.adapter:
            self.routed += 1
        else:
            self.fallbacks += 1
        headers = dict(request.trace_headers or {})
        headers[DECISION_HEADER] = encode_decision(h.key, decision)
        request.trace_headers = headers
        if not self.hold_cold(request, h.args, h.kwargs):
            self._add(request, *h.args, **h.kwargs)
        if self._on_routed is not None:
            try:
                self._on_routed(request, decision)
            except Exception:  # noqa: BLE001 — bookkeeping must not fail serving
                pass

    def status(self) -> dict:
        out = {"held": len(self.held), "encode_steps": self.encode_steps,
               "routed": self.routed, "fallbacks": self.fallbacks,
               "dropped": self.dropped, "cold_held": len(self.cold),
               "cold_holds": self.cold_holds, "encode_by": dict(self.encode_by),
               "encoded_requests": self.encoded_requests,
               "split_batches": self.split_batches,
               "encode_wall_ms": round(self.encode_wall_s * 1e3, 1)}
        if self.mirror is not None:
            out["adapters"] = self.mirror.status()
        return out


def attach_adapter(request, lora) -> None:
    """Give a request its adapter AFTER it was built, and recompute what vLLM
    derived from "no adapter": the prefix-cache block hashes (the adapter's
    name is part of every block's key)."""
    request.lora_request = lora
    hashes = getattr(request, "block_hashes", None)
    if hashes is not None and hasattr(request, "update_block_hashes"):
        request.block_hashes = []
        request.update_block_hashes()
