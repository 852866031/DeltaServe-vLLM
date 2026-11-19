#!/usr/bin/env python
"""Adapter routing inside the engine: the scheduler chooses a routed request's
adapter (deltaserve.adapter_routing.engine_router + the plugin's hooks + the
API-server middleware).

CPU-only, no model:

    python tests/test_engine_router.py

Checks
  * the route payload and the decision survive their trip through
    ``trace_headers``;
  * a routed request is held, encoded by ONE encode step with the others,
    given its adapter and released to vLLM's own add_request in arrival order;
  * an encode step carries at most ``route_encode_max_tokens`` padded tokens,
    the rest is encoded by the next step; one request is always taken;
  * a request is never stuck: failed encode / failed decision / an adapter
    that left the pool → the base model, with the reason;
  * a held request that is aborted is dropped;
  * INVARIANT: attaching the adapter recomputes vLLM's prefix-cache block
    hashes (on a real vLLM Request: the hashes of a routed request equal the
    hashes of the same prompt sent with that adapter by name, and differ from
    the base model's);
  * the plugin hooks on a scheduler object: held requests count as requests,
    ``schedule()`` routes them first, aborts reach the router; ``schedule()``
    calls the router's ``on_busy`` (indexing leaves the GPU) while a routed
    request is held or an inference request — not a finetuning sample — is in
    the scheduler;
  * the middleware: the request goes on as a base-model request with the
    encoder tokens for the engine; the response start waits for the engine's
    decision and carries the x-deltaserve-* headers; an error response does
    not wait; routing off / no text are answered without the engine;
  * the worker mixin: padding, which rank holds the encoder;
  * the engine's mirror of vLLM's adapter caches (host memory / GPU slots);
  * COLD START: a request (named or routed) whose adapter is on disk is held
    for one step while other work exists, the step carries a preload hint, the
    worker reads the adapter on a background thread while it runs the step,
    and vLLM's load takes that read (waiting for one in flight) — never reads
    twice; with nothing else to run the request is not held.
"""

import asyncio
import json
import sys
from types import SimpleNamespace

import numpy as np

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


def fake_request(rid, ids, key=None, extra=None):
    from deltaserve.route_middleware import ROUTE_HEADER, encode_route

    headers = dict(extra or {})
    if ids is not None:
        headers[ROUTE_HEADER] = encode_route(key or f"k-{rid}", ids)
    return SimpleNamespace(request_id=rid, trace_headers=headers or None,
                           lora_request=None)


def make_router(decisions=None, encode=None, lora_for=None, max_tokens=1024):
    from deltaserve.adapter_routing.engine_router import EngineRouter
    from deltaserve.route_middleware import RouteDecision

    log = SimpleNamespace(encoded=[], added=[], routed=[])

    def default_encode(batch):
        log.encoded.append([list(b) for b in batch])
        return np.zeros((len(batch), 4), np.float32)

    def default_decide(embs):
        return [RouteDecision("ad-a", 0.9, "", 1.5, ("ad-a",)) for _ in embs]

    def default_lora(name):
        return SimpleNamespace(lora_name=name, lora_int_id=100001, lora_path="/p")

    def add(request, *args, **kwargs):
        log.added.append((request.request_id, args, kwargs))

    r = EngineRouter(encode or default_encode, decisions or default_decide,
                     lora_for or default_lora, add, max_tokens=max_tokens,
                     on_routed=lambda req, d: log.routed.append(
                         (req.request_id, d.adapter)))
    return r, log


def test_wire_format():
    print("Route payload and decision in trace_headers")
    from deltaserve.route_middleware import (
        RouteDecision,
        decode_decision,
        decode_route,
        encode_decision,
        encode_route,
    )

    key, ids = decode_route(encode_route("abc", [101, 7, 102]))
    check("route payload round trip", (key, ids) == ("abc", [101, 7, 102]))
    d = RouteDecision("ad-a", 0.8312, "", 1.25, ("ad-a", "ad-b"))
    k2, d2 = decode_decision(encode_decision("abc", d))
    check("decision round trip (adapter, score, confidence, candidates)",
          k2 == "abc" and d2 == d and d2.outcome == "routed")
    k3, d3 = decode_decision(encode_decision(
        "x", RouteDecision(None, None, "no-adapters")))
    check("fallback decision round trip",
          d3.adapter is None and d3.reason == "no-adapters"
          and d3.outcome == "fallback")


def test_hold_and_route():
    print("Hold → encode step → adapter → scheduler")
    from deltaserve.adapter_routing.engine_router import EngineRouter
    from deltaserve.route_middleware import DECISION_HEADER, decode_decision

    r, log = make_router()
    plain = fake_request("plain", None)
    check("a request without the route header is not taken",
          not EngineRouter.wants(plain))
    a = fake_request("a", [1, 2, 3], extra={"traceparent": "00-x"})
    b = fake_request("b", [4, 5])
    check("a routed request is taken", EngineRouter.wants(a))
    r.hold(a, (7,), {})
    r.hold(b)
    check("held: not in the scheduler, the router reports pending work",
          r.pending and not log.added)
    check("the route header is stripped, other trace headers are kept",
          a.trace_headers == {"traceparent": "00-x"} and b.trace_headers is None)
    n = r.route_pending()
    check("one encode step for both", n == 2 and len(log.encoded) == 1
          and log.encoded[0] == [[1, 2, 3], [4, 5]])
    check("released to vLLM's add_request in arrival order, arguments kept",
          log.added == [("a", (7,), {}), ("b", (), {})])
    check("each carries its adapter", a.lora_request.lora_name == "ad-a"
          and b.lora_request.lora_int_id == 100001)
    key, d = decode_decision(a.trace_headers[DECISION_HEADER])
    check("and its decision for the API server, under its own key",
          key == "k-a" and d.adapter == "ad-a" and d.score == 0.9
          and a.trace_headers["traceparent"] == "00-x")
    check("the pool is told which adapter was used",
          log.routed == [("a", "ad-a"), ("b", "ad-a")])
    check("nothing left; a second call does nothing",
          not r.pending and r.route_pending() == 0 and len(log.encoded) == 1)
    st = r.status()
    check("status counts", st["routed"] == 2 and st["encode_steps"] == 1
          and st["held"] == 0)


def test_encode_time():
    print("Encode time handed to the TBT check")
    import time

    import numpy as np

    def slow(spent):
        def encode(batch):
            time.sleep(0.03)          # the wait for the step in flight
            embs = np.zeros((len(batch), 4), dtype=np.float32)
            return embs if spent is None else (embs, spent)
        return encode

    r, log = make_router(encode=slow(0.004))
    r.hold(fake_request("a", [1, 2]))
    n = r.route_pending()
    check("the worker's own encode time is used, not the engine's wall time "
          f"({r.encode_s * 1e3:.1f} ms of {r.encode_wall_s * 1e3:.0f} ms)",
          n == 1 and abs(r.encode_s - 0.004) < 1e-9
          and r.encode_wall_s >= 0.03
          and r.status()["encode_wall_ms"] >= 30)
    r, log = make_router(encode=slow(None))
    r.hold(fake_request("b", [1, 2]))
    r.route_pending()
    check("a worker that reports no time: the wall time is used",
          r.encode_s >= 0.03 and r.encode_s == r.encode_wall_s)
    r, log = make_router(encode=slow(9.0))
    r.hold(fake_request("c", [1, 2]))
    r.route_pending()
    check("never more than the wall time", r.encode_s == r.encode_wall_s < 1)


def test_token_cap():
    print("An encode step's token cap")
    from deltaserve.adapter_routing.engine_router import _Held, take_batch

    def held(*lens):
        return [_Held(None, (), {}, str(i), [0] * n) for i, n in enumerate(lens)]

    check("everything fits", take_batch(held(100, 100, 100), 1024) == 3)
    check("padded size counts: 3 x 400 > 1024 → two", take_batch(held(400, 400, 400), 1024) == 2)
    check("a long one makes the short ones before it expensive",
          take_batch(held(10, 10, 512), 1024) == 2)
    check("one request is always taken, whatever its length",
          take_batch(held(4000, 10), 1024) == 1)
    r, log = make_router(max_tokens=1024)
    for i, n in enumerate((500, 500, 500, 20)):
        r.hold(fake_request(f"r{i}", [1] * n))
    first = r.route_pending()
    check("first step takes two of four", first == 2 and r.pending
          and [x[0] for x in log.added] == ["r0", "r1"])
    second = r.route_pending()
    check("the next step takes the rest, still in order",
          second == 2 and not r.pending
          and [x[0] for x in log.added] == ["r0", "r1", "r2", "r3"]
          and len(log.encoded) == 2)


def test_never_stuck():
    print("A request is never stuck")
    from deltaserve.route_middleware import (
        DECISION_HEADER,
        NO_ADAPTERS,
        UNAVAILABLE,
        RouteDecision,
        decode_decision,
    )

    def boom(batch):
        raise RuntimeError("no worker holds the route encoder")

    r, log = make_router(encode=boom)
    q = fake_request("q", [1, 2])
    r.hold(q)
    r.route_pending()
    _, d = decode_decision(q.trace_headers[DECISION_HEADER])
    check("failed encode step → base model, reason router-unavailable",
          log.added == [("q", (), {})] and q.lora_request is None
          and d.adapter is None and d.reason == UNAVAILABLE)

    r, log = make_router(encode=lambda batch: None)
    q = fake_request("q", [1])
    r.hold(q)
    r.route_pending()
    check("an encode step that returns nothing → base model",
          log.added and q.lora_request is None)

    def bad_decide(embs):
        raise ValueError("index is being rewritten")

    r, log = make_router(decisions=bad_decide)
    q = fake_request("q", [1])
    r.hold(q)
    r.route_pending()
    check("failed decision → base model", log.added and q.lora_request is None)

    r, log = make_router(lora_for=lambda name: None)
    q = fake_request("q", [1])
    r.hold(q)
    r.route_pending()
    _, d = decode_decision(q.trace_headers[DECISION_HEADER])
    check("an adapter chosen from the index but gone from the pool → base "
          "model, and the response says fallback",
          q.lora_request is None and d.adapter is None
          and d.reason == UNAVAILABLE and r.status()["fallbacks"] == 1)

    r, log = make_router(decisions=lambda embs: [
        RouteDecision(None, None, NO_ADAPTERS) for _ in embs])
    q = fake_request("q", [1])
    r.hold(q)
    r.route_pending()
    _, d = decode_decision(q.trace_headers[DECISION_HEADER])
    check("an empty pool → base model, reason no-adapters; the pool is not "
          "told of a hit", d.reason == NO_ADAPTERS and q.lora_request is None)

    r, log = make_router()
    weird = SimpleNamespace(request_id="w", lora_request=None,
                            trace_headers={"x-deltaserve-route": "not json"})
    r.hold(weird)
    check("an unreadable route payload → served at once as it is",
          log.added == [("w", (), {})] and not r.pending)


def test_abort():
    print("Aborting a held request")
    r, log = make_router()
    for rid in ("a", "b", "c"):
        r.hold(fake_request(rid, [1, 2]))
    r.abort(["b", "not-held"])
    check("the aborted one is dropped, the others stay in order",
          [h.request.request_id for h in r.held] == ["a", "c"]
          and r.status()["dropped"] == 1)
    r.route_pending()
    check("it never reaches the scheduler", [x[0] for x in log.added] == ["a", "c"])
    r.abort(["a"])
    check("aborting a released request is vLLM's business (no-op here)",
          not r.pending)


def test_mirror():
    print("The engine's view of vLLM's adapter caches")
    from deltaserve.adapter_routing.engine_router import LoraCacheMirror

    m = LoraCacheMirror(cpu_capacity=4, gpu_slots=3, pinned=(1000,))
    check("the finetuning adapter is in memory and in a slot from the start",
          m.in_memory(1000) and m.in_slot(1000))
    check("a first step with two new adapters: two disk reads, two slot copies",
          m.note_step([1, 2]) == (2, 2) and m.in_slot(1) and m.in_slot(2))
    check("the same adapters again cost nothing", m.note_step([2, 1]) == (0, 0))
    check("load_cost predicts without changing anything",
          m.load_cost([1, 3]) == (1, 1) and not m.in_memory(3))
    m.note_step([3])
    check("slots full: the least recently used goes, never the pinned one",
          m.in_slot(3) and m.in_slot(1000) and not m.in_slot(2) and m.in_slot(1)
          and m.in_memory(2))
    check("back from host memory: a slot copy, no disk read",
          m.note_step([2]) == (1, 0))
    m.note_step([4])
    check("host memory full: the oldest leaves it (and its slot)",
          not m.in_memory(1) and not m.in_slot(1) and m.in_memory(1000))
    m.note_read(7)
    check("a background read puts an adapter in memory, not in a slot",
          m.in_memory(7) and not m.in_slot(7) and m.load_cost([7]) == (1, 0))
    m.forget(7)
    check("remove_lora is mirrored", not m.in_memory(7))
    m2 = LoraCacheMirror(cpu_capacity=8, gpu_slots=2)
    m2.note_step([1, 2])
    m2.note_step([3, 1])
    check("a step never evicts one of its own adapters",
          m2.in_slot(3) and m2.in_slot(1) and not m2.in_slot(2))


def test_cold_start():
    print("Cold start: the adapter is on disk")
    from deltaserve.adapter_routing.engine_router import LoraCacheMirror
    from deltaserve.route_middleware import RouteDecision

    def lora(i, name="ad"):
        return SimpleNamespace(lora_name=f"{name}-{i}", lora_int_id=i,
                               lora_path=f"/p/{i}")

    busy = {"v": True}
    r, log = make_router(lora_for=lambda name: lora(5))
    r.mirror = LoraCacheMirror(8, 3)
    r.busy = lambda: busy["v"]

    named = SimpleNamespace(request_id="n", trace_headers=None, lora_request=lora(9))
    check("a named request for an adapter on disk is held while other work "
          "exists", r.hold_cold(named, (1,), {}) and r.pending and not log.added)
    preload, copies, reads = r.step_issued([])
    check("the step being issued carries the preload hint for it",
          [p.lora_int_id for p in preload] == [9] and (copies, reads) == (0, 0))
    check("the hint is given once", r.step_issued([])[0] == [])
    check("the next schedule() releases it, arguments kept",
          r.release_cold() == 1 and log.added == [("n", (1,), {})] and not r.pending)
    check("that step copies the adapter into a slot and reads nothing",
          r.step_issued([9])[1:] == (1, 0))
    again = SimpleNamespace(request_id="n2", trace_headers=None, lora_request=lora(9))
    check("the adapter is in memory now: the next request for it is not held",
          not r.hold_cold(again))

    r.hold(fake_request("q", [1, 2]))
    r.route_pending()
    check("a ROUTED request whose adapter is cold is held the same way "
          "(encoded now, scheduled next step)",
          [h.request.request_id for h in r.cold] == ["q"]
          and r.cold[0].request.lora_request.lora_int_id == 5
          and "q" not in [x[0] for x in log.added])
    check("with its decision already attached for the API server",
          "x-deltaserve-routed" in r.cold[0].request.trace_headers)
    r.release_cold()
    check("and released by the next schedule()", log.added[-1][0] == "q")

    busy["v"] = False
    lone = SimpleNamespace(request_id="l", trace_headers=None, lora_request=lora(11))
    check("nothing else to run: not held — the step loads the adapter itself",
          not r.hold_cold(lone))
    check("which the step's load cost shows: one copy, one disk read",
          r.step_issued([11])[1:] == (1, 1))
    busy["v"] = True
    base = SimpleNamespace(request_id="b", trace_headers=None, lora_request=None)
    check("a base-model request is never held", not r.hold_cold(base))

    c1 = SimpleNamespace(request_id="c1", trace_headers=None, lora_request=lora(21))
    c2 = SimpleNamespace(request_id="c2", trace_headers=None, lora_request=lora(22))
    r.hold_cold(c1)
    r.hold_cold(c2)
    r.abort(["c1"])
    check("a cold-held request that is aborted is dropped",
          [h.request.request_id for h in r.cold] == ["c2"])
    r2, log2 = make_router()
    check("without a mirror nothing is ever held (the previous behaviour)",
          not r2.hold_cold(SimpleNamespace(request_id="x", trace_headers=None,
                                           lora_request=lora(1))))
    del RouteDecision


def test_worker_preload():
    print("Worker: background read of a cold adapter")
    import threading
    import time

    from deltaserve.route_worker import RouteEncodeMixin

    reads, gate = [], threading.Event()

    def read(req):
        reads.append((req.lora_int_id, threading.current_thread().name))
        gate.wait(2.0)
        return f"weights-{req.lora_int_id}"

    mgr = SimpleNamespace(_load_adapter=read, list_adapters=lambda: {3})
    seen = {}

    class Base:
        def execute_model(self, scheduler_output):
            seen["ran_while_reading"] = not gate.is_set()
            return "out"

    class W(RouteEncodeMixin, Base):
        model_runner = SimpleNamespace(lora_manager=mgr)

    def req(i):
        return SimpleNamespace(lora_int_id=i, lora_path=f"/p/{i}")

    w = W()
    out = w.execute_model(SimpleNamespace(_ds_preload=[req(1), req(3)]))
    time.sleep(0.05)
    check("the step runs while the adapter is being read on another thread",
          out == "out" and seen["ran_while_reading"]
          and reads == [(1, "deltaserve-adapter-preload")])
    check("an adapter already in vLLM's memory is not read again",
          3 not in [r[0] for r in reads])
    gate.set()
    got = mgr._load_adapter(req(1))
    check("vLLM's load takes the finished read instead of reading again",
          got == "weights-1" and len(reads) == 1)
    got = mgr._load_adapter(req(2))
    check("an adapter that was not preloaded is read by the step itself",
          got == "weights-2" and reads[-1][0] == 2
          and reads[-1][1] != "deltaserve-adapter-preload")
    gate.clear()
    w.execute_model(SimpleNamespace(_ds_preload=[req(8)]))
    t0 = time.monotonic()
    threading.Timer(0.2, gate.set).start()
    got = mgr._load_adapter(req(8))
    check("a read still in flight is waited for, not repeated",
          got == "weights-8" and time.monotonic() - t0 >= 0.15
          and [r[0] for r in reads].count(8) == 1)
    check("a step without a hint is untouched",
          w.execute_model(SimpleNamespace()) == "out")


def test_block_hashes():
    print("INVARIANT: prefix-cache hashes follow the attached adapter")
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256_cbor
    from vllm.v1.core import kv_cache_utils as k
    from vllm.v1.request import Request

    from deltaserve.adapter_routing.engine_router import attach_adapter

    k.init_none_hash(sha256_cbor)
    hasher = k.get_request_block_hasher(16, sha256_cbor)
    tokens = list(range(1000, 1100))

    def build(lora):
        return Request(request_id="r", prompt_token_ids=list(tokens),
                       sampling_params=SamplingParams(max_tokens=4),
                       pooling_params=None, client_index=0, arrival_time=0.0,
                       lora_request=lora, block_hasher=hasher)

    lora = LoRARequest(lora_name="ad-a", lora_int_id=100001, lora_path="/p")
    base, named = build(None), build(lora)
    check("vLLM keys a prompt's blocks by adapter: base != adapter",
          len(base.block_hashes) == 6 and base.block_hashes != named.block_hashes)
    routed = build(None)
    attach_adapter(routed, lora)
    check("after the route the request has the adapter's hashes — it can "
          "never reuse the base model's KV blocks",
          routed.block_hashes == named.block_hashes
          and routed.block_hashes != base.block_hashes
          and routed.lora_request is lora)
    stale = build(None)
    stale.lora_request = lora
    check("(setting the adapter alone would have left the base model's hashes)",
          stale.block_hashes == base.block_hashes)


def test_scheduler_hooks():
    print("Held requests are part of the scheduler's work")
    from deltaserve.plugin import _route_in_scheduler

    r, log = make_router()
    calls = []
    sched = SimpleNamespace(
        has_requests=lambda: False, has_unfinished_requests=lambda: False,
        requests={},
        schedule=lambda: calls.append("schedule") or SimpleNamespace(
            num_scheduled_tokens={}))
    _route_in_scheduler(sched, r)
    check("nothing held: the scheduler's own answers", not sched.has_requests()
          and not sched.has_unfinished_requests())
    r.hold(fake_request("a", [1, 2]))
    check("a held request counts as a request (the engine loop must not "
          "sleep on it)", sched.has_requests() and sched.has_unfinished_requests())
    out = sched.schedule()
    check("schedule() runs the encode step first, then schedules",
          out is not None and calls == ["schedule"]
          and log.added == [("a", (), {})] and len(log.encoded) == 1)
    check("and the held request is gone", not sched.has_requests())

    # [indexing] the scheduler's busy signal: schedule() says "inference is
    # present" (IndexingPlacement.note_inference) while a routed request is
    # held or the scheduler has an inference request.
    r, log = make_router()
    sched = SimpleNamespace(
        has_requests=lambda: False, has_unfinished_requests=lambda: False,
        requests={},
        schedule=lambda: SimpleNamespace(num_scheduled_tokens={}))
    _route_in_scheduler(sched, r)
    check("no on_busy callback by default; schedule() works without one",
          r.on_busy is None and sched.schedule() is not None)
    busy = []
    r.on_busy = lambda: busy.append(int(r.pending))
    sched.schedule()
    check("idle engine: schedule() does not report inference", busy == [])
    r.hold(fake_request("b", [1, 2]))
    sched.schedule()
    check("a held routed request → on_busy, before the encode step",
          busy == [1] and len(log.encoded) == 1)
    sched.requests["__ft__0"] = object()
    sched.schedule()
    check("finetuning samples alone are not inference", busy == [1])
    sched.requests["chat-1"] = object()
    sched.schedule()
    sched.schedule()
    check("an inference request in the scheduler → on_busy at every schedule()",
          busy == [1, 0, 0])
    del sched.requests["chat-1"]
    sched.schedule()
    check("... and no longer once it is gone", busy == [1, 0, 0])


async def _asgi(mw, body, path="/v1/chat/completions", app=None, registry=None):
    seen, out = {}, {"headers": {}, "chunks": [], "status": None}

    async def default_app(scope, receive, send):
        chunks = []
        while True:
            m = await receive()
            chunks.append(m.get("body", b""))
            if not m.get("more_body"):
                break
        seen["body"] = b"".join(chunks)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    mw.app = app or default_app
    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.sleep(3600)

    async def send(m):
        if m["type"] == "http.response.start":
            out["status"] = m["status"]
            out["headers"] = {k.decode(): v.decode() for k, v in m["headers"]}
        else:
            out["chunks"].append(m.get("body", b""))

    scope = {"type": "http", "method": "POST", "path": path,
             "headers": [(b"content-length", str(len(body)).encode())]}
    await mw(scope, receive, send)
    return seen, out


def test_middleware():
    print("API-server middleware")
    from deltaserve.route_middleware import (
        DECISION_HEADER,
        ROUTE_PAYLOAD,
        AdapterRouteMiddleware,
        DecisionRegistry,
        RouteDecision,
        decode_route,
        encode_decision,
    )

    reg = DecisionRegistry()
    mw = AdapterRouteMiddleware(None, lambda text: [101, len(text), 102], "auto",
                                "org/base", registry=reg)
    body = json.dumps({"model": "auto", "temperature": 0,
                       "messages": [{"role": "user", "content": "translate"}]}).encode()

    async def run_stream():
        captured = {}

        async def app(scope, receive, send):
            m = await receive()
            captured["body"] = json.loads(m["body"])
            captured["payload"] = ROUTE_PAYLOAD.get()
            key, ids = decode_route(captured["payload"])
            captured["key"], captured["ids"] = key, ids

            async def engine_answers():
                await asyncio.sleep(0.05)
                captured["start_sent_before_decision"] = bool(out_ref.get("status"))
                reg.note_headers({DECISION_HEADER: encode_decision(
                    key, RouteDecision("ad-a", 0.91, "", 2.0, ("ad-a", "ad-b")))})

            asyncio.ensure_future(engine_answers())
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"tok"})

        out_ref = {}
        mw.app = app
        sent = False

        async def receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.sleep(3600)

        async def send(m):
            if m["type"] == "http.response.start":
                out_ref["status"] = m["status"]
                out_ref["headers"] = {k.decode(): v.decode() for k, v in m["headers"]}

        await mw({"type": "http", "method": "POST", "path": "/v1/chat/completions",
                  "headers": []}, receive, send)
        return captured, out_ref

    captured, out = asyncio.run(run_stream())
    check("the request goes on as a base-model request, the rest unchanged",
          captured["body"]["model"] == "org/base"
          and captured["body"]["temperature"] == 0)
    check("with the route encoder's tokens for the engine",
          captured["ids"] == [101, len("translate"), 102] and captured["key"])
    check("a streamed response's start waits for the engine's decision",
          captured["start_sent_before_decision"] is False and out["status"] == 200)
    h = out["headers"]
    check("the response says: routed, adapter, score, confidence, candidates",
          h.get("x-deltaserve-routing") == "routed"
          and h.get("x-deltaserve-adapter") == "ad-a"
          and h.get("x-deltaserve-routing-score") == "0.9100"
          and h.get("x-deltaserve-routing-confidence") == "2.000"
          and h.get("x-deltaserve-routing-candidates") == "ad-a,ad-b")
    check("nothing is left waiting", not reg.waiting)

    async def err_app(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 400, "headers": []})
        await send({"type": "http.response.body", "body": b"bad"})

    _, out = asyncio.run(_asgi(mw, body, app=err_app))
    check("an error response that never reached the router is sent at once, "
          "without routing headers",
          out["status"] == 400 and not any(k.startswith("x-deltaserve")
                                           for k in out["headers"])
          and not reg.waiting)

    # How vLLM really streams: the request is submitted to the engine by the
    # body generator, i.e. only AFTER the app's send(start) has returned. A
    # middleware that made the start wait for the decision would wait forever
    # (found on the production trace: every streamed routed request took the
    # full timeout).
    async def lazy_stream():
        events, got = [], {}

        async def app(scope, receive, send):
            await receive()
            key, _ = decode_route(ROUTE_PAYLOAD.get())
            await send({"type": "http.response.start", "status": 200, "headers": []})
            events.append("start returned")
            # only now is the request "submitted": the engine decides
            reg.note_headers({DECISION_HEADER: encode_decision(
                key, RouteDecision("ad-b", 0.7, "", None, ()))})
            await asyncio.sleep(0)
            await send({"type": "http.response.body", "body": b"tok1",
                        "more_body": True})
            await send({"type": "http.response.body", "body": b"tok2"})

        mw.app = app
        sent = False

        async def receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.sleep(3600)

        async def send(m):
            events.append(m["type"] if m["type"].endswith("start")
                          else m.get("body"))
            if m["type"] == "http.response.start":
                got.update({k.decode(): v.decode() for k, v in m["headers"]})

        await asyncio.wait_for(mw({"type": "http", "method": "POST",
                                   "path": "/v1/chat/completions", "headers": []},
                                  receive, send), 3.0)
        return events, got

    try:
        events, got = asyncio.run(lazy_stream())
    except asyncio.TimeoutError:
        events, got = ["DEADLOCK"], {}
    check("a stream whose request is submitted only after send(start) returns "
          "does not deadlock: the start is kept back, not waited on",
          events[0] == "start returned")
    check("the client gets the start with the first token, headers included, "
          "then the tokens in order",
          events[1:] == ["http.response.start", b"tok1", b"tok2"]
          and got.get("x-deltaserve-adapter") == "ad-b" and not reg.waiting)

    async def no_body_app(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})

    import deltaserve.route_middleware as rm

    old, rm.DECISION_WAIT_S = rm.DECISION_WAIT_S, 0.05
    try:
        _, out = asyncio.run(_asgi(mw, body, app=no_body_app))
    finally:
        rm.DECISION_WAIT_S = old
    check("a response that ends without a body message still gets its start; "
          "no decision ever came → the headers say router-timeout",
          out["status"] == 200
          and out["headers"].get("x-deltaserve-routing-reason") == "router-timeout")

    named = json.dumps({"model": "ad-a", "messages": []}).encode()
    seen, out = asyncio.run(_asgi(mw, named))
    check("a request naming an adapter passes byte-for-byte, no headers",
          seen["body"] == named and not out["headers"])

    no_text = json.dumps({"model": "auto", "prompt": [1, 2, 3]}).encode()
    seen, out = asyncio.run(_asgi(mw, no_text))
    check("no text to route on → base model, answered without the engine's "
          "router", json.loads(seen["body"])["model"] == "org/base"
          and out["headers"].get("x-deltaserve-routing-reason") == "no-text")

    def bad_tok(text):
        raise RuntimeError("tokenizer missing")

    mw2 = AdapterRouteMiddleware(None, bad_tok, "auto", "org/base", registry=reg)
    seen, out = asyncio.run(_asgi(mw2, body))
    check("a failing tokenizer never fails the request",
          out["status"] == 200
          and out["headers"].get("x-deltaserve-routing-reason") == "router-unavailable")

    off = AdapterRouteMiddleware(None, None, "auto", "org/base", registry=reg)
    seen, out = asyncio.run(_asgi(off, body))
    check("adapter routing off → base model + 'adapter-routing-disabled'",
          json.loads(seen["body"])["model"] == "org/base"
          and out["headers"].get("x-deltaserve-routing-reason")
          == "adapter-routing-disabled")

    reg.note_headers({DECISION_HEADER: "garbage"})
    reg.note_headers(None)
    check("unreadable or missing decision headers are ignored", True)


def test_worker_mixin():
    print("Worker side")
    from deltaserve.route_worker import encoder_rank, pad_batch

    ids, mask = pad_batch([[5, 6, 7], [8]], pad_id=0)
    check("padding to the longest, with the attention mask",
          ids.tolist() == [[5, 6, 7], [8, 0, 0]]
          and mask.tolist() == [[1, 1, 1], [1, 0, 0]])
    rc = SimpleNamespace(router_gpu=None)
    check("the last TP rank holds the encoder by default",
          encoder_rank(rc, 1) == 0 and encoder_rank(rc, 2) == 1
          and encoder_rank(rc, 4) == 3)
    check("router_gpu picks the rank when it is one of the model's cards",
          encoder_rank(SimpleNamespace(router_gpu=0), 2) == 0
          and encoder_rank(SimpleNamespace(router_gpu=5), 2) == 1)


def main():
    test_wire_format()
    test_hold_and_route()
    test_encode_time()
    test_token_cap()
    test_never_stuck()
    test_abort()
    test_mirror()
    test_cold_start()
    test_worker_preload()
    test_block_hashes()
    test_scheduler_hooks()
    test_middleware()
    test_worker_mixin()
    print()
    if _failed:
        print(f"{_RED}{_failed} failed{_RESET}, {_passed} passed")
        sys.exit(1)
    print(f"{_GREEN}all {_passed} checks passed{_RESET}")


if __name__ == "__main__":
    main()
