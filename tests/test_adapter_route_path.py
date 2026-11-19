#!/usr/bin/env python
"""The request path of adapter routing: the decision (``router.Router``), the
API-server middleware (``deltaserve.route_middleware``) and the loop it closes
with the engine's router (``adapter_routing.engine_router``).

CPU-only. The encoder is a deterministic fake except in the last, opt-in test:

    python tests/test_adapter_route_path.py
    DSERVE_TEST_REAL_ENCODER=1 python tests/test_adapter_route_path.py   # + real e5 on CPU

Checks
  * which text a request is routed on (chat: last user message; completions:
    the prompt; token-id prompts: none);
  * the router: top-1, fallback below the threshold / on an empty index / on no
    text, and it follows the index file as it changes;
  * the middleware: only ``model == route name`` is touched — its text is
    tokenized for the route encoder, ``model`` becomes the BASE model, the
    tokens go to the engine (``ROUTE_PAYLOAD`` → ``trace_headers``, by the
    plugin's wrapper on ``process_inputs``, which also stamps the arrival
    time) and the response start is held until the engine's decision is back;
    the x-deltaserve-* headers then say which adapter served it. Everything
    else passes through byte-for-byte; streaming responses and chunked request
    bodies work; a failing tokenizer or hook never fails the request; an
    error response is not held; an engine that never answers is waited for a
    bounded time only;
  * routing off (``tokenize=None``): base model + 'adapter-routing-disabled';
  * the closed loop: middleware → route payload → ``EngineRouter`` (fake encode
    step + ``Router.route_embeddings`` on a real index file) → decision header
    → ``DecisionRegistry`` → response headers.
"""

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_passed = 0
_failed = 0
_TTY = sys.stdout.isatty()
_GREEN = "\033[92m" if _TTY else ""
_RED = "\033[91m" if _TTY else ""
_RESET = "\033[0m" if _TTY else ""
DIM = 8
E = np.eye(DIM, dtype=np.float32)
TOPICS = {"translate": 0, "sql": 1, "poem": 2}


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


class FakeEncoder:
    """Embeds a text on the axis of the first topic word it contains."""

    def __init__(self, delay=0.0):
        self.delay, self.calls = delay, 0

    def encode(self, texts):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        out = []
        for t in texts:
            v = np.full(DIM, 0.01, dtype=np.float32)
            for word, axis in TOPICS.items():
                if word in t.lower():
                    v[axis] = 1.0
                    break
            else:
                v[DIM - 1] = 1.0            # unrelated text
            out.append(v)
        return np.array(out)


def write_index(path, names):
    from deltaserve.adapter_routing.index import CentroidIndex

    idx = CentroidIndex("fake", "", 512)
    for n in names:
        idx.add(f"dserve-routing-{n}", E[TOPICS[n]])
    idx.save(path)


def test_extract_text():
    green("which text is routed on:")
    from deltaserve.adapter_routing.middleware import extract_text

    check("chat: the last user message",
          extract_text({"messages": [
              {"role": "system", "content": "be brief"},
              {"role": "user", "content": "first"},
              {"role": "assistant", "content": "ok"},
              {"role": "user", "content": "second question"}]}) == "second question")
    check("chat: text parts of a multi-part message",
          extract_text({"messages": [{"role": "user", "content": [
              {"type": "text", "text": "describe"},
              {"type": "image_url", "image_url": {"url": "x"}},
              {"type": "text", "text": "this"}]}]}) == "describe\nthis")
    check("chat without a user message → none",
          extract_text({"messages": [{"role": "system", "content": "s"}]}) is None)
    check("completions: the prompt", extract_text({"prompt": "hello"}) == "hello")
    check("completions: first prompt of a list",
          extract_text({"prompt": ["a", "b"]}) == "a")
    check("token-id prompt → none", extract_text({"prompt": [1, 2, 3]}) is None)
    check("nothing to route on → none", extract_text({}) is None)
    check("responses API: a string input", extract_text({"input": "hi there"}) == "hi there")
    check("responses API: the last user message of an input list",
          extract_text({"input": [
              {"role": "user", "content": "first"},
              {"role": "assistant", "content": "ok"},
              {"role": "user", "content": [
                  {"type": "input_text", "text": "translate this"},
                  {"type": "input_image", "image_url": "x"}]}]}) == "translate this")


def test_router():
    green("router:")
    from deltaserve.adapter_routing.router import Router

    with tempfile.TemporaryDirectory() as d:
        f = str(Path(d) / "centroids.npz")
        enc = FakeEncoder()
        r = Router(enc, f, threshold=0.5, reload_s=0.0)
        dec = r.route("translate this")
        check("no index file → fallback 'no-adapters', encoder not called",
              dec.adapter is None and dec.reason == "no-adapters" and enc.calls == 0)
        write_index(f, ["translate", "sql"])
        dec = r.route("please translate this sentence")
        check("index appears → picked up; top-1 routed",
              dec.adapter == "dserve-routing-translate" and dec.outcome == "routed"
              and dec.score > 0.9 and dec.reason == "")
        dec = r.route("write a poem")
        check("no adapter close enough → fallback 'below-threshold' with score",
              dec.adapter is None and dec.outcome == "fallback"
              and dec.reason == "below-threshold" and dec.score < 0.5)
        os.utime(f, ns=(1, 1))             # make the rewrite visible to mtime
        write_index(f, ["translate", "sql", "poem"])
        check("index file updated → the new adapter is routable without a restart",
              r.route("write a poem").adapter == "dserve-routing-poem")
        check("empty / missing text → fallback 'no-text'",
              r.route("  ").reason == "no-text" and r.route(None).reason == "no-text")
        os.unlink(f)
        check("index file removed → fallback 'no-adapters'",
              r.route("translate").reason == "no-adapters")


def tokenize(text):
    """Stand-in for the route tokenizer: the text's bytes as token ids."""
    return list(text.encode())


class Engine:
    """The engine side of a routed request: an ``EngineRouter`` whose encode
    step is the fake encoder and whose decisions come from a real index file
    (``Router.route_embeddings``). A request released to "the scheduler"
    answers through the API server's ``DecisionRegistry``, as its outputs do."""

    def __init__(self, index_file, registry, threshold=0.5, auto=True):
        from deltaserve.adapter_routing.engine_router import EngineRouter
        from deltaserve.adapter_routing.router import Router

        self.registry, self.auto = registry, auto
        self.encoder = FakeEncoder()
        self.decider = Router(None, index_file, threshold=threshold, reload_s=0.0)
        self.batches, self.added, self.gone, self.n = [], [], set(), 0
        self.router = EngineRouter(self._encode, self.decider.route_embeddings,
                                   self._lora_for, self._add)

    def _encode(self, batch):
        self.batches.append(len(batch))
        return self.encoder.encode([bytes(ids).decode() for ids in batch])

    def _lora_for(self, name):
        from types import SimpleNamespace

        if name in self.gone:
            return None
        return SimpleNamespace(lora_name=name, lora_int_id=7, lora_path=f"/p/{name}")

    def _add(self, request, *args, **kwargs):
        self.added.append(request)
        self.registry.note_headers(request.trace_headers)

    def take(self, payload):
        """add_request for a request carrying ``payload`` in its trace headers
        (and, with ``auto``, the next schedule())."""
        from types import SimpleNamespace

        from deltaserve.route_middleware import ROUTE_HEADER

        self.n += 1
        request = SimpleNamespace(request_id=f"r{self.n}", lora_request=None,
                                  trace_headers={ROUTE_HEADER: payload})
        self.router.hold(request)
        if self.auto:
            self.router.route_pending()
        return request

    def adapter_of_last(self):
        lora = self.added[-1].lora_request
        return lora.lora_name if lora is not None else None


async def call(mw, body, path="/v1/chat/completions", method="POST",
               split=False, app=None, engine=None):
    """Run one request through the middleware; returns what the app saw and
    what the client got. ``engine``: where a routed request's payload goes."""
    from deltaserve.route_middleware import ROUTE_PAYLOAD

    seen = {}

    async def default_app(scope, receive, send):
        chunks = []
        while True:
            m = await receive()
            chunks.append(m.get("body", b""))
            if not m.get("more_body", False):
                break
        seen["body"] = b"".join(chunks)
        seen["headers"] = dict(scope["headers"])
        seen["payload"] = ROUTE_PAYLOAD.get()
        if seen["payload"] is not None and engine is not None:
            engine.take(seen["payload"])
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"text/event-stream")]})
        for part in (b"data: a\n\n", b"data: b\n\n"):
            await send({"type": "http.response.body", "body": part,
                        "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    mw.app = app or default_app
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    parts = [raw[:len(raw) // 2], raw[len(raw) // 2:]] if split else [raw]
    queue = [{"type": "http.request", "body": p, "more_body": i < len(parts) - 1}
             for i, p in enumerate(parts)]

    async def receive():
        return queue.pop(0) if queue else {"type": "http.disconnect"}

    out = {"headers": {}, "body": b"", "chunks": 0}

    async def send(m):
        if m["type"] == "http.response.start":
            out["status"] = m["status"]
            out["headers"] = {k.decode(): v.decode() for k, v in m["headers"]}
        else:
            out["body"] += m.get("body", b"")
            out["chunks"] += 1

    scope = {"type": "http", "method": method, "path": path,
             "headers": [(b"content-type", b"application/json"),
                         (b"content-length", str(len(raw)).encode())]}
    await mw(scope, receive, send)
    return seen, out


def no_headers(out):
    return not any(k.startswith("x-deltaserve") for k in out["headers"])


def test_middleware():
    green("API-server middleware:")
    from deltaserve import route_middleware as rm
    from deltaserve.adapter_routing.middleware import AdapterRouteMiddleware
    from deltaserve.route_middleware import (
        DECISION_HEADER,
        ROUTE_HEADER,
        ROUTE_PAYLOAD,
        DecisionRegistry,
        RouteDecision,
        decode_route,
        encode_decision,
    )

    calls = []

    def tok(text):
        calls.append(text)
        return tokenize(text)

    run = asyncio.run
    with tempfile.TemporaryDirectory() as d:
        f = str(Path(d) / "centroids.npz")
        write_index(f, ["translate", "sql"])
        reg = DecisionRegistry()
        eng = Engine(f, reg)
        mw = AdapterRouteMiddleware(None, tok, "auto",
                                    lambda: "Qwen/Qwen3-14B-Base", registry=reg)

        body = {"model": "auto", "stream": True,
                "messages": [{"role": "user", "content": "translate: hallo"}]}
        seen, out = run(call(mw, body, engine=eng))
        got = json.loads(seen["body"])
        check("model 'auto' goes on as a base-model request",
              got["model"] == "Qwen/Qwen3-14B-Base")
        check("the rest of the request is unchanged",
              {k: v for k, v in got.items() if k != "model"}
              == {k: v for k, v in body.items() if k != "model"})
        check("content-length matches the rewritten body",
              int(seen["headers"][b"content-length"]) == len(seen["body"]))
        key, ids = decode_route(seen["payload"])
        check("routed on the last user message: its encoder tokens travel to "
              "the engine under a fresh key",
              calls[-1] == "translate: hallo" and ids == tokenize("translate: hallo")
              and len(key) == 32)
        check("the engine attaches the adapter the router chose",
              eng.adapter_of_last() == "dserve-routing-translate")
        h = out["headers"]
        check("response says: routed, which adapter, score",
              h.get("x-deltaserve-routing") == "routed"
              and h.get("x-deltaserve-adapter") == "dserve-routing-translate"
              and float(h.get("x-deltaserve-routing-score", 0)) > 0.9
              and len(h["x-deltaserve-routing-score"].split(".")[1]) == 4
              and "x-deltaserve-routing-reason" not in h)
        check("streaming response passes through chunk by chunk",
              out["body"] == b"data: a\n\ndata: b\n\n" and out["chunks"] == 3
              and h.get("content-type") == "text/event-stream")
        check("nothing is left waiting for a decision", not reg.waiting)

        seen, out = run(call(mw, {"model": "auto", "prompt": "write a poem"},
                             path="/v1/completions", engine=eng))
        h = out["headers"]
        check("fallback → served by the base model, no adapter attached",
              json.loads(seen["body"])["model"] == "Qwen/Qwen3-14B-Base"
              and eng.adapter_of_last() is None)
        check("response says: fallback, why, score, a message; no adapter header",
              h.get("x-deltaserve-routing") == "fallback"
              and h.get("x-deltaserve-routing-reason") == "below-threshold"
              and float(h.get("x-deltaserve-routing-score", 1)) < 0.5
              and h.get("x-deltaserve-routing-message", "").startswith(
                  "no adapter matched")
              and "x-deltaserve-adapter" not in h)

        # A stream's response starts before its first token: the start is held
        # until the engine's decision is back.
        state = {}

        async def early_app(scope, receive, send):
            await receive()
            payload = ROUTE_PAYLOAD.get()
            start = asyncio.ensure_future(send(
                {"type": "http.response.start", "status": 200, "headers": []}))
            await asyncio.sleep(0.03)
            state["held"] = not start.done()
            eng.take(payload)
            await start
            await send({"type": "http.response.body", "body": b"tok"})

        seen, out = run(call(mw, body, app=early_app))
        check("the response start waits for the engine's decision and carries it",
              state["held"] and out["status"] == 200
              and out["headers"].get("x-deltaserve-adapter")
              == "dserve-routing-translate" and out["body"] == b"tok")

        async def err_app(scope, receive, send):
            await receive()
            await send({"type": "http.response.start", "status": 400, "headers": []})
            await send({"type": "http.response.body", "body": b"bad"})

        t0 = time.time()
        seen, out = run(call(mw, body, app=err_app))
        check("an error response that never reached the engine is not held, "
              "and has no routing headers",
              out["status"] == 400 and no_headers(out) and time.time() - t0 < 1
              and not reg.waiting)

        keep_wait, rm.DECISION_WAIT_S = rm.DECISION_WAIT_S, 0.05
        try:
            t0 = time.time()
            seen, out = run(call(mw, body))             # nobody answers
            took = time.time() - t0
        finally:
            rm.DECISION_WAIT_S = keep_wait
        check("an engine that never answers is waited for a bounded time: "
              "'router-timeout', the response still goes out",
              out["status"] == 200 and 0.04 <= took < 1
              and out["headers"].get("x-deltaserve-routing-reason") == "router-timeout"
              and not reg.waiting)

        # arrival_time: stamped when the request reaches the API server, before
        # anything is done for the route; the encoder tokens are handed to the
        # engine request by the plugin's wrapper on process_inputs.
        from deltaserve import plugin as _plugin
        from deltaserve.route_middleware import REQUEST_ARRIVAL

        seen_at = {}

        def slow_tok(text):
            time.sleep(0.05)
            return tokenize(text)

        class Proc:
            def process_inputs(self, request_id, prompt, params,
                               supported_tasks=(), arrival_time=None,
                               lora_request=None, tokenization_kwargs=None,
                               trace_headers=None):
                return arrival_time, trace_headers
        make = {}
        orig_wrap = _plugin._wrap
        _plugin._wrap = lambda cls, name, mk: make.setdefault(name, mk)
        _plugin._REGISTERED = False         # collect the wrappers even if loaded
        try:
            _plugin.register()
        finally:
            _plugin._wrap = orig_wrap
        wrapped = make["process_inputs"](Proc.process_inputs)

        async def app(scope, receive, send):
            await receive()
            seen_at["var"] = REQUEST_ARRIVAL.get()
            seen_at["payload"] = ROUTE_PAYLOAD.get()
            seen_at["stamp"], seen_at["trace"] = wrapped(
                Proc(), "r", "p", None, supported_tasks=(),
                trace_headers={"traceparent": "00-x"})
            seen_at["given"], seen_at["second"] = wrapped(
                Proc(), "r", "p", None, supported_tasks=(), arrival_time=123.0)
            seen_at["now"] = time.time()
            eng.take(seen_at["trace"][ROUTE_HEADER])
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        slow = AdapterRouteMiddleware(app, slow_tok, "auto", lambda: "base",
                                      registry=reg)
        t_in = time.time()
        seen, out = run(call(slow, {"model": "auto", "prompt": "translate"},
                             path="/v1/completions", app=app))
        check("the request's arrival time is taken before it is prepared "
              f"({(seen_at['now'] - seen_at['stamp']) * 1e3:.0f} ms earlier than the "
              "moment it reaches vLLM)",
              seen_at["var"] is not None and seen_at["stamp"] == seen_at["var"]
              and t_in <= seen_at["stamp"] <= t_in + 0.02
              and seen_at["now"] - seen_at["stamp"] >= 0.05)
        check("an arrival_time given by the caller is kept", seen_at["given"] == 123.0)
        check("the route payload is put into the engine request's trace_headers "
              "under ROUTE_HEADER, other trace headers kept",
              seen_at["payload"] is not None
              and seen_at["trace"] == {"traceparent": "00-x",
                                       ROUTE_HEADER: seen_at["payload"]})
        check("... once: a second engine request of the same HTTP request "
              "carries no payload", seen_at["second"] is None)
        check("and the decision made from that payload reaches the client",
              out["headers"].get("x-deltaserve-adapter") == "dserve-routing-translate")
        check("outside a routed endpoint vLLM stamps the arrival itself and "
              "nothing is added (nothing passed)",
              wrapped(Proc(), "r", "p", None, supported_tasks=()) == (None, None))

        n = len(calls)
        named = json.dumps({"model":   "my-adapter", "prompt": "translate"},
                           indent=3).encode()          # odd formatting on purpose
        seen, out = run(call(mw, named, path="/v1/completions", engine=eng))
        check("a request naming an adapter passes byte-for-byte: not tokenized, "
              "no payload, no headers",
              seen["body"] == named and len(calls) == n
              and seen["payload"] is None and no_headers(out))
        base = json.dumps({"model": "Qwen/Qwen3-14B-Base", "prompt": "translate"}).encode()
        seen, out = run(call(mw, base, path="/v1/completions", engine=eng))
        check("a request naming the base model passes untouched",
              seen["body"] == base and len(calls) == n and no_headers(out))

        seen, out = run(call(mw, body, split=True, engine=eng))
        check("a request body arriving in two chunks is reassembled and routed",
              json.loads(seen["body"])["model"] == "Qwen/Qwen3-14B-Base"
              and decode_route(seen["payload"])[1] == tokenize("translate: hallo")
              and out["headers"].get("x-deltaserve-adapter")
              == "dserve-routing-translate")
        seen, out = run(call(mw, b"not json {", path="/v1/completions", engine=eng))
        check("a non-JSON body passes through for vLLM to reject",
              seen["body"] == b"not json {" and out["status"] == 200
              and no_headers(out))
        seen, out = run(call(mw, {"model": "auto", "prompt": "translate"},
                             path="/v1/embeddings", engine=eng))
        check("other endpoints are not routed",
              json.loads(seen["body"])["model"] == "auto" and len(calls) == n + 1
              and seen["payload"] is None and no_headers(out))
        added = len(eng.added)
        seen, out = run(call(mw, {"model": "auto", "prompt": [1, 2, 3]},
                             path="/v1/completions", engine=eng))
        check("token-id prompt: nothing to route on → base model, 'no-text', "
              "answered without the engine's router",
              len(calls) == n + 1 and seen["payload"] is None
              and len(eng.added) == added
              and json.loads(seen["body"])["model"] == "Qwen/Qwen3-14B-Base"
              and out["headers"].get("x-deltaserve-routing") == "fallback"
              and out["headers"].get("x-deltaserve-routing-reason") == "no-text")

        def boom(text):
            raise RuntimeError("tokenizer exploded")

        mwb = AdapterRouteMiddleware(None, boom, "auto", "base-model", registry=reg)
        seen, out = run(call(mwb, body, engine=eng))
        check("a failing tokenizer never fails the request: base model, "
              "'router-unavailable'",
              json.loads(seen["body"])["model"] == "base-model"
              and seen["payload"] is None
              and out["headers"].get("x-deltaserve-routing-reason")
              == "router-unavailable" and out["status"] == 200)

        names = []
        mwh = AdapterRouteMiddleware(None, tok, "auto", "base-model",
                                     before_model=names.append, registry=reg)
        run(call(mwh, body, engine=eng))
        run(call(mwh, {"model": "my-adapter", "prompt": "x"}, path="/v1/completions",
                 engine=eng))
        seen, out = run(call(mwh, {"model": "auto", "input": "translate this"},
                             path="/v1/responses", engine=eng))
        check("the before-model hook sees every request's final model name "
              "(a routed request's is the base model)",
              names == ["base-model", "my-adapter", "base-model"])
        check("/v1/responses is routed too",
              out["headers"].get("x-deltaserve-adapter") == "dserve-routing-translate")

        def bad_hook(name):
            raise RuntimeError("hook exploded")

        mwx = AdapterRouteMiddleware(None, tok, "auto", "base-model",
                                     before_model=bad_hook, registry=reg)
        seen, out = run(call(mwx, body, engine=eng))
        check("a failing hook never fails the request",
              out["status"] == 200
              and out["headers"].get("x-deltaserve-routing") == "routed")

        mwa = AdapterRouteMiddleware(None, tok, "pick", "base-model", registry=reg)
        seen, out = run(call(mwa, {"model": "pick", "prompt": "a sql query"},
                             path="/v1/completions", engine=eng))
        check("custom route name",
              json.loads(seen["body"])["model"] == "base-model"
              and out["headers"].get("x-deltaserve-adapter") == "dserve-routing-sql")
        raw = json.dumps({"model": "auto", "prompt": "x"}).encode()
        seen, out = run(call(mwa, raw, path="/v1/completions", engine=eng))
        check("'auto' is an ordinary name when the route name is something else",
              seen["body"] == raw and seen["payload"] is None and no_headers(out))

        # The registry itself.
        async def registry_cases():
            r2 = DecisionRegistry()
            fut = r2.expect("k1")
            r2.note_headers({DECISION_HEADER: "garbage"})
            r2.note_headers(None)
            r2.note_headers({"traceparent": "00-x"})
            r2.note_headers({DECISION_HEADER: encode_decision(
                "other", RouteDecision("a", 0.9))})
            await asyncio.sleep(0)
            ignored = not fut.done() and list(r2.waiting) == ["k1"]
            r2.note_headers({DECISION_HEADER: encode_decision(
                "k1", RouteDecision("a", 0.9))})
            got = await asyncio.wait_for(fut, 1)
            return ignored, got, dict(r2.waiting)

        ignored, got, left = run(registry_cases())
        check("decision registry: unreadable / missing / someone else's decisions "
              "are ignored; its own resolves the waiting request once",
              ignored and got.adapter == "a" and got.score == 0.9 and left == {})


class FakeApp:
    """Records what attach_router adds to vLLM's FastAPI app."""

    def __init__(self, additional_config, served=None):
        from types import SimpleNamespace

        self.state = SimpleNamespace(args=SimpleNamespace(
            additional_config=additional_config, model="org/base",
            served_model_name=served))
        self.middlewares, self.routers = [], []

    def add_middleware(self, cls, **kw):
        self.middlewares.append((cls, kw))

    def include_router(self, router):
        self.routers.append(router)


def test_routing_off():
    green("adapter routing OFF: auto requests are served by the base model:")
    from deltaserve.api import attach_router
    from deltaserve.route_middleware import AdapterRouteMiddleware

    mw = AdapterRouteMiddleware(None, None, "auto", "org/base")
    body = {"model": "auto", "messages": [{"role": "user", "content": "translate"}]}
    seen, out = asyncio.run(call(mw, body))
    h = out["headers"]
    check("auto request → base model, nothing sent to the engine's router",
          json.loads(seen["body"])["model"] == "org/base"
          and seen["payload"] is None)
    check("headers: fallback, reason 'adapter-routing-disabled'",
          h.get("x-deltaserve-routing") == "fallback"
          and h.get("x-deltaserve-routing-reason") == "adapter-routing-disabled"
          and "x-deltaserve-adapter" not in h
          and "x-deltaserve-routing-score" not in h)
    check("and a plain-language message",
          h.get("x-deltaserve-routing-message", "").startswith(
              "adapter routing is not enabled on this server"))
    named = json.dumps({"model": "my-adapter", "prompt": "x"}).encode()
    seen, out = asyncio.run(call(mw, named, path="/v1/completions"))
    check("named requests still pass untouched, no headers",
          seen["body"] == named and no_headers(out))

    app = FakeApp({"deltaserve": {"enable_finetuning": True}}, served=["served-name"])
    attach_router(app)
    check("finetuning server without a routing section → the routing-off "
          "middleware (no tokenizer) is installed, route name 'auto'",
          len(app.middlewares) == 1
          and app.middlewares[0][0] is AdapterRouteMiddleware
          and app.middlewares[0][1]["tokenize"] is None
          and app.middlewares[0][1]["route_model_name"] == "auto")
    check("fallback target is the served model name",
          app.middlewares[0][1]["base_model_name"]() == "served-name")
    app = FakeApp({"deltaserve": {"enable_finetuning": False},
                   "deltaserve_adapter_routing": {
                       "enable_adapter_routing": False, "route_model_name": "pick"}})
    attach_router(app)
    check("routing section present but off → same, with its route name; "
          "falls back to the model path when no served name is set",
          app.middlewares[0][1]["tokenize"] is None
          and app.middlewares[0][1]["route_model_name"] == "pick"
          and app.middlewares[0][1]["base_model_name"]() == "org/base")
    app = FakeApp({"some_other_plugin": {}})
    attach_router(app)
    check("a plain vLLM launch (no DeltaServe entry) gets no middleware",
          app.middlewares == [] and app.routers == [])
    app = FakeApp(None)
    attach_router(app)
    check("no additional_config at all → nothing installed", app.middlewares == [])

    from deltaserve.adapter_routing.middleware import RouteTokenizer
    from deltaserve.config import AdapterRoutingConfig

    with tempfile.TemporaryDirectory() as d:
        app = FakeApp(json.dumps({"deltaserve_adapter_routing": {
            "enable_adapter_routing": True, "index_path": d}}))
        attach_router(app)
        kw = app.middlewares[0][1]
        t = kw["tokenize"]
        cfg = AdapterRoutingConfig()
        check("routing ON → the route encoder's tokenizer is plugged in (same "
              "model, prefix and truncation as the index; loaded on first use), "
              "with the cache-invalidation hook",
              len(app.middlewares) == 1 and isinstance(t, RouteTokenizer)
              and (t.model, t.prefix, t.max_tokens)
              == (cfg.encoder, cfg.encoder_query_prefix, cfg.encoder_max_tokens)
              and t._tok is None and callable(kw["before_model"])
              and kw["route_model_name"] == "auto")


def test_end_to_end():
    green("middleware → route payload → engine router → decision → headers:")
    from deltaserve.adapter_routing.middleware import AdapterRouteMiddleware
    from deltaserve.route_middleware import DecisionRegistry

    run = asyncio.run
    with tempfile.TemporaryDirectory() as d:
        f = str(Path(d) / "centroids.npz")
        write_index(f, ["translate", "sql"])
        reg = DecisionRegistry()
        eng = Engine(f, reg)
        mw = AdapterRouteMiddleware(None, tokenize, "auto", "base-model",
                                    registry=reg)

        def chat(text):
            return {"model": "auto", "messages": [{"role": "user", "content": text}]}

        seen, out = run(call(mw, chat("write a SQL query for top users"), engine=eng))
        check("auto request → base-model request; the engine attaches the sql "
              "adapter; the header names it",
              json.loads(seen["body"])["model"] == "base-model"
              and eng.adapter_of_last() == "dserve-routing-sql"
              and out["headers"]["x-deltaserve-adapter"] == "dserve-routing-sql")
        seen, out = run(call(mw, chat("tell me a joke"), engine=eng))
        check("unmatched request → base model + fallback header",
              json.loads(seen["body"])["model"] == "base-model"
              and eng.adapter_of_last() is None
              and out["headers"]["x-deltaserve-routing"] == "fallback"
              and out["headers"]["x-deltaserve-routing-reason"] == "below-threshold")

        seen, out = run(call(mw, chat("write a poem"), engine=eng))
        before = out["headers"].get("x-deltaserve-routing")
        os.utime(f, ns=(1, 1))             # make the rewrite visible to mtime
        write_index(f, ["translate", "sql", "poem"])
        seen, out = run(call(mw, chat("write a poem"), engine=eng))
        check("index file updated → the new adapter is chosen without a restart",
              before == "fallback"
              and out["headers"].get("x-deltaserve-adapter") == "dserve-routing-poem")

        # Requests that arrive together share ONE encode step, and each gets
        # the decision for its own text.
        lazy = Engine(f, reg, auto=False)
        texts = ["a sql query", "translate this", "tell me a joke", "a poem please"]

        async def together():
            async def scheduler():
                while len(lazy.router.held) < len(texts):
                    await asyncio.sleep(0.001)
                lazy.router.route_pending()

            res = await asyncio.gather(
                *(call(mw, chat(t), engine=lazy) for t in texts), scheduler())
            return res[:len(texts)]

        res = run(together())
        got = [o["headers"].get("x-deltaserve-adapter") for _, o in res]
        check(f"four concurrent requests → one encode step (sizes {lazy.batches})",
              lazy.batches == [4])
        check("every caller gets the decision for ITS text (no cross-wiring)",
              got == ["dserve-routing-sql", "dserve-routing-translate", None,
                      "dserve-routing-poem"]
              and res[2][1]["headers"].get("x-deltaserve-routing") == "fallback"
              and not reg.waiting)

        eng.gone.add("dserve-routing-sql")
        seen, out = run(call(mw, chat("a sql query"), engine=eng))
        check("an adapter chosen from the index but gone from the pool → base "
              "model, and the response says fallback 'router-unavailable'",
              eng.adapter_of_last() is None
              and out["headers"]["x-deltaserve-routing"] == "fallback"
              and out["headers"]["x-deltaserve-routing-reason"] == "router-unavailable"
              and "x-deltaserve-adapter" not in out["headers"])

        class Broken(FakeEncoder):
            def encode(self, texts):
                raise RuntimeError("cuda out of memory")

        eng.encoder = Broken()
        seen, out = run(call(mw, {"model": "auto", "prompt": "translate"},
                             path="/v1/completions", engine=eng))
        check("encode step failing mid-flight → request still served by the "
              "base model",
              json.loads(seen["body"])["model"] == "base-model"
              and out["status"] == 200 and eng.adapter_of_last() is None
              and out["headers"]["x-deltaserve-routing-reason"]
              == "router-unavailable")
        eng.encoder = FakeEncoder()
        seen, out = run(call(mw, chat("translate"), engine=eng))
        check("... and the next request is routed again",
              out["headers"].get("x-deltaserve-adapter") == "dserve-routing-translate")

        os.unlink(f)
        seen, out = run(call(mw, chat("translate"), engine=eng))
        check("index file removed → fallback 'no-adapters'",
              out["headers"]["x-deltaserve-routing-reason"] == "no-adapters"
              and out["headers"].get("x-deltaserve-routing-message", "").startswith(
                  "no routable adapter"))
        st = eng.router.status()
        check("the engine router's counters add up; nothing held",
              st["held"] == 0 and st["routed"] + st["fallbacks"] == len(eng.added)
              and st["routed"] == 3 and not reg.waiting)


def test_real_encoder():
    if os.environ.get("DSERVE_TEST_REAL_ENCODER") != "1":
        green("real encoder on CPU: skipped (set DSERVE_TEST_REAL_ENCODER=1)")
        return
    green("real e5 on CPU: index two tiny datasets, then route:")
    from deltaserve.adapter_routing.encoder import SentenceEncoder
    from deltaserve.adapter_routing.index import CentroidIndex, Indexer, fingerprint
    from deltaserve.adapter_routing.router import Router

    data = {
        "dserve-routing-translate": [
            "Translate to English: Guten Morgen, wie geht es dir?",
            "Translate to English: Das Wetter ist heute schön.",
            "Translate to English: Ich möchte einen Kaffee bestellen.",
            "Translate to English: Wo ist der Bahnhof?"],
        "dserve-routing-sql": [
            "Write a SQL query that returns the ten newest orders.",
            "Write a SQL query counting users per country.",
            "Write a SQL query joining customers and invoices.",
            "Write a SQL query that finds duplicate email addresses."],
    }
    enc = SentenceEncoder("intfloat/e5-large-v2", "query: ", 512, "cpu")
    with tempfile.TemporaryDirectory() as d:
        f = str(Path(d) / "centroids.npz")
        idx = CentroidIndex(enc.model_name, enc.prefix, enc.max_tokens)
        for name, texts in data.items():
            ix = Indexer(texts, fingerprint(texts, enc.model_name, enc.prefix, 512))
            while not ix.done:
                s, chunk = ix.next_chunk(2)
                ix.commit(s, enc.encode(chunk))
            idx.add(name, ix.centroid())
        idx.save(f)
        r = Router(enc, f, threshold=0.0)
        a = r.route("Translate to English: Wir fahren morgen nach Berlin.")
        b = r.route("Write a SQL query listing products that were never sold.")
        c = r.route("What is the capital of Australia?")
        print(f"    scores: translate {a.score:.3f}, sql {b.score:.3f}, "
              f"unrelated {c.score:.3f} (→ {c.adapter})")
        check("an unseen translation prompt routes to the translation adapter",
              a.adapter == "dserve-routing-translate")
        check("an unseen SQL prompt routes to the SQL adapter",
              b.adapter == "dserve-routing-sql")
        check("an unrelated prompt scores lower than both matches",
              c.score < min(a.score, b.score))


def main():
    green("=== adapter routing: request path ===")
    test_extract_text()
    test_router()
    test_middleware()
    test_routing_off()
    test_end_to_end()
    test_real_encoder()
    print()
    if _failed:
        print(f"{_RED}{_failed} failed{_RESET}, {_passed} passed")
        sys.exit(1)
    green(f"all {_passed} checks passed")


if __name__ == "__main__":
    main()
