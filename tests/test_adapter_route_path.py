#!/usr/bin/env python
"""The request path of adapter routing: router, router service + client, and
the API-server middleware (deltaserve.adapter_routing.{service,middleware}).

CPU-only. The encoder is a deterministic fake except in the last, opt-in test:

    python tests/test_adapter_route_path.py
    DSERVE_TEST_REAL_ENCODER=1 python tests/test_adapter_route_path.py   # + real e5 on CPU

Checks
  * which text a request is routed on (chat: last user message; completions:
    the prompt; token-id prompts: none);
  * the router: top-1, fallback below the threshold / on an empty index / on no
    text, and it follows the index file as it changes;
  * the service over its unix socket; a client that never raises — down, slow
    or broken service → fallback, and it reconnects;
  * the middleware: only ``model == route name`` is rewritten (to the adapter,
    or to the base model on fallback) and gets the x-deltaserve-* headers;
    everything else passes through byte-for-byte; streaming responses and
    chunked request bodies work; a failing router never fails the request.
"""

import asyncio
import json
import os
import sys
import tempfile
import threading
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
    from deltaserve.adapter_routing.service import Router

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


class Service:
    """The router service on a thread, for the protocol tests."""

    def __init__(self, router, path):
        from deltaserve.adapter_routing.service import serve

        self.stop, ready = threading.Event(), threading.Event()
        self.thread = threading.Thread(
            target=serve, args=(router, path, self.stop, ready), daemon=True)
        self.thread.start()
        assert ready.wait(5)

    def close(self):
        self.stop.set()
        self.thread.join(5)


def test_service_and_client():
    green("router service + client:")
    from deltaserve.adapter_routing.service import Router, RouterClient, socket_path

    with tempfile.TemporaryDirectory() as d:
        f = str(Path(d) / "centroids.npz")
        write_index(f, ["translate", "sql"])
        path = socket_path(d)
        check("socket path is short, per user, per index directory",
              len(path) < 100 and str(os.getuid()) in path
              and path != socket_path(d + "x"))

        client = RouterClient(path, timeout_s=1.0)
        dec = client.route("translate")
        check("service not running → fallback 'router-unavailable', no exception",
              dec.adapter is None and dec.reason == "router-unavailable")
        check("... and the client keeps why (it is the only trace of a fallback)",
              client.failures == 1 and bool(client.last_error))

        enc = FakeEncoder()
        svc = Service(Router(enc, f, threshold=0.5, reload_s=0.0), path)
        try:
            check("socket file is owner-only",
                  (os.stat(path).st_mode & 0o077) == 0)
            dec = client.route("translate to english")
            check("client reconnects once the service is up; routed",
                  dec.adapter == "dserve-routing-translate" and dec.score > 0.9)
            check("fallback reason travels through the service",
                  client.route("a poem").reason == "below-threshold")
            check("ping reports the index size", client.ping()[0] == 2)

            results = []

            def worker(i):
                c = RouterClient(path, timeout_s=2.0)
                for _ in range(10):
                    results.append(c.route("sql query" if i % 2 else "translate").adapter)
                c.close()

            ts = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
            [t.start() for t in ts]
            [t.join() for t in ts]
            check("four concurrent clients (API servers) all get right answers",
                  len(results) == 40 and results.count("dserve-routing-sql") == 20
                  and results.count("dserve-routing-translate") == 20)

            enc.delay = 0.3
            slow = RouterClient(path, timeout_s=0.05)
            dec = slow.route("translate")
            check("slow service → fallback 'router-timeout'",
                  dec.adapter is None and dec.reason == "router-timeout")
            enc.delay = 0.0
            time.sleep(0.4)
            check("after a timeout the next answer is not a stale one",
                  slow.route("sql please").adapter == "dserve-routing-sql")
        finally:
            svc.close()
        check("service stopped → fallback again, and the socket file is gone",
              client.route("translate").reason == "router-unavailable"
              and not os.path.exists(path))
        svc = Service(Router(FakeEncoder(), f, 0.5, 0.0), path)
        try:
            check("service restarted → the same client works again",
                  client.route("translate").adapter == "dserve-routing-translate")
        finally:
            svc.close()
            client.close()


def test_batching():
    green("request batching in the router service:")
    from deltaserve.adapter_routing.service import (
        RouteBatcher,
        Router,
        RouterClient,
        socket_path,
    )

    class CountingEncoder(FakeEncoder):
        def __init__(self, delay=0.0):
            super().__init__(delay)
            self.sizes = []

        def encode(self, texts):
            self.sizes.append(len(texts))
            return super().encode(texts)

    with tempfile.TemporaryDirectory() as d:
        f = str(Path(d) / "centroids.npz")
        write_index(f, ["translate", "sql"])

        enc = CountingEncoder()
        r = Router(enc, f, threshold=0.5, reload_s=0.0)
        ds = r.route_many(["translate a", "", "sql b", "a poem", None])
        check("route_many: one encoder call for the texts that need encoding",
              enc.sizes == [3] and [x.adapter for x in ds]
              == ["dserve-routing-translate", None, "dserve-routing-sql", None, None])
        check("...and per-item fallback reasons",
              [x.reason for x in ds]
              == ["", "no-text", "", "below-threshold", "no-text"])

        enc = CountingEncoder(delay=0.05)
        b = RouteBatcher(Router(enc, f, 0.5, 0.0), max_batch=8)
        t0 = time.time()
        one = b.submit("translate this")
        alone = time.time() - t0
        check("a request to an idle service is encoded alone, at once "
              f"({alone * 1e3:.0f} ms for a 50 ms encode)",
              one.adapter == "dserve-routing-translate" and enc.sizes == [1]
              and alone < 0.12)

        n = 24
        results = [None] * n

        def worker(i):
            text = f"sql query {i}" if i % 2 else f"translate {i}"
            results[i] = (text, b.submit(text))

        ts = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        t0 = time.time()
        [t.start() for t in ts]
        [t.join() for t in ts]
        took = time.time() - t0
        calls = enc.sizes[1:]
        check(f"24 concurrent requests → {len(calls)} encoder calls "
              f"(sizes {calls}), not 24",
              sum(calls) == n and len(calls) <= 6)
        check("no batch exceeds route_batch_max", max(calls) <= 8)
        check("every caller gets the decision for ITS text (no cross-wiring)",
              all(dec.adapter == ("dserve-routing-sql" if "sql" in text
                                  else "dserve-routing-translate")
                  for text, dec in results))
        check(f"batched: {took * 1e3:.0f} ms total; serial would be "
              f"{n * 50} ms", took < n * 0.05 * 0.5)
        check("batch statistics are kept", b.largest == max(calls)
              and b.batches == len(enc.sizes))
        b.stop()

        enc1 = CountingEncoder(delay=0.01)
        b1 = RouteBatcher(Router(enc1, f, 0.5, 0.0), max_batch=1)
        ts = [threading.Thread(target=b1.submit, args=("translate",))
              for _ in range(6)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        check("route_batch_max = 1 keeps the old one-at-a-time behaviour",
              enc1.sizes == [1] * 6)
        b1.stop()

        class Broken(FakeEncoder):
            def encode(self, texts):
                raise RuntimeError("cuda out of memory")

        bb = RouteBatcher(Router(Broken(), f, 0.5, 0.0))
        errs = []

        def bad():
            try:
                bb.submit("translate")
            except RuntimeError as e:
                errs.append(str(e))

        ts = [threading.Thread(target=bad) for _ in range(3)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        check("an encoder failure reaches every waiting request (nobody hangs)",
              errs == ["cuda out of memory"] * 3)
        bb.stop()

        encw = CountingEncoder()
        bw = RouteBatcher(Router(encw, f, 0.5, 0.0), max_batch=8, wait_s=0.15)
        ts = [threading.Thread(target=bw.submit, args=("translate",))
              for _ in range(4)]
        for t in ts:
            t.start()
            time.sleep(0.02)
        [t.join() for t in ts]
        check("route_batch_wait_ms holds a batch open so staggered requests "
              f"share it (sizes {encw.sizes})", encw.sizes == [4])
        bw.stop()

        enc = CountingEncoder(delay=0.03)
        path = socket_path(d)
        svc = Service(Router(enc, f, 0.5, 0.0), path)
        try:
            out = []

            def client(i):
                c = RouterClient(path, timeout_s=5.0)
                for _ in range(5):
                    out.append(c.route("sql" if i % 2 else "translate").adapter)
                c.close()

            ts = [threading.Thread(target=client, args=(i,)) for i in range(8)]
            [t.start() for t in ts]
            [t.join() for t in ts]
            ping = RouterClient(path).ping()
            check("through the socket: 8 API-server clients × 5 requests, all "
                  f"correct, in {len(enc.sizes)} encoder calls",
                  len(out) == 40 and out.count("dserve-routing-sql") == 20
                  and len(enc.sizes) < 40)
            check("ping reports batching (batches, largest batch)",
                  ping is not None and ping[2] == len(enc.sizes) and ping[3] > 1)
        finally:
            svc.close()


def test_indexing_borrows_the_encoder():
    green("indexing chunks borrow the router's encoder:")
    from deltaserve.adapter_routing.runner import RouterEncodeExecutor
    from deltaserve.adapter_routing.service import (
        RouteBatcher,
        Router,
        RouterClient,
        socket_path,
    )

    order = []

    class Enc(FakeEncoder):
        def encode(self, texts):
            order.append(("chunk" if texts[0].startswith("sample") else "route",
                          len(texts)))
            return super().encode(texts)

    with tempfile.TemporaryDirectory() as d:
        f = str(Path(d) / "centroids.npz")
        write_index(f, ["translate", "sql"])
        enc = Enc(delay=0.03)
        b = RouteBatcher(Router(enc, f, 0.5, 0.0), max_batch=8)
        got = {}

        def chunk(i):
            got[i] = b.encode([f"sample {i} a", f"sample {i} b"])

        def route(i):
            got[f"r{i}"] = b.submit("translate this")

        ts = [threading.Thread(target=chunk, args=(i,)) for i in range(4)]
        [t.start() for t in ts]
        time.sleep(0.045)                      # chunks are being worked through
        rs = [threading.Thread(target=route, args=(i,)) for i in range(3)]
        [t.start() for t in rs]
        [t.join() for t in ts + rs]
        check("a chunk returns the embeddings of its samples, in order",
              got[0].shape == (2, DIM) and all(k in got for k in range(4)))
        first_route = next(i for i, (k, _) in enumerate(order) if k == "route")
        chunks_after = sum(1 for k, _ in order[first_route:] if k == "chunk")
        check(f"route requests jump ahead of waiting chunks ({order})",
              chunks_after >= 1 and first_route <= 2)
        check("chunks are never merged with route requests or each other",
              all(n == 2 for k, n in order if k == "chunk") and b.chunks == 4)
        b.stop()

        path = socket_path(d)
        svc = Service(Router(FakeEncoder(), f, 0.5, 0.0), path)
        try:
            ex = RouterEncodeExecutor(path, timeout_s=5.0)
            check("the GPU executor reports itself as always loaded",
                  ex.device == "gpu" and ex.loaded)
            ex.submit(["sample x", "sample y", "sample z"])
            t0, emb = time.time(), None
            while emb is None and time.time() - t0 < 5:
                emb = ex.poll(0.05)
            check("executor: a chunk goes over the socket and comes back",
                  emb is not None and emb.shape == (3, DIM))
            ex.submit(["sample q"])
            ex.cancel()
            time.sleep(0.2)
            check("a cancelled chunk's result is dropped", ex.poll(0.0) is None)
            ex.submit(["sample again"])
            t0, emb = time.time(), None
            while emb is None and time.time() - t0 < 5:
                emb = ex.poll(0.05)
            check("...and the executor is usable for the next chunk",
                  emb is not None and emb.shape == (1, DIM))
            check("ping counts the chunks", RouterClient(path).ping()[4] >= 2)
        finally:
            svc.close()
        ex.submit(["sample after shutdown"])
        failed = False
        t0 = time.time()
        while time.time() - t0 < 5:
            try:
                if ex.poll(0.05) is not None:
                    break
            except Exception:  # noqa: BLE001
                failed = True
                break
        check("with the service gone the executor raises (the runner then uses "
              "the CPU)", failed)
        ex.stop()


def test_placement():
    green("which device holds the encoder:")
    from deltaserve.adapter_routing.service import (
        apply_router_placement,
        router_placement,
    )
    from deltaserve.config import AdapterRoutingConfig

    def cfg(**kw):
        return AdapterRoutingConfig(enable_adapter_routing=True, **kw)

    keep_vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        check("default (auto): the last GPU the engine can see",
              router_placement(cfg()) == ("cuda:1", None))
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        check("auto with one visible GPU: that GPU",
              router_placement(cfg()) == ("cuda:0", None))
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        check("auto with no GPU: the CPU",
              router_placement(cfg()) == ("cpu", ""))
    finally:
        if keep_vis is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = keep_vis
    check("router_device: cpu → CPU, every GPU hidden from the service",
          router_placement(cfg(router_device="cpu")) == ("cpu", ""))
    check("router_gpu: 3 → the service sees only physical GPU 3, as cuda:0",
          router_placement(cfg(router_gpu=3)) == ("cuda:0", "3"))
    check("router_gpu wins over router_device",
          router_placement(cfg(router_gpu=1, router_device="cpu")) == ("cuda:0", "1"))
    check("router_device cuda:N is used as is (within the engine's GPUs)",
          router_placement(cfg(router_device="cuda:1")) == ("cuda:1", None))
    check("negative GPU index rejected",
          error_raised(lambda: cfg(router_gpu=-1)))
    check("batch size 0 rejected", error_raised(lambda: cfg(route_batch_max=0)))
    keep = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        dev = apply_router_placement(cfg(router_gpu=1))
        check("applying it restricts the process to that card",
              dev == "cuda:0" and os.environ["CUDA_VISIBLE_DEVICES"] == "1")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        apply_router_placement(cfg(router_device="cuda:1"))
        check("router_device leaves the visible GPUs alone",
              os.environ["CUDA_VISIBLE_DEVICES"] == "0,1")
    finally:
        if keep is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = keep


def error_raised(fn):
    try:
        fn()
    except Exception:  # noqa: BLE001
        return True
    return False


async def call(mw, body, path="/v1/chat/completions", method="POST",
               split=False, app=None):
    """Run one request through the middleware; returns what the app saw and
    what the client got."""
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


def test_middleware():
    green("API-server middleware:")
    from deltaserve.adapter_routing.middleware import AdapterRouteMiddleware
    from deltaserve.adapter_routing.service import RouteDecision

    calls = []

    def route(text):
        calls.append(text)
        if text and "translate" in text:
            return RouteDecision("dserve-routing-translate", 0.9321)
        return RouteDecision(None, 0.4123, "below-threshold")

    mw = AdapterRouteMiddleware(None, route, "auto", lambda: "Qwen/Qwen3-14B-Base")
    run = asyncio.run

    body = {"model": "auto", "stream": True,
            "messages": [{"role": "user", "content": "translate: hallo"}]}
    seen, out = run(call(mw, body))
    got = json.loads(seen["body"])
    check("model 'auto' rewritten to the routed adapter",
          got["model"] == "dserve-routing-translate")
    check("the rest of the request is unchanged",
          {k: v for k, v in got.items() if k != "model"}
          == {k: v for k, v in body.items() if k != "model"})
    check("content-length matches the rewritten body",
          int(seen["headers"][b"content-length"]) == len(seen["body"]))
    h = out["headers"]
    check("response says: routed, which adapter, score",
          h.get("x-deltaserve-routing") == "routed"
          and h.get("x-deltaserve-adapter") == "dserve-routing-translate"
          and h.get("x-deltaserve-routing-score") == "0.9321"
          and "x-deltaserve-routing-reason" not in h)
    check("streaming response passes through chunk by chunk",
          out["body"] == b"data: a\n\ndata: b\n\n" and out["chunks"] == 3
          and h.get("content-type") == "text/event-stream")
    check("routed on the last user message", calls[-1] == "translate: hallo")

    seen, out = run(call(mw, {"model": "auto", "prompt": "write a poem"},
                         path="/v1/completions"))
    h = out["headers"]
    check("fallback → served by the base model",
          json.loads(seen["body"])["model"] == "Qwen/Qwen3-14B-Base")
    check("response says: fallback, why, score; no adapter header",
          h.get("x-deltaserve-routing") == "fallback"
          and h.get("x-deltaserve-routing-reason") == "below-threshold"
          and h.get("x-deltaserve-routing-score") == "0.4123"
          and "x-deltaserve-adapter" not in h)

    n = len(calls)
    named = json.dumps({"model":   "my-adapter", "prompt": "translate"},
                       indent=3).encode()          # odd formatting on purpose
    seen, out = run(call(mw, named, path="/v1/completions"))
    check("a request naming an adapter passes byte-for-byte, router not asked",
          seen["body"] == named and len(calls) == n
          and not any(k.startswith("x-deltaserve") for k in out["headers"]))
    base = json.dumps({"model": "Qwen/Qwen3-14B-Base", "prompt": "translate"}).encode()
    seen, out = run(call(mw, base, path="/v1/completions"))
    check("a request naming the base model passes untouched",
          seen["body"] == base and len(calls) == n)

    seen, out = run(call(mw, body, split=True))
    check("a request body arriving in two chunks is reassembled and routed",
          json.loads(seen["body"])["model"] == "dserve-routing-translate")
    seen, out = run(call(mw, b"not json {", path="/v1/completions"))
    check("a non-JSON body passes through for vLLM to reject",
          seen["body"] == b"not json {" and out["status"] == 200)
    seen, out = run(call(mw, {"model": "auto", "prompt": "translate"},
                         path="/v1/embeddings"))
    check("other endpoints are not routed",
          json.loads(seen["body"])["model"] == "auto" and len(calls) == n + 1)
    seen, out = run(call(mw, {"model": "auto", "prompt": [1, 2, 3]},
                         path="/v1/completions"))
    check("token-id prompt: router asked with no text",
          calls[-1] is None and json.loads(seen["body"])["model"]
          == "Qwen/Qwen3-14B-Base")

    def boom(text):
        raise RuntimeError("router exploded")

    mwb = AdapterRouteMiddleware(None, boom, "auto", "base-model")
    seen, out = run(call(mwb, body))
    check("a failing router never fails the request: base model, "
          "'router-unavailable'",
          json.loads(seen["body"])["model"] == "base-model"
          and out["headers"].get("x-deltaserve-routing-reason")
          == "router-unavailable" and out["status"] == 200)

    async def aroute(text):
        return RouteDecision("async-adapter", 0.8)

    names = []
    mwh = AdapterRouteMiddleware(None, route, "auto", "base-model",
                                 before_model=names.append)
    run(call(mwh, body))
    run(call(mwh, {"model": "my-adapter", "prompt": "x"}, path="/v1/completions"))
    run(call(mwh, {"model": "auto", "input": "translate this"},
             path="/v1/responses"))
    check("the before-model hook sees every request's final model name; "
          "/v1/responses is routed too",
          names == ["dserve-routing-translate", "my-adapter",
                    "dserve-routing-translate"])

    def bad_hook(name):
        raise RuntimeError("hook exploded")

    mwx = AdapterRouteMiddleware(None, route, "auto", "base-model",
                                 before_model=bad_hook)
    seen, out = run(call(mwx, body))
    check("a failing hook never fails the request", out["status"] == 200)

    mwa = AdapterRouteMiddleware(None, aroute, "pick", "base-model")
    seen, out = run(call(mwa, {"model": "pick", "prompt": "x"},
                         path="/v1/completions"))
    check("custom route name + async route function",
          json.loads(seen["body"])["model"] == "async-adapter")
    seen, out = run(call(mwa, {"model": "auto", "prompt": "x"},
                         path="/v1/completions"))
    check("'auto' is an ordinary name when the route name is something else",
          json.loads(seen["body"])["model"] == "auto")


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
    from deltaserve.route_middleware import (
        AdapterRouteMiddleware,
        routing_disabled,
    )

    mw = AdapterRouteMiddleware(None, routing_disabled, "auto", "org/base")
    body = {"model": "auto", "messages": [{"role": "user", "content": "translate"}]}
    seen, out = asyncio.run(call(mw, body))
    h = out["headers"]
    check("auto request → base model", json.loads(seen["body"])["model"] == "org/base")
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
          seen["body"] == named
          and not any(k.startswith("x-deltaserve") for k in out["headers"]))

    app = FakeApp({"deltaserve": {"enable_finetuning": True}}, served=["served-name"])
    attach_router(app)
    check("finetuning server without a routing section → the routing-off "
          "middleware is installed, route name 'auto'",
          len(app.middlewares) == 1
          and app.middlewares[0][1]["route_fn"] is routing_disabled
          and app.middlewares[0][1]["route_model_name"] == "auto")
    check("fallback target is the served model name",
          app.middlewares[0][1]["base_model_name"]() == "served-name")
    app = FakeApp({"deltaserve": {"enable_finetuning": False},
                   "deltaserve_adapter_routing": {
                       "enable_adapter_routing": False, "route_model_name": "pick"}})
    attach_router(app)
    check("routing section present but off → same, with its route name; "
          "falls back to the model path when no served name is set",
          app.middlewares[0][1]["route_fn"] is routing_disabled
          and app.middlewares[0][1]["route_model_name"] == "pick"
          and app.middlewares[0][1]["base_model_name"]() == "org/base")
    app = FakeApp({"some_other_plugin": {}})
    attach_router(app)
    check("a plain vLLM launch (no DeltaServe entry) gets no middleware",
          app.middlewares == [] and app.routers == [])
    app = FakeApp(None)
    attach_router(app)
    check("no additional_config at all → nothing installed", app.middlewares == [])

    with tempfile.TemporaryDirectory() as d:
        app = FakeApp(json.dumps({"deltaserve_adapter_routing": {
            "enable_adapter_routing": True, "index_path": d}}))
        attach_router(app)
        fn = app.middlewares[0][1]["route_fn"]
        check("routing ON → the router client is the route function",
              len(app.middlewares) == 1 and fn is not routing_disabled
              and fn("translate").reason == "router-unavailable")


def test_end_to_end():
    green("middleware → client → service → router:")
    from deltaserve.adapter_routing.middleware import AdapterRouteMiddleware
    from deltaserve.adapter_routing.service import Router, RouterClient, socket_path

    with tempfile.TemporaryDirectory() as d:
        f = str(Path(d) / "centroids.npz")
        write_index(f, ["translate", "sql"])
        path = socket_path(d)
        svc = Service(Router(FakeEncoder(), f, 0.5, 0.0), path)
        client = RouterClient(path, timeout_s=2.0)
        mw = AdapterRouteMiddleware(None, client.route, "auto", "base-model")
        try:
            seen, out = asyncio.run(call(mw, {"model": "auto", "messages": [
                {"role": "user", "content": "write a SQL query for top users"}]}))
            check("auto request → sql adapter, through the real socket",
                  json.loads(seen["body"])["model"] == "dserve-routing-sql"
                  and out["headers"]["x-deltaserve-adapter"] == "dserve-routing-sql")
            seen, out = asyncio.run(call(mw, {"model": "auto", "messages": [
                {"role": "user", "content": "tell me a joke"}]}))
            check("unmatched request → base model + fallback header",
                  json.loads(seen["body"])["model"] == "base-model"
                  and out["headers"]["x-deltaserve-routing"] == "fallback")
        finally:
            svc.close()
        seen, out = asyncio.run(call(mw, {"model": "auto", "prompt": "translate"},
                                     path="/v1/completions"))
        check("service gone mid-flight → request still served by the base model",
              json.loads(seen["body"])["model"] == "base-model"
              and out["headers"]["x-deltaserve-routing-reason"]
              == "router-unavailable")
        client.close()


def test_real_encoder():
    if os.environ.get("DSERVE_TEST_REAL_ENCODER") != "1":
        green("real encoder on CPU: skipped (set DSERVE_TEST_REAL_ENCODER=1)")
        return
    green("real e5 on CPU: index two tiny datasets, then route:")
    from deltaserve.adapter_routing.encoder import SentenceEncoder
    from deltaserve.adapter_routing.index import CentroidIndex, Indexer, fingerprint
    from deltaserve.adapter_routing.service import Router

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
    test_service_and_client()
    test_batching()
    test_indexing_borrows_the_encoder()
    test_placement()
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
