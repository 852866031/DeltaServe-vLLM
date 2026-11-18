#!/usr/bin/env python
"""The routing decision (deltaserve.adapter_routing.decision) and the reference
samples it is calibrated from.

CPU-only, synthetic embeddings shaped like real ones: every vector shares a
large common component (the "narrow cone" of sentence encoders), so raw cosines
are all high and close together.

    python tests/test_adapter_decision.py

Checks
  * reference samples: kept during indexing at no extra encoding, spread over
    the dataset, survive a resume, travel with the centroid through the index
    file and the pool;
  * the confidence threshold comes from the reference samples and keeps the
    configured share of them; it moves when the pool changes;
  * three outcomes: a clear winner, several close (chosen by preference, the
    choice reported), low confidence → base;
  * small pools and pools without reference samples degrade as documented;
  * the decision reaches the client: service protocol, log, and — the path a
    request takes — middleware → route payload → the engine's router (fake
    encode step + ``Router.route_embeddings``) → decision header → response
    headers (the body's ``model`` is the base model; the adapter is in the
    headers and on the engine request).
"""

import asyncio
import json
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

_passed = 0
_failed = 0
_TTY = sys.stdout.isatty()
_GREEN = "\033[92m" if _TTY else ""
_RED = "\033[91m" if _TTY else ""
_RESET = "\033[0m" if _TTY else ""
DIM = 64


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


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


class World:
    """K topics in a narrow cone: vector = big common part + topic + noise."""

    def __init__(self, k=8, seed=0, twins=True):
        self.rng = np.random.default_rng(seed)
        self.common = unit(self.rng.normal(size=DIM)) * 4.0
        self.topics = unit(self.rng.normal(size=(k, DIM)))
        if twins:                       # topics 0 and 1 are near-duplicates
            self.topics[1] = unit(self.topics[0] + 0.15 * self.topics[1])
        self.k = k

    def sample(self, topic, n=1, noise=0.25):
        v = self.common + self.topics[topic] + noise * self.rng.normal(size=(n, DIM))
        return unit(v).astype(np.float32)

    def unrelated(self, n=1):
        v = self.common + unit(self.rng.normal(size=(n, DIM))) * 0.6 \
            + 0.25 * self.rng.normal(size=(n, DIM))
        return unit(v).astype(np.float32)

    def index(self, topics=None, refs=48):
        from deltaserve.adapter_routing.index import CentroidIndex

        idx = CentroidIndex("enc", "", 512)
        for t in (range(self.k) if topics is None else topics):
            train = self.sample(t, 400)
            idx.add(f"a{t}", train.mean(0),
                    refs=train[:refs] if refs else None)
        return idx


def test_reference_samples():
    green("reference samples kept during indexing:")
    from deltaserve.adapter_routing.index import CentroidIndex, Indexer

    texts = [f"t{i:03d}" for i in range(100)]
    calls = []

    def embed(chunk):
        calls.append(len(chunk))
        return np.array([[int(t[1:]) + 1.0] + [1.0] * 7 for t in chunk], np.float32)

    def run(ix, upto=None):
        while not ix.done and (upto is None or ix.state.cursor < upto):
            s, chunk = ix.next_chunk(7)
            ix.commit(s, embed(chunk))

    ix = Indexer(texts, "fp", reference_samples=10)
    run(ix)
    refs = ix.references()
    check("the requested number of samples is kept", refs.shape == (10, 8))
    check("no sample is encoded twice for it (100 texts → 100 encodes)",
          sum(calls) == 100)
    first = refs[:, 0] / refs[:, 1]          # undo the normalisation: index + 1
    check("they are spread evenly over the dataset, first to last",
          np.allclose(first, np.linspace(0, 99, 10).round() + 1, atol=1e-3))
    check("they are unit vectors",
          np.allclose(np.linalg.norm(refs, axis=1), 1.0, atol=1e-5))
    check("no references asked → none kept",
          (lambda j: (run(j), j.references())[1])(Indexer(texts, "fp")) is None)
    check("more references than samples → every sample",
          (lambda j: (run(j), j.references())[1])(
              Indexer(texts[:5], "fp5", reference_samples=64)).shape[0] == 5)

    with tempfile.TemporaryDirectory() as d:
        ck = Path(d) / "c.npz"
        a = Indexer(texts, "fp", ck, reference_samples=10)
        run(a, upto=40)
        b = Indexer(texts, "fp", ck, reference_samples=10)
        check("a resume keeps the references collected so far",
              b.resumed_from >= 40)
        run(b)
        check("resumed job ends with the same references",
              np.allclose(b.references(), refs))
        c = Indexer(texts, "fp", ck, reference_samples=20)
        check("a different reference setting restarts the job",
              c.resumed_from == 0)

        idx = CentroidIndex("enc", "q: ", 512)
        idx.add("x", ix.centroid(), refs=refs)
        idx.add("y", np.ones(8))
        p = Path(d) / "i.npz"
        idx.save(p)
        back = CentroidIndex.load(p)
        check("references travel with the centroid through the index file",
              np.allclose(back.refs["x"], refs) and "y" not in back.refs
              and back.names == ["x", "y"])
        back.remove("x")
        check("removing an adapter drops its references", "x" not in back.refs)


def test_threshold():
    green("threshold from the pool's reference samples:")
    from deltaserve.adapter_routing.decision import CENTRED, RoutingModel

    w = World()
    idx = w.index()
    m80 = RoutingModel(idx, keep_percent=80)
    check("a pool of 8 with references uses the centred confidence",
          m80.mode == CENTRED and m80.n_refs == 8 * 48 and m80.threshold is not None)
    refs = np.concatenate([idx.refs[n] for n in idx.names])
    kept = np.mean([m80.decide(r).adapter is not None for r in refs])
    check(f"keep 80% → about 80% of the reference samples are routed "
          f"({100 * kept:.0f}%)", 0.76 <= kept <= 0.84)
    m60 = RoutingModel(idx, keep_percent=60)
    check("a lower keep setting gives a higher threshold",
          m60.threshold > m80.threshold)
    m100 = RoutingModel(idx, keep_percent=100)
    check("keep 100% → no threshold, nothing falls back on confidence",
          m100.threshold is None
          and all(m100.decide(r).adapter is not None for r in refs[:50]))
    ms = RoutingModel(idx, keep_percent=80, slack=1.5)
    check("slack lowers the threshold by that many confidence units",
          abs((m80.threshold - ms.threshold) - 1.5) < 1e-6
          and ms.describe()["slack"] == 1.5)
    idx.add("a_new", w.sample(3, 400).mean(0) * 0 + unit(w.common + w.rng.normal(size=DIM)),
            refs=w.unrelated(48))
    m80b = RoutingModel(idx, keep_percent=80)
    check("the threshold is recomputed when the pool changes",
          abs(m80b.threshold - m80.threshold) > 1e-6 and len(m80b.names) == 9)


def test_three_outcomes():
    green("three outcomes:")
    from deltaserve.adapter_routing.decision import RoutingModel

    w = World(seed=1)
    m = RoutingModel(w.index(), keep_percent=80, close_band=0.05)

    hits, single = 0, 0
    for _ in range(200):
        t = int(w.rng.integers(2, 8))            # the non-twin topics
        c = m.decide(w.sample(t)[0])
        hits += c.adapter == f"a{t}"
        single += len(c.candidates) == 1
    check(f"a request from a distinct topic goes to its adapter "
          f"({hits / 2:.0f}%), mostly with one clear candidate ({single / 2:.0f}%)",
          hits >= 150 and single >= 120)

    raw = np.array([float(c.score) for c in
                    (m.decide(w.sample(int(w.rng.integers(2, 8)))[0])
                     for _ in range(50))])
    rawu = np.array([float(m.decide(u).score) for u in w.unrelated(50)])
    check(f"raw scores barely differ (matching {raw.mean():.3f}, unrelated "
          f"{rawu.mean():.3f}) — the cone", abs(raw.mean() - rawu.mean()) < 0.06)
    low = [m.decide(u) for u in w.unrelated(300)]
    frac = np.mean([c.adapter is None and c.low_confidence for c in low])
    check(f"yet most unrelated requests fall back as low-confidence "
          f"({100 * frac:.0f}%)", frac >= 0.6)
    check("a fallback still reports score, confidence and the nearest candidates",
          low[0].score is not None and low[0].confidence is not None
          and len(low[0].candidates) >= 1)

    twin = [m.decide(w.sample(0)[0]) for _ in range(100)]
    both = np.mean([set(c.candidates[:2]) == {"a0", "a1"} for c in twin
                    if c.adapter])
    check(f"requests for a near-duplicate pair list both as candidates "
          f"({100 * both:.0f}%)", both >= 0.7)
    pref1 = [m.decide(w.sample(0)[0], prefer=lambda n: n == "a1")
             for _ in range(100)]
    routed = [c for c in pref1 if c.adapter and len(c.candidates) > 1]
    check("among close candidates the preferred one (e.g. in a GPU slot) is used",
          routed and all(c.adapter == "a1" for c in routed
                         if "a1" in c.candidates))
    solo = [c for c in (m.decide(w.sample(5)[0], prefer=lambda n: n == "a1")
                        for _ in range(60)) if c.adapter]
    check("a preference never overrides a clear winner",
          all(c.adapter in c.candidates for c in solo)
          and np.mean([c.adapter == "a5" for c in solo]) > 0.8)
    c = m.decide(w.sample(5)[0], raw_floor=0.9999)
    check("the manual raw floor is applied first",
          c.adapter is None and not c.low_confidence)


def test_degraded_modes():
    green("small pools and pools without reference samples:")
    from deltaserve.adapter_routing.decision import RAW, UNGATED, RoutingModel
    from deltaserve.adapter_routing.index import CentroidIndex

    w = World(seed=2, twins=False)
    small = RoutingModel(w.index(topics=[0, 1]), keep_percent=80, min_pool=4)
    check("fewer adapters than route_min_pool → raw-score confidence",
          small.mode == RAW and small.threshold is not None)
    c = small.decide(w.sample(0)[0])
    check("it still routes and reports a confidence; one candidate only",
          c.confidence is not None and len(c.candidates) == 1)
    one = RoutingModel(w.index(topics=[3]), keep_percent=80)
    check("a single adapter works (no margin, no centring)",
          one.mode == RAW and one.decide(w.sample(3)[0]).margin is None)
    norefs = RoutingModel(w.index(refs=0), keep_percent=80)
    c = norefs.decide(w.unrelated(1)[0])
    check("no reference samples → nothing is gated (always top-1), stated as such",
          norefs.mode == UNGATED and norefs.threshold is None
          and c.adapter is not None and c.confidence is None)
    empty = RoutingModel(CentroidIndex(), keep_percent=80)
    check("an empty pool decides nothing", empty.decide(np.ones(DIM)).adapter is None)
    check("describe() says which mode is in force",
          small.describe()["mode"] == RAW and norefs.describe()["mode"] == UNGATED)


def test_reaches_the_client():
    green("the decision reaches the client:")
    from deltaserve.adapter_routing.pool import AdapterPool
    from deltaserve.adapter_routing.service import (
        Router,
        RouterClient,
        serve,
        socket_path,
    )
    from types import SimpleNamespace

    from deltaserve.adapter_routing.engine_router import EngineRouter
    from deltaserve.route_middleware import (
        ROUTE_HEADER,
        ROUTE_PAYLOAD,
        AdapterRouteMiddleware,
        DecisionRegistry,
    )

    w = World(seed=3)

    class Enc:
        def encode(self, texts):
            out = []
            for t in texts:
                kind, _, n = t.partition(":")
                out.append(w.unrelated(1)[0] if kind == "u" else w.sample(int(n))[0])
            return np.array(out)

    class Null:
        def load(self, e): pass
        def activate(self, e): pass
        def deactivate(self, e): pass
        def unload(self, e): pass

    with tempfile.TemporaryDirectory() as d:
        src = w.index()
        pool = AdapterPool(Null(), type(src)("enc", "", 512), gpu_slots=2,
                           directory=d)
        for n in src.names:
            pool.register(n, f"/p/{n}", centroid=src.centroid_of(n),
                          refs=src.refs[n], load=False)
        from deltaserve.adapter_routing.index import CentroidIndex
        from deltaserve.adapter_routing.pool import ROUTABLE_FILE, routable_path

        saved = CentroidIndex.load(routable_path(d))
        check("the pool publishes reference samples with the routable centroids",
              len(saved.refs) == 8 and saved.refs["a0"].shape == (48, DIM))

        log = str(Path(d) / "decisions.jsonl")
        router = Router(Enc(), str(Path(d) / ROUTABLE_FILE), keep_percent=80,
                        reload_s=0.0, decision_log=log)
        path = socket_path(d)
        stop, ready = threading.Event(), threading.Event()
        th = threading.Thread(target=serve, args=(router, path, stop, ready),
                              daemon=True)
        th.start()
        ready.wait(5)
        try:
            client = RouterClient(path, timeout_s=5.0)
            d5 = client.route("t:5")
            check("routed decision over the socket carries confidence and candidates",
                  d5.adapter == "a5" and d5.confidence is not None
                  and d5.candidates and d5.candidates[0] == "a5")
            lows = [client.route("u:") for _ in range(30)]
            low = next(x for x in lows if x.adapter is None)
            check("a low-confidence fallback has its own reason",
                  low.reason == "low-confidence" and low.confidence is not None)

            pool.note_hit("a1")                 # a1 now sits in a GPU slot
            pool.save(d)
            router.refresh(force=True)
            twins = [client.route("t:0") for _ in range(60)]
            both = [x for x in twins if x.adapter and "a1" in x.candidates
                    and len(x.candidates) > 1]
            check("close candidates: the router picks the one the pool has in a "
                  f"GPU slot ({len(both)} such requests)",
                  len(both) >= 20 and all(x.adapter == "a1" for x in both))

            # The path of a request: the API server tokenizes (here: the
            # text's bytes), the engine's router encodes the held requests in
            # one step, decides from the embeddings and attaches the adapter;
            # the decision returns on the request's trace_headers.
            reg = DecisionRegistry()
            enc = Enc()

            def engine_router(added):
                def add(request, *args, **kwargs):
                    added.append(request)
                    reg.note_headers(request.trace_headers)

                return EngineRouter(
                    lambda batch: enc.encode([bytes(ids).decode() for ids in batch]),
                    router.route_embeddings,
                    lambda name: SimpleNamespace(lora_name=name, lora_int_id=1,
                                                 lora_path=f"/p/{name}"),
                    add)

            async def one(prompt):
                seen, out, added = {}, {}, []
                er = engine_router(added)

                async def app(scope, receive, send):
                    seen["body"] = (await receive())["body"]
                    er.hold(SimpleNamespace(
                        request_id="r", lora_request=None,
                        trace_headers={ROUTE_HEADER: ROUTE_PAYLOAD.get()}))
                    er.route_pending()
                    await send({"type": "http.response.start", "status": 200,
                                "headers": []})
                    await send({"type": "http.response.body", "body": b""})

                mw = AdapterRouteMiddleware(app, lambda text: list(text.encode()),
                                            "auto", "base", registry=reg)
                msgs = [{"type": "http.request", "more_body": False,
                         "body": json.dumps({"model": "auto",
                                             "prompt": prompt}).encode()}]

                async def receive():
                    return msgs.pop(0) if msgs else {"type": "http.disconnect"}

                async def send(m):
                    if m["type"] == "http.response.start":
                        out.update({k.decode(): v.decode() for k, v in m["headers"]})

                await mw({"type": "http", "method": "POST",
                          "path": "/v1/completions", "headers": []}, receive, send)
                lora = added[0].lora_request
                return (json.loads(seen["body"])["model"], out,
                        lora.lora_name if lora is not None else None)

            for _ in range(40):
                model, h, lora = asyncio.run(one("t:0"))
                if "x-deltaserve-routing-candidates" in h \
                        and h.get("x-deltaserve-routing") == "routed":
                    break
            check("response headers: confidence, and the candidates when several "
                  "were close",
                  "x-deltaserve-routing-confidence" in h
                  and set(h["x-deltaserve-routing-candidates"].split(",")[:2])
                  == {"a0", "a1"})
            check("the request went on as a base-model request; the adapter is "
                  "on the engine request and in the header",
                  model == "base" and lora == h["x-deltaserve-adapter"]
                  and lora in ("a0", "a1"))
            for _ in range(40):
                model, h, lora = asyncio.run(one("u:"))
                if h.get("x-deltaserve-routing") == "fallback":
                    break
            check("response headers on a low-confidence fallback: base model, "
                  "reason, message, confidence",
                  model == "base" and lora is None
                  and h.get("x-deltaserve-routing-reason") == "low-confidence"
                  and "not a confident match" in h.get("x-deltaserve-routing-message", "")
                  and "x-deltaserve-routing-confidence" in h)
        finally:
            stop.set()
            th.join(5)
        rows = [json.loads(x) for x in open(log)]
        check("every decision is logged for calibration, without the request text",
              len(rows) > 50 and {"outcome", "adapter", "confidence", "margin",
                                   "candidates", "reason", "score", "t"}
              == set(rows[0]) and any(r["outcome"] == "fallback" for r in rows))


def test_config():
    green("config:")
    from deltaserve.config import AdapterRoutingConfig

    c = AdapterRoutingConfig()
    check("defaults: keep 90%, band 0.05, 64 reference samples, min pool 4",
          (c.route_keep_percent, c.route_close_band, c.route_reference_samples,
           c.route_min_pool) == (90.0, 0.05, 64, 4))
    bad = False
    try:
        AdapterRoutingConfig(route_keep_percent=120)
    except Exception:  # noqa: BLE001
        bad = True
    check("keep percent outside [0, 100] is rejected", bad)


def main():
    green("=== adapter routing: the decision ===")
    test_reference_samples()
    test_threshold()
    test_three_outcomes()
    test_degraded_modes()
    test_reaches_the_client()
    test_config()
    print()
    if _failed:
        print(f"{_RED}{_failed} failed{_RESET}, {_passed} passed")
        sys.exit(1)
    green(f"all {_passed} checks passed")


if __name__ == "__main__":
    main()
