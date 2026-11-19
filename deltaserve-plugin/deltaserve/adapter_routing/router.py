# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""``Router``: the routable centroid index + the decision model → a
``RouteDecision`` for a request embedding.

The index file is re-read when it changes (an adapter was indexed, retired,
...), so decisions follow the pool without a restart. The encode itself happens
elsewhere: for a request, in the engine's encode step
(``engine_router`` → ``route_embeddings``); ``route_many`` / ``route`` take
texts and need an ``encoder`` (tests, offline use).
"""

import os
import time

from deltaserve import dprint
from deltaserve.adapter_routing.decision import RoutingModel
from deltaserve.adapter_routing.index import CentroidIndex
from deltaserve.route_middleware import (  # noqa: F401 — re-exported
    BELOW_THRESHOLD,
    FALLBACK,
    LOW_CONFIDENCE,
    NO_ADAPTERS,
    NO_TEXT,
    ROUTED,
    UNAVAILABLE,
    RouteDecision,
)


class Router:
    def __init__(self, encoder, index_file: str, threshold: float = 0.0,
                 reload_s: float = 1.0, keep_percent: float = 100.0,
                 close_band: float = 0.05, min_pool: int = 4,
                 decision_log: str | None = None, slack: float = 0.0) -> None:
        self.encoder, self.index_file = encoder, index_file
        self.threshold, self.reload_s = float(threshold), float(reload_s)
        self._model_args = (float(keep_percent), float(close_band), int(min_pool),
                            float(slack))
        self.index = CentroidIndex()
        self.model = RoutingModel(self.index, *self._model_args)
        self._mtime = None
        self._checked = 0.0
        self.requests = 0
        # Among close candidates: an adapter in a GPU slot first (the pool's
        # last published view), then the one this router has routed to most.
        self._tier: dict[str, str] = {}
        self._pool_mtime = None
        self._published: str | None = None
        self._routed: dict[str, int] = {}
        self._log = decision_log

    def _prefer(self, name: str):
        return (self._tier.get(name) == "gpu", self._routed.get(name, 0))

    def _refresh_pool_view(self) -> None:
        from deltaserve.adapter_routing.pool import REGISTRY_FILE, read_registry

        directory = os.path.dirname(self.index_file)
        try:
            mtime = os.stat(os.path.join(directory, REGISTRY_FILE)).st_mtime_ns
        except OSError:
            return
        if mtime == self._pool_mtime:
            return
        reg = read_registry(directory)
        if reg is not None:
            self._pool_mtime = mtime
            self._tier = {a["name"]: a.get("tier", "") for a in reg["adapters"]}
            self._published = (os.path.join(directory, reg["routable"])
                               if reg.get("routable") else None)

    def refresh(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._checked < self.reload_s:
            return
        self._checked = now
        self._refresh_pool_view()
        # The pool's registry names the routable index that is current (one
        # atomic switch); without a registry the given file is used directly.
        index_file = self._published or self.index_file
        try:
            mtime = (index_file, os.stat(index_file).st_mtime_ns)
        except OSError:
            if self._mtime is not None:
                self.index, self._mtime = CentroidIndex(), None
                self.model = RoutingModel(self.index, *self._model_args)
            return
        if mtime != self._mtime:
            index = CentroidIndex.load(index_file)
            model = RoutingModel(index, *self._model_args)
            # one assignment each: a batch being routed keeps the snapshot it has
            self.index, self.model, self._mtime = index, model, mtime
            dprint(f"[router] index loaded: {len(index)} adapters | "
                   f"{model.describe()}")

    def _record(self, decision: RouteDecision, choice) -> None:
        if not self._log:
            return
        import json

        try:
            with open(self._log, "a") as f:
                f.write(json.dumps({
                    "t": round(time.time(), 3), "outcome": decision.outcome,
                    "adapter": decision.adapter, "reason": decision.reason,
                    "score": decision.score, "confidence": decision.confidence,
                    "margin": None if choice is None else choice.margin,
                    "candidates": list(decision.candidates)}) + "\n")
        except OSError:
            pass

    def route_many(self, texts: list) -> list[RouteDecision]:
        """One decision per text, with ONE encoder call for all of them."""
        self.requests += len(texts)
        self.refresh()
        model = self.model
        out: list = [None] * len(texts)
        todo = []
        for i, text in enumerate(texts):
            if not text or not text.strip():
                out[i] = RouteDecision(None, None, NO_TEXT)
            elif not model.names:
                out[i] = RouteDecision(None, None, NO_ADAPTERS)
            else:
                todo.append(i)
        if todo:
            embs = self.encoder.encode([texts[i] for i in todo])
            for i, emb in zip(todo, embs):
                out[i] = self._decide(model, emb)
        return out

    def _decide(self, model, emb) -> RouteDecision:
        c = model.decide(emb, prefer=self._prefer, raw_floor=self.threshold)
        reason = "" if c.adapter else (
            LOW_CONFIDENCE if c.low_confidence else BELOW_THRESHOLD)
        if c.adapter:
            self._routed[c.adapter] = self._routed.get(c.adapter, 0) + 1
        d = RouteDecision(c.adapter, c.score, reason, c.confidence,
                          tuple(c.candidates))
        self._record(d, c)
        return d

    def route_embeddings(self, embs: list) -> list[RouteDecision]:
        """One decision per request embedding — the encode happened elsewhere
        (the engine's encode step). None = it could not be encoded: the base
        model."""
        self.requests += len(embs)
        self.refresh()
        model = self.model
        out = []
        for emb in embs:
            if emb is None:
                out.append(RouteDecision(None, None, UNAVAILABLE))
            elif not model.names:
                out.append(RouteDecision(None, None, NO_ADAPTERS))
            else:
                out.append(self._decide(model, emb))
        return out

    def route(self, text: str | None) -> RouteDecision:
        return self.route_many([text])[0]
