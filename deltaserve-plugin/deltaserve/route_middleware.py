# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""API-server middleware for auto-routed requests (``model: "<route name>"``).

Lives outside ``deltaserve.adapter_routing`` on purpose: a DeltaServe server
with adapter routing OFF still answers such a request — with the base model and
a header saying routing is not enabled — without loading any routing component.
With routing on, ``deltaserve.adapter_routing.middleware.install`` plugs in the
e5 tokenizer.

The adapter is chosen INSIDE the engine, by a step the scheduler issues (see
``deltaserve.adapter_routing.engine_router``); the API server only prepares the
request and reports the outcome:

1. A POST to a generation endpoint whose JSON ``model`` equals the route name
   is taken. Its text is tokenized for the route encoder (CPU, well under a
   millisecond), the ``model`` field is rewritten to the base model, and the
   request goes into the engine at once — its ``arrival_time`` is its real
   arrival. The encoder tokens travel with it, in the engine request's
   ``trace_headers`` (``ROUTE_HEADER``; the plugin's wrapper on
   ``InputProcessor.process_inputs`` puts them there from ``ROUTE_PAYLOAD``).
2. The engine's decision comes back on the request's outputs, again in
   ``trace_headers`` (``DECISION_HEADER``); the plugin's wrapper on
   ``OutputProcessor.process_outputs`` hands it to ``DECISIONS``.
3. The response headers say what happened::

    x-deltaserve-routing: routed | fallback
    x-deltaserve-adapter: <adapter name>          (routed only)
    x-deltaserve-routing-score: <best cosine>     (when one was computed)
    x-deltaserve-routing-confidence: <pool-relative confidence>   (when computed)
    x-deltaserve-routing-candidates: <a,b,...>    (several adapters were close)
    x-deltaserve-routing-reason: <why>            (fallback only)
    x-deltaserve-routing-message: <the same, as a sentence>   (fallback only)

   They reach the client with the first token, streaming or not: the response
   start is held until the decision is known. The response body's ``model``
   field is the base model's name — the adapter is reported in the headers
   only.

A request naming an adapter or the base model passes through byte-for-byte,
without headers. Pure ASGI (no framework import), so it is testable alone.
"""

import asyncio
import contextvars
import json
import time
import uuid

from dataclasses import dataclass

ROUTED, FALLBACK = "routed", "fallback"
# Why a request fell back to the base model.
BELOW_THRESHOLD = "below-threshold"
NO_ADAPTERS = "no-adapters"
NO_TEXT = "no-text"
UNAVAILABLE = "router-unavailable"
TIMEOUT = "router-timeout"
LOW_CONFIDENCE = "low-confidence"
DISABLED = "adapter-routing-disabled"
# Request → engine: the route encoder's token ids; engine → API server: the
# decision. Both ride in the engine request's ``trace_headers``.
ROUTE_HEADER = "x-deltaserve-route"
DECISION_HEADER = "x-deltaserve-routed"
MESSAGES = {
    BELOW_THRESHOLD: "no adapter matched this request closely enough; "
                     "served by the base model",
    LOW_CONFIDENCE: "the best adapter is not a confident match for this "
                    "request; served by the base model",
    NO_ADAPTERS: "no routable adapter is available; served by the base model",
    NO_TEXT: "the request has no text to route on; served by the base model",
    UNAVAILABLE: "the adapter router is unavailable; served by the base model",
    TIMEOUT: "the adapter router did not answer in time; "
             "served by the base model",
    DISABLED: "adapter routing is not enabled on this server; "
              "served by the base model",
}


@dataclass
class RouteDecision:
    adapter: str | None            # None = serve with the base model
    score: float | None = None     # best cosine, when one was computed
    reason: str = ""               # set on a fallback
    confidence: float | None = None   # the pool-relative confidence, when computed
    candidates: tuple = ()         # the adapters that scored close to the best

    @property
    def outcome(self) -> str:
        return ROUTED if self.adapter else FALLBACK


ROUTED_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/responses")


def _content_text(content) -> str:
    """A chat message's text: the string, or the text parts of a part list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content
                         if isinstance(p, dict)
                         and p.get("type") in ("text", "input_text"))
    return ""


def extract_text(body: dict) -> str | None:
    """The text a request is routed on.

    - chat: the last user message (the task being asked now);
    - completions: the prompt string (the first one of a list);
    - responses: the ``input`` string, or the last user message of its list.

    None when there is no text to encode (e.g. a token-id prompt)."""
    messages = body.get("messages")
    if messages is None and "input" in body:
        if isinstance(body["input"], str):
            return body["input"] or None
        messages = body["input"]
    if isinstance(messages, list):
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                text = _content_text(m.get("content"))
                if text.strip():
                    return text
        return None
    prompt = body.get("prompt")
    if isinstance(prompt, str):
        return prompt or None
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], str):
        return prompt[0] or None
    return None


# When the request being handled reached the API server (time.time()); read by
# the plugin's wrapper on vLLM's InputProcessor.process_inputs. A contextvar:
# the middleware and the endpoint run in the same task.
REQUEST_ARRIVAL: contextvars.ContextVar = contextvars.ContextVar(
    "deltaserve_request_arrival", default=None)
# The ROUTE_HEADER value of the request being handled (None: not routed).
ROUTE_PAYLOAD: contextvars.ContextVar = contextvars.ContextVar(
    "deltaserve_route_payload", default=None)


def encode_route(key: str, ids: list) -> str:
    return json.dumps({"k": key, "ids": list(ids)}, separators=(",", ":"))


def decode_route(value: str) -> tuple[str, list]:
    d = json.loads(value)
    return str(d["k"]), [int(t) for t in d["ids"]]


def encode_decision(key: str, d: RouteDecision) -> str:
    return json.dumps({"k": key, "a": d.adapter, "s": d.score, "r": d.reason,
                       "c": d.confidence, "n": list(d.candidates)},
                      separators=(",", ":"))


def decode_decision(value: str) -> tuple[str, RouteDecision]:
    d = json.loads(value)
    return str(d["k"]), RouteDecision(d.get("a"), d.get("s"), d.get("r") or "",
                                      d.get("c"), tuple(d.get("n") or ()))


class DecisionRegistry:
    """The decisions this API server process is waiting for, by route key."""

    def __init__(self) -> None:
        self.waiting: dict = {}

    def expect(self, key: str):
        fut = asyncio.get_running_loop().create_future()
        self.waiting[key] = fut
        return fut

    def forget(self, key: str) -> None:
        self.waiting.pop(key, None)

    def note_headers(self, trace_headers) -> None:
        """An engine output's ``trace_headers``: resolve the decision it
        carries, if this process waits for it. Never raises."""
        value = trace_headers.get(DECISION_HEADER) if trace_headers else None
        if value is None:
            return
        try:
            key, decision = decode_decision(value)
        except (ValueError, KeyError, TypeError):
            return
        fut = self.waiting.pop(key, None)
        if fut is not None and not fut.done():
            fut.get_loop().call_soon_threadsafe(_resolve, fut, decision)


def _resolve(fut, decision) -> None:
    if not fut.done():
        fut.set_result(decision)


DECISIONS = DecisionRegistry()

# A streamed response starts when its first token exists; its headers wait
# for the decision, which the engine makes before the prefill. This only
# bounds a request whose engine never answers at all.
DECISION_WAIT_S = 120.0


def decision_headers(decision: RouteDecision) -> list:
    extra = [(b"x-deltaserve-routing", decision.outcome.encode())]
    if decision.adapter:
        extra.append((b"x-deltaserve-adapter", decision.adapter.encode()))
    if decision.score is not None:
        extra.append((b"x-deltaserve-routing-score",
                      f"{decision.score:.4f}".encode()))
    if decision.confidence is not None:
        extra.append((b"x-deltaserve-routing-confidence",
                      f"{decision.confidence:.3f}".encode()))
    if len(decision.candidates) > 1:
        extra.append((b"x-deltaserve-routing-candidates",
                      ",".join(decision.candidates).encode()))
    if decision.reason:
        extra.append((b"x-deltaserve-routing-reason", decision.reason.encode()))
        if decision.reason in MESSAGES:
            extra.append((b"x-deltaserve-routing-message",
                          MESSAGES[decision.reason].encode()))
    return extra


class AdapterRouteMiddleware:
    """``tokenize(text) -> list[int]``: the route encoder's token ids for a
    request text (prefix included, truncated). None = adapter routing is off
    on this server: a routed request is served by the base model and told so."""

    def __init__(self, app, tokenize, route_model_name: str, base_model_name,
                 paths=ROUTED_PATHS, before_model=None, registry=None) -> None:
        self.app, self.tokenize = app, tokenize
        # Called with the model name every generation request ends up with:
        # lets the server drop what it cached about an adapter that has since
        # changed. Must not raise.
        self.before_model = before_model
        self.route_model_name = route_model_name
        # str, or a callable evaluated per request (the app's state is only
        # complete once the server has started).
        self._base = base_model_name
        self.paths = tuple(paths)
        self.registry = registry if registry is not None else DECISIONS

    def _base_name(self) -> str:
        return self._base() if callable(self._base) else self._base

    def _prepare(self, text):
        """(decision, route key): a decision when the request does not go
        through the engine's router at all, else the key its decision will
        come back under (and ``ROUTE_PAYLOAD`` set for the engine)."""
        if self.tokenize is None:
            return RouteDecision(None, None, DISABLED), None
        if not text or not text.strip():
            return RouteDecision(None, None, NO_TEXT), None
        try:
            ids = self.tokenize(text)
        except Exception:  # noqa: BLE001 — routing must never fail a request
            return RouteDecision(None, None, UNAVAILABLE), None
        key = uuid.uuid4().hex
        ROUTE_PAYLOAD.set(encode_route(key, ids))
        return None, key

    async def __call__(self, scope, receive, send):
        if (scope["type"] != "http" or scope.get("method") != "POST"
                or scope.get("path") not in self.paths):
            return await self.app(scope, receive, send)
        REQUEST_ARRIVAL.set(time.time())
        ROUTE_PAYLOAD.set(None)

        chunks, first = [], None
        while True:
            msg = await receive()
            if msg["type"] != "http.request":
                first = msg                     # client went away while sending
                break
            chunks.append(msg.get("body", b""))
            if not msg.get("more_body", False):
                break
        body = b"".join(chunks)

        routed, decision, key = False, None, None
        if first is None:
            try:
                data = json.loads(body)
            except (ValueError, UnicodeDecodeError):
                data = None
            if isinstance(data, dict) and data.get("model") == self.route_model_name:
                routed = True
                decision, key = self._prepare(extract_text(data))
                data["model"] = self._base_name()
                body = json.dumps(data).encode()
                headers = [(k, v) for k, v in scope.get("headers", [])
                           if k.lower() != b"content-length"]
                headers.append((b"content-length", str(len(body)).encode()))
                scope = {**scope, "headers": headers}
            if self.before_model is not None and isinstance(data, dict) \
                    and isinstance(data.get("model"), str):
                try:
                    self.before_model(data["model"])
                except Exception:  # noqa: BLE001 — never fail a request on this
                    pass

        sent = False

        async def replay():
            nonlocal sent
            if first is not None:
                return first
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        if not routed:
            return await self.app(scope, replay, send)

        fut = self.registry.expect(key) if key is not None else None

        async def send_with_headers(message):
            nonlocal decision
            if message["type"] == "http.response.start":
                if decision is None and fut is not None:
                    if fut.done():
                        decision = fut.result()
                    elif int(message.get("status", 200)) < 400:
                        # A stream starts before its first token: wait for
                        # the engine's decision (it precedes the prefill).
                        try:
                            decision = await asyncio.wait_for(
                                asyncio.shield(fut), DECISION_WAIT_S)
                        except asyncio.TimeoutError:
                            decision = RouteDecision(None, None, TIMEOUT)
                    # an error response that never reached the engine's
                    # router carries no routing headers
                if decision is not None:
                    message = {**message, "headers": [
                        *message.get("headers", []), *decision_headers(decision)]}
            await send(message)

        try:
            return await self.app(scope, replay, send_with_headers)
        finally:
            if key is not None:
                self.registry.forget(key)


def install_middleware(app, tokenize, route_model_name: str,
                       before_model=None) -> None:
    """Add the middleware to vLLM's FastAPI app (before the server starts)."""

    def base_model_name() -> str:
        args = app.state.args
        names = getattr(args, "served_model_name", None)
        return names[0] if names else args.model

    app.add_middleware(AdapterRouteMiddleware, tokenize=tokenize,
                       route_model_name=route_model_name,
                       base_model_name=base_model_name, before_model=before_model)
