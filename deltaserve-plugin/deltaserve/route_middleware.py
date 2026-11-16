# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""API-server middleware for auto-routed requests: turn ``model: "<route
name>"`` into an adapter, or into the base model.

Lives outside ``deltaserve.adapter_routing`` on purpose: a DeltaServe server
with adapter routing OFF still answers such a request — with the base model and
a header saying routing is not enabled — without loading any routing component
(no encoder, no index, no service). With routing on,
``deltaserve.adapter_routing.middleware.install`` plugs the router in.

Only a request that asks for it is touched: a POST to a generation endpoint
whose JSON ``model`` equals the configured route name (``auto`` by default).
Its text is sent to the router; the ``model`` field is rewritten to the chosen
adapter, or to the base model when the router falls back. From there on it is
an ordinary named-adapter request — vLLM builds the ``LoRARequest`` itself.

Every routed request is answered with headers saying what happened::

    x-deltaserve-routing: routed | fallback
    x-deltaserve-adapter: <adapter name>          (routed only)
    x-deltaserve-routing-score: <best cosine>     (when one was computed)
    x-deltaserve-routing-confidence: <pool-relative confidence>   (when computed)
    x-deltaserve-routing-candidates: <a,b,...>    (several adapters were close)
    x-deltaserve-routing-reason: <why>            (fallback only)
    x-deltaserve-routing-message: <the same, as a sentence>   (fallback only)

and the response's own ``model`` field carries the adapter (or base) name.
Headers reach the client before the first token, streaming or not.

A request naming an adapter or the base model passes through byte-for-byte,
without headers. Pure ASGI (no framework import), so it is testable alone.
"""

import asyncio
import json

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


async def routing_disabled(text) -> RouteDecision:
    """The route function of a server whose adapter routing is off."""
    return RouteDecision(None, None, DISABLED)


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


class AdapterRouteMiddleware:
    """``route_fn(text) -> RouteDecision`` may be sync (run in a thread) or async."""

    def __init__(self, app, route_fn, route_model_name: str, base_model_name,
                 paths=ROUTED_PATHS, before_model=None) -> None:
        self.app, self.route_fn = app, route_fn
        # Called with the model name every generation request ends up with
        # (after routing): lets the server drop what it cached about an
        # adapter that has since changed. Must not raise.
        self.before_model = before_model
        self.route_model_name = route_model_name
        # str, or a callable evaluated per request (the app's state is only
        # complete once the server has started).
        self._base = base_model_name
        self.paths = tuple(paths)

    def _base_name(self) -> str:
        return self._base() if callable(self._base) else self._base

    async def _decide(self, text) -> RouteDecision:
        try:
            if asyncio.iscoroutinefunction(self.route_fn):
                return await self.route_fn(text)
            return await asyncio.to_thread(self.route_fn, text)
        except Exception:  # noqa: BLE001 — routing must never fail a request
            return RouteDecision(None, None, UNAVAILABLE)

    async def __call__(self, scope, receive, send):
        if (scope["type"] != "http" or scope.get("method") != "POST"
                or scope.get("path") not in self.paths):
            return await self.app(scope, receive, send)

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

        decision = None
        if first is None:
            try:
                data = json.loads(body)
            except (ValueError, UnicodeDecodeError):
                data = None
            if isinstance(data, dict) and data.get("model") == self.route_model_name:
                decision = await self._decide(extract_text(data))
                data["model"] = decision.adapter or self._base_name()
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

        if decision is None:
            return await self.app(scope, replay, send)

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

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                message = {**message,
                           "headers": [*message.get("headers", []), *extra]}
            await send(message)

        return await self.app(scope, replay, send_with_headers)


def install_middleware(app, route_fn, route_model_name: str,
                       before_model=None) -> None:
    """Add the middleware to vLLM's FastAPI app (before the server starts)."""

    def base_model_name() -> str:
        args = app.state.args
        names = getattr(args, "served_model_name", None)
        return names[0] if names else args.model

    app.add_middleware(AdapterRouteMiddleware, route_fn=route_fn,
                       route_model_name=route_model_name,
                       base_model_name=base_model_name, before_model=before_model)
