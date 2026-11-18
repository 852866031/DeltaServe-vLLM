# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Plug adapter routing into the API server's auto-route middleware.

The middleware itself (``deltaserve.route_middleware``) is routing-agnostic and
also serves the routing-off case; this module is the routing-ON wiring: the
route encoder's tokenizer (the encode itself runs inside the engine, on token
ids — see ``engine_router``), the pool's LoRA resolver, and the invalidation
of what vLLM cached about pool adapters.
"""

from deltaserve.route_middleware import (  # noqa: F401 — re-exported
    ROUTED_PATHS,
    AdapterRouteMiddleware,
    extract_text,
    install_middleware,
)


class RouteTokenizer:
    """``text -> token ids`` exactly as ``SentenceEncoder`` feeds its model
    (query prefix, truncation, special tokens). Loaded on first use."""

    def __init__(self, model: str, prefix: str, max_tokens: int) -> None:
        self.model, self.prefix, self.max_tokens = model, prefix, int(max_tokens)
        self._tok = None

    def __call__(self, text: str) -> list:
        if self._tok is None:
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.model)
        return self._tok(self.prefix + text, truncation=True,
                         max_length=self.max_tokens)["input_ids"]


def install(app, cfg) -> None:
    """Called by the plugin, only with adapter routing on."""
    from deltaserve.adapter_routing.resolver import install_resolver, make_invalidator

    resolver = install_resolver(cfg)
    tokenize = RouteTokenizer(cfg.encoder, cfg.encoder_query_prefix,
                              cfg.encoder_max_tokens)
    install_middleware(app, tokenize, cfg.route_model_name,
                       before_model=make_invalidator(app, resolver))
