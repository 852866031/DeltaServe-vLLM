# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Make pool adapters known to vLLM's API servers, through vLLM's own
``LoRAResolver`` hook.

An API server only accepts a ``model`` name it has in its adapter table. When a
name is unknown and runtime LoRA updating is on, vLLM asks the registered
resolvers for it; a hit is cached in that API server and loaded into the
engine. ``PoolLoRAResolver`` answers from the adapter pool's registry file
(``pool.json`` in the index directory), so an adapter that joined the pool
after launch — a finetuning job that reached ``ready`` — is servable by name,
and therefore by the route middleware, without restarting anything.

**LoRA ids.** vLLM identifies an adapter by ``lora_int_id`` everywhere in the
engine, and after a resolver returns it overwrites that id with a counter that
is private to each API server process. With several API servers
(``api_server_count > 1``) two processes would hand the engine different
adapters under one id, or one adapter under two. The pool therefore assigns
each adapter a stable id (``pool.LORA_ID_BASE`` + n, recorded in the registry),
and the request the resolver returns keeps it: ``PoolLoRARequest`` ignores
assignments to ``lora_int_id``. On the wire it is a plain ``LoRARequest``.
"""

import os

from vllm.lora.request import LoRARequest
from vllm.lora.resolver import LoRAResolver, LoRAResolverRegistry

from deltaserve import dprint
from deltaserve.adapter_routing.pool import REGISTRY_FILE, read_registry

RESOLVER_NAME = "deltaserve-adapter-pool"


class PoolLoRARequest(LoRARequest):
    """A ``LoRARequest`` whose id is the pool's and cannot be reassigned."""

    def __setattr__(self, name, value):
        if name == "lora_int_id":
            return
        super().__setattr__(name, value)


class PoolLoRAResolver(LoRAResolver):
    def __init__(self, directory: str) -> None:
        self.directory = directory
        self._mtime = None
        self._adapters: dict[str, dict] = {}

    def _refresh(self) -> None:
        try:
            mtime = os.stat(os.path.join(self.directory, REGISTRY_FILE)).st_mtime_ns
        except OSError:
            self._mtime, self._adapters = None, {}
            return
        if mtime == self._mtime:
            return
        reg = read_registry(self.directory)
        if reg is None:
            return            # being rewritten: keep what we have
        self._mtime = mtime
        self._adapters = {a["name"]: a for a in reg["adapters"]}

    def lookup(self, lora_name: str) -> PoolLoRARequest | None:
        self._refresh()
        a = self._adapters.get(lora_name)
        if a is None or not a.get("lora_int_id") or not a.get("path"):
            return None
        if not os.path.isdir(a["path"]):
            dprint(f"[adapter-pool] '{lora_name}' is in the pool but its "
                   f"weights are missing: {a['path']}")
            return None
        return PoolLoRARequest(lora_name=lora_name,
                               lora_int_id=int(a["lora_int_id"]),
                               lora_path=a["path"])

    async def resolve_lora(self, base_model_name: str,
                           lora_name: str) -> LoRARequest | None:
        return self.lookup(lora_name)


def make_invalidator(app, resolver: PoolLoRAResolver):
    """vLLM's API server caches a resolved adapter (name → LoRARequest) for
    good. This returns a function for the route middleware to call with every
    request's model name: when that name is a pool adapter the server has
    cached and the pool has since removed it or replaced it with a new version
    (another path / LoRA id), the cached entry is dropped, so vLLM resolves it
    again — and gets the new version, or "model does not exist"."""

    def before_model(name: str) -> None:
        models = getattr(app.state, "openai_serving_models", None)
        cached = None if models is None else models.lora_requests.get(name)
        if not isinstance(cached, PoolLoRARequest):
            return                       # not ours (or not cached): untouched
        current = resolver.lookup(name)
        if current is None or current.lora_int_id != cached.lora_int_id \
                or current.lora_path != cached.lora_path:
            models.lora_requests.pop(name, None)
            dprint(f"[adapter-pool] '{name}' changed in the pool: dropped from "
                   f"this API server's adapter table")

    return before_model


def install_resolver(cfg) -> PoolLoRAResolver:
    """Register the pool resolver in this API server process and switch on
    vLLM's resolve-unknown-adapter path. Must run before vLLM builds its model
    registry (the plugin calls it while the app is being built)."""
    from deltaserve.adapter_routing.stages import index_dir

    resolver = PoolLoRAResolver(index_dir(cfg))
    LoRAResolverRegistry.register_resolver(RESOLVER_NAME, resolver)
    if os.environ.get("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "0").strip().lower() \
            not in ("1", "true"):
        # vLLM consults resolvers only with this on. Set here, after vLLM has
        # mounted its routers, it does not expose the /v1/load_lora_adapter
        # endpoints — only the resolve-by-name path.
        os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"
        dprint("[adapter-pool] enabled vLLM runtime LoRA resolution for pool "
               "adapters")
    return resolver
