# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Adapter routing inside the EngineCore process: the pool, its link to vLLM's
LoRA manager, the router service, and publishing a finished job's adapter.

Created by the plugin right after the engine is built, only with adapter
routing on. One object ties the loop together::

    job reaches 'ready' → trained adapter saved → publish(): centroid + path
    into the pool → registry + routable index written → the API servers'
    resolver and the router service pick them up → model="auto" is routed.

``VllmLoRABackend`` maps the pool's tier moves onto the engine's own LoRA
calls. Loading and unloading are explicit; which adapter sits in a GPU slot is
still decided by vLLM (it activates an adapter when a batch needs it and evicts
least-recently-used), so the pool's ``gpu`` tier is its intent, not yet a fact.
"""

import multiprocessing as mp
import os

from deltaserve import dprint
from deltaserve.adapter_routing.index import CentroidIndex
from deltaserve.adapter_routing.pool import AdapterPool
from deltaserve.adapter_routing.stages import INDEX_FILE, index_dir


class VllmLoRABackend:
    """The adapter pool's backend over ``EngineCore.add_lora`` / ``remove_lora``
    (both reach every TP rank)."""

    def __init__(self, engine) -> None:
        self.engine = engine

    def load(self, e) -> None:
        from vllm.lora.request import LoRARequest

        self.engine.add_lora(LoRARequest(lora_name=e.name,
                                         lora_int_id=int(e.lora_int_id),
                                         lora_path=e.path))

    def unload(self, e) -> None:
        self.engine.remove_lora(int(e.lora_int_id))

    def activate(self, e) -> None:
        """vLLM puts an adapter in a GPU slot when a batch first uses it."""

    def deactivate(self, e) -> None:
        """vLLM frees a slot by its own least-recently-used eviction."""


def _router_service_main(routing_kwargs: dict, device: str) -> None:
    from deltaserve.adapter_routing.service import build_router, serve
    from deltaserve.config import AdapterRoutingConfig

    try:
        from deltaserve.backward_process import arm_parent_death_signal

        arm_parent_death_signal("router")
    except Exception:  # noqa: BLE001 — never block the service on this
        pass
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg = AdapterRoutingConfig(**routing_kwargs)
    router, path = build_router(cfg, device)
    router.encoder.load()
    router.refresh(force=True)
    dprint(f"[router] serving on {path} | encoder={cfg.encoder} on {device} | "
           f"{len(router.index)} adapters")
    serve(router, path)


class RouterServiceProcess:
    """The router service as a child of the engine."""

    def __init__(self, cfg, device: str) -> None:
        from dataclasses import asdict

        from deltaserve.adapter_routing.service import socket_path

        self._args = (asdict(cfg), device)
        self._socket = socket_path(index_dir(cfg))
        self._proc = None

    def start(self) -> None:
        ctx = mp.get_context("spawn")
        self._proc = ctx.Process(target=_router_service_main, args=self._args,
                                 name="deltaserve-router", daemon=True)
        cur = mp.current_process()
        was = cur._config.get("daemon", False)   # the engine may be daemonic
        cur._config["daemon"] = False
        try:
            self._proc.start()
        finally:
            cur._config["daemon"] = was

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        proc.terminate()
        proc.join(timeout=3.0)
        if proc.is_alive():
            proc.kill()
        try:                        # a terminated service cannot remove it
            os.unlink(self._socket)
        except OSError:
            pass


class AdapterRoutingEngine:
    def __init__(self, engine, vllm_config, backend=None,
                 start_service: bool = True) -> None:
        cfg = vllm_config.adapter_routing_config
        lora = vllm_config.lora_config
        if lora is None:
            raise ValueError("adapter routing needs LoRA serving (enable_lora)")
        self.cfg, self.engine = cfg, engine
        self.dir = index_dir(cfg)
        os.makedirs(self.dir, exist_ok=True)
        # The finetuning job's adapter keeps one pinned slot for itself.
        ft_slot = 1 if vllm_config.finetune_config.enable_finetuning else 0
        gpu_slots = max(1, int(lora.max_loras) - ft_slot)
        self.pool = AdapterPool(
            backend if backend is not None else VllmLoRABackend(engine),
            CentroidIndex(cfg.encoder, cfg.encoder_query_prefix,
                          cfg.encoder_max_tokens),
            capacity=cfg.max_pool_adapters, gpu_slots=gpu_slots,
            protected=cfg.pool_protected_slots,
            half_life_s=cfg.pool_hit_half_life_s, directory=self.dir)
        restored = self.pool.restore(self.dir, centroids=self._centroids())
        dprint(f"[adapter-routing] pool: {restored} adapters restored, "
               f"{gpu_slots} GPU slots, index dir {self.dir}")
        self.service = None
        if start_service:
            self.service = RouterServiceProcess(cfg, cfg.router_device)
            self.service.start()

    def _centroids(self) -> CentroidIndex | None:
        """Every centroid built so far (the indexing stage's file)."""
        path = os.path.join(self.dir, INDEX_FILE)
        return CentroidIndex.load(path) if os.path.exists(path) else None

    def adapter_dir(self, name: str) -> str:
        """Default location of a job's saved adapter."""
        return os.path.join(self.dir, "adapters", name)

    def publish(self, job) -> None:
        """A finished auto-routable job's adapter joins the pool."""
        if not job.saved_path:
            raise RuntimeError(f"{job.name}: the trained adapter was not saved")
        centroids = self._centroids()
        centroid = None
        if centroids is not None and job.name in centroids:
            centroid = centroids._matrix[centroids.names.index(job.name)].copy()
        else:
            dprint(f"[adapter-routing] {job.name} has no centroid (no 'indexing' "
                   f"stage?): servable by name, not auto-routable")
        self.pool.register(job.name, job.saved_path, centroid=centroid,
                           job=job.name)

    def note_request(self, request) -> None:
        """Count a request that uses a pool adapter (drives GPU-slot retention)."""
        lora = getattr(request, "lora_request", None)
        name = getattr(lora, "lora_name", None)
        if name is None or name not in self.pool:
            return
        try:
            self.pool.note_hit(name)
        except Exception as e:  # noqa: BLE001 — bookkeeping must not fail serving
            dprint(f"[adapter-routing] hit on {name} not counted: {e}")

    def status(self) -> dict:
        return {**self.pool.status(),
                "router_service_alive": bool(self.service and self.service.is_alive()),
                "index_dir": self.dir}

    def shutdown(self) -> None:
        if self.service is not None:
            self.service.stop()
