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
import threading

from deltaserve import dprint
from deltaserve.adapter_routing.index import CentroidIndex
from deltaserve.adapter_routing.pool import AdapterPool
from deltaserve.adapter_routing.stages import (
    INDEX_FILE,
    IndexingStage,
    SharedIndexer,
    index_dir,
)
from deltaserve.ft_job import FinetuneJobSpec, FtJobManager


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


def _router_service_main(routing_kwargs: dict) -> None:
    from deltaserve.adapter_routing.service import (
        apply_router_placement,
        serve_config,
    )
    from deltaserve.config import AdapterRoutingConfig

    cfg = AdapterRoutingConfig(**routing_kwargs)
    device = apply_router_placement(cfg)     # before anything touches CUDA

    try:
        from deltaserve.backward_process import arm_parent_death_signal

        arm_parent_death_signal("router")
    except Exception:  # noqa: BLE001 — never block the service on this
        pass
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import signal
    import sys

    # SIGTERM (engine shutdown, or the whole server being stopped) → unwind
    # through serve()'s cleanup, which removes the socket file.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    serve_config(cfg, device)


class RouterServiceProcess:
    """The router service as a child of the engine."""

    def __init__(self, cfg) -> None:
        from dataclasses import asdict

        from deltaserve.adapter_routing.service import socket_path

        self._args = (asdict(cfg),)
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
                 start_service: bool = True, start_driver: bool = True) -> None:
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
            capacity=cfg.max_active_adapters, gpu_slots=gpu_slots,
            protected=cfg.pool_protected_slots,
            half_life_s=cfg.pool_hit_half_life_s, directory=self.dir,
            when_full=cfg.when_full, grace_s=cfg.retire_grace_s,
            min_pool=cfg.route_min_pool,
            min_self_recall=cfg.admission_min_self_recall,
            max_stolen=cfg.admission_max_stolen)
        restored = self.pool.restore(self.dir, centroids=self._centroids())
        dprint(f"[adapter-routing] pool: {restored} adapters restored, "
               f"{gpu_slots} GPU slots, index dir {self.dir}")
        self.service = None
        if start_service:
            self.service = RouterServiceProcess(cfg)
            self.service.start()

        # One encoder process for every job of this server.
        self.shared_indexer = SharedIndexer(
            lambda: IndexingStage.make_executors(cfg))
        # Jobs that do not train — an existing adapter being indexed and
        # published. They need nothing of the finetuning machinery or of the
        # engine loop, so they run on their own driver thread, next to a
        # training job, one at a time.
        self.jobs = FtJobManager(adapter_routing=True)
        self._jobs_lock = threading.RLock()
        self._external: dict[str, dict] = {}
        self._stop = threading.Event()
        self._driver = None
        for spec in cfg.adapters or []:
            try:
                self.submit_adapter(**spec)
            except Exception as e:  # noqa: BLE001 — one bad entry must not stop the server
                dprint(f"[adapter-routing] adapter {spec.get('name') or spec.get('path')}"
                       f" not added: {e}")
        if start_driver:
            self._driver = threading.Thread(target=self._drive, daemon=True,
                                            name="deltaserve-adapter-jobs")
            self._driver.start()

    # ------------------------------------------------- jobs that do not train
    @property
    def adapter_routing_config(self):
        return self.cfg

    def submit_adapter(self, path: str, name: str | None = None,
                       data_path: str | None = None,
                       centroid_path: str | None = None,
                       prewarm: bool = False) -> dict:
        """Add an existing adapter to the pool through a job with stages
        ``[init, indexing, admission, ready]`` — without ``indexing`` when its
        centroid is given (``centroid_path``, a .npy vector) or there is no dataset (then it
        is servable by name only). Nothing is trained, exported or copied, and
        the weights are loaded only when a request first needs them (unless
        ``prewarm``)."""
        stages = ["init", "admission", "ready"]
        if data_path and not centroid_path:
            stages.insert(1, "indexing")
        spec = FinetuneJobSpec(name=name or None, stages=tuple(stages),
                               auto_routing=True, adapter_path=path,
                               data_path=data_path)
        with self._jobs_lock:
            if name and name in self.pool \
                    and self.pool.entries[name].path != path:
                raise ValueError(f"adapter '{name}' is already in the pool with "
                                 f"another path")
            job = self.jobs.submit(spec, replace_finished=True)
            self._external[job.name] = {"centroid_path": centroid_path,
                                        "prewarm": bool(prewarm)}
            return job.describe()

    def tick_jobs(self) -> None:
        """Advance the non-training jobs (the driver thread calls this)."""
        with self._jobs_lock:
            self.jobs.tick(self)
            idle = self.jobs.current is None
        if idle and self.shared_indexer.idle_and_loaded:
            self.shared_indexer.unload()      # give the encoder's memory back
        # The router prefers, among close candidates, adapters that are in a
        # GPU slot / used most: let it see the pool's view now and then.
        self._ticks = getattr(self, "_ticks", 0) + 1
        if getattr(self, "_usage_dirty", False) and self._ticks % 25 == 0:
            self._usage_dirty = False
            self.pool.save(self.dir)

    def _drive(self) -> None:
        while not self._stop.wait(0.2):
            try:
                self.tick_jobs()
            except Exception as e:  # noqa: BLE001 — keep the driver alive
                dprint(f"[adapter-routing] job driver error: {e}")

    def admit_job(self, job) -> None:
        """'admission' of a job that did not train: its adapter IS the one on
        disk — nothing is exported or copied."""
        import numpy as np

        extra = self._external.get(job.name, {})
        job.saved_path = job.spec.adapter_path
        centroid = None
        if extra.get("centroid_path"):
            centroid = np.load(extra["centroid_path"]).reshape(-1)
        self.publish(job, load=bool(extra.get("prewarm")), centroid=centroid)

    def on_job_ready(self, job) -> None:
        if job.admission is None:        # a job listed without an admission stage
            self.admit_job(job)

    def adapter_op(self, name: str, op: str) -> dict:
        """Operator actions on a pool adapter (the /adapters routes)."""
        pool = self.pool
        if op == "retire":
            pool.retire(name)
        elif op == "reactivate":
            return {"admission": pool.reactivate(name)}
        elif op == "pin":
            pool.pin(name, True)
        elif op == "unpin":
            pool.pin(name, False)
        elif op == "remove":
            pool.remove(name)
            return {"removed": name}
        else:
            raise ValueError(f"unknown adapter operation '{op}'")
        return {"adapter": pool.entries[name].describe(pool._clock(),
                                                       pool.half_life_s)}

    def _centroids(self) -> CentroidIndex | None:
        """Every centroid built so far (the indexing stage's file)."""
        path = os.path.join(self.dir, INDEX_FILE)
        return CentroidIndex.load(path) if os.path.exists(path) else None

    def adapter_dir(self, name: str) -> str:
        """Default location of a job's saved adapter."""
        return os.path.join(self.dir, "adapters", name)

    def publish(self, job, load: bool = True, centroid=None) -> None:
        """A finished auto-routable job's adapter goes through the pool's
        admission (``AdapterPool.admit``). ``load=False`` only registers it
        (weights loaded on first use)."""
        if not job.saved_path:
            raise RuntimeError(f"{job.name}: the adapter was not saved")
        refs = None
        if centroid is None:
            centroids = self._centroids()
            if centroids is not None and job.name in centroids:
                centroid = centroids.centroid_of(job.name)
                refs = centroids.refs.get(job.name)
        job.admission = self.pool.admit(
            job.name, job.saved_path, centroid=centroid, refs=refs,
            job=job.name, load=load, prewarm=load)

    def note_request(self, request) -> None:
        """Count a request that uses a pool adapter (drives GPU-slot retention)."""
        lora = getattr(request, "lora_request", None)
        name = getattr(lora, "lora_name", None)
        if name is None or name not in self.pool:
            return
        try:
            self.pool.note_hit(name)
            self._usage_dirty = True
        except Exception as e:  # noqa: BLE001 — bookkeeping must not fail serving
            dprint(f"[adapter-routing] hit on {name} not counted: {e}")

    def status(self) -> dict:
        with self._jobs_lock:
            jobs = self.jobs.status()
        return {**self.pool.status(), "jobs": jobs,
                "router_service_alive": bool(self.service and self.service.is_alive()),
                "index_dir": self.dir}

    def shutdown(self) -> None:
        self._stop.set()
        if self._driver is not None:
            self._driver.join(timeout=3.0)
        self.shared_indexer.stop()
        if self.service is not None:
            self.service.stop()
