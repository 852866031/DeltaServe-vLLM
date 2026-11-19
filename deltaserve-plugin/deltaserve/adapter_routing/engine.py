# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Adapter routing inside the EngineCore process: the pool, the router that
chooses a routed request's adapter, indexing's placement, and publishing a
finished job's adapter.

Created by the plugin right after the engine is built, only with adapter
routing on. One object ties the loop together::

    job reaches 'ready' → trained adapter saved → publish(): centroid + path
    into the pool → registry + routable index written → the API servers'
    resolver and the engine's router pick them up → model="auto" is routed.

``VllmLoRABackend`` is the pool's link to vLLM. Nothing is loaded through it:
vLLM reads an adapter when a step first uses it, and the scheduler arranges a
background read first (cold start, ``engine_router``). Which adapter sits in a
GPU slot is decided by vLLM (least-recently-used); ``LoraCacheMirror`` is the
engine's replay of it, and the pool's own tier is its bookkeeping only.
"""

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
        self.mirror = None          # the engine's view of vLLM's adapter caches
        self.preload = None         # EngineRouter.request_preload

    def load(self, e) -> None:
        """Ask for the adapter to be read into host memory — in the background,
        by the workers, with the next step the scheduler issues (the cold-start
        preload, ``engine_router``). Never ``EngineCore.add_lora``: that blocks
        the engine loop for the read on every rank (26 ms on Qwen3-14B TP=2).
        vLLM puts it in a GPU slot when a step first uses it."""
        if self.preload is not None and e.lora_int_id:
            from vllm.lora.request import LoRARequest

            self.preload(LoRARequest(lora_name=e.name,
                                     lora_int_id=int(e.lora_int_id),
                                     lora_path=e.path))

    def unload(self, e) -> None:
        self.engine.remove_lora(int(e.lora_int_id))
        if self.mirror is not None:
            self.mirror.forget(int(e.lora_int_id))

    def activate(self, e) -> None:
        """vLLM puts an adapter in a GPU slot when a batch first uses it."""

    def deactivate(self, e) -> None:
        """vLLM frees a slot by its own least-recently-used eviction."""


def route_encoder_on_cpu(cfg) -> bool:
    """The route encoder was asked onto the CPU (then nothing indexes on a GPU)."""
    return cfg.router_gpu is None and str(cfg.router_device).lower() == "cpu"


class AdapterRoutingEngine:
    def __init__(self, engine, vllm_config, backend=None,
                 start_driver: bool = True, grant=None) -> None:
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
        if gpu_slots < 2:
            dprint(f"[adapter-routing] WARNING: lora.max_loras={lora.max_loras} leaves "
                   f"{gpu_slots} adapter per batch for routed requests"
                   + (" (the finetuning adapter takes one)" if ft_slot else "")
                   + ". Requests routed to different adapters cannot share a "
                   "batch and queue behind each other (production trace: TTFT "
                   "p95 89 ms → 1.5 s). Raise lora.max_loras to the number of "
                   "adapters in use at once"
                   + (" + 1." if ft_slot else "."))
        # The adapter of a routed request is chosen here, in the engine: the
        # scheduler issues the encode step (see engine_router).
        self.router = self._make_router(engine, vllm_config)
        if hasattr(self.pool.backend, "mirror"):
            self.pool.backend.mirror = self.router.mirror
            self.pool.backend.preload = self.router.request_preload
            # Where an adapter is, is what vLLM did with it: the pool reads
            # its tiers from the mirror instead of keeping its own.
            self.pool.mirror = self.router.mirror
        # Among close candidates the decision prefers an adapter that is in a
        # GPU slot right now (no load), then the most used one.
        self.decider._prefer = self._prefer

        # Indexing runs on its own (IndexingRunner thread); the scheduler only
        # tells it where it may run, when that changes: on the GPU — the
        # indexing process that shares the route encoder's weights — while no
        # inference request is present, on the CPU otherwise.
        from deltaserve.adapter_routing.runner import IndexingPlacement

        self.grant = grant
        self.placement = IndexingPlacement(
            cfg.indexing_gpu_after_quiet_s, grant=grant,
            idle_threads=cfg.indexing_cpu_max_threads,
            busy_threads=cfg.indexing_cpu_busy_threads)
        self.router.on_busy = self.note_inference

        def executors():
            from deltaserve.adapter_routing.index_process import (
                SharedEncoderExecutor,
                socket_path,
            )

            ex = dict(IndexingStage.make_executors(cfg))
            if grant is not None and cfg.indexing_on_gpu \
                    and not route_encoder_on_cpu(cfg):
                ex["gpu"] = SharedEncoderExecutor(socket_path(self.dir))
            return ex

        self.shared_indexer = SharedIndexer(executors, placement=self.placement)
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

    # ------------------------------------------------------ routed requests
    def _make_router(self, engine, vllm_config):
        from deltaserve.adapter_routing.engine_router import EngineRouter
        from deltaserve.adapter_routing.pool import ROUTABLE_FILE
        from deltaserve.adapter_routing.router import Router

        cfg = self.cfg
        decider = Router(
            None, os.path.join(self.dir, ROUTABLE_FILE),
            cfg.fallback_score_threshold, keep_percent=cfg.route_keep_percent,
            close_band=cfg.route_close_band, min_pool=cfg.route_min_pool,
            slack=cfg.route_confidence_slack,
            decision_log=(os.path.join(self.dir, "decisions.jsonl")
                          if cfg.route_decision_log else None))
        self.decider = decider

        def encode(batch):
            """The encode step: every rank is called, one holds the encoder."""
            replies = engine.model_executor.collective_rpc(
                "deltaserve_route_encode", args=(batch,))
            for r in replies:
                if isinstance(r, dict):
                    raise RuntimeError(r.get("error", "encode failed"))
            for r in replies:
                if r is not None:
                    return r      # (embeddings, seconds on the GPU)
            raise RuntimeError("no worker holds the route encoder")

        from deltaserve.adapter_routing.engine_router import LoraCacheMirror

        lora_cfg = vllm_config.lora_config
        pinned = ()
        if vllm_config.finetune_config.enable_finetuning:
            from deltaserve.ft_injector import FT_LORA_INT_ID

            pinned = (FT_LORA_INT_ID,)
        mirror = LoraCacheMirror(
            getattr(lora_cfg, "max_cpu_loras", None) or lora_cfg.max_loras,
            lora_cfg.max_loras, pinned)
        return EngineRouter(encode, decider.route_embeddings, self.lora_for,
                            None, max_tokens=cfg.route_encode_max_tokens,
                            on_routed=self._on_routed, mirror=mirror)

    def lora_for(self, name: str):
        """The pool adapter ``name`` as vLLM takes it, or None."""
        from vllm.lora.request import LoRARequest

        e = self.pool.entries.get(name)
        if e is None or not e.lora_int_id:
            return None
        return LoRARequest(lora_name=e.name, lora_int_id=int(e.lora_int_id),
                           lora_path=e.path)

    def _prefer(self, name: str):
        e = self.pool.entries.get(name)
        mirror = self.router.mirror
        in_slot = bool(e is not None and mirror is not None
                       and mirror.in_slot(e.lora_int_id))
        return (in_slot, 0 if e is None else e.hits)

    def _on_routed(self, request, decision) -> None:
        if decision.adapter:
            self.note_request(request)

    def note_inference(self) -> None:
        """Inference work is present (indexing leaves the GPU)."""
        self.placement.note_inference()

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
        """Every request the engine receives: inference is active (indexing
        leaves the GPU), and a request that uses a pool adapter is counted
        (drives GPU-slot retention and auto-retirement)."""
        self.placement.note_inference()
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
        self.pool.sync_tiers()
        return {**self.pool.status(), "jobs": jobs,
                "router": self.router.status(),
                "indexing": {"level": self.placement.level,
                             "level_changes": self.placement.changes,
                             "gpu_granted": bool(self.grant is not None
                                                 and self.grant.is_set())},
                "index_dir": self.dir}

    def worker_stats(self) -> list:
        """Every rank's route counters. One worker call — for GET /adapters,
        not part of ``status()``."""
        try:
            return [r for r in self.engine.model_executor.collective_rpc(
                "deltaserve_route_stats") if r is not None]
        except Exception:  # noqa: BLE001 — a stock worker, a test engine
            return []

    def shutdown(self) -> None:
        self._stop.set()
        if self._driver is not None:
            self._driver.join(timeout=3.0)
        self.shared_indexer.stop()
