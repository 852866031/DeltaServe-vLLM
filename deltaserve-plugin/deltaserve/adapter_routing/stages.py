# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Finetuning-job stages that exist only with adapter routing on.

Importing this module registers them (``deltaserve.ft_job`` does so when the
feature is enabled). Light at import: the encoder loads in its own process.
"""

import json
import os
import tempfile
import threading

from deltaserve import dprint
from deltaserve.adapter_routing.governor import CpuGovernor, parse_cpu_list
from deltaserve.adapter_routing.index import (
    CentroidIndex,
    Indexer,
    fingerprint,
    select_samples,
)
from deltaserve.adapter_routing.runner import IndexingPlacement, IndexingRunner
from deltaserve.ft_job import Stage, register_stage

INDEX_FILE = "centroids.npz"
# name → fingerprint (samples + encoder) its centroid was built from.
META_FILE = "centroids.json"


def index_dir(cfg) -> str:
    """Where centroids and indexing checkpoints live: ``index_path``, or a
    per-user temp directory when none is configured (lost on reboot)."""
    return cfg.index_path or os.path.join(
        tempfile.gettempdir(), f"deltaserve-adapter-index-{os.getuid()}")


def make_cpu_executor(cfg):
    from deltaserve.adapter_routing.encoder import CpuIndexerProcess

    return CpuIndexerProcess(cfg.encoder, cfg.encoder_query_prefix,
                             cfg.encoder_max_tokens,
                             parse_cpu_list(cfg.indexing_cpus))


class SharedIndexer:
    """One set of indexing executors (one encoder process) for every job of a
    server, used by one job at a time. Without it each job would start its own
    process and load the encoder again."""

    def __init__(self, factory) -> None:
        self._factory = factory
        self._lock = threading.Lock()
        self._executors = None
        self._busy = False
        self.created = 0

    def acquire(self):
        """The executors, or None while another job is using them."""
        with self._lock:
            if self._busy:
                return None
            if self._executors is None:
                self._executors = self._factory()
                self.created += 1
            self._busy = True
            return self._executors

    def release(self) -> None:
        with self._lock:
            self._busy = False

    @property
    def idle_and_loaded(self) -> bool:
        with self._lock:
            return (not self._busy and self._executors is not None
                    and any(getattr(ex, "loaded", False)
                            for ex in self._executors.values()))

    def unload(self) -> None:
        """Free the encoder's memory when no job needs it (process stays)."""
        with self._lock:
            if self._busy or self._executors is None:
                return
            for ex in self._executors.values():
                ex.unload()

    def stop(self) -> None:
        with self._lock:
            executors, self._executors = self._executors, None
        for ex in (executors or {}).values():
            try:
                ex.stop()
            except Exception:  # noqa: BLE001
                pass


# centroids.npz is updated by whichever thread finishes a job's indexing.
_INDEX_FILE_LOCK = threading.Lock()


def _read_meta(directory: str) -> dict:
    try:
        with open(os.path.join(directory, META_FILE)) as f:
            meta = json.load(f)
        return meta if isinstance(meta, dict) else {}
    except (OSError, ValueError):
        return {}


def centroid_is_current(directory: str, name: str, fp: str) -> bool:
    """A centroid for ``name`` built from exactly this data + encoder exists."""
    if _read_meta(directory).get(name) != fp:
        return False
    path = os.path.join(directory, INDEX_FILE)
    return os.path.exists(path) and name in CentroidIndex.load(path)


@register_stage
class IndexingStage(Stage):
    """Build the job adapter's centroid from its dataset and add it to the
    centroid index. Depends only on the dataset, so it runs next to training.

    Kept cheap:
    - nothing is encoded when a centroid built from the same samples with the
      same encoder already exists (a restart, or an adapter listed again);
    - the samples are encoded shortest-first (the mean does not care about
      order; chunks of similar length waste no padding);
    - a server's jobs share one encoder process (``SharedIndexer``).

    Runs on its own thread (``IndexingRunner``); the stage only reports."""

    name = "indexing"
    # Private executors, used when the server offers no shared indexer.
    # Replaced in tests; the GPU executor joins this dict when it exists.
    make_executors = staticmethod(lambda cfg: {"cpu": make_cpu_executor(cfg)})

    def requirements(self, stage_names):
        return ("init",)

    def start(self, job, ctx) -> None:
        cfg = ctx.adapter_routing_config
        if not job.spec.data_path:
            raise RuntimeError("the job has no dataset to index")
        texts = select_samples(job.spec.data_path, cfg.indexing_max_samples,
                               input_until=cfg.indexing_input_until)
        if not texts:
            raise RuntimeError(f"no sample to index in {job.spec.data_path}")
        texts.sort(key=len)
        self.cfg, self.dir, self.runner = cfg, index_dir(cfg), None
        self.fp = fingerprint(texts, cfg.encoder, cfg.encoder_query_prefix,
                              cfg.encoder_max_tokens)
        self.cached = centroid_is_current(self.dir, job.name, self.fp)
        if self.cached:
            dprint(f"[indexing] {job.name}: centroid already built from this "
                   f"dataset, nothing to encode")
            return
        self._indexer = Indexer(texts, self.fp, os.path.join(
            self.dir, "checkpoints", f"{job.name}.npz"),
            reference_samples=cfg.route_reference_samples)
        if self._indexer.resumed_from:
            dprint(f"[indexing] {job.name} resumes at "
                   f"{self._indexer.resumed_from}/{len(texts)}")
        self._shared = getattr(ctx, "shared_indexer", None) or getattr(
            getattr(ctx, "_adapter_routing", None), "shared_indexer", None)

    def _try_start(self, job) -> bool:
        cfg = self.cfg
        if self._shared is not None:
            executors = self._shared.acquire()
            if executors is None:
                return False            # another job is indexing: wait our turn
            share = dict(stop_executors=False, on_exit=self._shared.release)
        else:
            executors, share = self.make_executors(cfg), {}
        governor = CpuGovernor(
            max_threads=cfg.indexing_cpu_max_threads,
            reserve_cores=cfg.indexing_cpu_reserve_cores,
            min_free_memory_bytes=int(cfg.indexing_min_free_memory_gb * (1 << 30)),
            cpus=parse_cpu_list(cfg.indexing_cpus))
        self.runner = IndexingRunner(
            self._indexer, executors, governor=governor,
            placement=IndexingPlacement(cfg.indexing_gpu_after_quiet_s),
            chunk_samples=cfg.indexing_chunk_samples, name=job.name, **share)
        self.runner.start()
        return True

    def poll(self, job, ctx) -> bool:
        if self.cached:
            return True
        if self.runner is None and not self._try_start(job):
            return False
        runner = self.runner
        if runner.error:
            raise RuntimeError(runner.error)
        if not runner.done:
            return False
        path = os.path.join(self.dir, INDEX_FILE)
        cfg = self.cfg
        with _INDEX_FILE_LOCK:
            index = (CentroidIndex.load(path) if os.path.exists(path) else
                     CentroidIndex(cfg.encoder, cfg.encoder_query_prefix,
                                   cfg.encoder_max_tokens))
            if len(index) and (index.encoder, index.prefix, index.max_tokens) != (
                    cfg.encoder, cfg.encoder_query_prefix, cfg.encoder_max_tokens):
                raise RuntimeError(
                    f"{path} was built with encoder '{index.encoder}', not "
                    f"'{cfg.encoder}'; centroids of different encoders cannot mix")
            index.add(job.name, runner.indexer.centroid(),
                      refs=runner.indexer.references())
            index.save(path)
            meta = _read_meta(self.dir)
            meta[job.name] = self.fp
            tmp = os.path.join(self.dir, META_FILE + ".tmp")
            with open(tmp, "w") as f:
                json.dump(meta, f, indent=1)
            os.replace(tmp, os.path.join(self.dir, META_FILE))
        dprint(f"[indexing] {job.name} done: centroid added to {path} "
               f"({len(index)} adapters) | {runner.describe()}")
        return True


@register_stage
class AdmissionStage(Stage):
    """The adapter joins the pool: it is compared with the active adapters,
    the new pool state (centroid mean, statistics, threshold) is built on the
    side and published in one step — the pool keeps serving throughout, and a
    failed admission leaves it as it was. For a job that trained, the adapter
    is saved first. The report is kept on the job."""

    name = "admission"

    def requirements(self, stage_names):
        return tuple(s for s in stage_names if s not in (self.name, "ready"))

    def start(self, job, ctx) -> None:
        ctx.admit_job(job)
