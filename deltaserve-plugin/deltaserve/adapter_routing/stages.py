# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Finetuning-job stages that exist only with adapter routing on.

Importing this module registers them (``deltaserve.ft_job`` does so when the
feature is enabled). Light at import: the encoder loads in its own process.
"""

import os
import tempfile

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


@register_stage
class IndexingStage(Stage):
    """Build the job adapter's centroid from its dataset and add it to the
    centroid index. Depends only on the dataset, so it runs next to training.

    Runs on its own thread (``IndexingRunner``); the stage only reports."""

    name = "indexing"
    # Replaced in tests; the GPU executor joins this dict when it exists.
    make_executors = staticmethod(lambda cfg: {"cpu": make_cpu_executor(cfg)})

    def requirements(self, stage_names):
        return ("init",)

    def start(self, job, ctx) -> None:
        cfg = ctx.adapter_routing_config
        if not job.spec.data_path:
            raise RuntimeError("the job has no dataset to index")
        texts = select_samples(job.spec.data_path, cfg.indexing_max_samples)
        if not texts:
            raise RuntimeError(f"no sample to index in {job.spec.data_path}")
        self.cfg, self.dir = cfg, index_dir(cfg)
        fp = fingerprint(texts, cfg.encoder, cfg.encoder_query_prefix,
                         cfg.encoder_max_tokens)
        indexer = Indexer(texts, fp, os.path.join(
            self.dir, "checkpoints", f"{job.name}.npz"))
        if indexer.resumed_from:
            dprint(f"[indexing] {job.name} resumes at "
                   f"{indexer.resumed_from}/{len(texts)}")
        governor = CpuGovernor(
            max_threads=cfg.indexing_cpu_max_threads,
            reserve_cores=cfg.indexing_cpu_reserve_cores,
            min_free_memory_bytes=int(cfg.indexing_min_free_memory_gb * (1 << 30)),
            cpus=parse_cpu_list(cfg.indexing_cpus))
        self.runner = IndexingRunner(
            indexer, self.make_executors(cfg), governor=governor,
            placement=IndexingPlacement(cfg.indexing_gpu_after_quiet_s),
            chunk_samples=cfg.indexing_chunk_samples, name=job.name)
        self.runner.start()

    def poll(self, job, ctx) -> bool:
        runner = self.runner
        if runner.error:
            raise RuntimeError(runner.error)
        if not runner.done:
            return False
        path = os.path.join(self.dir, INDEX_FILE)
        cfg = self.cfg
        index = (CentroidIndex.load(path) if os.path.exists(path) else
                 CentroidIndex(cfg.encoder, cfg.encoder_query_prefix,
                               cfg.encoder_max_tokens))
        if len(index) and (index.encoder, index.prefix, index.max_tokens) != (
                cfg.encoder, cfg.encoder_query_prefix, cfg.encoder_max_tokens):
            raise RuntimeError(
                f"{path} was built with encoder '{index.encoder}', not "
                f"'{cfg.encoder}'; centroids of different encoders cannot mix")
        index.add(job.name, runner.indexer.centroid())
        index.save(path)
        dprint(f"[indexing] {job.name} done: centroid added to {path} "
               f"({len(index)} adapters) | {runner.describe()}")
        return True
