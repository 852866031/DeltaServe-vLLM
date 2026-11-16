# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Finetuning jobs and their stages.

A ``FinetuneJob`` is one finetuning run: a dataset, the LoRA adapter it trains
and the list of **stages** it goes through. Stages belong to the job (not to the
adapter), and the job's config says which ones it has::

    ft_job:
      stages: [init, training, ready]

A stage is a small class registered by name (``register_stage``). It declares
which other stages must be done before it may run (``requirements``); the
``FtJobManager`` runs every stage whose requirements are met, so stages that
only share a prerequisite (training and, with adapter routing, indexing) run
side by side — the list is a set of stages, not an execution order. Adding a
stage later (validation, data collection, ...) is one class plus its name in a
job's list.

One job runs at a time; further jobs queue.

Without an ``ft_job`` section the flat ``finetune:`` settings describe an
implicit job with stages ``[init, training]`` — the behaviour before jobs
existed. The built-in stages only OBSERVE the existing co-serving machinery
(store, coordinator); they never drive it, so the job layer cannot change what
is admitted, saved or trained.

Pure Python: no vLLM / torch import, usable from the config loader and tests.
"""

import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from deltaserve import dprint

PENDING, RUNNING, DONE, FAILED = "pending", "running", "done", "failed"

DEFAULT_STAGES = ("init", "training")

# ft_job keys that are job metadata (kept in ``FinetuneConfig.ft_job``) ...
JOB_META_KEYS = ("name", "stages", "auto_routing", "adapter_init", "save_path")
# ... and ft_job keys that set an existing flat FinetuneConfig field, which the
# co-serving machinery already reads.
JOB_FLAT_KEYS = {
    "adapter_path": "finetuning_lora_path",
    "data_path": "data_path",
    "num_epochs": "num_epochs",
    "learning_rate": "learning_rate",
    "weight_decay": "weight_decay",
    "gamma": "gamma",
}
# How the job's adapter gets its initial weights. Only loading an existing
# adapter directory exists today; new methods are added here.
ADAPTER_INIT_METHODS = ("from_path",)

# Stages that exist only while a feature is on: name → the switch to name in
# the error when a job lists the stage with the feature off.
GATED_STAGES = {"indexing": "adapter_routing.enable_adapter_routing",
                "admission": "adapter_routing.enable_adapter_routing"}


# ----------------------------------------------------------------- stages
class Stage:
    """One step of a finetuning job. Subclass, set ``name``, register."""

    name: str = ""

    def requirements(self, stage_names: tuple[str, ...]) -> tuple[str, ...]:
        """The stages of this job that must be DONE before this one starts."""
        return ()

    def start(self, job: "FinetuneJob", ctx: Any) -> None:
        """Called once, when the requirements are met."""

    def poll(self, job: "FinetuneJob", ctx: Any) -> bool:
        """Called every manager tick while running; True once the stage is done."""
        return True


_STAGES: dict[str, type[Stage]] = {}


def register_stage(cls: type[Stage]) -> type[Stage]:
    if not cls.name:
        raise ValueError(f"{cls.__name__} has no stage name")
    _STAGES[cls.name] = cls
    return cls


def known_stages() -> tuple[str, ...]:
    return tuple(_STAGES)


def load_gated_stages(adapter_routing: bool) -> None:
    """Register the stages a feature owns. The only place the job layer reaches
    into ``deltaserve.adapter_routing``, and only with the feature on."""
    if adapter_routing and "indexing" not in _STAGES:
        import deltaserve.adapter_routing.stages  # noqa: F401 — registers


@register_stage
class InitStage(Stage):
    """The job's adapter weights are loaded and its dataset tokenized. Both
    happen while the engine starts (worker load, scheduler init), so this stage
    only confirms the dataset is there."""

    name = "init"

    def start(self, job, ctx) -> None:
        begin = getattr(ctx, "begin_job", None)
        if begin is not None:
            begin(job)       # a job submitted while serving: prepare its dataset

    def poll(self, job, ctx) -> bool:
        ready = getattr(ctx, "job_ready_to_train", None)
        if ready is not None and not ready(job):
            return False     # still preparing / waiting to switch adapters
        store = getattr(ctx, "store", None)
        if store is None:
            # A job that does not train (an existing adapter being indexed):
            # nothing is loaded or tokenized, the inputs only have to exist.
            import os

            spec = job.spec
            if not spec.adapter_path or not os.path.isfile(os.path.join(
                    spec.adapter_path, "adapter_model.safetensors")):
                raise RuntimeError(f"no adapter weights at {spec.adapter_path}")
            if "indexing" in spec.stages and not (
                    spec.data_path and os.path.isfile(spec.data_path)):
                raise RuntimeError(f"no dataset at {spec.data_path}")
            return True
        if store.data_path is not None and len(store) == 0:
            raise RuntimeError(
                f"finetuning dataset has no usable sample: {store.data_path}")
        return True


@register_stage
class TrainingStage(Stage):
    """The co-serving finetuning loop. Done when the last epoch's samples are
    all trained: nothing left to select or in flight, the activation buffer is
    empty and no backward is pending."""

    name = "training"

    def requirements(self, stage_names):
        return ("init",)

    def poll(self, job, ctx) -> bool:
        store, coord = ctx.store, ctx.coordinator
        if store.data_path is None:
            return True
        return (not store.has_next()
                and store.current_epoch >= store.total_epochs - 1
                and not store.has_claimed()
                and not coord.pending_backward
                and coord.fill_count + coord.reserved_fill == 0)


@register_stage
class ReadyStage(Stage):
    """Terminal stage: every other stage of the job is done. The server's
    ``on_job_ready`` hook then saves the trained adapter and, for an
    auto-routable job, adds it to the adapter pool; a failure there fails the
    job (an adapter that could not be saved is not ready)."""

    name = "ready"

    def requirements(self, stage_names):
        return tuple(s for s in stage_names if s != self.name)

    def start(self, job, ctx) -> None:
        hook = getattr(ctx, "on_job_ready", None)
        if hook is not None:
            hook(job)


# ------------------------------------------------------------------- spec
@dataclass
class FinetuneJobSpec:
    """What a finetuning job is: resolved from ``FinetuneConfig`` (the flat
    fields + the ``ft_job`` metadata)."""

    name: str | None = None
    stages: tuple[str, ...] = DEFAULT_STAGES
    auto_routing: bool = False
    adapter_init: str = "from_path"
    save_path: str | None = None
    adapter_path: str | None = None
    data_path: str | None = None
    num_epochs: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    gamma: float = 1.0

    @classmethod
    def from_finetune_config(cls, ft_cfg) -> "FinetuneJobSpec":
        meta = dict(getattr(ft_cfg, "ft_job", None) or {})
        unknown = sorted(set(meta) - set(JOB_META_KEYS))
        if unknown:
            raise ValueError(f"unknown ft_job key(s) {unknown}; known: "
                             f"{sorted((*JOB_META_KEYS, *JOB_FLAT_KEYS))}")
        stages = meta.get("stages")
        if stages is None:
            stages = DEFAULT_STAGES
        elif isinstance(stages, str) or not all(isinstance(s, str) for s in stages):
            raise ValueError("ft_job.stages must be a list of stage names")
        return cls(
            name=meta.get("name") or None,
            stages=tuple(stages),
            auto_routing=bool(meta.get("auto_routing", False)),
            adapter_init=meta.get("adapter_init") or "from_path",
            save_path=meta.get("save_path") or None,
            adapter_path=ft_cfg.finetuning_lora_path,
            data_path=ft_cfg.data_path,
            num_epochs=int(ft_cfg.num_epochs),
            learning_rate=float(ft_cfg.learning_rate),
            weight_decay=float(ft_cfg.weight_decay),
            gamma=float(ft_cfg.gamma),
        )

    def validate(self, adapter_routing: bool = False) -> None:
        """Reject a job that cannot run. ``adapter_routing``: whether the
        adapter-routing feature is on (it owns some stages and the pool)."""
        load_gated_stages(adapter_routing)
        stages = self.stages
        if not stages:
            raise ValueError("ft_job.stages is empty")
        if len(set(stages)) != len(stages):
            raise ValueError(f"ft_job.stages has a duplicate: {list(stages)}")
        for s in stages:
            if s in _STAGES and (adapter_routing or s not in GATED_STAGES):
                continue
            gate = GATED_STAGES.get(s)
            if gate is not None and not adapter_routing:
                raise ValueError(
                    f"ft_job stage '{s}' needs {gate}: true (it is off)")
            raise ValueError(f"unknown ft_job stage '{s}'; available: "
                             f"{list(_STAGES)}")
        if "init" not in stages:
            raise ValueError("ft_job.stages must contain 'init'")
        for s in stages:
            missing = [r for r in _STAGES[s]().requirements(stages)
                       if r not in stages]
            if missing:
                raise ValueError(
                    f"ft_job stage '{s}' requires {missing}, not in the job's "
                    f"stages {list(stages)}")
        if self.adapter_init not in ADAPTER_INIT_METHODS:
            raise ValueError(
                f"ft_job.adapter_init '{self.adapter_init}' is not supported; "
                f"available: {list(ADAPTER_INIT_METHODS)}")
        if self.auto_routing and not adapter_routing:
            raise ValueError(
                "ft_job.auto_routing needs "
                "adapter_routing.enable_adapter_routing: true (it is off)")


def generate_adapter_name(auto_routing: bool, taken=()) -> str:
    """A random adapter name for a job that was given none:
    ``dserve-routing-<8 hex>`` for an auto-routable adapter, ``dserve-<8 hex>``
    otherwise; never one in ``taken``."""
    prefix = "dserve-routing-" if auto_routing else "dserve-"
    while True:
        name = prefix + secrets.token_hex(4)
        if name not in taken:
            return name


# -------------------------------------------------------------------- job
@dataclass
class FinetuneJob:
    spec: FinetuneJobSpec
    name: str
    stage_state: dict[str, str] = field(default_factory=dict)
    stage_times: dict[str, list] = field(default_factory=dict)
    error: str | None = None
    saved_path: str | None = None     # where 'ready' wrote the trained adapter
    admission: dict | None = None     # the pool's admission report, once admitted
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        for s in self.spec.stages:
            self.stage_state.setdefault(s, PENDING)
        self._stages = {s: _STAGES[s]() for s in self.spec.stages}

    @property
    def active_stages(self) -> tuple[str, ...]:
        return tuple(s for s in self.spec.stages
                     if self.stage_state[s] == RUNNING)

    @property
    def failed(self) -> bool:
        return self.error is not None

    @property
    def finished(self) -> bool:
        return self.failed or all(
            v == DONE for v in self.stage_state.values())

    @property
    def stage(self) -> str:
        """The job's stage as one string: the running stages joined by '+'
        (``training+indexing``), else ``failed`` / ``done`` / ``pending``."""
        if self.failed:
            return FAILED
        active = self.active_stages
        if active:
            return "+".join(active)
        return DONE if self.finished else PENDING

    def tick(self, ctx: Any) -> bool:
        """Start every stage whose requirements are done, poll the running
        ones. Returns True when a stage changed state."""
        if self.finished:
            return False
        changed = False
        names = self.spec.stages
        for s in names:
            state = self.stage_state[s]
            if state in (DONE, FAILED):
                continue
            stage = self._stages[s]
            try:
                if state == PENDING:
                    if any(self.stage_state[r] != DONE
                           for r in stage.requirements(names)):
                        continue
                    stage.start(self, ctx)
                    self.stage_state[s] = RUNNING
                    self.stage_times[s] = [time.time(), None]
                    changed = True
                if stage.poll(self, ctx):
                    self.stage_state[s] = DONE
                    self.stage_times[s][1] = time.time()
                    changed = True
            except Exception as e:  # noqa: BLE001 — a stage must not kill serving
                self.stage_state[s] = FAILED
                self.error = f"stage '{s}': {type(e).__name__}: {e}"
                dprint(f"[ft-job] {self.name} FAILED in {self.error}")
                return True
        return changed

    def describe(self) -> dict:
        return {
            "name": self.name,
            "stage": self.stage,
            "stages": {s: self.stage_state[s] for s in self.spec.stages},
            "stage_times": {s: list(t) for s, t in self.stage_times.items()},
            "auto_routing": self.spec.auto_routing,
            "adapter_path": self.spec.adapter_path,
            "saved_path": self.saved_path,
            "admission": self.admission,
            "data_path": self.spec.data_path,
            "num_epochs": self.spec.num_epochs,
            "error": self.error,
            # what a running stage wants to show (e.g. indexing progress)
            "detail": {s: st.runner.describe()
                       for s, st in self._stages.items()
                       if hasattr(getattr(st, "runner", None), "describe")},
        }


# ---------------------------------------------------------------- manager
class FtJobManager:
    """The finetuning jobs of one server: at most one runs, the rest queue."""

    def __init__(self, adapter_routing: bool = False) -> None:
        self.adapter_routing = bool(adapter_routing)
        self.current: FinetuneJob | None = None
        self.queue: list[FinetuneJob] = []
        self.finished: list[FinetuneJob] = []

    def _names(self) -> set[str]:
        jobs = [*self.queue, *self.finished]
        if self.current is not None:
            jobs.append(self.current)
        return {j.name for j in jobs}

    def submit(self, spec: FinetuneJobSpec,
               replace_finished: bool = False) -> FinetuneJob:
        """``replace_finished``: a job that already finished under this name is
        superseded (the same adapter submitted again, e.g. with a new dataset)."""
        spec.validate(self.adapter_routing)
        if replace_finished and spec.name:
            self.finished = [j for j in self.finished if j.name != spec.name]
        taken = self._names()
        name = spec.name or generate_adapter_name(spec.auto_routing, taken)
        if name in taken:
            raise ValueError(f"a finetuning job named '{name}' already exists")
        job = FinetuneJob(spec=spec, name=name)
        if self.current is None:
            self.current = job
        else:
            self.queue.append(job)
        dprint(f"[ft-job] {name} submitted | stages={list(spec.stages)} "
               f"auto_routing={spec.auto_routing}")
        return job

    def tick(self, ctx: Any) -> None:
        """Advance the running job. Cheap when there is nothing to do."""
        job = self.current
        if job is None:
            return
        if job.tick(ctx):
            dprint(f"[ft-job] {job.name} → {job.stage} {job.stage_state}")
        if job.finished:
            self.finished.append(job)
            self.current = self.queue.pop(0) if self.queue else None

    @property
    def busy(self) -> bool:
        """A job is running or waiting (the engine keeps ticking for it)."""
        return self.current is not None or bool(self.queue)

    def status(self) -> dict:
        return {
            "current": None if self.current is None else self.current.describe(),
            "queued": [j.describe() for j in self.queue],
            "finished": [j.describe() for j in self.finished],
        }
