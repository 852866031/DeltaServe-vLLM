# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The vLLM general plugin (entry point ``vllm.general_plugins: deltaserve``).

``register()`` runs in every vLLM process — the API server (from EngineArgs),
the EngineCore and every worker (``load_general_plugins``) — and again, as a
safety net, whenever a co-serving config is unpickled (``DeltaServeConfig``),
which in a spawned EngineCore / worker happens before vLLM builds anything.
It is idempotent.

Co-serving is switched on through vLLM's own ``additional_config``::

    vllm serve <model> --additional-config '{"deltaserve": {"enable_finetuning": true, ...}}'

(``deltaserve.config_loader.finetune_cli_args`` builds that from a YAML). The
classes that do the work are selected through vLLM's seams — ``worker_cls``
(``DeltaServeWorker``, which adopts ``DeltaServeModelRunner``) and
``scheduler_cls`` (``FinetuneScheduler``). Everything else is wrapped here,
around upstream methods, never replacing their bodies:

- ``VllmConfig.__post_init__``: turn the ``deltaserve`` entry into a
  ``DeltaServeConfig`` (its compile-cache hash covers only the settings that
  change the traced forward) and apply the co-serving defaults
  (``engine_config.configure_vllm_config``) before the platform fills in its own.
  ``VllmConfig.finetune_config`` reads it back (a disabled config otherwise).
- ``VllmConfig._get_v2_model_runner_unsupported_features``: co-serving runs on
  the v1 model runner.
- ``EngineArgs.create_engine_config``: ``disable_log_stats`` defaults on (the
  per-step stats stream stalls the API frontend; see engine_config).
- ``EngineCore`` / ``EngineCoreProc``: the ``EngineHooks`` (arrival signal
  before the executor, profiling pass before the busy loop, estimator save at
  shutdown, the input thread's ADD signal) and the
  ``deltaserve_save_estimator_state`` utility.
- ``register_vllm_serve_api_routers``: the /start_finetuning, /stop_finetuning
  and /save_estimator_state routes.
"""

import functools
import json
from collections.abc import Mapping

_REGISTERED = False


class DeltaServeConfig(Mapping):
    """``VllmConfig.additional_config`` of a co-serving run: the parsed
    ``FinetuneConfig`` next to any other additional_config entries. A read-only
    mapping like the dict it replaces; ``compute_hash`` keys vLLM's compile cache
    on the finetuning settings that change the traced forward only, so tuning an
    SLO or throttle knob does not recompile."""

    KEY = "deltaserve"

    def __init__(self, finetune, extra: dict) -> None:
        self.finetune = finetune
        self._extra = dict(extra)

    def _as_dict(self) -> dict:
        from dataclasses import asdict

        return {**self._extra, self.KEY: asdict(self.finetune)}

    def __getitem__(self, key):
        return self._as_dict()[key]

    def __iter__(self):
        return iter(self._as_dict())

    def __len__(self) -> int:
        return len(self._extra) + 1

    def compute_hash(self) -> str:
        from vllm.utils.hashing import safe_hash

        extra = json.dumps(self._extra, sort_keys=True, default=str)
        return safe_hash(f"{self.finetune.compute_hash()}|{extra}".encode(),
                         usedforsecurity=False).hexdigest()

    def __repr__(self) -> str:
        return f"DeltaServeConfig({self.finetune!r}, extra={self._extra!r})"

    def __setstate__(self, state) -> None:
        self.__dict__.update(state)
        register()   # a spawned process sees this before vLLM builds anything


def _disabled():
    from deltaserve.config import FinetuneConfig

    global _DISABLED
    if _DISABLED is None:
        _DISABLED = FinetuneConfig()
    return _DISABLED


_DISABLED = None


def _wrap(cls, name: str, make):
    orig = getattr(cls, name)
    if getattr(orig, "_deltaserve", False):
        return
    new = functools.wraps(orig)(make(orig))
    new._deltaserve = True
    setattr(cls, name, new)


def _patch_config() -> None:
    from vllm.config import VllmConfig
    from vllm.engine.arg_utils import EngineArgs

    from deltaserve.config import FinetuneConfig
    from deltaserve.engine_config import configure_vllm_config

    def finetune_config(self):
        ac = self.additional_config
        return ac.finetune if isinstance(ac, DeltaServeConfig) else _disabled()

    VllmConfig.finetune_config = property(finetune_config)

    def post_init(orig):
        def __post_init__(self):
            ac = self.additional_config
            if isinstance(ac, dict) and DeltaServeConfig.KEY in ac:
                extra = {k: v for k, v in ac.items() if k != DeltaServeConfig.KEY}
                self.additional_config = DeltaServeConfig(
                    FinetuneConfig(**ac[DeltaServeConfig.KEY]), extra)
            configure_vllm_config(self)
            return orig(self)
        return __post_init__

    _wrap(VllmConfig, "__post_init__", post_init)

    def v2_unsupported(orig):
        def _get_v2_model_runner_unsupported_features(self):
            out = orig(self)
            if self.finetune_config.enable_finetuning:
                # Our hooks live in the v1 runner: route there, and make an
                # explicit VLLM_USE_V2_MODEL_RUNNER=1 fail loudly.
                out.append("DeltaServe co-serving finetuning")
            return out
        return _get_v2_model_runner_unsupported_features

    _wrap(VllmConfig, "_get_v2_model_runner_unsupported_features", v2_unsupported)

    def create_engine_config(orig):
        def wrapper(self, *args, **kwargs):
            ac = self.additional_config or {}
            ft = ac.get(DeltaServeConfig.KEY) if isinstance(ac, dict) else None
            if ft and ft.get("enable_finetuning") and not self.disable_log_stats:
                from deltaserve import dprint

                # The per-step scheduler_stats stream saturates the API
                # frontend's event loop (TTFT spikes → FT admission backs off);
                # finetuning keeps its own engine-side timing.
                self.disable_log_stats = True
                dprint("enable_finetuning → disabling vLLM log_stats")
            return orig(self, *args, **kwargs)
        return wrapper

    _wrap(EngineArgs, "create_engine_config", create_engine_config)


def _patch_engine_core() -> None:
    from vllm.v1.engine import EngineCoreRequestType
    from vllm.v1.engine import core as core_mod

    from deltaserve.engine_core import (
        FtInputQueue,
        attach_engine_hooks,
        save_estimator_state,
    )

    EngineCore, EngineCoreProc = core_mod.EngineCore, core_mod.EngineCoreProc

    def init(orig):
        def __init__(self, vllm_config, *args, **kwargs):
            # Before the executor spawns the workers (arrival signal).
            self._ds = attach_engine_hooks(self, vllm_config)
            return orig(self, vllm_config, *args, **kwargs)
        return __init__

    _wrap(EngineCore, "__init__", init)

    def shutdown(orig):
        def wrapper(self, *args, **kwargs):
            ds = getattr(self, "_ds", None)
            if ds is not None:
                ds.on_shutdown()
            try:
                return orig(self, *args, **kwargs)
            finally:
                if ds is not None:
                    ds.after_shutdown()
        return wrapper

    _wrap(EngineCore, "shutdown", shutdown)

    def busy_loop(orig):
        def wrapper(self, *args, **kwargs):
            ds = getattr(self, "_ds", None)
            if ds is not None:
                ds.before_busy_loop()   # profiling pass, engine inputs
            return orig(self, *args, **kwargs)
        return wrapper

    for cls in {EngineCoreProc, getattr(core_mod, "DPEngineCoreProc", EngineCoreProc)}:
        if "run_busy_loop" in cls.__dict__:
            _wrap(cls, "run_busy_loop", busy_loop)

    def handle_shutdown(orig):
        def wrapper(self, *args, **kwargs):
            ds = getattr(self, "_ds", None)
            if (ds is not None and self.shutdown_state
                    == core_mod.EngineShutdownState.REQUESTED):
                ds.save_estimator_state()   # before the process is gone
            return orig(self, *args, **kwargs)
        return wrapper

    _wrap(EngineCoreProc, "_handle_shutdown", handle_shutdown)

    def input_sockets(orig):
        def wrapper(self, *args, **kwargs):
            ds = getattr(self, "_ds", None)
            if ds is not None:
                # Every ADD the input thread queues signals a mid-forward abort.
                FtInputQueue.adopt(self.input_queue, ds, EngineCoreRequestType.ADD)
            return orig(self, *args, **kwargs)
        return wrapper

    _wrap(EngineCoreProc, "process_input_sockets", input_sockets)

    if not hasattr(EngineCore, "deltaserve_save_estimator_state"):
        def deltaserve_save_estimator_state(self, path=None) -> dict:
            """Utility behind POST /save_estimator_state."""
            return save_estimator_state(self, path)

        EngineCore.deltaserve_save_estimator_state = deltaserve_save_estimator_state


def _patch_api_server() -> None:
    import sys

    # Only where the API server is (it imports this package before plugins
    # load); keeps fastapi out of the EngineCore and worker processes.
    serve = sys.modules.get("vllm.entrypoints.serve")
    if serve is None:
        return

    def routers(orig):
        def register_vllm_serve_api_routers(app):
            orig(app)
            from deltaserve.api import attach_router

            attach_router(app)
        return register_vllm_serve_api_routers

    if not getattr(serve.register_vllm_serve_api_routers, "_deltaserve", False):
        new = routers(serve.register_vllm_serve_api_routers)
        new._deltaserve = True
        serve.register_vllm_serve_api_routers = new


def register() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True
    _patch_config()
    _patch_engine_core()
    _patch_api_server()
