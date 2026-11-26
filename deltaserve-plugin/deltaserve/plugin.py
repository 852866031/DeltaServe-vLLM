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
- ``VllmConfig._get_v2_model_runner_unsupported_features`` /
  ``use_v2_model_runner`` (vLLM <= 0.33, which ships both runners): co-serving
  runs on the runner ``finetune_features.model_runner`` names.
- ``EngineArgs.create_engine_config``: ``disable_log_stats`` defaults on (the
  per-step stats stream stalls the API frontend; see engine_config).
- ``EngineCore`` / ``EngineCoreProc``: the ``EngineHooks`` (arrival signal
  before the executor, profiling pass before the busy loop, estimator save at
  shutdown, the input thread's ADD signal) and the
  ``deltaserve_save_estimator_state`` / ``deltaserve_finetuning_status``
  utilities.
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
    ADAPTER_ROUTING_KEY = "deltaserve_adapter_routing"

    def __init__(self, finetune, extra: dict, adapter_routing=None) -> None:
        self.finetune = finetune
        self._extra = dict(extra)
        # AdapterRoutingConfig, or None when the launch carried no adapter-routing
        # entry (the mapping and its hash are then exactly what they were without it).
        self.adapter_routing = adapter_routing

    def _as_dict(self) -> dict:
        from dataclasses import asdict

        out = {**self._extra, self.KEY: asdict(self.finetune)}
        if self.adapter_routing is not None:
            out[self.ADAPTER_ROUTING_KEY] = asdict(self.adapter_routing)
        return out

    def __getitem__(self, key):
        return self._as_dict()[key]

    def __iter__(self):
        return iter(self._as_dict())

    def __len__(self) -> int:
        return len(self._extra) + 1 + (self.adapter_routing is not None)

    def compute_hash(self) -> str:
        from vllm.utils.hashing import safe_hash

        # Adapter routing is not part of the hash: it never changes the traced forward.
        extra = json.dumps(self._extra, sort_keys=True, default=str)
        return safe_hash(f"{self.finetune.compute_hash()}|{extra}".encode(),
                         usedforsecurity=False).hexdigest()

    def compile_factors(self) -> dict:
        """The same, for vLLM >= 0.34 (``VllmConfig.compile_factors``; without
        it vLLM would key the cache on every entry of the mapping)."""
        return {"deltaserve": self.finetune.compute_hash(), "extra": self._extra}

    def __repr__(self) -> str:
        routing = ("" if self.adapter_routing is None
                   else f", adapter_routing={self.adapter_routing!r}")
        return (f"DeltaServeConfig({self.finetune!r}, extra={self._extra!r}"
                f"{routing})")

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


def _adapter_routing_disabled():
    from deltaserve.config import AdapterRoutingConfig

    global _ADAPTER_ROUTING_DISABLED
    if _ADAPTER_ROUTING_DISABLED is None:
        _ADAPTER_ROUTING_DISABLED = AdapterRoutingConfig()
    return _ADAPTER_ROUTING_DISABLED


_ADAPTER_ROUTING_DISABLED = None


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

    from deltaserve.config import FinetuneConfig, AdapterRoutingConfig
    from deltaserve.engine_config import configure_vllm_config

    def finetune_config(self):
        ac = self.additional_config
        return ac.finetune if isinstance(ac, DeltaServeConfig) else _disabled()

    VllmConfig.finetune_config = property(finetune_config)

    def adapter_routing_config(self):
        ac = self.additional_config
        routing = ac.adapter_routing if isinstance(ac, DeltaServeConfig) else None
        return routing if routing is not None else _adapter_routing_disabled()

    VllmConfig.adapter_routing_config = property(adapter_routing_config)

    def post_init(orig):
        def __post_init__(self):
            ac = self.additional_config
            keys = (DeltaServeConfig.KEY, DeltaServeConfig.ADAPTER_ROUTING_KEY)
            if isinstance(ac, dict) and any(k in ac for k in keys):
                extra = {k: v for k, v in ac.items() if k not in keys}
                routing = ac.get(DeltaServeConfig.ADAPTER_ROUTING_KEY)
                self.additional_config = DeltaServeConfig(
                    FinetuneConfig(**ac.get(DeltaServeConfig.KEY, {})), extra,
                    None if routing is None else AdapterRoutingConfig(**routing))
            configure_vllm_config(self)
            rc = self.adapter_routing_config
            lora = getattr(self, "lora_config", None)
            if rc.enable_adapter_routing and lora is not None:
                # vLLM keeps only max_cpu_loras adapters in host memory
                # (default: max_loras) and re-reads the others from disk each
                # time they are used: room for the whole active set.
                want = int(rc.max_active_adapters) + int(lora.max_loras)
                if (lora.max_cpu_loras or 0) < want:
                    lora.max_cpu_loras = want
            return orig(self)
        return __post_init__

    _wrap(VllmConfig, "__post_init__", post_init)

    from deltaserve.runner_select import wants_v2

    def v2_unsupported(orig):
        def _get_v2_model_runner_unsupported_features(self):
            out = orig(self)
            ft = self.finetune_config
            if ft.enable_finetuning and not wants_v2(ft):
                # The hooks of this launch live in the v1 runner: route there,
                # and make an explicit VLLM_USE_V2_MODEL_RUNNER=1 fail loudly.
                out.append("DeltaServe co-serving finetuning on the v1 runner "
                           "(finetune_features.model_runner)")
            return out
        return _get_v2_model_runner_unsupported_features

    # vLLM <= 0.33 ships both runners and picks one per launch; later versions
    # have only v2 (neither name exists there).
    if hasattr(VllmConfig, "_get_v2_model_runner_unsupported_features"):
        _wrap(VllmConfig, "_get_v2_model_runner_unsupported_features",
              v2_unsupported)
    use_v2 = VllmConfig.__dict__.get("use_v2_model_runner")
    if isinstance(use_v2, property) and not getattr(
            use_v2.fget, "_deltaserve", False):
        def use_v2_model_runner(self):
            ft = self.finetune_config
            if ft.enable_finetuning and wants_v2(ft):
                # finetune_features.model_runner: v2 — whatever vLLM would
                # pick for this model.
                return True
            return use_v2.fget(self)

        use_v2_model_runner._deltaserve = True
        VllmConfig.use_v2_model_runner = property(use_v2_model_runner)

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
        finetuning_status,
        metrics_summary,
        save_estimator_state,
        telemetry,
    )

    EngineCore, EngineCoreProc = core_mod.EngineCore, core_mod.EngineCoreProc

    def init(orig):
        def __init__(self, vllm_config, *args, **kwargs):
            # Before the executor spawns the workers (arrival signal).
            self._ds = attach_engine_hooks(self, vllm_config)
            self._ds_handover = None
            if vllm_config.finetune_config.enable_finetuning:
                # [adapter handover] Where every rank's backward child listens
                # for the engine; the name travels in the workers' environment.
                import os

                from deltaserve.handover_channel import (
                    ENV,
                    HandoverClient,
                    make_directory,
                )

                try:
                    directory = make_directory()
                    os.environ[ENV] = directory
                    self._ds_handover = HandoverClient(
                        directory,
                        vllm_config.parallel_config.tensor_parallel_size)
                except Exception as e:  # noqa: BLE001 — handover via the workers
                    from deltaserve import dprint

                    dprint(f"[ft-job] no direct handover line ({e})")
            self._ds_index_grant = None
            if vllm_config.adapter_routing_config.enable_adapter_routing:
                # [indexing] May indexing use the GPU? The engine writes it,
                # the indexing process (a child of a worker) obeys: its name
                # travels in the workers' environment.
                import os

                from deltaserve.adapter_routing.index_process import (
                    GRANT_ENV,
                    IndexingGrant,
                )

                try:
                    from deltaserve.adapter_routing.index_process import socket_path
                    from deltaserve.adapter_routing.stages import index_dir

                    try:    # a previous run's socket: its process is gone
                        os.unlink(socket_path(index_dir(
                            vllm_config.adapter_routing_config)))
                    except OSError:
                        pass
                    self._ds_index_grant = IndexingGrant.create()
                    os.environ[GRANT_ENV] = self._ds_index_grant.name
                except Exception as e:  # noqa: BLE001 — indexing stays on the CPU
                    from deltaserve import dprint

                    dprint(f"[indexing] no GPU grant ({e}): indexing stays on the CPU")
            orig(self, vllm_config, *args, **kwargs)
            # [ft-job] the scheduler saves a finished job's adapter through the
            # engine (worker RPC).
            if vllm_config.finetune_config.enable_finetuning:
                self.scheduler._ds_engine = self
            self._ds_routing = None
            if vllm_config.adapter_routing_config.enable_adapter_routing:
                # The only place the engine reaches into deltaserve.adapter_routing.
                from deltaserve.adapter_routing.engine import AdapterRoutingEngine

                self._ds_routing = AdapterRoutingEngine(
                    self, vllm_config, grant=self._ds_index_grant)
                self.scheduler._adapter_routing = self._ds_routing
                _route_in_scheduler(self.scheduler, self._ds_routing.router)
        return __init__

    _wrap(EngineCore, "__init__", init)

    def shutdown(orig):
        def wrapper(self, *args, **kwargs):
            ds = getattr(self, "_ds", None)
            if ds is not None:
                ds.on_shutdown()
            routing = getattr(self, "_ds_routing", None)
            if routing is not None:
                routing.shutdown()
            handover = getattr(self, "_ds_handover", None)
            if handover is not None:
                import shutil

                handover.close()
                shutil.rmtree(handover.directory, ignore_errors=True)
                self._ds_handover = None
            grant = getattr(self, "_ds_index_grant", None)
            if grant is not None:
                grant.set(False)
                grant.close()
                self._ds_index_grant = None
            try:
                return orig(self, *args, **kwargs)
            finally:
                if ds is not None:
                    ds.after_shutdown()
        return wrapper

    _wrap(EngineCore, "shutdown", shutdown)

    def add_request(orig):
        def wrapper(self, request, *args, **kwargs):
            routing = getattr(self, "_ds_routing", None)
            if routing is not None:
                router = routing.router
                if router.wants(request):
                    # [adapter routing] A routed request: the scheduler chooses
                    # its adapter (an encode step) before it joins the queue.
                    if router._add is None:
                        router._add = functools.partial(orig, self)
                    routing.note_inference()
                    router.hold(request, args, kwargs)
                    return None
                # Every request that names a pool adapter, with or without
                # finetuning: the pool's usage count.
                routing.note_request(request)
                if router._add is None:
                    router._add = functools.partial(orig, self)
                if router.hold_cold(request, args, kwargs):
                    # [cold start] its adapter is on disk: it joins the next
                    # step, this one reads the adapter in the background.
                    return None
            return orig(self, request, *args, **kwargs)
        return wrapper

    _wrap(EngineCore, "add_request", add_request)

    def abort_requests(orig):
        def wrapper(self, request_ids, *args, **kwargs):
            routing = getattr(self, "_ds_routing", None)
            if routing is not None:
                # A held request is not in the scheduler: vLLM's abort would
                # not find it.
                routing.router.abort(request_ids)
            return orig(self, request_ids, *args, **kwargs)
        return wrapper

    _wrap(EngineCore, "abort_requests", abort_requests)

    def process_inputs(orig):
        def wrapper(self, *args, **kwargs):
            from deltaserve.route_middleware import (
                REQUEST_ARRIVAL,
                ROUTE_HEADER,
                ROUTE_PAYLOAD,
            )

            # [adapter routing] arrival_time = when the request reached the API
            # server (set by the route middleware). None outside a routed
            # endpoint: vLLM's own stamp.
            if len(args) < 5 and kwargs.get("arrival_time") is None:
                t = REQUEST_ARRIVAL.get()
                if t is not None:
                    kwargs["arrival_time"] = t
            # [adapter routing] The route encoder's token ids go into the
            # engine with the request (trace_headers is the 8th parameter).
            payload = ROUTE_PAYLOAD.get()
            if payload is not None:
                ROUTE_PAYLOAD.set(None)        # one engine request per route
                if len(args) >= 8:             # given positionally
                    args = (*args[:7], {**(args[7] or {}), ROUTE_HEADER: payload},
                            *args[8:])
                else:
                    kwargs["trace_headers"] = {
                        **(kwargs.get("trace_headers") or {}), ROUTE_HEADER: payload}
            return orig(self, *args, **kwargs)
        return wrapper

    from vllm.v1.engine.input_processor import InputProcessor

    _wrap(InputProcessor, "process_inputs", process_inputs)

    def process_outputs(orig):
        def wrapper(self, engine_core_outputs, *args, **kwargs):
            from deltaserve.route_middleware import DECISIONS

            # [adapter routing] The engine's decision for a routed request
            # rides back on its outputs; the route middleware waits for it.
            if DECISIONS.waiting:
                for out in engine_core_outputs:
                    headers = out.trace_headers
                    if headers:
                        DECISIONS.note_headers(headers)
            return orig(self, engine_core_outputs, *args, **kwargs)
        return wrapper

    from vllm.v1.engine.output_processor import OutputProcessor

    _wrap(OutputProcessor, "process_outputs", process_outputs)

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
                ds.on_shutdown_requested()  # estimator + metrics, before the process is gone
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

    if not hasattr(EngineCore, "deltaserve_adapters"):
        def deltaserve_adapters(self) -> dict:
            """Utility behind GET /adapters: the adapter pool and its jobs."""
            routing = getattr(self, "_ds_routing", None)
            if routing is None:
                return {"error": "adapter routing is off"}
            return {**routing.status(), "workers": routing.worker_stats()}

        def deltaserve_add_adapter(self, spec: dict) -> dict:
            """Utility behind POST /adapters: add an existing adapter."""
            routing = getattr(self, "_ds_routing", None)
            if routing is None:
                return {"error": "adapter routing is off"}
            try:
                return {"job": routing.submit_adapter(**spec)}
            except Exception as e:  # noqa: BLE001 — reported to the caller
                return {"error": f"{type(e).__name__}: {e}"}

        def deltaserve_adapter_op(self, name: str, op: str) -> dict:
            """Utility behind POST /adapters/{name}/{op} and DELETE."""
            routing = getattr(self, "_ds_routing", None)
            if routing is None:
                return {"error": "adapter routing is off"}
            try:
                return routing.adapter_op(name, op)
            except Exception as e:  # noqa: BLE001 — reported to the caller
                return {"error": f"{type(e).__name__}: {e}"}

        EngineCore.deltaserve_adapter_op = deltaserve_adapter_op
        EngineCore.deltaserve_adapters = deltaserve_adapters
        EngineCore.deltaserve_add_adapter = deltaserve_add_adapter

    if not hasattr(EngineCore, "deltaserve_submit_ft_job"):
        def deltaserve_submit_ft_job(self, spec: dict) -> dict:
            """Utility behind POST /ft_jobs: queue a finetuning job."""
            submit = getattr(self.scheduler, "submit_ft_job", None)
            if submit is None:
                return {"error": "finetuning is off on this server"}
            try:
                return {"job": submit(spec)}
            except Exception as e:  # noqa: BLE001 — reported to the caller
                return {"error": f"{type(e).__name__}: {e}"}

        EngineCore.deltaserve_submit_ft_job = deltaserve_submit_ft_job

    if not hasattr(EngineCore, "deltaserve_ft_jobs"):
        def deltaserve_ft_jobs(self) -> dict:
            """Utility behind GET /ft_jobs: the finetuning jobs and their stages."""
            jobs = getattr(self.scheduler, "_ft_jobs", None)
            return {"current": None, "queued": [], "finished": []} \
                if jobs is None else jobs.status()

        EngineCore.deltaserve_ft_jobs = deltaserve_ft_jobs


    if not hasattr(EngineCore, "deltaserve_finetuning_status"):
        def deltaserve_finetuning_status(self) -> dict:
            """Utility behind GET /finetuning_status."""
            return finetuning_status(self)

        EngineCore.deltaserve_finetuning_status = deltaserve_finetuning_status

    if not hasattr(EngineCore, "deltaserve_telemetry"):
        def deltaserve_telemetry(self, op, arg=None) -> dict:
            """Utility behind /telemetry/{data,start,stop}."""
            return telemetry(self, op, arg)

        EngineCore.deltaserve_telemetry = deltaserve_telemetry

    if not hasattr(EngineCore, "deltaserve_metrics_summary"):
        def deltaserve_metrics_summary(self) -> dict:
            """Utility behind GET /system_metrics."""
            return metrics_summary(self)

        EngineCore.deltaserve_metrics_summary = deltaserve_metrics_summary


def _route_in_scheduler(scheduler, router) -> None:
    """[adapter routing] Make held (routed, adapter not chosen yet) requests
    part of the scheduler's work: they count as requests — otherwise the
    engine loop would sleep on them — and every ``schedule()`` starts with
    the encode step that releases them. Wrapped on the object: any scheduler
    class, with or without finetuning."""
    for name in ("has_requests", "has_unfinished_requests"):
        orig = getattr(scheduler, name)
        setattr(scheduler, name,
                (lambda orig: lambda *a, **k: router.pending or orig(*a, **k))(orig))
    schedule = scheduler.schedule
    requests = scheduler.requests
    load_seconds = getattr(scheduler, "_adapter_load_seconds",
                           lambda copies, reads: 0.0)

    def busy() -> bool:
        # Inference requests in the scheduler (finetuning samples do not
        # count: a step with only those is not worth waiting behind).
        return any(not str(rid).startswith("__ft__") for rid in requests)

    router.busy = busy

    def schedule_routed(*args, **kwargs):
        if router.on_busy is not None and (router.pending or busy()):
            router.on_busy()          # [indexing] inference is present
        router.encode_s = 0.0
        router.release_cold()
        router.route_pending()
        out = schedule(*args, **kwargs)
        loras = {}
        for rid in getattr(out, "num_scheduled_tokens", None) or ():
            req = requests.get(rid)
            lora = getattr(req, "lora_request", None)
            if lora is not None:
                loras[lora.lora_int_id] = lora
        preload, copies, reads = router.step_issued(loras)
        if preload:
            out._ds_preload = preload     # pickled with the output (TP)
        if copies or reads:
            out._ds_adapter_load = (copies, reads)
            feats = getattr(out, "_ft_step_features", None)
            if feats is not None:
                # [estimator] the step's adapter load is not the forward's
                # cost: taken off the measured duration before the fit.
                feats.load_s = load_seconds(copies, reads)
        return out

    scheduler.schedule = schedule_routed


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
