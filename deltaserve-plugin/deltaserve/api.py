# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""HTTP control for co-serving finetuning admission.

POST /start_finetuning opens FT admission (when the server was launched with
finetune_param.start_on_launch=false). It runs `deltaserve_start_finetuning` on the
worker via collective_rpc; under single-GPU/uniproc the worker shares the
process with the scheduler + coordinator, so the flag flip is seen immediately.

POST /stop_finetuning closes FT admission again — a stop / start cycle resumes
from the same buffer + counter state (the coordinator's ``stop_finetuning``
intentionally preserves in-flight work). Useful for pausing FT during a
benchmark phase or to bracket a deliberate idle window.

POST /save_adapter writes the trained adapter (PEFT format); GET
/finetuning_status reports admission + training progress; GET
/system_metrics the system metrics (telemetry.metrics_dir); /telemetry/*
the live feed of dserve-top (start / stop recording at runtime, data); POST
/shutdown stops the server the way Ctrl-C does (dserve-top's ■ STOP button),
accepted only from the server's own machine.

Only attached when finetuning is enabled.
"""

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, Response

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.post("/start_finetuning")
async def start_finetuning(raw_request: Request):
    logger.info("Starting finetuning (opening FT admission)...")
    try:
        results = await engine_client(raw_request).collective_rpc(
            method="deltaserve_start_finetuning")
    except Exception as e:
        logger.exception("start_finetuning failed")
        return JSONResponse(status_code=500, content={"error": str(e)})
    ok = bool(results and all(results))
    logger.info("Finetuning %s.", "started" if ok else "NOT started")
    return JSONResponse(status_code=200 if ok else 409,
                        content={"started": ok})


@router.post("/stop_finetuning")
async def stop_finetuning(raw_request: Request):
    """Close FT admission. Idempotent — a stop on an already-stopped
    session returns ok=True. Buffer state and in-flight backwards are
    preserved so a follow-up POST /start_finetuning resumes cleanly."""
    logger.info("Stopping finetuning (closing FT admission)...")
    try:
        results = await engine_client(raw_request).collective_rpc(
            method="deltaserve_stop_finetuning")
    except Exception as e:
        logger.exception("stop_finetuning failed")
        return JSONResponse(status_code=500, content={"error": str(e)})
    ok = bool(results and all(results))
    logger.info("Finetuning %s.", "stopped" if ok else "NOT stopped")
    return JSONResponse(status_code=200 if ok else 409,
                        content={"stopped": ok})


@router.post("/save_estimator_state")
async def save_estimator_state(raw_request: Request):
    """Write the SLO estimator's state (fitted coefficients + the recorded
    step samples) to a JSON file NOW. Optional JSON body ``{"path": ...}``;
    default is ``slo.estimator_state_save_path``. A later launch restores
    it with ``slo.estimator_state_load_path`` — this is how a long trace
    is replayed as separate runs with one continuous estimator. Call it before
    stopping the server: the shutdown-time save is only best effort."""
    try:
        body = await raw_request.json()
    except Exception:
        body = {}
    path = (body or {}).get("path")
    try:
        result = await engine_client(raw_request).engine_core.call_utility_async(
            "deltaserve_save_estimator_state", path)
    except Exception as e:
        logger.exception("save_estimator_state failed")
        return JSONResponse(status_code=500, content={"error": str(e)})
    return JSONResponse(status_code=200 if result.get("saved") else 409,
                        content=result)


def _ft_config(raw_request: Request) -> dict:
    ac = getattr(raw_request.app.state.args, "additional_config", None) or {}
    if isinstance(ac, str):
        import json
        ac = json.loads(ac)
    return ac.get("deltaserve") or {}


@router.post("/save_adapter")
async def save_adapter(raw_request: Request):
    """Write the adapter being trained to a PEFT directory NOW (fp32 masters
    + the source ``adapter_config.json`` + ``trainer_state.json``). Optional
    JSON body: ``{"path": dir, "wait": true, "timeout_s": 300}``; ``path``
    defaults to ``<finetune_param.adapter_save_path>/manual-<time>``. The save runs
    in the backward child between cycles; with ``wait`` (default) the route
    returns once it is on disk (200, the trainer state) or failed (500), else
    202 right away. Loadable as a vLLM ``--lora-modules`` adapter, with PEFT,
    or as ``finetune_param.finetuning_lora_path`` to continue training."""
    import asyncio
    import os
    import time
    import uuid

    from deltaserve.adapter_io import read_save_result

    try:
        body = await raw_request.json()
    except Exception:
        body = {}
    body = body if isinstance(body, dict) else {}
    path = body.get("path")
    if not path:
        root = _ft_config(raw_request).get("adapter_save_path")
        if not root:
            return JSONResponse(status_code=400, content={
                "error": "no path given and finetune_param.adapter_save_path is not set"})
        path = os.path.join(root, time.strftime("manual-%Y%m%d-%H%M%S"))
    path = os.path.abspath(str(path))
    save_id = uuid.uuid4().hex
    try:
        results = await engine_client(raw_request).collective_rpc(
            method="deltaserve_save_adapter", args=(path, save_id))
    except Exception as e:
        logger.exception("save_adapter failed")
        return JSONResponse(status_code=500, content={"error": str(e)})
    if not (results and all(results)):
        return JSONResponse(status_code=409, content={
            "error": "no live backward child to save from", "path": path})
    if not body.get("wait", True):
        return JSONResponse(status_code=202, content={
            "scheduled": True, "path": path, "save_id": save_id})
    deadline = time.monotonic() + float(body.get("timeout_s", 300))
    while time.monotonic() < deadline:
        res = read_save_result(path, save_id)
        if res is not None:
            logger.info("save_adapter → %s: %s", path,
                        "saved" if res["saved"] else res.get("error"))
            return JSONResponse(status_code=200 if res["saved"] else 500,
                                content={"path": path, **res})
        await asyncio.sleep(0.1)
    return JSONResponse(status_code=202, content={
        "pending": True, "path": path, "save_id": save_id})


@router.get("/finetuning_status")
async def finetuning_status(raw_request: Request):
    """Admission + training progress: started / disabled (and why), epoch,
    backward cycles, tokens trained, last and recent loss, buffer fill."""
    try:
        result = await engine_client(raw_request).engine_core.call_utility_async(
            "deltaserve_finetuning_status")
    except Exception as e:
        logger.exception("finetuning_status failed")
        return JSONResponse(status_code=500, content={"error": str(e)})
    return JSONResponse(status_code=200, content=result)


@router.get("/system_metrics")
async def metrics_summary(raw_request: Request):
    """System metrics (telemetry.metrics_dir): the last full window and the
    one in progress — TTFT / TBT percentiles and SLO attainment, inference and
    FT throughput, GPU time by step kind, memory, run totals."""
    try:
        result = await engine_client(raw_request).engine_core.call_utility_async(
            "deltaserve_metrics_summary")
    except Exception as e:
        logger.exception("metrics_summary failed")
        return JSONResponse(status_code=500, content={"error": str(e)})
    return JSONResponse(status_code=200 if result.get("enabled") else 409,
                        content=result)


async def _telemetry(raw_request: Request, op: str, arg=None):
    return await engine_client(raw_request).engine_core.call_utility_async(
        "deltaserve_telemetry", op, arg)


@router.get("/telemetry/data")
async def telemetry_data(raw_request: Request, steps_since: int = 0,
                         series_since: float = 0.0):
    """[dserve-top] The live feed: steps with seq ≥ steps_since, per-second
    rows after series_since, current window, totals, memory, FT status.
    ``{"active": false}`` while telemetry is not recording."""
    try:
        res = await _telemetry(raw_request, "data", {
            "steps_since": steps_since, "series_since": series_since})
    except Exception as e:
        logger.exception("telemetry data failed")
        return JSONResponse(status_code=500, content={"error": str(e)})
    return JSONResponse(status_code=200, content=res)


async def _memory_snapshot(raw_request: Request, root: str) -> None:
    try:
        await engine_client(raw_request).collective_rpc(
            method="deltaserve_memory_snapshot", args=(root,))
    except Exception:
        logger.exception("memory snapshot failed")


@router.post("/telemetry/start")
async def telemetry_start(raw_request: Request):
    """[dserve-top] Start recording into ``{"dir": ...}`` unless telemetry is
    already on (config or an earlier start), and take each GPU's memory
    snapshot."""
    try:
        body = await raw_request.json()
    except Exception:
        body = {}
    if not (body or {}).get("dir"):
        return JSONResponse(status_code=400, content={"error": "dir is required"})
    import os
    body["dir"] = os.path.abspath(str(body["dir"]))
    try:
        res = await _telemetry(raw_request, "start", body)
    except Exception as e:
        logger.exception("telemetry start failed")
        return JSONResponse(status_code=500, content={"error": str(e)})
    await _memory_snapshot(raw_request, res.get("dir") or body["dir"])
    return JSONResponse(status_code=200, content=res)


@router.post("/telemetry/memory")
async def telemetry_memory(raw_request: Request):
    """[dserve-top] Refresh every GPU's memory_rank<r>.json."""
    res = await _telemetry(raw_request, "data", {"steps_since": 1 << 62,
                                                 "series_since": 1e18})
    if not res.get("active"):
        return JSONResponse(status_code=409, content={"error": "not recording"})
    await _memory_snapshot(raw_request, res["dir"])
    return JSONResponse(status_code=200, content={"dir": res["dir"]})


@router.post("/telemetry/stop")
async def telemetry_stop(raw_request: Request):
    """[dserve-top] Stop a recording started by /telemetry/start (one from
    the config keeps running)."""
    try:
        res = await _telemetry(raw_request, "stop")
    except Exception as e:
        logger.exception("telemetry stop failed")
        return JSONResponse(status_code=500, content={"error": str(e)})
    return JSONResponse(status_code=200, content=res)


_LOOPBACK = ("127.0.0.1", "::1", "localhost")


@router.post("/shutdown")
async def shutdown(raw_request: Request):
    """[dserve-top] Stop the server as Ctrl-C would: SIGINT to this API
    server process, so vLLM shuts the engine down normally (estimator state and
    telemetry are saved on the way, as on Ctrl-C). Loopback clients only."""
    import asyncio
    import os
    import signal

    peer = raw_request.client.host if raw_request.client else ""
    if peer not in _LOOPBACK and not peer.startswith("127."):
        return JSONResponse(status_code=403, content={
            "error": f"shutdown is accepted only from the server's own machine (not {peer})"})
    logger.warning("POST /shutdown from %s: stopping the server", peer)
    # After the response is on its way.
    asyncio.get_running_loop().call_later(0.3, os.kill, os.getpid(), signal.SIGINT)
    return JSONResponse(status_code=200, content={"status": "stopping"})


@router.post("/ft_jobs")
async def submit_ft_job(raw_request: Request):
    """Queue a finetuning job; jobs run one after another. JSON body with the
    keys of the ``ft_job`` YAML section: ``adapter_path`` and ``data_path``
    (required), ``name``, ``stages``, ``num_epochs``, ``learning_rate``,
    ``weight_decay``, ``gamma``, ``auto_routing``, ``save_path``. The adapter
    must have the shape of the one the server was launched with (same LoRA
    tensors, rank, alpha/r). Watch GET /ft_jobs."""
    try:
        body = await raw_request.json()
    except Exception:
        body = None
    if not isinstance(body, dict):
        return JSONResponse(status_code=400,
                            content={"error": "JSON object body required"})
    try:
        result = await engine_client(raw_request).engine_core.call_utility_async(
            "deltaserve_submit_ft_job", body)
    except Exception as e:
        logger.exception("submit_ft_job failed")
        return JSONResponse(status_code=500, content={"error": str(e)})
    return JSONResponse(status_code=400 if "error" in result else 202,
                        content=result)


@router.get("/ft_jobs")
async def ft_jobs(raw_request: Request):
    """The finetuning jobs of this server: the running one with the state of
    each of its stages, the queued ones and the finished ones."""
    try:
        result = await engine_client(raw_request).engine_core.call_utility_async(
            "deltaserve_ft_jobs")
    except Exception as e:
        logger.exception("ft_jobs failed")
        return JSONResponse(status_code=500, content={"error": str(e)})
    return JSONResponse(status_code=200, content=result)


routing_router = APIRouter()


@routing_router.get("/adapters")
async def adapters(raw_request: Request):
    """The adapter pool: every auto-routable adapter with its tier, hit count
    and state, plus the jobs that are adding adapters."""
    try:
        result = await engine_client(raw_request).engine_core.call_utility_async(
            "deltaserve_adapters")
    except Exception as e:
        logger.exception("adapters failed")
        return JSONResponse(status_code=500, content={"error": str(e)})
    return JSONResponse(status_code=409 if "error" in result else 200,
                        content=result)


def available_adapters(directory: str, route_model: str = "auto") -> dict:
    """What a client can ask for right now, from the pool's registry file."""
    from deltaserve.adapter_routing.pool import read_registry

    reg = read_registry(directory) or {"adapters": [], "version": 0}
    return {
        "route_model": route_model, "version": reg.get("version", 0),
        "adapters": [{"name": a["name"], "level": a.get("level"),
                      # a candidate for model=<route_model>; every listed
                      # adapter can be asked for by name
                      "routable": a.get("level") == "active"
                      and bool(a.get("enabled"))}
                     for a in reg["adapters"]]}


@routing_router.get("/adapters/available")
async def adapters_available(raw_request: Request):
    """The adapters a client can use now: by name (``"model": "<name>"``), and
    — those marked routable — through ``"model": "<route_model>"``. An adapter
    appears here the moment its finetuning job has saved it and the pool has
    admitted it. Answered by the API server from the pool's registry file, no
    engine call: cheap enough to poll."""
    cfg = getattr(raw_request.app.state, "deltaserve_adapter_routing", None)
    if cfg is None:
        return JSONResponse(status_code=409,
                            content={"error": "adapter routing is off"})
    from deltaserve.adapter_routing.stages import index_dir

    try:
        return JSONResponse(status_code=200, content=available_adapters(
            index_dir(cfg), cfg.route_model_name))
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": str(e)})


@routing_router.post("/adapters")
async def add_adapter(raw_request: Request):
    """Add an existing adapter to the pool without training. JSON body:
    ``{"path": ..., "name"?: ..., "data_path"?: ..., "centroid_path"?: ...,
    "prewarm"?: bool}``. With ``data_path`` its centroid is built in the
    background (a job with stages [init, indexing, ready]); it becomes
    auto-routable when that job is done — watch GET /adapters."""
    try:
        body = await raw_request.json()
    except Exception:
        body = None
    if not isinstance(body, dict) or not body.get("path"):
        return JSONResponse(status_code=400,
                            content={"error": "JSON body with 'path' required"})
    try:
        result = await engine_client(raw_request).engine_core.call_utility_async(
            "deltaserve_add_adapter", body)
    except Exception as e:
        logger.exception("add_adapter failed")
        return JSONResponse(status_code=500, content={"error": str(e)})
    return JSONResponse(status_code=400 if "error" in result else 202,
                        content=result)


async def _adapter_op(raw_request: Request, name: str, op: str):
    try:
        result = await engine_client(raw_request).engine_core.call_utility_async(
            "deltaserve_adapter_op", name, op)
    except Exception as e:
        logger.exception("adapter %s failed", op)
        return JSONResponse(status_code=500, content={"error": str(e)})
    return JSONResponse(status_code=400 if "error" in result else 200,
                        content=result)


@routing_router.post("/adapters/{name}/{op}")
async def adapter_op(name: str, op: str, raw_request: Request):
    """``retire`` (stop selecting it, free its memory; still servable by name),
    ``reactivate`` (back through admission), ``pin`` / ``unpin`` (never / again
    auto-retired)."""
    if op not in ("retire", "reactivate", "pin", "unpin"):
        return JSONResponse(status_code=404,
                            content={"error": f"unknown operation '{op}'"})
    return await _adapter_op(raw_request, name, op)


@routing_router.delete("/adapters/{name}")
async def remove_adapter(name: str, raw_request: Request):
    """Take an adapter out of the pool entirely."""
    return await _adapter_op(raw_request, name, "remove")


def attach_router(app: FastAPI):
    """Called after vLLM registers its own routers (plugin wrapper)."""
    ac = getattr(app.state.args, "additional_config", None) or {}
    if isinstance(ac, str):
        import json
        ac = json.loads(ac)
    if (ac.get("deltaserve") or {}).get("enable_finetuning", False):
        app.include_router(router)
    routing = ac.get("deltaserve_adapter_routing") or {}
    if routing.get("enable_adapter_routing", False):
        # The only place the API server reaches into deltaserve.adapter_routing.
        from deltaserve.adapter_routing.middleware import install
        from deltaserve.config import AdapterRoutingConfig

        cfg = AdapterRoutingConfig(**routing)
        # Route middleware + the pool's LoRA resolver (pool adapters become
        # servable by name) + cache invalidation.
        install(app, cfg)
        app.state.deltaserve_adapter_routing = cfg
        app.include_router(routing_router)
        logger.info("Adapter routing on: requests with model=%r are routed.",
                    routing.get("route_model_name", "auto"))
    elif "deltaserve" in ac or routing:
        # A DeltaServe server with adapter routing off still answers an
        # auto-route request: base model + "routing is not enabled" headers.
        # No routing component is loaded for this.
        from deltaserve.route_middleware import install_middleware

        install_middleware(app, None,
                           routing.get("route_model_name") or "auto")
