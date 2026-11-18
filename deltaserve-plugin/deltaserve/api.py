# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""HTTP control for co-serving finetuning admission.

POST /start_finetuning opens FT admission (when the server was launched with
finetune.start_on_launch=false). It runs `deltaserve_start_finetuning` on the
worker via collective_rpc; under single-GPU/uniproc the worker shares the
process with the scheduler + coordinator, so the flag flip is seen immediately.

POST /stop_finetuning closes FT admission again — a stop / start cycle resumes
from the same buffer + counter state (the coordinator's ``stop_finetuning``
intentionally preserves in-flight work). Useful for pausing FT during a
benchmark phase or to bracket a deliberate idle window.

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
    default is ``finetune.estimator_state_save_path``. A later launch restores
    it with ``finetune.estimator_state_load_path`` — this is how a long trace
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
