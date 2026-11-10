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
/system_metrics the system metrics (telemetry.metrics_dir).

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


def attach_router(app: FastAPI):
    """Called after vLLM registers its own routers (plugin wrapper)."""
    ac = getattr(app.state.args, "additional_config", None) or {}
    if isinstance(ac, str):
        import json
        ac = json.loads(ac)
    if (ac.get("deltaserve") or {}).get("enable_finetuning", False):
        app.include_router(router)
