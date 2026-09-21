# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""[DeltaServe] HTTP control for co-serving finetuning admission.

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


def attach_router(app: FastAPI):
    ft_cfg = getattr(app.state.args, "finetune_config", None)
    if ft_cfg is not None and getattr(ft_cfg, "enable_finetuning", False):
        app.include_router(router)
