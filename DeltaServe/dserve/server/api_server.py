# Adapted from vllm/entrypoints/api_server.py
# of the vllm-project/vllm GitHub repository.
#
# Copyright 2023 ModelTC Team
# Copyright 2023 vLLM Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import time
import torch
import uvloop
import sys
from pprint import pprint
from .build_prompt import build_prompt
from typing import List, Dict, Any, Optional, Tuple

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import argparse
import json
from http import HTTPStatus
import uuid
import multiprocessing as mp
from typing import AsyncGenerator

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import Response, StreamingResponse, JSONResponse
import uvicorn
from .sampling_params import SamplingParams
from .httpserver.manager import HttpServerManager
from .detokenization.manager import start_detokenization_process
from .router.manager import start_router_process

from dserve.utils.net_utils import alloc_can_use_network_port
from .config import ServerConfig, load_config, apply_overrides
from .api_models import (
    ChatCompletionRequest,
    UsageInfo,
    ChatMessage,
    ChatCompletionResponseChoice,
    ChatCompletionResponse,
    DeltaMessage,
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
)

from dserve.mprophet.measure import ModelProphet
from dserve.mprophet.lora_stats import LoRAProphet


GB = 1024 ** 3
MB = 1024 ** 2

TIMEOUT_KEEP_ALIVE = 5  # seconds.

app = FastAPI()

isFirst = True
firstChecking = True

from pprint import pprint

def create_error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
    return JSONResponse({"message": message}, status_code=status_code.value)


@app.get("/healthz")
@app.get("/health")
def healthcheck():
    return "OK"

@app.post("/start_finetuning")
async def finetuning_status(request: Request) -> Response:
    print("api_server.py: start_finetuning called")
    await httpserver_manager.start_finetuning()
    print("api_server.py: start_finetuning finished")
    return Response(
        content=json.dumps({"message": "Finetuning started"}),
        status_code=200,
        media_type="application/json"
    )

@app.post("/exit_finetuning")
async def exit_finetuning(request: Request) -> Response:
    await httpserver_manager.exit_finetuning()
    return Response(
        content=json.dumps({"message": "Finetuning exited"}),
        status_code=200,
        media_type="application/json"
    )
    
    
@app.post("/finetuning_status")
async def finetuning_status(request: Request) -> Response:
    finished = await httpserver_manager.check_finetune_status_once()
    if finished:
        return Response(
            content=json.dumps({"finished": "true"}),
            status_code=200,
            media_type="application/json"
        )
    else:
        return Response(
            content=json.dumps({"finished": "false"}),
            status_code=200,
            media_type="application/json"
        )
@app.post("/start_finetuning")
async def finetuning_status(request: Request) -> Response:
    await httpserver_manager.start_finetuning()
    return Response(
        content=json.dumps({"message": "Finetuning started"}),
        status_code=200,
        media_type="application/json"
    )


@app.post("/generate")
async def generate(request: Request) -> Response:
    global isFirst
    if isFirst:
        loop = asyncio.get_event_loop()
        loop.create_task(httpserver_manager.handle_loop())
        isFirst = False

    request_dict = await request.json()
    adapter_dir = request_dict["lora_dir"] if "lora_dir" in request_dict else None
    prompt = request_dict.pop("inputs")
    sample_params_dict = request_dict["parameters"]
    return_details = sample_params_dict.pop("return_details", False)
    sampling_params = SamplingParams(**sample_params_dict)
    sampling_params.verify()

    if "req_id" in request_dict:
        request_id = request_dict["req_id"]
    else:
        request_id = uuid.uuid4().hex
    results_generator = httpserver_manager.generate(adapter_dir, prompt, sampling_params, request_id)
    # Non-streaming case
    final_output = []
    count_output_tokens = 0
    tokens = []
    perf_metrics = None
    async for request_output, metadata, finished, perf_metrics in results_generator:
        count_output_tokens += 1
        if finished == -1:
            return Response(status_code=499)
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await httpserver_manager.abort(request_id)
            return Response(status_code=499)
        final_output.append(request_output)
        if return_details:
            metadata["text"] = request_output
            tokens.append(metadata)

    assert final_output is not None
    ret = {
        "generated_text": ["".join(final_output)],
        "count_output_tokens": count_output_tokens,
        "ttft": perf_metrics[0] if perf_metrics is not None else None,
        "avg_tbt": perf_metrics[1] if perf_metrics is not None else None,
        "worst_tbt": perf_metrics[2] if perf_metrics is not None else None,
    }
    if return_details:
        ret["tokens"] = tokens
    return Response(content=json.dumps(ret, ensure_ascii=False).encode("utf-8"))


@app.post("/generate_stream")
async def generate_stream(request: Request) -> Response:
    print("api_server.py: generate_stream called, inference received")
    global isFirst
    if isFirst:
        loop = asyncio.get_event_loop()
        loop.create_task(httpserver_manager.handle_loop())
        isFirst = False

    request_dict = await request.json()
    print(request_dict)
    adapter_dir = request_dict["lora_dir"] if "lora_dir" in request_dict else None
    prompt = request_dict.pop("inputs")
    sample_params_dict = request_dict["parameters"]
    pprint(sample_params_dict)
    return_details = sample_params_dict.pop("return_details", False)
    sampling_params = SamplingParams(**sample_params_dict)
    sampling_params.verify()

    if "req_id" in request_dict:
        request_id = request_dict["req_id"]
    else:
        request_id = uuid.uuid4().hex
    results_generator = httpserver_manager.generate(adapter_dir, prompt, sampling_params, request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output, metadata, finished, _perf in results_generator:
            ret = {
                "token": {
                    "id": metadata.get("id", None),
                    "text": request_output,
                    "logprob": metadata.get("logprob", None),
                    "special": False
                },
                "generated_text": None,
                "finished": finished,
                "details": None
            }

            yield ("data:" + json.dumps(ret, ensure_ascii=False) + f"\n\n").encode(
                "utf-8"
            )

    async def abort_request() -> None:
        await httpserver_manager.abort(request_id)

    background_tasks = BackgroundTasks()
    # Abort the request if the client disconnects.
    background_tasks.add_task(abort_request)

    return StreamingResponse(
        stream_results(), media_type="text/event-stream", background=background_tasks
    )

@app.post("/generate_batch_stream")
async def generate_batch_stream(raw_request: Request) -> Response:
    """
    Accepts a batch of requests and streams *one* JSON payload per finished item
    using Server-Sent Events (SSE). Items are emitted in completion order.
    """
    # Ensure background handle loop is running once
    global isFirst
    if isFirst:
        loop = asyncio.get_event_loop()
        loop.create_task(httpserver_manager.handle_loop())
        isFirst = False

    body = await raw_request.json()
    items: List[Dict[str, Any]] = body.get("items", [])
    if not isinstance(items, list) or len(items) == 0:
        return create_error_response(HTTPStatus.BAD_REQUEST, "items must be a non-empty list")

    # Pre-parse & verify inputs; build immutable plan to avoid repeated dict lookups.
    parsed: List[Tuple[str, Optional[str], str, SamplingParams, bool, int]] = []
    # (req_id, lora_dir, prompt, sampling_params, return_details, index)
    for idx, item in enumerate(items):
        try:
            lora_dir = item.get("lora_dir", None)
            prompt = item["inputs"]
            params_dict = dict(item["parameters"])  # shallow copy; we'll pop
            return_details = bool(params_dict.pop("return_details", False))
            sp = SamplingParams(**params_dict)
            sp.verify()
            req_id = item.get("req_id", uuid.uuid4().hex)
            parsed.append((req_id, lora_dir, prompt, sp, return_details, idx))
        except Exception as e:
            return create_error_response(
                HTTPStatus.BAD_REQUEST,
                f"Invalid item at index {idx}: {e}"
            )

    # Helper: run one request to completion and return final payload
    async def run_one(req_id: str,
                      lora_dir: Optional[str],
                      prompt: str,
                      sp: SamplingParams,
                      return_details: bool,
                      index: int) -> Dict[str, Any]:
        gen = httpserver_manager.generate(lora_dir, prompt, sp, req_id)

        final_chunks: List[str] = []
        count_output_tokens = 0
        tokens: List[Dict[str, Any]] = []

        async for request_output, metadata, finished, _perf in gen:
            count_output_tokens += 1
            final_chunks.append(request_output)
            if return_details:
                md = dict(metadata) if metadata is not None else {}
                md["text"] = request_output
                tokens.append(md)

        result = {
            "index": index,
            "req_id": req_id,
            "generated_text": "".join(final_chunks),
            "count_output_tokens": count_output_tokens,
        }
        if return_details:
            result["tokens"] = tokens
        return result

    # For abort-on-disconnect, track pending req_ids
    pending_req_ids = {req_id for (req_id, *_rest) in parsed}

    async def abort_all_pending():
        # Called when client disconnects or stream ends unexpectedly
        await asyncio.gather(*(httpserver_manager.abort(rid) for rid in list(pending_req_ids)), return_exceptions=True)

    # Stream results as each task finishes (unordered by design)
    async def stream_results() -> AsyncGenerator[bytes, None]:
        tasks = [asyncio.create_task(run_one(*tup)) for tup in parsed]
        try:
            # While there are tasks, wait for whichever finishes first
            while tasks:
                # If client disconnected, abort immediately
                if await raw_request.is_disconnected():
                    await abort_all_pending()
                    return

                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

                # Emit all completed results
                for t in done:
                    try:
                        payload = t.result()
                        # Remove from pending set for abort
                        pending_req_ids.discard(payload["req_id"])
                        # SSE event
                        yield ("data:" + json.dumps(payload, ensure_ascii=False) + "\n\n").encode("utf-8")
                    except Exception as e:
                        # Emit an error event for this index (non-fatal for others)
                        err_payload = {"error": str(e)}
                        yield ("data:" + json.dumps(err_payload, ensure_ascii=False) + "\n\n").encode("utf-8")

                tasks = list(pending)

            # Send a final terminator event for clients that expect it (optional)
            yield b"data:{\"finished\": true}\n\n"

        finally:
            # Safety net: if anything breaks mid-stream, abort leftovers
            if pending_req_ids:
                await abort_all_pending()

    background_tasks = BackgroundTasks()
    # Best-effort cleanup on disconnect (FastAPI won't call this automatically on SSE close, so we also check in loop)
    background_tasks.add_task(lambda: None)

    return StreamingResponse(
        stream_results(),
        media_type="text/event-stream",
        background=background_tasks
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest, raw_request: Request
) -> Response:
    global isFirst
    if isFirst:
        loop = asyncio.get_event_loop()
        loop.create_task(httpserver_manager.handle_loop())
        isFirst = False

    if request.logit_bias is not None:
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            "The logit_bias parameter is not currently supported",
        )

    if request.n > 1:
        return create_error_response(
            HTTPStatus.BAD_REQUEST, "The n parameter currently only supports 1"
        )

    if request.function_call != "none":
        return create_error_response(
            HTTPStatus.BAD_REQUEST, "The function call feature is not supported"
        )

    created_time = int(time.time())
    prompt = await build_prompt(request)
    sampling_params = SamplingParams(
        do_sample=request.do_sample,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        ignore_eos=request.ignore_eos,
        max_new_tokens=request.max_tokens,
        stop_sequences=request.stop
    )
    sampling_params.verify()

    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    results_generator = httpserver_manager.generate(prompt, sampling_params, request_id)

    # Non-streaming case
    if not request.stream:
        final_output = []
        prompt_tokens = -1
        completion_tokens = 0
        async for request_output, metadata in results_generator:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await httpserver_manager.abort(request_id)
                return Response(status_code=499)
            completion_tokens += 1
            if prompt_tokens == -1:
                prompt_tokens = metadata["prompt_tokens"]
            final_output.append(request_output)

        usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
        chat_message = ChatMessage(role="assistant", content="".join(final_output))
        choice = ChatCompletionResponseChoice(index=0, message=chat_message)
        resp = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=request.model,
            choices=[choice],
            usage=usage
        )
        return resp

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output, metadata in results_generator:
            delta_message = DeltaMessage(role="assistant", content=request_output)

            stream_choice = ChatCompletionStreamResponseChoice(
                index=0, delta=delta_message
            )

            stream_resp = ChatCompletionStreamResponse(
                id=request_id,
                created=created_time,
                model=request.model,
                choices=[stream_choice],
            )
            yield ("data: " + stream_resp.json(ensure_ascii=False) + f"\n\n").encode("utf-8")

    async def abort_request() -> None:
        await httpserver_manager.abort(request_id)

    background_tasks = BackgroundTasks()
    # Abort the request if the client disconnects.
    background_tasks.add_task(abort_request)

    return StreamingResponse(
        stream_results(), media_type="text/event-stream", background=background_tasks
    )


def print_mem_stats(args):
    model_dir = args.model_dir
    model_name = args.model_dir.split("/")[-1]
    try:
        fake_model = ModelProphet(model_name, model_dir=model_dir)
    except:
        fake_model = ModelProphet(model_name)
    model_size = fake_model.get_model_size()
    print(f"{model_name}: {model_size / GB:.2f} GB")
    peak_working_memory = fake_model.get_peak_working_memory(
            bs=20, context_len=512, tiling_dim=512)
    print(f"peak working mem for (bs=20, seqlen=512): {peak_working_memory / GB:.2f} GB")
    peak_working_memory = fake_model.get_peak_working_memory(
            bs=100, context_len=512, tiling_dim=512)
    print(f"peak working mem for (bs=100, seqlen=512): {peak_working_memory / GB:.2f} GB")
 
    tot_lora_size = 0
    for lora_dir in args.lora_dirs:
        lora_name = lora_dir.split("/")[-1]
        if args.dummy:
            fake_model = LoRAProphet(lora_name, model_name)
            try:
                fake_model = LoRAProphet(lora_name, model_name)
            except NotImplementedError as e:
                fake_model = LoRAProphet(lora_name, model_name,
                                         adapter_dir=lora_dir,
                                         base_model_dir=model_dir)
        else:
            fake_model = LoRAProphet(lora_name, model_name,
                                     adapter_dir=lora_dir,
                                     base_model_dir=model_dir)
        lora_size = fake_model.get_adapter_size()
        tot_lora_size += lora_size
        # print(f"{lora_name}, {base_name}: {lora_size / GB:.3f} GB")
    print(f"all adapters ({len(args.lora_dirs)}) estimated size: {tot_lora_size / GB:.2f} GB")
    print(f"avg adapter estimated size: {tot_lora_size / len(args.lora_dirs) / MB:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="DeltaServe API server. Configured via a single "
                    "serving_config.yaml; see eval/llama3/config/serving_config.yaml.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to a serving_config.yaml (the single source of truth).")
    parser.add_argument("--override", type=str, default=[], action="append",
                        help="Dotted-path override, e.g. "
                             "--override serving.max_total_token_num=30000. "
                             "May be passed multiple times.")
    # Per-invocation shortcuts. These are the two knobs that vary essentially
    # every launch (port collisions, multi-GPU rank assignment) so they get
    # direct flags. Anything else uses --override.
    parser.add_argument("--port", type=int, default=None,
                        help="Override server.port from the YAML.")
    parser.add_argument("--rank_id", type=int, default=None,
                        help="Override server.rank_id from the YAML.")

    args = parser.parse_args()

    # ── Build the single ServerConfig that drives everything downstream ──
    cfg = load_config(args.config)
    apply_overrides(cfg, args.override)
    if args.port is not None:
        cfg.server.port = args.port
    if args.rank_id is not None:
        cfg.server.rank_id = args.rank_id

    # Post-cfg setup that applies regardless of source.
    if cfg.finetune.enabled:
        # Append the finetune adapter to the loaded LoRA list.
        if cfg.finetune.lora_path \
                and cfg.finetune.lora_path not in cfg.lora.adapter_dirs:
            cfg.lora.adapter_dirs.append(cfg.finetune.lora_path)
        # Finetuning runs on the dserve scheduler.
        cfg.scheduler.name = "dserve"

    assert cfg.serving.max_req_input_len < cfg.serving.max_req_total_len
    if cfg.serving.batch_max_tokens is None:
        batch_max_tokens = int(cfg.serving.max_total_token_num / 5)
        batch_max_tokens = max(batch_max_tokens, cfg.serving.max_req_total_len)
        cfg.serving.batch_max_tokens = batch_max_tokens
        print(f"batch_max_tokens auto-derived: {batch_max_tokens} "
              f"(max(1/5*max_total_token_num={cfg.serving.max_total_token_num // 5}, "
              f"max_req_total_len={cfg.serving.max_req_total_len}))")
    else:
        assert cfg.serving.batch_max_tokens >= cfg.serving.max_req_total_len, \
            "batch_max_tokens must be >= max_req_total_len"

    args.cfg = cfg
    # Mirror a few cfg fields back into args so downstream code that still
    # reads args.X (e.g. detokenization manager) keeps working without a
    # second migration pass in this PR.
    args.model_dir = cfg.model.dir
    args.tokenizer_mode = cfg.model.tokenizer_mode
    args.trust_remote_code = cfg.model.trust_remote_code
    args.dummy = cfg.debug.dummy
    args.lora_dirs = list(cfg.lora.adapter_dirs)
    args.mode = list(cfg.model.mode)

    print("Resolved ServerConfig:")
    print(cfg.pretty())

    can_use_ports = alloc_can_use_network_port(
        num=3 + cfg.server.tp, used_nccl_port=cfg.server.nccl_port
    )
    router_port, detokenization_port, httpserver_port = can_use_ports[0:3]
    router_port += cfg.server.rank_id * 10
    detokenization_port += cfg.server.rank_id * 10
    httpserver_port += cfg.server.rank_id * 10
    print(f"router_port: {router_port}, detokenization_port: {detokenization_port}, httpserver_port: {httpserver_port}")
    model_rpc_ports = can_use_ports[3:]
    global httpserver_manager
    httpserver_manager = HttpServerManager(
        cfg.model.dir,
        cfg.model.tokenizer_mode,
        router_port=router_port,
        httpserver_port=httpserver_port,
        total_token_num=cfg.serving.max_total_token_num,
        max_req_input_len=cfg.serving.max_req_input_len,
        max_req_total_len=cfg.serving.max_req_total_len,
        trust_remote_code=cfg.model.trust_remote_code,
        dummy=cfg.debug.dummy,
    )
    if cfg.finetune.enabled and not cfg.finetune.start_on_launch:
        httpserver_manager.finetuning_finished = True
    pipe_router_reader, pipe_router_writer = mp.Pipe(duplex=False)
    pipe_detoken_reader, pipe_detoken_writer = mp.Pipe(duplex=False)
    proc_router = mp.Process(
        target=start_router_process,
        args=(
            args,
            router_port,
            detokenization_port,
            model_rpc_ports,
            args.mode,
            pipe_router_writer,
        ),
    )
    proc_router.start()
    proc_detoken = mp.Process(
        target=start_detokenization_process,
        args=(
            args,
            detokenization_port,
            httpserver_port,
            pipe_detoken_writer,
            args.trust_remote_code,
        ),
    )
    proc_detoken.start()

    # wait load model ready
    router_init_state = pipe_router_reader.recv()
    detoken_init_state = pipe_detoken_reader.recv()

    if router_init_state != "init ok" or detoken_init_state != "init ok":
        proc_router.kill()
        proc_detoken.kill()
        print(
            "router init state:",
            router_init_state,
            "detoken init state:",
            detoken_init_state,
        )
        sys.exit(1)

    assert proc_router.is_alive() and proc_detoken.is_alive()

    print_mem_stats(args)

    uvicorn.run(
        app,
        host=cfg.server.host,
        port=cfg.server.port,
        log_level="debug",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        loop="uvloop",
        access_log=False
    )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')      # this code will not be ok for settings to fork to subprocess
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
