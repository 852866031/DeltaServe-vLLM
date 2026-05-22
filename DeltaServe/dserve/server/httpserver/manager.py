import json
import zmq
import zmq.asyncio
import asyncio
import uvloop
from typing import Union
import time

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from ..tokenizer import get_tokenizer
from ..io_struct import BatchStrOut, AbortReq, BatchAbortReq, FinetuneReq, FinetuneStatusReq

class HttpServerManager:
    def __init__(
        self,
        model_weightdir,
        tokenizor_mode,
        router_port,
        httpserver_port,
        total_token_num,
        max_req_input_len,
        max_req_total_len,
        trust_remote_code,
        dummy=False,
    ):
        context = zmq.asyncio.Context(3)
        self.send_to_router = context.socket(zmq.PUSH)
        self.send_to_router.connect(f"tcp://127.0.0.1:{router_port}")

        self.recv_from_detokenization = context.socket(zmq.PULL)
        self.recv_from_detokenization.bind(f"tcp://127.0.0.1:{httpserver_port}")

        try:
            self.tokenizer = get_tokenizer(model_weightdir, tokenizor_mode, trust_remote_code=trust_remote_code)
        except:
            if dummy:
                self.tokenizer = get_tokenizer("huggyllama/llama-7b", tokenizor_mode)

        self.req_id_to_out_inf = {}  # value type (out_str, metadata, finished, event)

        self.total_token_num = total_token_num
        print("httpserver: total_token_num", total_token_num)
        self.max_req_input_len = max_req_input_len
        self.max_req_total_len = max_req_total_len

        self._arrival_count = 0          # how many requests have arrived (since process start)
        self._t_begin = None        # start time of current 1s window
        self._win_count = 0              # arrivals in (t_begin, t_begin+1.0]
        self.finetuning_finished = False

    async def start_finetuning(self):
        print("httpserver: sending start finetuning request to router")
        self.send_to_router.send_pyobj(FinetuneReq())
        return
    
    async def exit_finetuning(self):
        self.send_to_router.send_pyobj(FinetuneReq(exit_finetuning=True))
        return

    def _record_arrival(self):
        """
        Rolling 1s windows:
        - First, wait until the 6th arrival; that arrival becomes t_begin.
        - Count how many further arrivals occur in (t_begin, t_begin+1.0].
        - On the first arrival AFTER that 1s window, print the count,
            then set THIS arrival as the new t_begin and start a fresh window.
        """
        now = time.time()
        self._arrival_count += 1

        # Initialize first window at the 6th arrival
        if self._t_begin is None:
            if self._arrival_count == 6:
                self._t_begin = now
                self._win_count = 0
            return

        # We have an active window
        dt = now - self._t_begin
        if dt <= 1.0:
            # Count arrivals strictly after the begin
            self._win_count += 1
        else:
            # Window elapsed: report and start a new window with THIS arrival as begin
            self._t_begin = now
            self._win_count = 0  # current arrival is the new begin, not counted
    
    async def check_finetune_status_once(self) -> bool:
        self.send_to_router.send_pyobj(FinetuneStatusReq())
        await asyncio.sleep(1)
        if self.finetuning_finished:
            return True
        return False


    async def generate(self, adapter_dir, prompt, sampling_params, request_id):
        loop = asyncio.get_running_loop()
        prompt_ids = await loop.run_in_executor(None, self.tokenizer.encode, prompt)
        prompt_tokens = len(prompt_ids)
        if prompt_tokens > self.max_req_input_len:
            raise ValueError(
                f"the input prompt token len {prompt_tokens} is too long > {self.max_req_input_len}"
            )
        req_total_len = prompt_tokens + sampling_params.max_new_tokens
        if req_total_len > self.max_req_total_len:
            raise ValueError(
                f"the req token total len (input len + output len) is too long > max_req_total_len:{self.max_req_total_len}"
            )
        if req_total_len + 1 > self.total_token_num:
            print(f"req_total_len:{req_total_len}, max_total_token_num:{self.total_token_num}")
            print("prompt_tokens:", prompt_tokens)
            print("sampling_params.max_new_tokens:", sampling_params.max_new_tokens)
            raise ValueError(
                f"the req token total len {req_total_len} + 1 (input len + output len + 1) is too long > max_total_token_num:{self.total_token_num}"
            )
        
        sampling_params.stop_sentences_to_token_ids(self.tokenizer)
        self.send_to_router.send_pyobj((adapter_dir, prompt_ids, sampling_params, request_id))
        event = asyncio.Event()
        self.req_id_to_out_inf[request_id] = ("", {}, False, event)
        while True:
            try:
                await asyncio.wait_for(event.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass
            event.clear()
            # request_id is aborted by the backend system for traffic control
            if request_id not in self.req_id_to_out_inf:
                yield "", {}, -1, False
                break
            out = self.req_id_to_out_inf[request_id]
            if len(out) != 5:
                out_str, metadata, finished, event = out
                perf_metrics = {}
            else:
                out_str, metadata, finished, event, perf_metrics = out
            if len(metadata) != 0:
                self.req_id_to_out_inf[request_id] = ("", {}, finished, event)
                metadata["prompt_tokens"] = prompt_tokens
                yield out_str, metadata, finished, perf_metrics
            if finished:
                try:
                    del self.req_id_to_out_inf[request_id]
                except:
                    pass
                break
        return

    async def abort(self, request_id):
        abort_req = AbortReq(req_id=request_id)
        self.send_to_router.send_pyobj(abort_req)
        try:
            del self.req_id_to_out_inf[request_id]
        except:
            pass
        return

    async def handle_loop(self):
        while True:
            recv_ans:Union(BatchStrOut, BatchAbortReq, FinetuneStatusReq) = await self.recv_from_detokenization.recv_pyobj()
            assert isinstance(recv_ans, (BatchStrOut, BatchAbortReq, FinetuneStatusReq)), f"error recv type {type(recv_ans)}"
            if isinstance(recv_ans, BatchStrOut):
                for req_id, text, metadata, finished, abort, perf_metrics in recv_ans.reqs_infs:
                    try:
                        if not abort:
                            _, _, _, event = self.req_id_to_out_inf[req_id]
                            self.req_id_to_out_inf[req_id] = (
                                text,
                                metadata,
                                finished,
                                event,
                                perf_metrics
                            )
                            event.set()
                        else:
                            del self.req_id_to_out_inf[req_id]
                    except:
                        pass
            elif isinstance(recv_ans, BatchAbortReq):
                print("httpserver: received abort from detokenization")
                print("abort reqs:", recv_ans.reqs)
                for req_id in recv_ans.reqs:
                    try:
                        del self.req_id_to_out_inf[req_id]
                    except:
                        pass
            elif isinstance(recv_ans, FinetuneStatusReq):
                self.finetuning_finished = recv_ans.finished

        return
