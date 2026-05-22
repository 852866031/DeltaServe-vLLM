import asyncio
from multiprocessing.dummy import Pipe
import multiprocessing as mp
import threading

import numpy as np
import rpyc
from dserve.models.llama.SFT_service import LlamaSFTBackwardService
from dserve.models.llama3.model import Llama3TpPartModel
import torch
import torch.distributed as dist
import traceback
import time
import os
import json
from collections import defaultdict
from multiprocessing import Process, Pipe

from datetime import timedelta
from tqdm import tqdm
from typing import Dict, List, Tuple
from rpyc.utils.classic import obtain

from transformers.configuration_utils import PretrainedConfig
from dserve.server.router.model_infer.infer_batch import InferBatch

from dserve.common.configs.config import set_active_config
from dserve.models.llama.model import LlamaTpPartModel
from dserve.models.llama2.model import Llama2TpPartModel
from dserve.models.peft.lora_adapter import LoraTpPartAdapter
from dserve.models.peft.lora_unordered_batch_mixed import LoraUnorderedBatchMixed
from dserve.server.router.model_infer.infer_adapter import InferAdapter
from dserve.server.router.model_infer.infer_adapter_alt import InferAdapterAlt
from dserve.server.router.model_infer.naive_infer_adapter import NaiveInferAdapter
from dserve.utils.infer_utils import set_random_seed
from dserve.utils.infer_utils import calculate_time, mark_start, mark_end
from dserve.utils.model_utils import get_model_config
from .post_process import sample

from ..mixed_req_queue import rprint
from pprint import pprint

from enum import Enum

class BackwardResumePoint(Enum):
    BEFORE_OPTIMIZER = 0
    OPTIMIZER = 1

class ModelRpcServer(rpyc.Service):

    def exposed_init_model(self, rank_id, world_size, weight_dir, adapter_dirs,
                           max_total_token_num, load_way, mode, input_params,
			   prefetch_stream, 
               half_model=False, mem_manager_log_path=None, 
               gpu_profiler=None, unified_mem_manager_max_size=0, use_rank_id=0):
        if world_size != 1:
            trans_list = [obtain(e) for e in (rank_id, world_size, weight_dir, adapter_dirs,
                                              max_total_token_num, load_way, mode)]
            rank_id, world_size, weight_dir, adapter_dirs, max_total_token_num, load_way, mode = trans_list
        self.use_rank_id = use_rank_id
        torch.cuda.set_device(self.use_rank_id)

        self.tp_rank = rank_id
        self.world_size = world_size
        self.load_way = load_way
        self.mode = mode
        self.input_params = input_params
        # Install the active ServerConfig in this (possibly remote) process so
        # downstream hot-path code (infer_batch, etc.) can read get_active_config().
        set_active_config(input_params.cfg)
        self.prefetch_stream = prefetch_stream

        self.cache = {}
        self.original_weights = {}
        self.backward_status = BackwardResumePoint.BEFORE_OPTIMIZER
        nccl_port = input_params.cfg.server.nccl_port + use_rank_id
        dist.init_process_group('nccl', init_method=f'tcp://127.0.0.1:{nccl_port}', rank=rank_id, world_size=world_size)
        model_cfg = get_model_config(weight_dir, dummy=input_params.dummy)
        if half_model:
            model_cfg["num_hidden_layers"] = int(model_cfg["num_hidden_layers"] / 2)
        max_finetuning_tokens = input_params.cfg.memory.max_finetuning_tokens
        try:
            self.model_type = model_cfg["model_type"]
            if self.model_type == "llama":
                if "Llama-3" in weight_dir:
                    self.model = Llama3TpPartModel(rank_id, world_size, weight_dir,
                                                    max_total_token_num,
                                                    mem_adapter_size=input_params.pool_size_lora,
                                                    load_way=load_way, mode=mode,
                                                    dummy=input_params.dummy,
                                                    half_model=half_model,
                                                    mem_manager_log_path=mem_manager_log_path,
                                                    unified_mem_manager_max_size=unified_mem_manager_max_size,
                                                    max_finetuning_tokens=max_finetuning_tokens)
                elif "num_key_value_heads" in model_cfg.keys():
                    self.model = Llama2TpPartModel(rank_id, world_size, weight_dir,
                                                    max_total_token_num,
                                                    mem_adapter_size=input_params.pool_size_lora,
                                                    load_way=load_way, mode=mode,
                                                    dummy=input_params.dummy)

                else:
                    self.model = LlamaTpPartModel(rank_id, world_size, weight_dir,
                                                    max_total_token_num,
                                                    mem_adapter_size=input_params.pool_size_lora,
                                                    load_way=load_way, mode=mode,
                                                    dummy=input_params.dummy,
                                                    half_model=half_model,
                                                    mem_manager_log_path=mem_manager_log_path,
                                                    unified_mem_manager_max_size=unified_mem_manager_max_size,
                                                    max_finetuning_tokens=max_finetuning_tokens)
                    if gpu_profiler is not None: gpu_profiler.mark_annotation("model_load")
            else:
                raise Exception(f"can not support {self.model_type} now")
        except Exception as e:
            print("#" * 16)
            print("load model error:", str(e), e, type(e))
            raise e

        ''' init adapters '''
        self.adapters = []
        self.adapter_id = {}
        num = 0
        print(f"Prepare to load {len(adapter_dirs)} adapters")
        for adapter_dir in tqdm(adapter_dirs, desc="load adapters"):
            print(f"Adding adapter from {adapter_dir}, number {num}")
            num += 1
            self.adapter_id[adapter_dir] = len(self.adapters)
            self.adapters.append(LoraTpPartAdapter(rank_id, world_size, adapter_dir, model_cfg,
                                                   swap=input_params.swap, dummy=input_params.dummy,
                                                   no_lora_swap=input_params.no_lora_swap,
						   prefetch_stream=prefetch_stream))

        finetuning_lora_path = input_params.finetuning_params.finetuning_lora_path
        self.finetuning_adapter = None
        self.backward_service = None
        if finetuning_lora_path != None:
            self.adapter_id[finetuning_lora_path] = len(self.adapters)
            print("Loading finetuning adapter", finetuning_lora_path)
            self.adapters.append(LoraTpPartAdapter(rank_id, world_size, finetuning_lora_path, model_cfg,
                                                    swap=input_params.swap, dummy=input_params.dummy,
                                                    no_lora_swap=input_params.no_lora_swap,
                                                        prefetch_stream=prefetch_stream, is_finetuning=True))

            self.finetuning_adapter = self.adapters[-1]
            self.finetuning_adapter.is_finetuning = True
            self.current_epoch = 0
            self.total_epochs = input_params.finetuning_params.num_epochs
            self.bwd_pause_event = None
            if True:
                rpc_recv, bwd_send = Pipe()
                bwd_recv, rpc_send = Pipe()
                self.bwd_pause_event = mp.Event()
                self.bwd_pause_event.set()   # set = RUNNING; clear = PAUSED
                cg = input_params.cfg.cuda_graph
                backward_service_obj = self.model.backward_service_class(
                    self.model.config, bwd_recv, bwd_send,
                    lr=input_params.finetuning_params.learning_rate,
                    weight_decay=input_params.finetuning_params.weight_decay,
                    gamma=input_params.finetuning_params.gamma,
                    use_rank_id=self.use_rank_id,
                    enable_bwd_graph=cg.enable_bwd_cuda_graph,
                    max_saved_finetuning_tokens=input_params.finetuning_params.max_saved_finetuning_tokens,
                    use_graphed_bwd_attention=cg.use_graphed_bwd_attention,
                    attn_bn_max=cg.attn_bn_max,
                    attn_l_max=cg.attn_l_max,
                )
                backward_service_obj.bwd_pause_event = self.bwd_pause_event
                backward_service_obj.receive_model_dict(self.model.export_model_dict())
                self.rpc_recv = rpc_recv
                self.rpc_send = rpc_send
                _ = torch.cuda.current_device()
                os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = "10"
                os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
                self.backward_service = Process(target=backward_service_obj.start_service, daemon=True)
                self.backward_service.start()
                os.environ.pop("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", None)
                os.environ.pop("CUDA_DEVICE_MAX_CONNECTIONS", None)
                self.rpc_send.send(self.finetuning_adapter.load_gpu_fp32_dict())
                self.rpc_send.send(self.model.mem_manager.share_activation_dict())
                self.rpc_recv.recv()

        self.adapter_id[None] = len(self.adapters)
        self.adapters.append(None)

        if input_params.no_mem_pool:
            head_num = self.model.config["num_attention_heads"]
            self.infer_adapter = NaiveInferAdapter.init(self.model.config["num_hidden_layers"],
                                                        head_num,
                                                        self.model.config["hidden_size"] // head_num)
        else:
            self.infer_adapter = InferAdapterAlt.init(self.model.mem_manager)

        set_random_seed(2147483647)
        return

    @torch.no_grad()
    def exposed_load_adapters(self, adapter_dirs, prefetch=False):
        adapters = []
        for adapter_dir in adapter_dirs:
            if adapter_dir is not None:
                adapters.append(self.adapters[self.adapter_id[adapter_dir]])
        self.infer_adapter.load_adapters(adapters)


    @torch.no_grad()
    def exposed_offload_adapters(self, reserve_dirs=None, prefetch=False):
        self.infer_adapter.offload_adapters(reserve_dirs if reserve_dirs is not None else [])

    # @calculate_time(show=True, min_cost_ms=0.1)
    def exposed_add_batch(self, batch_id, reqs, dtype):
        if self.world_size != 1:
            batch_id, reqs, dtype = obtain(batch_id), obtain(reqs), obtain(dtype)
        if dtype == "fp16":
            dtype = torch.float16
        else:
            assert False, "error dtype"
        batch_data = InferBatch.init_batch(batch_id, reqs, dtype, torch.cuda.current_device(), 
                                        self.model.mem_manager, self.model.vocab_size)
        self.cache[batch_id] = batch_data
        return
    
    # @calculate_time(show=True, min_cost_ms=300)
    # @calculate_time(show=True, min_cost_ms=0)
    def exposed_prefill_batch(self, batch_id, prefill_interrupt_event=None):
        return self.forward(batch_id, is_prefill=True, prefill_interrupt_event=prefill_interrupt_event)

    # @calculate_time(show=True, min_cost_ms=200)
    # @calculate_time(show=True, min_cost_ms=0)
    def exposed_decode_batch(self, batch_id, decode_count=-1):
        return self.forward(batch_id, is_prefill=False, decode_count=decode_count)

    # @calculate_time(show=True, min_cost_ms=0.1)
    def exposed_filter_batch(self, batch_id, req_id_list):
        if self.world_size != 1:
            batch_id, req_id_list = obtain(batch_id), obtain(req_id_list)
        batch = self.cache.pop(batch_id)
        filter_batch = batch.filter(req_id_list)
        del batch
        self.cache[batch_id] = filter_batch
        return

    # @calculate_time(show=True, min_cost_ms=0.1)
    def exposed_merge_batch(self, batch_id1, batch_id2):
        batch1 = self.cache.pop(batch_id1)
        batch2 = self.cache.pop(batch_id2)
        m_batch = InferBatch.merge(batch1, batch2)
        del batch1
        del batch2
        self.cache[batch_id1] = m_batch
        return

    # @calculate_time(show=True, min_cost_ms=10)
    def exposed_remove_batch(self, batch_id):
        batch = self.cache.pop(batch_id)
        batch.free_self()
        del batch
        # torch.cuda.empty_cache()
        return

    def forward(self, batch_id, is_prefill, prefill_interrupt_event=None, decode_count=-1):
        if self.use_rank_id != self.tp_rank:
            torch.cuda.set_device(self.use_rank_id)
        batch: InferBatch = self.cache.pop(batch_id)
        # print(batch.requests)
        # print([req["request_id"] for req in batch.requests])
        kwargs = {
            "batch_size": len(batch),
            "total_token_num": batch.nopad_total_token_num,
            "max_len_in_batch": batch.nopad_max_len_in_batch,
            "input_ids": batch.input_ids,
            "b_loc": batch.nopad_b_loc,
            "b_start_loc": batch.nopad_b_start_loc,
            "b_seq_len": batch.nopad_b_seq_len,
            "is_prefill": is_prefill,
        }

        # assert False, f"{kwargs}"

        assert len(batch.adapter_dirs) == len(batch), "batch.adapter_dirs != batch"
            # always use lora batch infer
        if (self.input_params.no_lora or self.input_params.no_kernel or
            self.input_params.scheduler == "peft" or set(batch.adapter_dirs) == {None}):
            engine = self.model
        else:
            adapters = [self.adapters[self.adapter_id[adapter_dir]] for adapter_dir in batch.adapter_dirs]
            engine = LoraUnorderedBatchMixed(
                    self.model,
                    adapters,
                    infer_adapter=self.infer_adapter,
                    finetuning_adapter=self.finetuning_adapter,
                    enable_cuda_graph=getattr(self.input_params, 'enable_cuda_graph', False),
                    enable_prefill_cuda_graph=getattr(self.input_params, 'enable_prefill_cuda_graph', False))
            kwargs["finetune_mask"] = batch.finetune_mask
            kwargs["b_loc_key"] = batch.nopad_b_loc_key
            kwargs["b_loc_value"] = batch.nopad_b_loc_value
            kwargs["prefill_interrupt_event"] = prefill_interrupt_event
            kwargs["no_lora_compute"] = self.input_params.no_lora_compute
            # kwargs["no_lora_copy"] = self.input_params.no_lora_copy 
        if not is_prefill and decode_count == 2:
            kwargs["print_time_profile"] = True
        logits = engine.forward(**kwargs)
        #print(f"Engine execution time: {time.time() - start:.3f}s")

        if logits is None:
            batch.nopad_max_len_in_batch += 1
            batch.nopad_b_seq_len += 1
            self.cache[batch.batch_id] = batch
            return None

        next_token_ids, next_token_probs = sample(logits, batch)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
        output_dict = {}
        new_input_ids = []        
        for i, (r, all_input_ids, next_token_id, next_token_logprob) in enumerate(zip(batch.requests, batch.all_input_ids, next_token_ids, next_token_logprobs)):
            all_input_ids.append(int(next_token_id))
            new_input_ids.append(next_token_id)
            batch.all_input_ids[i] = all_input_ids
            batch.input_lengths[i] += 1
            batch.out_token_id_counts[i][next_token_id] += 1
            metadata = {
                'id': int(next_token_id),
                'logprob': float(next_token_logprob),
            }
            output_dict[r['request_id']] = (int(next_token_id), metadata)
        
        batch.input_ids = torch.tensor(new_input_ids, dtype=torch.long).cuda()
        batch.nopad_b_start_loc = batch.nopad_b_start_loc + torch.arange(0, len(batch), dtype=torch.int32, device="cuda")
        batch.nopad_total_token_num += len(batch)
        batch.nopad_max_len_in_batch += 1
        batch.nopad_b_seq_len += 1
        self.cache[batch.batch_id] = batch
        return output_dict
            
    def exposed_back_batch(self, current_epoch):
        self.current_epoch = current_epoch
        result = self.backward()
        return result
    
    def backward(self):
        requests_info_dict = self.model.mem_manager.export_requests_info()
        requests_info_dict["current_epoch"] = self.current_epoch
        self.rpc_send.send(requests_info_dict)
        finished, loss, total_token_processed = self.rpc_recv.recv()
        if finished:
            self.model.mem_manager.reset_activation_pool()
            return True, loss, total_token_processed
        else:
            return False, None, None
    
    def exposed_shutdown_backward(self):
        if self.backward_service is not None:
            self.rpc_send.send("EXIT")
            self.backward_service.join()

    def exposed_pause_backward(self):
        self.bwd_pause_event.clear()

    def exposed_resume_backward(self):
        self.bwd_pause_event.set()

    def exposed_get_graph_mem_stats(self):
        """Return total CUDA-graph memory captured so far on the shared
        runners owned by LoraUnorderedBatchMixed. Returned as a dict of
        {kind: (bytes, num_graphs)} for printing on the manager side."""
        from dserve.models.peft.lora_unordered_batch_mixed import LoraUnorderedBatchMixed
        stats = {}
        r = LoraUnorderedBatchMixed._shared_cuda_graph_runner
        if r is not None:
            stats["decode"] = (r.total_decode_graph_bytes(), r.num_decode_graphs())
            stats["prefill"] = (r.total_prefill_graph_bytes(), r.num_prefill_graphs())
        return stats

    def exposed_get_all_captured_buckets(self):
        """Return the full set of currently captured (bs, bucket) keys —
        used by the manager to seed its GraphEligibility mirror after
        offline profiling. Returns {'decode': [...], 'prefill': [...]}."""
        from dserve.models.peft.lora_unordered_batch_mixed import LoraUnorderedBatchMixed
        r = LoraUnorderedBatchMixed._shared_cuda_graph_runner
        if r is None:
            return {"decode": [], "prefill": []}
        return {
            "decode": r.all_decode_buckets(),
            "prefill": r.all_prefill_buckets(),
        }

    def exposed_pop_pending_captures(self):
        """Drain & return buckets newly captured since last call. Manager polls
        this on each scheduler iteration to keep its eligibility mirror current.
        Returns {'decode': [...], 'prefill': [...]}."""
        from dserve.models.peft.lora_unordered_batch_mixed import LoraUnorderedBatchMixed
        r = LoraUnorderedBatchMixed._shared_cuda_graph_runner
        if r is None:
            return {"decode": [], "prefill": []}
        return {
            "decode": r.pop_pending_decode_captures(),
            "prefill": r.pop_pending_prefill_captures(),
        }
    
    def compare_lora_drift_per_layer(self):
        """
        Compares current LoRA weights to the originally saved weights,
        and prints per-layer total drift (sum of L2 norms across q/k/v/o).
        """
        print("=== LoRA Parameter Drift by Layer ===")
        for layer_id, layer in enumerate(self.finetuning_adapter.layers):
            total_drift = 0.0
            for name in [
                "w_combined_home",
            ]:
                key = f"layer_{layer_id}.{name}"
                original = self.original_weights[key].to(layer.__dict__[name].device)
                current = getattr(layer, name)
                
                drift = torch.norm(current - original).item()
                total_drift += drift
                weight_norm = original.norm().item()
                ratio = total_drift / weight_norm

            if total_drift!=0:
                print(f"Layer {layer_id:02d}: total ∆W = {total_drift:.6f} |  step/weight ratio = {ratio:.4e}")


    def check_adapter_update(self):
        for idx, layer in enumerate(self.finetuning_adapter.layers):
            w_fp32 = layer.w_combined_home_fp32       # Tensor on GPU, dtype=float32
            w = layer.w_combined_home          # Tensor on GPU, dtype=float16
            if torch.isnan(w).any() or torch.isinf(w).any():
                print(f"[non-finite] layer {idx} → w_combined_home has NaN/Inf")
                if torch.isnan(w_fp32).any() or torch.isinf(w_fp32).any():
                    print(f"And w_combined_home_fp32 has NaN/Inf")
                else:
                    print(f"but w_combined_home_fp32 does not has NaN/Inf")
                break   


class ModelRpcClient:
    def __init__(self, model_rpc, world_size, rpc_server_process=None):
        self.model: ModelRpcServer = model_rpc
        self.world_size = world_size
        self.rpc_server_process = rpc_server_process
        self.use_rpc = self.world_size != 1
        if self.use_rpc:
            def async_wrap(f):
                f = rpyc.async_(f)
                async def _func(*args, **kwargs):
                    ans = f(*args, **kwargs)
                    await asyncio.to_thread(ans.wait)
                    # raise if exception
                    return ans.value
                return _func
            self._init_model = async_wrap(self.model.init_model)
            self._load_adapters = rpyc.async_(self.model.load_adapters)
            self._offload_adapters = rpyc.async_(self.model.offload_adapters)
            self._add_batch = async_wrap(self.model.add_batch)
            self._prefill_batch = async_wrap(self.model.prefill_batch)

            self._back_batch = async_wrap(self.model.back_batch)
            self._pause_backward = async_wrap(self.model.pause_backward)
            self._resume_backward = async_wrap(self.model.resume_backward)
            self._shutdown_backward = async_wrap(self.model.shutdown_backward)

            self._decode_batch = async_wrap(self.model.decode_batch)
            self._filter_batch = async_wrap(self.model.filter_batch)
            self._merge_batch = async_wrap(self.model.merge_batch)
            self._remove_batch = async_wrap(self.model.remove_batch)
            self._get_graph_mem_stats = self.model.get_graph_mem_stats
            self._get_all_captured_buckets = self.model.get_all_captured_buckets
            self._pop_pending_captures = self.model.pop_pending_captures
        else:
            self._init_model = self.model.exposed_init_model
            self._load_adapters = self.model.exposed_load_adapters
            self._offload_adapters = self.model.exposed_offload_adapters
            self._add_batch = self.model.exposed_add_batch
            self._prefill_batch = self.model.exposed_prefill_batch

            self._back_batch = self.model.exposed_back_batch
            self._pause_backward = self.model.exposed_pause_backward
            self._resume_backward = self.model.exposed_resume_backward
            self._shutdown_backward = self.model.exposed_shutdown_backward
            
            self._decode_batch = self.model.exposed_decode_batch
            self._filter_batch = self.model.exposed_filter_batch
            self._merge_batch = self.model.exposed_merge_batch
            self._remove_batch = self.model.exposed_remove_batch
            self._get_graph_mem_stats = self.model.exposed_get_graph_mem_stats
            self._get_all_captured_buckets = self.model.exposed_get_all_captured_buckets
            self._pop_pending_captures = self.model.exposed_pop_pending_captures
        return

    async def init_model(self, rank_id, world_size, weight_dir, adapter_dirs,
                         max_total_token_num, load_way, mode, input_params,
			                prefetch_stream, half_model=False, mem_manager_log_path=None,
                            unified_mem_manager_max_size=0,
                            gpu_profiler=None, use_rank_id=0):
        ans : rpyc.AsyncResult = self._init_model(rank_id, world_size, weight_dir, adapter_dirs,
                                                  max_total_token_num, load_way, mode, input_params,
						                            prefetch_stream, half_model, mem_manager_log_path,
                                                    gpu_profiler, unified_mem_manager_max_size, use_rank_id)
        if self.use_rpc:
            await ans
            return
        else:
            return

    async def load_adapters(self, reqs, prefetch=False):
        self._load_adapters(reqs, prefetch=prefetch)


    async def offload_adapters(self, reserved_reqs=None, prefetch=False):
        self._offload_adapters(reserved_reqs, prefetch=prefetch)


    async def init_batch(self, batch_id, reqs):
        ans = self._add_batch(batch_id, reqs, "fp16")
        if self.use_rpc:
            await ans
            return
        else:
            return
    
    async def prefill_batch(self, batch_id, prefill_interrupt_event=None):
        if self.use_rpc:
            ans = self._prefill_batch(batch_id, prefill_interrupt_event)
            return await ans
        else:
            # run blocking call in a thread
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._prefill_batch, batch_id, prefill_interrupt_event)
        
    async def back_batch(self, current_epoch):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._back_batch, current_epoch)
    
    def back_batch_threading(self, current_epoch):
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(None, self._back_batch, current_epoch)
    
    async def shutdown_backward(self):
        if self.use_rpc:
            ans = self._shutdown_backward()
            return await ans
        else:
            return self._shutdown_backward()

    async def pause_backward(self):
        if self.use_rpc:
            ans = self._pause_backward()
            return await ans
            
        else:
            return self._pause_backward()

    async def resume_backward(self):
        if self.use_rpc:
            ans = self._resume_backward()
            return await ans
        else:
            return self._resume_backward()
    
    def reset_activation_pool(self):
        self.model.model.mem_manager.reset_activation_pool()
        return

    def get_graph_mem_stats(self):
        """Synchronous accessor — returns dict {kind: (bytes, num_graphs)}."""
        if self.use_rpc:
            from rpyc.utils.classic import obtain
            return obtain(self._get_graph_mem_stats())
        return self._get_graph_mem_stats()

    def get_all_captured_buckets(self):
        """Snapshot of currently captured (bs, bucket) keys per modality.
        Used to seed the manager's GraphEligibility mirror once after
        offline profiling. Returns {'decode': [...], 'prefill': [...]}."""
        if self.use_rpc:
            from rpyc.utils.classic import obtain
            return obtain(self._get_all_captured_buckets())
        return self._get_all_captured_buckets()

    def pop_pending_captures(self):
        """Drain newly captured buckets from the runner. Cheap call —
        manager polls at scheduler-iteration cadence."""
        if self.use_rpc:
            from rpyc.utils.classic import obtain
            return obtain(self._pop_pending_captures())
        return self._pop_pending_captures()

    async def decode_batch(self, batch_id, decode_count=-1):
        ans = self._decode_batch(batch_id, decode_count=decode_count)
        if self.use_rpc:
            return await ans
        else:
            return ans

    async def filter_batch(self, batch_id, req_id_list):
        ans = self._filter_batch(batch_id, req_id_list)
        if self.use_rpc:
            await ans
            return
        else:
            return 

    async def merge_batch(self, batch_id1, batch_id2):
        ans = self._merge_batch(batch_id1, batch_id2)
        if self.use_rpc:
            await ans
            return
        else:
            return

    async def remove_batch(self, batch_id):
        ans = self._remove_batch(batch_id)
        if self.use_rpc:
            await ans
            return
        else:
            return
    

def _init_env(port):
    from rpyc.utils.server import ThreadedServer
    t = ThreadedServer(ModelRpcServer(), port=port, protocol_config={"allow_pickle": True})
    t.start()
    return


async def start_model_process(port, world_size):
    # 单卡时不使用 rpc
    if world_size == 1:
        return ModelRpcClient(ModelRpcServer(), world_size)
    
    import multiprocessing
    proc = multiprocessing.Process(target=_init_env, args=(port,))
    proc.start()
    await asyncio.sleep(2)
    repeat_count = 0
    while repeat_count < 20:
        try:
            con = rpyc.connect("localhost", port, config={"allow_pickle": True})
            break
        except BaseException:
            await asyncio.sleep(1)
        repeat_count += 1
    if repeat_count == 20:
        raise Exception("init rpc env error!")

    assert proc.is_alive()
    return ModelRpcClient(con.root, world_size, rpc_server_process=proc)