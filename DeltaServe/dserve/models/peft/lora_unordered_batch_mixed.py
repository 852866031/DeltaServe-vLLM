import time
import numpy as np
from dserve.common.unified_mem_allocator import PageType
from dserve.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from dserve.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from dserve.models.peft.layer_weights.lora_layer_weight import LoraLayerWeight
import torch
import torch.nn as nn
from typing import final
from typing import Tuple
from dserve.common.infer_utils import init_bloc
from dserve.models.llama.triton_kernel.context_flashattention_nopad import context_attention_fwd
from dserve.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from dserve.models.peft.triton_kernel.lora.lora_prefill import lora_get_qkvo_fwd_shrink, lora_get_qkvo_fwd_expand
from dserve.server.router.model_infer.naive_infer_adapter import NaiveInferAdapter
from dserve.utils.infer_utils import mark_cost_time, set_random_seed
from dserve.utils.infer_utils import calculate_time, mark_start, mark_end
from dserve._kernels import dispatch_bgmv
from ...server.router.mixed_req_queue import rprint
import hashlib
from dserve.models.peft.alt_to_slora_kernel import dispatch_bgmv_pt, compare_tensors, dispatch_bgmv_pt_exact
import math
from dserve.common.cuda_graph_runner import CudaGraphRunner


def _resolve_max_graph_memory_bytes() -> "int | None":
    """Read `cuda_graph.max_graph_memory_gb` from active cfg and return
    bytes, or None if unset / non-positive (no cap)."""
    try:
        from dserve.common.configs.config import get_active_config
        cap_gb = get_active_config().cuda_graph.max_graph_memory_gb
    except Exception:
        return None
    if cap_gb is None or cap_gb < 0:
        return None
    return int(cap_gb * (1024 ** 3))


import torch
import hashlib

def tensor_hash(t: torch.Tensor, algo="sha256") -> str:
    h = hashlib.new(algo)
    h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def kernel_bgmv(y: torch.Tensor , x: torch.Tensor, w: torch.Tensor, start_indicies: torch.Tensor,
                   lora_ranks: torch.Tensor, loc_indicies: torch.Tensor, indicies: torch.Tensor,
                   qkvo: int, lora_scales: torch.Tensor):
    w_ld = w.size(1) * w.size(2)
    h_in = x.size(1)
    h_out = y.size(1)
    w_valid = h_in if (h_out < h_in) else h_out
    dispatch_bgmv(y, x, w, start_indicies, lora_ranks, loc_indicies, indicies, qkvo, lora_scales, w_ld, w_valid)

class LoraUnorderedBatchMixed:
    # Class-level shared runner (persists across engine instances)
    _shared_cuda_graph_runner = None

    def __init__(self, base_model, adapters, infer_adapter=None, finetuning_adapter= None,
                 enable_cuda_graph=False, enable_prefill_cuda_graph=False):
        self.base_model = base_model
        self.enable_cuda_graph = enable_cuda_graph
        self.enable_prefill_cuda_graph = enable_prefill_cuda_graph

        lora_layer_dim = [adapter.r if adapter is not None else 0 for adapter in adapters]
        self.max_lora_dim = max(lora_layer_dim)
        self.req_bins = torch.zeros(len(adapters), dtype=torch.long, device="cuda")
        self.finetuning_adapter = finetuning_adapter
        self.is_finetuning_batch = False
        self.infer_adapter = infer_adapter
        if isinstance(infer_adapter, NaiveInferAdapter):
            self.key_buffer = infer_adapter.key_buffer
            self.value_buffer = infer_adapter.value_buffer
            for i, adapter in enumerate(adapters):
                # FIX ME @TODO: currently not supporting adapter is None
                if adapter is None: continue
                idx = infer_adapter.adapter_dirs.index(adapter.lora_dir)
                self.req_bins[i] = idx
        else:
            for i, adapter in enumerate(adapters):
                if adapter is None: continue
                idx = infer_adapter.adapter_dirs.index(adapter.lora_dir)
                self.req_bins[i] = idx
                
        
        self.kv_embed_dim = base_model.tp_k_head_num_ * base_model.head_dim_


    @torch.no_grad()
    def forward(
            self,
            batch_size, # number of request
            total_token_num,
            max_len_in_batch,
            input_ids, # 1D input tensor
            b_loc, # mapping to memory pool
            b_loc_key, # mapping to key memory pool
            b_loc_value, # mapping to value memory pool
            b_start_loc, # the start index of each request
            b_seq_len, # the current length of each request
            finetune_mask,
            is_prefill=True,
            use_bmm=True,
            no_lora_compute=False,
            no_lora_copy=False,
            prefill_interrupt_event=None,
            print_time_profile = False):

        # Notice that batch_lora only support decoding
        assert len(b_loc) == len(b_start_loc) == len(b_seq_len)
        self.delta = []
        self.max_b_seq_len = torch.max(b_seq_len).item()

        if is_prefill:
            assert(len(self.req_bins)==len(b_seq_len))
            self.batch_req_bins = torch.repeat_interleave(self.req_bins, b_seq_len)
            # self.b_start_loc = torch.cumsum(torch.cat([torch.tensor([0], dtype=torch.long, device="cuda"), b_seq_len[:-1]]), dim=0)
            for _ in range(3):
                self.delta.append(torch.zeros((len(self.batch_req_bins), self.max_lora_dim), dtype=torch.float16, device="cuda"))

            out = self._prefill(batch_size, total_token_num, max_len_in_batch,
                                    input_ids,
                                    b_loc, b_loc_key, b_loc_value, b_start_loc, b_seq_len,
                                    finetune_mask, no_lora_compute, prefill_interrupt_event)
            return out
        else:
            for _ in range(3):
                self.delta.append(torch.zeros((len(b_seq_len), self.max_lora_dim), dtype=torch.float16, device="cuda"))
            out = self._decode(batch_size, total_token_num, max_len_in_batch,
                                input_ids,
                                b_loc, b_loc_key, b_loc_value, b_start_loc, b_seq_len,
                                no_lora_compute, no_lora_copy, print_time_profile)
            #print(tensor_hash(out))
            return out

    # Prefill functions for inference and finetuning
    def _prefill(self, batch_size, total_token_num, max_len_in_batch,
                 input_ids,
                 b_loc, b_loc_key, b_loc_value, b_start_loc, b_seq_len, finetune_mask,
                   no_lora_compute=False, prefill_interrupt_event=None):

        infer_state = self.base_model.infer_state_class()
        infer_state.is_prefill = True
        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.max_len_in_batch = max_len_in_batch

        assert (input_ids.shape[0] == total_token_num)
        assert (b_loc.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0])

        infer_state.finetune_mask = torch.zeros(input_ids.shape[0], dtype=torch.bool, device="cuda")
        nr_finetuning_reqs = 0
        finetuning_start = input_ids.shape[0]
        if finetune_mask is not None:
            for i in range(batch_size):
                if finetune_mask[i] == 1:
                    nr_finetuning_reqs += 1
                    start = b_start_loc[i].item()
                    length = b_seq_len[i].item()
                    if finetuning_start == input_ids.shape[0]:
                        finetuning_start = start
                    infer_state.finetune_mask[start : start + length] = True

        b_seq_len_numpy = b_seq_len.cpu().numpy()
        position_ids = torch.from_numpy(np.concatenate([
            np.arange(0, b_seq_len_numpy[i]) for i in range(len(b_seq_len_numpy))
            ], axis=0)).cuda()
        infer_state.position_cos = torch.index_select(
                self.base_model._cos_cached, 0, position_ids).view(position_ids.shape[0], -1)
        infer_state.position_sin = torch.index_select(
                self.base_model._sin_cached, 0, position_ids).view(position_ids.shape[0], -1)
        position_ids = None
        infer_state.b_start_loc = b_start_loc
        infer_state.b_seq_len = b_seq_len
        infer_state.mem_manager = self.base_model.mem_manager
        infer_state.b_loc_key = b_loc_key
        infer_state.b_loc_value = b_loc_value

        # Decide CUDA graph path. Full-graph captures attention inside the graph
        # and supports batch_size in {1, 2, 4, 8, 16, 32, 64} via padding.
        # Prefill CG is gated off for any batch with active finetune tokens or
        # an interrupt event in flight.
        has_ft = bool(torch.any(infer_state.finetune_mask))
        use_prefill_cg = (
            self.enable_prefill_cuda_graph
            and not has_ft
            and prefill_interrupt_event is None
            and batch_size <= 64  # bs_bucket support: {1, 2, 4, 8, 16, 32, 64}
        )
        if use_prefill_cg:
            T_bucket = CudaGraphRunner.get_prefill_token_bucket(total_token_num)
            alloc_size = T_bucket
        else:
            alloc_size = total_token_num

        infer_state.prefill_mem_index_key = self.base_model.mem_manager.alloc(alloc_size, PageType.KV_CACHE)
        infer_state.prefill_mem_index_value = self.base_model.mem_manager.alloc(alloc_size, PageType.KV_CACHE)
        infer_state.prefill_mem_index_cat = torch.cat([infer_state.prefill_mem_index_key, infer_state.prefill_mem_index_value], dim=0)
        # init_bloc uses the first total_token_num indices for the real tokens.
        # Padding indices [total_token_num: T_bucket] are dedicated scratch slots.
        init_bloc(infer_state.b_loc_key, b_seq_len, max_len_in_batch,
                  infer_state.prefill_mem_index_key[:total_token_num])
        init_bloc(infer_state.b_loc_value, b_seq_len, max_len_in_batch,
                  infer_state.prefill_mem_index_value[:total_token_num])

        infer_state.prefill_key_buffer = torch.empty(
                (alloc_size, self.base_model.tp_k_head_num_, self.base_model.head_dim_),
                dtype=torch.float16, device="cuda")
        infer_state.prefill_value_buffer = torch.empty(
                (alloc_size, self.base_model.tp_k_head_num_, self.base_model.head_dim_),
                dtype=torch.float16, device="cuda")

        if use_prefill_cg:
            predict_logics = self._prefill_with_cuda_graph(
                batch_size, total_token_num, input_ids, infer_state,
                self.batch_req_bins, no_lora_compute)
        else:
            predict_logics = self._context_forward(
                input_ids, infer_state, no_lora_compute,
                prefill_interrupt_event=prefill_interrupt_event)

        # Free the prefill scratch slots that backed the padded positions
        # `[total_token_num, alloc_size)`. They were allocated from the KV
        # pool so the padded forward had somewhere to write throwaway K/V
        # for padding tokens, but they're not registered in b_loc_key/value,
        # so the request-finish cleanup (free_self / filter / clip) will
        # never reclaim them. The captured graph doesn't pin these specific
        # slot ids — `prefill_replay` overwrites the static index buffer
        # with new ids on every replay — so freeing them here is safe.
        if use_prefill_cg and alloc_size > total_token_num:
            scratch_k = infer_state.prefill_mem_index_key[total_token_num:alloc_size]
            scratch_v = infer_state.prefill_mem_index_value[total_token_num:alloc_size]
            self.base_model.mem_manager.free_kv(scratch_k)
            self.base_model.mem_manager.free_kv(scratch_v)
        return predict_logics

    def interrupt_and_clean(
        self,
        prefill_interrupt_event,
        infer_state,
        FFN_input_vpids: torch.Tensor = None,
        attention_input_vpids: torch.Tensor = None,
    ) -> bool:
        """
        Tensor-based version: if interrupted, free all given vpids (on GPU).
        """
        vpids_to_free = None
        if prefill_interrupt_event is not None and prefill_interrupt_event.is_set():
            print("Prefill interrupted, cleaning up…")
            if FFN_input_vpids is not None and attention_input_vpids is not None:
                vpids_to_free = torch.cat((FFN_input_vpids, attention_input_vpids))
            elif FFN_input_vpids is not None:
                vpids_to_free = FFN_input_vpids
            elif attention_input_vpids is not None:
                vpids_to_free = attention_input_vpids
            if vpids_to_free is not None and vpids_to_free.numel() > 0:
                infer_state.mem_manager.free(vpids_to_free)
                self.infer_adapter.unpin_adapters_pages() 
                self.base_model.mem_manager.reset_b_loc_kv(None, None)
            return True
        return False
    
    @torch.no_grad()
    def sanitize_logits(self, logits: torch.Tensor, clamp_min: float = -1e4, clamp_max: float = 1e4) -> torch.Tensor:
        return logits
        logits_fp32 = logits.float()
        logits_fp32 = torch.nan_to_num(logits_fp32, nan=0.0, posinf=0.0, neginf=0.0)
        logits_fp32 = torch.clamp(logits_fp32, min=clamp_min, max=clamp_max)

        if torch.isnan(logits_fp32).any() or torch.isinf(logits_fp32).any():
            print("⚠️ NaN or Inf remain in logits after cleanup!")
        return logits_fp32

    def _post_infer_inference_only(self, input_embdings, infer_state):
        """Graph-friendly prefill output: return last-token logits for each request.
        Pure GPU ops — no .item(), no Python batch loop, so the graph captures cleanly.
        """
        layer_weight = self.base_model.pre_post_weight
        last_index = torch.cumsum(infer_state.b_seq_len, dim=0, dtype=torch.long) - 1
        last_input = input_embdings[last_index, :]
        last_input = self.base_model.post_infer._norm(last_input, infer_state, layer_weight)
        last_token_logits = torch.mm(last_input, layer_weight.lm_head_weight_.T)
        return last_token_logits

    def _prefill_with_cuda_graph(self, batch_size, total_token_num, input_ids,
                                  infer_state, real_batch_req_bins, no_lora_compute=False):
        """Capture or replay a full prefill CUDA graph (inference-only).

        Supports batch_size in {1, 2, 4, 8, 16, 32, 64} via padding to bs_bucket;
        captured graph runs as if there were bs_bucket requests, padded slots
        have b_seq_len=0 so attention is a no-op for them.

        Must be called AFTER `_prefill` has populated infer_state (memory indices,
        position_cos/sin, b_loc_key/value, prefill_key/value_buffer).
        """
        if LoraUnorderedBatchMixed._shared_cuda_graph_runner is None:
            LoraUnorderedBatchMixed._shared_cuda_graph_runner = CudaGraphRunner(
                max_total_tokens=self.base_model.max_total_token_num if hasattr(self.base_model, 'max_total_token_num') else 25000,
                num_layers=self.base_model.layers_num,
                tp_q_head_num=self.base_model.tp_q_head_num_,
                tp_k_head_num=self.base_model.tp_k_head_num_,
                head_dim=self.base_model.head_dim_,
                max_graph_memory_bytes=_resolve_max_graph_memory_bytes(),
            )
        runner = LoraUnorderedBatchMixed._shared_cuda_graph_runner

        # Skip all finetune_mask checks inside _context_forward.
        infer_state._skip_finetune_checks = True

        forward_kwargs = {
            'no_lora_compute': no_lora_compute,
            'prefill_interrupt_event': None,
        }

        # DEBUG: set DSERVE_PREFILL_CG_DEBUG=pad_only to run the padded forward
        # without graph capture/replay — isolates whether issues are in padding
        # or graph machinery.
        import os
        if os.environ.get("DSERVE_PREFILL_CG_DEBUG", "") == "pad_only":
            return self._prefill_padded_no_graph(
                batch_size, total_token_num, input_ids, infer_state, real_batch_req_bins, forward_kwargs)

        if runner.has_prefill_graph(batch_size, total_token_num):
            logits = runner.prefill_replay(
                self, batch_size, total_token_num, input_ids, infer_state, real_batch_req_bins)
        elif runner.can_capture_more():
            logits = runner.prefill_capture(
                self, batch_size, total_token_num, self._context_forward,
                input_ids, infer_state, real_batch_req_bins, forward_kwargs)
        else:
            # Memory cap reached — run the padded eager forward instead
            # of a fresh capture. Same math as the captured path, no graph.
            runner.note_capture_refused()
            logits = self._prefill_padded_no_graph(
                batch_size, total_token_num, input_ids, infer_state,
                real_batch_req_bins, forward_kwargs)
        # logits is bs_bucket-sized; the caller expects exactly batch_size rows.
        return logits[:batch_size]

    def _prefill_padded_no_graph(self, batch_size, total_token_num, input_ids,
                                  infer_state, real_batch_req_bins, forward_kwargs):
        """Run a padded prefill forward WITHOUT CUDA graph capture.
        Mirrors the padding logic in CudaGraphRunner.prefill_capture but runs eagerly.
        """
        runner = LoraUnorderedBatchMixed._shared_cuda_graph_runner
        T_bucket = runner.get_prefill_token_bucket(total_token_num)

        self.delta = [
            torch.zeros((T_bucket, self.max_lora_dim), dtype=torch.float16, device="cuda")
            for _ in range(3)
        ]

        new_input = torch.zeros(T_bucket, dtype=input_ids.dtype, device="cuda")
        new_input[:total_token_num].copy_(input_ids)

        pc = infer_state.position_cos
        ps = infer_state.position_sin
        new_pc = torch.zeros((T_bucket, pc.shape[1]), dtype=pc.dtype, device="cuda")
        new_ps = torch.zeros((T_bucket, ps.shape[1]), dtype=ps.dtype, device="cuda")
        new_pc[:total_token_num].copy_(pc)
        new_ps[:total_token_num].copy_(ps)
        infer_state.position_cos = new_pc
        infer_state.position_sin = new_ps

        new_brb = torch.zeros(T_bucket, dtype=real_batch_req_bins.dtype, device="cuda")
        new_brb[:total_token_num].copy_(real_batch_req_bins)
        self.batch_req_bins = new_brb

        # prefill_mem_index_{key,value} + prefill_key/value_buffer are already
        # T_bucket-sized (allocated in _prefill when use_prefill_cg=True).
        assert infer_state.prefill_mem_index_key.shape[0] == T_bucket
        assert infer_state.prefill_key_buffer.shape[0] == T_bucket
        infer_state.prefill_mem_index_cat = torch.cat(
            [infer_state.prefill_mem_index_key, infer_state.prefill_mem_index_value], dim=0)

        infer_state.finetune_mask = torch.zeros(T_bucket, dtype=torch.bool, device="cuda")
        infer_state.total_token_num = T_bucket

        return self._context_forward(new_input, infer_state, **forward_kwargs)

    @final
    def _context_forward(self, input_ids, infer_state, no_lora_compute=False, prefill_interrupt_event=None):
        cuda_input_ids = input_ids
        input_embs = self.base_model.pre_infer.context_forward(
                cuda_input_ids, infer_state, self.base_model.pre_post_weight)
        if self.interrupt_and_clean(prefill_interrupt_event, infer_state):
            return None
        # When driving this forward inside a CUDA-graph capture/replay, skip all
        # `torch.any(finetune_mask)` branches — they force host<->device syncs
        # that break graph capture. The prefill-CG path sets this flag.
        skip_ft = getattr(infer_state, '_skip_finetune_checks', False)
        if not skip_ft and torch.any(infer_state.finetune_mask):
            infer_state.mem_manager.save_embedding_output(input_embs, infer_state)
        FFN_input_vpids = None
        attention_input_vpids = None
        for i in range(self.base_model.layers_num):
            input_embs = self._lora_context_forward(i, input_embs, infer_state, no_lora_compute)
            if self.interrupt_and_clean(prefill_interrupt_event, infer_state, FFN_input_vpids, attention_input_vpids):
                return None
            if not skip_ft and torch.any(infer_state.finetune_mask):
                FFN_input_vpids = infer_state.mem_manager.save_activations_by_layer(i, input_embs, infer_state,
                                                                            PageType.FFN_INPUT_ACTIVATION, FFN_input_vpids)
            self.base_model.layers_infer[i]._context_ffn(input_embs, infer_state, self.base_model.trans_layers_weight[i])
            if self.interrupt_and_clean(prefill_interrupt_event, infer_state, FFN_input_vpids, attention_input_vpids):
                return None
            if not skip_ft and torch.any(infer_state.finetune_mask):
                attention_input_vpids = infer_state.mem_manager.save_activations_by_layer(i, input_embs, infer_state,
                                                                            PageType.ATTENTION_INPUT_ACTIVATION, attention_input_vpids)
        if self.interrupt_and_clean(prefill_interrupt_event, infer_state, FFN_input_vpids, attention_input_vpids):
            return None
        if skip_ft:
            # Graph-friendly inference-only last-token logits (no .item(), no Python loop).
            predict_logics = self._post_infer_inference_only(input_embs, infer_state)
            return predict_logics
        finetune_logits_per_request = []
        predict_logics = self.base_model.post_infer.token_forward_with_finetune_outputs(
                input_embs, finetune_logits_per_request, infer_state, self.base_model.pre_post_weight)
        #predict_logics = self.sanitize_logits(predict_logics)
        if torch.any(infer_state.finetune_mask):
            infer_state.mem_manager.write_to_logit_tensor(finetune_logits_per_request, FFN_input_vpids, attention_input_vpids)
        #print(f"Embedding time: {input_embs_layer_end_time - start:.3f}s")
        #print(f"Transformer layer time: {transformer_layer_end_time - input_embs_layer_end_time:.3f}s")
        #print(f"Output layer time: {output_layer_end_time - transformer_layer_end_time:.3f}s")
        #self.check_invalid_probs(predict_logics)
        return predict_logics

    @final
    def _lora_context_forward(self, layer_id, input_embs, infer_state, no_lora_compute=False):
        self._lora_context_attention(layer_id, input_embs, infer_state, no_lora_compute)
        return input_embs
    
    # @mark_cost_time("trans context flash forward time cost")  # dont to remove this, will make performence down, did not know why
    def _lora_context_attention(self, layer_id, input_embs, infer_state, no_lora_compute=False):
        layer_weight = self.base_model.trans_layers_weight[layer_id]
        layer_infer = self.base_model.layers_infer[layer_id]
        # layer normalization
        input1 = layer_infer._att_norm(input_embs, infer_state, layer_weight)
        # fetch k, v
        cache_k, cache_v = layer_infer._pre_cache_kv(infer_state, layer_weight)
        # gen new q, k, v (batch different adapters)
        q = self._lora_get_qkv(layer_id, input1, cache_k, cache_v, infer_state, no_lora_compute)
        input1 = None
        layer_infer._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        # compute attention
        o = layer_infer._context_attention_kernel(q, cache_k, cache_v, infer_state, layer_weight)
        o = self._lora_get_o(layer_id, o, infer_state, no_lora_compute)
        input_embs.add_(o.view(-1, layer_infer.embed_dim_))
        return

    def _lora_get_qkv(self, layer_id, input_embs, cache_k, cache_v, infer_state, no_lora_compute=False)->torch.Tensor:
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]
        # q (S, H)
        q = torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_),
                     base_layer_weight.q_weight_)
        buffer_address, a_start_lora, a_len_lora, gpu_a_loc_lora_a, gpu_a_loc_lora_b, a_scaling = \
                    self.infer_adapter.get_lora_params_at_layer(layer_id)
        #if layer_id==16: print(f"\t\t\tLayer {layer_id} address conversion time: {time.time() - start:.5f}s")
        assert(len(q)==len(self.batch_req_bins))
        if not no_lora_compute:
            # fix me: @TODO we need to filter out requests querying only base model
            if self.max_b_seq_len >= 200 and self.max_lora_dim >= 64  and len(infer_state.b_seq_len) >= 2:
            # if 1 == 0:
                lora_get_qkvo_fwd_shrink(input_embs.view(-1, base_layer_infer.embed_dim_), 
                                         buffer_address.view(-1, self.kv_embed_dim), 
                                         self.delta[0], gpu_a_loc_lora_a, a_start_lora, 
                                         a_len_lora, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         0, self.max_lora_dim, self.max_b_seq_len)
                lora_get_qkvo_fwd_expand(self.delta[0], buffer_address.view(-1, self.kv_embed_dim), 
                                         q, a_scaling, 
                                         gpu_a_loc_lora_b, a_start_lora, 
                                         a_len_lora, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, self.kv_embed_dim, 
                                         0, self.max_lora_dim, self.max_b_seq_len)
            else:
                kernel_bgmv(                                             
                    self.delta[0],
                    input_embs.view(-1, base_layer_infer.embed_dim_),
                    buffer_address,
                    a_start_lora,
                    a_len_lora,
                    gpu_a_loc_lora_a,
                    self.batch_req_bins,
                    0,                                   # qkvo
                    a_scaling,
                )

                kernel_bgmv(                                             # 2️⃣ EXPAND (B-side)
                    q,
                    self.delta[0],
                    buffer_address,
                    a_start_lora,
                    a_len_lora,
                    gpu_a_loc_lora_b,
                    self.batch_req_bins,
                    0,
                    a_scaling,
                )

        rotary_emb_fwd(q.view(-1, base_layer_infer.tp_q_head_num_, base_model.head_dim_),
                          infer_state.position_cos, infer_state.position_sin)

        # k (S, H)
        torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.k_weight_,
                 out=cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_))
        if not no_lora_compute:
            if self.max_b_seq_len >= 200 and self.max_lora_dim >= 64  and len(infer_state.b_seq_len) >= 2:
            # if 1 == 0:
                lora_get_qkvo_fwd_shrink(input_embs.view(-1, base_layer_infer.embed_dim_), 
                                         buffer_address.view(-1, self.kv_embed_dim), 
                                         self.delta[1], gpu_a_loc_lora_a, a_start_lora, 
                                         a_len_lora, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         1, self.max_lora_dim, self.max_b_seq_len)
                lora_get_qkvo_fwd_expand(self.delta[1], buffer_address.view(-1, self.kv_embed_dim), 
                                         cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_), 
                                         a_scaling, 
                                         gpu_a_loc_lora_b, a_start_lora, 
                                         a_len_lora, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, self.kv_embed_dim, 
                                         1, self.max_lora_dim, self.max_b_seq_len)
            else:
                kernel_bgmv(                                             
                    self.delta[1],
                    input_embs.view(-1, base_layer_infer.embed_dim_),
                    buffer_address,
                    a_start_lora,
                    a_len_lora,
                    gpu_a_loc_lora_a,
                    self.batch_req_bins,
                    1,                                   # qkvo
                    a_scaling,
                )

                kernel_bgmv(                                             # 2️⃣ EXPAND (B-side)
                    cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_),
                    self.delta[1],
                    buffer_address,
                    a_start_lora,
                    a_len_lora,
                    gpu_a_loc_lora_b,
                    self.batch_req_bins,
                    1,
                    a_scaling,
                )
               
        rotary_emb_fwd(cache_k, infer_state.position_cos, infer_state.position_sin)

        # v (S, H)
        torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.v_weight_,
                 out=cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_))
       
        if not no_lora_compute:
            if self.max_b_seq_len >= 200 and self.max_lora_dim >= 64 and len(infer_state.b_seq_len) >= 2:
            # if 1 ==0:
                lora_get_qkvo_fwd_shrink(input_embs.view(-1, base_layer_infer.embed_dim_), 
                                         buffer_address.view(-1, self.kv_embed_dim), 
                                         self.delta[2], gpu_a_loc_lora_a, a_start_lora, 
                                         a_len_lora, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         2, self.max_lora_dim, self.max_b_seq_len)       
                lora_get_qkvo_fwd_expand(self.delta[2], buffer_address.view(-1, self.kv_embed_dim), 
                                         cache_v.view(-1, base_model.tp_v_head_num_ * base_model.head_dim_), 
                                         a_scaling, 
                                         gpu_a_loc_lora_b, a_start_lora, 
                                         a_len_lora, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, self.kv_embed_dim, 
                                         2, self.max_lora_dim, self.max_b_seq_len)
            else:
                kernel_bgmv(                                             
                    self.delta[2],
                    input_embs.view(-1, base_layer_infer.embed_dim_),
                    buffer_address,
                    a_start_lora,
                    a_len_lora,
                    gpu_a_loc_lora_a,
                    self.batch_req_bins,
                    2,                                   # qkvo
                    a_scaling,
                )

                kernel_bgmv(                                             # 2️⃣ EXPAND (B-side)
                    cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_),
                     self.delta[2],
                    buffer_address,
                    a_start_lora,
                    a_len_lora,
                    gpu_a_loc_lora_b,
                    self.batch_req_bins,
                    2,
                    a_scaling,
                )
        return q

    def _lora_get_o(self, layer_id, input, infer_state, no_lora_compute=False)->torch.Tensor:
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]

        o = torch.mm(input.view(-1, base_layer_infer.embed_dim_),
                          base_layer_weight.o_weight_)
        if not no_lora_compute:
            delta_oA = self.delta[0]
            buffer_address, a_start_lora, a_len_lora, gpu_a_loc_lora_a, gpu_a_loc_lora_b, a_scaling = \
                self.infer_adapter.get_lora_params_at_layer(layer_id)
            if self.max_b_seq_len >= 200 and self.max_lora_dim >= 64  and len(infer_state.b_seq_len) >= 2:
            # if 1 == 0:
                lora_get_qkvo_fwd_shrink(input.view(-1, base_layer_infer.embed_dim_), 
                                         buffer_address.view(-1, self.kv_embed_dim), 
                                         delta_oA, gpu_a_loc_lora_a, a_start_lora, 
                                         a_len_lora, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         3, self.max_lora_dim, self.max_b_seq_len)
                lora_get_qkvo_fwd_expand(delta_oA, buffer_address.view(-1, self.kv_embed_dim), 
                                         o, a_scaling, 
                                         gpu_a_loc_lora_b, a_start_lora, 
                                         a_len_lora, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         3, self.max_lora_dim, self.max_b_seq_len)
            else:
                
                kernel_bgmv(delta_oA, input.view(-1, base_layer_infer.embed_dim_), 
                            buffer_address, 
                            a_start_lora, a_len_lora, 
                            gpu_a_loc_lora_a, self.batch_req_bins, 3, a_scaling)

                kernel_bgmv(o, delta_oA, buffer_address, a_start_lora, 
                            a_len_lora, gpu_a_loc_lora_b, 
                            self.batch_req_bins, 3, a_scaling)   
        return o

    # Decoding functions for inference
    def _prepare_decode_infer_state(self, batch_size, total_token_num, max_len_in_batch,
                                     input_ids, b_loc, b_loc_key, b_loc_value, b_start_loc, b_seq_len):
        """Prepare infer_state for decode: memory allocation + state setup.
        This must run OUTSIDE any CUDA graph capture since it contains CPU logic.

        When enable_cuda_graph=True, forces non-contiguous decode path so that
        KV cache writes use static-address buffers (decode_key_buffer/decode_value_buffer)
        compatible with CUDA graph capture/replay.
        """
        infer_state = self.base_model.infer_state_class()
        infer_state.is_prefill = False
        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.max_len_in_batch = max_len_in_batch
        infer_state._att_m_buffers = None  # set by CudaGraphRunner if needed
        assert (b_loc.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0])
        infer_state.b_start_loc = b_start_loc
        infer_state.b_seq_len = b_seq_len

        # When CUDA graph is enabled, force non-contiguous path.
        # The contiguous path creates views into gpu_pool with addresses that change
        # each step (decode_mem_start/end), which breaks CUDA graph replay.
        force_non_contiguous = self.enable_cuda_graph

        infer_state.mem_manager = self.base_model.mem_manager
        infer_state.b_loc_key = b_loc_key
        infer_state.b_loc_value = b_loc_value

        if not force_non_contiguous:
            alloc_mem = self.base_model.mem_manager.alloc_contiguous_kv(batch_size, PageType.KV_CACHE)
        else:
            alloc_mem = None  # skip contiguous, go to non-contiguous path

        if alloc_mem is not None:
            infer_state.decode_is_contiguous = True
            infer_state.decode_mem_index_key = alloc_mem[0]
            infer_state.decode_mem_start_key = alloc_mem[1]
            infer_state.decode_mem_end_key = alloc_mem[2]
            infer_state.decode_mem_index_value = alloc_mem[3]
            infer_state.decode_mem_start_value = alloc_mem[4]
            infer_state.decode_mem_end_value = alloc_mem[5]
            b_loc_key[:, max_len_in_batch - 1] = infer_state.decode_mem_index_key
            b_loc_value[:, max_len_in_batch - 1] = infer_state.decode_mem_index_value
            self.base_model.mem_manager.reset_b_loc_kv(b_loc_key, b_loc_value)
        else:
            infer_state.decode_is_contiguous = False
            infer_state.decode_mem_index_key = self.base_model.mem_manager.alloc(batch_size, PageType.KV_CACHE)
            infer_state.decode_mem_index_value = self.base_model.mem_manager.alloc(batch_size, PageType.KV_CACHE)
            infer_state.decode_mem_index_cat = torch.cat([infer_state.decode_mem_index_key, infer_state.decode_mem_index_value], dim=0)
            infer_state.decode_key_buffer = torch.empty(
                        (batch_size, self.base_model.tp_k_head_num_, self.base_model.head_dim_),
                        dtype=torch.float16, device="cuda")
            infer_state.decode_value_buffer = torch.empty(
                        (batch_size, self.base_model.tp_k_head_num_, self.base_model.head_dim_),
                        dtype=torch.float16, device="cuda")
            b_loc_key[:, max_len_in_batch - 1] = infer_state.decode_mem_index_key
            b_loc_value[:, max_len_in_batch - 1] = infer_state.decode_mem_index_value
            self.base_model.mem_manager.reset_b_loc_kv(b_loc_key, b_loc_value)

        infer_state.init_some_extra_state(self.base_model, batch_size, total_token_num, max_len_in_batch,
                                          input_ids, b_loc, b_start_loc, b_seq_len, False)
        return infer_state

    def _decode(self, batch_size, total_token_num, max_len_in_batch,
                input_ids, b_loc, b_loc_key, b_loc_value,
                b_start_loc, b_seq_len, no_lora_compute=False, no_lora_copy=False, print_time_profile=False):
        infer_state = self._prepare_decode_infer_state(
            batch_size, total_token_num, max_len_in_batch,
            input_ids, b_loc, b_loc_key, b_loc_value, b_start_loc, b_seq_len)

        if self.enable_cuda_graph:
            predict_logics = self._decode_with_cuda_graph(
                batch_size, max_len_in_batch, input_ids, infer_state, no_lora_compute, no_lora_copy)
        else:
            predict_logics = self._token_forward(input_ids, infer_state, no_lora_compute, no_lora_copy, print_time_profile=print_time_profile)
        return predict_logics

    def _decode_with_cuda_graph(self, batch_size, max_len_in_batch, input_ids, infer_state, no_lora_compute, no_lora_copy):
        """Execute decode forward pass using CUDA graph capture/replay."""
        if LoraUnorderedBatchMixed._shared_cuda_graph_runner is None:
            LoraUnorderedBatchMixed._shared_cuda_graph_runner = CudaGraphRunner(
                max_total_tokens=self.base_model.max_total_token_num if hasattr(self.base_model, 'max_total_token_num') else 25000,
                num_layers=self.base_model.layers_num,
                tp_q_head_num=self.base_model.tp_q_head_num_,
                tp_k_head_num=self.base_model.tp_k_head_num_,
                head_dim=self.base_model.head_dim_,
                max_graph_memory_bytes=_resolve_max_graph_memory_bytes(),
            )

        forward_kwargs = {
            'no_lora_compute': no_lora_compute,
            'no_lora_copy': no_lora_copy,
            'print_time_profile': False,
        }

        runner = LoraUnorderedBatchMixed._shared_cuda_graph_runner
        if runner.has_graph(batch_size, max_len_in_batch):
            predict_logics = runner.replay(
                batch_size, max_len_in_batch, input_ids, infer_state)
        elif runner.can_capture_more():
            predict_logics = runner.capture(
                batch_size, max_len_in_batch, self._token_forward,
                input_ids, infer_state, forward_kwargs)
        else:
            # Memory cap reached — bypass the runner and run eager.
            runner.note_capture_refused()
            predict_logics = self._token_forward(
                input_ids, infer_state,
                no_lora_compute=no_lora_compute,
                no_lora_copy=no_lora_copy,
                print_time_profile=False)
        return predict_logics
    
    @final
    def _token_forward(self, input_ids, infer_state, no_lora_compute=False, no_lora_copy=False, print_time_profile=True):
        cuda_input_ids = input_ids
        input_embs = self.base_model.pre_infer.token_forward(
                cuda_input_ids, infer_state, self.base_model.pre_post_weight)
        for i in range(self.base_model.layers_num):
            input_embs = self._lora_token_forward(i, input_embs, infer_state, no_lora_compute, no_lora_copy)
        predict_logics = self.base_model.post_infer.token_forward(
                input_embs, infer_state, self.base_model.pre_post_weight, return_logics=True)
        #predict_logics = self.sanitize_logits(predict_logics)
        return predict_logics

    @final
    # @calculate_time(show=True, min_cost_ms=0)
    def _lora_token_forward(self, layer_id, input_embs, infer_state, no_lora_compute=False, no_lora_copy=False):
        self._lora_token_attention(layer_id, input_embs, infer_state, no_lora_compute, no_lora_copy)
        layer_weight = self.base_model.trans_layers_weight[layer_id]
        layer_infer = self.base_model.layers_infer[layer_id]
        # mark_start("token_ffn")
        layer_infer._token_ffn(input_embs, infer_state, layer_weight)
        # mark_end("token_ffn")
        return input_embs

    # @calculate_time(show=True, min_cost_ms=0)
    # this impl dont to use @mark_cost_time
    def _lora_token_attention(self, layer_id, input_embs, infer_state, no_lora_compute=False, no_lora_copy=False):
        layer_weight = self.base_model.trans_layers_weight[layer_id]
        layer_infer = self.base_model.layers_infer[layer_id]
        # layer normalization
        input1 = layer_infer._att_norm(input_embs, infer_state, layer_weight)
        #self.check_invalid_probs(input1)
        # fetch k, v
        cache_k, cache_v = layer_infer._pre_cache_kv(infer_state, layer_weight)
        q = self._batch_lora_get_qkv(layer_id, input1, cache_k, cache_v, infer_state, no_lora_compute, no_lora_copy)
        input1 = None
        layer_infer._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        o = layer_infer._token_attention_kernel(q, infer_state, layer_weight)
        q = None
        o = self._batch_lora_get_o(layer_id, o, infer_state, no_lora_compute)

        input_embs.add_(o.view(-1, layer_infer.embed_dim_))
        return
    
    
    def _batch_lora_get_qkv(self, layer_id, input_embs, cache_k, cache_v, infer_state, no_lora_compute=False, no_lora_copy=False)->torch.Tensor:
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]
        if layer_id==0:
            self.infer_adapter.pin_adapters_pages()
        # q (bs, H)
        q = torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.q_weight_)
        assert(len(q)==len(self.req_bins))
        buffer_address, a_start_lora, a_len_lora, gpu_a_loc_lora_a, gpu_a_loc_lora_b, a_scaling = \
                    self.infer_adapter.get_lora_params_at_layer(layer_id)
    
        if not no_lora_compute:
            kernel_bgmv(                                             
                self.delta[0], input_embs.view(-1, base_layer_infer.embed_dim_),
                buffer_address, a_start_lora, a_len_lora, gpu_a_loc_lora_a,
                self.req_bins, 0, a_scaling,
            )

            kernel_bgmv(                                             # 2️⃣ EXPAND (B-side)
                q, self.delta[0],
                buffer_address, a_start_lora, a_len_lora, gpu_a_loc_lora_b,
                self.req_bins, 0, a_scaling,
            )

        rotary_emb_fwd(q.view(-1, base_layer_infer.tp_q_head_num_, base_model.head_dim_),
                          infer_state.position_cos, infer_state.position_sin)

        # k (bs, H)
        torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.k_weight_,
                 out=cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_))
        
        if not no_lora_compute:
            kernel_bgmv(                                             
                self.delta[1],
                input_embs.view(-1, base_layer_infer.embed_dim_),
                buffer_address,
                a_start_lora,
                a_len_lora,
                gpu_a_loc_lora_a,
                self.req_bins,
                1,                                   # qkvo
                a_scaling,
            )
            kernel_bgmv(                                             # 2️⃣ EXPAND (B-side)
                cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_),
                self.delta[1],
                buffer_address,
                a_start_lora,
                a_len_lora,
                gpu_a_loc_lora_b,
                self.req_bins,
                1,
                a_scaling,
            )
            # delta_kA = None
            # mark_end("get_k")

        rotary_emb_fwd(cache_k, infer_state.position_cos, infer_state.position_sin)

        # v (bs, H)
        torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.v_weight_,
                 out=cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_))

        if not no_lora_compute:
            # mark_start("get_v")
            kernel_bgmv(                                             
                self.delta[2],
                input_embs.view(-1, base_layer_infer.embed_dim_),
                buffer_address,
                a_start_lora,
                a_len_lora,
                gpu_a_loc_lora_a,
                self.req_bins,
                2,                                   # qkvo
                a_scaling,
            )

            kernel_bgmv(                                             # 2️⃣ EXPAND (B-side)
                cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_),
                    self.delta[2],
                buffer_address,
                a_start_lora,
                a_len_lora,
                gpu_a_loc_lora_b,
                self.req_bins,
                2,
                a_scaling,
            )  
        return q
    
    def _batch_lora_get_o(self, layer_id, input, infer_state, no_lora_compute=False)->torch.Tensor:
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]

        o = torch.mm(input.view(-1, base_layer_infer.embed_dim_),
                          base_layer_weight.o_weight_)
        #self.check_invalid_probs(o)
        if not no_lora_compute:
            # mark_start("get_o")
            buffer_address, a_start_lora, a_len_lora, gpu_a_loc_lora_a, gpu_a_loc_lora_b, a_scaling = \
                self.infer_adapter.get_lora_params_at_layer(layer_id)
            delta_oA = self.delta[0]
            kernel_bgmv(delta_oA, input.view(-1, base_layer_infer.embed_dim_), 
                            buffer_address, 
                            a_start_lora, a_len_lora, 
                            gpu_a_loc_lora_a, self.req_bins, 3, a_scaling)
            #self.check_invalid_probs(delta_oA)
            kernel_bgmv(o, delta_oA, buffer_address, a_start_lora, 
                            a_len_lora, gpu_a_loc_lora_b, 
                            self.req_bins, 3, a_scaling) 


        if layer_id==self.base_model.layers_num-1:
            self.infer_adapter.unpin_adapters_pages() 
   
        return o