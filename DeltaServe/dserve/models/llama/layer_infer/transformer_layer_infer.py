import time
from dserve.common.unified_mem_allocator import UnifiedMemoryAllocator
import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
from typing import Tuple
import triton

from dserve.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from dserve.models.llama.triton_kernel.context_flashattention_nopad import context_attention_fwd
from dserve.models.llama.triton_kernel.token_attention_nopad_att1 import token_att_fwd, token_att_fwd_int8k
from dserve.models.llama.triton_kernel.token_attention_nopad_softmax import token_softmax_fwd
from dserve.models.llama.triton_kernel.token_attention_nopad_reduceV import token_att_fwd2, token_att_fwd2_int8v
from dserve.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from dserve.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd

from dserve.models.llama.infer_struct import LlamaInferStateInfo
from dserve.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv, destindex_copy_quantize_kv
from dserve.common.basemodel import TransformerLayerInferTpl

@torch.no_grad()
def context_attention_fwd_pytorch(q, k, v, o, b_start_loc, b_seq_len, max_input_len):
    B = b_seq_len.shape[0]
    S, H, D = k.shape
    hidden_dim = H * D
    scale = 1.0 / (D ** 0.5)
    q = q.view(S, H, D)

    for i in range(B):
        start = b_start_loc[i].item()
        seqlen = b_seq_len[i].item()
        end = start + seqlen

        q_i = q[start:end]
        k_i = k[start:end]
        v_i = v[start:end]
        q_i = q_i.transpose(0, 1)  # [H, L, D]
        k_i = k_i.transpose(0, 1)  # [H, L, D]
        v_i = v_i.transpose(0, 1)  # [H, L, D]

        attn_scores = torch.matmul(q_i, k_i.transpose(-1, -2)) * scale

        mask = torch.tril(torch.ones((seqlen, seqlen), dtype=torch.bool, device=q.device))
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

        attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_probs, v_i)  # [H, L, D]
        context = context.transpose(0, 1).contiguous().view(seqlen, hidden_dim)

        # Write into output buffer
        o[start:end].copy_(context)
 

class LlamaTransformerLayerInfer(TransformerLayerInferTpl):
    """
    """

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        self.eps_ = network_config["rms_norm_eps"]
        self.tp_q_head_num_ = network_config["num_attention_heads"] // self.world_size_
        self.tp_k_head_num_ = self.tp_q_head_num_
        self.tp_v_head_num_ = self.tp_q_head_num_
        self.tp_o_head_num_ = self.tp_q_head_num_
        self.head_dim_ = network_config["hidden_size"] // network_config["num_attention_heads"]
        self.embed_dim_ = network_config["hidden_size"]
        return

    
    def _att_norm(self, input, infer_state:LlamaInferStateInfo, layer_weight:LlamaTransformerLayerWeight)->torch.Tensor:
        tmp = rmsnorm_forward(input, weight=layer_weight.att_norm_weight_, eps=self.eps_)
        return tmp
    
    def _ffn_norm(self, input, infer_state:LlamaInferStateInfo, layer_weight:LlamaTransformerLayerWeight)->torch.Tensor:
        return rmsnorm_forward(input, weight=layer_weight.ffn_norm_weight_, eps=self.eps_)

    def _get_qkv(self, input, cache_k, cache_v, infer_state:LlamaInferStateInfo, layer_weight:LlamaTransformerLayerWeight)->torch.Tensor:
        q = torch.mm(input.view(-1, self.embed_dim_), layer_weight.q_weight_)
        rotary_emb_fwd(q.view(-1, self.tp_q_head_num_, self.head_dim_), infer_state.position_cos, infer_state.position_sin)
        torch.mm(input.view(-1, self.embed_dim_), layer_weight.k_weight_,
                    out=cache_k.view(-1, self.tp_k_head_num_ * self.head_dim_))
        rotary_emb_fwd(cache_k, infer_state.position_cos, infer_state.position_sin)
        torch.mm(input.view(-1, self.embed_dim_), layer_weight.v_weight_,
                    out=cache_v.view(-1, self.tp_v_head_num_ * self.head_dim_))
        return q
    
    def _post_cache_kv(self, cache_k, cache_v, infer_state:LlamaInferStateInfo, layer_weight:LlamaTransformerLayerWeight):
        if not infer_state.is_prefill and infer_state.decode_is_contiguous:
            return
        if infer_state.is_prefill:
            mem_manager = infer_state.mem_manager
            mem_manager.pin_pages(infer_state.prefill_mem_index_cat)
            key_mem_index = mem_manager.to_gpu_index(infer_state.prefill_mem_index_key)
            value_mem_index = mem_manager.to_gpu_index(infer_state.prefill_mem_index_value)
            destindex_copy_kv(cache_k, key_mem_index, mem_manager.get_kv_pool(self.layer_num_))
            destindex_copy_kv(cache_v, value_mem_index, mem_manager.get_kv_pool(self.layer_num_))
            mem_manager.unpin_pages(infer_state.prefill_mem_index_cat)
            return
        else:
            if not infer_state.decode_is_contiguous:
                pass  # non-contiguous decode path (used by CUDA graph)
                mem_manager = infer_state.mem_manager
                mem_manager.pin_pages(infer_state.decode_mem_index_key)
                key_mem_index = mem_manager.to_gpu_index(infer_state.decode_mem_index_key)
                value_mem_index = mem_manager.to_gpu_index(infer_state.decode_mem_index_value)
                destindex_copy_kv(cache_k, key_mem_index, mem_manager.get_kv_pool(self.layer_num_))
                destindex_copy_kv(cache_v, value_mem_index, mem_manager.get_kv_pool(self.layer_num_))
                return
        return
    
    def _context_attention_kernel(self, q, k, v, infer_state:LlamaInferStateInfo, layer_weight)->torch.Tensor:
        o_tensor = torch.empty_like(q)
        context_attention_fwd(q.view(-1, self.tp_q_head_num_, self.head_dim_),
                              k.view(-1, self.tp_k_head_num_, self.head_dim_),
                              v.view(-1, self.tp_v_head_num_, self.head_dim_),
                              o_tensor.view(-1, self.tp_q_head_num_, self.head_dim_),
                              infer_state.b_start_loc,
                              infer_state.b_seq_len,
                              infer_state.max_len_in_batch)
        return o_tensor
    
    def _token_attention_kernel(self, q, infer_state:LlamaInferStateInfo, layer_weight)->torch.Tensor:
        return self._token_decode_attention_mode(q, infer_state)

    def _get_o(self, input, infer_state:LlamaInferStateInfo, layer_weight:LlamaTransformerLayerWeight)->torch.Tensor:
        o_tensor = torch.mm(input.view(-1, self.tp_o_head_num_ * self.head_dim_), layer_weight.o_weight_)
        return o_tensor

    def _ffn(self, input, infer_state:LlamaInferStateInfo, layer_weight:LlamaTransformerLayerWeight)->torch.Tensor:
        gate_out = torch.mm(input.view(-1, self.embed_dim_), layer_weight.gate_proj)
        torch.nn.functional.silu(gate_out, inplace=True)
        up_out = torch.mm(input.view(-1, self.embed_dim_), layer_weight.up_proj)
        input = None
        ffn1_out = gate_out * up_out
        gate_out, up_out = None, None
        ffn2_out = torch.mm(ffn1_out, layer_weight.down_proj)
        ffn1_out = None
        return ffn2_out
    
    def _copy_kv_to_mem_cache(self, key_buffer, value_buffer, mem_index, mem_manager):
        if "int8kv" in self.mode:
            destindex_copy_quantize_kv(key_buffer,
                                       mem_index,
                                       mem_manager.key_buffer[self.layer_num_],
                                       mem_manager.key_scale_buffer[self.layer_num_])
            #print("Access key buffer from LlamaTransformerLayerInfer")
            destindex_copy_quantize_kv(value_buffer,
                                       mem_index,
                                       mem_manager.value_buffer[self.layer_num_],
                                       mem_manager.value_scale_buffer[self.layer_num_])
            #print("Access value buffer from LlamaTransformerLayerInfer")
        else:
            destindex_copy_kv(key_buffer, mem_index, mem_manager.key_buffer[self.layer_num_])
            destindex_copy_kv(value_buffer, mem_index, mem_manager.value_buffer[self.layer_num_])
            #print("Access value buffer from LlamaTransformerLayerInfer")
        
    def _token_decode_attention_mode(self, q, infer_state: LlamaInferStateInfo):
        if "int8kv" in self.mode:
            return self._token_decode_attention_int8kv(q, infer_state)
        else:
            o = self._token_decode_attention_normal(q, infer_state)
            return o

    def _token_decode_attention_normal(self, q, infer_state: LlamaInferStateInfo):
        total_token_num = infer_state.total_token_num
        batch_size = infer_state.batch_size
        calcu_shape1 = (batch_size, self.tp_q_head_num_, self.head_dim_)

        # For CUDA graph compatibility, use pre-allocated att_m buffer if available
        # (avoids dynamic-shape torch.empty that would break graph replay)
        if hasattr(infer_state, '_att_m_buffers') and infer_state._att_m_buffers is not None:
            att_m_tensor = infer_state._att_m_buffers[self.layer_num_]
        else:
            att_m_tensor = torch.empty((self.tp_q_head_num_, total_token_num), dtype=q.dtype, device="cuda")

        buffer_address, gpu_b_loc_key, gpu_b_loc_value = infer_state.mem_manager.prepare_b_locs_for_layer(
                infer_state.b_loc_key, infer_state.b_loc_value, infer_state.b_seq_len, self.layer_num_)
        kv_heads = self.tp_k_head_num_
        token_att_fwd(q.view(calcu_shape1),
                          buffer_address,
                          att_m_tensor,
                          gpu_b_loc_key,
                          infer_state.b_start_loc,
                          infer_state.b_seq_len,
                          infer_state.max_len_in_batch,
                          kv_heads=kv_heads)

        o_tensor = torch.empty_like(q)
        from dserve.models.llama.triton_kernel.token_attention_softmax_and_reducev import token_softmax_reducev_fwd
        token_softmax_reducev_fwd(att_m_tensor,
                                        buffer_address,
                                        o_tensor.view(calcu_shape1),
                                        gpu_b_loc_value,
                                        infer_state.b_start_loc,
                                        infer_state.b_seq_len,
                                        infer_state.max_len_in_batch,
                                        infer_state.other_kv_index,
                                        kv_heads=kv_heads)
        return o_tensor

    def _token_decode_attention_int8kv(self, q, infer_state: LlamaInferStateInfo):
        total_token_num = infer_state.total_token_num
        batch_size = infer_state.batch_size
        calcu_shape1 = (batch_size, self.tp_q_head_num_, self.head_dim_)
        att_m_tensor = torch.empty((self.tp_q_head_num_, total_token_num), dtype=q.dtype, device="cuda")
        token_att_fwd_int8k(q.view(calcu_shape1),
                            infer_state.mem_manager.key_buffer[self.layer_num_],
                            infer_state.mem_manager.key_scale_buffer[self.layer_num_],
                            att_m_tensor,
                            infer_state.b_loc,
                            infer_state.b_start_loc,
                            infer_state.b_seq_len,
                            infer_state.max_len_in_batch)
        #print("Access key buffer from LlamaTransformerLayerInfer")

        prob = torch.empty_like(att_m_tensor)
        token_softmax_fwd(att_m_tensor, infer_state.b_start_loc, infer_state.b_seq_len, prob, infer_state.max_len_in_batch)
        att_m_tensor = None

        o_tensor = torch.empty_like(q)
        token_att_fwd2_int8v(prob,
                                infer_state.mem_manager.value_buffer[self.layer_num_],
                                infer_state.mem_manager.value_scale_buffer[self.layer_num_],
                                o_tensor.view(calcu_shape1),
                                infer_state.b_loc,
                                infer_state.b_start_loc,
                                infer_state.b_seq_len,
                                infer_state.max_len_in_batch)
        #print("Access value buffer from LlamaTransformerLayerInfer")
        prob = None
        return o_tensor