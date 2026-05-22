import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
import triton

from dserve.models.llama3.layer_weights.transformer_layer_weight import Llama3TransformerLayerWeight
from dserve.models.llama2.triton_kernel.context_flashattention_nopad import context_attention_fwd
from dserve.models.llama.triton_kernel.token_attention_nopad_att1 import token_att_fwd
from dserve.models.llama2.triton_kernel.token_attention_nopad_softmax import token_softmax_fwd
from dserve.models.llama2.triton_kernel.token_attention_nopad_reduceV import token_att_fwd2
from dserve.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from dserve.models.llama.infer_struct import LlamaInferStateInfo
from dserve.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer

class Llama3TransformerLayerInfer(LlamaTransformerLayerInfer):

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)

        # Override KV head counts for GQA
        kv_heads = network_config.get("num_key_value_heads", None)
        if kv_heads is None:
            kv_heads = network_config["num_attention_heads"]  # fallback to MHA

        assert kv_heads % self.world_size_ == 0, "KV-head sharding requires num_key_value_heads % world_size == 0"

        self.tp_k_head_num_ = kv_heads // self.world_size_
        self.tp_v_head_num_ = self.tp_k_head_num_

        # Convenience invariant (often required by GQA kernels)
        assert self.tp_q_head_num_ % self.tp_k_head_num_ == 0, "GQA grouping must be integer per TP rank"
        self.kv_group_size_ = self.tp_q_head_num_ // self.tp_k_head_num_

        return

    def _get_qkv(
        self,
        input,
        cache_k,
        cache_v,
        infer_state: LlamaInferStateInfo,
        layer_weight: Llama3TransformerLayerWeight,
    ) -> torch.Tensor:
        # Q: out dim == tp_q_heads * head_dim
        q = torch.mm(input.view(-1, self.embed_dim_), layer_weight.q_weight_)
        rotary_emb_fwd(q.view(-1, self.tp_q_head_num_, self.head_dim_), infer_state.position_cos, infer_state.position_sin)
        torch.mm(
            input.view(-1, self.embed_dim_),
            layer_weight.k_weight_,
            out=cache_k.view(-1, self.tp_k_head_num_ * self.head_dim_),
        )
        rotary_emb_fwd(
            cache_k.view(-1, self.tp_k_head_num_, self.head_dim_),
            infer_state.position_cos,
            infer_state.position_sin,
        )
        torch.mm(
            input.view(-1, self.embed_dim_),
            layer_weight.v_weight_,
            out=cache_v.view(-1, self.tp_v_head_num_ * self.head_dim_),
        )
        return q
    
    # gqa attention
    def _context_attention_kernel(self, q, k, v, infer_state: LlamaInferStateInfo, layer_weight:Llama3TransformerLayerWeight) -> torch.Tensor:
        o_tensor = torch.empty_like(q)
        context_attention_fwd(q.view(-1, self.tp_q_head_num_, self.head_dim_),
                              k.view(-1, self.tp_k_head_num_, self.head_dim_),
                              v.view(-1, self.tp_v_head_num_, self.head_dim_),
                              o_tensor.view(-1, self.tp_q_head_num_, self.head_dim_),
                              infer_state.b_start_loc,
                              infer_state.b_seq_len,
                              infer_state.max_len_in_batch)
        return o_tensor
