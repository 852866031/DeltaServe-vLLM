import torch
import math
import numpy as np
from dserve.common.basemodel import TransformerLayerWeight

from dserve.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight

class Llama3TransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)
        return
    
    def load_hf_weights(self, weights, dummy=False):
        self._load_qkvo_weights(weights)
        self._load_ffn_weights(weights)

    def _load_qkvo_weights(self, weights):
        # input layernorm params
        key_ln = f"model.layers.{self.layer_num_}.input_layernorm.weight"
        if key_ln in weights:
            self.att_norm_weight_ = self._cuda(weights[key_ln])

        hidden_size = self.network_config_["hidden_size"]
        n_heads = self.network_config_["num_attention_heads"]
        n_kv_heads = self.network_config_.get("num_key_value_heads", None)
        if n_kv_heads is None:
            # fallback to MHA behavior
            n_kv_heads = n_heads

        assert hidden_size % n_heads == 0, "hidden_size must be divisible by num_attention_heads"
        head_dim = hidden_size // n_heads

        # TP sharding strategy assumed by your Llama3TpPartModel:
        # num_attention_heads % world_size == 0 and num_key_value_heads % world_size == 0
        assert hidden_size % self.world_size_ == 0, "hidden_size must be divisible by world_size"
        assert n_kv_heads % self.world_size_ == 0, "num_key_value_heads must be divisible by world_size for KV sharding"

        # Q out dim == hidden_size
        split_q_out = hidden_size // self.world_size_

        # KV out dim == num_key_value_heads * head_dim
        kv_out = n_kv_heads * head_dim
        assert kv_out % self.world_size_ == 0, "kv_out must be divisible by world_size"
        split_kv_out = kv_out // self.world_size_

        # q_proj
        key_q = f"model.layers.{self.layer_num_}.self_attn.q_proj.weight"
        if key_q in weights:
            wq = weights[key_q][split_q_out * self.tp_rank_: split_q_out * (self.tp_rank_ + 1), :]
            self.q_weight_ = self._cuda(wq.transpose(0, 1))

        # k_proj (GQA: smaller out dim)
        key_k = f"model.layers.{self.layer_num_}.self_attn.k_proj.weight"
        if key_k in weights:
            wk = weights[key_k][split_kv_out * self.tp_rank_: split_kv_out * (self.tp_rank_ + 1), :]
            self.k_weight_ = self._cuda(wk.transpose(0, 1))

        # v_proj (GQA: smaller out dim)
        key_v = f"model.layers.{self.layer_num_}.self_attn.v_proj.weight"
        if key_v in weights:
            wv = weights[key_v][split_kv_out * self.tp_rank_: split_kv_out * (self.tp_rank_ + 1), :]
            self.v_weight_ = self._cuda(wv.transpose(0, 1))

        # o_proj: same sharding scheme as LLaMA (column split on input dimension)
        key_o = f"model.layers.{self.layer_num_}.self_attn.o_proj.weight"
        if key_o in weights:
            wo = weights[key_o][:, split_q_out * self.tp_rank_: split_q_out * (self.tp_rank_ + 1)]
            self.o_weight_ = self._cuda(wo.transpose(0, 1))