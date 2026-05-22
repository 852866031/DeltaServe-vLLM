import os
import json
from dserve.common.build_utils import repair_config
from dserve.models.llama3.SFT_service import Llama3SFTBackwardService
import torch

from dserve.models.llama3.layer_infer.transformer_layer_infer import Llama3TransformerLayerInfer
from dserve.models.llama3.layer_weights.transformer_layer_weight import Llama3TransformerLayerWeight

from dserve.models.llama.model import LlamaTpPartModel


class Llama3TpPartModel(LlamaTpPartModel):
    """
    LLaMA-3 TP-part model = LLaMA-1 implementation + GQA (num_key_value_heads).

    Key responsibilities here:
      - Ensure config exposes num_key_value_heads (fallback to MHA)
      - Set per-TP Q/K/V head counts correctly
      - Initialize KV-cache allocator using KV head count (not Q head count)
      - Swap transformer layer weight + infer implementations
    """
    # weight class
    transformer_weight_class = Llama3TransformerLayerWeight

    # infer class
    transformer_layer_infer_class = Llama3TransformerLayerInfer

    backward_service_class = Llama3SFTBackwardService

    def __init__(self, tp_rank, world_size, weight_dir,
                 max_total_token_num, mem_adapter_size, load_way="HF", mode=[],
                 dummy=False, half_model=False, mem_manager_log_path=None, unified_mem_manager_max_size=0,
                 max_finetuning_tokens=1024):
        super().__init__(tp_rank, world_size, weight_dir,
                         max_total_token_num, mem_adapter_size, load_way, mode, dummy=dummy,
                         half_model=half_model, mem_manager_log_path=mem_manager_log_path,
                         unified_mem_manager_max_size=unified_mem_manager_max_size,
                         max_finetuning_tokens=max_finetuning_tokens)

    def _init_config(self):
        super()._init_config()
        repair_config(self.config, same_names=["num_key_value_heads", "n_kv_head", "num_kv_heads"])
        if self.config.get("num_key_value_heads", None) is None:
            self.config["num_key_value_heads"] = self.config["num_attention_heads"]
        return 
    
    def _verify_must(self):
        # Parent checks num_attention_heads % world_size == 0
        super()._verify_must()
        # GQA requires attention heads be divisible by KV heads (grouping is integer)
        assert self.config["num_attention_heads"] % self.config["num_key_value_heads"] == 0, \
            "GQA requires num_attention_heads % num_key_value_heads == 0"

        # Strategy here: shard KV heads across TP ranks (recommended for memory efficiency)
        assert self.config["num_key_value_heads"] % self.world_size_ == 0, \
            "KV-head sharding requires num_key_value_heads % world_size == 0"
        return
    
    def _init_some_value(self):
        # head_dim is still defined by Q heads
        self.head_dim_ = self.config["hidden_size"] // self.config["num_attention_heads"]
        # Q heads per TP rank
        self.tp_q_head_num_ = self.config["num_attention_heads"] // self.world_size_
        # KV heads per TP rank (sharded strategy)
        self.tp_k_head_num_ = self.config["num_key_value_heads"] // self.world_size_
        self.tp_v_head_num_ = self.tp_k_head_num_
        self.tp_kv_head_num_ = self.tp_k_head_num_  # for convenience
        # For attention implementation convenience
        assert self.tp_q_head_num_ % self.tp_k_head_num_ == 0, \
            "Per-TP grouping must be integer (tp_q_head_num_ % tp_k_head_num_ == 0)"
        self.kv_group_size_ = self.tp_q_head_num_ // self.tp_k_head_num_
        self.layers_num = self.config["n_layer"]
        self.vocab_size = self.config["vocab_size"]
        return
    
    def _init_mem_manager(self):
        """
        KV cache allocator dispatch via cfg.memory.allocator. The "packed_kv"
        variant uses num_key_value_heads to pack F = num_attention_heads /
        num_key_value_heads KV slots per page; "unified" keeps the legacy
        page=1-KV-slot layout.
        """
        head_dim = self.config["hidden_size"] // self.config["num_attention_heads"]

        from dserve.common.allocator_factory import make_allocator
        from dserve.common.configs.config import get_active_config
        cfg = get_active_config()
        self.mem_manager = make_allocator(
            cfg.memory.allocator,
            head_num=self.config["num_attention_heads"],
            head_dim=head_dim,
            layer_num=self.config["num_hidden_layers"],
            vocab_size=self.config["vocab_size"],
            dtype=torch.float16,
            max_pool_size=self.unified_mem_manager_max_size,
            log_path=self.mem_manager_log_path,
            max_finetuning_tokens=self.max_finetuning_tokens,
            num_kv_heads=self.config["num_key_value_heads"],
        )
        return
    
