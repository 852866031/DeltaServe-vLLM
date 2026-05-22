import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np

from dserve.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from einops import rearrange
from dserve.models.llama.infer_struct import LlamaInferStateInfo
from dserve.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from dserve.common.basemodel import PostLayerInferTpl

class LlamaPostLayerInfer(PostLayerInferTpl):
    """
    """

    def __init__(self, tp_rank, world_size, network_config, mode):
        super().__init__(tp_rank, world_size, network_config, mode)
        self.eps_ = network_config["rms_norm_eps"]
        self.vocab_size_ = network_config["vocab_size"]
        self.embed_dim_ = network_config["n_embed"]
        return
    
    def _norm(self, input, infer_state, layer_weight:LlamaPreAndPostLayerWeight) -> torch.Tensor:
        return rmsnorm_forward(input, layer_weight.final_norm_weight_, eps=self.eps_)

    def soft_max(self, data):
        return torch.softmax(data.permute(1, 0).float(), dim=-1)

    def token_forward(self, input_embdings, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight, return_logics=False):
        batch_size = infer_state.batch_size
        last_input = torch.empty((batch_size, self.embed_dim_), device=input_embdings.device, dtype=torch.float16)
        if infer_state.is_prefill:
            last_index = torch.cumsum(infer_state.b_seq_len, dim=0, dtype=torch.long) - 1
            last_input[:, :] = input_embdings[last_index, :]
        else:
            last_input[:, :] = input_embdings[-batch_size:, :]
        input_embdings = None
        last_input = self._norm(last_input, infer_state, layer_weight)
        last_input = rearrange(last_input, "batch embed_dim -> embed_dim batch").contiguous().reshape(-1, batch_size)
        logic_batch = torch.mm(layer_weight.lm_head_weight_, last_input)
        last_input = None
        if self.world_size_ == 1:
            gather_data = logic_batch
        else:
            gather_data = torch.empty((self.vocab_size_, batch_size), device=logic_batch.device, dtype=torch.float16)
            split_size = self.vocab_size_ // self.world_size_
            dist.all_gather([gather_data[i * split_size: (i + 1) * split_size, :]
                            for i in range(self.world_size_)], logic_batch, group=None, async_op=False)
        logic_batch = None

        if not return_logics:
            prob_out = self.soft_max(gather_data)
            gather_data = None
            return prob_out
        else:
            ans_logics = gather_data.permute(1, 0).float()
            gather_data = None
            return ans_logics   
    
    def token_forward_with_finetune_outputs(self, input_embdings, finetune_logits_per_request, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight):
        batch_size = infer_state.batch_size
        embed_dim = self.embed_dim_

        token_mask = infer_state.finetune_mask  # [total_token_num], bool tensor

        # Project all finetune token embeddings
        finetune_input = input_embdings[token_mask]                          # [N, D]
        finetune_input = self._norm(finetune_input, infer_state, layer_weight)  # [N, D]
        finetune_logits = torch.mm(finetune_input, layer_weight.lm_head_weight_.T)  # [N, vocab_size]

        # Compute last-token logits for all requests
        last_input = torch.empty((batch_size, embed_dim), device=input_embdings.device, dtype=torch.float16)
        if infer_state.is_prefill:
            last_index = torch.cumsum(infer_state.b_seq_len, dim=0, dtype=torch.long) - 1
            last_input[:, :] = input_embdings[last_index, :]
        else:
            last_input[:, :] = input_embdings[-batch_size:, :]

        last_input = self._norm(last_input, infer_state, layer_weight)  # [B, D]
        last_token_logits = torch.mm(last_input, layer_weight.lm_head_weight_.T)  # [B, vocab_size]

        # Split finetune logits by request using start/length info
        token_idx = 0
        for i in range(batch_size):
            start = infer_state.b_start_loc[i].item()
            length = infer_state.b_seq_len[i].item()
            if torch.any(token_mask[start : start + length]):
                finetune_logits_per_request.append(finetune_logits[token_idx : token_idx + length])
                token_idx += length

        return last_token_logits