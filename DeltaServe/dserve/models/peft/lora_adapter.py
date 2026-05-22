import re
import torch
import os
import json
from dserve.mprophet.lora_config import get_lora_config_json
from dserve.models.peft.layer_weights.hf_load_utils import load_hf_weights
from dserve.models.peft.layer_weights.lora_layer_weight import LoraLayerWeight
from dserve.utils.model_load import hf_load_config
from dserve.server.router.mixed_req_queue import rprint


def get_lora_config_finetune():
    config = {"base_model_name_or_path": "decapoda-research/llama-7b-hf",
                  "bias": "none",
                  "enable_lora": None,
                  "fan_in_fan_out": False,
                  "inference_mode": True,
                  "lora_alpha": 16,
                  "lora_dropout": 0.05,
                  "merge_weights": False,
                  "modules_to_save": None,
                  "peft_type": "LORA",
                  "r": 16,
                  "target_modules": [
                  "q_proj",
                  "k_proj",
                  "v_proj",
                  "o_proj"
                  ],
                  "task_type": "CAUSAL_LM"
                 }
    return config


def get_lora_config(lora_dir, dummy):
    if dummy:
        return get_lora_config_json(lora_dir), lora_dir
    else:
        lora_dir = re.sub(r'-(\d+)$', '', lora_dir)
        return hf_load_config(lora_dir)

        

class LoraTpPartAdapter:

    def __init__(self, tp_rank, world_size, lora_dir, network_config,
                 swap=False, dummy=False, no_lora_swap=False, prefetch_stream=None, is_finetuning=False):
        assert world_size == 1
        self.tp_rank_ = tp_rank
        self.world_size_ = world_size
        self.lora_dir = lora_dir
        self.network_config = network_config
        self.lora_config, lora_dir = get_lora_config(lora_dir, dummy)
        self.r = self.lora_config["r"]
        self.lora_alpha = self.lora_config["lora_alpha"]
        self.scaling = self.lora_alpha / self.r
        self.is_finetuning = is_finetuning
        
        self.layers = [
            LoraLayerWeight(i, tp_rank, world_size, self.lora_config, network_config, torch.float16,
                            no_lora_swap=no_lora_swap, prefetch_stream=prefetch_stream, is_finetuning=is_finetuning)
            for i in range(network_config["num_hidden_layers"])
        ]

        self.prefetch_stream = prefetch_stream
        print("Loading adapter ", lora_dir)
        load_hf_weights("fp16", lora_dir, transformer_layer_list=self.layers,
                        swap=swap, dummy=dummy)
        
    def check_all_lora_b_zero(self):
        for id , layer in enumerate(self.layers):
            if torch.all(layer.q_lora_B_home == 0) and torch.all(layer.k_lora_B_home == 0) \
                and torch.all(layer.v_lora_B_home == 0) and torch.all(layer.o_lora_B_home == 0):
                print(f"LoRA B are all zeros  in layer {id}")
            else:
                rprint(f"[Not all zeros] in layer {id}")

    def is_on_gpu(self,):
        return (self.layers[0].w_combined is not None)


    def load_to_gpu(self, prefetch=False, bmm=False, both=False):
        #print("Loading LoRA weights to GPU")
        if prefetch:
            with self.prefetch_stream:
                for layer_weight in self.layers:
                    layer_weight.load_to_gpu(bmm=bmm, both=both)
        else:
            for layer_weight in self.layers:
                layer_weight.load_to_gpu(bmm=bmm, both=both)

    # def load_gpu_fp32_dict(self):
    #     adapter_dict = {
    #         "scaling": self.scaling,
    #         "lora_weights": [layer.w_combined_home.float().to("cuda", non_blocking=True).requires_grad_() for layer in self.layers]
    #     }
    #     return adapter_dict
    
    def load_gpu_fp32_dict(self):
        adapter_dict = {
            "scaling": self.scaling,
            "lora_weights": [layer.w_combined_home_fp32 for layer in self.layers]
        }
        return adapter_dict


    def offload_from_gpu(self):
        for layer_weight in self.layers:
            layer_weight.offload_from_gpu()

    def get_all_items(self):
        all_items = {}
        for layer, layer_weight in enumerate(self.layers):
            all_items[f"layer_{layer}.q_lora_A"] = layer_weight.q_lora_A
            all_items[f"layer_{layer}.k_lora_A"] = layer_weight.k_lora_A
            all_items[f"layer_{layer}.v_lora_A"] = layer_weight.v_lora_A
            all_items[f"layer_{layer}.o_lora_A"] = layer_weight.o_lora_A
            all_items[f"layer_{layer}.q_lora_B"] = layer_weight.q_lora_B
            all_items[f"layer_{layer}.k_lora_B"] = layer_weight.k_lora_B
            all_items[f"layer_{layer}.v_lora_B"] = layer_weight.v_lora_B
            all_items[f"layer_{layer}.o_lora_B"] = layer_weight.o_lora_B
        return all_items
