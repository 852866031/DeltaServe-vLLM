from dataclasses import dataclass
import numpy as np
from dserve.common.unified_mem_allocator import PageType, UnifiedMemoryAllocator
import torch
from typing import List, Dict, Any
import time

from dserve.common.mem_allocator import MemoryAllocator
from dserve.utils.infer_utils import calculate_time, mark_start, mark_end


@dataclass
class InferAdapter:
    adapter_dirs: List[str]  # all adapters on the server
    a_loc: torch.Tensor  # a_loc[i] is a list of indices occupied by adapter i
    a_start: torch.Tensor  # a_start[i] is the start location of adapter i
    a_len: torch.Tensor  # a_len[i] is the number of cells occupied by adapter i
    a_scaling: torch.Tensor  # a_scaling[i] is the scaling factor of adapter i
    mem_manager: MemoryAllocator

    # adapter_ids_lora_a: List[int]  # adapter_ids_lora_a[i] is the indices of adapter i in a_loc_lora_a
    # adapter_ids_lora_b: List[int]  # adapter_ids_lora_b[i] is the indices of adapter i in a_loc_lora_b

    idx_map: Dict[str, int]
    prefetch_tag: Dict[str, int]
    cur_tag: int
    print_loading: bool
    prefetch_stream: Any

    @classmethod
    def init(cls, mem_manager, prefetch_stream):
        return cls(
            adapter_dirs=[],
            a_loc=torch.empty(0, dtype=torch.long, device="cuda"),
            a_start=torch.empty(0, dtype=torch.long, device="cuda"),
            a_len=torch.empty(0, dtype=torch.long, device="cuda"),
            a_scaling=torch.empty(0, dtype=torch.float16, device="cuda"),
            mem_manager=mem_manager,
            idx_map={},
            prefetch_tag={},
            cur_tag=0,
            print_loading=True,
            prefetch_stream=prefetch_stream,
        )


    def get_lora_a_at_layer(self,
                            adapter_dir: str,
                            layer_id:    int) -> List[torch.Tensor]:

        # 1) Locate the adapter in the bookkeeping tensors
        try:
            adapter_idx = self.idx_map[adapter_dir]
        except KeyError:
            raise ValueError(f"Adapter '{adapter_dir}' is not loaded.")

        start  = int(self.a_start[adapter_idx].item())
        length = int(self.a_len[adapter_idx].item())        # rows == 4*r
        vpids  = self.a_loc[start : start + length]         # Tensor[int64]

        # 2) Grab the rows from mem_manager.key_buffer for this layer
        #    key_buffer[layer] has shape [ pool_size, hidden_dim ]
        rows_tensor = self.mem_manager.key_buffer[layer_id][vpids]

        # 3) Clone each row so the caller owns an independent copy
        return [row.clone() for row in rows_tensor]
    
    def get_lora_b_at_layer(self,
                            adapter_dir: str,
                            layer_id:    int) -> List[torch.Tensor]:

        # 1) Locate the adapter in the bookkeeping tensors
        try:
            adapter_idx = self.idx_map[adapter_dir]
        except KeyError:
            raise ValueError(f"Adapter '{adapter_dir}' is not loaded.")

        start  = int(self.a_start[adapter_idx].item())
        length = int(self.a_len[adapter_idx].item())        # rows == 4*r
        vpids  = self.a_loc[start : start + length]         # Tensor[int64]

        # 2) Grab the rows from mem_manager.key_buffer for this layer
        #    key_buffer[layer] has shape [ pool_size, hidden_dim ]
        rows_tensor = self.mem_manager.value_buffer[layer_id][vpids]

        # 3) Clone each row so the caller owns an independent copy
        return [row.clone() for row in rows_tensor]

    # @calculate_time(show=True, min_cost_ms=0)
    def load_lora_A(self, adapter, loc, prefetch=False):
        r = adapter.r
        h = adapter.network_config["hidden_size"]
        head_num = adapter.network_config["num_attention_heads"]
        head_dim = h // head_num

        for i in range(adapter.network_config["num_hidden_layers"]):
            adapter.layers[i].load_to_gpu(prefetch=prefetch)
            w_combined = adapter.layers[i].w_combined
            self.mem_manager.key_buffer[i][loc] = w_combined[0]
            #print("Access key buffer from InferAdapter")
            adapter.layers[i].offload_from_gpu()


    # @calculate_time(show=True, min_cost_ms=0)
    def load_lora_B(self, adapter, loc, prefetch=False):
        r = adapter.r
        h = adapter.network_config["hidden_size"]
        head_num = adapter.network_config["num_attention_heads"]
        head_dim = h // head_num
        for i in range(adapter.network_config["num_hidden_layers"]):
            adapter.layers[i].load_to_gpu(prefetch=prefetch)
            w_combined = adapter.layers[i].w_combined
            self.mem_manager.value_buffer[i][loc] = w_combined[1]
            adapter.layers[i].offload_from_gpu()

    # @calculate_time(show=True, min_cost_ms=0)
    def load_adapters(self, adapters, prefetch=False):
        # func_name = "realload" if not prefetch else "prefetch"
        # mark_start(func_name)
        if len(adapters) == 0:
            if self.print_loading: print(f"load 0 adapters, {len(self.adapter_dirs)} in total")
            return

        if prefetch:
            self.cur_tag ^= 1
            capacity = self.mem_manager.can_use_mem_size
            new_adapters = []
            tot_size = 0
            # mark_start("load scan")
            for adapter in adapters:
                self.prefetch_tag[adapter.lora_dir] = self.cur_tag
                if adapter is not None and adapter.lora_dir not in self.idx_map:
                    if tot_size + adapter.r * 4 > capacity:
                        break
                    new_adapters.append(adapter)
                    tot_size += adapter.r * 4
            # mark_end("load scan")
            if self.print_loading: print(f"prefetch {len(new_adapters)} adapters, "
                  f"{len(self.adapter_dirs) + len(new_adapters)} in total")
        else:
            new_adapters = []
            tot_size = 0
            # mark_start("load scan")
            for adapter in adapters:
                if adapter is not None and adapter.lora_dir not in self.idx_map:
                    new_adapters.append(adapter)
                    tot_size += adapter.r * 4
            # mark_end("load scan")
            if self.print_loading: print(f"load {len(new_adapters)} adapters, {len(self.adapter_dirs) + len(new_adapters)} in total")

        new_loc = self.mem_manager.alloc(tot_size)
        # assert len(new_loc) == tot_size
        start_offset = self.a_start.shape[0]
        self.a_start = torch.cat((self.a_start, torch.empty(len(new_adapters,), dtype=torch.long, device="cuda")))
        len_offset = self.a_len.shape[0]
        self.a_len = torch.cat((self.a_len, torch.empty(len(new_adapters,), dtype=torch.long, device="cuda")))
        loc_offset = self.a_loc.shape[0]
        self.a_loc = torch.cat((self.a_loc, torch.empty(tot_size, dtype=torch.long, device="cuda")))

        cum_loc = 0
        cum_loc_list = []
        for i, new_adapter in enumerate(new_adapters):
            cum_loc_list.append(cum_loc)
            self.idx_map[new_adapter.lora_dir] = len(self.adapter_dirs)
            self.adapter_dirs.append(new_adapter.lora_dir)
            self.a_start[start_offset + i] = loc_offset + cum_loc
            self.a_len[len_offset + i] = new_adapter.r * 4
            self.a_loc[loc_offset + cum_loc: loc_offset + cum_loc + new_adapter.r * 4] = (
                    new_loc[cum_loc: cum_loc + new_adapter.r * 4])
            cum_loc += new_adapter.r * 4
        self.a_scaling = torch.cat((self.a_scaling, torch.tensor([adapter.scaling for adapter in new_adapters], dtype=torch.float16, device="cuda")))

        if prefetch:
            with torch.cuda.stream(self.prefetch_stream):
                new_loc = new_loc.clone()
                for i, new_adapter in enumerate(new_adapters):

                    cum_loc = cum_loc_list[i]
                    self.load_lora_A(new_adapter, new_loc[cum_loc: cum_loc + new_adapter.r * 4], prefetch)
                    self.load_lora_B(new_adapter, new_loc[cum_loc: cum_loc + new_adapter.r * 4], prefetch)

        else:
            for i, new_adapter in enumerate(new_adapters):
                cum_loc = cum_loc_list[i]
                self.load_lora_A(new_adapter, new_loc[cum_loc: cum_loc + new_adapter.r * 4], prefetch)
                self.load_lora_B(new_adapter, new_loc[cum_loc: cum_loc + new_adapter.r * 4], prefetch)

    def offload_target_adapters(self, remove_adapter_dirs):
        print(f"Offloading adapters: {remove_adapter_dirs}")
        reserve_adapter_dirs = set(self.adapter_dirs) - set(remove_adapter_dirs)
        self.offload_adapters(reserve_adapter_dirs)
        

    # @calculate_time(show=True, min_cost_ms=0)
    def offload_adapters(self, reserve_adapter_dirs):
        if len(reserve_adapter_dirs) == len(self.adapter_dirs):
            if self.print_loading: print(f"offload 0 adapters, {len(self.adapter_dirs)} remains")
            return
        if len(reserve_adapter_dirs) == 0:
            if self.print_loading: print(f"offload {len(self.adapter_dirs)} adapters, 0 remains")
            self.mem_manager.free(self.a_loc)
            self.adapter_dirs=[]
            self.a_loc=torch.empty(0, dtype=torch.long, device="cuda")
            self.a_start=torch.empty(0, dtype=torch.long, device="cuda")
            self.a_len=torch.empty(0, dtype=torch.long, device="cuda")
            self.a_scaling=torch.empty(0, dtype=torch.float16, device="cuda")
            self.idx_map={}
            return

        # mark_start("offload scan")
        remove_ind = []
        left_ind = []
        new_adapter_dirs = []
        self.idx_map = {}
        removed_adapter_dirs = []
        for i, adapter_dir in enumerate(self.adapter_dirs):
            if (adapter_dir not in reserve_adapter_dirs and
                (adapter_dir not in self.prefetch_tag or
                 self.prefetch_tag[adapter_dir] != self.cur_tag)):
                remove_ind.append(self.a_loc[self.a_start[i]:self.a_start[i] + self.a_len[i]])
                removed_adapter_dirs.append(adapter_dir)
            else:
                left_ind.append(i)
                self.idx_map[adapter_dir] = len(new_adapter_dirs)
                new_adapter_dirs.append(adapter_dir)
        if len(remove_ind) == 0:
            return
        # mark_end("offload scan")
        self.adapter_dirs = new_adapter_dirs
        tot_size = torch.sum(self.a_len[left_ind]).item()
        if self.print_loading: print(f"offload {len(remove_ind)} adapters, {len(left_ind)} remains")

        # mark_start("offload cat")
        remove_ind = torch.cat(remove_ind)
        # mark_end("offload cat")
        # release memory
        # mark_start("offload free mem manager")
        self.mem_manager.free(remove_ind)
        # mark_end("offload free mem manager")
        
        # reset indexing
        # mark_start("offload torch.empty")
        new_a_len = torch.empty(len(left_ind), dtype=torch.long, device="cuda")
        new_a_start = torch.empty(len(left_ind), dtype=torch.long, device="cuda")
        new_a_scaling = torch.empty(len(left_ind), dtype=torch.float16, device="cuda")
        new_a_loc = torch.empty(tot_size, dtype=torch.long, device="cuda")
        # mark_end("offload torch.empty")

        new_a_len[:] = self.a_len[left_ind]
        new_a_start[0] = 0
        new_a_start[1:] = torch.cumsum(new_a_len, dim=0)[:-1]
        new_a_scaling[:] = self.a_scaling[left_ind]
        # mark_start("offload a_loc update")
        launch_var_len_copy_triton(self.a_start[left_ind], new_a_len,
                                   self.a_loc, new_a_start, new_a_loc)
        # mark_end("offload a_loc update")

        self.a_start = new_a_start
        self.a_len = new_a_len
        self.a_loc = new_a_loc
        self.a_scaling = new_a_scaling

        # print(f"current adapters on batch (offloaded {len(remove_ind)})",
        #       len(self.adapter_dirs), self.adapter_dirs)
        # print(self.mem_manager.can_use_mem_size_suffix // 4 / 32)


import triton
import triton.language as tl


@triton.jit
def var_len_copy_kernel_triton(old_a_start, old_a_len, old_a_location, new_a_start, new_a_location,
                               BLOCK_SIZE: tl.constexpr):
    a_id = tl.program_id(0)
    length = tl.load(old_a_len + a_id)
    old_start = tl.load(old_a_start + a_id)
    new_start = tl.load(new_a_start + a_id)
    old_offset = tl.arange(0, BLOCK_SIZE)
    new_offset = tl.arange(0, BLOCK_SIZE)
    for i in range(0, length, BLOCK_SIZE):
        v = tl.load(old_a_location + old_start + i + old_offset, mask=old_offset < length)
        tl.store(new_a_location + new_start + i + new_offset, v, mask=new_offset < length)


def launch_var_len_copy_triton(old_a_start, old_a_len, old_location, new_a_start, new_a_location):
    BLOCK_SIZE = 256
    grid_size = (len(old_a_start),)

    var_len_copy_kernel_triton[grid_size](
        old_a_start, old_a_len, old_location, new_a_start, new_a_location, BLOCK_SIZE)


"""
from cupyx import jit
import cupy
import torch


@jit.rawkernel()
def var_len_copy_kernel(old_a_start, old_a_len, old_a_location, new_a_start, new_a_location,
                        BLOCK_SIZE):
    a_id = jit.blockIdx.x
    t_id = jit.threadIdx.x
    for i in range(t_id, old_a_len[a_id], BLOCK_SIZE):
        new_a_location[new_a_start[a_id] + i] = old_a_location[old_a_start[a_id] + i]


def launch_var_len_copy(old_a_start, old_a_len, old_location, new_a_start, new_a_location):
    BLOCK_SIZE = 128
    print(BLOCK_SIZE)
    grid_size = (len(old_a_start),)
    assert len(old_a_start) == len(new_a_start) == len(old_a_len)
    block_size = (BLOCK_SIZE,)

    var_len_copy_kernel(grid_size, block_size,
        (cupy.asarray(old_a_start),
         cupy.asarray(old_a_len),
         cupy.asarray(old_location),
         cupy.asarray(new_a_start),
         cupy.asarray(new_a_location),
         BLOCK_SIZE))
"""
