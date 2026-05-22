from dataclasses import dataclass
import threading
import numpy as np
from dserve.common.unified_mem_allocator import PageType, UnifiedMemoryAllocator
import torch
from typing import List, Dict, Any
import time

from dserve.common.mem_allocator import MemoryAllocator
from dserve.utils.infer_utils import calculate_time, mark_start, mark_end


@dataclass
class InferAdapterAlt:
    adapter_dirs: List[str]  # all adapters on the server
    idx_map: Dict[str, int]

    a_start_lora: torch.Tensor  # track the start index in a_loc_lora_a/b for each adapter
    a_len_lora: torch.Tensor  # tells the length of each adapter's LoRA A/B, use a_start_lora and a_len_lora to get the adapter in a_loc_lora_a/b

    a_loc_lora_a: torch.Tensor  # a_loc[i] is a list of indices occupied by adapter i lora A
    a_loc_lora_b: torch.Tensor  # a_loc[i] is a list of indices occupied by adapter i lora B

    a_loc_cat: torch.Tensor  # concatenated a_loc_lora_a and a_loc_lora_b for pinning

    a_scaling: torch.Tensor  # a_scaling[i] is the scaling factor of adapter i
    mem_manager: UnifiedMemoryAllocator
    finetuning_adapter_dir: str = None
    gpu_a_loc_lora_a = None
    gpu_a_loc_lora_b = None

    @classmethod
    def init(cls, mem_manager: UnifiedMemoryAllocator):
        return cls(
            adapter_dirs       = [],
            idx_map            = {},
            a_start_lora       = torch.empty(0, dtype=torch.long, device='cuda'),
            a_len_lora         = torch.empty(0, dtype=torch.long, device='cuda'),
            a_loc_lora_a       = torch.empty(0, dtype=torch.long, device='cuda'),
            a_loc_lora_b       = torch.empty(0, dtype=torch.long, device='cuda'),
            a_loc_cat          = torch.empty(0, dtype=torch.long, device='cuda'),
            a_scaling          = torch.empty(0, dtype=torch.float16, device='cuda'),
            mem_manager        = mem_manager,
            finetuning_adapter_dir = None,
        )

    def load_lora_A(self, adapter, vpids_a):
        rows_per_layer = adapter.r * 4
        assert len(vpids_a) == rows_per_layer

        self.mem_manager.pin_pages(vpids_a)  # Ensure pages are resident on GPU
        for layer_id, layer in enumerate(adapter.layers):
            if adapter.is_finetuning:
                rows = layer.w_combined_home_fp32[0].clamp_(-6.5e4, 6.5e4).to(dtype=layer.w_combined_home.dtype)
            else:
                layer.load_to_gpu()
                rows = layer.w_combined[0]                # [4*r, head_dim]
            self.mem_manager.copy_rows_to_layer(layer_id, vpids_a, rows)
            layer.offload_from_gpu()
        self.mem_manager.unpin_pages(vpids_a)  # Unpin after copying


    def load_lora_B(self, adapter, vpids_b):
        rows_per_layer = adapter.r * 4
        assert len(vpids_b) == rows_per_layer

        self.mem_manager.pin_pages(vpids_b)  # Ensure pages are resident on GPU
        for layer_id, layer in enumerate(adapter.layers):
            if adapter.is_finetuning:
                rows = layer.w_combined_home_fp32[1].clamp_(-6.5e4, 6.5e4).to(dtype=layer.w_combined_home.dtype)
            else:
                layer.load_to_gpu()
                rows = layer.w_combined[1]                # [4*r, head_dim]
            self.mem_manager.copy_rows_to_layer(layer_id, vpids_b, rows)
            layer.offload_from_gpu()
        self.mem_manager.unpin_pages(vpids_b)  # Unpin after copying

    # @calculate_time(show=True, min_cost_ms=0)
    def load_adapters(self, adapters):
        new_adapters = [a for a in adapters if a and a.lora_dir not in self.idx_map]
        if not new_adapters:
            return
        add_count   = len(new_adapters)
        start_base  = self.a_start_lora.numel()

        self.a_start_lora = torch.cat(
            (self.a_start_lora,
            torch.empty(add_count, dtype=torch.long, device='cuda'))
        )
        self.a_len_lora   = torch.cat(
            (self.a_len_lora,
            torch.empty(add_count, dtype=torch.long, device='cuda'))
        )
        self.a_scaling    = torch.cat(
            (self.a_scaling,
            torch.tensor([a.scaling for a in new_adapters],
                        dtype=torch.float16, device='cuda'))
        )

        # finetune_a_vpids = []
        # finetune_b_vpids = []

        for i, adapter in enumerate(new_adapters):
            rows_needed = adapter.r * 4                    # for A *or* B
            self.idx_map[adapter.lora_dir] = len(self.adapter_dirs)
            self.adapter_dirs.append(adapter.lora_dir)

            free_gpu = self.mem_manager._num_free_gpu_slots()
            gpu_rows = min(rows_needed * 2, free_gpu)      # A+B budget
            cpu_rows = rows_needed * 2 - gpu_rows

            # vpids_gpu = (self.mem_manager.alloc(gpu_rows, PageType.ADAPTER_WEIGHT)
            #             if gpu_rows else [])
            # vpids_cpu = (self.mem_manager.alloc_cpu(cpu_rows, PageType.ADAPTER_WEIGHT)
            #             if cpu_rows else [])
            # vpids_total = vpids_gpu + vpids_cpu

            vpids_gpu = (
                self.mem_manager.alloc(gpu_rows, PageType.ADAPTER_WEIGHT)
                if gpu_rows else torch.empty(0, dtype=torch.long, device='cuda')
            )
            vpids_cpu = (
                self.mem_manager.alloc_cpu(cpu_rows, PageType.ADAPTER_WEIGHT)
                if cpu_rows else torch.empty(0, dtype=torch.long, device='cuda')
            )
            vpids_total = torch.cat((vpids_gpu, vpids_cpu), dim=0)
            assert vpids_total.numel() == rows_needed * 2

            vpids_a, vpids_b = vpids_total[:rows_needed], vpids_total[rows_needed:]

            # update global vpid tables
            start_idx = self.a_loc_lora_a.numel()
            self.a_start_lora[start_base + i] = start_idx
            self.a_len_lora[start_base + i]   = rows_needed

            self.a_loc_lora_a = torch.cat(
                (self.a_loc_lora_a, vpids_a)
            )
            self.a_loc_lora_b = torch.cat(
                (self.a_loc_lora_b, vpids_b)
            )
            self.a_loc_cat = torch.cat((self.a_loc_lora_a, self.a_loc_lora_b), dim=0)

            self.load_lora_A(adapter, vpids_a)
            self.load_lora_B(adapter, vpids_b)
        self.gpu_a_loc_lora_a = None
        self.gpu_a_loc_lora_b = None


    def offload_target_adapters(self, remove_adapter_dirs):
        print(f"Offloading adapters: {remove_adapter_dirs}")
        reserve_adapter_dirs = set(self.adapter_dirs) - set(remove_adapter_dirs)
        self.offload_adapters(reserve_adapter_dirs)

    # @calculate_time(show=True, min_cost_ms=0)
    def offload_adapters(self, reserve_adapter_dirs):
        if not self.adapter_dirs:
            return
        if len(reserve_adapter_dirs) == len(self.adapter_dirs) and all(
                d in reserve_adapter_dirs for d in self.adapter_dirs):
            # nothing to remove
            return
        # for dir in self.adapter_dirs:
        #     if dir not in reserve_adapter_dirs:
        #         print(f"  Removing adapter: {dir}", end=' ')
        if len(reserve_adapter_dirs) == 0:
            # free everything
            all_vpids = torch.cat((self.a_loc_lora_a, self.a_loc_lora_b)).tolist()
            self.mem_manager.free(all_vpids)

            # reset all bookkeeping
            self.adapter_dirs = []
            self.idx_map      = {}
            self.a_start_lora = self.a_len_lora = torch.empty(0, dtype=torch.long,  device='cuda')
            self.a_loc_lora_a = self.a_loc_lora_b = torch.empty(0, dtype=torch.long, device='cuda')
            self.a_loc_cat = torch.empty(0, dtype=torch.long, device='cuda')
            self.a_scaling    = torch.empty(0, dtype=torch.float16, device='cuda')
            return
        
        left_indices   = []
        remove_indices = []
        for i, adir in enumerate(self.adapter_dirs):
            if adir in reserve_adapter_dirs:
                left_indices.append(i)
            else:
                remove_indices.append(i)

        if not remove_indices:
            return                                        

        vpids_to_free = []
        for idx in remove_indices:
            start = self.a_start_lora[idx].item()
            length = self.a_len_lora[idx].item()
            vpids_to_free.extend(
                self.a_loc_lora_a[start:start + length].tolist())
            vpids_to_free.extend(
                self.a_loc_lora_b[start:start + length].tolist())

        # Free through allocator
        self.mem_manager.free(vpids_to_free)

        new_adapter_dirs = []
        new_idx_map      = {}

        for new_pos, old_idx in enumerate(left_indices):
            adir = self.adapter_dirs[old_idx]
            new_adapter_dirs.append(adir)
            new_idx_map[adir] = new_pos

        # Copy a_len / a_scaling of kept adapters
        new_a_len     = self.a_len_lora[left_indices].clone()
        new_a_scaling = self.a_scaling[left_indices].clone()

        # Build new a_start (prefix sum) and new a_loc tables
        new_a_start   = torch.empty_like(new_a_len)
        new_a_start[0] = 0
        if new_a_start.numel() > 1:
            new_a_start[1:] = torch.cumsum(new_a_len, dim=0)[:-1]

        total_rows = int(new_a_len.sum().item())
        new_loc_a  = torch.empty(total_rows, dtype=torch.long, device='cuda')
        new_loc_b  = torch.empty(total_rows, dtype=torch.long, device='cuda')

        write_cursor = 0
        for new_pos, old_idx in enumerate(left_indices):
            rows = new_a_len[new_pos].item()
            old_start = self.a_start_lora[old_idx].item()
            new_loc_a[write_cursor:write_cursor+rows] = \
                self.a_loc_lora_a[old_start:old_start+rows]
            new_loc_b[write_cursor:write_cursor+rows] = \
                self.a_loc_lora_b[old_start:old_start+rows]
            write_cursor += rows
        
        '''
        # ------------------------------------------------------------------
        # Fast GPU copy with Triton (LoRA-A table)
        # ------------------------------------------------------------------
        launch_var_len_copy_triton(
            self.a_start_lora[left_indices],        # old_a_start
            new_a_len,                              # old_a_len (same as new)
            self.a_loc_lora_a,                      # old_a_location  (src)
            new_a_start,                            # new_a_start
            new_loc_a                               # new_a_location  (dst)
        )

        # ------------------------------------------------------------------
        # Fast GPU copy with Triton (LoRA-B table)
        # ------------------------------------------------------------------
        launch_var_len_copy_triton(
            self.a_start_lora[left_indices],        # old_a_start
            new_a_len,                              # old_a_len
            self.a_loc_lora_b,                      # old_a_location  (src)
            new_a_start,                            # new_a_start
            new_loc_b                               # new_a_location  (dst)
        )
        '''
        self.adapter_dirs  = new_adapter_dirs
        self.idx_map       = new_idx_map
        self.a_start_lora  = new_a_start
        self.a_len_lora    = new_a_len
        self.a_loc_lora_a  = new_loc_a
        self.a_loc_lora_b  = new_loc_b
        self.a_loc_cat     = torch.cat((self.a_loc_lora_a, self.a_loc_lora_b), dim=0)
        self.a_scaling     = new_a_scaling
        self.gpu_a_loc_lora_a = None
        self.gpu_a_loc_lora_b = None
    
    def pin_adapters_pages(self):
        self.mem_manager.pin_pages(self.a_loc_cat)

    def unpin_adapters_pages(self):
        self.mem_manager.unpin_pages(self.a_loc_cat)
        self.gpu_a_loc_lora_a = None
        self.gpu_a_loc_lora_b = None

    def get_lora_params_at_layer(self, layer_id):
        if self.gpu_a_loc_lora_a is None or self.gpu_a_loc_lora_b is None:
            self.gpu_a_loc_lora_a = self.mem_manager.to_gpu_index(self.a_loc_lora_a) 
            self.gpu_a_loc_lora_b = self.mem_manager.to_gpu_index(self.a_loc_lora_b)
        return self.mem_manager.get_adapter_pool(layer_id), self.a_start_lora, self.a_len_lora, self.gpu_a_loc_lora_a, self.gpu_a_loc_lora_b, self.a_scaling

   
import triton
import triton.language as tl


@triton.jit
def var_len_copy_kernel(
        old_start_ptr,      # [N]  int64
        old_len_ptr,        # [N]  int64
        old_loc_ptr,        # [total_rows]  int64  (or int32)
        new_start_ptr,      # [N]  int64
        new_loc_ptr,        # [total_rows]  int64  (or int32)
        BLOCK_SIZE: tl.constexpr):

    a_id   = tl.program_id(axis=0)                 # adapter index i
    length = tl.load(old_len_ptr   + a_id)         # rows = 4*r
    if length == 0:                                # (skip empty slots)
        return

    old_s  = tl.load(old_start_ptr + a_id)
    new_s  = tl.load(new_start_ptr + a_id)

    offs   = tl.arange(0, BLOCK_SIZE)              # [0, …, 255]
    for i in range(0, length, BLOCK_SIZE):
        mask = offs < (length - i)
        vals = tl.load(old_loc_ptr + old_s + i + offs, mask=mask, other=0)
        tl.store(new_loc_ptr + new_s + i + offs, vals, mask=mask)


# ---------------------------------------------------------------------------
def launch_var_len_copy_triton(old_start: torch.Tensor,
                               old_len:   torch.Tensor,
                               old_loc:   torch.Tensor,
                               new_start: torch.Tensor,
                               new_loc:   torch.Tensor):
    """
    A thin wrapper that dispatches the kernel once and waits for completion.
    Tensors must all reside on the same GPU device.
    """
    assert old_start.is_cuda and old_loc.is_cuda and new_loc.is_cuda
    BLOCK_SIZE = 256
    grid = (old_start.numel(),)                    # one program per adapter

    var_len_copy_kernel[grid](
        old_start, old_len, old_loc,
        new_start, new_loc,
        BLOCK_SIZE
    )