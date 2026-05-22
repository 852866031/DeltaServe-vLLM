import os
import threading
import time
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

import torch


def get_tensor_size_kb(numel: int, dtype: torch.dtype) -> float:
    """Helper to compute tensor element size in KB."""
    dtype_size_map = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float64: 8,
        torch.int8: 1,
        torch.int16: 2,
        torch.int32: 4,
        torch.int64: 8,
        torch.bool: 1,
    }
    bytes_per_element = dtype_size_map[dtype]
    return (numel * bytes_per_element) / 1024


class _OccupancyTracker:
    """Lightweight background sampler logging (used pages / total pages).

    Runs as a daemon thread; no explicit shutdown needed (dies with the
    process). Samples at `interval_s` (default 1.0s, env-overridable via
    DSERVE_OCCUPANCY_INTERVAL_S). Each sample reads the allocator's
    free_bitmap once, derives used = tot_size - free, and appends a row
    to the configured CSV.
    """

    def __init__(self, allocator, log_path: str, interval_s: float = 1.0,
                 label: str = ""):
        self.allocator = allocator
        self.log_path = log_path
        self.interval_s = float(interval_s)
        self.label = label or type(allocator).__name__
        self._stop = threading.Event()
        self._thread = None
        self._file = None
        self._t0 = None

    def start(self):
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.log_path, "w", buffering=1)
        self._file.write(
            "timestamp,t_rel_s,allocator,used_pages,total_pages,occupancy_pct\n"
        )
        self._t0 = time.monotonic()
        self._thread = threading.Thread(
            target=self._loop, daemon=True,
            name=f"OccupancyTracker-{self.label}",
        )
        self._thread.start()
        print(f"[occupancy:{self.label}] tracking → {self.log_path} "
              f"(interval={self.interval_s}s)", flush=True)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval_s + 0.5, 1.0))
        if self._file is not None:
            self._file.close()
            self._file = None

    def _loop(self):
        # Event.wait returns True when set(), False on timeout — so loop
        # while it timed out (i.e. we weren't asked to stop yet).
        while not self._stop.wait(self.interval_s):
            try:
                self._sample()
            except Exception as e:
                print(f"[occupancy:{self.label}] sample error: {e}", flush=True)

    def _sample(self):
        # CPU-only read. Doing GPU work here can interfere with CUDA-graph
        # capture/replay running on the inference thread (graph pool
        # aliasing → silent NaN logits down the line). _used_pages is
        # maintained incrementally under page_table_lock by every
        # allocator path that mutates free_bitmap.
        with self.allocator.page_table_lock:
            used = int(self.allocator._used_pages)
            total = int(self.allocator.tot_size)
        pct = 100.0 * used / total if total > 0 else 0.0
        ts = datetime.now().isoformat(timespec="milliseconds")
        t_rel = time.monotonic() - self._t0
        self._file.write(
            f"{ts},{t_rel:.3f},{self.label},{used},{total},{pct:.3f}\n"
        )


class PageType(Enum):
    FREE = 0
    KV_CACHE = auto()
    ADAPTER_WEIGHT = auto()
    ATTENTION_INPUT_ACTIVATION = auto()
    FFN_INPUT_ACTIVATION = auto()
    EMBEDDING = auto()


class UnifiedMemoryAllocator:
    """
    Simplified allocator assuming everything fits in GPU memory.
    Uses global free bitmap and page_type_map shared by all layers.
    All layers allocate/free the same physical pages together.
    """

    def __init__(self, head_num, head_dim, vocab_size, layer_num: int,
                 max_pool_size: int, dtype=torch.float16, device="cuda", log_path=None,
                 max_finetuning_tokens: int = 1024):
        self.head_dim = head_dim
        self.head_num = head_num
        self.hidden_dim = head_num * head_dim
        self.layer_num = layer_num
        self.device = device
        self.dtype = dtype
        self.vocab_size = vocab_size

        # total number of slots per layer
        self.tot_size = int(
            max_pool_size * 1024 * 1024
            / self.layer_num
            / get_tensor_size_kb(self.head_num * self.head_dim, self.dtype)
        )

        # contiguous tensor pool for each layer.
        # NOTE: this attribute is allocator-internal storage.
        self.gpu_pools = [
            torch.empty((self.tot_size, self.head_num, self.head_dim),
                        device=self.device, dtype=self.dtype)
            for _ in range(self.layer_num)
        ]
        print(f"UnifiedMemoryAllocator initialized with {self.tot_size} of {self.gpu_pools[0].shape[1:]} pages per layer, Page size={get_tensor_size_kb(self.head_num * self.head_dim, self.dtype):.2f} KB, dtype={self.dtype}")
        # bitmaps shared by all layers
        self.page_type_map = torch.zeros(self.tot_size, dtype=torch.long, device=self.device)
        self.free_bitmap = torch.ones(self.tot_size, dtype=torch.bool, device=self.device)  # True=free

        self.page_table_lock = threading.RLock()

        # bookkeeping
        self.request_token_info = []
        self.activation_page_indices = []  # list of tuples: (FFN_input_phys_ids, ATTENTION_input_phys_ids)
        self.finetune_input_ids = []
        self.finetune_logits_per_request = []
        self.shared_transformer_out_activations = None
        self.shared_attention_out_activations = None
        self.embedding_output = None
        self.max_finetuning_tokens = max_finetuning_tokens
        self.init_shared_activation_memory()

        # CPU-side mirror of (tot_size - free_bitmap.sum()). Maintained
        # incrementally under page_table_lock by every path that mutates
        # free_bitmap. The tracker reads this instead of running a GPU
        # reduction; that matters because GPU work in a background thread
        # can interfere with CUDA-graph capture/replay on the inference
        # thread (graph pool aliasing) and produce silent corruption.
        self._used_pages: int = 0

        # Optional page-occupancy tracker. Runs as a daemon thread so no
        # explicit shutdown coordination is needed. Activate by setting
        # cfg.memory.unified_mem_manager_log_path (or passing log_path
        # directly). Subclasses inherit this for free — the tracker only
        # reads self._used_pages and tot_size.
        self.log_path = log_path
        self._occupancy_tracker = None
        if log_path:
            interval = float(os.environ.get("DSERVE_OCCUPANCY_INTERVAL_S", "1.0"))
            self._occupancy_tracker = _OccupancyTracker(
                self, log_path, interval_s=interval,
                label=type(self).__name__,
            )
            self._occupancy_tracker.start()


    def _num_free_gpu_slots(self) -> int:
        """
        Return the number of currently free GPU slots (global across all layers).
        """
        with self.page_table_lock:
            return int(self.free_bitmap.sum().item())

    def get_kv_pool(self, layer_id: int) -> torch.Tensor:
        return self.gpu_pools[layer_id]

    def get_adapter_pool(self, layer_id: int) -> torch.Tensor:
        return self.gpu_pools[layer_id]

    def get_activation_pool(self, layer_id: int) -> torch.Tensor:
        return self.gpu_pools[layer_id]

    def alloc(self, num_pages: int, page_type: PageType) -> torch.Tensor:
        """
        Allocate `num_pages` GPU slots globally across all layers.
        Returns tensor of physical indices shared by all layers.
        """
        with self.page_table_lock:
            free_idx = torch.nonzero(self.free_bitmap, as_tuple=False).flatten()
            if free_idx.numel() < num_pages:
                raise RuntimeError(
                    f"Not enough free pages: need {num_pages}, have {free_idx.numel()}."
                )

            alloc_ids = free_idx[:num_pages]
            self.free_bitmap[alloc_ids] = False
            self.page_type_map[alloc_ids] = int(page_type.value)
            self._used_pages += int(num_pages)
            return alloc_ids

    def free(self, phys_ids: torch.Tensor):
        """
        Free pages globally for all layers.
        """
        with self.page_table_lock:
            if not isinstance(phys_ids, torch.Tensor):
                phys_ids = torch.as_tensor(phys_ids, dtype=torch.long, device=self.device)

            self.page_type_map[phys_ids] = int(PageType.FREE.value)
            self.free_bitmap[phys_ids] = True
            self._used_pages -= int(phys_ids.numel())

    def free_kv(self, kv_ids: torch.Tensor):
        """Free KV slots. In the base allocator a KV slot IS a page, so
        this delegates to free(). PackedKVMemoryAllocator overrides with
        sub-slot semantics. Callers that free KV (b_loc_key/value entries)
        MUST go through free_kv — never free() — so the right path is
        taken regardless of which allocator is in use."""
        return self.free(kv_ids)

    def alloc_contiguous_kv(self, need_size: int, page_type: PageType):
        with self.page_table_lock:
            free_mask = self.free_bitmap
            free_idx = torch.nonzero(free_mask, as_tuple=False).flatten()
            if free_idx.numel() < 2 * need_size:
                return None  # not enough total free slots
            # find contiguous free runs
            diffs = free_idx[1:] - free_idx[:-1]
            # gaps = 1 means contiguous, 0 means break
            # create run IDs for consecutive groups
            run_starts = torch.cat((
                torch.tensor([0], device=self.device),
                torch.nonzero(diffs != 1, as_tuple=False).flatten() + 1
            ))
            run_ends = torch.cat((run_starts[1:], torch.tensor([free_idx.numel()], device=self.device)))
            # scan for a contiguous run of at least 2 * need_size
            start_idx = None
            for s, e in zip(run_starts.tolist(), run_ends.tolist()):
                run_len = e - s
                if run_len >= 2 * need_size:
                    start_idx = free_idx[s].item()
                    break

            if start_idx is None:
                return None  # no large enough contiguous segment found

            end_idx = start_idx + 2 * need_size
            phys_all = torch.arange(start_idx, end_idx, dtype=torch.long, device=self.device)

            # mark them as used
            self.free_bitmap[phys_all] = False
            self.page_type_map[phys_all] = int(page_type.value)
            self._used_pages += int(2 * need_size)

            # split into K/V halves
            phys_k = phys_all[:need_size]
            phys_v = phys_all[need_size:]
            return phys_k, start_idx, start_idx + need_size, phys_v, start_idx + need_size, end_idx


    def reset_activation_pool(self):
        """
        Clears activation-related states and frees activation pages globally.
        """
        with self.page_table_lock:
            self.request_token_info.clear()
            self.finetune_input_ids.clear()
            self.finetune_logits_per_request.clear()
            self.activation_page_indices.clear()

            mask = (self.page_type_map == int(PageType.ATTENTION_INPUT_ACTIVATION.value)) | (
                self.page_type_map == int(PageType.FFN_INPUT_ACTIVATION.value)
            )
            idx = torch.nonzero(mask, as_tuple=False).flatten()
            if idx.numel() > 0:
                for pool in self.gpu_pools:
                    pool[idx].zero_()
                self.page_type_map[idx] = int(PageType.FREE.value)
                self.free_bitmap[idx] = True
                self._used_pages -= int(idx.numel())


    def page_size_kb(self) -> float:
        t = self.gpu_pools[0][0]
        s_kb = t.numel() * t.element_size() / 1024.0
        return s_kb * self.layer_num
    
    def init_shared_activation_memory(self):
        self.shared_transformer_out_activations = [
            torch.zeros((self.max_finetuning_tokens, self.head_num * self.head_dim),
                         dtype=self.dtype, device=self.device)
            for _ in range(self.layer_num)
        ]
        self.shared_attention_out_activations = [
            torch.zeros((self.max_finetuning_tokens, self.head_num * self.head_dim),
                         dtype=self.dtype, device=self.device)
            for _ in range(self.layer_num)
        ]
        self.embedding_output = torch.zeros((self.max_finetuning_tokens, self.head_num * self.head_dim),
                                        dtype=self.dtype, device=self.device)
        
        self.concat_input_ids = torch.zeros(self.max_finetuning_tokens*2, dtype=torch.int64, device=self.device) 
        self.logit_tensor = torch.zeros((self.max_finetuning_tokens, self.vocab_size), dtype=self.dtype, device=self.device)
    
    def share_activation_dict(self):
        return {
            "logit_tensor": self.logit_tensor,
            "concat_input_ids": self.concat_input_ids,
            "transformer_out_activations": self.shared_transformer_out_activations,
            "attention_out_activations": self.shared_attention_out_activations,
            "input_layer_output": self.embedding_output
        }

    def get_concatenated_finetune_input_ids(self):
        if not self.finetune_input_ids:
            return self.concat_input_ids[:0]
        cat_ids = torch.cat(self.finetune_input_ids, dim=0).to(self.concat_input_ids.device)
        n = cat_ids.numel()
        if n > self.concat_input_ids.numel():
            raise ValueError(f"concat_input_ids capacity {self.concat_input_ids.numel()} < needed {n}")
        self.concat_input_ids[:n].copy_(cat_ids)
        return self.concat_input_ids[:n]
    
    def copy_rows_to_layer(self, layer_id: int, phys_ids, rows: torch.Tensor):
        """
        Copy `rows` into self.gpu_pools[layer_id][vpids], paging-in any CPU pages first.
        """
        # print the device of gpu pool and rows:
        assert rows.dim() == 2 or rows.dim() == 3, \
            f"Expected 2D or 3D tensor for rows, got shape {rows.shape}"
        assert layer_id < self.layer_num, f"Invalid layer_id {layer_id}"
        if rows.device != self.device or rows.dtype != self.gpu_pools[layer_id].dtype:
            rows = rows.to(device=self.device, dtype=self.gpu_pools[layer_id].dtype,non_blocking=True)
        # ensure shape consistency
        rows_reshaped = rows.view(-1, self.head_num, self.head_dim)
        self.gpu_pools[layer_id][phys_ids] = rows_reshaped


    def save_activations_by_layer(self, layer_id, input_embs, infer_state, page_type, phys_ids=None):
        finetune_mask = infer_state.finetune_mask  # shape: [total_token_num]
        #finetune_activations = input_embs[finetune_mask].clone()
        finetune_activations = input_embs[finetune_mask]
        num_new_tokens = finetune_activations.shape[0]
        if phys_ids is None:
            phys_ids = self.alloc(num_new_tokens, page_type)
        else:
            if len(phys_ids) != num_new_tokens:
                raise ValueError(f"Expected {num_new_tokens} phys_ids, got {len(phys_ids)}")
        self.gpu_pools[layer_id][phys_ids] = finetune_activations.view(-1, self.head_num, self.head_dim)
        return phys_ids
        
    def save_embedding_output(self, input_embs, infer_state):
        finetune_mask = infer_state.finetune_mask  # shape: [total_token_num]
        finetune_activations = input_embs[finetune_mask]
        prev_total = sum(self.request_token_info)
        num_new_tokens = finetune_activations.shape[0]
        self.embedding_output[prev_total : prev_total + num_new_tokens] = finetune_activations
    
    def write_to_logit_tensor(self, logits, FFN_input_pids, attention_input_pids):
        #self.finetune_logits_per_request.extend(logits)
        accumlate_len = sum(self.request_token_info)
        for logit in logits:
            n = logit.size(0)
            accumlate_len += n
            self.request_token_info.append(n)
        self.activation_page_indices.append((FFN_input_pids, attention_input_pids))
        flat_logits = torch.cat(logits, dim=0)
        total_tokens = flat_logits.size(0)
        end_pos = min(accumlate_len, self.logit_tensor.size(0))
        self.logit_tensor[accumlate_len - total_tokens:end_pos].copy_(flat_logits, non_blocking=True)
    

    def export_requests_info(self):
        self.get_concatenated_finetune_input_ids()
        self.saved_layer_0_activations = None
        for layer_id in range(self.layer_num):
            self.fill_activations_by_layer(layer_id, PageType.FFN_INPUT_ACTIVATION, 
                                         self.shared_attention_out_activations[layer_id])
            self.fill_activations_by_layer(layer_id, PageType.ATTENTION_INPUT_ACTIVATION, 
                                         self.shared_transformer_out_activations[layer_id])
        requests_info_dict = {
            "request_token_info": self.request_token_info,
            #"finetuning_logits_per_request": self.finetune_logits_per_request,
        }
        return requests_info_dict

    def fill_activations_by_layer(self, layer_id, page_type, dest):
        """
        Gather activations of the given PageType for a specific layer, in the same
        order as requests were recorded in self.activation_page_indices.

        Args:
            layer_id (int): Which layer's pool to read from.
            page_type (PageType): Which activation type to export (FFN_INPUT_ACTIVATION or ATTENTION_INPUT_ACTIVATION).
            dest (torch.Tensor): Destination tensor to fill with shape [max_finetuning_tokens, hidden_dim].

        Returns:
            dest (torch.Tensor): Filled up to total_tokens rows.
        """
        if not self.activation_page_indices or len(self.request_token_info) == 0:
            return None

        total_tokens = sum(self.request_token_info)
        layer_pool = self.gpu_pools[layer_id]
        collected = []
        # Map PageType → index position in the tuple
        # (FFN_INPUT_ACTIVATION, ATTENTION_INPUT_ACTIVATION)
        idx_pos = 0 if page_type == PageType.FFN_INPUT_ACTIVATION else 1

        # Collect activation tensors per request
        for ffn_phys_ids, attn_phys_ids in self.activation_page_indices:
            phys_ids = ffn_phys_ids if idx_pos == 0 else attn_phys_ids
            if phys_ids is not None and len(phys_ids) > 0:
                collected.append(layer_pool.index_select(0, phys_ids))

        if not collected:
            raise ValueError(f"No activations found for {page_type.name} in layer {layer_id}.")

        # Concatenate in request order and flatten to [total_tokens, hidden_dim]
        flat = torch.cat(collected, dim=0).reshape(total_tokens, -1)

        # Copy into destination tensor
        if dest.device != flat.device or dest.dtype != flat.dtype:
            dest = dest.to(device=flat.device, dtype=flat.dtype)
        dest[:total_tokens].copy_(flat, non_blocking=True)

        return dest

    def prepare_b_locs_for_layer(
        self,
        b_loc_key:   torch.Tensor,
        b_loc_value: torch.Tensor,
        b_seq_len:   torch.Tensor,
        layer_id:    int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with self.page_table_lock:
            return self.get_kv_pool(layer_id), b_loc_key, b_loc_value
    
    def to_gpu_index(self, vpids) -> torch.Tensor:
        with self.page_table_lock:
            return vpids
    
    def pin_pages(self, vpids):
        return
    
    def unpin_pages(self, vpids):
        return
    
    def alloc_cpu(self, num_pages: int, page_type: PageType) -> torch.Tensor:
        raise NotImplementedError("UnifiedMemoryAllocator does not support CPU allocation.")
    
    def reset_b_loc_kv(self, b_loc_key, b_loc_value):
        return