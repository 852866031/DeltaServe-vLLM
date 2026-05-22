from dserve.common.unified_mem_allocator import UnifiedMemoryAllocator, PageType
import torch
import numpy as np
import collections

from dserve.common.configs.config import get_active_config
from dataclasses import dataclass
from typing import List, Dict
from dserve.common.mem_manager import MemoryManager
from dserve.utils.infer_utils import mark_start, mark_end
from dserve.utils.infer_utils import calculate_time


# ─── Merge-trace debug toggle ────────────────────────────────────────
# When True, every InferBatch.merge() call logs a one-line `[merge]`
# marker showing both source dims and the allocator's KV-page count at
# that moment. Pair with PackedKVMemoryAllocator._DEBUG_FREE to see
# free events and merge events on the same timeline — KV climbing in
# step with [merge] markers (without a matching [mem_free] event)
# confirms merge is dropping live slot ids.
#
# We also emit a rough "expected KV" estimate based on in-flight
# b_seq_len, so the "residual" (= actual KV pages held by the
# allocator − pages strictly needed to back the in-flight tokens) is
# directly visible. Zero or near-zero residual = no leak; growing
# residual = orphaned slots.
#
# Off-path cost is one LOAD_GLOBAL + branch per merge — merges happen
# at most a few times per second, so even when on the print cost is
# negligible vs. the GPU work merge itself does.
_DEBUG_MERGE = False


class InferSamplingParams:

    def __init__(
        self,
        do_sample: bool = False,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        vocab_size: int = -1,
    ) -> None:
        self.do_sample = do_sample
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        if self.top_k == -1:
            self.top_k = vocab_size
        return


from ..mixed_req_queue import rprint


@dataclass
class InferBatch:
    batch_id: int
    requests: List
    requests_idx_mapping: Dict[int, int]

    input_ids: torch.Tensor

    all_input_ids: List[List[int]]
    input_lengths: List[int]
    
    out_token_id_counts: List
    sampling_param_list : List[InferSamplingParams]

    input_ids: torch.Tensor

    nopad_total_token_num: int
    nopad_max_len_in_batch: int
    nopad_b_loc: torch.Tensor
    nopad_b_start_loc: torch.Tensor
    nopad_b_seq_len: torch.Tensor

    # nopad_b_start_loc[0] nopad_b_seq_len[0]
    # ith request in the batch: nopad_b_loc[nopad_b_start_loc[i]: nopad_b_start_loc[i] + nopad_b_seq_len[i]]
    # DO NOT CROSS REQUEST ATTENTION SCORE
    mem_manager: UnifiedMemoryAllocator

    nopad_b_loc_key: torch.Tensor
    nopad_b_loc_value: torch.Tensor

    adapter_dirs: List[str]

    finetune_mask: torch.Tensor  # Added

    @classmethod
    @torch.no_grad()
    def init_batch(cls, batch_id, requests, dtype: torch.dtype, device: torch.device, mem_manager: MemoryManager, vocab_size: int):
        input_lengths = []
        all_input_ids = []
        requests_idx_mapping = {}

        out_token_id_counts = []
        sampling_param_list = []
        finetune_flags = []

        nopad_total_token_num = 0
        nopad_max_len_in_batch = 0
        max_req_total_len = get_active_config().serving.max_req_total_len
        nopad_b_loc = torch.zeros((len(requests), max_req_total_len + 12), dtype=torch.long, device='cuda')
        nopad_b_loc_key = torch.zeros((len(requests), max_req_total_len + 12), dtype=torch.long, device='cuda')
        nopad_b_loc_value = torch.zeros((len(requests), max_req_total_len + 12), dtype=torch.long, device='cuda')

        nopad_b_start_loc = torch.zeros(len(requests), dtype=torch.int32, device='cuda')

        adapter_dirs = []

        for i, r in enumerate(requests):
            requests_idx_mapping[r['request_id']] = i

            tokenized_input = r['input_id']
            is_finetuning = r.get("is_finetuning", False)

            if is_finetuning:
                full_input_tensor = torch.tensor(tokenized_input, dtype=torch.int64, device=device)
                mem_manager.finetune_input_ids.append(full_input_tensor)

            if is_finetuning and len(tokenized_input) > 1:
                tokenized_input = tokenized_input[:-1]

            input_length = len(tokenized_input)
            input_lengths.append(input_length)
            all_input_ids.append(tokenized_input)
            out_token_id_counts.append(collections.defaultdict(int))

            sampling_param = r["sampling_param"]
            sampling_param["vocab_size"] = vocab_size
            sampling_param_list.append(InferSamplingParams(**sampling_param))

            nopad_total_token_num += input_length
            nopad_max_len_in_batch = max(nopad_max_len_in_batch, input_length)

            adapter_dirs.append(r["adapter_dir"])
            finetune_flags.append(1 if is_finetuning else 0)

        # Create masks and tensor metadata
        finetune_mask = torch.tensor(finetune_flags, dtype=torch.uint8, device=device)
        nopad_b_seq_len = torch.tensor(input_lengths, dtype=torch.int32, device=device)

        if len(requests) > 1:
            nopad_b_start_loc[1:] = torch.cumsum(nopad_b_seq_len, dim=0, dtype=torch.int32)[:-1]

        # Flatten all input ids into a single tensor
        if len(requests) > 1:
            input_ids = np.concatenate(all_input_ids, dtype=np.int64)
        else:
            input_ids = all_input_ids[0]

        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=device)
        rprint("input_ids shape", input_ids.shape)
        return cls(
            batch_id=batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            input_lengths=input_lengths,
            all_input_ids=all_input_ids,
            nopad_total_token_num=nopad_total_token_num,
            nopad_max_len_in_batch=nopad_max_len_in_batch,
            nopad_b_loc=nopad_b_loc,
            nopad_b_loc_key=nopad_b_loc_key,
            nopad_b_loc_value=nopad_b_loc_value,
            nopad_b_start_loc=nopad_b_start_loc,
            nopad_b_seq_len=nopad_b_seq_len,
            out_token_id_counts=out_token_id_counts,
            sampling_param_list=sampling_param_list,
            mem_manager=mem_manager,
            adapter_dirs=adapter_dirs,
            finetune_mask=finetune_mask,
        )
    
    def count_tokens(self):
        input_tokens = self.nopad_total_token_num
        kv_tokens = int(torch.sum(self.nopad_b_seq_len))
        generated_tokens = kv_tokens - input_tokens
        return input_tokens, generated_tokens, kv_tokens
    
    @torch.no_grad()
    def free_self(self):
        # b_loc_*[idx, max_len - seq_len : max_len] holds all seq_len KV
        # slot ids for request idx. Previous code ended the slice at
        # max_len - 1, leaking the slot at column max_len - 1 (the most
        # recent prefill/decode allocation) — one K + one V sub-slot per
        # finished request, never returned to the pool.
        remove_index = []
        max_len = self.nopad_max_len_in_batch
        for idx in range(len(self)):
            seq_len = self.nopad_b_seq_len[idx]
            remove_index.append(self.nopad_b_loc_key[idx, max_len - seq_len : max_len])
            remove_index.append(self.nopad_b_loc_value[idx, max_len - seq_len : max_len])
        remove_index = torch.cat(remove_index, dim=-1)
        self.mem_manager.free_kv(remove_index)
        return
        
    # @calculate_time(show=True, min_cost_ms=0)
    @torch.no_grad()
    def filter(self, request_ids: List[int]):
        if len(request_ids) == 0:
            raise ValueError("Batch must have at least one request")
        if len(request_ids) == len(self):
            return self
        requests_idx_mapping = {}
        indices = []
        requests = []
        all_input_ids = []
        input_lengths = []

        nopad_total_token_num = 0
        nopad_max_len_in_batch = 0
        max_req_total_len = get_active_config().serving.max_req_total_len
        nopad_b_loc = torch.zeros((len(request_ids), max_req_total_len + 12),
                                  dtype=torch.long, device='cuda')
        nopad_b_loc_key = torch.zeros((len(request_ids), max_req_total_len + 12),
                                  dtype=torch.long, device='cuda')
        nopad_b_loc_value = torch.zeros((len(request_ids), max_req_total_len + 12),
                                  dtype=torch.long, device='cuda')
        nopad_b_start_loc = torch.zeros(len(request_ids), dtype=torch.int32, device='cuda')
        nopad_b_seq_len = torch.zeros(len(request_ids), dtype=torch.int32, device='cuda')

        left_idx = []
        for i, request_id in enumerate(request_ids):
            idx = self.requests_idx_mapping[request_id]
            left_idx.append(idx)
        
        left_idx_set = set(left_idx)
        # Slice covers all seq_len populated columns (was ending at
        # max_len - 1, dropping the most recently allocated slot — see
        # free_self comment for the leak details).
        remove_index_kv = []
        max_len = self.nopad_max_len_in_batch
        for idx in range(len(self)):
            if idx not in left_idx_set:
                seq_len = self.nopad_b_seq_len[idx]
                remove_index_kv.append(self.nopad_b_loc_key[idx, max_len - seq_len : max_len])
                remove_index_kv.append(self.nopad_b_loc_value[idx, max_len - seq_len : max_len])
        remove_index_kv = torch.cat(remove_index_kv, dim=-1)
        self.mem_manager.free_kv(remove_index_kv)


        nopad_max_len_in_batch = 0
        for i, request_id in enumerate(request_ids):
            idx = self.requests_idx_mapping[request_id]
            indices.append(idx)
        
        nopad_b_seq_len[:] = self.nopad_b_seq_len[indices]
        nopad_max_len_in_batch = torch.max(nopad_b_seq_len).item()
        nopad_b_start_loc[1:] = torch.cumsum(nopad_b_seq_len, dim=0, dtype=torch.int32)[0:-1]
        nopad_total_token_num = torch.sum(nopad_b_seq_len).item()
        
        nopad_b_loc[:, 0 : (nopad_max_len_in_batch - 1)] = self.nopad_b_loc[indices, (self.nopad_max_len_in_batch - 1) - (nopad_max_len_in_batch - 1): (self.nopad_max_len_in_batch - 1)]
        nopad_b_loc_key[:, 0 : (nopad_max_len_in_batch - 1)] = self.nopad_b_loc_key[indices, (self.nopad_max_len_in_batch - 1) - (nopad_max_len_in_batch - 1): (self.nopad_max_len_in_batch - 1)]
        nopad_b_loc_value[:, 0 : (nopad_max_len_in_batch - 1)] = self.nopad_b_loc_value[indices, (self.nopad_max_len_in_batch - 1) - (nopad_max_len_in_batch - 1): (self.nopad_max_len_in_batch - 1)]
        adapter_dirs = []
        for i, request_id in enumerate(request_ids):
            idx = self.requests_idx_mapping[request_id]
            requests_idx_mapping[request_id] = i
            requests.append(self.requests[idx])
            all_input_ids.append(self.all_input_ids[idx])
            input_lengths.append(self.input_lengths[idx])
            adapter_dirs.append(self.requests[idx]["adapter_dir"])
        
        input_ids = self.input_ids[indices]

        return InferBatch(
            batch_id=self.batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            input_lengths=input_lengths,
            all_input_ids=all_input_ids,
            nopad_total_token_num=nopad_total_token_num,
            nopad_max_len_in_batch=nopad_max_len_in_batch,
            nopad_b_loc=nopad_b_loc,
            nopad_b_loc_key=nopad_b_loc_key,
            nopad_b_loc_value=nopad_b_loc_value,
            nopad_b_start_loc=nopad_b_start_loc,
            nopad_b_seq_len=nopad_b_seq_len,
            out_token_id_counts=[self.out_token_id_counts[_i] for _i in indices],
            sampling_param_list=[self.sampling_param_list[_i] for _i in indices],
            mem_manager=self.mem_manager,
            adapter_dirs=adapter_dirs,
            finetune_mask=None,
        )


    @torch.no_grad()
    def clip(self, x: int):
        if x <= 0 or x > len(self):
            raise ValueError(f"x must be between 1 and {len(self)}")
        if x == len(self):
            return self

        # --- Step 1: indices to keep/drop ---
        keep_indices = list(range(x))
        drop_indices = list(range(x, len(self)))
        keep_tensor = torch.tensor(keep_indices, dtype=torch.long, device="cuda")

        # --- Step 2: free dropped requests' memory ---
        # Slice covers all seq_len populated columns (was ending at
        # max_len - 1, dropping the most recently allocated slot — see
        # free_self comment for the leak details).
        if drop_indices:
            remove_index_kv = []
            max_len = self.nopad_max_len_in_batch
            for idx in drop_indices:
                seq_len = self.nopad_b_seq_len[idx]
                remove_index_kv.append(
                    self.nopad_b_loc_key[idx, max_len - seq_len : max_len]
                )
                remove_index_kv.append(
                    self.nopad_b_loc_value[idx, max_len - seq_len : max_len]
                )
            remove_index_kv = torch.cat(remove_index_kv, dim=-1)
            self.mem_manager.free_kv(remove_index_kv)

        # --- Step 3: recompute metadata for survivors ---
        new_seq_len = self.nopad_b_seq_len[keep_tensor]
        nopad_max_len_in_batch = torch.max(new_seq_len).item()
        nopad_total_token_num = torch.sum(new_seq_len).item()

        nopad_b_start_loc = torch.zeros(x, dtype=torch.int32, device="cuda")
        if x > 1:
            nopad_b_start_loc[1:] = torch.cumsum(new_seq_len[:-1], dim=0, dtype=torch.int32)

        nopad_b_loc = torch.zeros((x, self.nopad_b_loc.size(1)), dtype=torch.long, device="cuda")
        nopad_b_loc_key = torch.zeros_like(nopad_b_loc)
        nopad_b_loc_value = torch.zeros_like(nopad_b_loc)

        span = nopad_max_len_in_batch - 1
        nopad_b_loc[:, :span] = self.nopad_b_loc[keep_tensor,
                            (self.nopad_max_len_in_batch - 1) - span:(self.nopad_max_len_in_batch - 1)]
        nopad_b_loc_key[:, :span] = self.nopad_b_loc_key[keep_tensor,
                            (self.nopad_max_len_in_batch - 1) - span:(self.nopad_max_len_in_batch - 1)]
        nopad_b_loc_value[:, :span] = self.nopad_b_loc_value[keep_tensor,
                            (self.nopad_max_len_in_batch - 1) - span:(self.nopad_max_len_in_batch - 1)]

        # --- Step 4: rebuild InferBatch with first x requests ---
        requests = self.requests[:x]
        all_input_ids = self.all_input_ids[:x]
        input_lengths = self.input_lengths[:x]
        adapter_dirs = self.adapter_dirs[:x]

        requests_idx_mapping = {r["request_id"]: i for i, r in enumerate(requests)}

        return InferBatch(
            batch_id=self.batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=self.input_ids[keep_tensor],
            input_lengths=input_lengths,
            all_input_ids=all_input_ids,
            nopad_total_token_num=nopad_total_token_num,
            nopad_max_len_in_batch=nopad_max_len_in_batch,
            nopad_b_loc=nopad_b_loc,
            nopad_b_loc_key=nopad_b_loc_key,
            nopad_b_loc_value=nopad_b_loc_value,
            nopad_b_start_loc=nopad_b_start_loc,
            nopad_b_seq_len=new_seq_len,
            out_token_id_counts=self.out_token_id_counts[:x],
            sampling_param_list=self.sampling_param_list[:x],
            mem_manager=self.mem_manager,
            adapter_dirs=adapter_dirs,
            finetune_mask=None,
        )

    @classmethod
    @torch.no_grad()
    def merge(cls, batch1, batch2):
        if _DEBUG_MERGE:
            cls._debug_print_merge_entry(batch1, batch2)
        requests = batch1.requests + batch2.requests
        requests_idx_mapping = {}
        new_batch_size = len(batch1) + len(batch2)

        input_ids = batch1.input_ids.new_empty(new_batch_size)
        all_input_ids = []
        input_lengths = []
        out_token_id_counts=[]
        sampling_param_list=[]

        cumulative_batch_size = 0
        nopad_total_token_num = batch1.nopad_total_token_num + batch2.nopad_total_token_num
        nopad_max_len_in_batch = max(batch1.nopad_max_len_in_batch, batch2 .nopad_max_len_in_batch)
        
        max_req_total_len = get_active_config().serving.max_req_total_len
        nopad_b_loc = torch.zeros((new_batch_size, max_req_total_len + 12), dtype=torch.long, device='cuda')
        nopad_b_loc_key = torch.zeros((new_batch_size, max_req_total_len + 12), dtype=torch.long, device='cuda')
        nopad_b_loc_value = torch.zeros((new_batch_size, max_req_total_len + 12), dtype=torch.long, device='cuda')

        nopad_b_start_loc = torch.zeros(new_batch_size, dtype=torch.int32, device='cuda')
        nopad_b_seq_len = torch.zeros(new_batch_size, dtype=torch.int32, device='cuda')
        nopad_start_loc_len_temp = 0
        adapter_dirs = []
        batches = [batch1, batch2]
        for i, batch in enumerate(batches):
            if i == 0:
                requests_idx_mapping = batch.requests_idx_mapping
            else:
                for k, v in batch.requests_idx_mapping.items():
                    requests_idx_mapping[k] = v + cumulative_batch_size
            start_index = cumulative_batch_size
            end_index = cumulative_batch_size + len(batch)
            input_ids[start_index:end_index] = batch.input_ids
            nopad_b_seq_len[start_index: end_index] = batch.nopad_b_seq_len
            nopad_b_start_loc[start_index: end_index] = batch.nopad_b_start_loc + nopad_start_loc_len_temp
            nopad_start_loc_len_temp = nopad_b_start_loc[end_index - 1] + nopad_b_seq_len[end_index - 1]
            nopad_b_loc[start_index: end_index, nopad_max_len_in_batch - batch.nopad_max_len_in_batch: nopad_max_len_in_batch -
                        1] = batch.nopad_b_loc[:, :batch.nopad_max_len_in_batch - 1]
            nopad_b_loc_key[start_index: end_index, nopad_max_len_in_batch - batch.nopad_max_len_in_batch: nopad_max_len_in_batch -
                        1] = batch.nopad_b_loc_key[:, :batch.nopad_max_len_in_batch - 1]
            nopad_b_loc_value[start_index: end_index, nopad_max_len_in_batch - batch.nopad_max_len_in_batch: nopad_max_len_in_batch -
                        1] = batch.nopad_b_loc_value[:, :batch.nopad_max_len_in_batch - 1]

            adapter_dirs += batch.adapter_dirs

            all_input_ids.extend(batch.all_input_ids)

            input_lengths.extend(batch.input_lengths)
            out_token_id_counts.extend(batch.out_token_id_counts)
            sampling_param_list.extend(batch.sampling_param_list)
            # Update
            cumulative_batch_size += len(batch)
        
        nopad_b_loc[:, nopad_max_len_in_batch - 1] = nopad_total_token_num - \
            new_batch_size + torch.arange(0, new_batch_size, dtype=torch.int32, device='cuda')
        nopad_b_loc_key[:, nopad_max_len_in_batch - 1] = nopad_total_token_num - \
            new_batch_size + torch.arange(0, new_batch_size, dtype=torch.int32, device='cuda')
        nopad_b_loc_value[:, nopad_max_len_in_batch - 1] = nopad_total_token_num - \
            new_batch_size + torch.arange(0, new_batch_size, dtype=torch.int32, device='cuda')

        return InferBatch(
            batch_id=batches[0].batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            input_lengths=input_lengths,
            all_input_ids=all_input_ids,
            nopad_total_token_num=nopad_total_token_num,
            nopad_max_len_in_batch=nopad_max_len_in_batch,
            nopad_b_loc=nopad_b_loc,
            nopad_b_loc_key=nopad_b_loc_key,
            nopad_b_loc_value=nopad_b_loc_value,
            nopad_b_start_loc=nopad_b_start_loc,
            nopad_b_seq_len=nopad_b_seq_len,
            out_token_id_counts=out_token_id_counts,
            sampling_param_list=sampling_param_list,
            mem_manager=batches[0].mem_manager,
            adapter_dirs=adapter_dirs,
            finetune_mask=None,
        )

    @staticmethod
    def _debug_print_merge_entry(batch1, batch2) -> None:
        """One-line `[merge]` marker. Reads:
          - both source dims (so you can see when small mini-batches join a
            big running batch),
          - the allocator's current KV-page count (so you can correlate
            with [mem_free] traces),
          - the sub-slot demand implied by the in-flight b_seq_len totals
            (so a growing gap = `actual − demand_pages` is the residual /
            orphaned-slot count).

        One GPU sync per merge (bincount + .tolist()). Merge fires only
        when a new prefill mini-batch joins the running batch — at most
        a few times per second — so the cost is irrelevant next to the
        actual merge work."""
        mm = batch1.mem_manager
        # Per-type page histogram. Cheap (single bincount + 1 sync) and
        # only runs when _DEBUG_MERGE is True.
        try:
            type_counts = torch.bincount(
                mm.page_type_map.to(torch.long),
                minlength=max(pt.value for pt in PageType) + 1,
            ).tolist()
            kv_pages = type_counts[PageType.KV_CACHE.value]
        except Exception:
            kv_pages = -1

        # In-flight KV "demand": one K + one V sub-slot per token across
        # both batches. Under packed_kv with packing factor F, that maps
        # to ceil(2 * total_tokens / F) pages (lower bound — partial
        # last pages may add up to F-1 sub-slots of slack).
        try:
            total_tokens = int(batch1.nopad_b_seq_len.sum().item()) + \
                           int(batch2.nopad_b_seq_len.sum().item())
            F = getattr(mm, "F", 1) or 1
            # Two pools (K, V) of `total_tokens` sub-slots each.
            demand_pages = (2 * total_tokens + F - 1) // F
        except Exception:
            total_tokens = -1
            demand_pages = -1

        residual = (kv_pages - demand_pages) if (kv_pages >= 0 and demand_pages >= 0) else "?"
        print(
            f"[merge] b1(len={len(batch1)}, max={batch1.nopad_max_len_in_batch}) "
            f"+ b2(len={len(batch2)}, max={batch2.nopad_max_len_in_batch}) | "
            f"alloc KV pages={kv_pages} | in-flight demand={demand_pages} "
            f"({total_tokens} tokens, F={getattr(mm, 'F', 1)}) | "
            f"residual={residual}"
        )

    def __len__(self):
        return len(self.requests)

    def is_finetuning_only(self):
        return torch.all(self.finetune_mask).item() == 1
    
    
    def get_post_sample_tensors(self):
        presence_penalties: List[float] = []
        frequency_penalties: List[float] = []
        temperatures: List[float] = []
        top_ps: List[float] = []
        top_ks: List[int] = []
        p_token_ids: List[int] = []
        p_token_counts: List[int] = []
        p_seq_len: List[int] = [0,]
        p_max_len_in_batch: int = 0
        for i, id_to_count in enumerate(self.out_token_id_counts):
            sample_param = self.sampling_param_list[i]
            presence_penalties.append(sample_param.presence_penalty)
            frequency_penalties.append(sample_param.frequency_penalty)
            temperatures.append(sample_param.temperature)
            top_ps.append(sample_param.top_p)
            top_ks.append(sample_param.top_k)
            
            for token_id, count in id_to_count.items():
                p_token_ids.append(token_id)
                p_token_counts.append(count)
            p_seq_len.append(len(id_to_count))
            p_max_len_in_batch = max(p_max_len_in_batch, len(id_to_count))
        
        presence_penalties = torch.tensor(presence_penalties, dtype=torch.float, device="cuda")
        frequency_penalties = torch.tensor(frequency_penalties, dtype=torch.float, device="cuda")
        temperatures = torch.tensor(temperatures, dtype=torch.float, device="cuda")
        top_ps = torch.tensor(top_ps, dtype=torch.float, device="cuda")
        top_ks = torch.tensor(top_ks, dtype=torch.int32, device="cuda")
        p_token_ids = torch.tensor(p_token_ids, dtype=torch.int32, device="cuda")
        p_token_counts = torch.tensor(p_token_counts, dtype=torch.int32, device="cuda")
        p_seq_len = torch.tensor(p_seq_len, dtype=torch.int32, device="cuda")
        p_cumsum_seq_len = torch.cumsum(p_seq_len, dim=0, dtype=torch.int32)
        return presence_penalties, frequency_penalties, temperatures, top_ps, top_ks, p_token_ids, p_token_counts, p_cumsum_seq_len, p_max_len_in_batch
