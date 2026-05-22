"""
CUDA Graph runner for decode (token generation) steps.

Key design decisions:
  - Cache key is (batch_size_bucket, max_len_bucket).
  - All batch-dimensional tensors are padded to bs_bucket during capture.
  - b_loc is RIGHT-ALIGNED to ml_bucket before capture/replay.
  - Force non-contiguous decode path for stable buffer addresses.
  - Pre-allocate att_m_buffers (one per layer) with max_total_tokens size.
"""

import torch
from typing import Callable, Dict, Optional, Tuple


# Shared CUDA graph memory pool for all FORWARD graphs (decode + prefill).
# Without this, every torch.cuda.graph(g) capture allocates its own private
# workspace; sharing one pool lets the captures alias non-overlapping live
# tensors and sizes total memory to the largest single captured set instead
# of the sum.
#
# DO NOT share this pool with backward graphs (SFT_service_graph) — those run
# on bwd_stream concurrently with forward, and pool sharing across concurrent
# streams is unsafe.
_FORWARD_GRAPH_POOL: Optional[Tuple[int, int]] = None


def _get_forward_graph_pool() -> Tuple[int, int]:
    """Lazy-init the shared forward-graph memory pool. CUDA must be live."""
    global _FORWARD_GRAPH_POOL
    if _FORWARD_GRAPH_POOL is None:
        _FORWARD_GRAPH_POOL = torch.cuda.graph_pool_handle()
    return _FORWARD_GRAPH_POOL


def _pad_to(tensor: torch.Tensor, target_dim0: int) -> torch.Tensor:
    """Pad tensor's first dimension to target_dim0 with zeros."""
    if tensor.shape[0] >= target_dim0:
        return tensor[:target_dim0].clone()
    pad_shape = list(tensor.shape)
    pad_shape[0] = target_dim0
    out = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    out[:tensor.shape[0]] = tensor
    return out


class CudaGraphRunner:
    """Caches and replays CUDA graphs keyed by (batch_size_bucket, max_len_bucket)."""

    BATCH_BUCKETS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
    MAX_LEN_BUCKET_SIZE = 128

    # Prefill bucket size: total_token_num is padded to nearest multiple of this.
    PREFILL_TOKEN_BUCKET_SIZE = 128

    def __init__(self, max_total_tokens: int = 25000, num_layers: int = 32,
                 tp_q_head_num: int = 32, tp_k_head_num: int = 8,
                 head_dim: int = 128,
                 embed_dim: int = 4096,
                 b_loc_width: int = None,
                 max_graph_memory_bytes: Optional[int] = None):
        self.max_total_tokens = max_total_tokens
        self.num_layers = num_layers
        self.tp_q_head_num = tp_q_head_num
        self.tp_k_head_num = tp_k_head_num
        self.head_dim = head_dim
        self.embed_dim = embed_dim
        self.b_loc_width = b_loc_width  # second dim of b_loc_key/value tensor

        # Decode cache: (bs, ml_bucket) -> (graph, bufs, output, ml_bucket)
        self._cache: Dict[Tuple[int, int], Tuple[torch.cuda.CUDAGraph, dict, torch.Tensor, int]] = {}
        # Prefill cache: (bs, T_bucket) -> (graph, bufs, output, T_bucket)
        self._prefill_cache: Dict[Tuple[int, int], Tuple[torch.cuda.CUDAGraph, dict, torch.Tensor, int]] = {}
        # Per-capture memory cost (bytes) — measured as torch.cuda.memory_allocated()
        # delta across each capture call. Used to log totals.
        self._decode_graph_bytes: Dict[Tuple[int, int], int] = {}
        self._prefill_graph_bytes: Dict[Tuple[int, int], int] = {}
        # Buckets newly captured since the last `pop_pending_*_captures()` call.
        # The manager polls these to keep its GraphEligibility mirror in sync.
        self._pending_decode_captures: list = []
        self._pending_prefill_captures: list = []
        self._warmup_iters = 3
        # Cap on combined decode+prefill graph memory. None or any
        # negative value disables the cap (capture freely). Measured
        # against `total_graph_bytes()` — the running sum of
        # `_decode_graph_bytes` + `_prefill_graph_bytes`.
        self._max_graph_memory_bytes: Optional[int] = (
            None if (max_graph_memory_bytes is None or max_graph_memory_bytes < 0)
            else int(max_graph_memory_bytes)
        )
        # One-time log when the cap first refuses a capture.
        self._cap_warned: bool = False

    def total_decode_graph_bytes(self) -> int:
        return sum(self._decode_graph_bytes.values())

    def total_prefill_graph_bytes(self) -> int:
        return sum(self._prefill_graph_bytes.values())

    def total_graph_bytes(self) -> int:
        """Combined decode + prefill graph memory (bytes). Source of truth
        for the cap and for the manager-side memory readout."""
        return self.total_decode_graph_bytes() + self.total_prefill_graph_bytes()

    def can_capture_more(self) -> bool:
        """True iff the runtime is allowed to capture another graph.
        False once `total_graph_bytes()` has reached the configured cap;
        callers should fall back to eager forward in that case."""
        if self._max_graph_memory_bytes is None:
            return True
        return self.total_graph_bytes() < self._max_graph_memory_bytes

    def note_capture_refused(self) -> None:
        """Hook for callers to record that a capture was skipped due to
        the memory cap. Emits a one-time INFO log so the operator knows
        the cap is now active."""
        if not self._cap_warned:
            self._cap_warned = True
            cap_gb = (self._max_graph_memory_bytes or 0) / (1024.0 ** 3)
            cur_gb = self.total_graph_bytes() / (1024.0 ** 3)
            print(
                f"[CudaGraphRunner] graph memory cap reached "
                f"({cur_gb:.2f}/{cap_gb:.2f} GB); further runtime captures "
                f"are disabled — uncaptured buckets will run eager."
            )

    def num_decode_graphs(self) -> int:
        return len(self._cache)

    def num_prefill_graphs(self) -> int:
        return len(self._prefill_cache)

    def all_decode_buckets(self) -> list:
        """Snapshot of all currently captured decode bucket keys (for mirror seeding)."""
        return list(self._cache.keys())

    def all_prefill_buckets(self) -> list:
        """Snapshot of all currently captured prefill bucket keys (for mirror seeding)."""
        return list(self._prefill_cache.keys())

    def pop_pending_decode_captures(self) -> list:
        """Drain & return decode bucket keys captured since last call. Manager-side
        GraphEligibility mirror appends these on each scheduler iteration."""
        out = self._pending_decode_captures
        self._pending_decode_captures = []
        return out

    def pop_pending_prefill_captures(self) -> list:
        """Drain & return prefill bucket keys captured since last call."""
        out = self._pending_prefill_captures
        self._pending_prefill_captures = []
        return out

    @staticmethod
    def get_batch_bucket(batch_size: int) -> int:
        for b in CudaGraphRunner.BATCH_BUCKETS:
            if batch_size <= b:
                return b
        return batch_size

    @staticmethod
    def get_max_len_bucket(max_len: int) -> int:
        bs = CudaGraphRunner.MAX_LEN_BUCKET_SIZE
        return ((max_len + bs - 1) // bs) * bs

    def get_cache_key(self, batch_size: int, max_len_in_batch: int) -> Tuple[int, int]:
        # Use exact batch_size (not bucketed) because LoRA dispatch requires
        # exact adapter mappings that can't be padded
        return (batch_size, self.get_max_len_bucket(max_len_in_batch))

    def has_graph(self, batch_size: int, max_len_in_batch: int) -> bool:
        return self.get_cache_key(batch_size, max_len_in_batch) in self._cache

    # ─── Prefill bucketing helpers ──────────────────────────────────────
    PREFILL_BS_BUCKETS = [1, 2, 4, 8, 16, 32, 64]

    @staticmethod
    def get_prefill_token_bucket(total_token_num: int) -> int:
        bs = CudaGraphRunner.PREFILL_TOKEN_BUCKET_SIZE
        return ((total_token_num + bs - 1) // bs) * bs

    @staticmethod
    def get_prefill_bs_bucket(batch_size: int) -> int:
        for b in CudaGraphRunner.PREFILL_BS_BUCKETS:
            if batch_size <= b:
                return b
        return batch_size

    def get_prefill_cache_key(self, batch_size: int, total_token_num: int) -> Tuple[int, int]:
        return (self.get_prefill_bs_bucket(batch_size),
                self.get_prefill_token_bucket(total_token_num))

    def has_prefill_graph(self, batch_size: int, total_token_num: int) -> bool:
        return self.get_prefill_cache_key(batch_size, total_token_num) in self._prefill_cache

    def _realign_b_loc(self, static_buf, real_buf, batch_size, actual_max_len, bucket_max_len):
        shift = bucket_max_len - actual_max_len
        static_buf[:batch_size].zero_()
        if actual_max_len > 0 and shift >= 0:
            src_cols = min(actual_max_len, real_buf.shape[1])
            static_buf[:batch_size, shift:shift + src_cols].copy_(
                real_buf[:batch_size, :src_cols])

    def _create_att_m_buffers(self, dtype):
        buffers = {}
        for layer_id in range(self.num_layers):
            buffers[layer_id] = torch.empty(
                (self.tp_q_head_num, self.max_total_tokens),
                dtype=dtype, device="cuda")
        return buffers

    def capture(self, batch_size, max_len_in_batch, token_forward_fn,
                input_ids, infer_state, forward_kwargs) -> torch.Tensor:
        infer_state.mem_manager.page_table_lock.acquire()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()
        ml_bucket = self.get_max_len_bucket(max_len_in_batch)
        cache_key = (batch_size, ml_bucket)

        # Pre-allocate att_m_buffers (must be float16, NOT input_ids.dtype which is int)
        att_m_buffers = self._create_att_m_buffers(torch.float16)
        infer_state._att_m_buffers = att_m_buffers

        # Clone all tensors (exact batch_size, no padding)
        static_input_ids = input_ids.clone()
        static_b_seq_len = infer_state.b_seq_len.clone()
        static_b_start_loc = infer_state.b_start_loc.clone()
        static_position_cos = infer_state.position_cos.clone()
        static_position_sin = infer_state.position_sin.clone()

        # b_loc (unified mem path) - re-align to ml_bucket
        static_b_loc_key = None
        static_b_loc_value = None
        if hasattr(infer_state, 'b_loc_key') and infer_state.b_loc_key is not None:
            static_b_loc_key = torch.zeros_like(infer_state.b_loc_key)
            self._realign_b_loc(static_b_loc_key, infer_state.b_loc_key,
                                batch_size, max_len_in_batch, ml_bucket)
            static_b_loc_value = torch.zeros_like(infer_state.b_loc_value)
            self._realign_b_loc(static_b_loc_value, infer_state.b_loc_value,
                                batch_size, max_len_in_batch, ml_bucket)

        # b_loc (regular mem path)
        static_b_loc = None
        if hasattr(infer_state, 'b_loc') and infer_state.b_loc is not None:
            static_b_loc = torch.zeros_like(infer_state.b_loc)
            self._realign_b_loc(static_b_loc, infer_state.b_loc,
                                batch_size, max_len_in_batch, ml_bucket)

        # Decode mem indices
        static_decode_mem_index_key = None
        static_decode_mem_index_value = None
        static_decode_mem_index = None
        if hasattr(infer_state, 'decode_mem_index_key') and infer_state.decode_mem_index_key is not None:
            static_decode_mem_index_key = infer_state.decode_mem_index_key.clone()
            static_decode_mem_index_value = infer_state.decode_mem_index_value.clone()
        if hasattr(infer_state, 'decode_mem_index') and infer_state.decode_mem_index is not None:
            static_decode_mem_index = infer_state.decode_mem_index.clone()

        # Decode KV scratch buffers
        static_decode_key_buffer = None
        static_decode_value_buffer = None
        if hasattr(infer_state, 'decode_key_buffer') and infer_state.decode_key_buffer is not None:
            static_decode_key_buffer = infer_state.decode_key_buffer.clone()
            static_decode_value_buffer = infer_state.decode_value_buffer.clone()

        # Wire static buffers into infer_state
        infer_state.b_seq_len = static_b_seq_len
        infer_state.b_start_loc = static_b_start_loc
        infer_state.position_cos = static_position_cos
        infer_state.position_sin = static_position_sin
        infer_state.max_len_in_batch = ml_bucket

        if static_b_loc_key is not None:
            infer_state.b_loc_key = static_b_loc_key
            infer_state.b_loc_value = static_b_loc_value
        if static_b_loc is not None:
            infer_state.b_loc = static_b_loc
        if static_decode_mem_index_key is not None:
            infer_state.decode_mem_index_key = static_decode_mem_index_key
            infer_state.decode_mem_index_value = static_decode_mem_index_value
        if static_decode_mem_index is not None:
            infer_state.decode_mem_index = static_decode_mem_index
        if static_decode_key_buffer is not None:
            infer_state.decode_key_buffer = static_decode_key_buffer
            infer_state.decode_value_buffer = static_decode_value_buffer

        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(self._warmup_iters):
                _ = token_forward_fn(static_input_ids, infer_state, **forward_kwargs)
        torch.cuda.current_stream().wait_stream(s)

        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=_get_forward_graph_pool()):
            static_output = token_forward_fn(static_input_ids, infer_state, **forward_kwargs)

        input_buffers = {
            'input_ids': static_input_ids,
            'b_seq_len': static_b_seq_len,
            'b_start_loc': static_b_start_loc,
            'position_cos': static_position_cos,
            'position_sin': static_position_sin,
            'b_loc_key': static_b_loc_key,
            'b_loc_value': static_b_loc_value,
            'b_loc': static_b_loc,
            'decode_mem_index_key': static_decode_mem_index_key,
            'decode_mem_index_value': static_decode_mem_index_value,
            'decode_mem_index': static_decode_mem_index,
            'decode_key_buffer': static_decode_key_buffer,
            'decode_value_buffer': static_decode_value_buffer,
            'att_m_buffers': att_m_buffers,
        }

        self._cache[cache_key] = (graph, input_buffers, static_output, ml_bucket)
        self._pending_decode_captures.append(cache_key)
        torch.cuda.synchronize()
        self._decode_graph_bytes[cache_key] = torch.cuda.memory_allocated() - mem_before
        infer_state.mem_manager.page_table_lock.release()
        return static_output[:batch_size]

    def replay(self, batch_size, max_len_in_batch, input_ids, infer_state) -> torch.Tensor:
        cache_key = self.get_cache_key(batch_size, max_len_in_batch)
        graph, bufs, static_output, ml_bucket = self._cache[cache_key]

        # Copy current values into static buffers (exact size, no padding)
        bufs['input_ids'].copy_(input_ids)
        bufs['b_seq_len'].copy_(infer_state.b_seq_len)
        bufs['b_start_loc'].copy_(infer_state.b_start_loc)
        bufs['position_cos'].copy_(infer_state.position_cos)
        bufs['position_sin'].copy_(infer_state.position_sin)

        # Re-align b_loc to ml_bucket
        if bufs['b_loc_key'] is not None and hasattr(infer_state, 'b_loc_key') and infer_state.b_loc_key is not None:
            self._realign_b_loc(bufs['b_loc_key'], infer_state.b_loc_key,
                                batch_size, max_len_in_batch, ml_bucket)
            self._realign_b_loc(bufs['b_loc_value'], infer_state.b_loc_value,
                                batch_size, max_len_in_batch, ml_bucket)
        if bufs['b_loc'] is not None and hasattr(infer_state, 'b_loc') and infer_state.b_loc is not None:
            self._realign_b_loc(bufs['b_loc'], infer_state.b_loc,
                                batch_size, max_len_in_batch, ml_bucket)

        # Update decode mem indices
        if bufs['decode_mem_index_key'] is not None and hasattr(infer_state, 'decode_mem_index_key') and infer_state.decode_mem_index_key is not None:
            bufs['decode_mem_index_key'].copy_(infer_state.decode_mem_index_key)
            bufs['decode_mem_index_value'].copy_(infer_state.decode_mem_index_value)
        if bufs['decode_mem_index'] is not None and hasattr(infer_state, 'decode_mem_index') and infer_state.decode_mem_index is not None:
            bufs['decode_mem_index'].copy_(infer_state.decode_mem_index)

        # Point infer_state to static buffers
        infer_state.b_seq_len = bufs['b_seq_len']
        infer_state.b_start_loc = bufs['b_start_loc']
        infer_state.position_cos = bufs['position_cos']
        infer_state.position_sin = bufs['position_sin']
        infer_state.max_len_in_batch = ml_bucket
        infer_state._att_m_buffers = bufs['att_m_buffers']

        if bufs['b_loc_key'] is not None:
            infer_state.b_loc_key = bufs['b_loc_key']
            infer_state.b_loc_value = bufs['b_loc_value']
        if bufs['b_loc'] is not None:
            infer_state.b_loc = bufs['b_loc']
        if bufs['decode_mem_index_key'] is not None:
            infer_state.decode_mem_index_key = bufs['decode_mem_index_key']
            infer_state.decode_mem_index_value = bufs['decode_mem_index_value']
        if bufs['decode_mem_index'] is not None:
            infer_state.decode_mem_index = bufs['decode_mem_index']
        if bufs['decode_key_buffer'] is not None:
            infer_state.decode_key_buffer = bufs['decode_key_buffer']
            infer_state.decode_value_buffer = bufs['decode_value_buffer']

        graph.replay()
        return static_output

    # ─── Prefill capture / replay ───────────────────────────────────────
    #
    # Scope: inference-only (no finetune), non-int8 KV.
    # Cache key: (bs_bucket, total_token_num_bucket).
    #
    # Captures the full _context_forward path (incl. attention) so attention's
    # max_input_len, b_start_loc, b_seq_len are baked into the captured kernels.
    # Supports batch_size in {1, 2, 4, 8, 16, 32, 64} via padding to bs_bucket:
    # padded request slots have b_seq_len=0 (attention skips them) and write
    # K/V into dedicated scratch slots that are never read.
    def prefill_capture(self, engine, batch_size, total_token_num, context_forward_fn,
                        input_ids, infer_state, batch_req_bins, forward_kwargs) -> torch.Tensor:
        """Capture a CUDA graph for prefill.

        `engine` is the LoraUnorderedBatchMixed instance — we update engine.batch_req_bins
        and engine.delta to point at static padded buffers so the captured kernels
        reference stable addresses.
        `batch_req_bins` is the real (unpadded) [total_token_num] index tensor.
        """
        infer_state.mem_manager.page_table_lock.acquire()
        try:
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated()
            T_bucket = self.get_prefill_token_bucket(total_token_num)
            bs_bucket = self.get_prefill_bs_bucket(batch_size)
            cache_key = (bs_bucket, T_bucket)

            # Swap in T_bucket-sized delta scratch buffers for LoRA (persistent across
            # capture/replay of the same bucket).
            max_lora_dim = engine.max_lora_dim
            static_delta = [
                torch.zeros((T_bucket, max_lora_dim), dtype=torch.float16, device="cuda")
                for _ in range(3)
            ]
            engine.delta = static_delta

            infer_state._att_m_buffers = self._create_att_m_buffers(torch.float16)

            # Pad input_ids
            static_input_ids = torch.zeros(T_bucket, dtype=input_ids.dtype, device="cuda")
            static_input_ids[:total_token_num].copy_(input_ids)

            # Pad position_cos / position_sin
            pc = infer_state.position_cos
            ps = infer_state.position_sin
            static_position_cos = torch.zeros((T_bucket, pc.shape[1]), dtype=pc.dtype, device="cuda")
            static_position_sin = torch.zeros((T_bucket, ps.shape[1]), dtype=ps.dtype, device="cuda")
            static_position_cos[:total_token_num].copy_(pc)
            static_position_sin[:total_token_num].copy_(ps)

            # b_start_loc / b_seq_len padded to bs_bucket. Padded slots have
            # b_seq_len=0 so attention treats them as zero-length requests.
            # b_start_loc for padded slots is set to total_token_num (one past
            # the last real token) so the kernel grid math is sane.
            static_b_start_loc = torch.zeros(bs_bucket, dtype=infer_state.b_start_loc.dtype, device="cuda")
            static_b_seq_len = torch.zeros(bs_bucket, dtype=infer_state.b_seq_len.dtype, device="cuda")
            static_b_start_loc[:batch_size].copy_(infer_state.b_start_loc)
            static_b_seq_len[:batch_size].copy_(infer_state.b_seq_len)
            if batch_size < bs_bucket:
                static_b_start_loc[batch_size:].fill_(int(total_token_num))

            # Pad batch_req_bins
            static_batch_req_bins = torch.zeros(T_bucket, dtype=batch_req_bins.dtype, device="cuda")
            static_batch_req_bins[:total_token_num].copy_(batch_req_bins)

            # prefill_mem_index_{key,value} are already T_bucket-sized (allocated by
            # _prefill when CUDA graph is enabled). Clone for stable addresses.
            mk = infer_state.prefill_mem_index_key
            mv = infer_state.prefill_mem_index_value
            assert mk.shape[0] == T_bucket, (
                f"prefill_mem_index_key size {mk.shape[0]} != T_bucket {T_bucket}; "
                f"_prefill must allocate T_bucket KV slots when use_prefill_cg=True")
            static_mem_index_key = mk.clone()
            static_mem_index_value = mv.clone()
            static_mem_index_cat = torch.cat([static_mem_index_key, static_mem_index_value], dim=0)

            # Prefill K/V scratch buffers sized to T_bucket
            static_prefill_key_buffer = torch.empty(
                (T_bucket, self.tp_k_head_num, self.head_dim),
                dtype=torch.float16, device="cuda")
            static_prefill_value_buffer = torch.empty(
                (T_bucket, self.tp_k_head_num, self.head_dim),
                dtype=torch.float16, device="cuda")

            # b_loc_key/value padded to bs_bucket so the captured buffer can
            # accept any real batch_size up to bs_bucket. Padded request rows
            # are zeros — kernels skip them because the matching b_seq_len=0.
            static_b_loc_key = torch.zeros(
                (bs_bucket, infer_state.b_loc_key.shape[1]),
                dtype=infer_state.b_loc_key.dtype, device="cuda")
            static_b_loc_value = torch.zeros(
                (bs_bucket, infer_state.b_loc_value.shape[1]),
                dtype=infer_state.b_loc_value.dtype, device="cuda")
            static_b_loc_key[:batch_size].copy_(infer_state.b_loc_key)
            static_b_loc_value[:batch_size].copy_(infer_state.b_loc_value)

            # finetune_mask: all zeros (inference-only path)
            static_finetune_mask = torch.zeros(T_bucket, dtype=torch.bool, device="cuda")

            # Wire statics into infer_state + engine
            infer_state.b_start_loc = static_b_start_loc
            infer_state.b_seq_len = static_b_seq_len
            infer_state.position_cos = static_position_cos
            infer_state.position_sin = static_position_sin
            infer_state.prefill_mem_index_key = static_mem_index_key
            infer_state.prefill_mem_index_value = static_mem_index_value
            infer_state.prefill_mem_index_cat = static_mem_index_cat
            infer_state.prefill_key_buffer = static_prefill_key_buffer
            infer_state.prefill_value_buffer = static_prefill_value_buffer
            infer_state.b_loc_key = static_b_loc_key
            infer_state.b_loc_value = static_b_loc_value
            infer_state.finetune_mask = static_finetune_mask
            infer_state.total_token_num = T_bucket
            infer_state.max_len_in_batch = int(total_token_num)
            engine.batch_req_bins = static_batch_req_bins

            # Prefill warmup: 1 iter (not 3 like decode). Empirically, more warmup
            # iters on the side stream leave state that makes the captured forward
            # produce wrong logits on the first request. The post-capture replay()
            # below ensures static_output reflects the current input.
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                _ = context_forward_fn(static_input_ids, infer_state, **forward_kwargs)
            torch.cuda.current_stream().wait_stream(s)

            # Capture, then replay once so static_output is populated for the CAPTURE
            # caller (otherwise the first request returns stale/uninitialized values).
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=_get_forward_graph_pool()):
                static_output = context_forward_fn(static_input_ids, infer_state, **forward_kwargs)
            graph.replay()
            torch.cuda.current_stream().synchronize()

            bufs = {
                'input_ids': static_input_ids,
                'b_start_loc': static_b_start_loc,
                'b_seq_len': static_b_seq_len,
                'position_cos': static_position_cos,
                'position_sin': static_position_sin,
                'batch_req_bins': static_batch_req_bins,
                'prefill_mem_index_key': static_mem_index_key,
                'prefill_mem_index_value': static_mem_index_value,
                'prefill_mem_index_cat': static_mem_index_cat,
                'prefill_key_buffer': static_prefill_key_buffer,
                'prefill_value_buffer': static_prefill_value_buffer,
                'b_loc_key': static_b_loc_key,
                'b_loc_value': static_b_loc_value,
                'finetune_mask': static_finetune_mask,
                'att_m_buffers': infer_state._att_m_buffers,
                'delta': static_delta,
            }
            self._prefill_cache[cache_key] = (graph, bufs, static_output, T_bucket)
            self._pending_prefill_captures.append(cache_key)
            torch.cuda.synchronize()
            self._prefill_graph_bytes[cache_key] = torch.cuda.memory_allocated() - mem_before
            return static_output
        finally:
            infer_state.mem_manager.page_table_lock.release()

    def prefill_replay(self, engine, batch_size, total_token_num, input_ids, infer_state,
                       batch_req_bins) -> torch.Tensor:
        cache_key = self.get_prefill_cache_key(batch_size, total_token_num)
        graph, bufs, static_output, T_bucket = self._prefill_cache[cache_key]
        bs_bucket = self.get_prefill_bs_bucket(batch_size)

        # Swap engine's delta to the persistent static buffers captured with this graph.
        engine.delta = bufs['delta']

        bufs['input_ids'][total_token_num:].zero_()
        bufs['input_ids'][:total_token_num].copy_(input_ids)

        bufs['position_cos'][total_token_num:].zero_()
        bufs['position_cos'][:total_token_num].copy_(infer_state.position_cos)
        bufs['position_sin'][total_token_num:].zero_()
        bufs['position_sin'][:total_token_num].copy_(infer_state.position_sin)

        # Pad b_start_loc / b_seq_len to bs_bucket — padded slots have seq_len=0
        # so attention treats them as no-ops.
        bufs['b_start_loc'].zero_()
        bufs['b_seq_len'].zero_()
        bufs['b_start_loc'][:batch_size].copy_(infer_state.b_start_loc)
        bufs['b_seq_len'][:batch_size].copy_(infer_state.b_seq_len)
        if batch_size < bs_bucket:
            bufs['b_start_loc'][batch_size:].fill_(int(total_token_num))

        bufs['batch_req_bins'][total_token_num:].zero_()
        bufs['batch_req_bins'][:total_token_num].copy_(batch_req_bins)

        # prefill_mem_index_{key,value} are already T_bucket-sized (dedicated padding slots)
        assert infer_state.prefill_mem_index_key.shape[0] == T_bucket
        bufs['prefill_mem_index_key'].copy_(infer_state.prefill_mem_index_key)
        bufs['prefill_mem_index_value'].copy_(infer_state.prefill_mem_index_value)
        bufs['prefill_mem_index_cat'][:T_bucket].copy_(bufs['prefill_mem_index_key'])
        bufs['prefill_mem_index_cat'][T_bucket:].copy_(bufs['prefill_mem_index_value'])

        bufs['b_loc_key'].zero_()
        bufs['b_loc_value'].zero_()
        bufs['b_loc_key'][:batch_size].copy_(infer_state.b_loc_key)
        bufs['b_loc_value'][:batch_size].copy_(infer_state.b_loc_value)

        bufs['finetune_mask'].zero_()

        # Rewire
        infer_state.b_start_loc = bufs['b_start_loc']
        infer_state.b_seq_len = bufs['b_seq_len']
        infer_state.position_cos = bufs['position_cos']
        infer_state.position_sin = bufs['position_sin']
        infer_state.prefill_mem_index_key = bufs['prefill_mem_index_key']
        infer_state.prefill_mem_index_value = bufs['prefill_mem_index_value']
        infer_state.prefill_mem_index_cat = bufs['prefill_mem_index_cat']
        infer_state.prefill_key_buffer = bufs['prefill_key_buffer']
        infer_state.prefill_value_buffer = bufs['prefill_value_buffer']
        infer_state.b_loc_key = bufs['b_loc_key']
        infer_state.b_loc_value = bufs['b_loc_value']
        infer_state.finetune_mask = bufs['finetune_mask']
        infer_state._att_m_buffers = bufs['att_m_buffers']
        infer_state.total_token_num = T_bucket
        infer_state.max_len_in_batch = int(total_token_num)
        engine.batch_req_bins = bufs['batch_req_bins']

        graph.replay()
        return static_output

    def reset(self):
        self._cache.clear()
        self._prefill_cache.clear()
        self._decode_graph_bytes.clear()
        self._prefill_graph_bytes.clear()


