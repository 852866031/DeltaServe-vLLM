import torch

from dserve.common.unified_mem_allocator import (
    UnifiedMemoryAllocator,
    PageType,
)


# ─── Free-trace debug toggle ─────────────────────────────────────────
# Set True to print one line per page-free call so leaks (pages alloc'd
# but never freed → `used` climbs forever) become visible. Each gated
# block is wrapped in `if _DEBUG_FREE:` so when False the cost is one
# LOAD_GLOBAL + branch per free call — sub-microsecond, negligible
# relative to the GPU work.
_DEBUG_FREE = False


class PackedKVMemoryAllocator(UnifiedMemoryAllocator):
    """
    Packs F = num_attention_heads / num_key_value_heads KV sub-slots into
    each physical page. KV-reading kernels see a reshape view where each
    row is a [num_kv_heads, head_dim] sub-slot; activation, adapter, and
    embedding pages keep the original [num_attention_heads, head_dim]
    layout.

    Allocation strategy: always claim whole pages from the parent
    allocator (`ceil(n / F)` pages), expose the first `n` sub-slots from
    those pages to the caller, and track per-page refcount so the page
    returns to FREE only when every sub-slot the caller received has
    been freed. Trade-off: a partial last page's unused sub-slots are
    "wasted" until the page fully drains — but we keep alloc fast
    (parent's hot path with no extra per-sub-slot scatter) and avoid the
    sync-heavy partial-page scan that the previous design needed.

    Index spaces:
      - alloc(KV_CACHE) / alloc_contiguous_kv -> sub-slot ids in [0, tot_size * F)
      - alloc(other PageType)                 -> page ids in [0, tot_size)

    KV freers MUST call free_kv (sub-slot semantics). free() is inherited
    from the parent and is for PAGE-level frees (activation, adapter,
    embedding); sub-slot ids in [0, tot_size) collide with page ids and
    cannot be safely disambiguated by range.
    """

    def __init__(self, head_num, head_dim, vocab_size, layer_num,
                 max_pool_size, dtype=torch.float16, device="cuda",
                 log_path=None, max_finetuning_tokens=1024,
                 num_kv_heads=None):
        super().__init__(head_num, head_dim, vocab_size, layer_num,
                         max_pool_size, dtype, device, log_path,
                         max_finetuning_tokens)

        kv = num_kv_heads if num_kv_heads is not None else head_num
        assert head_num % kv == 0, (
            f"num_attention_heads={head_num} must be divisible by "
            f"num_key_value_heads={kv}"
        )
        self.F = head_num // kv
        self.num_kv_heads = kv

        # Reshape view — same physical memory, F * tot_size rows of
        # [num_kv_heads, head_dim] each. Free metadata change.
        self._gpu_pools_kv = [
            p.view(self.tot_size * self.F, kv, head_dim)
            for p in self.gpu_pools
        ]

        # Per-page refcount: number of sub-slots from this page currently
        # held by the caller. Set on alloc, decremented in free_kv via
        # `kv_refcount -= bincount(page_ids)`, page returns to FREE when
        # it hits 0. int64 dtype matches torch.bincount's output so the
        # decrement is a single in-place subtract with no cast kernel.
        # Only meaningful when page_type_map[p] == KV_CACHE.
        self.kv_refcount = torch.zeros(
            self.tot_size, dtype=torch.int64, device=device,
        )

        print(f"PackedKVMemoryAllocator: F={self.F}, "
              f"num_kv_heads={kv}, KV view shape={tuple(self._gpu_pools_kv[0].shape)}")

    # ------------------------------------------------------------------
    # Pool accessor — only KV view is overridden; activation and adapter
    # accessors inherit from the parent.
    # ------------------------------------------------------------------
    def get_kv_pool(self, layer_id: int) -> torch.Tensor:
        return self._gpu_pools_kv[layer_id]

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------
    def alloc(self, num_pages: int, page_type: PageType) -> torch.Tensor:
        try:
            if page_type == PageType.KV_CACHE:
                return self._alloc_kv(num_pages)
            return super().alloc(num_pages, page_type)
        except RuntimeError as e:
            # Parent raises RuntimeError("Not enough free pages: …") on OOP.
            # Emit a structured diagnosis of where the pages went, plus the
            # GPU's view of memory pressure, then re-raise so the caller's
            # stack trace still points at the failing allocation site.
            self._print_oop_diagnosis(num_pages, page_type, original_error=e)
            raise

    def _print_oop_diagnosis(self,
                             num_pages: int,
                             page_type: "PageType",
                             original_error: BaseException) -> None:
        """Page-level breakdown + GPU memory snapshot at OOP time."""
        MB = 1024.0 * 1024.0
        GB = 1024.0 * 1024.0 * 1024.0
        sep = "═" * 72
        lines = [
            "",
            f"\033[31m{sep}",
            f"[PackedKVMemoryAllocator] OUT OF PAGES while allocating "
            f"{num_pages} page(s) of {page_type.name}",
            sep + "\033[0m",
            f"Failure: {original_error}",
            "",
            f"Pool capacity        : tot_size = {self.tot_size} pages "
            f"(F={self.F}, num_kv_heads={self.num_kv_heads})",
        ]

        # ── Per-PageType page counts. Use the page_type_map (single GPU
        # sync) instead of free_bitmap so the breakdown is authoritative
        # even if _used_pages and free_bitmap ever drift.
        try:
            ptm = self.page_type_map.detach().to("cpu", non_blocking=False)
            counts = torch.bincount(
                ptm.to(torch.long),
                minlength=max(pt.value for pt in PageType) + 1,
            ).tolist()
            total_counted = sum(counts)
            lines.append(
                f"Page accounting      : used={self._used_pages}, "
                f"free(bitmap)={self.tot_size - self._used_pages}, "
                f"sum(page_type_map)={total_counted}"
            )
            lines.append("Pages by type        :")
            for pt in PageType:
                n = counts[pt.value] if pt.value < len(counts) else 0
                pct = (n / max(self.tot_size, 1)) * 100.0
                lines.append(f"  {pt.name:<28} {n:>10} pages  ({pct:5.1f}%)")
        except Exception as e:
            lines.append(f"Page-type breakdown unavailable: {e}")

        # ── Packed-KV specific: refcount distribution on KV pages.
        try:
            kv_mask = self.page_type_map == int(PageType.KV_CACHE.value)
            kv_n = int(kv_mask.sum().item())
            if kv_n > 0:
                refs = self.kv_refcount[kv_mask].to(torch.long)
                full = int((refs == self.F).sum().item())
                partial = int(((refs > 0) & (refs < self.F)).sum().item())
                zero = int((refs == 0).sum().item())
                mean_ref = float(refs.float().mean().item())
                wasted_slots = (self.F * full + partial * self.F  # noqa: F841
                                ) - int(refs.sum().item())
                lines.append(
                    f"KV refcount (F={self.F}): full={full}, partial={partial}, "
                    f"zero={zero}, mean={mean_ref:.2f} / {self.F}"
                )
                if partial > 0:
                    lines.append(
                        f"  → partial pages hold "
                        f"{int(refs[(refs > 0) & (refs < self.F)].sum().item())} "
                        f"of {partial * self.F} possible sub-slots "
                        f"({partial * self.F - int(refs[(refs > 0) & (refs < self.F)].sum().item())} "
                        f"sub-slot(s) wasted until those pages fully drain)"
                    )
        except Exception as e:
            lines.append(f"KV refcount breakdown unavailable: {e}")

        # ── GPU-side memory snapshot. cuda.mem_get_info() returns
        # (free_bytes, total_bytes) at the driver level — this is the
        # true free including OTHER processes (e.g. backward subproc).
        # PyTorch's memory_allocated()/memory_reserved() only reports
        # this process's caching allocator usage.
        lines.append("")
        free_gb: "float | None" = None
        try:
            if torch.cuda.is_available():
                free_b, total_b = torch.cuda.mem_get_info()
                free_gb = free_b / GB
                lines.append(
                    f"Driver memory        : free={free_gb:.2f} GB / "
                    f"total={total_b / GB:.2f} GB  "
                    f"(used by all procs={(total_b - free_b) / GB:.2f} GB)"
                )
                lines.append(
                    f"This process (PyTorch): "
                    f"allocated={torch.cuda.memory_allocated() / GB:.2f} GB, "
                    f"reserved={torch.cuda.memory_reserved() / GB:.2f} GB, "
                    f"reserved-but-unallocated="
                    f"{(torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / MB:.1f} MB"
                )
                max_alloc = torch.cuda.max_memory_allocated() / GB
                lines.append(
                    f"This process peak    : max_allocated={max_alloc:.2f} GB"
                )
            else:
                lines.append("CUDA unavailable; skipping driver memory snapshot.")
        except Exception as e:
            lines.append(f"Driver memory snapshot failed: {e}")

        # ── Sizing suggestion. Compare the gpu_pools footprint (what
        # `memory.unified_mem_manager_max_size_gb` controls) against
        # how much free driver memory is left. If there's meaningful
        # headroom, suggest growing the pool; otherwise point at other
        # knobs to relieve pressure.
        lines.append("")
        try:
            elem_bytes = torch.empty(0, dtype=self.dtype).element_size()
            page_bytes = int(self.head_num) * int(self.head_dim) * int(elem_bytes)
            current_pool_gb = (self.tot_size * self.layer_num * page_bytes) / GB
            lines.append(
                f"Current pool footprint: {current_pool_gb:.2f} GB "
                f"(tot_size={self.tot_size} × layers={self.layer_num} × "
                f"page={page_bytes / 1024.0:.2f} KB)"
            )
            # Reserve some driver-side headroom for the caching allocator,
            # graph-capture pools, transient activations, and other procs.
            HEADROOM_GB = 2.0
            MIN_GROWTH_GB = 0.5
            if free_gb is None:
                lines.append(
                    "Suggestion: driver free memory unavailable — can't "
                    "advise on pool sizing."
                )
            elif free_gb - HEADROOM_GB >= MIN_GROWTH_GB:
                growth_gb = free_gb - HEADROOM_GB
                suggested_gb = current_pool_gb + growth_gb
                lines.append(
                    f"Suggestion: \033[33m~{growth_gb:.1f} GB headroom\033[0m "
                    f"at the driver (free={free_gb:.2f} GB minus a "
                    f"{HEADROOM_GB:.1f} GB safety margin for caching "
                    f"allocator + graph pools)."
                )
                lines.append(
                    f"  → raise \033[33mmemory.unified_mem_manager_max_size_gb\033[0m "
                    f"from {current_pool_gb:.1f} → ~{suggested_gb:.1f} "
                    f"in the active YAML (or pass "
                    f"--override memory.unified_mem_manager_max_size_gb="
                    f"{suggested_gb:.1f} to launch_*.py)."
                )
            else:
                lines.append(
                    f"Suggestion: only {free_gb:.2f} GB free at driver level; "
                    "growing the pool won't help. Try instead:"
                )
                lines.append(
                    "  - lower cuda_graph.prefill_sweep_max_tokens / set "
                    "cuda_graph.max_graph_memory_gb to cap graph memory"
                )
                lines.append(
                    "  - lower serving.batch_max_tokens / serving.max_req_total_len"
                )
                lines.append(
                    "  - shrink cuda_graph.attn_l_max / attn_bn_max so the "
                    "padded-attn ctx and per-layer activation pools are smaller"
                )
                lines.append(
                    "  - export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
                    "to fight fragmentation"
                )
        except Exception as e:
            lines.append(f"Pool-sizing suggestion unavailable: {e}")

        lines.append(f"\033[31m{sep}\033[0m")
        print("\n".join(lines), flush=True)

    def _alloc_kv(self, n: int) -> torch.Tensor:
        """Whole-page allocation: claim ceil(n/F) pages from the parent,
        return the first n sub-slots derived arithmetically from those
        page ids. The last page may be partial (refcount < F)."""
        if self.F == 1:
            return super().alloc(n, PageType.KV_CACHE)
        if n == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)

        with self.page_table_lock:
            need_pages = (n + self.F - 1) // self.F
            page_ids = super().alloc(need_pages, PageType.KV_CACHE)

            # Refcount: write F to every picked page, patch the last one
            # if the trailing page is only partially filled. Aligned
            # allocations (n divisible by F) skip the patch entirely —
            # one kernel total in the fast path.
            self.kv_refcount[page_ids] = self.F
            if n % self.F != 0:
                self.kv_refcount[page_ids[-1]] = n - (need_pages - 1) * self.F

            # Sub-slot ids: each page contributes F consecutive ids
            # starting at page_id*F. Take first n in the flattened view.
            offsets = torch.arange(self.F, device=self.device)
            sub_ids = (page_ids.unsqueeze(1) * self.F + offsets
                       ).flatten()[:n]
            return sub_ids

    def alloc_contiguous_kv(self, need_size: int, page_type: PageType):
        """Returns 2*need_size consecutive sub-slot ids (K half then V
        half) backed by 2 * ceil(need_size / F) consecutive FREE pages,
        or None if no such run exists. The K half and V half each have
        their last page possibly partial."""
        assert page_type == PageType.KV_CACHE
        if self.F == 1:
            return super().alloc_contiguous_kv(need_size, page_type)

        with self.page_table_lock:
            need_pages_per_half = (need_size + self.F - 1) // self.F
            result = super().alloc_contiguous_kv(need_pages_per_half, page_type)
            if result is None:
                return None
            phys_k_pages, sk_pg, ek_pg, phys_v_pages, sv_pg, ev_pg = result

            # Refcount on K and V halves: write F to every picked page,
            # patch the trailing page if need_size isn't divisible by F.
            self.kv_refcount[phys_k_pages] = self.F
            self.kv_refcount[phys_v_pages] = self.F
            if need_size % self.F != 0:
                last_used = need_size - (need_pages_per_half - 1) * self.F
                self.kv_refcount[phys_k_pages[-1]] = last_used
                self.kv_refcount[phys_v_pages[-1]] = last_used

            # Sub-slot ranges. K = [sk_pg*F, sk_pg*F + need_size).
            # V = [sv_pg*F, sv_pg*F + need_size).
            sub_k_start = sk_pg * self.F
            sub_v_start = sv_pg * self.F
            phys_k_subs = torch.arange(
                sub_k_start, sub_k_start + need_size,
                dtype=torch.long, device=self.device,
            )
            phys_v_subs = torch.arange(
                sub_v_start, sub_v_start + need_size,
                dtype=torch.long, device=self.device,
            )
            return (phys_k_subs, sub_k_start, sub_k_start + need_size,
                    phys_v_subs, sub_v_start, sub_v_start + need_size)

    # ------------------------------------------------------------------
    # Free
    # ------------------------------------------------------------------
    # Note: free() is inherited from UnifiedMemoryAllocator and is for
    # PAGE-level frees (activation, adapter, embedding). KV freers MUST
    # call free_kv() — sub-slot ids in [0, tot_size) collide with page
    # ids and cannot be safely disambiguated by range.

    def free_kv(self, sub_ids):
        if not isinstance(sub_ids, torch.Tensor):
            sub_ids = torch.as_tensor(
                sub_ids, dtype=torch.long, device=self.device
            )
        if sub_ids.numel() == 0:
            return
        if self.F == 1:
            # Sub-slot id == page id; delegate to parent's page-level free
            # which also maintains the counter. The parent's free() is
            # overridden below to honor the debug toggle, so we don't
            # need a separate print branch here.
            return super().free(sub_ids)

        with self.page_table_lock:
            page_ids = sub_ids // self.F

            # bincount folds three things into one kernel:
            #   - per-page count of sub-slots being freed (handles
            #     duplicate page_ids correctly via accumulation),
            #   - the "touched pages" mask (decrements > 0),
            #   - source values for the refcount decrement.
            # Output size is fixed at tot_size, so no host sync (unlike
            # torch.unique). int64 dtype matches kv_refcount, no cast.
            decrements = torch.bincount(page_ids, minlength=self.tot_size)
            self.kv_refcount -= decrements

            # A page transitions KV → FREE iff it was touched in this
            # batch AND its refcount is now 0. Pure tensor algebra —
            # no boolean-index sync.
            empty_mask = (decrements > 0) & (self.kv_refcount == 0)

            # Masked scatter writes are no-ops when the mask is empty
            # (typical during steady-state decode); the kernels still
            # launch but write zero positions.
            self.page_type_map[empty_mask] = int(PageType.FREE.value)
            self.free_bitmap |= empty_mask

            # One sync to keep _used_pages coherent on the CPU side.
            # Reuse the result for the optional debug print so we don't
            # duplicate the .item() sync when tracing is on.
            n_freed_pages = int(empty_mask.sum().item())
            self._used_pages -= n_freed_pages

            if _DEBUG_FREE:
                self._debug_print_free(
                    kind="free_kv",
                    freed_pages=n_freed_pages,
                    extra=f"subslots={int(sub_ids.numel())}",
                )

    def free(self, phys_ids):
        """Page-level free for non-KV page types (adapter / activation /
        embedding). Overrides the parent only to optionally trace; the
        actual bookkeeping stays in UnifiedMemoryAllocator.free()."""
        if isinstance(phys_ids, torch.Tensor):
            n = int(phys_ids.numel())
        else:
            n = len(phys_ids)
        super().free(phys_ids)
        if _DEBUG_FREE:
            self._debug_print_free(kind="free", freed_pages=n)

    # Short labels for the per-type breakdown in the free trace.
    # FREE is reported separately as `free=…`, so it's excluded here.
    _DEBUG_TYPE_LABELS = {
        PageType.KV_CACHE: "KV",
        PageType.ADAPTER_WEIGHT: "ADP",
        PageType.ATTENTION_INPUT_ACTIVATION: "ATT",
        PageType.FFN_INPUT_ACTIVATION: "FFN",
        PageType.EMBEDDING: "EMB",
    }

    def _debug_print_free(self, *, kind: str, freed_pages: int,
                          extra: str = "") -> None:
        """One-line trace per free call. Off-path is guarded at the
        call site so this method is never entered when _DEBUG_FREE is
        False. Adds a per-PageType breakdown of `used` so a creeping
        leak shows which page-type is hogging pages (e.g. KV stuck
        despite frees vs. ADP staying constant). One GPU sync per
        call (bincount + .tolist()) — acceptable in the debug path."""
        used = self._used_pages
        free = self.tot_size - used
        # Per-type histogram of page_type_map. minlength keeps the
        # bincount output well-formed even when some PageTypes have
        # zero pages assigned.
        type_counts = torch.bincount(
            self.page_type_map.to(torch.long),
            minlength=max(pt.value for pt in PageType) + 1,
        ).tolist()
        by_type = " ".join(
            f"{label}={type_counts[pt.value]}"
            for pt, label in self._DEBUG_TYPE_LABELS.items()
        )
        suffix = f" {extra}" if extra else ""
        print(
            f"[mem_free] {kind}: freed={freed_pages} pages → "
            f"used={used} [{by_type}] free={free} (tot={self.tot_size}){suffix}"
        )
