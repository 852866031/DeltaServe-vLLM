import math
import random
import string
import uuid
from typing import List, Optional, Tuple
import numpy as np

from ..io_struct import Batch, Req
from ..tokenizer import get_tokenizer
from ..input_params import FinetuneParams
from ..sampling_params import SamplingParams

# ---------------------- Helpers ----------------------

def _infer_sampling_params(max_new_tokens: int) -> SamplingParams:
    return SamplingParams(
        do_sample=False,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        ignore_eos=True,
        max_new_tokens=max_new_tokens,
        stop_sequences=[],
    )


def _ft_sampling_params() -> SamplingParams:
    return SamplingParams(
        do_sample=False,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        ignore_eos=False,
        max_new_tokens=1,
        stop_sequences=[],
    )


def _random_sentence(num_words: int) -> str:
    words = ["".join(random.choices(string.ascii_lowercase, k=random.randint(3, 8))) for _ in range(max(1, num_words))]
    return " ".join(words).capitalize() + "."


# ---------------------- Main Class ----------------------
class ProfilingBatchGenerator:
    """
    Generates dummy profiling batches whose shapes are derived from runtime
    budgets rather than hand-picked. The downstream prefill/decode estimators
    in tracker.py fit:

        T_prefill ≈ α·Σnᵢ² + β·Σnᵢ + γ·T_ft + c
        T_decode  ≈ δ·B + ε·K + d

    To identify all four prefill coefficients we need diversity along three
    axes: total inference tokens, request decomposition (how Σnᵢ² differs
    from Σnᵢ for the same total), and FT-token fraction. This generator
    produces:

      * `warmup_batches` — discarded, run before stats collection so the
        first real samples are not contaminated by kernel/JIT warmup.
      * `inference_batches` — geometric token sweep + decomposition variants
        for fitting α, β, c (and feeding decode samples via the
        post-prefill decode step).
      * `coserving_batches` — (inf, ft) grid with FT-dominant and
        full-batch edge cases for fitting γ.

    Budget caps respected:
      * inf_total          ≤ INF_CAP = min(batch_max_tokens,
                                            max_total_token_num // 2)
      * inf_total + n_ft   ≤ batch_max_tokens
      * n_ft               ≤ FT_CAP   = min(memory.max_finetuning_tokens,
                                            finetune.max_saved_finetuning_tokens)

    `unified_mem_manager_max_size_gb` is informational; it backs
    `max_total_token_num`, which is the binding constraint here.
    """

    def __init__(
        self,
        finetune_params: FinetuneParams,
        inference_adapter_dir: str,
        *,
        batch_max_tokens: int,
        max_finetuning_tokens: int,
        max_saved_finetuning_tokens: int,
        max_total_token_num: int,
        max_req_total_len: int,
        unified_mem_manager_max_size_gb: float,
        num_repeats: int = 2,
        inf_req_lens: Optional[List[int]] = None,
        model_weightdir: Optional[str] = None,
        tokenizor_mode: Optional[str] = None,
        trust_remote_code: bool = False,
        max_new_tokens_infer: int = 1,
        rng_seed: int = 42,
    ) -> None:
        self.ft_params = finetune_params
        self.inference_adapter_dir = inference_adapter_dir
        self.max_new_tokens_infer = max_new_tokens_infer
        self.rng = random.Random(rng_seed)
        random.seed(rng_seed)
        np.random.seed(rng_seed)

        try:
            self.tokenizer = get_tokenizer(
                model_weightdir or finetune_params.model_weightdir,
                tokenizor_mode or finetune_params.tokenizor_mode,
                trust_remote_code=trust_remote_code or finetune_params.trust_remote_code,
            )
        except Exception:
            self.tokenizer = get_tokenizer("huggyllama/llama-7b", tokenizor_mode or "auto")

        # Budget caps
        self.batch_max_tokens = int(batch_max_tokens)
        self.FT_CAP = max(1, min(int(max_finetuning_tokens), int(max_saved_finetuning_tokens)))
        self.INF_CAP = max(64, min(int(batch_max_tokens), int(max_total_token_num) // 2))
        # Per-request hard cap. The runtime allocates b_loc[:, max_req_total_len + pad]
        # per request; exceeding this crashes init_bloc. Reserve a 16-tok margin.
        self.MAX_REQ_LEN = max(64, int(max_req_total_len) - 16)
        self.unified_mem_manager_max_size_gb = unified_mem_manager_max_size_gb
        self.num_repeats = max(1, int(num_repeats))

        # Test override: when set, replaces the principled sweep+decomposition
        # logic for inference batches with a flat list — each entry produces a
        # single-request inference batch of that exact prompt length. Coserve
        # batches are unaffected. Lengths violating MAX_REQ_LEN or
        # batch_max_tokens are skipped with a warning.
        self._inf_req_lens_override: Optional[List[int]] = (
            list(inf_req_lens) if inf_req_lens is not None else None
        )

        self.warmup_batches: List[Batch] = []
        self.inference_batches: List[Batch] = []
        self.coserving_batches: List[Batch] = []

        # Pre-tokenized pool used to slice exact-length prompts
        self._token_pool: List[int] = []
        self._pool_cursor: int = 0
        self._POOL_MIN_LEN = max(10000, 4 * self.batch_max_tokens)

    # ---------------------- Public ----------------------
    def prepare(self) -> None:
        """Populate warmup_batches, inference_batches, coserving_batches.

        For each unique shape, the *first* run is placed in `warmup_batches`
        (the manager runs these without recording stats) so that CUDA graph
        capture happens during warmup. The following `num_repeats` runs go
        to the recorded lists, where they hit the now-warm graph cache and
        measure replay timing — which is what live serving will also see.

        Without this split, the offline fit blends one capture-cost sample
        with one replay sample per shape, biasing predictions toward eager
        and making the scheduler over-conservative until the first online
        refit (and over-aggressive afterwards, when the fit suddenly shifts
        to pure replay).
        """
        self._ensure_token_pool()

        co_pairs = self._coserve_pairs()

        print(
            f"[ProfilingBatchGenerator] caps: batch_max={self.batch_max_tokens}, "
            f"INF_CAP={self.INF_CAP}, FT_CAP={self.FT_CAP}, "
            f"MAX_REQ_LEN={self.MAX_REQ_LEN}, "
            f"mem_pool={self.unified_mem_manager_max_size_gb}GB, "
            f"num_repeats={self.num_repeats} "
            f"(each shape: 1 capture-priming pass in warmup + {self.num_repeats} recorded passes)"
        )

        # Generic kernel warmup — primes cuBLAS / mempool / autotune caches
        # before any graph-capture pass runs.
        self.warmup_batches.append(self._build_inference_batch_exact(256, n_reqs=2))
        if self.FT_CAP >= 8 and self.batch_max_tokens >= 128 + 8:
            ft_warm = max(8, self.FT_CAP // 4)
            self.warmup_batches.append(self._build_coserve_batch_exact(128, ft_warm))

        # Inference profiling
        if self._inf_req_lens_override is not None:
            valid_lens: List[int] = []
            for L in self._inf_req_lens_override:
                if L < 1:
                    print(f"  [override] skipping L={L} (< 1)")
                    continue
                if L > self.MAX_REQ_LEN:
                    print(f"  [override] skipping L={L} (> MAX_REQ_LEN={self.MAX_REQ_LEN})")
                    continue
                if L > self.batch_max_tokens:
                    print(f"  [override] skipping L={L} (> batch_max_tokens={self.batch_max_tokens})")
                    continue
                valid_lens.append(L)
            print(f"  [override] inf_req_lens active: {valid_lens} "
                  f"(skipping sweep + decomposition; one 1-req batch per length)")
            # Capture-priming pass → warmup (no stats)
            for L in valid_lens:
                self.warmup_batches.append(self._build_inference_batch_exact(L, n_reqs=1))
            # Recorded passes → stats
            for _ in range(self.num_repeats):
                for L in valid_lens:
                    self.inference_batches.append(self._build_inference_batch_exact(L, n_reqs=1))
        else:
            inf_targets = self._inference_token_targets()
            inf_decomp = self._inference_decomposition_variants()
            #print(f"  inference token sweep: {inf_targets}")
            #print(f"  inference decomposition variants (total, n_reqs): {inf_decomp}")
            # Capture-priming pass → warmup (no stats)
            for total in inf_targets:
                self.warmup_batches.append(self._build_inference_batch_exact(total))
            for total, n_reqs in inf_decomp:
                self.warmup_batches.append(self._build_inference_batch_exact(total, n_reqs=n_reqs))
            # Recorded passes → stats
            for _ in range(self.num_repeats):
                for total in inf_targets:
                    self.inference_batches.append(self._build_inference_batch_exact(total))
                for total, n_reqs in inf_decomp:
                    self.inference_batches.append(self._build_inference_batch_exact(total, n_reqs=n_reqs))

        print(f"  coserving (inf, ft) pairs: {co_pairs}")
        # Coserving: capture-priming pass → warmup (no stats), then recorded passes
        for n_inf, n_ft in co_pairs:
            self.warmup_batches.append(self._build_coserve_batch_exact(n_inf, n_ft))
        for _ in range(self.num_repeats):
            for n_inf, n_ft in co_pairs:
                self.coserving_batches.append(self._build_coserve_batch_exact(n_inf, n_ft))

        n_warm = len(self.warmup_batches)
        n_inf = len(self.inference_batches)
        n_co = len(self.coserving_batches)
        print(f"  total batches: {n_warm} warmup + {n_inf} inference + {n_co} coserve = {n_warm + n_inf + n_co}")

    # ---------------------- Target generation ----------------------
    def _inference_token_targets(self) -> List[int]:
        """Geometric sweep of total inference tokens, capped at INF_CAP."""
        cap = self.INF_CAP
        base = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
        targets = {t for t in base if t <= cap}
        # Useful fractions of the cap — catches the upper end on small configs
        for frac in (0.5, 0.75, 0.9, 1.0):
            targets.add(max(64, int(cap * frac)))
        return sorted(targets)

    def _inference_decomposition_variants(self) -> List[Tuple[int, int]]:
        """
        (total, n_reqs) pairs at two pivots, three decompositions each.
        Without this axis Σnᵢ² is collinear with Σnᵢ in the lstsq matrix
        and α (the quadratic term) is poorly identified.

        Per-request cap (MAX_REQ_LEN) is respected: the "big and few"
        variant uses the smallest n_reqs that keeps each piece ≤ cap,
        not n_reqs=1.
        """
        cap = self.INF_CAP
        pivots: List[int] = []
        for frac in (0.25, 0.6):
            t = int(cap * frac)
            if 256 <= t <= cap:
                pivots.append(t)
        if not pivots:
            pivots = [min(512, cap)]

        out: List[Tuple[int, int]] = []
        seen = set()
        for total in pivots:
            n_min = max(1, math.ceil(total / self.MAX_REQ_LEN))
            # 'big and few' (= n_min); 'medium' (4× the floor); 'many small' (~64-tok pieces)
            candidates = [n_min, max(n_min, 4), max(n_min, total // 64)]
            for n_reqs in candidates:
                n_reqs = max(1, min(n_reqs, total))
                key = (total, n_reqs)
                if key in seen:
                    continue
                seen.add(key)
                out.append(key)
        return out

    def _coserve_pairs(self) -> List[Tuple[int, int]]:
        """
        (n_inf, n_ft) grid + edge cases. Constraints:
          n_inf + n_ft ≤ batch_max_tokens
          n_ft         ≤ FT_CAP
        """
        cap_total = self.batch_max_tokens
        ft_cap = self.FT_CAP

        ft_levels = sorted({
            max(8, ft_cap // 8),
            max(16, ft_cap // 4),
            max(32, ft_cap // 2),
            max(48, int(ft_cap * 0.75)),
            ft_cap,
        })
        inf_anchor = max(256, int(cap_total * 0.6))
        inf_levels = sorted({128, 512, 1024, 2048, inf_anchor})

        pairs: List[Tuple[int, int]] = []
        seen = set()

        def _add(p: Tuple[int, int]) -> None:
            if p[0] >= 1 and p[1] >= 1 and p[0] + p[1] <= cap_total and p not in seen:
                seen.add(p)
                pairs.append(p)

        for ft in ft_levels:
            for inf in inf_levels:
                _add((inf, ft))

        # FT-dominant: tiny inf with full FT cap (forces γ to absorb FT cost)
        _add((32, ft_cap))
        # Full-batch: maximize inf alongside full FT cap
        inf_full = cap_total - ft_cap
        if inf_full >= 64:
            _add((inf_full, ft_cap))

        return pairs

    # ---------------------- Pool & Slicing ----------------------
    def _ensure_token_pool(self) -> None:
        if len(self._token_pool) >= self._POOL_MIN_LEN:
            return
        pool: List[int] = []
        while len(pool) < self._POOL_MIN_LEN:
            sent = _random_sentence(self.rng.randint(8, 20))
            ids = self.tokenizer(sent).get("input_ids", [])
            if not ids:
                continue
            pool.extend(ids)
        self._token_pool = pool
        self._pool_cursor = 0

    def _take_slice(self, length: int) -> List[int]:
        """Return a contiguous slice of exactly `length` token IDs from the pool (wraps if needed)."""
        assert length > 0
        pool = self._token_pool
        n = len(pool)
        if self._pool_cursor + length <= n:
            sl = pool[self._pool_cursor : self._pool_cursor + length]
            self._pool_cursor += length
            return sl
        part1 = pool[self._pool_cursor :]
        needed = length - len(part1)
        part2 = pool[:needed]
        self._pool_cursor = needed
        return part1 + part2

    # ---------------------- Builders ----------------------
    def _safe_n_reqs(self, total: int, n_reqs: int) -> int:
        """Bump n_reqs upward so no single piece exceeds MAX_REQ_LEN."""
        if total <= 0:
            return 0
        n_min = max(1, math.ceil(total / self.MAX_REQ_LEN))
        return max(n_reqs, n_min)

    def _build_inference_batch_exact(self, total_tokens: int, n_reqs: Optional[int] = None) -> Batch:
        if n_reqs is None:
            # Default: ~128 tokens/req, capped at 8 reqs
            n_reqs = max(1, min(8, total_tokens // 128))
        n_reqs = self._safe_n_reqs(total_tokens, max(1, min(n_reqs, total_tokens)))
        lengths = self._exact_partition(total_tokens, n_reqs)
        reqs: List[Req] = [
            self._new_infer_req_from_ids(self._take_slice(L), max_new_tokens=2)
            for L in lengths
        ]
        return Batch(uuid.uuid4().hex, reqs)

    def _build_coserve_batch_exact(self, n_inf: int, n_ft: int) -> Batch:
        n_inf_reqs = max(1, min(8, n_inf // 128)) if n_inf > 0 else 0
        n_ft_reqs = max(1, min(4, n_ft // 64)) if n_ft > 0 else 0
        n_inf_reqs = self._safe_n_reqs(n_inf, n_inf_reqs)
        n_ft_reqs = self._safe_n_reqs(n_ft, n_ft_reqs)
        inf_lengths = self._exact_partition(n_inf, n_inf_reqs) if n_inf_reqs else []
        ft_lengths = self._exact_partition(n_ft, n_ft_reqs) if n_ft_reqs else []

        reqs: List[Req] = []
        for L in inf_lengths:
            reqs.append(self._new_infer_req_from_ids(self._take_slice(L)))
        for L in ft_lengths:
            reqs.append(self._new_ft_req_from_ids(self._take_slice(L)))

        infer_tokens = sum(r.input_len for r in reqs if not r.is_finetuning)
        ft_tokens = sum(r.input_len for r in reqs if r.is_finetuning)
        assert infer_tokens == n_inf and ft_tokens == n_ft, (
            f"Exact totals mismatch: inf={infer_tokens}!={n_inf} or ft={ft_tokens}!={n_ft}"
        )
        return Batch(uuid.uuid4().hex, reqs)

    # ---------------------- Partitioning ----------------------
    def _exact_partition(self, total: int, n_parts: int) -> List[int]:
        """Split `total` into `n_parts` positive integers that sum exactly to `total`."""
        if n_parts <= 1:
            return [total]
        base = total // n_parts
        rem = total % n_parts
        parts = [base + 1] * rem + [base] * (n_parts - rem)
        self.rng.shuffle(parts)
        return parts

    # ---------------------- Request Factories ----------------------
    def _new_infer_req_from_ids(self, ids: List[int], max_new_tokens=None) -> Req:
        try:
            text = self.tokenizer.decode(ids)
        except Exception:
            text = f"<synthetic {len(ids)} tok>"
        return Req(
            adapter_dir=self.inference_adapter_dir,
            request_id=uuid.uuid4().hex,
            prompt_ids=ids,
            sample_params=_infer_sampling_params(self.max_new_tokens_infer if max_new_tokens is None else max_new_tokens),
            is_finetuning=False,
            needs_to_notify_detokenize=True,
            text=text,
        )

    def _new_ft_req_from_ids(self, ids: List[int]) -> Req:
        try:
            text = self.tokenizer.decode(ids)
        except Exception:
            text = f"<synthetic {len(ids)} tok>"
        return Req(
            adapter_dir=self.ft_params.finetuning_lora_path,
            request_id=uuid.uuid4().hex,
            prompt_ids=ids,
            sample_params=_ft_sampling_params(),
            is_finetuning=True,
            needs_to_notify_detokenize=False,
            text=text,
        )


# ---------------------- Optional: pretty summary ----------------------
def summarize_batches(batches: List[Batch]) -> List[Tuple[str, int, int]]:
    out = []
    for b in batches:
        total = sum(r.input_len for r in b.reqs)
        out.append((b.batch_id, total, len(b.reqs)))
    return out
