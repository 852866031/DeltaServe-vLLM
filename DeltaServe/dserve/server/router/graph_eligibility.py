"""
Manager-side mirror of the runner's captured CUDA-graph bucket sets.

Why this exists
---------------
The CUDA graph runner lives in the model_rpc subprocess. The manager (router)
process makes scheduling decisions and calls into the prefill/decode estimators
to predict execution time. Predictions need to know which regime the upcoming
batch will run in:

  * graph regime — bucket is in the runner's cache, replay is fast and ~constant
  * eager regime — bucket isn't cached (fall back to full forward), slower and
    quadratic-ish in tokens

This class is a pure Python mirror that:
  • is seeded from `model_rpc.get_all_captured_buckets()` once after offline
    profiling completes;
  • is updated by the manager on each scheduler iteration via
    `model_rpc.pop_pending_captures()`;
  • answers `will_*_use_graph(...)` predicates for the scheduler.

Mirror staleness is bounded by the scheduler iteration cadence (~µs to ms).
The first batch of a never-before-seen shape is mirror-MISS → predicted as
eager (pessimistic-safe). After that batch's capture is reported back, the
mirror updates and subsequent batches at that shape are predicted as graph.

Co-serving invariant
--------------------
The runtime hard-disables the prefill graph for any batch with FT tokens
(`lora_unordered_batch_mixed.py`, gate `not has_ft`). So
`will_prefill_use_graph(has_ft=True, ...)` always returns False, regardless
of the bucket state — encoded directly here so callers don't have to remember.

Bucket-rounding logic mirrors the runner's static helpers exactly. We import
them rather than duplicate so any change to bucketing in the runner stays in
sync automatically.
"""

from typing import Set, Tuple

from dserve.common.cuda_graph_runner import CudaGraphRunner


class GraphEligibility:
    def __init__(
        self,
        *,
        decode_enabled: bool,
        prefill_enabled: bool,
    ) -> None:
        self.decode_enabled = decode_enabled
        self.prefill_enabled = prefill_enabled
        self._decode_buckets: Set[Tuple[int, int]] = set()
        self._prefill_buckets: Set[Tuple[int, int]] = set()

    # ─── Mirror updates ────────────────────────────────────────────────
    def seed(self, captured: dict) -> None:
        """Seed from `model_rpc.get_all_captured_buckets()` output."""
        for k in captured.get("decode", []) or []:
            self._decode_buckets.add(tuple(k))
        for k in captured.get("prefill", []) or []:
            self._prefill_buckets.add(tuple(k))

    def note_pending(self, captured: dict) -> int:
        """Apply the output of `model_rpc.pop_pending_captures()`. Returns the
        number of new buckets added (for logging / debug)."""
        n_new = 0
        for k in captured.get("decode", []) or []:
            t = tuple(k)
            if t not in self._decode_buckets:
                self._decode_buckets.add(t)
                n_new += 1
        for k in captured.get("prefill", []) or []:
            t = tuple(k)
            if t not in self._prefill_buckets:
                self._prefill_buckets.add(t)
                n_new += 1
        return n_new

    # ─── Predicates ────────────────────────────────────────────────────
    def will_prefill_use_graph(
        self, has_ft: bool, batch_size: int, total_tokens: int
    ) -> bool:
        """True iff a prefill batch with this shape would replay a captured
        graph. Co-serving (has_ft=True) is unconditionally eager."""
        if has_ft:
            return False
        if not self.prefill_enabled:
            return False
        if batch_size > CudaGraphRunner.PREFILL_BS_BUCKETS[-1]:
            return False  # bs above the largest bucket → runner falls back to eager
        key = (
            CudaGraphRunner.get_prefill_bs_bucket(batch_size),
            CudaGraphRunner.get_prefill_token_bucket(total_tokens),
        )
        return key in self._prefill_buckets

    def will_prefill_capture_on_hit(
        self, has_ft: bool, batch_size: int, total_tokens: int
    ) -> bool:
        """True iff a prefill batch with this shape would trigger a
        first-touch CUDA-graph *capture* at runtime (the slow path that
        runs warmup forward + capture forward + replay).

        A bucket is captureable iff (a) prefill graph capture is enabled,
        (b) the batch is inference-only — co-serve is hard-gated to
        eager, (c) the batch fits within the runner's bs bucketing
        scheme. If those hold and the bucket is *not* currently in the
        captured set, the next hit will trigger capture.
        """
        if has_ft:
            return False
        if not self.prefill_enabled:
            return False
        if batch_size > CudaGraphRunner.PREFILL_BS_BUCKETS[-1]:
            return False
        key = (
            CudaGraphRunner.get_prefill_bs_bucket(batch_size),
            CudaGraphRunner.get_prefill_token_bucket(total_tokens),
        )
        return key not in self._prefill_buckets

    def will_decode_use_graph(self, batch_size: int, max_len_in_batch: int) -> bool:
        """True iff a decode step with this shape would replay a captured graph.
        Decode key uses exact batch_size (not bucketed) — mirrors the runner."""
        if not self.decode_enabled:
            return False
        key = (batch_size, CudaGraphRunner.get_max_len_bucket(max_len_in_batch))
        return key in self._decode_buckets

    def will_decode_capture_on_hit(self, batch_size: int, max_len_in_batch: int) -> bool:
        """True iff a decode step with this shape would trigger a
        first-touch capture. Decode has no batch-size cap (the runner
        keys on exact bs), so any uncaptured bucket is captureable when
        decode graphs are enabled."""
        if not self.decode_enabled:
            return False
        key = (batch_size, CudaGraphRunner.get_max_len_bucket(max_len_in_batch))
        return key not in self._decode_buckets

    # ─── Introspection (logging / debug) ───────────────────────────────
    def num_decode_buckets(self) -> int:
        return len(self._decode_buckets)

    def num_prefill_buckets(self) -> int:
        return len(self._prefill_buckets)

    def summary(self) -> str:
        return (
            f"GraphEligibility(decode={self.num_decode_buckets()} buckets, "
            f"prefill={self.num_prefill_buckets()} buckets, "
            f"decode_enabled={self.decode_enabled}, "
            f"prefill_enabled={self.prefill_enabled})"
        )
