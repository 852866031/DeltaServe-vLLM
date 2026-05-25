# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Finetuning sample store — port of DeltaServe's FinetuningManager (data parts).

Loads + tokenizes a finetuning corpus at startup (one sample per non-empty
line) and serves samples by length-bucketed selection:

  - ``pop_best_under(max_tokens)`` returns the *untrained* sample with the
    largest ``input_len <= max_tokens`` (does NOT mark it trained), so a step
    packs the biggest sample that fits its remaining FT token budget.
  - ``claim(samples)`` reserves samples for an in-flight FT step: removes
    them from the selectable pool but does NOT yet mark them trained.
    ``commit_claimed(samples)`` finalises (sets ``trained=True``) after the
    backward has actually trained on their activations.
    ``release_claimed(samples)`` is the rollback path — returns samples to
    the selectable pool (used when an FT-only step is pre-empted by a late
    inference arrival under ``forward_interruptible``).
  - ``advance_epoch()`` resets marks for another pass; will refuse while any
    sample is claimed-but-not-yet-committed, so an in-flight backward can't
    be silently orphaned across an epoch boundary.

This is pure Python (no GPU / vLLM coupling). The DeltaServe original lives at
``dserve/server/router/finetuning_store.py``; we drop the parts coupled to
DeltaServe's Req/Batch + the bwd-loss bookkeeping (those return in Phase 3).
"""

import uuid
from bisect import bisect_left, bisect_right
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable

from vllm.deltaserve import dprint


@dataclass
class FinetuningSample:
    """One tokenized finetuning sample (prefill-only; never decoded)."""

    request_id: str
    prompt_ids: list[int]
    text: str
    adapter: str | None = None

    @property
    def input_len(self) -> int:
        return len(self.prompt_ids)


class FinetuningStore:
    """Length-bucketed store of tokenized finetuning samples."""

    def __init__(
        self,
        data_path: str | None,
        tokenize: Callable[[str], list[int]],
        adapter: str | None = None,
        total_epochs: int = 1,
        max_saved_finetuning_tokens: int = 256,
        max_prepare: int | None = None,
    ) -> None:
        self.data_path = data_path
        self.tokenize = tokenize
        self.adapter = adapter
        self.total_epochs = int(total_epochs)
        self.max_saved_finetuning_tokens = int(max_saved_finetuning_tokens)
        self.max_prepare = max_prepare

        self.samples: list[FinetuningSample] = []
        self.id2idx: dict[str, int] = {}
        self.trained: list[bool] = []
        self.current_epoch = 0
        self.total_tokens_in_memory = 0

        # length -> deque of untrained sample indices (current epoch)
        self.len_buckets: dict[int, deque] = defaultdict(deque)
        self.sorted_lengths: list[int] = []
        # Indices claimed for an in-flight FT step / backward (not in buckets,
        # not yet trained). On rollback (release_claimed) they go back to the
        # buckets; on completion (commit_claimed) they get trained=True. Used
        # by forward_interruptible — keeps epoch boundaries honest by
        # blocking advance_epoch until all in-flight samples are reconciled.
        self._claimed: set[int] = set()
        # immutable templates rebuilt per epoch
        self._bucket_template: dict[int, tuple[int, ...]] = {}
        self._sorted_template: list[int] = []

    # -- loading -----------------------------------------------------------

    def load(self) -> int:
        """Read + tokenize the corpus. Returns the number of samples loaded."""
        if self.data_path is None:
            return 0
        loaded = 0
        with open(self.data_path, encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                prompt_ids = list(self.tokenize(text))
                sample = FinetuningSample(
                    request_id=uuid.uuid4().hex,
                    prompt_ids=prompt_ids,
                    text=text,
                    adapter=self.adapter,
                )
                idx = len(self.samples)
                self.samples.append(sample)
                self.id2idx[sample.request_id] = idx
                self.total_tokens_in_memory += sample.input_len
                loaded += 1
                if self.max_prepare is not None and loaded >= self.max_prepare:
                    break
        self._build_templates()
        self._reset_epoch_structures()
        dprint(
            f"[ft-store] loaded {loaded} samples from {self.data_path} | "
            f"{self.total_tokens_in_memory} tokens, "
            f"{len(self._sorted_template)} distinct lengths "
            f"(min={self._sorted_template[0] if self._sorted_template else 0}, "
            f"max={self._sorted_template[-1] if self._sorted_template else 0})"
        )
        return loaded

    def _build_templates(self) -> None:
        tmp: dict[int, list[int]] = defaultdict(list)
        for idx, sample in enumerate(self.samples):
            tmp[sample.input_len].append(idx)
        self._bucket_template = {length: tuple(idxs) for length, idxs in tmp.items()}
        self._sorted_template = sorted(self._bucket_template)

    def _reset_epoch_structures(self) -> None:
        self.trained = [False] * len(self.samples)
        self.len_buckets = {
            length: deque(idxs) for length, idxs in self._bucket_template.items()
        }
        self.sorted_lengths = list(self._sorted_template)
        # Defensive: advance_epoch refuses to fire with _claimed non-empty, so
        # this should already be (). Clearing anyway to keep state coherent
        # if the guard is ever bypassed (e.g. profiling reset paths).
        self._claimed = set()

    # -- selection ---------------------------------------------------------

    def pop_best_under(self, max_tokens: int,
                       exclude: list[FinetuningSample] | None = None
                       ) -> FinetuningSample | None:
        """Largest untrained sample with input_len <= max_tokens (peek, no mark)."""
        if not self.sorted_lengths:
            return None
        exclude_ids = {s.request_id for s in exclude} if exclude else set()
        pos = bisect_right(self.sorted_lengths, max_tokens) - 1
        while pos >= 0:
            length = self.sorted_lengths[pos]
            for idx in self.len_buckets.get(length, ()):  # type: ignore[arg-type]
                if self.trained[idx]:
                    continue
                if self.samples[idx].request_id in exclude_ids:
                    continue
                return self.samples[idx]
            pos -= 1
        return None

    def pop_next(self, exclude: list[FinetuningSample] | None = None
                 ) -> FinetuningSample | None:
        """Smallest untrained sample (ascending length; peek, no mark)."""
        if not self.sorted_lengths:
            return None
        exclude_ids = {s.request_id for s in exclude} if exclude else set()
        for length in self.sorted_lengths:
            dq = self.len_buckets.get(length)
            if not dq:
                continue
            for idx in dq:
                if self.trained[idx]:
                    continue
                if self.samples[idx].request_id in exclude_ids:
                    continue
                return self.samples[idx]
        return None

    # -- marking / epochs --------------------------------------------------

    def claim(self, samples: list[FinetuningSample]) -> int:
        """Reserve samples for an in-flight FT step: remove them from the
        selectable pool and track in ``_claimed``. Does NOT yet mark them
        trained — that waits for ``commit_claimed`` after the backward acks.
        Symmetric with ``release_claimed`` for rollback. Returns count
        actually claimed (skips already-claimed or already-trained samples).
        """
        by_len: dict[int, set] = {}
        for sample in samples:
            idx = self.id2idx.get(sample.request_id)
            if idx is None or self.trained[idx] or idx in self._claimed:
                continue
            self._claimed.add(idx)
            by_len.setdefault(self.samples[idx].input_len, set()).add(idx)

        marked = 0
        for length, to_remove in by_len.items():
            dq = self.len_buckets.get(length)
            if not dq:
                continue
            kept = [i for i in dq if i not in to_remove]
            if kept:
                self.len_buckets[length] = deque(kept)
            else:
                del self.len_buckets[length]
                p = bisect_left(self.sorted_lengths, length)
                if p < len(self.sorted_lengths) and self.sorted_lengths[p] == length:
                    del self.sorted_lengths[p]
            marked += len(to_remove)
        return marked

    def commit_claimed(self, samples: list[FinetuningSample]) -> int:
        """Finalise: mark claimed samples as trained for the current epoch
        and drop them from ``_claimed``. Called from the coordinator's
        backward-done hook, so a sample is only ``trained=True`` once the
        backward has actually processed its activations. Idempotent on
        already-committed samples; silently skips samples not in
        ``_claimed`` (e.g. spurious double-commit).
        """
        marked = 0
        for sample in samples:
            idx = self.id2idx.get(sample.request_id)
            if idx is None or idx not in self._claimed:
                continue
            self._claimed.discard(idx)
            self.trained[idx] = True
            marked += 1
        return marked

    def release_claimed(self, samples: list[FinetuningSample]) -> int:
        """Rollback: return claimed samples to the selectable pool. Used when
        an FT-only step is pre-empted by a late inference arrival and the FT
        scheduling has to be undone. ``trained`` stays False; the indices
        reappear in ``len_buckets`` / ``sorted_lengths``. Idempotent.
        """
        by_len: dict[int, list[int]] = {}
        for sample in samples:
            idx = self.id2idx.get(sample.request_id)
            if idx is None or idx not in self._claimed:
                continue
            self._claimed.discard(idx)
            by_len.setdefault(self.samples[idx].input_len, []).append(idx)
        n = 0
        for length, idxs in by_len.items():
            dq = self.len_buckets.get(length)
            if dq is None:
                self.len_buckets[length] = deque(idxs)
                p = bisect_left(self.sorted_lengths, length)
                if (p == len(self.sorted_lengths)
                        or self.sorted_lengths[p] != length):
                    self.sorted_lengths.insert(p, length)
            else:
                dq.extend(idxs)
            n += len(idxs)
        return n

    def advance_epoch(self) -> bool:
        """Start the next epoch (reset marks). False if no epochs remain OR
        if any sample is still claimed-but-not-yet-committed (we must not
        cross an epoch boundary with in-flight samples — they'd be silently
        orphaned). Caller retries on a later step once the backward
        finishes or a rollback releases the claimed samples."""
        if self.current_epoch >= self.total_epochs:
            return False
        if self._claimed:
            return False
        self.current_epoch += 1
        self._reset_epoch_structures()
        return True

    def has_next(self) -> bool:
        """True iff a sample can be selected right now (not counting claimed
        in-flight samples — those can't be re-admitted until released or
        committed). Combined with ``current_epoch < total_epochs`` and
        ``len(_claimed) > 0`` by callers to decide if FT has any work."""
        return bool(self.len_buckets)

    def has_claimed(self) -> bool:
        """True iff any sample is currently claimed in-flight (not yet
        committed or released). Used by the scheduler to decide whether to
        keep stepping the engine even with an empty selectable pool."""
        return bool(self._claimed)

    def __len__(self) -> int:
        return len(self.samples)
