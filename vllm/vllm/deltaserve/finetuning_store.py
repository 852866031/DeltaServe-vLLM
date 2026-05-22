# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Finetuning sample store — port of DeltaServe's FinetuningManager (data parts).

Loads + tokenizes a finetuning corpus at startup (one sample per non-empty
line) and serves samples by length-bucketed selection:

  - ``pop_best_under(max_tokens)`` returns the *untrained* sample with the
    largest ``input_len <= max_tokens`` (does NOT mark it trained), so a step
    packs the biggest sample that fits its remaining FT token budget.
  - ``confirmed_trained(samples)`` marks samples trained for the current epoch.
  - ``advance_epoch()`` resets marks for another pass.

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

    def confirmed_trained(self, samples: list[FinetuningSample]) -> int:
        """Mark samples trained this epoch and drop them from the buckets."""
        by_len: dict[int, set] = {}
        for sample in samples:
            idx = self.id2idx.get(sample.request_id)
            if idx is None or self.trained[idx]:
                continue
            self.trained[idx] = True
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

    def advance_epoch(self) -> bool:
        """Start the next epoch (reset marks). False if no epochs remain."""
        if self.current_epoch >= self.total_epochs:
            return False
        self.current_epoch += 1
        self._reset_epoch_structures()
        return True

    def has_next(self) -> bool:
        return bool(self.len_buckets)

    def __len__(self) -> int:
        return len(self.samples)
