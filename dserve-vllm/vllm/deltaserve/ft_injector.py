# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Builds finetuning-sample Requests for injection into the scheduler.

Each FT sample becomes an ordinary vLLM ``Request`` with ``max_tokens=1`` (so it
does one prefill, allocates minimal KV, and is retired the same step), tagged
``is_finetuning=True`` and routed to the dedicated FT LoRA adapter. The samples
come from the length-bucketed ``FinetuningStore``; we greedily pack samples up
to a per-step token budget, then mark them trained so the next step draws fresh
samples (cycling epochs).
"""

from collections.abc import Callable

from vllm.deltaserve import dprint
from vllm.deltaserve.finetuning_store import FinetuningStore
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request

# Reserved LoRA id for the FT adapter — high enough to never collide with
# inference adapter ids (launch_deltaserve.py assigns 1, 2, ...).
FT_LORA_INT_ID = 1000


class FinetuneInjector:
    """Owns the FinetuningStore + FT LoRARequest; emits FT Requests per step."""

    def __init__(
        self,
        finetune_config,
        tokenize: Callable[[str], list[int]],
        block_hasher=None,
    ) -> None:
        self.cfg = finetune_config
        self.block_hasher = block_hasher
        self.store = FinetuningStore(
            data_path=finetune_config.data_path,
            tokenize=tokenize,
            adapter=finetune_config.finetuning_lora_path,
            total_epochs=finetune_config.num_epochs,
            max_saved_finetuning_tokens=finetune_config.max_saved_finetuning_tokens,
            max_prepare=finetune_config.max_prepare,
        )
        self.store.load()

        self.ft_lora: LoRARequest | None = None
        if finetune_config.finetuning_lora_path:
            self.ft_lora = LoRARequest(
                lora_name="__finetune__",
                lora_int_id=FT_LORA_INT_ID,
                lora_path=finetune_config.finetuning_lora_path,
            )
        self._counter = 0

    def next_ft_requests(self, token_budget: int) -> list[Request]:
        """Greedily pack FT samples up to `token_budget` tokens into Requests.
        Claims the chosen samples in the store (removing them from the
        selectable pool); the scheduler is responsible for
        ``commit_claimed`` after the backward finishes, or
        ``release_claimed`` on rollback. Each emitted Request carries its
        source sample at ``req._ft_sample`` so downstream code can route
        commit/release without a secondary lookup table."""
        if token_budget <= 0 or self.store.data_path is None:
            return []
        popped = []
        remaining = token_budget
        while remaining > 0:
            if not self.store.has_next() and not self.store.advance_epoch():
                break
            sample = self.store.pop_best_under(remaining, exclude=popped)
            if sample is None:
                break
            popped.append(sample)
            remaining -= sample.input_len

        if not popped:
            return []
        # Claim samples: remove from selectable pool but defer trained=True
        # until the backward acks via commit_claimed.
        self.store.claim(popped)
        return [self._make_request(s) for s in popped]

    def _make_request(self, sample) -> Request:
        self._counter += 1
        req_id = f"__ft__{self._counter}"
        req = Request(
            request_id=req_id,
            prompt_token_ids=list(sample.prompt_ids),
            sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
            pooling_params=None,
            lora_request=self.ft_lora,
            block_hasher=self.block_hasher,
            # [DeltaServe] Unique per-request cache salt so this FT prompt can
            # NEVER get a prefix-cache hit — not against sibling FT samples that
            # share the corpus template prefix (e.g. "### Instruction:"), nor
            # against an earlier-epoch repeat of the same sample. A cache hit
            # would skip re-forwarding the cached prefix tokens, so their
            # activations would never be saved and the backward would train on a
            # truncated sample (only the uncached suffix). The unique salt makes
            # every block-hash unique → the FULL prompt is always forwarded →
            # all per-token activations are captured. No effect when prefix
            # caching is off; inference requests are unaffected.
            cache_salt=req_id,
        )
        req.is_finetuning = True
        # Source sample reference for the store's commit/release routing.
        req._ft_sample = sample
        return req
