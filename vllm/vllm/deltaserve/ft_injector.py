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
        """Greedily pack FT samples up to `token_budget` tokens into Requests."""
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
        # Mark trained so the next step draws different samples (cycles epochs).
        self.store.confirmed_trained(popped)
        return [self._make_request(s) for s in popped]

    def _make_request(self, sample) -> Request:
        self._counter += 1
        req = Request(
            request_id=f"__ft__{self._counter}",
            prompt_token_ids=list(sample.prompt_ids),
            sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
            pooling_params=None,
            lora_request=self.ft_lora,
            block_hasher=self.block_hasher,
        )
        req.is_finetuning = True
        return req
