#!/usr/bin/env python
"""Finetuning with the per-batch adapter limit full of inference adapters.

vLLM lets a batch hold ``max_loras`` distinct adapters and parks a request
that would exceed it in ``scheduler.skipped_waiting``. With one inference
adapter the FT adapter always fits; with adapter routing several inference
adapters share a batch, the limit fills, and FT requests get parked there.
The scheduler's cleanup of unscheduled FT requests must take them out of that
queue too — left behind, one is scheduled later with no entry in
``scheduler.requests`` and the engine dies (KeyError in
_update_after_schedule; found on Qwen3-14B TP=2 with routed requests).

Qwen3-0.6B, one GPU, max_loras=2, two inference adapters in flight together
while finetuning is open:

    CUDA_VISIBLE_DEVICES=0 python tests/test_ft_lora_limit.py
"""

import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_CONFIG = _REPO / "configs" / "serving_config_finetuning_qwen3_0.6b.yaml"
_PAIRS = 16
# Pair k: a request on adapter 0 at k*_PERIOD, one on adapter 1 _STAGGER later —
# its prefill is scheduled while the first is decoding (both adapters in the
# batch: the limit is full), then the engine idles so FT-only steps and
# backwards run and admission reopens before the next pair.
_PERIOD, _STAGGER = 1.2, 0.15
_ARRIVALS = sorted((k * _PERIOD + j * _STAGGER, j) for k in range(_PAIRS) for j in range(2))
_PROMPTS = [f"Question {i}: explain in a few words why the sky is blue."
            for i in range(len(_ARRIVALS))]


def main() -> None:
    os.environ.setdefault("HF_HOME", "/home/jiaxuan/.cache/huggingface")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import RequestOutputKind

    from deltaserve.config_loader import load_yaml_config, split_config

    raw = load_yaml_config(str(_CONFIG), verbose=False)
    engine_kwargs, ft_cfg, _ = split_config(raw)
    engine_kwargs["enforce_eager"] = True
    engine_kwargs["max_loras"] = 2
    infer = str(raw["adapters"]["lora_path_0"])
    ft = dict(enable_finetuning=True, start_on_launch=True,
              finetuning_lora_path=ft_cfg.finetuning_lora_path,
              data_path=ft_cfg.data_path, num_epochs=ft_cfg.num_epochs,
              max_saved_finetuning_tokens=ft_cfg.max_saved_finetuning_tokens)
    llm = LLM(**engine_kwargs, additional_config={"deltaserve": ft})
    eng = llm.llm_engine
    loras = [LoRARequest(f"infer-{i}", 11 + i, infer) for i in range(2)]
    sp = SamplingParams(temperature=0.0, max_tokens=48,
                        output_kind=RequestOutputKind.FINAL_ONLY)
    llm.generate(["Warm-up:"], SamplingParams(temperature=0.0, max_tokens=8))

    done, nxt, error = set(), 0, None
    t0 = time.monotonic()
    try:
        while len(done) < len(_PROMPTS) and time.monotonic() - t0 < 180:
            now = time.monotonic() - t0
            while nxt < len(_PROMPTS) and _ARRIVALS[nxt][0] <= now:
                eng.add_request(str(nxt), _PROMPTS[nxt], sp,
                                lora_request=loras[_ARRIVALS[nxt][1]])
                nxt += 1
            if eng.has_unfinished_requests():
                for out in eng.step():
                    if out.finished:
                        done.add(out.request_id)
            else:
                time.sleep(0.002)
    except Exception as e:  # noqa: BLE001 — the failure this test is about
        error = f"{type(e).__name__}: {e}"

    passed = failed = 0

    def check(name, cond):
        nonlocal passed, failed
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        passed, failed = passed + bool(cond), failed + (not cond)

    print("=== finetuning with the per-batch adapter limit full ===")
    check(f"the engine kept stepping (error: {error})", error is None)
    check(f"every inference request finished ({len(done)}/{len(_PROMPTS)})",
          len(done) == len(_PROMPTS))
    print(f"\n{passed} passed, {failed} failed")
    sys.stdout.flush()
    try:
        eng.engine_core.shutdown()   # stops the engine core and the backward child
    except Exception:  # noqa: BLE001 — a dead engine is already reported above
        pass
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
