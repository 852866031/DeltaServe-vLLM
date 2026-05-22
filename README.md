# DeltaServe-vLLM

Re-hosting DeltaServe's LLM **co-serving** layer on top of vLLM's V1 engine:
interleave a LoRA-SFT backward pass with ongoing inference on the same GPU
(activation capture → backward subprocess → SLO-aware FT-admission scheduler),
keeping vLLM's production multi-LoRA batching + multi-process engine as the
substrate.

## Layout
- `vllm/` — our coding dir: a vendored fork of `vllm-project/vllm`
  (base commit `117afeea4`, ~v0.21.1rc0). DeltaServe code lives in
  `vllm/vllm/deltaserve/`; upstream edits are small and tagged `[DeltaServe]`.
- `DeltaServe/` — read-only reference (the original co-serving framework).
- `configs/`, `scripts/`, `eval/`, `adapters/`, `tests/` — integration assets.

## Read first
- `CLAUDE.md` — architecture, box mapping, build/precision rules, current status.
- `INTEGRATION_PROGRESS.md` — phased plan + per-stage progress.
- `VLLM_FORK_CHANGES.md` — every change vs upstream vLLM (navigate the fork).
- `vllm_setup_5090.md` — reproducible build/run on the RTX 5090.

## Note on a fresh clone
Compiled kernels (`*.so`) and model weights (`*.safetensors`, tokenizer blobs)
are **gitignored** — rebuild/reinstall vLLM (`VLLM_USE_PRECOMPILED=1 …`, see
`vllm_setup_5090.md`) and regenerate/download the toy LoRA adapters before
running the eval.
