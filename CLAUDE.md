# DeltaServe-vLLM — Project Context

## What this project is

We are **re-hosting DeltaServe's LLM co-serving layer on top of vLLM's V1 engine**.
DeltaServe interleaves a LoRA SFT **backward pass** with ongoing **inference** on
the same GPU: it injects finetuning samples into inference batches, captures their
activations during the forward, hands them to a **backward process** that trains a
dedicated LoRA adapter, and an **SLO-aware scheduler** decides each step how much
finetuning work to admit without blowing inference TTFT/latency.

The bet: vLLM already gives us the two hardest pieces for free — (1) a production
multi-LoRA batching pipeline (punica/S-LoRA kernels, adapter pool) and (2) a
multi-process engine with a real scheduler + continuous batching. So this is **not**
"port DeltaServe's inference engine to vLLM." It is "keep DeltaServe's *co-serving
value-add* (activation capture, backward process, SLO-aware FT-admission scheduler +
estimator) and re-host it on vLLM's inference + LoRA + scheduling substrate."

## Directory layout

```
DeltaServe-vLLM/                    ← this repo root (where we write integration code & this file)
├── DeltaServe/                     ← READ-ONLY reference. The original co-serving framework.
│   └── CLAUDE.md                   ← authoritative DeltaServe architecture doc (read it)
├── vllm/                           ← OUR CODING DIR. Cloned vLLM, verified to run on this 5090.
│   └── AGENTS.md                   ← upstream vLLM contribution rules (mostly N/A — we're a research fork)
├── configs/                        ← serving_config_finetuning_{opt,llama3}.yaml (DeltaServe-style YAML)
├── scripts/                        ← entry points: ft_experiment_{opt,llama3}.py, launch_deltaserve.py, …
├── adapters/                       ← toy LoRA adapters (opt125m / llama3; inference + "-ft" FT target)
├── INTEGRATION_PROGRESS.md         ← plan + per-stage progress. Source of truth for *what* to build & *how far*.
├── VLLM_FORK_CHANGES.md            ← every change vs upstream vLLM (navigate the fork)
└── vllm_setup_5090.md              ← reproducible build/run setup for the RTX 5090
```

- **`DeltaServe/` is read-only** — reference for ideas and implementation details.
  Never edit it; use it to verify how a mechanism worked in the original.
- **`vllm/` is where we write integration code.** It is a normal git checkout of vLLM
  (not yet a divergent fork — see version note below).

## Current status

**Phases 1–4 code-complete & GPU-validated end-to-end on the 5090 (Phases 1–3 gradcheck-verified).**
The real co-serving training loop works on Llama-3-8B; opt-125m is a frozen loss-only reference path.
The `_maybe_pause` GPU-yielding contract is wired (prefill-gated, fire-and-forget).

**Key Phase-4 finding (read this):** the inference-TTFT spikes that closed FT admission were **not**
co-serving / GPU / backward contention — they were a **vLLM frontend** stall. vLLM attaches per-step
`scheduler_stats` to the rank-0 API frontend's output stream, saturating that one asyncio loop and
stalling HTTP accept + SSE streaming. Fix: **`disable_log_stats` now auto-defaults ON whenever
`enable_finetuning`** (`engine/arg_utils.py:create_engine_config`); the SLO estimator uses its own
engine-side timing, so it's unaffected. Frontend output processing can also be sharded with
`--api-server-count N` (1 shared EngineCore + N frontends). **Current focus: optimizing inference
E2E latency under co-serving — the SLO gate is over-admitting FT.** Lever:
`finetune.ft_tokens_admission_constrain_factor` (cap FT tokens ≤ `prefill_tokens · factor` per step;
`-1` disables).

Two living docs track the detail:

- **`INTEGRATION_PROGRESS.md`** — plan + per-stage progress + how each step was verified.
- **`VLLM_FORK_CHANGES.md`** — every change vs upstream vLLM (new files + edits),
  with "what it does" / "used by". **Read this to navigate the fork.**

All net-new code lives in **`vllm/vllm/deltaserve/`** (logging, config loader, backward
process, per-model backward services in `bwd_services/`, finetuning store, FT injector, FT
scheduler, activation accumulator, coordinator). Upstream edits are small and tagged `[DeltaServe]`.

What works today (single-GPU, verified on the 5090):
- `--enable-finetuning` (a `FinetuneConfig` sub-config); a per-model backward subprocess
  (`bwd_services/`, selected by HF arch) spawned from the Worker with child-only MPS env;
  base + FT-adapter weights shared zero-copy via CUDA IPC.
- FT samples injected into real inference batches as `max_tokens=1` prefill-only requests
  routed to a dedicated FT LoRA adapter, marked with a `finetune_mask`, run **eager**, retired
  same-step, invisible to the frontend; real inference output unaffected. (Async scheduling is
  now **enabled** for co-serving via reserve-at-inject buffer accounting — see Phase 4b.)
- Per-token FT activations **accumulated** into shared GPU buffers: residual-stream layer inputs
  (`layer_in[i]`), `final_in`, `final_hidden`, `concat_input_ids`, and the MLP pre-activations
  (`mlp_gate_up[i]`, to skip the gate_up recompute in the backward).
- **Real LoRA SFT backward** (`Llama3BackwardService`): manual fp32 per-layer backward
  (re-materialized from the saved layer inputs), AdamW + StepLR on the fp32 master, then the
  trained weights are **published into vLLM's served LoRA buffers** so inference uses them. The
  backward fires on buffer-full **or** epoch-end. Gradchecked vs autograd to ~1e-7.
- A co-serving **coordinator**: signals the backward when the activation buffer is full, can't grow
  (the **peek-next** smallest-untrained sample won't fit the free space → `note_injection` raises the
  flush flag), or at an epoch boundary; reopens admission when done. Per-step FT budget is the full
  buffer capacity (was `0.5·capacity`), so an idle step fills the buffer in one eager forward.
  Admission is **SLO-aware** (Phase 4) and gated off until a POST `/start_finetuning` (so profiling +
  warmup run FT-free).
- **SLO-aware FT admission + execution-time estimator** (Phase 4): a merged 6-param step-time
  model (`deltaserve/estimator.py`), seeded by an offline profiling pass at launch and refit
  online every 256 steps, gates how many FT tokens to admit so inference TTFT/TBT stays within
  configured SLOs. The backward yields the GPU to prefill via `_maybe_pause`.
- **`scripts/ft_experiment_{llama3,opt}.py`** — launch a real `vllm serve` HTTP server with
  finetuning and fire periodic prompts so co-serving decisions stream live. **`eval/`** —
  `auto_benchmark.py` replays request timelines (`eval/timelines/5090/`) against a co-serving
  server and `auto_plot.py` renders the 4-panel latency/throughput/SLO figure.
  (Configs: `configs/serving_config_finetuning_{llama3,opt}.yaml`.)

Next: **optimize inference-request E2E latency under co-serving** — the SLO gate currently
over-admits FT, so co-serving inflates inference E2E more than the SLOs intend. Tune the admission
budgets / estimator and use `ft_tokens_admission_constrain_factor`. Then **Phase 5** — backward CUDA
graphs + attention batching. Known open issues: FT loss divergence in the loose-co eval run
(training-quality, not the SLO gate); avg-TBT admission gate deferred.

> Historical note: the original Phase-1 plan called the activation save "capture"; it
> was renamed **accumulate** to avoid confusion with CUDA-graph capture.

## vLLM V1 architecture (the substrate we integrate against)

```
 entrypoints/openai/api_server.py   ── HTTP
      │  ZMQ
 v1/engine/async_llm.py (AsyncLLM)  ── frontend, API process
      │
 v1/engine/core.py (EngineCoreProc) ── separate process: scheduler + executor driver
      │   Scheduler.schedule() → SchedulerOutput
 v1/core/sched/scheduler.py         ── one token-budget batch per step (unified prefill+decode)
      │
 v1/executor → v1/worker/gpu_worker.py (Worker)  ── separate process(es): own the GPU
      │
 v1/worker/gpu_model_runner.py (GPUModelRunner)  ── input prep + forward + sampling
```

Insertion points for DeltaServe pieces:
- **Backward process** spawns from the **Worker** (GPU-owning process) — the analogue
  of DeltaServe's `model_rpc.py`. Mental substitution: `model_rpc.py → gpu_worker.py`.
- **Activation capture** hooks the model forward inside `GPUModelRunner`.
- **FT injection + `finetune_mask`** splits across the **`Scheduler`** (admission decision)
  and **`GPUModelRunner` input prep** (lay out tokens, build mask, route to FT adapter).
- **SLO scheduler + estimator** wrap/subclass **`Scheduler`** (the estimator is net-new;
  vLLM has no execution-time predictor).

### Hard invariants / mismatches to design around (plan §2)

1. **Any batch containing FT tokens runs eager.** Capturing side-effecting activation
   copies inside a piecewise CUDA graph reintroduces the pool-aliasing NaN trap. This is
   the same gate DeltaServe enforces at `lora_unordered_batch_mixed.py:171-177` (`not has_ft`).
2. **FT samples are prefill-only.** They go through the forward once to produce
   activations, then the backward process consumes them — they must never enter the decode
   loop, hold KV past the step, or emit sampler output. Single-step-prefill-then-retire.
3. **Last-token-only logits.** V1 only materializes logits for sampled positions.
   *Recommended choice:* save FT **hidden states** (pre-LM-head) to the shared buffer and
   run the LM head inside the backward process — keeps the forward's extra work to a memcpy.
4. **Cross-process GPU tensor sharing under `spawn`** needs explicit CUDA IPC (torch.mp
   reductions), not DeltaServe's fork-style reference passing. (Phase 1 risk above.)

## Version & build reality on THIS machine

- **Plan recommended `v0.15.1`; the actual checkout in `vllm/` is `v0.21.1rc0` (123 commits
  past the tag).** Treat `vllm/` as the working baseline — it was verified to run on the
  5090. File paths in the plan are for the ~v0.10–0.15 line and **will drift**; trust the
  live tree over the plan's path references.
- **Two model runners exist in this version.** `GPUModelRunner` v1 (`vllm/v1/worker/gpu_model_runner.py`)
  vs v2 (`vllm/v1/worker/gpu/model_runner.py`), selected by `VllmConfig.use_v2_model_runner`.
  **v2 is default ONLY for `Qwen3ForCausalLM`** (`DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES`,
  `vllm/config/vllm.py:68`). We target **Llama-3 → v1 model runner**, which matches the plan.
  Don't accidentally patch the v2 runner.
- **Build mode: Python-only, precompiled kernels.** Nothing we add touches vLLM's C++/CUDA
  (forward LoRA = vLLM's precompiled punica kernels; backward = hand-rolled PyTorch + CUDA
  graphs). Editable install via `VLLM_USE_PRECOMPILED=1`.
- **RTX 5090 = Blackwell sm_120.** vLLM's precompiled kernels cover sm_120 (no full source
  build needed). FlashInfer JIT-compiles sm_120 kernels on first run (one-time). See
  `vllm_setup_5090.md` for the exact CUDA-13.0-in-conda-env recipe — the env (`dserve-vllm`)
  must own its `nvcc ≥ 12.9` or FlashInfer fails with a misleading "requires sm75" error.
- **Setup uses conda** (`dserve-vllm` env), per `vllm_setup_5090.md`. vLLM's `AGENTS.md`
  prescribes `uv`/`.venv` and strict upstream-PR rules — those are for upstream contributors;
  we are a research fork, so follow `vllm_setup_5090.md` for env management.
- Models live in `HF_HOME=/mnt/storage/huggingface` (set `HF_HUB_OFFLINE=1`):
  `meta-llama/Meta-Llama-3-8B` (base), `-Instruct`, plus LoRA adapters for the multi-LoRA path.

## DeltaServe → vLLM box mapping (the value-add we port)

| DeltaServe | Disposition on vLLM |
|---|---|
| Multi-LoRA inference batching, adapter pool (`naive_infer_adapter.py`, `lora_unordered_batch_mixed.py` LoRA half) | **Replaced** by vLLM native multi-LoRA (`vllm/lora/`) |
| Base model + KV memory manager (`unified_mem_allocator.py`, `packed_kv_mem_allocator.py`) | **Replaced** by vLLM model + paged KV |
| Two-process launch + GPU buffer share + pause event (`model_rpc.py`) | **Ported** → `gpu_worker.py` (Phase 1) |
| Finetuning sample store (`router/finetuning_store.py`) | **Ported** ~as-is (pure Python) — Phase 2 |
| FT injection + `finetune_mask` | **Ported** into `Scheduler` + `GPUModelRunner` input prep — Phase 2 |
| Activation capture (`lora_unordered_batch_mixed.py:394-421`) | **Re-implemented** as forward hooks / arch subclass — Phase 2 |
| Backward SFT service (`models/{llama,llama3}/SFT_service*.py`) | **Ported** ~verbatim into backward process (framework-agnostic PyTorch) — Phase 3 |
| SLO scheduler + 3-regime estimator + tracker (`mixed_req_queue.py`, `tracker.py`) | **Ported** as a layer on V1 `Scheduler` — Phase 4 |
| Allocators / occupancy / packed-KV | **Dropped** initially (vLLM owns KV); revisit only for FT activation pool |

## Phased plan summary (see `INTEGRATION_PROGRESS.md` for per-phase detail + status)

Each phase ends in an independently testable state; don't start a phase until the prior
phase's test passes.

1. **Phase 1 — backward process + shared-memory IPC** (no backward logic). ← *current focus.*
   Deliverable: shared-buffer hash round-trips cross-process; MPS vars child-only; clean shutdown.
2. **Phase 2 — activation capture + FT injection + dedicated FT adapter.** FT samples flow
   through real batches via a dedicated LoRA adapter; their (and only their) activations land
   in shared buffers; cross-process hash matches; inference correctness for real requests
   is unaffected. (No backward yet.)
3. **Phase 3 — real backward pass.** Backward process trains the FT adapter from captured
   activations; pause/resume wired to the `mp.Event`; FT loss decreases; inference still serves;
   no NaNs. Working co-serving with *fixed* admission.
4. **Phase 4 — SLO-aware scheduler + estimator.** Replace fixed FT injection with DeltaServe's
   SLO-aware admission gate + cost estimator. SLO satisfaction near target at non-trivial FT
   throughput; admission backs off under inference bursts.
5. **Phase 5 (later) — optimizations & assets.** Backward CUDA graphs, dedicated FT activation
   pool if needed, eval/analysis tooling port, multi-TP correctness.

## Key DeltaServe co-serving contracts to preserve (from DeltaServe/CLAUDE.md)

- `_maybe_pause()` at **every layer boundary** in the backward path is how backward yields
  the GPU to inference. It is load-bearing.
- Backward runs on its own CUDA stream; time it with `torch.cuda.synchronize()` before
  reading the wall clock (else you measure host dispatch, not GPU completion).
- MPS partitioning is the mechanism for true concurrent execution (the env-var wrap above).
- fp32 LM head / final norm precision rule; fp32 `scores` matmul for GQA attention backward
  (downgrading to fp16 caused llama3 loss to plateau). Relevant in Phase 3.
- CUDA-graph pool aliasing: persistent buffers (LoRA `.grad`, attention `ctx`) must live
  outside the graph pool. Relevant in Phase 3/5.

## Good first reads

1. This file.
2. `INTEGRATION_PROGRESS.md` — plan + progress: design invariants, runtime pipeline, per-phase status, next step, risks.
3. `DeltaServe/CLAUDE.md` — original architecture; co-serving contract; SFT backward.
4. `vllm_setup_5090.md` — how to build/run on this machine.
5. For Phase 1: `DeltaServe/dserve/server/router/model_infer/model_rpc.py:120-195`
   (spawn + MPS + buffer share) and `vllm/vllm/v1/worker/gpu_worker.py`
   (`Worker.__init__`, `init_device`, `load_model`) + `vllm/vllm/v1/engine/tensor_ipc.py`.
