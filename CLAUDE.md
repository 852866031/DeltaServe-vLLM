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
├── dserve-vllm/                    ← OUR CODING DIR + the published package source (name="dserve-vllm").
│   ├── pyproject.toml              ← distribution name & `dserve-vllm` console script
│   ├── vllm/                       ← inner Python package (still `import vllm`)
│   └── AGENTS.md                   ← upstream vLLM contribution rules (mostly N/A — we're a research fork)
├── configs/                        ← serving_config_finetuning_{opt,llama3}.yaml (DeltaServe-style YAML)
├── scripts/                        ← entry points: ft_experiment_{opt,llama3}.py, launch_deltaserve.py, …
├── adapters/                       ← toy LoRA adapters (opt125m / llama3; inference + "-ft" FT target)
├── INTEGRATION_PROGRESS.md         ← plan + per-stage progress. Source of truth for *what* to build & *how far*.
├── VLLM_FORK_CHANGES.md            ← every change vs upstream vLLM (navigate the fork)
└── README.md                       ← install guide (general + full RTX 5090 section)
```

- **`DeltaServe/` is read-only** — reference for ideas and implementation details.
  Never edit it; use it to verify how a mechanism worked in the original.
- **`dserve-vllm/` is where we write integration code.** It is a vendored fork of vLLM;
  the distribution name + CLI are `dserve-vllm`, the Python import name is still `vllm`.

## Current status

**Phases 1–4, 5.2, 5.3, 5.4, 5.5, 6, 6.1 all code-complete (Phases 1–3
gradcheck-verified, Phases 5.2 + 5.4 + 5.5 graph/eager parity-verified —
111/111 + 12/12 in `tests/test_llama3_backward{,_graph}.py`); Phase 6 GPU
validation of the pre-emption pipeline is the user's next run.** The real
co-serving training loop works on Llama-3-8B; opt-125m is a frozen loss-only
reference path. The `_maybe_pause` GPU-yielding contract is wired
(prefill-gated, fire-and-forget). The `forward_interruptible` pre-emption
pipeline defaults to OFF — when off, behaviour is bit-identical to
pre-Phase-6. The unified-phase scheduler (`coserving_admission_phase: both`)
also defaults to OFF — when off, behaviour is bit-identical to today's
prefill-only FT admission.

**Key Phase-4 finding (read this):** the inference-TTFT spikes that closed FT admission were **not**
co-serving / GPU / backward contention — they were a **vLLM frontend** stall. vLLM attaches per-step
`scheduler_stats` to the rank-0 API frontend's output stream, saturating that one asyncio loop and
stalling HTTP accept + SSE streaming. Fix: **`disable_log_stats` now auto-defaults ON whenever
`enable_finetuning`** (`engine/arg_utils.py:create_engine_config`); the SLO estimator uses its own
engine-side timing, so it's unaffected. Frontend output processing can also be sharded with
`--api-server-count N` (1 shared EngineCore + N frontends).

**Phase 6 (`forward_interruptible`) — three-tier inference pre-emption of FT-only stepping.**
Behind one config flag (`finetune.forward_interruptible`, default `False` → bit-identical
to today when off). When on, late-arriving inference requests pre-empt FT-only stepping at
three windows: **(A)** pre-schedule grace poll on `input_queue` (default 2 ms via
`ft_only_admission_grace_ms`); **(B)** post-schedule rollback when an arrival lands between
`schedule()` and `execute_model()` (releases KV + claimed samples + reservation, re-schedules
once); **(C)** mid-forward abort — per-layer hooks in `accumulate.py` raise `FTAborted` when
the input-socket thread sets `coord.ft_abort_event`, runner zeros the partial-write tail
and returns a sentinel, engine rolls back. Also introduces the 3-phase store API
(`claim` / `commit_claimed` / `release_claimed`), which fixes the pre-existing bookkeeping
flaw of marking samples `trained=True` at admit time before any backward processed them.

**Phase 6.1 — slice-based FT activation save.** Per-layer hooks gather FT rows via
`val[start:start+n]` (a view) instead of `val[mask]` (an `index_select` kernel) on the
fast path; mask gather kept as silent fallback when the FT-True positions are interleaved
(which can happen when an FT request fills a freed inference slot mid-batch).

**Phase 5.2 — backward CUDA graphs. ✅ code + parity verified.** Per-layer FFN-backward
graph + ONE shared padded-attention backward graph (the core has no per-layer weights, so
32 captures would be wasted), pre-captured up-front in
`Llama3GraphedBackward.prepare()` against zero-initialized static buffers. First real
backward sees only replay cost. Gated by `finetune.backward_cuda_graph` (default OFF).
Bounds: `backward_cuda_graph_attn_{bn_max,l_max}`. Silent eager fallback per-layer on
capture or shape-fit failure. `_maybe_pause()` is preserved at the once-per-layer
cadence — relocated to between Graph A and Graph B (mp.Event.wait can't run inside a
captured region). Gradcheck + graph parity in `tests/test_llama3_backward{,_graph}.py`
(12 + 21 cases).

**Phase 5.4 — forward-recompute CUDA graph (per layer). ✅ code + parity verified.** The
per-layer forward rematerialization inside `process_backward` (RMSNorm in_ln + Q/K/V proj
+ RoPE + padded attention + O proj + residual) is now also captured as a CUDA graph — one
per layer — extending the Phase 5.2 backend from 2 captured regions to **3 per layer**
(forward + FFN-bwd + attn-bwd) under the same `finetune.backward_cuda_graph` flag.
The forward graph's output buffers ARE the FFN-bwd / padded-attn-bwd input buffers
(`static_resid_mid`, `static_gate`, `static_up`, `static_qh/kh/vh_pad`) — eliminates the
intermediate copy-in steps. Per-sample attention forward is replaced by a captureable
padded variant mirroring the existing padded-attn backward. Pre-capture cost goes from 33
→ 65 captures at child startup (~few hundred ms). Eager fallback when `_attn_fit=False`
or the layer's saved gate||up is absent. Gradcheck parity 111/111 in
`tests/test_llama3_backward_graph.py`.

**Phase 5.5 (a.k.a. F1) — save post-RoPE qh/kh/vh per layer. ✅ code + parity verified.**
Optional, opt-in via `finetune.save_attn_qkv: bool = False`. When ON, a
`forward_pre_hook` on each `self_attn.attn` module saves the FT rows of post-RoPE q, k, v
to new buffers in the accumulator (`attn_qh/kh/vh[i]`, +~96 MB at s_max=256). The
backward — both eager `layer_forward` AND the Phase 5.4 forward graph — short-circuits
Q/K/V projection + RoPE entirely, only recomputing RMSNorm in_ln (cheap, needed for Q/K/V
LoRA-A grad). Eliminates ~400-500 GFLOPs per backward (~5 ms on the 5090). Per-layer
silent eager fallback when `saved_gu` is absent for that layer. Parity verified vs the
recompute path in `test_layer_forward_saved_qkv_parity` + `test_forward_graph_saved_qkv_parity`.

**Unified-phase FT scheduling — opt-in via `slo.coserving_admission_phase: both`. ✅ code.**
The original FT-rides-prefill rule is now selectable: `"prefill"` (default — today's
behaviour) keeps the `decode_only → 0` short-circuit; `"both"` selects
`BothPhaseFinetuneScheduler` (`deltaserve/ft_scheduler_both.py`), which removes the
short-circuit so the SLO estimator decides every step composition (prefill, decode,
mixed, idle). New `slo.decode_only_ft_safety_margin: float = 0.7` scales `max_tbt_slo`
on decode-only steps (justification: eager penalty + estimator γ cold start). The
`ft_tokens_admission_constrain_factor` proportional cap is prefill-relative — when
`!= -1` AND `phase=="both"`, selection soft-falls to `"prefill"` with a startup warning.
`match_prefill_workload_factor` self-gates on `feats.t_in > 0` so its counter only
accumulates on prefill — long decode-only stretches naturally rate-limit FT once banked
credit is consumed. New scheduler config at `configs/serving_config_finetuning_llama3_both.yaml`;
new `auto_benchmark.py --scheduler {prefill,both}` flag picks the YAML and tags outputs
with `_phase_<phase>`; new `eval/auto_plot_schedulers.py` emits two A/B PNGs
(`both_vs_inf-only`, `both_vs_prefill`).

**Phase 5.3 — perf polish + admission strategies + bug fixes.** Several shipped:
- **`match_prefill_workload_factor: float`** — leaky-bucket admission
  strategy (`config/finetune.py`, default 0.0). Accumulates inference-prefill
  tokens seen but not yet "spent" on FT; admits ONE FT sample sized to the
  next-sample's `input_len` (capped by SLO budget) when
  `(counter + t_in) * factor >= next_sample.input_len`. Factor scales how
  much credit each prefill token earns: `1.0` ≡ the previous boolean-on
  behaviour, `>1` more aggressive, `<1` more conservative, `0` disables.
  Mutually exclusive with `ft_tokens_admission_constrain_factor`.
- **Oversized-sample drop fix** (`finetuning_store.py:load()`): samples with
  `input_len > max_saved_finetuning_tokens` are dropped at load with a warning.
  Previously they sat in the pool forever, deadlocking FT admission once
  fittable samples ran out (`has_next()=True` but `pop_best_under(cap)=None`,
  `advance_epoch` never fires). Surfaced by `pure_ft_bench.py` on alpaca_1000
  (3 samples >256-token cap).
- **Fused AdamW** + **persistent grad_qh/kh/vh buffers** in the backward — ~3-5
  ms / backward saved (8 LoRA tensors × 32 layers → one fused kernel; 96
  zero-fills/backward eliminated).
- **One-shot `set_corpus_meta` IPC** from scheduler → child after
  `FinetuningStore.load()` — replaces sending corpus size on every
  `notify_buffer_full`.
- **Per-cycle backward log** is now one line:
  `[backward] Xms (graph/eager) loss=… total_trained=… n=… epoch=… N/total tokens`
  (CUDA-event timing; 2 syncs/cycle → 1 via cleanup-sync drain).

**Eval tooling fixes.** Throughput-panel alignment: `auto_benchmark.py` writes
`bench_meta<suffix>.json` with the recording-phase wall-clock t=0, and `auto_plot.py`
anchors the FT (wall-clock) series to it. Backward-log timestamps are now ms-resolution.
**Output files now carry FT factor + scheduler phase:**
`_co_factor_<X>_phase_<Y>_<mode>.csv` (factor `off` = `-1` sentinel; phase = `prefill` |
`both`) so A/B runs across factors AND schedulers don't overwrite. **`auto_plot.py`**
has a 4-panel single-row layout (timeline / E2E vs time / throughput / TTFT
satisfaction) and a `--factor X` CLI arg that auto-detects the smallest factor present
in `output/` if not specified. **`auto_plot_schedulers.py`** emits two A/B PNGs
centered on `phase=both`: `both_vs_inf-only` (co-serving overhead vs no-co baseline) and
`both_vs_prefill` (head-to-head scheduler comparison). **`eval/pure_ft_bench.py`** is
the pure-FT (no inference traffic) benchmark — launches the server, POSTs
`/start_finetuning`, idles for `--duration`, trims + summarizes the bwd_log.

**Current focus:** GPU-validating the `forward_interruptible` pipeline + the unified-phase
scheduler on the existing `eval/auto_benchmark.py` replay (P99 TTFT outlier reduction +
FT throughput uplift are the headline metrics). Independent levers:
- `slo.coserving_admission_phase: both` (new) vs `prefill` (default) — A/B with
  `auto_benchmark.py --co --scheduler both` and the new `eval/auto_plot_schedulers.py`
  PNGs.
- `finetune.match_prefill_workload_factor` (P5.3 leaky-bucket admission, float) vs
  `ft_tokens_admission_constrain_factor` (proportional cap) — A/B with
  `auto_benchmark.py --co` and the `_factor_*` suffix.
- `finetune.backward_cuda_graph` (P5.2 + 5.4) — compare backward latency in the
  per-cycle log between on/off. 3 captured regions per layer now (forward + FFN-bwd +
  attn-bwd).
- `finetune.save_attn_qkv` (P5.5 / F1) — toggles the post-RoPE qh/kh/vh save (+~96 MB
  activation pool for ~5 ms backward speedup).

Two living docs track the detail:

- **`INTEGRATION_PROGRESS.md`** — plan + per-stage progress + how each step was verified.
- **`VLLM_FORK_CHANGES.md`** — every change vs upstream vLLM (new files + edits),
  with "what it does" / "used by". **Read this to navigate the fork.**

All net-new code lives in **`dserve-vllm/vllm/deltaserve/`** (logging, config loader, backward
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
  (`layer_in[i]`), `final_in`, `final_hidden`, `concat_input_ids`, the MLP pre-activations
  (`mlp_gate_up[i]`, to skip the gate_up recompute in the backward), and — when
  `finetune.save_attn_qkv: true` (P5.5) — post-RoPE q/k/v per layer
  (`attn_qh/kh/vh[i]`, to skip Q/K/V proj + RoPE recompute too).
- **Real LoRA SFT backward** (`Llama3BackwardService`): manual per-layer backward
  in bf16 (cdt) by default — re-materialized from the saved layer inputs — with fp32-strict
  attention scores / softmax / RMSNorm / LM head per the DeltaServe precision contract.
  **Fused AdamW** (one CUDA kernel for all 256 LoRA tensors) + StepLR on the fp32 master,
  then the trained weights are **published into vLLM's served LoRA buffers** so inference
  uses them (the served slot is reserved at startup; punica reads with scale=1.0 because
  we bake `α/r` into B at publish time). The backward fires on buffer-full **or**
  epoch-end. Gradchecked vs autograd to ~1e-7. Optional **CUDA-graph backward** (P5.2)
  cuts per-cycle dispatch ~5 ms on Llama-3-8B with `finetune.backward_cuda_graph: true`.
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
  server (output files now tagged `_factor_<X>_<mode>` so A/B runs across FT admission
  factors don't overwrite); `auto_plot.py` renders the **5-panel** latency / throughput /
  SLO / **E2E percentile (p99 highlighted)** figure and takes `--factor X` (auto-detects
  smallest if omitted); `pure_ft_bench.py` runs a pure-FT workload (no inference traffic)
  for isolated backward-throughput measurement. (Configs:
  `configs/serving_config_finetuning_{llama3,opt}.yaml`.)

Next: **GPU-validate `forward_interruptible` on the eval replay** (set
`finetune.forward_interruptible: true` in the YAML, re-run `eval/auto_benchmark.py --co`,
compare TTFT P50/P95/P99 vs the same replay with the feature off). Phase 5.2 (backward
CUDA graphs) has shipped; the **next backward-throughput lever is F1 — save post-RoPE
qh/kh/vh per layer** (mirror the existing `mlp_gate_up` save pattern). Documented as
future work in `INTEGRATION_PROGRESS.md` Phase 5 section; expected ~5 ms / backward win
for +99 MB at s_max=256 (best perf/MB ratio of the candidates).

Known open issues: FT loss divergence in the loose-co eval run (training-quality, not
the SLO gate); avg-TBT admission gate deferred; pre-existing minor leak in the
runner's `self.requests` (`CachedRequestState`) for FT requests — they're never added
to `finished_req_ids`, so the per-request state lingers (small, bounded by `num FT
requests ever`, not a correctness issue); dead-child deadlock surface (if the backward
subprocess crashes, `_claimed` stays non-empty forever → FT admission wedges silently
after a one-shot 5 s warning — documented in `.claude/plans/backward-review-issues.md`
as C7, deliberately deferred).

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
  `README.md` for the exact CUDA-13.0-in-conda-env recipe — the env (`dserve-vllm`)
  must own its `nvcc ≥ 12.9` or FlashInfer fails with a misleading "requires sm75" error.
- **Setup uses conda** (`dserve-vllm` env), per `README.md`. vLLM's `AGENTS.md`
  prescribes `uv`/`.venv` and strict upstream-PR rules — those are for upstream contributors;
  we are a research fork, so follow `README.md` for env management.
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
5. **Phase 5 — optimizations.**
   - **5.1 ✅** `_maybe_pause` GPU-yield contract (prefill-gated).
   - **5.2 ✅** Backward CUDA graphs: per-layer FFN-bwd + single shared
     padded-attn-bwd, pre-captured at child startup. Flag:
     `finetune.backward_cuda_graph`.
   - **5.3 ✅** Perf polish + admission strategies + bug fixes: fused AdamW,
     persistent grad buffers, sync coalescing, one-shot `set_corpus_meta` IPC,
     restructured per-cycle log, `match_prefill_workload_factor` leaky-bucket
     admission (float), oversized-sample drop at load (FT deadlock fix), eval-tooling
     additions (factor suffix, `pure_ft_bench.py`).
   - **5.4 ✅** Forward-recompute CUDA graph: per-layer capture of the layer
     forward (RMSNorm + Q/K/V/RoPE + padded attention + O proj + residual).
     Total = 3 captured regions per layer. Same `finetune.backward_cuda_graph`
     flag; forward graph outputs land directly in the existing FFN/attn-bwd
     input buffers (no copy-in). Eager fallback per-layer.
   - **5.5 ✅ (a.k.a. F1)** Save post-RoPE qh/kh/vh per layer
     (`finetune.save_attn_qkv: bool = False`, opt-in). +~96 MB activation pool
     for ~5 ms backward speedup (skips Q/K/V proj + RoPE recompute on
     Llama-3-8B at s_max=256). Forward graph automatically reads from the
     saved buffers when the mode is on.
   - **(future)** Dedicated FT activation pool if vLLM's allocator gets in
     the way; multi-TP correctness (backward per-rank); profiling-pass
     extension to cover `decode + FT` shapes (currently online-refit only).
6. **Phase 6 — inference pre-emption of FT-only stepping (`forward_interruptible`). ✅ code.**
   Three tiers (A: pre-schedule grace; B: post-schedule rollback; C: mid-forward abort via
   per-layer hooks) behind one config flag. Adds the 3-phase store API (`claim` /
   `commit_claimed` / `release_claimed`) and an `FTAborted` sentinel for the runner/engine
   to thread. Default off → bit-identical behaviour. Pending: GPU validation on the eval
   replay (target: P99 TTFT outlier reduction from ~80 ms toward ~30 ms).
7. **Phase 6.1 — slice-based FT activation save. ✅ code.** Per-layer hooks gather FT
   rows via a slice view instead of `index_select` on the fast path; mask gather is the
   silent fallback when FT positions are interleaved with non-FT.
8. **Unified-phase FT scheduler — opt-in via `slo.coserving_admission_phase`. ✅ code.**
   `"prefill"` (default) keeps today's FT-rides-prefill rule; `"both"` selects
   `BothPhaseFinetuneScheduler` (`deltaserve/ft_scheduler_both.py`) so FT can ride decode-
   only / mixed / idle steps under the SLO estimator. `slo.decode_only_ft_safety_margin`
   (default 0.7) tightens the TBT budget on decode-only to bound estimator-cold-start risk.
   Soft-fall to `"prefill"` when `ft_tokens_admission_constrain_factor != -1`.
6. **Phase 6 — inference pre-emption of FT-only stepping (`forward_interruptible`). ✅ code.**
   Three tiers (A: pre-schedule grace; B: post-schedule rollback; C: mid-forward abort via
   per-layer hooks) behind one config flag. Adds the 3-phase store API (`claim` /
   `commit_claimed` / `release_claimed`) and an `FTAborted` sentinel for the runner/engine
   to thread. Default off → bit-identical behaviour. Pending: GPU validation on the eval
   replay (target: P99 TTFT outlier reduction from ~80 ms toward ~30 ms).
7. **Phase 6.1 — slice-based FT activation save. ✅ code.** Per-layer hooks gather FT
   rows via a slice view instead of `index_select` on the fast path; mask gather is the
   silent fallback when FT positions are interleaved with non-FT.

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
4. `README.md` — how to build/run on this machine.
5. For Phase 1: `DeltaServe/dserve/server/router/model_infer/model_rpc.py:120-195`
   (spawn + MPS + buffer share) and `dserve-vllm/vllm/v1/worker/gpu_worker.py`
   (`Worker.__init__`, `init_device`, `load_model`) + `dserve-vllm/vllm/v1/engine/tensor_ipc.py`.
6. For Phase 6 (`forward_interruptible`): the plan file
   `.claude/plans/can-you-make-a-elegant-cherny.md` for the end-to-end design +
   verification path; in-tree the integration points are
   `dserve-vllm/vllm/deltaserve/coordinator.py` (`FTAborted`, abort event, snapshot/restore),
   `dserve-vllm/vllm/deltaserve/finetuning_store.py` (3-phase claim/commit/release API),
   `dserve-vllm/vllm/deltaserve/ft_scheduler.py:_rollback_ft_step`,
   `dserve-vllm/vllm/deltaserve/accumulate.py` (hook abort check + slice fast path),
   `dserve-vllm/vllm/v1/engine/core.py` (tiers A + B + sentinel routing), and
   `dserve-vllm/vllm/v1/worker/gpu_model_runner.py` (tier C abort wrap + entry-time check).
