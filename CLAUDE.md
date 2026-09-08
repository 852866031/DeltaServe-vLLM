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
├── configs/                        ← serving_config_finetuning_{opt,llama3,llama3_tp2}.yaml (DeltaServe-style YAML)
├── scripts/                        ← entry points: ft_experiment_{opt,llama3}.py, launch_deltaserve.py, …
├── eval/                           ← single-GPU eval: auto_benchmark.py, auto_plot*.py, pure_ft_bench.py
├── eval-tp/                        ← TP eval: launch_deltaserve.py, ft_bench_tp.py (per-TP logs, --kill-stale)
├── tests/                          ← standalone scripts (run with `python`, NOT pytest — not installed)
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

**Phases 1–4, 5.2, 5.3, 5.4, 5.5, 6, 6.1 all code-complete; Phase 7 (TP=2) M0–M4.2 GPU-validated + M5 (backward graphs under TP) 2-GPU-gated (Phases 1–3
gradcheck-verified, Phases 5.2 + 5.4 + 5.5 graph/eager parity-verified —
111/111 + 12/12 in `tests/test_llama3_backward{,_graph}.py`); Phase 6 GPU
validation of the pre-emption pipeline is the user's next run.** The real
co-serving training loop works on Llama-3-8B; opt-125m is a frozen loss-only
reference path. The `_maybe_pause` GPU-yielding contract is wired
(prefill-gated, fire-and-forget).

**Phase 8 — Qwen3 family + backward-service restructure. ✅ code + CPU gates + Qwen3-14B
TP=2 GPU-validated (trains and co-serves the loose / tight / nutanix timelines).** The
backward service is now split into a family-agnostic
stack (`bwd_services/common/`: ops / attention / ffn / head / tp / trainer / graph)
and one file per family (`bwd_services/llama3.py`, `bwd_services/qwen3.py`), selected by
`bwd_services/registry.py`. Qwen3 = Llama-3 + a per-head `q_norm`/`k_norm` between the
qkv projection and RoPE; its layer math backpropagates through that norm (so its
`save_attn_qkv` captures the PRE-norm q/k at the `q_norm`/`k_norm` inputs — exact, and
since 2026-09-08 on by default, matching Llama-3's level of saving), adds no TP collective,
and passes gradcheck 16/16, shard 42/42, real-gloo TP 10/10, an overfit run, and — with
its own `graph_forward_core` — CUDA-graph parity 92/92 (all three captured regions, like
Llama-3). Llama-3 stays bit-identical (all five gates re-pass). Two generic vLLM edits: finetuning now
lists itself as unsupported on the v2 model runner (so `Qwen3ForCausalLM` + finetuning
lands on the v1 runner our hooks live in), and the worker's served-LoRA publish gate +
`meta` dict went through `bwd_services.registry.is_trainer` / `deltaserve/ft_meta.py`.
**Also fixed a live bug found on the way:** under transformers 5.x the worker read
`rope_theta` from an attribute that no longer exists, so every backward remat used
theta=10000 (Llama-3: 5e5, Qwen3: 1e6) — see the open-issue paragraph below. Configs:
`configs/serving_config_finetuning_qwen3_{14b_tp2,0.6b}.yaml`; `eval-tp/*` take
`--family {llama3,qwen3-14b,qwen3-0.6b}`. The `forward_interruptible` pre-emption
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

**Phase 6 (`forward_interruptible`) — three-tier inference pre-emption of FT-only stepping.
Works under TP since 2026-09-08 (rank-symmetric tier C — see the Phase 7 invariants), and
`finetune.pause_until_prefill_done` (opt-in, ON in the TP YAMLs) keeps the backward child
paused until the prefill has actually completed on the GPU — without MPS the old
enqueue-time resume let the child time-slice against the prefill. Tight-trace result
(Qwen3-14B TP=2): burst-start TTFT 55–67 ms (inference-only 38–41; before: 40–458),
run-max TTFT 577 → 150 ms, 100 % satisfaction, FT −5 %.**
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
`GraphedBackward.prepare()` against zero-initialized static buffers. First real
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

**Phase 5.6 — save the post-attention residual per layer (`finetune.save_resid_mid`). ✅
code + parity verified (2026-09-08).** Opt-in, default off. A `forward_pre_hook` on each
`post_attention_layernorm` captures the FT rows of `o + residual` = `resid_mid` (the FFN
input) with the same fused add-norm idiom the `layer_in` hooks use. The backward — eager
`layer_forward` and the forward graph alike — then skips the O projection (base + LoRA
GEMM, ~8.6 GFLOPs/layer on Llama-3-8B) and the residual add; `ctx_flat` is still produced
for the O-proj backward. **Under TP this removes the forward remat's `o` all-reduce, the
only collective in the forward recompute: 7 → 6 per layer (Llama-3-8B: 224 → 192 per
cycle), because the saved value is the already-reduced full residual.** The graph runner
stages the saved residual straight into `static_resid_mid` (Graph A's input) and captures
the tail with the core again. With `save_attn_qkv` + `save_attn_ctx` + this, the per-layer
forward recompute is RMSNorm in_ln alone. Cost: one `[s_max, hidden]` buffer per layer
(+64 MB Llama-3-8B, +100 MB Qwen3-14B at s_max=256). ON in every shipped YAML. Gates:
`tests/test_accumulate_hooks.py` 49/49 (hook-level, fake vLLM-named decoder),
`test_forward_graph_saved_resid_mid_parity` (llama3) + `test_saved_resid_mid_forward_graph`
(qwen3), and both 2-GPU NCCL gates, which also count the collectives per layer (7 without,
6 with).

**Phase 5.7 — LM-head restructure. ✅ (2026-09-08).** The first kernel-level profile of the
backward (synthetic Qwen3-14B shape, 2 GPUs, graph + all saves, no inference) put the LM
head at ~63 ms of a ~168 ms cycle: per-sample GEMMs with ~31 rows and nine bf16→fp32
conversions of the whole head per cycle. `head_backward` now batches all predicting rows
and converts each vocab chunk once per pass — same math, same fp32 contract — cycle
168 → 127 ms. Remaining uncontended budget on that shape: NCCL 37 ms (240 reduces — the
M4.3 target), bf16 layer GEMMs ~22 ms, head ~18 ms, elementwise ~17 ms. Under 4 rps
inference without MPS the cycle roughly doubles from driver time-slicing; **no MPS
daemon runs on this box** (the env var is set on the child but inert — see the Phase 7
notes), so `backward_mps_percentage` currently does nothing.

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

**Phase 7 — tensor parallelism (TP=2). ✅ M0–M4.2 GPU-validated; M5 2-GPU-gated (live A/B pending); M4.3 open.**
Backward-per-rank: each rank trains its own shard in its own backward child, joined by a
dedicated NCCL group (7 all-reduces per layer). **Verified:** TP=2 trains equivalently to
TP=1 — cycle-for-cycle to ~0.01 loss on identical data; shard 23/23; real 2-process gloo
gradcheck 10/10; TP=1 gradcheck 12/12 unchanged. **M4.2** (estimator timing relay +
all-rank backward ack, incl. on idle steps) is GPU-validated on Qwen3-14B TP=2: the
profiling pass fits all three regimes from 146 relayed samples, admission is SLO-gated,
and the timeline replays (`eval-tp/auto_benchmark_tp.py`, `ttft_slo 0.4`) give loose /
tight / nutanix-600-800 TTFT satisfaction of 95.0 / 97.5 / 98.1 % at 467 / 206 / 239 FT
tok/s — full table in INTEGRATION_PROGRESS.md Phase 7. **M5 (2026-09-08):** backward
CUDA graphs are TP-safe — the forward graph is split at the o-proj output (`static_o`),
that all-reduce and the six backward-side reduces run eagerly between replays, no
collective is captured, and the force-eager guard is gone. Gated on the two 5090s with
real NCCL: `tests/test_tp_backward_graph_nccl.py` 1196/1196 (layer math, both families ×
fp32/bf16, incl. the overflow fallback) and `tests/test_tp_trainer_graph_nccl.py` 132/132
(the real trainer, 4 steps: graph == eager == tp=1), both also covering
`save_resid_mid` (P5.6: no forward collective at all — 6 per layer, counted). **Not
yet:** the live graph-vs-eager A/B on the real models.
→ Full detail, invariants, and run commands in **"Tensor parallelism (Phase 7)"** below.

**Current focus (next session):** the TP line is feature-complete for now — M5, M4.3,
`forward_interruptible` under TP, the pause fix and the head restructure all landed and
were validated on the Qwen3-14B timelines on 2026-09-08 (loose 98.3 % / nutanix 99.4 % /
tight 100 % TTFT satisfaction at 642 / 392 / 262 FT tok/s; the earlier Qwen3 baselines
were 95.0 / 98.1 %). Remaining, in priority order: the Llama-3 TP=2 replays with the same
stack; bringing up an MPS daemon (`nvidia-cuda-mps-control -d`) so `backward_mps_percentage`
actually partitions the SMs — the ~2× cycle inflation under load is driver time-slicing;
a `validate_estimator` TP-vs-tp1 residual check; the Llama-3 `rope_theta` DIAG re-check;
the Qwen3-0.6B single-GPU smoke. The ordered plan with gates is the "Next step" section of
INTEGRATION_PROGRESS.md.

**Open, and NOT a TP bug — likely cause found (Phase 8):** FT loss stalls ~4.3 after
~25 cycles where an earlier `pure_ft` run reached ~2.6 on the same samples, identically
at TP=1 and TP=2, with `DSERVE_TP_DIAG=1` showing the backward's forward-remat diverging
from vLLM's real forward. Phase 8 found that under the installed transformers 5.8.1 vLLM
normalizes RoPE into `hf_config.rope_parameters["rope_theta"]` and the worker's
`getattr(hf_config, "rope_theta", 10000.0)` silently returned **10000 instead of
500000**, so the remat's RoPE never matched the served model. Fixed generically in
`deltaserve/ft_meta.py:rope_theta_of`. **Not yet re-verified on GPU** — the first item
of the Phase 8 ladder is to re-run the DIAG + `eval/pure_ft_bench.py` and confirm the
remat error collapses and the 2.12 reference reproduces.

Single-GPU levers still pending GPU A/B: `forward_interruptible`,
`slo.coserving_admission_phase: both`, `finetune.match_prefill_workload_factor` vs
`ft_tokens_admission_constrain_factor`, `finetune.backward_cuda_graph`,
`finetune.save_attn_qkv`.

Two living docs track the detail:

- **`INTEGRATION_PROGRESS.md`** — plan + per-stage progress + how each step was verified.
- **`VLLM_FORK_CHANGES.md`** — every change vs upstream vLLM (new files + edits),
  with "what it does" / "used by". **Read this to navigate the fork.**

All net-new code lives in **`dserve-vllm/vllm/deltaserve/`** (logging, config loader, backward
process, per-model backward services in `bwd_services/`, `ft_meta.py` (the backward
`meta` dict + `rope_theta_of`), finetuning store, FT injector, FT scheduler, activation
accumulator, coordinator). Upstream edits are small and tagged `[DeltaServe]`.

```
deltaserve/bwd_services/
├── base.py        BackwardService: the recv/dispatch loop, _maybe_pause, service_main
├── registry.py    HF arch / alias → service class (lazy import); is_trainer(); get_family()
├── common/        family-agnostic LoRA-SFT trainer stack
│   ├── ops.py         rope_cos_sin / apply_rope / rope_backward / rmsnorm(+bwd) / proj(+bwd)
│   ├── attention.py   attn_forward_core / attn_backward_core (per-sample GQA, fp32 core)
│   ├── ffn.py         ffn_forward_tail / ffn_backward_core (SwiGLU, frozen)
│   ├── head.py        head_backward (final norm + chunked fp32 LM head)
│   ├── tp.py          lora_shard_slice / init_backward_tp_group / reduce_partial
│   ├── family.py      Family record: layer fns + frozen weight map + graph hooks + flags
│   ├── trainer.py     LoraSftTrainerService: _build_state, process_backward, publish
│   └── graph.py       GraphedBackward: forward / FFN-bwd / padded-attn-bwd CUDA graphs
├── llama3.py      layer_forward / layer_backward / graph_forward_core + LLAMA3 + service
├── qwen3.py       same with q/k-norm (+ QWEN3; all three graph regions; saved-qkv = pre-norm q/k)
└── opt.py         loss-only reference
```
A family file owns its layer composition in full (readability over de-duplication);
everything it calls is in `common/`. The trainer binds the family's functions once in
`_build_state` and calls them directly — no per-call dispatch.

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
  (`mlp_gate_up[i]`, to skip the gate_up recompute in the backward), and — opt-in —
  q/k/v per layer (`finetune.save_attn_qkv`, P5.5: skips the Q/K/V projection — post-RoPE
  on Llama-3; on Qwen3 the pre-norm q/k from the `q_norm`/`k_norm` inputs, with the
  elementwise norm + RoPE re-applied), the attention context (`finetune.save_attn_ctx`:
  skips the attention forward) and the
  post-attention residual (`finetune.save_resid_mid`, P5.6: skips the O projection and,
  under TP, the forward remat's only all-reduce). With all three, the backward's
  per-layer forward recompute is RMSNorm in_ln alone.
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
  `vllm/config/vllm.py:69`). All our hooks live in the **v1 runner**, so Phase 8 lists
  "DeltaServe co-serving finetuning" in `_get_v2_model_runner_unsupported_features`:
  any `enable_finetuning` run (Qwen3 included) is routed to v1 with a `warning_once`,
  and an explicit `VLLM_USE_V2_MODEL_RUNNER=1` fails loudly. Qwen3 *without* finetuning
  keeps v2. Don't accidentally patch the v2 runner.
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

1. **Phase 1 — backward process + shared-memory IPC** (no backward logic). ✅
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
   - **5.6 ✅** Save the post-attention residual per layer
     (`finetune.save_resid_mid: bool = False`, opt-in; ON in the shipped
     YAMLs). +64 MB on Llama-3-8B for the O-proj GEMM + residual add per
     layer — and under TP the forward remat's `o` all-reduce (7 → 6
     collectives per layer).
   - **(future)** Dedicated FT activation pool if vLLM's allocator gets in
     the way; profiling-pass extension to cover `decode + FT` shapes
     (currently online-refit only). *(multi-TP correctness — backward
     per-rank — has since landed as **Phase 7**.)*
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
9. **Phase 7 — tensor parallelism (TP=2). 🟡 M0–M4.2 GPU-validated; M5 2-GPU-gated.**
   Backward-per-rank on a dedicated NCCL group across the backward children; 7
   all-reduces per layer; lock-step guaranteed by broadcasting the trigger on
   `SchedulerOutput`. M1 non-daemon workers (+ `PR_SET_PDEATHSIG`), M2 shard-aware dims /
   `lora_shard_slice` / `lm_head` all-gather, M3 the surgical reduce set, M4.1 the
   scheduler↔worker relay, M4.2 the estimator timing relay + all-rank backward ack.
   `tp_size == 1` inert. M5 backward CUDA graphs under TP (forward graph split at the
   o-proj reduce, every collective eager; 2-GPU NCCL gated). **Open:** the M5 live A/B,
   M4.3 (gradient bucketing + comm stream + rank-symmetric clip).

## Tensor parallelism (Phase 7) — the essentials

Everything you need to work on TP without opening another file. Detail lives in
`INTEGRATION_PROGRESS.md` (Phase 7) and `VLLM_FORK_CHANGES.md` (stage P7).

**Principle — backward-per-rank.** Mirroring the original DeltaServe (`model_rpc.py`
inits NCCL per rank and builds per-rank backward services): each TP rank runs its **own**
backward child on its **own** weight shard. The only cross-rank traffic is a small set of
all-reduces on a **dedicated NCCL group across the backward children** — they are *not*
in vLLM's inference NCCL group. `tp_size == 1` leaves every path inert, so single-GPU
behaviour is bit-identical.

### Milestones + gates

| M | Scope | Gate | Status |
|---|---|---|---|
| M1 | Non-daemon workers so each rank can fork its backward child | tp=2 boots, child per rank, clean shutdown, **no orphans on crash** | ✅ GPU-verified |
| M2 | Shard-aware dims/weights/buffers; `lora_shard_slice`; `lm_head` all-gather | buffer + weight widths == local shard; no shape errors | ✅ 23/23 |
| M3 | Backward NCCL group + the gradient all-reduces | **summed shard grads ≈ single-GPU reference (~1e-6)** | ✅ 10/10 (real gloo) |
| M4.1 | Scheduler↔worker control-plane relay | real 2-GPU: loss ↓, no NaN, inference correct | ✅ trains == TP=1 |
| M4.2 | SLO estimator relay under TP + all-rank backward ack (incl. idle steps) | admission gated by SLO, not a fixed budget; ack relayed only when every rank's child is done | ✅ CPU 22/22 + GPU-validated (Qwen3-14B TP=2 timelines) |
| M4.3 | Gradient bucketing + comm stream + rank-symmetric clip | 240 → 86 collectives/cycle (Qwen3-14B); TP=2 == tp=1 WITH the clip firing; async comm bit-identical to sync | ✅ gloo 92/92 + NCCL trainer 220/220 (2026-09-08) |
| M5 | Backward CUDA graphs under TP (split the forward graph at o-proj; every collective eager) | graph path re-enabled; TP=2 graph == eager == tp=1 | ✅ 2-GPU NCCL 1196/1196 + trainer 132/132; live A/B pending |

### Cross-rank traffic — 2 residual reduces per layer + bucketed factor grads

Per layer the backward needs the two residual-stream reduces below (each ~2 MB on
Llama-3-8B at `s_max=256`, 2.5 MB on Qwen3-14B, issued synchronously on the compute
stream — the next op needs them). The four replicated LoRA-factor grads per layer are
**not** reduced inline any more (M4.3): the trainer copies them into a persistent
`FactorBucket` and reduces one contiguous slice per group of `BUCKET_LAYERS=8` layers on
a dedicated `CommQueue` stream, overlapping the next group's compute; one more tiny
`[L]` all-reduce carries the sharded factors' squared norms for the rank-symmetric clip.
The forward remat's `o` reduce exists only when `save_resid_mid` is off.
**Per cycle (Qwen3-14B, saves on): 80 + 5 + 1 = 86 collectives**, ~205 MB per rank
(was 240 / 224 on Llama-3 before P5.6 + M4.3). The graph and eager paths issue exactly
the same set (counted per cycle by `tests/test_tp_trainer_graph_nccl.py` and
`tests/test_tp_bucket_gloo.py`; per layer by `tests/test_tp_backward_graph_nccl.py`).

| Where | Tensor | Why |
|---|---|---|
| forward remat (only without `save_resid_mid`) | `o` | `o_proj` is row-parallel → partial sum |
| backward | `grad_resid_mid − grad_out`, then add back | reduce only the FFN partial; the residual passthrough is already full on every rank and would otherwise be counted `tp_size`× |
| backward | `grad_x − grad_resid_mid`, then add back | reduce only the attention-path partial |
| after the layer loop (bucketed, comm stream) | `grad_qA`, `grad_kA`, `grad_vA`, `grad_oB` for a group of layers | replicated factors — grads sum across shards; only the optimizer reads them |
| after the loop | `[L]` vector of the sharded factors' squared norms | the rank-symmetric per-layer clip |

`grad_{q,k,v}B` (output-sharded) and `grad_oA` (input-sharded) are already this rank's
correct shard — **no reduce**. Reducing `grad_x` wholesale double-counts the residual;
the gloo test caught exactly that.

> **This box has no P2P** (`custom_all_reduce` disabled on the 2× 5090), so every
> collective crosses PCIe through host memory: measured 0.36 ms per 2.5 MB reduce,
> 0.045 ms per 160 KB one. The 80 residual reduces per cycle (~29 ms on Qwen3-14B) are
> the TP floor on this hardware; M4.3 removed the other 160 from the critical path.

### Load-bearing invariants (break these and it hangs or silently degrades)

- **Lock-step.** Both ranks must fire the backward on the same step, or the collectives
  deadlock. Guaranteed by *broadcasting* the trigger on `SchedulerOutput` — never make FT
  admission a per-rank decision.
- **`_maybe_pause` stays outside any captured region.** It is how the backward yields the
  GPU to inference; an `mp.Event.wait` cannot be captured. This is why we can never
  collapse to one graph per step the way an FT-only system does.
- **Tier-C aborts must be rank-symmetric (2026-09-08).** An FT-only forward runs a TP
  collective every layer, so one rank leaving at layer k while the other continues hangs
  both. Under TP the abort is therefore a joint decision: the EngineCore input thread
  bumps a shared-memory arrival counter (`coordinator.FtArrivalSignal`, name passed to
  the workers via `DSERVE_FT_ARRIVAL_SHM`), and each worker's `FtAbortPoller`
  MAX-all-reduces its local "arrivals changed" bit over the TP group's gloo `cpu_group`
  once at entry and once per layer boundary (the `input_layernorm` pre-hook; the other
  hooks in a layer run no collective). The poller is active only between `entry()` and
  `finish()`, i.e. for FT-ONLY forwards — a co-serving batch with inference tokens is
  never aborted (an early version did, and took the inference requests' step with it).
  The engine reads the abort from `ModelRunnerOutput.finetune_aborted` (a real field —
  the dynamic `_ft_aborted` attribute does not survive the worker → engine hop), on the
  execute_model future under TP and from `sample_tokens`' pending-abort sentinel. tp=1
  keeps the `threading.Event` path. Gates: `tests/test_ft_abort_tp.py` (2-process gloo,
  asymmetric observations → same abort layer on both ranks, equal collective counts,
  inactive poller runs none) + `tests/test_accumulate_hooks.py::test_abort_poll_boundary`.
- **The per-layer clip is rank-symmetric (M4.3) — keep it that way.** Under TP the
  clip runs after the bucketed reduce via `tp.clip_layers_symmetric_`: the norm sums
  the replicated grads (identical on every rank) and the sharded grads' squared norms
  across ranks (one `[L]` all-reduce), so both ranks derive the same scale. The old
  per-layer `torch.nn.utils.clip_grad_norm_` over a rank-local set desynchronized the
  replicated masters whenever a layer's norm exceeded 1.0 — `tests/test_tp_bucket_gloo.py`
  and the NCCL trainer gate now check TP=2 == tp=1 with the clip firing. tp=1 keeps
  the original per-layer `clip_grad_norm_` unchanged. Anything that clips or scales
  grads under TP must go through the symmetric helper.
- **Stream ordering of the bucketed reduces lives in one place: `tp.CommQueue`.**
  `submit` waits on the compute stream before the collective and records a completion
  event; `wait` makes the compute stream wait before anything reads the bucket. Only
  persistent buffers are ever submitted. `DSERVE_BWD_COMM_SYNC=1` (wait after every
  submit) and `DSERVE_BWD_COMM_DELAY=<cycles>` (spin the comm stream before each
  collective) are the debug switches; the async path must stay bit-identical to both
  (checked by the NCCL trainer gate). The child no longer sets
  `CUDA_DEVICE_MAX_CONNECTIONS=1` — it would pin both streams to one hardware queue.
- **Vocab padding.** vLLM pads vocab before sharding. Llama-3's 128256 happens to need no
  padding, so the `lm_head` all-gather is exactly `[vocab, hidden]` — for other models use
  the real local `lm_head.shape[0]`, not `vocab_size // tp_size`.
- **No MPS by default (2026-09-08).** `backward_mps_percentage: 0` spawns the child as a
  plain CUDA context; no daemon runs on this box and none is assumed. Prefill latency is
  protected by the yield contract instead: `_maybe_pause` + `pause_until_prefill_done`
  (default on). Set a percentage > 0 only with `nvidia-cuda-mps-control -d` running.
- **MPS per device (only if MPS is enabled).** The child-only MPS env must partition each physical GPU
  independently.

### M4.2 — the SLO estimator under TP (GPU-validated 2026-09-01)

The single-GPU predictor pipeline was intact on both ends; only the wire between them
was missing. The coordinator is a per-process singleton, so under the multiproc executor
the runner's `push_sample` landed on the *worker's* coordinator while the scheduler
drained its *own*, always-empty one — the estimator never became ready and admission
sat on the cold-start buffer-cap budget. M4.2 extends the existing per-signal TP relay:

| Signal | Carrier | Where |
|---|---|---|
| timing samples `(StepFeatures, dur, was_graph, predicted)` | `ModelRunnerOutput.finetune_timing` (worker → scheduler, drained every `sample_tokens`) | `gpu_model_runner.sample_tokens` → `ft_scheduler.update_from_output` → `coord.push_sample` → the unchanged `schedule()` drain / tracker / refit / validation CSV |
| profiling warmup-vs-recorded gate | `SchedulerOutput.finetune_record_timing` (scheduler → worker, stamped per step) | `ft_scheduler.schedule` → the runner's timing ring |
| `was_graph` | the runner now records the CUDA-graph mode it **actually used** (the scheduler cannot see the dispatcher under TP; the old value was a guess even on one GPU) | timing ring owner tuple |
| backward ack | relayed only once **every** rank's child is done: each rank polls its own child, MIN-all-reduces a done flag over the TP group's gloo `cpu_group` (no GPU sync, only while a backward is outstanding), all ranks take the ack on the same step. Relayed on **idle (0-token) steps too** — those are how the engine keeps stepping while a backward is outstanding, and their output is `execute_model`'s early return, not `sample_tokens`; without this FT stalled through every idle valley under TP | `coordinator.{relay_backward_outstanding,poll_own_backward_ack,take_relay_ack}` + `gpu_model_runner.{_ft_relay_backward_done,_ft_fill_relay_fields}` |

**No formula change.** `T ≈ α·S + β·T_in + γ·T_ft + δ·B_d + ε·K + c` stays: TP halves the
per-layer GEMMs and adds two all-reduces per layer whose bytes scale with tokens (→ β, δ)
and whose fixed latencies land in `c`; the backward child's PCIe traffic during FT lands
in γ. Calibration is the same launch-time manufactured-shape pass — it runs through the
real executor, so under a TP=2 launch it fits TP-specific coefficients. `profile_on_launch`
is now **on** in both TP YAMLs. Rank 0's timing is sufficient because the per-layer
all-reduces keep the ranks in lock-step. `tp_size == 1` is untouched.
**Verified:** `tests/test_tp_timing_relay.py` 22/22 (round trip through pickle → tracker →
ready estimator; record gate; ack agreement across two fake ranks). **GPU (Qwen3-14B
TP=2):** the profiling pass drains 146 relayed samples and fits all three regimes
(eager rmse 0.0010); per-step `ft=` varies 120–256 with load; `[backward] <ms>ms` prints
(the `nanms` was a commented-out event read, single GPU too). The first timeline run
exposed the idle-step gap (FT ran one cycle per burst) — fixed by relaying on the 0-token
early return; cycles then fire every ~0.35 s through the valleys. Still to run: the
inference-only baselines and a `validate_estimator` TP-vs-tp1 residual comparison.

### Open, in priority order

1. **M5 live A/B (next run).** M5 landed: `common/graph.py` has `static_o` +
   `forward_tail()`; the family cores end at `static_o`; `layer_backward_graphed` takes
   `all_reduce=` and issues the six backward reduces (they were absent before — the
   design note's "already there" was wrong); the trainer's force-eager guard is gone.
   Gate left: Llama-3 / Qwen3-14B TP=2 `ft_bench_tp.py` with `backward_cuda_graph`
   true vs false — `[bwd-graph] pre-captured forward 32/32 …` from each child, `(graph)`
   cycle lines, equal loss cycle-for-cycle, ~5 ms/cycle saved.
2. ~~**M4.3 — bucketing + comm stream.**~~ Landed 2026-09-08: `tp.FactorBucket` +
   `tp.CommQueue` + `tp.clip_layers_symmetric_`; families take `reduce_factors=False`;
   the child's `CUDA_DEVICE_MAX_CONNECTIONS=1` is gone. Qwen3-14B: 240 → 86 collectives
   per cycle; uncontended cycle 123 → 121 ms (the bucket reduces overlap, the 80
   residual reduces remain the floor). Gates: `tests/test_tp_bucket_gloo.py` 92/92 (CPU,
   clip firing, TP2 == tp1), NCCL trainer 220/220 (sync/delay bit-identical, counts).
3. ~~**`forward_interruptible` under TP.**~~ Landed 2026-09-08: tier C is rank-symmetric
   (see the invariant above); ON in both TP YAMLs with a 2 ms tier-A grace. Live: the
   Qwen3-14B `ft_bench` shows both ranks aborting FT-only forwards at the same layer
   (`[ft-abort] tier C: … aborted at layer k` from each worker). Trade-off measured on
   the tight trace — see the Phase 6 / TP section of INTEGRATION_PROGRESS.md.
4. ~~`set_corpus_meta` never reaches the children under TP~~ — fixed 2026-09-08: the
   scheduler keeps the corpus total on its coordinator (`corpus_total_tokens`), every
   relayed trigger command carries it, and the worker's `execute_trigger` forwards it to
   its child once, before the first work signal (`tests/test_tp_timing_relay.py`).

### Running it

```bash
python eval-tp/ft_bench_tp.py --duration 60 --tp 2                    # Llama-3 TP bench (default --family llama3)
python eval-tp/ft_bench_tp.py --family qwen3-14b --tp 2 --duration 120 --kill-stale   # Qwen3-14B TP=2
python eval-tp/ft_bench_tp.py --family qwen3-0.6b --tp 1 --duration 60               # Qwen3-0.6B single-GPU smoke
python eval-tp/auto_benchmark_tp.py --family qwen3-14b --tp 2 --co --loose --kill-stale   # timeline replay (real co-serving workload)
python eval-tp/auto_benchmark_tp.py --family llama3 --tp 2 --tight                        # inference-only baseline
python eval-tp/launch_deltaserve.py --start-finetuning  # just bring a server up (--family as above)
python tests/test_llama3_tp_shard.py                    # M2 gate (23 cases)
python tests/test_llama3_tp_backward_gloo.py            # M3 gate (10 cases, real gloo)
python tests/test_qwen3_tp_shard.py                     # Qwen3 M2 gate (42 cases)
python tests/test_qwen3_tp_backward_gloo.py             # Qwen3 M3 gate (10 cases, real gloo)
python tests/test_tp_backward_graph_nccl.py             # M5 gate: graphed layer math, 2 GPUs + NCCL, both families (1196 cases)
python tests/test_tp_trainer_graph_nccl.py              # M5 + M4.3 gate: real trainer loop, 2 GPUs + NCCL (220 cases: graph/eager/save/sync/delay, clip on, counts)
python tests/test_tp_bucket_gloo.py                     # M4.3 gate on CPU: bucketing + rank-symmetric clip through the real trainer, 2-process gloo (92 cases)
python tests/test_accumulate_hooks.py                   # accumulator hooks on a fake vLLM-named decoder (CPU; layer_in / resid_mid / gate_up / final_in)
```

`--family` picks the preset YAML, the served model name and the output-file tag; the
YAML is the source of truth for the base model (`model.model`) and the inference
adapter (`adapters.lora_path_0`). `--config` / `--model` still override.

- **Tests are standalone scripts — run with `python`, NOT pytest** (pytest is not
  installed in the `dserve-vllm` env).
- `auto_benchmark_tp.py` is the TP counterpart of `eval/auto_benchmark.py`: it imports
  that file's replay / results-CSV / bwd-log-trim helpers and uses the family-aware
  launcher, so the measurement is identical. Modes `--loose`, `--tight`, `--nutanix`,
  `--nutanix-600-800` (the 600–800 s slice of the original Nutanix trace) map to
  `eval/timelines/5090/`; `--co` toggles co-serving vs the inference-only baseline.
  Outputs go to `eval-tp/output/` as `timeline_results_<family>_tp<N>[_co_factor_<f>_phase_<p>]_<mode>.csv`
  (+ `bwd_log…`, `bench_meta…json`, `server…log`); `--publish` drops the tags.
  `eval-tp/auto_plot_tp.py --family F --tp N [--mode M]` renders the same 4-panel figure as
  `eval/auto_plot.py` (imported builder) into `eval-tp/plots/<mode>_<family>_tp<N>_co_….png`.
- `ft_bench_tp.py` writes **per-family, per-TP** filenames (`bwd_log_{family}_tp{N}.csv`,
  `server_{family}_tp{N}.log`)
  and **truncates** the bwd log per run — the server opens it in append mode, so runs
  used to silently concatenate. It counts cycles from the CSV, not the server log: under
  TP **every rank prints its own `[backward]` line**, so log-line counting reports
  `tp_size`× the real count.
- `--kill-stale` + a pre-flight check refuse to launch when a previous run left
  GPU-resident processes (otherwise the next run dies minutes later with a misleading
  out-of-memory error).
- Env: `DSERVE_BACKWARD_NCCL_PORT` (default 29677) for the backward group;
  `DSERVE_TP_DIAG=1` enables `LoraSftTrainerService._diag_remat_check`, which
  rematerializes the full forward from `layer_in[0]` and compares it against the captured
  activations (inert otherwise).

### Known open issue that is **not** a TP bug

FT loss stalls around ~4.3 after ~25 cycles where an earlier `pure_ft` run reached ~2.6 on
the same samples — identically at TP=1 and TP=2, with the `DSERVE_TP_DIAG` diagnostic
showing the backward's forward-remat diverging from vLLM's real forward (final_in relative
error 0.22 → 0.45 over cycles). **Likely cause found in Phase 8:** the worker read
`rope_theta` from an `hf_config` attribute that transformers 5.x no longer sets (the value
moved to `rope_parameters["rope_theta"]`), so the remat's RoPE used theta=10000 against a
served model at 500000. Fixed generically in `deltaserve/ft_meta.py:rope_theta_of`.
Verification pending on GPU: re-run with `DSERVE_TP_DIAG=1` (the remat error should
collapse to bf16 noise) and `eval/pure_ft_bench.py` (the 2.12 reference should reproduce).

## Key DeltaServe co-serving contracts to preserve (from DeltaServe/CLAUDE.md)

- `_maybe_pause()` at **every layer boundary** in the backward path is how backward yields
  the GPU to inference. It is load-bearing.
- Backward runs on its own CUDA stream; time it with `torch.cuda.synchronize()` before
  reading the wall clock (else you measure host dispatch, not GPU completion).
- MPS partitioning was DeltaServe's mechanism for true concurrent execution; this project
  runs **without MPS by default** (`backward_mps_percentage: 0`) and relies on the pause
  contract, extended to hold until the prefill completes (`pause_until_prefill_done`).
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
7. For Phase 7 (tensor parallelism): the **"Tensor parallelism (Phase 7)"** section above
   is self-contained — milestones + gates, the all-reduce table, the load-bearing
   invariants, open items, and run commands. In-tree the integration points are
   `dserve-vllm/vllm/deltaserve/bwd_services/common/tp.py` (`lora_shard_slice`,
   `init_backward_tp_group`, `reduce_partial`), the reduces in each family's
   `layer_forward`/`layer_backward` (`bwd_services/{llama3,qwen3}.py`), local dims in
   `common/trainer.py:_build_state`, `dserve-vllm/vllm/v1/worker/gpu_worker.py`
   (`_maybe_share_finetuning_weights` tp geometry + `lm_head` all-gather;
   `_maybe_setup_finetuning_accumulator` local buffer widths),
   `dserve-vllm/vllm/v1/executor/multiproc_executor.py` (non-daemon workers +
   `_arm_parent_death_signal`), and the relay across
   `dserve-vllm/vllm/deltaserve/{coordinator,ft_scheduler}.py` +
   `dserve-vllm/vllm/v1/{core/sched/output.py,outputs.py,worker/gpu_model_runner.py}`.
   `DeltaServe/dserve/server/router/model_infer/model_rpc.py:63-77` is the original
   per-rank NCCL init this mirrors.
8. For Phase 8 (a second model family): read `bwd_services/common/family.py` (the
   contract), then `bwd_services/qwen3.py` next to `bwd_services/llama3.py` — the two
   differ only at the q/k-norm insertion points; `common/trainer.py` drives both. To add
   a family: write `bwd_services/<family>.py` (layer functions + `Family` + a 5-line
   service subclass), register it in `bwd_services/registry.py`, add a `tests/test_<family>_*`
   set on `tests/bwd_harness.py`, a YAML in `configs/`, and a preset in
   `eval-tp/launch_deltaserve.py`. Upstream vLLM needs nothing unless the architecture
   breaks a generic assumption (module names, SwiGLU, RMSNorm).
