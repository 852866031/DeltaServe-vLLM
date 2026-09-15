# Branch `mixed-fwd-cuda-graph` — CUDA graphs for forward steps that carry finetuning samples

Status: **implemented and gated (2026-09-15)** — see "Implementation status" at the end;
the design below is what was built. This file states what the branch is for, why
the current code cannot do it, the proposed way, what it is expected to buy and cost, and
the order of work. Numbers are from the 2026-09-08 step traces on Qwen3-14B, two RTX 5090s
(TP=2), no MPS — see INTEGRATION_PROGRESS.md, Phase 7, "Estimator validation under TP".

## Purpose

Today every forward step that contains finetuning (FT) tokens runs *eager*: it bypasses
vLLM's compiled forward and every CUDA graph, and executes the plain Python module path
(`skip_compiled` on the forward context, `force_eager` in the graph dispatcher). The goal
of this branch is to let **mixed batches — inference requests plus FT samples — replay
vLLM's piecewise CUDA graphs like any other batch**, while keeping the activation saves
the backward needs.

FT-only batches (no inference tokens) are deliberately left eager: they are the only
batches the mid-forward abort applies to, and that abort is a Python exception raised
between layers, which a graph replay cannot do.

## Why the mixed step is eager now — the one real blocker

The activations are saved by **Python hooks on the layer modules that read per-step
state**. Each hook branches on `_active` / `_cur_n`, slices the FT rows with Python
integers (`val[start:start+n]`), and writes at `buf[off:off+n]`, an offset the scheduler
reserved for that step. A CUDA graph would bake in the branch outcome, the slice bounds and
the destination address of the one step it was captured on. It is also why FT batches skip
the compiled path entirely: vLLM traces the model once at warm-up (with no FT active) and
afterwards dispatches to that bytecode without re-evaluating guards, so the hooks are
frozen as "inactive" inside it.

Everything else is secondary or already solved:

- The "pool-aliasing NaN trap" that motivated the original rule was a consequence of
  saving into graph-pool memory; with a fixed destination and the copy inside the replay
  it cannot occur.
- The mid-forward abort only applies to FT-only steps, which stay eager.
- LoRA (the FT adapter is an ordinary adapter slot), tensor parallelism and the
  capture-size buckets already work under the piecewise graphs.

## What it costs today

From the launch profiling pass (same shapes, no backward running):

| tokens in step | graphed inference prefill | eager with FT | host issue time (eager) |
|---|---|---|---|
| 192 | ~41 ms | 53 ms | 46 ms |
| 256 | 52 ms | 65 ms | 62 ms |
| 576 | ~108 ms | 125 ms | 91 ms |
| 1088 | ~200 ms | 212 ms | 152 ms |

The eager penalty is a fixed 12–16 ms per step (launch + Python overhead, not compute) on
top of a CPU issue floor of ~46 ms. The mixed step that matters on this box — a 50-token
prefill, a few decodes, one 23-token sample — costs 49 ms eager against a 50 ms
time-between-tokens gate, so admission admits only the smallest samples and 438 of 493
prefill steps on the tight trace carried no FT at all; most FT runs in idle gaps.

## Proposed way

Make the save a **fixed-shape, device-side operation whose per-step inputs are data, not
code** — the same pattern vLLM uses to feed attention its per-step metadata through the
forward context.

1. **A custom op replaces the hooks.** `dserve::save_rows(x, slot)`, registered with
   `direct_register_custom_op` (opaque to Dynamo, captured cleanly), does one
   `index_select` of exactly `max_saved` (256) rows through a persistent *source-index*
   tensor and one `index_copy_` into the shared accumulator buffer through a persistent
   *destination-index* tensor. Unused slots point at a scratch row. The FT rows' positions
   in the padded batch and the reserved write offset become the **contents** of those two
   tensors, refreshed by one small host-to-device copy before the step — never shapes,
   never addresses. Contiguous vs interleaved FT rows stops mattering.
2. **The op is called from the layer code**, guarded by a load-time module attribute, at
   the seven save points: the residual entering `input_layernorm` and
   `post_attention_layernorm` (the fused add-norm mutates `residual`, so the sum is formed
   before it, as the hook did implicitly), the MLP `gate_up` output, q/k/v (post-RoPE on
   Llama-3; the `q_norm`/`k_norm` inputs on Qwen3), the attention output, and the final
   norm input. Two model files (`qwen3.py`, `llama.py`), a few lines each.
3. **Per-step data rides the forward context** (`ft_save` next to `attn_metadata`); the
   runner fills the index tensors where it builds `_ft_mask_gpu` / `_ft_offset` today, and
   stops setting `skip_compiled` / `force_eager` for batches that also contain inference
   tokens.
4. **A `has_ft` variant of the piecewise graphs.** `BatchDescriptor` gains an `has_ft`
   bit; the mixed prefill-decode graphs are captured a second time with the op present
   (index tensors aimed at the scratch row during capture). Uniform-decode full graphs never
   carry FT and are untouched. Capturing the variant only for the sizes FT steps actually
   use (a few hundred tokens) keeps the graph-pool cost small. The alternative — leaving
   the op unconditional in every graph — costs every inference step ~0.3–0.5 ms and is
   rejected.
5. **Behind one flag**, `finetune.graph_ft_batches` (default off), so the eager path stays
   selectable and behaviour is bit-identical when off.

Correctness by construction: the destination is the existing IPC buffer (outside any pool)
and the copy is inside the replay, so no captured activation can alias pool memory; the
dedicated test captures a fake decoder twice with different index contents and checks the
saved rows follow the contents, not the capture.

## Expected impact

- **Mixed step ~20 % faster:** ~49 → ~38 ms on the tight shape; the save op adds back
  ~1–1.5 ms; worker host time ~50–60 ms → ~2 ms per step.
- **FT throughput under load, potentially 2–3× on the bursty traces:** ~12 ms of headroom
  under the 50 ms TBT gate is 60–70 more prefill tokens per step (0.17 ms/token), so most
  prefill steps can carry a 60–90-token sample instead of nothing. Median inference TTFT
  moves up toward the SLO by design (Llama-3 tight last week: 47 → 121 ms at 96.9 %).
- **The hidden cost:** 3× the FT tokens is 3× the backward cycles (~17 → ~50 % of wall
  time on tight). Every decode-only step overlapping a cycle still runs at ~2× (no MPS, no
  pause on decode), so worst-TBT > 50 ms counts grow in proportion. MPS, or pausing on
  decode after all, become the levers at that point.
- **Inference-only steps:** unaffected with the variant approach.
- **Startup / memory:** a second capture set costs ~5–10 s at launch and graph-pool memory
  taken from the KV cache (vLLM's estimate for the current set is ~15 % of GPU memory);
  restricting the variant to small sizes keeps this to a fraction.
- **Estimator:** no formula change; the "eager" regime becomes "graphed with FT" and the
  launch profiling pass must be re-run under the new mode. Accuracy should match the
  inference-prefill regime (RMSE ~1.5 ms).

## Files involved (see the session notes for detail)

| area | files | change |
|---|---|---|
| the op | `deltaserve/ft_save_op.py` (new), `deltaserve/accumulate.py` | custom op; index tensors + scratch row replace the seven hook factories on the graphed path; `begin_step` fills indices |
| call sites | `model_executor/models/{qwen3,llama}.py` | one guarded call per save point in the decoder layer / attention / final norm |
| per-step data | `forward_context.py`, `v1/worker/gpu_model_runner.py`, `v1/worker/gpu_worker.py` | `ft_save` context field; runner fills indices, drops `skip_compiled`/`force_eager` for mixed batches; worker allocates indices + scratch |
| dispatch / capture | `v1/cudagraph_dispatcher.py`, `gpu_model_runner.py:capture_model`, `deltaserve/ft_scheduler.py:_will_use_graph` | `has_ft` key + second capture set; scheduler queries with the bit |
| config | `config/finetune.py` | `graph_ft_batches` flag |
| tests | `tests/test_accumulate_hooks.py` sibling; a capture-twice test; existing hash check + gradcheck gates re-run | parity hooks vs op (contiguous / interleaved / none); saved rows follow index contents, not the capture |
| docs | `VLLM_FORK_CHANGES.md`, `INTEGRATION_PROGRESS.md`, `CLAUDE.md` invariant 1 | "any batch with FT tokens runs eager" → "FT-only stays eager for the abort; mixed batches replay the `has_ft` graphs" |

## Order of work

1. The op + index-tensor save, with mixed batches running the **compiled but uncaptured**
   path. Verify with the hook-parity test and the cross-process hash; measure the drop in
   host issue time. This de-risks the trace-time-freezing question on its own.
2. The `has_ft` dispatch variant and capture set; measure the mixed step (target ~38 ms).
3. Re-run the launch profiler, then the loose / tight / nutanix replays with the step trace
   (`eval-tp/auto_benchmark_tp.py --step-trace`, `eval-tp/analyze_step_trace.py`): TTFT
   satisfaction, FT tokens per step, and the worst-TBT count against the inference-only
   baselines.

Main risk: vLLM's compile wrapper freezes Python-side behaviour at trace time, so every
per-step value must travel through the forward context or device buffers; a leaked Python
value does not fail, it silently freezes.

## Implementation status (2026-09-15)

Built as designed, with two deviations found while implementing:

- **Keeping the op alive.** In this torch build an output-less custom op with no
  declared mutation is dead-code-eliminated by Dynamo and Inductor (`is_impure` is
  mutation/effect based; vLLM's own `unified_kv_cache_update` survives only because its
  dummy output feeds the attention op). The save op therefore declares a mutation on a
  1-element `marker` buffer registered on the calling module — the same mechanism as
  the attention op's `output` — while the real destination comes from the forward
  context by slot id. Cost: one tiny in-place kernel per call.
- **Where the residual sum is formed.** The hooks summed `hidden + residual` over the
  whole batch before the fused add-norm; the op gathers both rows first and adds only
  the `max_saved` rows.

Gates: `tests/test_ft_save_op.py` 28/28 (a graph captured once writes the rows the
index tensors name at replay, interleaved rows, a second offset, `ft_save=None`
captures nothing; the accumulator's op path matches the hook path bit-for-bit on a fake
vLLM-named model), the CPU gates (`test_accumulate_hooks` 153/153, `test_merged_estimator`
57/57 incl. the `ft_mixed` split, relay 26/26, step trace 28/28), and the Qwen3-0.6B
single-GPU co-serving smoke (tight trace, `DSERVE_TP_DIAG=1`): 339 mixed FT batches
replayed `graph(PIECEWISE)`, all 331 FT-only batches eager, no tracebacks, loss
4.40 → 1.71 over 669 cycles, `ft_mixed` predicted to 0.2 ms RMSE. Capture set on the
0.6B: +51 `has_ft` piecewise graphs, 0.90 GiB total.

## Results (2026-09-15, Qwen3-14B, TP=2, TTFT SLO 0.4 s, max-TBT 50 ms unless noted)

Baseline = the same code with `graph_ft_batches: false` (the `tp` branch behaviour, runs of
2026-09-08 night); "graphed" = this branch. Same traces, same SLOs, same stack otherwise.
Per-step numbers from the step trace (`eval-tp/compare_runs.py`).

| | tight (dense) | | loose | | Nutanix 600–800 s | |
|---|---|---|---|---|---|---|
| | baseline | graphed | baseline | graphed | baseline | graphed |
| TTFT satisfaction | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % |
| TTFT p50 / p95 / p99 (ms) | 76 / 94 / 121 | 112 / 167 / 208 | 75 / 95 / 132 | 109 / 166 / 228 | 74 / 115 / 178 | 78 / 175 / 262 |
| FT tok/s | 289 | **400 (+39 %)** | 685 | 739 (+8 %) | 420 | **488 (+16 %)** |
| prefill steps carrying FT | 17 / 480 | **189 / 474** | — | 77 | 228 / 534 | 361 / 533 |
| FT tokens per mixed step (p50) | 23 | 24 | — | — | 29 | **62** |
| mixed step time (p50) | 49.6 ms | 46.8 ms | — | 48.4 ms | 47.9 ms | 46.6 ms |
| FT cost above the same inference composition (p50) | 14.0 ms | **4.3 ms** | — | — | 14.4 ms | **8.4 ms** |
| worker host time per mixed step | 45 ms | **2 ms** | 62 ms | 2 ms | 45 ms | 2 ms |
| requests with worst-TBT > 50 ms | 57 / 480 | **442 / 480** | 60 / 240 | 196 / 240 | 188 / 534 | 383 / 534 |
| avg-TBT p50 | 29 ms | 39 ms | 26 ms | 31 ms | 20 ms | 25 ms |
| backward cycles in window | 54 | 76 | 125 | 136 | 319 | 371 |

**What the mechanism delivered.** The marginal cost of putting FT samples into an inference
step fell from ~14 ms to 4–8 ms (the eager penalty is gone; what remains is the samples'
own compute plus the ~1–2 ms fixed-size save), and the worker CPU went from saturated
(45–62 ms of Python per mixed step) to 2 ms. Because the admission gate sits at the 50 ms
TBT limit, that headroom is spent immediately: 11× more prefill steps carry FT on tight,
and on Nutanix each mixed step carries twice the FT tokens. FT throughput +39 / +8 / +16 %.

**What it costs.** Exactly the trade the design note predicted, only sharper: every decoding
request now rides ~47 ms mixed steps far more often, so avg TBT rises ~10 ms on tight and
the worst-TBT > 50 ms count goes from 12 % to 92 % of requests. The gate admits up to a
*predicted* 50 ms with the (inert, 0.2 %) safety margin, so steps at 47–50 ms predicted
routinely measure 50–55 ms. TTFT stays at 100 % everywhere; medians move up toward the
SLO as designed. Memory: the second capture set raised graph memory 4.99 → 6.81 GiB and
halved the KV cache (59.7k → 32.1k tokens on 2048-token requests; still 15.7× concurrency,
enough here). Startup: +63 s compile once (cached), +13 s capture.

**Buying the TBT back.** Tight with `max_tbt_slo: 0.045`
(`configs/serving_config_finetuning_qwen3_14b_tp2_tbt45.yaml`): FT 289 → 324 tok/s (+12 %),
119 mixed steps at 43.7 ms, worst-TBT > 50 ms on 159 / 480 (was 442 with the 50 ms target,
57 in the baseline), TTFT 100 %, p50 81 ms. The gate does exactly what its target says; a
real safety margin (relative or absolute, see INTEGRATION_PROGRESS.md) would let the 50 ms
target mean 50 ms measured.

**Correctness.** Qwen3-0.6B single GPU, tight, hook path vs graph path with the
rematerialization check on: loss cycle-for-cycle 4.402/4.403 → 3.576/3.566 → 2.327/2.325 →
1.735/1.730, equal FT tokens (103k), remat errors at bf16 noise in both (the graph run's
are lower), no tracebacks. Under TP the `[batch]` log shows `graph(PIECEWISE)` on every
mixed FT batch and `eager` on every FT-only one; `ft_mixed` predicted to 1.1–1.5 ms RMSE.

**Observed, not resolved.** Two inference-prefill steps on loose ran ~2.6–2.9× their
prediction with no backward flagged, both right after a mixed FT step (seq 938, 1270); the
next-step-after-trigger contention seen last week now lands mid-burst more often. Rare (2 of
25 clean prefills), noted for follow-up.
