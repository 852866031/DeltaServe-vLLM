# Reference for paper revision: DeltaServe → DeltaServe-vLLM

This document is a reference for the writer revising the DeltaServe paper
for the vLLM-port version. It answers every question in
`QUESTIONS_FOR_IMPLEMENTER.md` directly, and adds context the writer will
need to reframe the paper's claims and related-work positioning.

For each mechanism the original paper described, the question is roughly
"is it still there, did it change shape, or was it absorbed into vLLM?"
That framing drives the structure below.

---

## TL;DR for the writer

DeltaServe-vLLM is **the same co-serving idea, re-hosted on vLLM**. The
mechanism contributions of the original paper survive — SLO-aware FT
admission, mixed inference+FT batches, decoupled backward subprocess
under MPS, per-layer cooperative pause, fixed-shape backward CUDA-graph
capture. Two things were forced to change because of vLLM's substrate:

1. **The estimator was merged.** vLLM's continuous batching mixes
   prefill + decode in one step, so the original paper's three separate
   regime estimators (prefill, decode, graph) collapse into one
   six-parameter linear model that predicts mixed-batch step time
   directly. Graph vs eager remains a regime split inside that one
   model.

2. **The backward subprocess does more work.** vLLM only materializes
   logits for sampled positions, so the LM-head matmul moved from the
   inference forward into the backward. The forward writes pre-LM-head
   hidden states to shared GPU memory; the backward reconstructs full
   logits (chunked over vocab) before the rest of the manual SFT
   backward.

Three things are new (no analog in the original):

- An **async-pipelining-safe scheduling layer** — queue-wait term in
  the TTFT slack formula, deferred GPU-event timing ring, reserve-at-
  inject activation-buffer accounting. Required because vLLM builds the
  next batch while the current one is still running on the GPU.

- A **mid-forward inference pre-emption pipeline** ("forward
  interruptible") at three windows — pre-schedule grace poll, post-
  schedule rollback, mid-forward abort. Closes the late-HTTP-arrival
  case that continuous batching creates.

- An **opt-in unified-phase admission mode** that lets FT ride
  decode-only and mixed-step batches under the SLO estimator's gate,
  not just prefill-carrying steps. The default still matches the
  original paper's conservative rule.

Several things from the original were dropped because vLLM provides
them natively (multi-LoRA inference kernel, paged-KV allocator,
sampler, tokenizer integration). The port is roughly an order of
magnitude smaller than the original DeltaServe codebase.

The remaining sections answer the questionnaire in order.

---

## Section 1. One-paragraph descriptions (Q1–Q2)

### Q1. What is DeltaServe-vLLM?

DeltaServe-vLLM is a **co-serving add-on for vLLM** that interleaves
LoRA fine-tuning with online inference on a single GPU. It injects
fine-tuning samples into ordinary inference batches as single-step
prefill-only requests routed to a dedicated FT LoRA adapter, captures
their activations during the forward, hands them to a separate backward
subprocess that runs the SFT gradient computation under an MPS-shared
GPU partition, and gates admission so inference TTFT/TBT SLOs stay
within target. To a new engineer:

(a) **What it does**: trains a LoRA adapter on a fine-tuning corpus
while the same GPU continues to serve inference, with explicit SLO
guardrails on the inference path.

(b) **What it sits on top of**: vLLM's V1 engine, near-trunk (~123
commits past `v0.21.1rc0`). It uses vLLM's continuous-batching scheduler
as a base class, vLLM's PagedAttention KV allocator unchanged, vLLM's
multi-LoRA inference path (punica) unchanged, and vLLM's OpenAI-
compatible HTTP frontend unchanged.

(c) **What the operator gets for free** vs vanilla vLLM: live
LoRA-SFT training of a chosen adapter from a chosen corpus, with the
trained weights published into the inference-served adapter slot at
every optimizer step, all without the operator having to take the
service offline for training. The fine-tuning side is invisible to API
clients (no extra requests appear in the output stream).

### Q2. vLLM-specific feature or reusable layer?

**Mostly reusable; vLLM-specific in a few load-bearing places.** The
code splits as follows:

- `dserve-vllm/vllm/deltaserve/` (~7000 LoC) — the co-serving layer.
  Mostly host-agnostic. The finetuning sample store, FT request
  injector, coordinator (activation-buffer state machine + backward
  triggers + IPC events), SLO estimator (linear model + offline
  profiler + online refit), backward subprocess parent, and per-model
  backward services (Llama-3, OPT) are all reusable. A small amount is
  vLLM-specific: the activation accumulator's forward-hook discovery
  relies on vLLM's module naming (`model.layers.{i}.input_layernorm`,
  etc.), and the scheduler is a vLLM-`AsyncScheduler` subclass. None of
  these are deep couplings — porting to another framework would touch
  these files but leave the math and the IPC alone.

- `dserve-vllm/vllm/v1/...` — small `[DeltaServe]`-tagged edits to
  vLLM internals (Worker init, GPUModelRunner input prep, EngineCore
  lifecycle, the Request and SchedulerOutput dataclasses to carry a
  `finetune_mask` and an `is_finetuning` flag). These are the
  vLLM-specific hooks.

If a follow-up paper revision wants to claim the port is "vLLM
plus a reusable co-serving plugin," that's a fair framing — most of the
plugin would also work on SGLang or TGI with a similar number of small
upstream-side edits.

---

## Section 2. What survives from the original DeltaServe

The table the writer asked for. Every mechanism from the original paper:

| Mechanism (original paper) | kept / modified / dropped | What changed in the port (if modified) |
|---|---|---|
| Mixed prefill batch fusing inference + LoRA forward | **modified** | Still a mixed batch, but now produced by vLLM's continuous-batching scheduler with an FT-admission gate on top. The FT samples are scheduler entries (`max_tokens=1`, FT adapter id) rather than rows in a custom kernel. The forward path is vLLM's, not a custom kernel. Activation capture is by Python forward hooks on the residual-stream norms and the FFN pre-activation, not in the kernel. |
| Closed-form prefill latency model | **modified** | Merged into a single six-parameter linear step-time model that predicts mixed prefill+decode+FT step time. The prefill quadratic term survives as `α·S` where `S = T_in² / P`. See §3 Q3 below for the full formula. |
| Closed-form decode latency model | **modified** | Same — merged into the single model. The decode contribution is `δ·B_d + ε·K` (batch size and total context length). |
| Two-mode (graph vs eager) coefficient sets in the estimator | **kept** | The merged model fits two separate coefficient sets — one for steps that will run as a CUDA graph, one for eager. Each step is stamped with the regime at schedule time; the estimator picks the right set at predict time. |
| Offline profiling pass that seeds the estimator | **kept** | Runs at launch before serving begins. Sweeps a representative set of batch shapes through the live scheduler (with FT admission suppressed) and fits both coefficient sets. Without this the estimator can't make decisions on the first inference burst. |
| Online least-squares refit of estimator coefficients | **kept** | Every 256 paired (predicted, actual) records, refit using the partitioned least-squares fit. Predicted-vs-actual stamps can be dumped to a CSV for offline analysis. |
| Greedy router that selects requests by SLO budget | **modified** | The "router" abstraction is gone — vLLM's `Scheduler` owns request selection. The SLO budget decision now sits inside `FinetuneScheduler` (subclass of `AsyncScheduler`) and decides only **how many FT tokens** to admit per step, not which inference requests to run. Inference request ordering is vLLM's. |
| BatchConstructor that gates on TTFT and TPOT | **modified** | TTFT gate kept (using the same `earliest_arrival_time + 0.9·ttft_slo` formula, with a new async-pipelining queue-wait term — see Q3). Per-step max-TBT gate kept. Average-TBT gate deferred (needs per-request last-token tracking under continuous batching; configurable but not enforced). |
| Decoupled backward subprocess | **kept** | Spawned by the Worker process under a child-only MPS env. Same `daemon=True` lifecycle. Uses `torch.multiprocessing` (registers CUDA-IPC reductions automatically). |
| Cooperative per-layer pause / `_maybe_pause()` contract | **kept (with two refinements)** | Same shared `multiprocessing.Event` checked at every layer boundary in the backward. Refinement 1: yield only on prefill-carrying forwards, decode-only forwards co-run with the backward. Refinement 2: fire-and-forget toggles, no blocking handshakes between pause and resume. |
| MPS partition for concurrent backward | **kept** | `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` set before backward spawn, applied to the child only via the env-wrap-then-restore pattern. Default 10% (same default as the original paper). |
| Unified paged pool extended with FT activation pages | **dropped** | KV cache is vLLM's `PagedAttention` block manager unchanged. FT activation buffers are pre-allocated flat tensors outside any vLLM allocator pool. The "single pool with mixed page types" framing from the original paper does not apply to the port. |
| GQA-packed KV view | **dropped** | vLLM owns KV layout; we don't touch it. |
| Backward CUDA graph capture at fixed `max_saved_finetuning_tokens` | **kept and extended** | Same fixed-shape strategy. The port captures three regions per layer instead of two: (i) **forward recompute**, (ii) FFN-backward, (iii) one **shared** padded-attention-backward (reused across all L layers, possible because the attn-bwd core has no per-layer weights). Pre-captured at child startup against zero-initialized static buffers. Per-layer eager fallback on capture failure. Default OFF (opt-in via `finetune.backward_cuda_graph`). |

Two of the original-paper mechanisms (unified pool, GQA-packed KV) are
genuinely gone because vLLM owns that part of the stack. The writer
should consider removing or substantially rephrasing those sections of
the paper; they no longer describe what the port does.

---

## Section 3. New / changed mechanisms (Q3–Q5)

### Q3. Fine-tuning admission code-path walk-through

End-to-end, from a fine-tuning corpus line to GPU tokens. Naming files
where useful; everything else is conceptual.

**At engine init:**

1. The FT corpus is loaded and tokenized into a length-bucketed pool
   (`deltaserve/finetuning_store.py:FinetuningStore`). Samples whose
   token length exceeds `max_saved_finetuning_tokens` are dropped at
   load time with a warning (avoids deadlocking admission once smaller
   samples drain).

2. The Worker spawns the backward subprocess and shares (via CUDA IPC,
   zero-copy): base model weights, fp32 FT adapter weights, all
   activation buffers, and vLLM's served LoRA stacked buffers
   (`deltaserve/backward_process.py:BackwardProcess`).

3. The FT injector creates an in-process `LoRARequest` for the FT
   adapter at a reserved slot id (`deltaserve/ft_injector.py`).

4. The Worker creates the `FinetuneCoordinator` singleton
   (`deltaserve/coordinator.py:FinetuneCoordinator`) — shared state
   machine tracking buffer fill, reservations, admission gates,
   backward triggers, and the GPU yield event.

5. An offline profiling pass sweeps representative batch shapes
   through the live scheduler and seeds the SLO estimator
   (`deltaserve/estimator.py:MergedExecutionEstimator`).

**At each scheduler step:**

6. `FinetuneScheduler.schedule()` (in `deltaserve/ft_scheduler.py`)
   runs before vLLM's base `super().schedule()`. It does:

   - Drain completed step durations from the deferred GPU-event ring
     into the estimator's tracker. Refit coefficients every 256 paired
     records.
   - Poll the backward subprocess for completion of the previous
     trigger (via the coordinator). If done, reopen FT admission and
     mark contributing samples as `trained=True`.
   - Compute the upcoming step's inference composition
     (`_current_step_features`): `(T_in, B_d, K, P)` from the running
     queue and the waiting queue head, and the earliest waiting
     request's arrival time.
   - Compute the FT budget for this step (`_initial_ft_budget` →
     `_slo_ft_budget`). See Q4 / §6 for the formula.
   - Pop FT samples from the store via `pop_best_under(budget)`,
     enqueue them as `Request` objects with `is_finetuning=True` and
     the FT adapter id, and reserve their token count in the
     activation buffer.

7. `super().schedule()` runs unchanged — vLLM admits FT requests like
   any other waiting request.

8. After schedule, `_features_from_output` stamps the actual realized
   batch's features for later online refit, and the runner reads the
   FT mask from the scheduler output.

**In the model runner:**

9. `GPUModelRunner` builds the `finetune_mask` and per-sample lengths
   from the FT request ids in the scheduler output (small
   `[DeltaServe]`-tagged edit). It forces the step eager
   (`force_eager=self._ft_has`) and disables CUDA-graph capture.

10. The forward runs. Per-layer forward hooks on `input_layernorm` and
    on `mlp.gate_up_proj` (and optionally on `self_attn.attn` with the
    F1 flag) copy the FT-row activations into the shared GPU buffers at
    the coordinator-supplied write offset.

11. After the forward, the runner records a CUDA event for the
    deferred timing ring and the coordinator advances `fill_count`.
    When the buffer fills (or an epoch boundary fires), the coordinator
    signals the backward subprocess.

**In the backward subprocess:**

12. The backward computes loss + per-layer gradients, runs the
    optimizer, publishes the updated LoRA into vLLM's served buffers
    (no copy — same memory), and acks back.

13. Throughout the backward, the per-layer pause check yields the GPU
    to any inference forward carrying prefill tokens.

**Admission decision — inputs and output:**

The admission decision (in `_slo_ft_budget`) reads:
- `T_in`, `B_d`, `K`, `P` — features for the upcoming step (computed
  before FT is added).
- `earliest_arrival_time` — wall-clock of the oldest waiting request,
  read off `self.waiting[0].arrival_time`.
- `ttft_slo`, `max_tbt_slo`, `decode_only_ft_safety_margin` — from
  config.
- `T_current = estimator.predict(features, will_use_graph=False)` —
  the predicted eager-step time without FT.
- `now`, `queue_wait` (= the previously scheduled step's predicted
  time under async pipelining).
- `coord.next_ft_budget()` — buffer-space cap.
- The estimator's `max_next_ft_tokens(slack_seconds)` — solves the
  estimator inverse for FT tokens that fit a given slack budget.

It produces a single integer: how many FT tokens the scheduler may
admit this step. Zero means no FT this step.

### Q4. Scheduling: top-level scheduler or hooks?

**A new top-level scheduler, but it inherits from vLLM's `AsyncScheduler`**
and only overrides a small set of methods. So it's better framed as
"vLLM's continuous-batching scheduler with FT admission and lifecycle
hooks on top," not as a replacement.

The interception points are:

- **`schedule()`** — the only deep override. Before calling
  `super().schedule()`, the FT scheduler runs the SLO budget
  computation, polls the backward, optionally injects FT requests into
  the waiting queue, and stamps deferred-event features. After, it
  records features against the realized batch.

- **`update_from_output()`** — light override. FT requests are retired
  here via `_free_blocks` *before* the base loop runs, so they free KV
  the same step and never produce an `EngineCoreOutput` for the
  frontend.

- **`_initial_ft_budget()` and `_slo_ft_budget()`** — new methods
  (not overrides). These are the admission gate. The "unified-phase"
  scheduler is a sibling subclass that overrides `_initial_ft_budget`
  to drop the decode-only short-circuit.

- **Hook on the coordinator's `on_backward_done` callback** — fires
  `store.commit_claimed(samples)` when the backward acks completion,
  flipping `trained=True` only after work was actually done (fixes a
  pre-existing bug — see Q5).

No interception of sequence selection beyond what FT injection does
(adding `is_finetuning=True` requests to `self.waiting`). No
interception of eviction or swap-out — FT requests are too short-lived
to be evicted. No interception of the block manager.

### Q5. Anything entirely new (no analog in the original)

Five things the writer can frame as "new in the vLLM port":

1. **Async-pipelining-safe admission + buffer accounting.**
   vLLM's `async_scheduling` builds the next batch while the current
   one is still running on the GPU. The port handles this with a
   queue-wait term in the TTFT slack formula
   (`queue_wait = previously scheduled step's predicted time`), a
   deferred GPU-event timing ring (RING=4 steps, read off the hot
   path), and reserve-at-inject buffer accounting (admitted-but-not-
   yet-saved FT tokens are reserved; subsequent steps see the
   reservation, backward triggers only fire when no reservations are
   outstanding). None of this has an analog in the original — DeltaServe's
   loop is fully serial.

2. **Mid-forward inference pre-emption ("forward_interruptible").**
   Three windows: (i) **pre-schedule grace poll** — when the next
   step would be FT-only, briefly block on the engine input queue so
   a late HTTP arrival can join this step. (ii) **post-schedule
   rollback** — if the scheduled batch is FT-only and an arrival
   landed since, release reservations and re-schedule once. (iii)
   **mid-forward abort** — a per-layer hook check raises an
   `FTAborted` sentinel; the runner zeros the partial-write tail; the
   engine treats the step as empty. Default OFF; opt-in via
   `finetune.forward_interruptible`. Code complete, GPU validation
   pending.

3. **Unified-phase admission mode.** New scheduler variant
   (`BothPhaseFinetuneScheduler`) that admits FT on any step
   composition under the SLO estimator's gate, including decode-only
   and mixed-step batches. Default mode (the conservative
   "FT only on prefill" rule) is preserved. The merged estimator is
   what makes this possible — DeltaServe's three separate models
   couldn't predict FT-on-decode-only because that case never appeared
   in their training data. Opt-in via
   `slo.coserving_admission_phase: both`. A new
   `decode_only_ft_safety_margin` (default 0.7) tightens the TBT
   budget on decode-only steps to bound estimator cold-start risk.
   Code complete, GPU A/B pending.

4. **Three-phase sample lifecycle (claim / commit / release).**
   Replaces the one-way `confirmed_trained=True` mark from the
   original DeltaServe. Samples are `claim`-ed at admit time (removed
   from the selectable pool), `commit_claimed`-ed at backward
   completion (`trained=True`), or `release_claimed`-ed on rollback
   (returned to the pool). Fixes a pre-existing bug in both the
   original DeltaServe and the port's earlier state where samples
   were marked trained at admit time — before any backward had actually
   processed them, so a rolled-back admission would still be counted as
   trained. Required for forward_interruptible's rollback paths but
   useful on its own.

5. **Three-region backward CUDA-graph design.** DeltaServe's
   `SFT_service_graph.py` captures two regions per layer (FFN-bwd
   and padded-attention-bwd, both per-layer). The port adds a third
   region — **per-layer forward recompute** — and collapses
   padded-attention-bwd to **one shared graph** across all L layers
   (possible because LoRA-grad writes are kept outside the captured
   region). End result: 3 captured regions per layer (32 forward, 32
   FFN-bwd, 1 shared attn-bwd) vs DeltaServe's effective 64.
   Pre-captured at child startup. Gradient values bit-identical to
   eager (gradcheck 111/111).

Plus an opt-in activation save (post-RoPE q/k/v per layer; `F1` /
`save_attn_qkv`) that lets the backward skip Q/K/V projection + RoPE
entirely. ~5 ms saved per backward at +96 MB activation pool on
Llama-3-8B. No analog in the original.

---

## Section 4. vLLM integration details (Q6–Q8)

### Q6. vLLM version

**vLLM `v0.21.1rc0` + 123 commits.** The integration uses vLLM's V1
engine (`vllm/v1/...`). Not the V0 engine path.

Two model runners exist in this version of V1
(`gpu_model_runner.py` v1 vs `gpu/model_runner.py` v2); the port
targets the v1 runner, which is the default for everything except
`Qwen3ForCausalLM`. Don't reference the v2 runner in the paper — the
port does not patch it.

Build mode: Python-only, precompiled kernels. We do not touch vLLM's
C++/CUDA. The forward LoRA path uses vLLM's precompiled punica
kernels unchanged; the backward is hand-rolled PyTorch + CUDA graphs.

### Q7. Where the hook sits

Sketch:

```
[client HTTP]
   └─> vLLM API frontend (sharded via --api-server-count N)
          [DeltaServe edit]: /start_finetuning endpoint registered
                              when enable_finetuning is on
        └─> ZMQ
   └─> vLLM EngineCore process
          [DeltaServe edit]: minor lifecycle hooks for FT-aware logging
                              and the forward_interruptible Tier A grace poll
        └─> vLLM Scheduler   <── HOOK: FinetuneScheduler subclasses AsyncScheduler;
                                       overrides schedule() and update_from_output()
                                       to inject FT, gate on SLO, and retire FT
                                       same-step before the frontend sees them
   └─> vLLM Worker (GPU-owning process)
          HOOK: BackwardProcess spawned here at init; CUDA-IPC weight/buffer
                share happens here; FinetuneCoordinator singleton lives here
        └─> vLLM GPUModelRunner
              HOOK: small [DeltaServe] edits read the FT mask from
                    SchedulerOutput, force eager when has_ft, and
                    drive the FinetuneAccumulator's per-step state.
                    Forward hooks (registered at model load) capture FT
                    activations into shared GPU buffers.
            └─> vLLM PagedAttention BlockManager  <── NO HOOK; untouched
            └─> vLLM Sampler                     <── NO HOOK; untouched
                                                       FT requests have
                                                       max_tokens=1 and are
                                                       retired pre-output

[separately spawned by Worker]
   └─> Backward subprocess (daemon, MPS-partitioned child)
          reads activation buffers via CUDA IPC; writes trained LoRA
          directly into vLLM's served LoRA stacked buffers via CUDA IPC
```

Summary in prose: the port adds a scheduler subclass, three small
`[DeltaServe]`-tagged edits to vLLM internals (Worker init,
GPUModelRunner input prep + eager forcing, EngineCore minor logging
hooks), one new HTTP endpoint, and a backward subprocess spawned by
the Worker. The BlockManager and Sampler are not touched. The frontend
is not modified beyond mounting the one new endpoint.

### Q8. CUDA graphs in vLLM, FT, and the estimator

vLLM captures CUDA graphs for decode steps and (in newer versions)
piecewise for prefill. The port adds the following rule:

> **Any batch carrying fine-tuning tokens runs eager.**

This is enforced by the model runner setting `force_eager=True` when
the scheduler output's FT mask has any True entry. Without this, the
activation-capture forward hooks would fire inside a captured graph and
their copy ops would alias with graph-pool memory, breaking subsequent
replays.

The **two-mode (graph vs eager) estimator from the original paper
still applies**, and is in fact more important here than in DeltaServe.
The merged estimator fits two separate coefficient sets — one for steps
that will run as a CUDA graph, one for eager. Each step is stamped with
the regime at schedule time using vLLM's real `CudagraphDispatcher`
(shared via the coordinator singleton; no mirror), and the right
coefficient set is used at predict time. The eager-vs-graph delta on
Llama-3-8B is ~10x at small batch sizes, so getting the regime right
matters.

One paper-side framing note: in the original paper, the eager/graph
split was about whether the prefill kernel ran as a graph. In the port,
it's about whether the *whole step* runs as a graph (vLLM doesn't graph
prefill-carrying steps by default but does graph decode-only steps).
The estimator's regime split adapts naturally to this — it just needs
to know "will this step be graphed" — but the semantic is slightly
different, which the writer may want to call out if the paper goes
into kernel-level detail.

---

## Section 5. Memory model (Q9–Q13)

### Q9. KV cache

**vLLM's native PagedAttention block manager, unchanged.** The port
does not bypass or replace the block manager. FT requests get KV
blocks the same way inference requests do; the FT scheduler explicitly
calls `_free_blocks` to release those blocks the same step the FT
request was admitted.

### Q10. FT activations

**Pre-allocated flat GPU buffers, outside any vLLM allocator pool.**
The activation accumulator allocates `torch.zeros(...)` tensors at
worker startup, sized at `[max_saved_finetuning_tokens, hidden_size]`
(or wider for the MLP gate||up and the optional F1 q/k/v saves). These
buffers are NOT registered with vLLM's `BlockAllocator`; they're
ordinary CUDA tensors in the default caching allocator.

Three reasons:

- vLLM's `BlockAllocator` is opinionated about block layout — page
  size, refcounts, prefix caching, copy-on-write. None of that matches
  the access pattern for FT activations (a flat write/read window of
  fixed total size).
- The CUDA-graph backward needs **stable addresses** for its static IO
  buffers (graph capture binds to tensor addresses). vLLM's allocator
  may move blocks around (e.g., during swap-out, eviction, or KV
  migration).
- The buffers need to be shared zero-copy with the backward subprocess
  via CUDA IPC. Sharing arbitrary vLLM blocks would require teaching
  the block manager about IPC; allocating outside it is simpler.

### Q11. Adapter weights

**vLLM's existing LoRA support (punica), with one piece of plumbing.**
At engine init, the FT-target adapter is `add_lora`-ed into a stable
served slot and `pin_lora`-ed (so it never gets evicted). The
inference path reads it like any other LoRA. After every optimizer
step, the backward subprocess writes the trained fp32 master values
directly into vLLM's served LoRA stacked buffers — same CUDA memory,
no copy step on the inference side.

One detail the writer may want to flag: vLLM's punica callsite
hardcodes the LoRA scale to 1.0, not the per-adapter α/r. The port
bakes α/r into the B matrix at publish time so the net inference effect
is correct. If vLLM ever switches that callsite to pass the per-adapter
scaling, the port's publish step must drop the bake-in. This is
documented in the publish function's docstring.

### Q12. "Unified pool" framing

**No longer accurate for the port.** In the original DeltaServe, KV +
adapter + FT activations live in one paged pool managed by the unified
allocator. In the port:

- KV lives in vLLM's PagedAttention block manager.
- Adapter weights live in vLLM's served LoRA stacked buffers.
- FT activations live in flat pre-allocated buffers outside both.

These are three separate pools. The paper section that describes a
unified pool either needs to be removed for the vLLM port, or
substantially rephrased — e.g., "in the vLLM port, the unified-pool
abstraction is dropped because vLLM owns KV and LoRA storage; FT
activations live in a separate pool sized at startup."

### Q13. Activation budget knob

**Same knob, same name, same role:** `max_saved_finetuning_tokens`.
It still:

- Sizes the per-layer activation buffers (`[max_saved_finetuning_tokens, hidden]`).
- Caps per-step FT admission (the coordinator's buffer-space budget).
- Drives the fixed shape of the backward CUDA-graph capture (the
  `s_max` in `Llama3GraphedBackward`).

Two new related knobs the writer may want to mention:

- `backward_cuda_graph_attn_bn_max` (default 8) — max distinct
  samples per backward in the padded-attention graph. Overflow falls
  back to eager attention for that backward.
- `backward_cuda_graph_attn_l_max` (default 64) — max per-sample
  sequence length in the padded-attention graph. Same overflow
  behavior.

These two knobs are vLLM-port-specific; the original paper didn't
need them because DeltaServe's padded-attention shape was set
implicitly by the unified pool's page geometry.

---

## Section 6. Backward execution (Q14–Q16)

### Q14. Separate GPU subprocess under MPS?

**Yes — same as the original.** Spawned by the Worker process (the
vLLM equivalent of DeltaServe's model server), with `daemon=True`,
under a child-only `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` env (set
immediately before `.start()` and restored in a `finally` block so
only the child inherits the constrained partition). Default 10% MPS
partition, configurable via `finetune.backward_mps_percentage`.

Single-GPU only currently. Multi-TP would need one backward subprocess
per worker rank; documented as a known limitation (see Q22).

### Q15. Per-layer cooperative pause: signaling path changed?

**Same shared-status contract as the original code** — a
`multiprocessing.Event` (the GPU grant), set means "may run", cleared
means "yield". The backward checks it at every layer boundary; the
event is created in the parent and passed to the child via the spawn
arguments. No CUDA events, no streams, no shared memory beyond the
event itself.

Two refinements:

- **Yield only on prefill-carrying forwards.** The model runner clears
  the event around any forward that carries prefill tokens (TTFT-
  critical) and leaves it set on decode-only steps (decodes co-run
  with the backward, recovering throughput DeltaServe leaves unused).
  Detected from the scheduler-stashed `_ft_step_features.t_in > 0`.

- **Fire-and-forget toggles.** The runner no longer does a blocking
  `event.synchronize()` between pausing the backward and resuming it.
  The cross-process visibility wait is scoped to a separate
  "capture-completion event" (recorded by the runner after the
  activation copies) rather than a full device-sync.

If the paper claims a specific pause-resume latency or contract, the
writer should rephrase: the port's pause is one-way and non-blocking,
not a synchronous handshake.

### Q16. Backward CUDA graph: fixed shape, same configuration?

**Yes, fixed shape, same config knob.** The capture width is
`s_max = max_saved_finetuning_tokens`. The padded-attention
sub-region is captured at `[bn_max, l_max]` — see Q13 for the two new
related knobs.

What's different from the original capture:

- **Three captured regions per layer** instead of two: forward
  recompute, FFN-backward, padded-attention-backward. The forward
  recompute region is new in the port.
- **Padded-attention-backward is one shared graph** across all L
  layers (the core has no per-layer weights). DeltaServe captures L
  attn-bwd graphs because it writes LoRA `.grad` inside the captured
  region.
- **Pre-captured at child startup**, against zero-initialized static
  buffers — capture only depends on shapes/addresses, not values, so
  the first real backward sees only replay cost (no warmup + capture
  stalls land on a live co-serving step).
- **Per-layer eager fallback** on capture or shape-fit failure (the
  failed layer is added to `ffn_failed` / `fwd_failed`; that layer
  runs eager forever after).

All of this is gated by the same flag — `finetune.backward_cuda_graph`,
default OFF. Gradient values are bit-identical to eager (the graphed
and eager paths share `ffn_backward_core` / `attn_backward_core` /
`layer_forward` as math primitives).

---

## Section 7. Experiments (Q17–Q20)

The writer should treat these as "what's actually been run and works"
vs "what's planned but not yet measured". I'm being explicit about
which is which.

### Q17. Configurations actually run end-to-end

**Hardware:** RTX 5090 (Blackwell sm_120, 32 GB). vLLM's precompiled
kernels cover sm_120; FlashInfer JIT-compiles sm_120 on first run
(one-time). There is also a profile of an A100 box in the eval scripts
but the headline runs are on the 5090.

**Base model:** `meta-llama/Meta-Llama-3-8B` (on the 5090 box).
`meta-llama/Meta-Llama-3.1-8B` is the A100-box default in the harness.
opt-125m is kept as a frozen loss-only reference path for verification
(its backward is loss-only, no per-layer SFT loop).

**Workloads:** request timelines from `eval/timelines/5090/`:
- `timeline_loose.csv` — loose-co (lighter request density).
- `timeline_tight.csv` — tight-co (heavier).
- `timeline_nutanix.csv` — Nutanix-like trace.

The benchmark harness (`eval/auto_benchmark.py`) replays one of these
against a co-serving server (`--co`) or against an inference-only
baseline (no `--co`), with the FT admission factor and the scheduler
phase tagged into the output suffix (`_factor_<X>_phase_<Y>_<mode>`)
so A/B runs land in distinct CSVs.

**Comparison points:**
- Inference-only baseline (vanilla vLLM serving the same workload).
- Co-serving baseline ("phase=prefill"): the FT-rides-prefill rule
  ported faithfully from the original paper.
- Co-serving unified-phase ("phase=both"): the new admission mode.

There is no head-to-head against LLMStation or FlexLLM yet.

### Q18. Headline measurements

**Verified (real):**

- **Gradient correctness vs PyTorch autograd:** manual SFT backward
  matches autograd to ~1e-7 relative error on synthetic fp32 shapes.
  111/111 graph-parity tests pass (forward graph, FFN-bwd graph,
  attn-bwd graph, all saved-q/k/v variants).
- **Bit-identical inference output** with the co-serving path enabled
  vs disabled on the same greedy prompts (3/3 pre-commit tests).
- **Activation capture verification:** `RMSNorm(final_in) ≈ final_hidden`
  within bf16 tolerance per backward cycle (debug-gated runtime check).
- **Per-cycle backward log** confirms graph vs eager mode, per-cycle
  loss, total tokens trained, epoch progress.
- **The TTFT-spike fix is verified end-to-end.** Before the fix,
  `auto_benchmark.py --co --loose` showed periodic 1.4–1.6s TTFT
  spikes that closed FT admission. After `disable_log_stats: true`
  (auto-defaulted when co-serving is enabled), the spikes go away.

**Planned but not yet measured (pending GPU validation runs):**

- **`forward_interruptible` P99 TTFT reduction.** Code complete.
  Target: residual TTFT outliers should drop from "~80 ms (one full
  FT-only forward)" toward "~30 ms (irrecoverable in-flight kernels
  only)" when the flag is on.
- **Unified-phase vs prefill-only A/B on Llama-3-8B**: the
  `auto_plot_schedulers.py` plotter emits two PNGs
  (`both_vs_inf-only`, `both_vs_prefill`) for visual comparison; the
  numbers aren't yet in the doc.
- **Backward latency improvement from Phase 5.2 + 5.4 (3-graph
  backward) and Phase 5.5 (F1 activation save).** Per-cycle log shows
  graph vs eager; the absolute speedup vs DeltaServe's original
  two-graph design hasn't been published.

The writer should treat anything in "Planned" as preliminary and either
omit from the contributions claim until measured, or qualify carefully
("preliminary; full evaluation in progress").

### Q19. `DeltaServe/eval_plan.md` obsolescence

I don't have direct access to `DeltaServe/eval_plan.md`. Based on what
I know about the port:

**Likely still applicable:**
- TTFT/TBT SLO measurement methodology.
- The three workload timelines (loose, tight, Nutanix).
- The framing of "fine-tuning throughput vs inference latency
  trade-off" as the key axis.

**Likely needs revision:**
- Any experiment that depended on DeltaServe's three separate
  estimators — the port has one merged estimator, so direct
  per-estimator measurements don't translate.
- Any experiment that depended on the unified-pool memory abstraction
  (multi-page-type accounting) — the port doesn't have that.
- Any experiment running against the LoraUnorderedBatchMixed kernel
  — that kernel doesn't exist in the port; vLLM's punica is used
  instead, and per-kernel microbenchmarks won't replicate.
- Any experiment expecting the original's per-tensor AdamW timing —
  the port's fused AdamW changes the optimizer-step latency
  significantly.

**Genuinely new evaluation axes for the port:**
- vLLM-async-scheduling on/off (the port now defaults async on; the
  original didn't have this).
- The unified-phase scheduler ("both") vs the prefill-only scheduler.
- The F1 activation save (`save_attn_qkv`) on/off.
- The proportional-cap factor and the leaky-bucket factor (two new
  admission shapers).

### Q20. Cold-start behavior

**The port still does the offline profiling pass at launch**, same as
the original. Implementation: `profiling_batch_generator.py` generates
representative batch shapes (prefill decomposition, decode `B×K`,
co-serve, mixed); `EngineCore.profile_execution_model` runs them
through the live scheduler before `run_busy_loop` starts; the resulting
records seed both coefficient sets (graph and eager) of the merged
estimator.

Approximate cost on the 5090 test rig: small fraction of total
startup time (most startup time is HF weight load + vLLM model
build). Exact number not measured. Configurable via
`finetune.profile_on_launch` (default True) and
`finetune.profile_num_repeats` (recorded passes per shape after one
unrecorded warmup; default 2).

There is one known follow-up: the offline pass doesn't yet cover the
`(B_d > 0, K > 0, T_in = 0, T_ft > 0)` shape needed for the new
unified-phase admission mode. γ converges for that shape via the 256-
step online refit, but the cold-start window is wider for that mode.

---

## Section 8. Pointers and known limitations (Q21–Q22)

### Q21. Pointers I can read

Inside the DeltaServe-vLLM tree:

- **`CLAUDE.md`** — project context, design constraints, build/precision rules, phased plan summary. Read this first.
- **`INTEGRATION_PROGRESS.md`** — plan + per-phase progress + how each step was verified. The authoritative per-mechanism status doc.
- **`VLLM_FORK_CHANGES.md`** — every change vs upstream vLLM (new files + edits), with "what it does" / "used by". Use this to navigate the fork.
- **`CO_SERVING_OPTIMIZATIONS.md`** — full catalogue of optimizations layered on top of co-serving infrastructure. Grouped by what each optimization targets.
- **`README.md`** — install guide, including the full RTX 5090 setup recipe (CUDA-13.0-in-conda env, FlashInfer JIT).
- **`DeltaServe/CLAUDE.md`** — the original DeltaServe's architecture doc (read-only reference; canonical statement of the original's design).

In-tree code worth reading (in order):

1. `dserve-vllm/vllm/config/finetune.py` — every config field, with paper-relevant docstring.
2. `dserve-vllm/vllm/deltaserve/ft_scheduler.py` — the FT-injecting scheduler.
3. `dserve-vllm/vllm/deltaserve/coordinator.py` — the activation-buffer state machine.
4. `dserve-vllm/vllm/deltaserve/estimator.py` — the merged SLO estimator.
5. `dserve-vllm/vllm/deltaserve/bwd_services/llama3.py` — the manual per-layer SFT backward.
6. `dserve-vllm/vllm/deltaserve/bwd_services/llama3_graph.py` — the CUDA-graph backward.
7. `dserve-vllm/vllm/deltaserve/ft_scheduler_both.py` — the unified-phase scheduler (one-page).

For experimental setup:

- `configs/serving_config_finetuning_llama3.yaml` — the default
  serving config. Every knob has a comment.
- `configs/serving_config_finetuning_llama3_both.yaml` — the
  unified-phase variant (same content, two SLO knobs differ).
- `eval/auto_benchmark.py` — the benchmark replay harness.
- `eval/auto_plot.py` — the single-run plotter (4 panels).
- `eval/auto_plot_schedulers.py` — the scheduler-A/B plotter (2 PNGs).

### Q22. Known limitations / things to keep out of contribution claims

**Inference features the port doesn't yet handle:**

- **Chunked prefill** awareness is not implemented. The current FT
  admission decision computes `T_in` from the waiting queue assuming
  un-chunked prefill. If vLLM is configured with chunked prefill, the
  estimator's prefill-quadratic term may be under-predicted (chunked
  prefill is split across multiple steps, so the per-step quadratic
  cost is smaller). Workload-dependent. The port should be configured
  with chunked prefill disabled for current measurements.
- **Speculative decoding**: untested. The activation-capture hooks
  fire on real forwards, so speculative drafts would not interfere,
  but the SLO estimator's decode coefficients are fit assuming
  non-speculative decode tokens. Configure speculative decoding off
  for current measurements.
- **Multi-TP** (tensor parallel > 1): single-GPU only currently. The
  Worker spawning is uniproc-only. Multi-TP correctness would need one
  backward subprocess per worker rank plus per-rank LoRA-grad
  aggregation. Documented as a known follow-up.
- **Multi-modal models**: untested.
- **Non-Llama-3 architectures**: only Llama-3 has the full per-layer
  manual SFT backward. opt-125m is a frozen loss-only reference path
  (the per-layer backward returns the loss but doesn't run the manual
  per-layer gradient). The activation-capture path (`accumulate.py`)
  auto-detects module names, so other Llama-family models would
  likely work with no code change; non-Llama families would need a
  new backward service.

**Mechanism caveats:**

- **Average-TBT SLO**: configurable (`slo.avg_tbt_slo`) but not
  enforced. Only TTFT and max-TBT are gated. Per-request last-token
  tracking under vLLM's continuous batching is the missing piece;
  documented as a known follow-up.
- **Profiling pass coverage gap for unified-phase mode**: see Q20 — γ
  converges via online refit but the cold-start window for that
  scheduler variant is wider.
- **FT loss divergence in the loose-co eval run** was observed during
  Phase 4d. This is a training-quality issue (LR / corpus / publish
  cadence), not a scheduler issue. The SLO gate is unaffected.
  Investigation deferred.
- **Pre-existing minor leak** in the model runner's `self.requests`
  for FT requests — they're never added to `finished_req_ids`, so
  the per-request `CachedRequestState` lingers. Small (bounded by
  total FT requests admitted), not a correctness issue, documented.
- **Dead-backward deadlock surface**: if the backward subprocess
  crashes, samples claimed for that cycle stay in `_claimed` forever
  → FT admission wedges silently after a one-shot 5 s warning.
  Documented; deliberately deferred.

**vLLM version drift caveat:**

The vLLM tree in the fork is `v0.21.1rc0 + 123 commits`. The original
paper plan referenced `v0.15.1`. File paths in any older design
references will drift; trust the live tree over older path references.
File paths and class names cited in this document are correct against
the live tree at the time of writing.

---

## Section 9. Additional context the writer should have

Things not in the questionnaire but useful for the paper revision.

### 9.1 Framing the contribution shift

The original paper's contributions (as best I can infer from the code)
were roughly:

- SLO-aware FT admission with a closed-form per-step latency model.
- Mixed inference+FT batches via a custom unordered-LoRA kernel.
- A decoupled MPS-partitioned backward subprocess with a cooperative
  per-layer pause contract.
- Backward CUDA-graph capture at fixed `s_max`.
- A unified paged memory pool covering KV, adapters, and FT
  activations.

The vLLM-port retains the first, the third, and the fourth contributions
**at the mechanism level**, with implementation surfaces that match
vLLM's substrate. The mixed-batch contribution morphed into "FT
requests as ordinary scheduler entries with an admission gate" —
mechanically the same idea, expressed against vLLM's continuous-
batching scheduler. The unified pool contribution doesn't apply.

If the paper revision wants to claim new contributions from the port:

- **Async-pipelining-safe co-serving** (queue-wait term, deferred
  GPU-event timing, reserve-at-inject) is genuinely new and required
  by vLLM's `async_scheduling`.
- **Mid-forward inference pre-emption** (`forward_interruptible`) is
  new.
- **The merged six-parameter estimator** is a substantive
  reformulation of the original's three-model approach, forced by
  continuous batching.
- **Unified-phase admission** (FT on any step composition under SLO)
  is new and only enabled by the merged estimator.
- **Three-region backward CUDA-graph design** (adding forward
  recompute + collapsing per-layer attn-bwd to one shared graph) is
  wider than the original's two-region design.

### 9.2 The "what runs on top of what" map

A useful diagram for the paper. The port is roughly:

```
Original DeltaServe contribution      In the port, lives where
─────────────────────────────────     ──────────────────────────
LoRA SFT backward (math)          →   bwd_services/llama3.py
SLO-aware admission gate          →   ft_scheduler.py (gate)
                                       + estimator.py (model)
                                       + coordinator.py (state)
Backward subprocess + MPS         →   backward_process.py + bwd_services/
Per-layer pause contract          →   bwd_services/base.py + llama3.py
Backward CUDA-graph capture       →   bwd_services/llama3_graph.py
Mixed inference+FT batch          →   ft_scheduler.py (injection)
                                       + gpu_model_runner.py edits
                                       + accumulate.py (hooks)
Unified paged memory pool         →   (dropped — vLLM owns KV)
Multi-LoRA inference kernel       →   (dropped — vLLM owns this)
Custom sampler / tokenizer        →   (dropped — vLLM owns this)

vLLM provides (without the port doing anything):
Continuous-batching scheduler     ←   vLLM v1 Scheduler / AsyncScheduler
PagedAttention KV manager         ←   vLLM v1 BlockAllocator
Multi-LoRA forward (punica)       ←   vLLM v1 lora/
OpenAI HTTP API + streaming       ←   vLLM v1 entrypoints/openai/
Sampler + structured outputs      ←   vLLM v1 sampler
Async-pipelined scheduling        ←   vLLM v1 AsyncScheduler
CUDA-graph capture for inference  ←   vLLM v1 cudagraph_dispatcher
```

### 9.3 What the writer can confidently claim is verified

To avoid overclaiming in the paper, here's the line between "verified"
and "in flight":

**Verified by tests + integration runs:**
- Gradient correctness vs autograd (gradcheck 12/12 + 111/111).
- Forward-graph parity vs eager (45/45).
- Bit-identical inference output with co-serving enabled vs disabled.
- The TTFT-spike root cause and the fix.
- The activation buffer reservation accounting under async scheduling.
- The serving + co-serving lifecycle (FT injection → buffer fill →
  backward → optimizer step → published-LoRA visible to inference)
  works end-to-end on Llama-3-8B on the 5090.

**Code complete but GPU A/B pending:**
- `forward_interruptible` end-to-end TTFT-outlier reduction.
- Unified-phase scheduler vs prefill-only scheduler on the loose /
  tight / Nutanix replays.
- Backward latency comparison vs the original DeltaServe's
  two-graph design (the port's 3-graph design should be faster but
  the headline number hasn't been published).

**Estimated, not measured:**
- The F1 (`save_attn_qkv`) ~5 ms speedup per backward.
- The fused-AdamW microsecond-level speedup vs per-tensor AdamW.

The paper should reflect this distinction.

### 9.4 What to expect when running the port

For the writer who wants to reproduce a measurement:

1. Install per `README.md` (conda env `dserve-vllm`, vLLM editable
   install with `VLLM_USE_PRECOMPILED=1`, CUDA-13.0 in conda env on
   the 5090 box).
2. Models cached under `HF_HOME=/mnt/storage/huggingface` with
   `HF_HUB_OFFLINE=1`.
3. Run the benchmark replay:
   ```
   python eval/auto_benchmark.py --loose                         # inf-only
   python eval/auto_benchmark.py --co --loose --scheduler prefill # default
   python eval/auto_benchmark.py --co --loose --scheduler both    # unified
   ```
4. Plot:
   ```
   python eval/auto_plot.py --suffix _co --factor off            # per-mode 4 panels
   python eval/auto_plot_schedulers.py --mode loose              # 2-PNG A/B
   ```

Output suffixes carry the FT factor and the scheduler phase
(`_co_factor_off_phase_prefill_loose`, etc.) so multiple runs land
in distinct files. The benchmark records a wall-clock origin
(`bench_meta<suffix>.json`) so the plotter can anchor the FT throughput
series at the same t=0 as the inference series — that fix is in the
code, the writer doesn't need to do anything.

---

*This document covers everything in `QUESTIONS_FOR_IMPLEMENTER.md` plus
the framing context I think the writer needs to revise the paper
faithfully. For any follow-up that requires reading code, the file
pointers in §8 (Q21) are the right entry points.*
