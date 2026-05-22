# Porting DeltaServe Co-Serving onto vLLM (V1) — Integration Plan

## 0. Purpose & framing

DeltaServe is an **LLM co-serving framework**: it interleaves a LoRA SFT
**backward pass** with ongoing **inference** on the same GPU, running the
backward work in a **separate process** and gated by an **SLO-aware scheduler**
so finetuning soaks up GPU slack without blowing inference TTFT/latency. The
core ideas are:

- Inject finetuning samples into ordinary inference batches, marked so the
  forward pass can tell them apart from inference tokens.
- Capture the finetuning samples' activations during that forward pass into
  pre-allocated buffers.
- Hand those activations to a backward process that trains a dedicated LoRA
  adapter, yielding the GPU back to inference at every layer boundary.
- A scheduler + cost estimator decide, each step, how much finetuning work to
  admit without violating inference SLOs.

The goal of this document is to plan re-implementing that co-serving layer **on
top of vLLM's V1 engine**, developed on **our own fork of vLLM**. The strategic
bet is that vLLM already gives us, for free, the two hardest pieces of
infrastructure DeltaServe had to build by hand:

1. **A production multi-LoRA batching pipeline** (punica/S-LoRA-style kernels,
   adapter pool, per-token adapter dispatch) — equivalent to DeltaServe's
   custom multi-LoRA inference pipeline.
2. **A multi-process engine with a real scheduler and continuous batching** —
   vLLM V1 already runs the scheduler + model executor in a dedicated
   `EngineCore` process, with token-budget continuous batching.

So the port is **not** "move DeltaServe's inference engine to vLLM." It is
"keep DeltaServe's *co-serving value-add* (activation capture, the backward
process, the SLO-aware FT-admission scheduler + estimator) and re-host it on
vLLM's inference + LoRA + scheduling substrate."

### What DeltaServe brings (the value-add to port)

| Component | DeltaServe role | Disposition on vLLM |
|---|---|---|
| Multi-LoRA inference batching | Custom inference pipeline + adapter pool | **Replaced** by vLLM native multi-LoRA |
| Base model + KV memory manager | Custom model + paged KV allocators | **Replaced** by vLLM model + paged KV |
| Two-process launch + GPU buffer sharing + pause event | Spawns the backward process, shares activation buffers, pause-event contract | **Ported** (adapted to vLLM's worker process) |
| Finetuning sample store | Loads + tokenizes FT samples at startup, length-bucketed selection | **Ported** ~as-is (pure Python) |
| FT injection into batches + `finetune_mask` | Adds FT samples to batches and marks their tokens | **Ported** into vLLM scheduler + input prep |
| Activation capture (attn out / FFN out / logits) | Saves FT-only activations into shared buffers during forward | **Re-implemented** as forward hooks on vLLM model |
| Backward SFT service (LoRA grads, optimizer, CUDA-graphed backward) | The actual finetuning backward pass | **Ported** ~as-is (PyTorch, framework-agnostic) |
| SLO-aware scheduler + 3-regime estimator + tracker | Decides FT admission per step under SLOs | **Ported** as a layer on vLLM's V1 scheduler |
| Allocators / occupancy / packed-KV | Custom KV/activation pool management | **Dropped** initially; vLLM owns KV. Revisit only for FT activation pool |

The backward pass itself is the part that is *least* coupled to any particular
inference engine — it operates on saved activation tensors and fp32 LoRA
weights with plain PyTorch + CUDA graphs. That code ports almost verbatim; the
work is in *feeding* it from vLLM.

---

## 1. vLLM target: version and build mode

### 1.1 Version — fork from a recent stable V1 tag

**Recommendation: fork vLLM from a specific recent stable release tag where the
V1 engine is the default and mature (the v0.10.x line or the nearest stable tag
at project start), not from `main`.**

Reasoning:

- **V1 is required and is the default.** The V1 engine (`vllm/v1/`) became the
  default execution path in the v0.8 line and matured through v0.9/v0.10. The
  legacy V0 engine is being removed. Everything below assumes `vllm/v1/`
  internals: `vllm/v1/engine/core.py` (`EngineCore`/`EngineCoreProc`),
  `vllm/v1/core/sched/scheduler.py`, `vllm/v1/worker/gpu_model_runner.py`.
- **Stay on V1 as much as possible.** Build *everything* against the V1 paths
  (`vllm/v1/...`) and avoid touching or depending on the legacy V0 engine
  (`vllm/engine/llm_engine.py`, `async_llm_engine.py`, V0 executor/worker). V0
  is on its way out, so any work tied to it is dead on arrival; the only
  V0-adjacent file we legitimately touch is the shared config entry
  `vllm/engine/arg_utils.py` (`EngineArgs`), which both engines reuse. If a
  feature seems to require the V0 path, treat that as a signal to re-check the
  V1 equivalent before writing code against V0.
- **Fork from a tag, not `main`.** Because we develop on our own fork, we can
  edit vLLM internals directly (subclass where clean, patch source where not) —
  the scheduler, the model runner, and model definitions are not stable public
  APIs and we will be modifying them. Branching from a fixed release tag freezes
  the baseline; rebasing onto a newer upstream tag becomes a deliberate, tested
  step rather than absorbing daily upstream churn.
- **Confirm at project start** which is the latest stable tag whose V1 path
  supports everything we need (multi-LoRA + the model family you target). Verify
  on a smoke test before branching the fork.

Action item for whoever starts this: run `pip index versions vllm` (or check the
releases page), pick the newest stable tag, and create the fork's working branch
off that tag.

### 1.2 Build mode — **Python-only (precompiled), not full source build**

**Recommendation: install vLLM editable against precompiled kernels
(`VLLM_USE_PRECOMPILED=1 pip install -e . --no-build-isolation`, or the
documented "Python-only build" path), and do all DeltaServe work in Python.**

Reasoning — enumerate exactly what we change, and whether any of it is C++/CUDA:

| Change | Layer | Needs kernel recompile? |
|---|---|---|
| Spawn backward process from worker | Python (`vllm/v1/worker`) | No |
| Share GPU buffers via torch multiprocessing / CUDA IPC | Python (torch) | No |
| Allocate FT activation buffers at startup | Python (torch tensors) | No |
| Save attn/FFN/hidden activations for FT tokens | Python (model forward hooks) | No |
| Inject FT samples + `finetune_mask` | Python (scheduler + input prep) | No |
| Dedicated FT LoRA adapter | Python (vLLM LoRA API) — **reuses vLLM's precompiled punica kernels** | No |
| Backward pass (LoRA grads, optimizer) | Python/PyTorch (eager + torch CUDA graph capture) | No |
| SLO scheduler + estimator | Python | No |

Nothing on the list touches vLLM's C++/CUDA. The forward LoRA kernels we *use*
(punica) already ship precompiled in the wheel. The backward pass is hand-rolled
PyTorch + `torch.cuda.CUDAGraph` capture (exactly as in
`SFT_service_graph.py`) — no new C++.

A **full source build** (compiling all CUDA kernels, ~30+ min, needs matching
CUDA toolkit) buys us nothing here and dramatically slows iteration. We only
graduate to a full build if a *future* optimization phase requires editing
vLLM's own kernels (not planned). So: **Python-only build for the whole
project**, full build only if/when we ever fork a vLLM kernel.

Caveat to verify early: `VLLM_USE_PRECOMPILED` pulls a prebuilt kernel image
matching the pinned tag; confirm the precompiled commit matches the source tree
you check out (mismatch → import-time ABI errors). The editable Python tree must
be the same tag the precompiled kernels were built from.

---

## 2. vLLM V1 architecture map (what we're integrating against)

Establish the mental model before phasing the work. In V1:

```
 Frontend (AsyncLLM / LLMEngine, API server process)
        │  ZMQ
        ▼
 EngineCoreProc  (separate process)
        │  Scheduler.schedule() → SchedulerOutput
        ▼
 Executor → Worker(s)  (own the GPU; one per TP rank)
        │
        ▼
 GPUModelRunner.execute_model()  — input prep, forward, sampling
        ├── model forward (model_executor/models/llama.py etc.)
        ├── KVCacheManager / paged blocks
        ├── LoRA: LoRAModelManager + punica kernels (multi-LoRA batching)
        └── CUDA graphs (piecewise via torch.compile + cudagraph)
```

Key insertion points for DeltaServe:

- **Backward process** is spawned from the **Worker** process (that's where the
  model + GPU context live), as a child of the worker — *not* from the engine
  core or frontend. This mirrors DeltaServe spawning the backward service from
  inside `exposed_init_model` (`model_rpc.py:150-176`), which is the
  GPU-owning process.
- **Activation capture** hooks into the **model forward** inside
  `GPUModelRunner` (`model_executor/models/<arch>.py`).
- **FT injection + `finetune_mask`** spans the **`Scheduler`** (decide how many
  FT tokens) and **`GPUModelRunner` input prep** (lay them into the batch, build
  the mask, route them to the FT adapter).
- **Full logits / hidden-state capture for FT** hooks the **logits/sampling**
  stage — by default V1 only materializes last-token logits.
- **SLO scheduler + estimator** wrap / subclass **`Scheduler`**.

### Notable mismatches to design around (flag these up front)

1. **Last-token-only logits.** V1 computes logits only for the positions it
   needs to sample (typically the last token per running request). DeltaServe
   needs *full-sequence* logits for FT samples (`post_layer_infer.py:60-91`
   computes `[N, vocab]` for all FT tokens). **Design choice (recommended):
   don't compute full logits in the forward — save the FT *hidden states*
   (pre-LM-head) into the shared buffer and run the LM head inside the backward
   process.** This keeps the forward's extra work to a memcpy and avoids
   inflating the forward with a `[N_ft × vocab]` matmul on the inference stream.
   (DeltaServe computes logits in-forward; we can improve on that.) Decide in
   Phase 2.

2. **CUDA graphs + activation capture don't mix.** vLLM decode runs under
   piecewise CUDA graphs. Capturing activations (a side-effecting copy to an
   external buffer) inside a captured graph reintroduces exactly the
   pool-aliasing NaN trap documented in `CLAUDE.md` ("Memory / pool gotchas").
   **Invariant to preserve: any batch containing FT tokens runs eager.** This is
   the same hard gate DeltaServe enforces at
   `lora_unordered_batch_mixed.py:171-177` (`not has_ft`). In vLLM terms: when
   the scheduler admits FT tokens to a step, that step is forced out of cudagraph
   mode (eager). FT samples are prefill-only anyway (see #3), so this only costs
   us the graph on the mixed prefill — acceptable and expected.

3. **FT samples are prefill-only.** A finetuning sample goes through the forward
   **once** to produce activations + hidden states, then the backward process
   consumes them; it never enters the decode loop. DeltaServe encodes this by
   stripping the FT request's last token and excluding FT reqs from decode
   counts (`_decode_active_count`). In vLLM we must ensure FT "requests" are
   scheduled for a single prefill step and then retired (not added to the
   running/decode set, no KV held past the step, no sampling output).

4. **Cross-process GPU tensor sharing under spawn.** vLLM uses `spawn`-based
   multiprocessing. Sharing a CUDA tensor across processes then requires CUDA
   IPC (torch does this automatically when a CUDA tensor is sent through a
   `multiprocessing` queue/pipe with the proper start method, or via
   `tensor.share_memory_()` + the reductions in `torch.multiprocessing`).
   DeltaServe relied on fork-style reference passing; on vLLM we go through
   `torch.multiprocessing` reductions (CUDA IPC handles) explicitly. Prove this
   works in Phase 1 before building anything on it.

---

## 3. DeltaServe → vLLM, walked through DeltaServe's own structure

This section is for reading vLLM *through DeltaServe's eyes*. We take
DeltaServe's architecture — the one in `CLAUDE.md` — box by box, and for each
box name the vLLM file/class that plays the same role, what's different, and
where the seams don't line up. Anchor diagram (DeltaServe):

```
 api_server.py                   ── HTTP + ServerConfig load
      │
      ▼
 router/manager.py               ── scheduler loop; picks inference batches,
                                    interleaves backward micro-batches,
                                    enforces co-serving policy
      │
      ▼
 router/model_infer/model_rpc.py ── owns the GPU process; constructs the
                                    forward runner + the backward service
      │
      ├── forward/inference runner (graph-captured decode optional)
      └── backward service (SFT)
```

vLLM's equivalent spine (you'll see these names below):

```
 entrypoints/openai/api_server.py ── HTTP
      │
      ▼
 v1/engine/async_llm.py (AsyncLLM)        ── frontend, in the API process
      │  ZMQ
      ▼
 v1/engine/core.py (EngineCoreProc)       ── separate process: scheduler + executor driver
      │
      ▼
 v1/core/sched/scheduler.py (Scheduler)   ── builds one token-budget batch per step
      │
      ▼
 v1/executor → v1/worker/gpu_worker.py    ── separate process(es): own the GPU
      │
      ▼
 v1/worker/gpu_model_runner.py            ── input prep + forward + sampling
```

> Paths are for the V1 engine around the v0.10 line and **will drift** between
> releases — that's exactly why §1.1 says fork from a pinned tag. Treat them as
> "look here," not gospel.

### 3.1 Box-by-box mapping

| DeltaServe (what you know) | Role | vLLM counterpart | Key difference |
|---|---|---|---|
| `api_server.py` | HTTP entry, loads `ServerConfig`, spawns subprocesses | `vllm/entrypoints/openai/api_server.py` + `vllm/v1/engine/async_llm.py` (`AsyncLLM`) | In vLLM the HTTP layer and the engine live in **different processes** by default, talking over ZMQ. DeltaServe's `api_server.main()` fans config out by spawning; vLLM's `AsyncLLM` launches the `EngineCore` process for you. |
| `dserve/server/config.py` (`ServerConfig` + sections) | One typed config object, validated, fanned out to subprocesses | `vllm/engine/arg_utils.py` (`EngineArgs`) → `vllm/config.py` (`VllmConfig` with `ModelConfig`, `CacheConfig`, `SchedulerConfig`, `LoRAConfig`, `ParallelConfig`, …) | Same idea (one validated config, sub-sections). vLLM splits it into many dataclasses and builds from CLI/`EngineArgs` rather than a single YAML. Our new finetuning knobs become a new sub-config we add on the fork (analogue of your `finetune`/`slo`/`cuda_graph` sections). |
| `router/manager.py` (the `_co_serving_step` loop) | The driving loop: decide prefill vs decode vs FT admission, drive the GPU, drain captures | `vllm/v1/engine/core.py` (`EngineCore.step()` / `EngineCoreProc`) | vLLM's core loop calls `Scheduler.schedule()` then `executor.execute_model()` then `update_from_output()`. **There is no separate "prefill batch then N decode steps"** — see §3.3. Our co-serving loop logic gets folded into this step loop on the fork. |
| `router/mixed_req_queue.py` (`generate_new_batch`, SLO gates) | Decides *what goes in the batch* this iteration | `vllm/v1/core/sched/scheduler.py` (`Scheduler.schedule()` → `SchedulerOutput` in `sched/output.py`) | This is the single most important mapping. vLLM's `Scheduler` is where our FT-admission gate + SLO logic must live. It already does token-budget continuous batching; we extend it to also admit FT tokens. |
| `router/tracker.py` (estimators + `BatchExecutionTracker`) | Predict batch execution time; SLO math | *No equivalent* | vLLM has no execution-time estimator. We **add** this as-is alongside the `Scheduler` on the fork. |
| `router/graph_eligibility.py` (capture mirror) | Tells the scheduler whether a batch will hit a captured graph | *Partial* — vLLM owns cudagraph dispatch internally (`vllm/compilation/`) | vLLM decides graph vs eager itself; there's no manager/runner mirror to keep in sync because scheduler and runner aren't split the same way. Much of this complexity *disappears*; what remains is "is this step eager?" which we control via the FT-runs-eager rule. |
| `router/model_infer/model_rpc.py` | Owns the GPU process; builds the forward runner + backward service; the RPC surface | `vllm/v1/worker/gpu_worker.py` (`Worker`) + `vllm/v1/executor/*` | The vLLM `Worker` is the GPU-owning process. **This is where we spawn the backward process** (§ Phase 1), the same way `model_rpc` does. The executor abstraction handles TP/multi-worker fan-out that DeltaServe does more manually. |
| forward/inference runner | Input layout + forward + (optional) graph | `vllm/v1/worker/gpu_model_runner.py` (`GPUModelRunner.execute_model`) | Combines what DeltaServe splits across `infer_batch.py` (token layout, `b_loc`, mem indices) and the forward path. Our activation-capture hooks and `finetune_mask` plumbing land here. |
| `models/peft/lora_unordered_batch_mixed.py` | Multi-LoRA batched forward, per-token adapter dispatch, **+ FT mask + activation save** | LoRA half → `vllm/lora/` (`models.py` `LoRAModelManager`, `worker_manager.py`, `punica_wrapper/`); forward half → the model in `vllm/model_executor/models/` | This one DeltaServe file does **two jobs** that vLLM splits: (a) multi-LoRA batching — fully replaced by vLLM's LoRA subsystem; (b) the FT-specific bits (mask, activation capture, FT-vs-inference logits) — these have **no vLLM equivalent** and are what we re-implement as hooks. |
| `models/peft/naive_infer_adapter.py` | The adapter weight pool (a_start/a_len, packed A/B) | `vllm/lora/punica_wrapper/` + `vllm/lora/models.py` (the LoRA weight buffers / slots) | vLLM has a mature equivalent (LRU adapter cache, slot management). Replaced wholesale. Our "dedicated FT adapter" is just one more registered LoRA here. |
| `models/{llama,llama3}/` (layer_infer, model.py) | The actual transformer | `vllm/model_executor/models/llama.py` (`LlamaModel`/`LlamaForCausalLM`) | Replaced by vLLM's model definitions. We subclass/patch the arch model to add forward hooks for activation capture (FT only). |
| `infer_batch.py` (`InferBatch.init_batch`, token layout) | Logical batch → GPU tensor layout (`input_ids`, `b_start_loc`, `b_seq_len`, `finetune_mask`) | `vllm/v1/worker/gpu_model_runner.py` input-prep + `vllm/v1/core/sched/output.py` | vLLM builds the flattened token tensors and metadata in the model runner from `SchedulerOutput`. Our `finetune_mask` gets built here. |
| `common/unified_mem_allocator.py`, `packed_kv_mem_allocator.py`, `allocator_factory.py` | KV pool **and** FT activation/logit buffers | KV half → `vllm/v1/core/kv_cache_manager.py` + `vllm/v1/core/block_pool.py` | vLLM owns paged KV entirely — we drop DeltaServe's KV allocators. **But vLLM's KV manager does *not* manage FT activation buffers** — that part has no equivalent, so we keep a small dedicated activation-buffer allocator (the shared buffers from Phase 1/2). Note: `packed_kv`'s reason to exist (GQA KV oversizing) is moot — vLLM stores KV at `num_kv_heads` natively. |
| `common/cuda_graph_runner.py` | Forward CUDA-graph capture/replay + bucketing | `vllm/compilation/*` + the model runner's piecewise cudagraph (via `torch.compile`) | vLLM captures graphs through its compile backend, not a hand-rolled bucket cache. We don't port this; we only need the rule "FT steps run eager." |
| `models/{llama,llama3}/SFT_service*.py` (backward) | The LoRA SFT backward pass + backward CUDA graphs | *No equivalent* | vLLM is inference-only. The backward service is **pure DeltaServe**, ported ~verbatim into the backward process (it's framework-agnostic PyTorch). |
| `router/finetuning_store.py` | Loads/tokenizes FT samples at startup; length-bucketed selection | *No equivalent* | **Added** ~as-is. |

### 3.2 The two big "shapes" to internalize

1. **DeltaServe splits "loop control" from "batch contents"; vLLM splits
   "scheduling" from "execution."**
   - DeltaServe: `manager.py` runs the loop and *calls* `mixed_req_queue.py` to
     fill the batch; both run in the router process, and `model_rpc.py` owns a
     *separate* GPU process reached by RPC.
   - vLLM: `Scheduler` (decides) and `GPUModelRunner` (executes) are the split,
     and they sit in **different processes** (`EngineCore` vs `Worker`)
     connected by the executor. So "what goes in the batch" (your
     `mixed_req_queue`) and "lay out the tensors + forward" (your `infer_batch`
     + forward runner) end up on **opposite sides of a process boundary** in
     vLLM. Our FT logic therefore splits the same way: admission decision in the
     `Scheduler`, mask/capture in the `GPUModelRunner`.

2. **Where the backward process attaches is the same in both.** In DeltaServe
   the backward service is spawned by `model_rpc.py` — the GPU-owning process.
   In vLLM the GPU-owning process is the `Worker`. So the mental substitution is
   simply **`model_rpc.py` → `gpu_worker.py`**: that's the process that spawns
   our backward child and shares the activation buffers with it.

### 3.3 vLLM things that don't exist in DeltaServe (so they may surprise you)

- **No prefill/decode phase split.** DeltaServe explicitly does "one prefill
  batch, then a run of decode steps." vLLM V1 has **one unified step**: each
  `schedule()` builds a single token-budget batch that can mix prefilling and
  decoding requests, and **chunked prefill** splits a long prompt across steps.
  Our co-serving invariants (FT runs eager, FT is prefill-only) must be
  expressed in this unified model, not the two-phase one.
- **Automatic prefix caching.** vLLM hashes KV blocks and reuses them across
  requests. DeltaServe has nothing like it. Mostly orthogonal to us, but it
  means KV blocks have a lifecycle (hashing, eviction) we shouldn't fight.
- **Preemption / recompute.** vLLM can evict a running request and recompute it
  later under memory pressure. FT "requests" must be designed so they're never
  left half-scheduled across a preemption (they're single-step prefill anyway).
- **ZMQ multiprocess frontend↔core.** DeltaServe's API and scheduler are closer
  together; in vLLM there's a serialization boundary between the HTTP layer and
  the engine. Our finetuning control surface (start/stop FT, report loss) has to
  cross it or live entirely engine-side.
- **`torch.compile` + piecewise cudagraph.** vLLM compiles the model. This is
  the deep reason FT batches must run **eager**: capturing side-effecting
  activation copies inside a compiled/graphed region reintroduces the pool
  aliasing problem (see §2 mismatch #2).
- **A real executor abstraction (TP/PP, Ray or mp).** DeltaServe wires TP more
  by hand. vLLM's executor fans a step out to all workers; our backward process
  story has to be per-worker (per GPU), mirroring how captures are lock-step
  under TP in DeltaServe.

### 3.4 DeltaServe things that don't exist in vLLM (what we are actually adding)

Everything in this list is net-new code on the fork — vLLM has no hook for it:

- The **backward/SFT process** and its CUDA-graphed backward (`SFT_service*.py`).
- **Activation + logit/hidden-state capture** for FT tokens during the forward.
- The **`finetune_mask`** and FT-vs-inference token distinction.
- **FT sample injection** into the scheduler's batch + the **finetuning store**.
- The **SLO-aware FT-admission scheduler logic + the cost estimator/tracker**.
- The **cross-process shared activation buffers** + **pause-event** co-serving
  contract + the **MPS partition split**.

This is the precise list the phased plan below builds, in order.

## 4. Phased build plan

The phases follow the user's proposed sequence, each ending in an independently
**testable** state. Do not start a phase until the previous one's test passes.

### Phase 1 — Backward process + shared-memory IPC (no backward logic)

**Goal:** stand up a second GPU process spawned at the right point in vLLM
startup, and prove we can share a GPU buffer with it. No SFT math yet.

Steps:

1. **Find the launch spot.** In the vLLM **Worker** init (after the model and
   CUDA context are up, analogous to the end of DeltaServe's
   `exposed_init_model`), spawn a child process. Add a config flag (e.g.
   `--enable-finetuning`, plumbed through `EngineArgs`/`VllmConfig`) gating it.
   The backward process is a plain `multiprocessing.Process` target (a stub that
   loops on a pipe), `daemon=True`.
2. **MPS env wrapping.** Reproduce `model_rpc.py:172-178`: set
   `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=10` and `CUDA_DEVICE_MAX_CONNECTIONS=1`
   immediately before `.start()`, pop them after, so only the child inherits the
   constrained MPS partition.
3. **Shared buffer.** Allocate one GPU tensor in the worker, hand it to the child
   via `torch.multiprocessing` (CUDA IPC). Child maps it.
4. **IPC handshake.** Reproduce DeltaServe's primitives:
   - `mp.Event()` for pause/resume (the `_maybe_pause` mechanism).
   - a `Pipe`/`Queue` for work handoff (worker → child) and result return
     (child → worker).
5. **Test (the deliverable):** worker writes a known tensor to the shared buffer;
   child reads it, computes a hash, sends the hash back; worker verifies it
   matches `torch.hash`/checksum of what it wrote. Flip a few values, repeat.
   This proves the GPU memory is genuinely shared (not copied) across the process
   boundary.

Exit criteria: backward stub process launches on `--enable-finetuning`, MPS vars
applied only to child, shared-buffer hash round-trips correctly, clean shutdown.

### Phase 2 — Activation capture + FT injection + dedicated FT adapter

**Goal:** during real inference, inject FT samples into batches, mark their
tokens, route them to a dedicated LoRA adapter, and save their activations into
the Phase-1 shared buffers. Verify the captured activations are correct by hash,
**without** doing any backward yet.

Steps:

1. **Finetuning sample store.** Port `finetuning_store.py`'s `FinetuningManager`
   (load + tokenize at startup, length-bucketed selection via `pop_best_under`).
   It's pure Python; minimal change. Load path plumbed via config
   (`finetune.data_path` equivalent).
2. **Dedicated FT adapter.** Register a finetuning LoRA adapter at startup
   through vLLM's LoRA API (alongside any inference adapters). Plumb a config
   param for its path. This is the analogue of `model_rpc.py:137-146`
   (`finetuning_adapter = adapters[-1]`, `is_finetuning=True`). vLLM's multi-LoRA
   dispatch will route any request tagged with this adapter id through it — we
   get S-LoRA batching for free.
3. **FT injection in the scheduler (fixed for now).** In a `Scheduler` subclass,
   after the normal inference batch is built, append a *fixed* number of FT
   tokens / one FT sample per step (no SLO logic yet — that's Phase 4). Tag those
   requests `is_finetuning=True` and assign them the FT adapter id. Ensure they
   are **single-step prefill** and retired after the step (mismatch #3).
4. **`finetune_mask` plumbing.** Build a per-token boolean mask in
   `GPUModelRunner` input prep (analogue of
   `lora_unordered_batch_mixed.py:152-163`) and stash it on the forward context
   (vLLM `ForwardContext` / attention metadata) so the model forward can see it.
5. **Force eager for FT batches.** When the step contains FT tokens, disable
   cudagraph for that step (mismatch #2).
6. **Activation buffers.** Allocate, at startup, the shared activation buffers
   sized by `max_saved_finetuning_tokens`: per-layer attention output and FFN
   output (and embedding output), shaped `[max_saved_finetuning_tokens,
   hidden]`. These are the Phase-1 shared buffers, now real. Mirror
   `unified_mem_allocator.py:303-324`.
7. **Capture hooks.** Hook the model forward (subclass the arch model, or
   register forward hooks on the attention and MLP submodules) to copy
   `output[finetune_mask]` into the per-layer shared buffer
   (`save_activations_by_layer` analogue,
   `lora_unordered_batch_mixed.py:394-421`). Save **only** FT-token rows.
8. **FT hidden states instead of full logits (recommended).** Save the final
   pre-LM-head hidden states for FT tokens into a shared buffer; the LM head runs
   later in the backward process. (Alternative: compute full logits in-forward
   like `post_layer_infer.py:60-91`. Pick the hidden-state path unless a
   correctness reason forces otherwise — decide here.)
9. **Test (the deliverable):** run inference with FT injection on. In the worker,
   compute a reference hash of the activations for the FT tokens (recompute from
   the same inputs, or hash the buffer slice). Send to the backward stub, which
   independently hashes its view of the shared buffer. Assert equality across the
   process boundary. Also assert: inference outputs for the non-FT requests are
   **unchanged** vs a no-finetuning baseline (FT injection must not corrupt
   inference results — guard the eager/graph and mask logic).

Exit criteria: FT samples flow through real batches via the dedicated adapter,
their (and only their) activations land in shared buffers, hashes match
cross-process, and inference correctness for real requests is unaffected.

### Phase 3 — Real backward pass (working co-serving, no scheduling)

**Goal:** the backward process actually trains the FT adapter from the captured
activations. End state: a functioning co-serving system with *naive* (fixed) FT
admission.

Steps:

1. **Port the SFT backward service.** Bring over `SFT_service.py`
   (`LlamaSFTBackwardService`) and the GQA subclass
   (`llama3/SFT_service.py`), plus `SFT_service_graph.py`
   (`GraphedBackwardRunner`). These operate on the `Activations` object (cloned
   slices of the shared buffers) + fp32 LoRA weights + logits/targets — all
   framework-agnostic PyTorch. The main edits are wiring the *inputs* to come
   from the vLLM-side buffers and the FT sample metadata (token counts, target
   ids) rather than from DeltaServe's existing engine structures.
2. **fp32 LoRA weights to the backward process.** Hand the FT adapter's weights
   to the backward process as a persistent fp32 copy (analogue of
   `lora_adapter.py:load_gpu_fp32_dict`, `model_rpc.py:179`). Keep the
   fp32-master / fp16-serve split. Honor the **fp32 LM head / final norm**
   precision rule (memory note `feedback_lm_head_fp32`,
   `SFT_service.py:321`) and the **fp32 `scores` matmul** rule for GQA attention
   backward (`CLAUDE.md` "Precision rule for llama3 attention").
3. **LM head in backward (if Phase 2 chose hidden-state capture).** Run the LM
   head over the saved FT hidden states inside the backward process to get
   logits, then cross-entropy against the target tokens
   (`get_logits_and_targets` analogue, `SFT_service.py:240-269`).
4. **Pause/resume contract.** Wire `_maybe_pause()` (`SFT_service.py:159-165`,
   `236`) to the Phase-1 `mp.Event`. The vLLM worker `.clear()`s the event around
   each inference step so backward yields the GPU at every layer boundary —
   preserving the core co-serving contract. Backward runs on its own
   `self.bwd_stream`; keep the `torch.cuda.synchronize()`-before-wall-clock
   timing discipline (`CLAUDE.md` co-serving contract).
5. **Backward CUDA graphs (optional within this phase).** `SFT_service_graph.py`
   captures the FFN/post-layer backward and (optionally) padded attention. These
   graphs live entirely in the backward process and are independent of vLLM's
   graphs. Bring them once eager backward is correct; honor the persistent-buffer
   rules for LoRA `.grad` and attention `ctx` (`CLAUDE.md` "Memory / pool
   gotchas") so graph-pool aliasing doesn't corrupt grads.
6. **Test (the deliverable):** run co-serving with fixed FT injection; verify FT
   loss decreases over steps (training actually works), and that inference TTFT/
   latency is degraded but bounded (the GPU is shared but not deadlocked). Sanity
   check adapter weights change. Compare a short loss curve against the existing
   DeltaServe implementation on the same data as a correctness anchor.

Exit criteria: loss converges on the FT adapter while inference continues
serving, pause/resume keeps both alive, no NaNs in inference logits (the
classic graph-pool-aliasing failure mode).

### Phase 4 — SLO-aware scheduler + estimator

**Goal:** replace the fixed FT injection with DeltaServe's SLO-aware admission so
backward work fills GPU slack without blowing TTFT/TBT SLOs.

Steps:

1. **Port the estimator + tracker.** Bring `tracker.py`
   (`PrefillExecutionEstimator`, `DecodeExecutionEstimator`,
   `BatchExecutionTracker`) over. Feed it vLLM batch features (Σnᵢ², Σnᵢ, T_ft,
   batch size, KV size) and measured per-step durations. The math is
   framework-independent.
2. **Graph-regime awareness.** vLLM's cudagraph dispatch differs from
   DeltaServe's bucket cache, but the *concept* (graph vs eager vs first-touch
   capture cost) maps onto vLLM's piecewise-cudagraph dispatch. Decide how much
   of the three-regime estimator + `graph_eligibility` mirror is worth porting
   vs. simplifying: for co-serve batches DeltaServe *always* uses the eager
   regime anyway (the co-serving invariant), so the FT-admission path mostly
   needs the eager prefill estimator + decode estimator. The graph/capture
   regimes matter only for predicting the *inference-only* steps' timing. Start
   with eager-only estimation; add graph-regime modeling only if SLO prediction
   is too pessimistic.
3. **FT-admission gate in the scheduler.** Port `generate_new_batch`'s FT loop
   (`mixed_req_queue.py:334-378`) and `max_next_ft_tokens` /
   `check_will_starve`: each step, compute the max FT tokens admissible under the
   TTFT / avg-TBT / max-TBT SLOs given the predicted inference cost, and admit up
   to that (constrained also by `max_saved_finetuning_tokens` activation budget
   and KV). Implement on the `Scheduler` subclass from Phase 2.
4. **Stats dump.** Optionally port `BatchExecutionTracker`'s per-batch
   predicted-vs-actual CSV (`scheduler.batch_prediction_stats_path`) for tuning.
5. **Test (the deliverable):** run a workload with an SLO target; verify TTFT SLO
   satisfaction stays near target while FT throughput (tokens/s) is non-trivial,
   and that FT admission backs off under inference bursts. Reuse the
   `auto_benchmark.py` / `auto_plot.py` style measurement (TTFT CDF, latency over
   time, FT tokens/s, SLO satisfaction rate) adapted to the vLLM server.

Exit criteria: SLO satisfaction comparable to the current DeltaServe at similar
FT throughput, with FT admission demonstrably responsive to inference load.

### Phase 5 (later) — optimizations & assets

Pull these in only after Phases 1–4 are solid:

- Backward CUDA graphs fully tuned (padded attention path, sizing via
  `analyze_finetuning_data.py` / `keep_p95.py`).
- A dedicated FT **activation pool** if vLLM's allocator gets in the way (the
  packed-KV / occupancy-tracker work is tied to DeltaServe's own allocators and
  likely *not* needed — vLLM owns KV; we only manage the separate FT activation
  buffers).
- Eval/analysis tooling port (`auto_benchmark`, `auto_plot`, comparison plots).
- Multi-TP correctness (DeltaServe captures are lock-step under TP; verify the
  backward process story under vLLM's TP worker layout — backward likely lives
  per-rank).

---

## 5. Top risks / things to resolve early

1. **Cross-process CUDA tensor sharing under vLLM's `spawn`** — prove in Phase 1.
   If torch CUDA IPC across `spawn` is fragile in the vLLM worker context, the
   fallback is to spawn the backward process via `fork` from the worker
   specifically (the worker already has the CUDA context), or to use explicit
   `cudaIpcGetMemHandle` plumbing. Resolve before anything else.
2. **Editing vLLM internals on the fork.** The `Scheduler`, `GPUModelRunner`,
   and model definitions are not stable public APIs. Prefer subclassing where
   the seams allow it and patch source narrowly where they don't; keep our
   changes localized so rebasing onto a newer upstream tag stays tractable.
3. **Forward-hook activation capture vs. torch.compile.** vLLM compiles the model
   forward. Forward hooks on submodules may not fire as expected inside a
   compiled/graphed region. Mitigated by the FT-batches-run-eager invariant — but
   confirm hooks fire in eager mode and that compile is actually bypassed for FT
   steps. If hooks are unreliable, subclass the arch model and capture inline.
4. **Full-logits vs hidden-state capture decision** (mismatch #1) — settle in
   Phase 2; it changes both the forward capture and the backward LM-head wiring.
5. **FT requests must not leak into decode / KV / sampling** (mismatch #3) — get
   the single-step-prefill-then-retire lifecycle right in Phase 2, or FT samples
   will hold KV blocks and emit spurious sampler output.

---

## 6. One-paragraph summary

Re-host DeltaServe's co-serving layer on **vLLM V1**, developed on **our own
fork** branched from a recent stable tag, using a **Python-only / precompiled
build** (nothing we add touches C++/CUDA). vLLM's native multi-LoRA replaces
DeltaServe's custom multi-LoRA pipeline and its multi-process engine replaces
DeltaServe's existing router. We **port** the genuinely novel parts:
the backward SFT process + GPU-buffer sharing + pause contract, the finetuning
sample store, FT-token injection with a `finetune_mask` and a dedicated FT
adapter, activation capture into shared buffers, the actual LoRA backward pass,
and finally the SLO-aware FT-admission scheduler + cost estimator. Build it in
four testable phases — (1) backward process + shared-memory hash test, (2)
activation capture + FT injection verified by cross-process hash with inference
correctness preserved, (3) real backward = working co-serving with fixed
admission, (4) SLO-aware scheduling — gated by the hard invariant that **any
batch containing FT tokens runs eager** to avoid the CUDA-graph pool-aliasing
NaN trap.
