# Integration Progress & Plan — DeltaServe co-serving on vLLM

Single source of truth for **what we're building and how far we've got** (the former
standalone `VLLM_INTEGRATION_PLAN.md` was merged in here). Companion docs: `CLAUDE.md`
(architecture, DeltaServe→vLLM box mapping, build/precision rules) and
`VLLM_FORK_CHANGES.md` (every file changed vs upstream, with what/why).

Legend: ✅ done & verified · 🟡 implemented, runtime-verify pending · ⬜ not started

## Goal

Re-host DeltaServe's **co-serving value-add** on vLLM's V1 engine: inject LoRA-SFT
finetuning samples into ordinary inference batches, capture their activations during the
forward, hand them to a **backward subprocess** that trains a dedicated FT LoRA adapter,
and (later) an **SLO-aware scheduler** that admits FT work into GPU slack without blowing
inference latency. vLLM provides the two hardest pieces for free (production multi-LoRA
batching + a multi-process engine with a real scheduler), so we port only the co-serving
layer, not an inference engine. Box-by-box DeltaServe→vLLM mapping lives in `CLAUDE.md`.

## Design constraints / invariants (load-bearing — design around these)

1. **Any batch with FT tokens runs eager.** Capturing side-effecting activation copies
   inside a piecewise CUDA graph reintroduces the pool-aliasing NaN trap, so FT steps
   force eager (`skip_compiled`). DeltaServe's gate: `lora_unordered_batch_mixed.py:171-177`.
2. **FT samples are prefill-only, single-step-then-retire.** One forward to produce
   activations; never enter decode, hold no KV past the step, emit no sampler output
   (`FinetuneScheduler` frees their KV before the base loop → invisible to the frontend).
3. **Last-token-only logits.** V1 only materializes logits for sampled positions, so we
   save FT **hidden states** (pre-LM-head `final_hidden`) and run the LM head in the
   backward process.
4. **Cross-process GPU sharing under `spawn`** needs explicit CUDA IPC
   (`torch.multiprocessing` reductions), not fork-style reference passing.
5. **Precision (DeltaServe SFT rule).** fp32 LM head + final norm + logits/softmax/CE +
   GQA attention `scores` matmul; fp32 LoRA master / fp16 compute copy; RMSNorm weights
   fp32. Full table in the auto-memory `deltaserve-backward-precision`.

## Runtime pipeline — what happens when `scripts/ft_experiment_{opt,llama3}.py` runs

1. **Launch.** The harness loads the YAML, builds `vllm serve <model> …` (engine args +
   `--finetune-config.*` flags) and starts the OpenAI server in its own process group,
   then fires one short completion every `--interval` s for `--iterations`.
2. **Engine startup.** `VllmConfig.__post_init__` selects `FinetuneScheduler` (because
   `enable_finetuning`). In the **Worker** (GPU process) `init_device`, `BackwardProcess`
   spawns the **backward subprocess** with a child-only MPS partition; the child runs
   `bwd_services.base.service_main` → `get_service(arch)` →
   `OPTBackwardService` / `Llama3BackwardService`.
3. **`load_model`.** The worker shares, via CUDA IPC (zero-copy): the frozen base weights,
   an fp32 copy of the FT adapter, and a small `meta` dict (lm_head key, vocab size,
   logit scale, rms_norm_eps, norm/embed weight keys). Then it builds the
   `FinetuneAccumulator` — for llama3 it auto-detects `input_layernorm`/`model.norm` and
   registers residual-stream `forward_pre_hook`s; for opt it finds none and captures only
   `final_hidden` — shares the buffers zero-copy, and creates the `FinetuneCoordinator`.
4. **Schedule (per step).** `FinetuneScheduler.schedule()` does normal inference
   scheduling; if real work is present and admission is open, it injects FT requests
   (`max_tokens=1`, FT adapter id, `is_finetuning=True`) up to `coordinator.next_ft_budget()`
   and records their ids in `SchedulerOutput.finetune_req_ids`.
5. **Forward.** `GPUModelRunner` builds the `finetune_mask` + per-sample lengths, forces
   the step eager, runs the forward. llama3 pre-hooks copy each layer's residual-stream
   input (`layer_in[i]`) + the pre-final-norm residual (`final_in`) for FT rows; after the
   forward, `accumulate_final` copies post-norm `final_hidden` + `concat_input_ids`. FT
   requests are retired the same step (KV freed, no output).
6. **Coordinator.** `record_capture(n, sample_lens)` advances the fill offset; when the
   buffer fills it `cuda.synchronize()`s and signals the child
   (`notify_buffer_full(n, sleep, sample_lens)`), closing FT admission.
7. **Backward child.** `process_activations` reconstructs fp32 logits from `final_hidden`
   via the shared LM head (chunked over vocab to bound memory), computes per-sample
   shift-by-1 CE loss + the CE logit gradient, prints them (llama3 also runs
   `verify_activations`), then zeroes the buffers, sleeps `backward_sleep_seconds`
   (**simulated** backward — the real LoRA backward is the next slice), and replies.
8. **Reopen.** Next step, `poll_backward` sees the reply, resets the fill offset +
   `sample_lens`, reopens FT admission. Real completions stay byte-identical to a no-FT
   baseline throughout.

## Next step

**State on 2026-09-08 (end of the M5 session).** Everything below is in the working
tree, uncommitted, on branch `tp`. **M5 — backward CUDA graphs under TP — has landed**
and is gated on the two 5090s with real NCCL (`tests/test_tp_backward_graph_nccl.py`
1196/1196, `tests/test_tp_trainer_graph_nccl.py` 132/132); every tp=1 gate re-passes
unchanged. Qwen3-14B at TP=2 co-serves the real timeline workloads with SLO-gated FT
admission; the estimator relay (M4.2) is GPU-validated.

**Next session, in priority order:**

0. **Act on the step-trace findings** (see "Estimator validation under TP (2026-09-08,
   night)" under Phase 7): the predictor is accurate; the misses are backward-child
   interference on decode-only steps (TBT) and on paused prefills (pause is late).

1. **M5 live A/B — graph vs eager under TP=2 on the real models.** Code + 2-GPU gates are
   done (see "M5 — backward CUDA graphs under TP ✅" under Phase 7). Run
   `python eval-tp/ft_bench_tp.py --family llama3 --tp 2 --duration 60 --kill-stale` twice,
   with `backward_cuda_graph: true` (now the YAML default) and `false` in
   `configs/serving_config_finetuning_llama3_tp2.yaml` (the Qwen3-14B YAML has it on too).
   The first run of the session used the YAML's old `false` and correctly reported
   `(eager)`. Expect from EACH child `[bwd-graph] pre-captured forward 32/32 + FFN 32/32 +
   attn 1/1`, `(graph)` in the cycle lines, cycle loss equal to the eager run
   cycle-for-cycle (same data), and ~5 ms/cycle less. Both TP YAMLs also turn on
   `save_resid_mid` (P5.6) — expect `resid-mid pre-hooks on 32` in the `[accumulate]`
   line and 192 collectives per cycle instead of 224. Then the same for
   `--family qwen3-14b`.
2. ~~**M4.3 — gradient bucketing + comm stream + rank-symmetric clip.**~~ Landed
   2026-09-08 — see "M4.3 — gradient bucketing + comm stream ✅" under Phase 7.
3. ~~**`forward_interruptible` under TP.**~~ Landed 2026-09-08 — see "forward_interruptible
   under TP (rank-symmetric tier C)" under Phase 7 for the design, the gates and the tight-
   trace A/B.
4. **Baselines + estimator residuals.** Inference-only baselines are DONE for Qwen3-14B
   and Llama-3 TP=2 on loose / tight / nutanix-600-800 (tables under "Validation
   2026-09-08 (evening)" in Phase 7). Still open, next session: one
   `validate_estimator: true` TP=2 replay vs a tp=1 run — compare per-regime RMSE (a
   residual growing with `t_in` would be the only reason for a TP term in the formula).
5. ~~**Llama-3 `rope_theta` re-verification**~~ and ~~**Qwen3-0.6B smoke**~~ — both DONE
   2026-09-08: the DIAG remat error is at bf16 noise and no longer grows (was 0.22 → 0.45),
   `pure_ft_bench` passes 2.10 at cycle ~30 and keeps descending (863 cycles, 2097 FT
   tok/s, all `(graph)`); the 0.6B smoke trains once the worker resolves the tied
   `lm_head` through the LoRA-wrapped embedding (see "Validation 2026-09-08 (evening)").
6. Small items: none open from this list — Qwen3's exact `save_attn_qkv` (pre-norm q/k
   hooks) and the `N/?` progress meter under TP (corpus total on the relayed trigger)
   both landed 2026-09-08.

**Phase 7 (TP=2) remains the active line.** M0–M4.2 are done and GPU-validated on 2× RTX
5090.

## Phase 1 — Backward process + shared-memory IPC

> Goal: stand up a second GPU process spawned at the right point in vLLM
> startup, and prove we can share a GPU buffer with it. No SFT math yet.

### Step 1 — `--enable-finetuning` config flag ✅

The master gate for the whole co-serving feature. Plumbed through vLLM's config
system as a dedicated `FinetuneConfig` sub-config (the analogue of DeltaServe's
`finetune.*` YAML section), so later phases have a clean home for `data_path`,
`finetuning_lora_path`, `learning_rate`, `max_saved_finetuning_tokens`, SLOs, etc.

**Files changed** (all in `vllm/`, follows the existing `profiler_config` pattern):

| File | Change |
|---|---|
| `vllm/config/finetune.py` *(new)* | `FinetuneConfig` dataclass; field `enable_finetuning: bool = False` |
| `vllm/config/__init__.py` | export `FinetuneConfig` (import + `__all__`) |
| `vllm/config/vllm.py` | import + attach `finetune_config` field to `VllmConfig` |
| `vllm/engine/arg_utils.py` | 3 sites: import, field declaration, `--finetune-config` CLI arg, pass into `VllmConfig` ctor |
| `vllm/deltaserve/__init__.py` *(new)* | `deltaserve/` package — home for all our net-new code; provides `dprint()` (green, `[deltaserve]`-prefixed, TTY-guarded) so our runtime signals stand out in vLLM's logs |
| `vllm/v1/worker/gpu_worker.py` | read flag + `dprint` at end of `init_device()` |

**Runtime signal** (green via `dprint`):
`[deltaserve] Worker rank=R local_rank=L init_device done | enable_finetuning=<bool>`
printed once per worker during startup. All future DeltaServe prints route through
`vllm.deltaserve.dprint` for consistent green output.

**Why this design:** a sub-config (not a loose top-level bool) mirrors DeltaServe's
config layout and gives Phases 2–4 a place to grow. The Worker print is placed at
the end of `init_device()` because that is the GPU-owning process, after the CUDA
context is up — exactly where Step 2 will spawn the backward child.

**Verification:**
- CPU-only (config + CLI plumbing): `tests/test_phase1_step1.py` ✅ passing
- GPU runtime print: ✅ verified in full-system launches (`[deltaserve] Worker ... enable_finetuning=True`)

### Step 1b — DeltaServe YAML config + loader layer ✅

A DeltaServe-style sectioned YAML for reproducible test runs, plus an additive
loader that maps it onto vLLM's `EngineArgs` — **no edits to upstream vLLM code**.

**Files added:**

| File | Purpose |
|---|---|
| `configs/serving_config_finetuning.yaml` | sectioned config: `finetune` (enable + MPS %), `server` (host/port/rank_id), `model`/`engine`/`parallel` (→ EngineArgs) |
| `vllm/deltaserve/config_loader.py` *(new)* | `load_yaml_config()`, `split_config()`, `build_engine_args()`, `engine_args_from_yaml()` |
| `vllm/config/finetune.py` *(edit)* | added `backward_mps_percentage: int = 10` (used when spawning the backward proc in Step 2) |

**Mapping rules:** `finetune` → `FinetuneConfig`; `server` → returned dict (host/port/
rank for the API server); every other section is a bag of `EngineArgs` field names,
merged and passed straight through (add any vLLM knob without touching the loader).
Unknown finetune keys rejected by pydantic; non-dict sections / missing files raise
clear errors.

**Verification:** `tests/test_config_loader.py` ✅ 12/12 passing (CPU-only, no model load).

### Tooling — OPT-125m toy LoRA adapters ✅

Test assets for the multi-LoRA / FT-adapter paths, using the `facebook/opt-125m`
tester model instead of Llama-3.

| File / dir | Purpose |
|---|---|
| `train_opt125m_lora.py` *(new)* | trains tiny PEFT LoRA adapter(s) on opt-125m; short run on an inline corpus |
| `adapters/opt125m-toy-lora/` *(generated)* | inference adapter |
| `adapters/opt125m-toy-lora-ft/` *(generated)* | finetuning-target adapter |

Config mirrors the existing `adapters/llama3-toy-lora*` (r=16, α=32, dropout=0.05,
bias=none, CAUSAL_LM) but with **OPT module names and FFN included**:
`q_proj, k_proj, v_proj, out_proj` (attention) + `fc1, fc2` (FFN). (The llama3 toy
adapters target attention only.)

**Verified:** loss dropped ~6.5 → ~0.18 over 60 steps; loading base+adapter and
generating "The capital of France is" → base "the French Republic", adapter "Paris".
Requires `peft` (installed into the env). Run with `HF_HOME=/mnt/storage/huggingface
HF_HUB_OFFLINE=1`.

### Step 2 — Spawn backward stub process + MPS env wrapping ✅

Spawn a `daemon=True` backward child from the Worker, gated on
`enable_finetuning`, with the MPS env applied to the child only. Child = a stub
that handshakes 'ready', then loops on a pipe answering ping/shutdown. No CUDA /
SFT yet. (DeltaServe ref: `model_rpc.py:150-178`.)

**Key finding (drove the design):** the EngineCore process (`vllm/v1/engine/utils.py:144`)
is **non-daemon**, and single-GPU uses `UniProcExecutor` which runs the worker
*in-process* — so a `daemon=True` child is allowed. But the multiproc executor
(TP>1) makes `WorkerProc` **daemonic** (`multiproc_executor.py:680`), and Python
forbids daemonic processes from having children. → Step 2 targets single-GPU and
**guards** with a clear error if spawned from a daemonic process (multi-TP = Phase 5).

**Files changed:**

| File | Change |
|---|---|
| `vllm/deltaserve/backward_process.py` *(new)* | `_backward_stub_main` (child entry) + `BackwardProcess` handle (`start`/`ping`/`shutdown`); spawn ctx, MPS env wrapping, daemon-process guard |
| `vllm/config/finetune.py` *(Step 1b)* | `backward_mps_percentage` (consumed here) |
| `vllm/v1/worker/gpu_worker.py` | after the Step-1 print, if `enable_finetuning`: construct + `start()` a `BackwardProcess`, store on `self.backward_process` |

**Runtime signals (green):** `[backward] spawning child with CUDA_MPS_..=N` /
`[backward] stub started pid=.. inherited CUDA_MPS_..=N` / `[backward] child ready ..`

**Design notes:**
- **spawn** context (not fork) — CUDA-safe and matches vLLM, so Step 3's CUDA-IPC
  buffer sharing works the same way.
- MPS env set immediately before `.start()`, restored in a `finally` immediately
  after, so only the child inherits the constrained partition; inference keeps the
  full GPU.
- `daemon=True` ⇒ child dies with the worker; `shutdown()` (pipe + join, terminate
  fallback) gives graceful exit and is exercised by the test.

**Verification:**
- CPU-only (spawn / MPS-child-only / ping / clean shutdown): `tests/test_phase1_step2.py` ✅ 13/13
- In-worker spawn during real startup: ✅ verified in full-system launches (`[backward] ...` lines)

### Step 3 — Share FT-adapter + base weights via CUDA IPC ✅

Give the system two LoRA adapters (FT one marked in YAML), and share the FT
adapter's fp32 weights **and** the base model weights with the backward process
via CUDA IPC (zero-copy). This resolves the plan's **#1 risk** (cross-process
CUDA tensor sharing under spawn).

**Files changed:**

| File | Change |
|---|---|
| `vllm/config/finetune.py` | `finetuning_lora_path` (the FT adapter only; inference adapter lives outside finetune) |
| `configs/serving_config_finetuning.yaml` | base model → `facebook/opt-125m`; `finetune.finetuning_lora_path`; **new `adapters:` section** (`inference_lora_path`); `lora:` section (`enable_lora`, `max_loras`, `max_lora_rank`) |
| `vllm/deltaserve/config_loader.py` | `adapters` is a passthrough section; loader returns `(engine_args, extras)` where `extras={"server":…, "adapters":…}`; resolve any `*_path` in `finetune`/`adapters` to absolute (vs YAML dir) |
| `vllm/deltaserve/backward_process.py` | switch to **torch.multiprocessing** (registers CUDA-IPC reductions); `share_weights`/`checksum` handlers; `weight_hash_report()`/`print_hash_report()` (first/last FFN + first/last q_proj LoRA-A); keep producer-side refs alive |
| `vllm/v1/worker/gpu_worker.py` | `_maybe_share_finetuning_weights()` at end of `load_model`: shares base `named_parameters` (frozen refs) + fp32 FT adapter; prints parent hash report + `parent==child` check |
| `launch_deltaserve.py`, `tests/test_config_loader.py` | updated for the `extras` return + `adapters` section |

**Why torch.multiprocessing:** sending a CUDA tensor over its pipe reduces to a
CUDA-IPC handle, so the child maps the *same* GPU memory instead of copying.
(Plain `multiprocessing` would not register these reductions.) Safe because
`CuMemAllocator` only engages under `enable_sleep_mode` (off by default), so base
weights live in the normal caching allocator and are IPC-shareable.

**Verification:**
- CPU: `tests/test_config_loader.py` ✅ 22/22 (incl. new adapter fields + path resolution)
- GPU (mechanism, self-allocated tensors): `tests/test_phase1_step3.py` ✅ 9/9 — counts/checksum
  match, and a parent in-place mutation is seen by the child (zero-copy proof)
- GPU (real worker): launched opt-125m with `enable_finetuning=true` → shared
  base=148 tensors / 125,263,872 elems + ft=144 tensors / 2,654,208 elems; backward
  process confirmed matching summary. **Per-layer hash check** (base FFN fc1 L0/L11 +
  adapter q_proj LoRA-A L0/L11) prints on both parent and child and matches
  (`parent==child: True`) → content-level zero-copy proof. ✅ (Note: serving then
  hit the FlashInfer sm_120 JIT issue — environmental, post-sharing, see
  `README.md`.)

### Tooling — logging, config print, launcher inference adapter ✅

- **Two-color logging:** `dprint` is **green in the main process**, **purple in the
  backward subprocess** (`mark_backward_process()` called at the child entry). Makes
  interleaved multi-process output easy to read. TTY-guarded.
- **Config category print:** `config_loader.print_loaded_config()` (called from
  `load_yaml_config`) prints every loaded section + its key/values, one line per
  section.
- **Launcher uses the inference adapter:** `launch_deltaserve.py` builds a
  `LoRARequest` from `adapters.inference_lora_path` and passes it to `generate`.

**End-to-end run (opt-125m, `VLLM_USE_FLASHINFER_SAMPLER=0` to dodge the sm_120
sampler JIT in this shell):** config printed by category → backward spawned →
weights shared, all 4 hashes `parent==child: True` → generated WITH the inference
adapter → output "Paris" (base would say "the French Republic"). ✅ Confirms the
inference adapter is applied and is distinct from the FT adapter (which is only
shared with the backward process, not applied to inference).

> Note: `VLLM_USE_FLASHINFER_SAMPLER=0` is a per-shell runtime workaround for the
> Blackwell FlashInfer sampler JIT; do NOT bake it into committed config (per
> `README.md` — keep arch-specific backend overrides out of the fork).

### Step 4 — IPC handshake (pause event + work pipe) ⬜

`mp.Event()` for pause/resume; `Pipe`/`Queue` for work handoff and result return.

### Step 5 — Cross-process hash round-trip (Phase 1 deliverable) ⬜

Worker writes a known tensor to the shared buffer; child hashes it and sends the
hash back; worker verifies. Flip values, repeat. Proves the GPU memory is
genuinely shared, not copied. Clean shutdown.

---

## Phase 2 — Activation capture + FT injection + dedicated FT adapter

> Note: Phase 1 Step 4 (pause/resume) intentionally deferred; Step 5 (hash
> round-trip) is effectively covered by the Step 3 weight-hash check.

### Step P2.1 — Finetuning sample store ✅

Port of DeltaServe's `FinetuningManager` (data parts): load + tokenize a corpus
at startup, length-bucketed selection. Pure Python, no GPU/vLLM coupling.

**Files changed:**

| File | Change |
|---|---|
| `vllm/config/finetune.py` | `data_path`, `num_epochs`, `max_prepare`, `max_saved_finetuning_tokens` |
| `configs/serving_config_finetuning.yaml` | `finetune.data_path: ../alpaca_1000.txt` (+ `num_epochs`, `max_saved_finetuning_tokens`) |
| `vllm/deltaserve/finetuning_store.py` *(new)* | `FinetuningSample` + `FinetuningStore` (`load`, `pop_best_under`, `pop_next`, 3-phase `claim`/`commit_claimed`/`release_claimed` (Phase 6, replaces the original `confirmed_trained`), `advance_epoch`, `has_next`, `has_claimed`). `load()` also drops samples with `input_len > max_saved_finetuning_tokens` (P5.3 fix — they'd otherwise sit in the pool forever and deadlock FT admission once fittable samples drain). |
| `alpaca_1000.txt` *(repo root, user-provided)* | 1000-sample corpus, one per line |

**Behavior ported faithfully:** `pop_best_under(max_tokens)` returns the largest
*untrained* sample with `input_len <= max_tokens` (peek, no mark); the 3-phase store
API (`claim` reserves, `commit_claimed` marks trained, `release_claimed` rolls back —
Phase 6 replacement for the original `confirmed_trained`) ensures `_claimed`
samples are reconciled per backward; `advance_epoch` resets marks (total_epochs gates
the count) and refuses while any sample is claimed in-flight. Dropped vs DeltaServe:
Req/Batch coupling, bwd-loss bookkeeping (Phase 3).

**Not yet wired** into the engine — the store is constructed/used during FT
injection (next step), scheduler-side, with the engine's tokenizer.

**Verification:** `tests/test_finetuning_store.py` ✅ 17/17 — loads the real
alpaca_1000.txt with the opt-125m tokenizer (1000 samples / 103,657 tokens / 187
distinct lengths, min=38 max=274); selection, marking, epochs, `max_prepare` cap
all correct. `tests/test_config_loader.py` ✅ 26/26.

### Step P2.2 (Milestone 1) — FT injection + finetune_mask + force-eager ✅

Inject finetuning samples into real batches as single-step prefill-only requests
(`max_tokens=1`), routed to the dedicated FT LoRA adapter, mark their tokens with a
`finetune_mask`, force the step eager, and retire them same-step — invisible to the
frontend. NO activation capture yet (Milestone 2). SLO admission deferred.

**Files changed:**

| File | Change |
|---|---|
| `vllm/v1/request.py` | `Request.is_finetuning = False` flag |
| `vllm/v1/core/sched/output.py` | `SchedulerOutput.finetune_req_ids: set[str]` |
| `vllm/deltaserve/ft_injector.py` *(new)* | `FinetuneInjector`: pulls FT samples (length-bucketed), builds `max_tokens=1` Requests tagged `is_finetuning`, FT `LoRARequest` (reserved id 1000) |
| `vllm/deltaserve/ft_scheduler.py` *(new)* | `FinetuneScheduler(Scheduler)`: inject into queues before `super().schedule()` (gated on real work), cleanup unscheduled FT, populate `finetune_req_ids`; in `update_from_output` retire FT via `_free_blocks` BEFORE super → base loop's `request is None` skips them (no `EngineCoreOutput`) |
| `vllm/config/vllm.py` | `__post_init__`: select `FinetuneScheduler` when `enable_finetuning` (originally also forced `async_scheduling=False`; Phase 4b made async the default — see below) |
| `vllm/v1/worker/gpu_model_runner.py` | `_build_finetune_mask` (flat-batch order via `req_ids`+`query_start_loc`); `force_eager=self._ft_has` into dispatch (+assert `CUDAGraphMode.NONE`); `skip_compiled` on FT steps |

**Design:** FT samples reuse vLLM's KV alloc + LoRA routing + lifecycle. `max_tokens=1`
⇒ one prefill, minimal KV. The scrub frees their KV the same step *without* touching
`finished_req_ids`, so the frontend (which never registered them) sees nothing. Mask
built CPU-side (no sync) in `InputBatch.req_ids` order. Block hasher mirrors EngineCore
so FT requests behave under prefix caching. (Phase 2 ran sync scheduling so inject→retire
stayed within one step; Phase 4b made async safe via reserve-at-inject — see below.)

**Verification:** `tests/test_phase1_m1.py` ✅ 3/3 (GPU) — spawns baseline
(`enable_finetuning=false`) vs FT-on subprocesses on identical greedy prompts:
real-request output token ids **byte-identical**; FT-on engine terminates and returns
exactly one output per prompt (no FT leak). FT firing confirmed in logs:
`[runner] FT step: 256 FT tokens / 262 total -> eager + capture`. Existing CPU tests
still green (config 26, ft-store 17, step2 13). Uses `VLLM_USE_FLASHINFER_SAMPLER=0`.

### Step P2.3 (Milestone 2) — activation buffers + per-layer capture + hash ✅

Allocate fixed-size shared GPU buffers (sized by `max_saved_finetuning_tokens`),
register per-layer forward hooks that copy FT-token-only rows of the attention-output
and FFN-output projections during the (eager) FT forward, capture the pre-LM-head
hidden states + target ids after the forward, share all buffers zero-copy with the
backward process, and verify cross-process by hash. No backward training yet.

**Files changed:**

| File | Change |
|---|---|
| `vllm/deltaserve/accumulate.py` *(new)* | `FinetuneCapture`: discovers `out_proj`/`fc2` per layer, allocates `attn_out`/`ffn_out` `[max_saved, hidden]`×L + `final_hidden` + `concat_input_ids`; `register_hooks`, `begin_step`/`capture_final`/`end_step` |
| `vllm/deltaserve/backward_process.py` | `share_activations`/`hash_activations` (BackwardProcess) + child handlers + `activation_hash_report`; `print_hash_report` handles plain-hash entries |
| `vllm/v1/worker/gpu_worker.py` | `_maybe_setup_finetuning_capture()` at end of `load_model`: alloc capture, register hooks, share buffers with backward, inject capture+backward into runner |
| `vllm/v1/worker/gpu_model_runner.py` | `_build_finetune_mask` also sets `self._ft_num`; `execute_model` calls `begin_step` (pre-forward) + `capture_final`/`end_step` (post-forward) + one-shot `_maybe_verify_ft_capture` (parent vs child hash) |

**Design:** hooks fire eager-only — fine since FT steps are forced eager + `skip_compiled`;
on non-FT steps they're no-ops (or don't fire under cudagraph). Buffers are plain
`torch.zeros` (outside any graph pool). Shared once via CUDA IPC; in-place hook writes
are visible to the backward process zero-copy. Captured: pre-LM-head hidden states
(LM head deferred to backward, per the chosen design) + FT input ids (targets).

**Verification:** `tests/test_phase1_m2.py` ✅ 5/5 (GPU) — hooks+buffers set up, shared
with backward, FT capture ran, and `[capture] activation hashes parent==child: True`
for per-layer attn/ffn (first+last), final_hidden, and concat_input_ids; no mismatch.
`tests/test_phase1_m1.py` ✅ still 3/3 — **real inference byte-identical with hooks
active** (no torch.compile/hook perturbation). Uses `VLLM_USE_FLASHINFER_SAMPLER=0`.
### Step P2.4 — Co-serving coordinator (fill tracking + admission control) ✅

Accumulating activation buffer + admission gate cycling with the backward process.

**The fill index:** `FinetuneCoordinator.fill_count` (new) is the activation-buffer
write offset / fill level vs `capacity = max_saved_finetuning_tokens`. (Before this,
capture overwrote from offset 0 each step — no accumulation.)

**Behavior:**
1. Never build an FT-only batch — inject FT only when real inference work is present.
2. Per inference batch, admit FT up to `0.5 * max_saved_finetuning_tokens` (capped by
   free space); capture accumulates at `fill_count`.
3. Buffer full (`space < min_sample_len`) → signal backward + CLOSE FT admission.
4. Backward gets the signal, cleans the shared buffer, sleeps 1s (simulated backward),
   responds.
5. Main process polls the response → reset `fill_count=0` → reOPEN admission.
6. All decisions printed (`[ft-sched]`/`[coord]` green, `[backward]` purple).

**Files changed:**

| File | Change |
|---|---|
| `vllm/deltaserve/coordinator.py` *(new)* | `FinetuneCoordinator` (singleton): `fill_count`/`capacity`, `next_ft_budget`, `current_offset`, `record_capture` (full→signal+close), `poll_backward` (reopen) |
| `vllm/deltaserve/backward_process.py` | `notify_buffer_full` (async) + `poll_response` (non-blocking) + child `process_activations` (clean buffer + sleep 1s + respond) |
| `vllm/deltaserve/accumulate.py` | `begin_step`/`capture_final` take an `offset` → write at `[offset:offset+n]` (accumulate) |
| `vllm/deltaserve/ft_scheduler.py` | `schedule()` polls backward, gates on real work + `next_ft_budget()` |
| `vllm/v1/worker/gpu_worker.py` | creates the coordinator (worker runs before scheduler), sets `backward_process`, injects into runner |
| `vllm/v1/worker/gpu_model_runner.py` | reads `current_offset()` pre-forward; `record_capture(n)` post-forward |

**Subtlety found:** in `EngineCore.__init__` the worker/`load_model` runs *before* the
scheduler is constructed (during `_initialize_kv_caches`), so the coordinator is
created by the worker (the scheduler reuses the singleton + sets `min_sample_len`).

**Verification:** `ft_experiment.py` (new harness: launch engine + fire a max_tokens=10
prompt every 1s for N iterations, then shutdown) shows the full cycle repeating across
requests: fill 0→128→240 → FULL → signal → CLOSED → backward clean+sleep → done →
reset → OPEN. Idle gaps between requests inject no FT (req. 1). `tests/test_phase1_m1.py`
✅ 3/3 (inference still byte-identical) and `test_phase1_m2.py` ✅ 5/5 (capture hashes
match) — no regression.

### Step P2.5 — HTTP experiment harness + observability ✅

Wrap-up of the co-serving experiment loop + log readability. **Also renamed
`capture` → `accumulate`** throughout (file `capture.py`→`accumulate.py`, class
`FinetuneCapture`→`FinetuneAccumulator`, `capture_final`→`accumulate_final`,
`[capture]`→`[accumulate]`) to avoid confusion with CUDA-graph capture — so the
earlier P2.3/P2.4 entries' `capture.py`/`FinetuneCapture` references are now those.

**Changes:**

| File | Change |
|---|---|
| `config/finetune.py` | `backward_sleep_seconds` (2.0); debug flags `print_weight_hash` / `print_activation_hash` / `print_step_mode` |
| `configs/serving_config_finetuning.yaml` | `max_saved_finetuning_tokens: 512`; `backward_sleep_seconds`; new `debug:` section |
| `deltaserve/config_loader.py` | `debug:` is a special section folded into FinetuneConfig kwargs |
| `deltaserve/backward_process.py` | child weight/activation hash prints gated on the debug flags; backward sleep duration plumbed via `notify_buffer_full(n, sleep_s)` |
| `deltaserve/coordinator.py` | `backward_sleep_s`; dropped the per-step `accumulated` print (noise) |
| `v1/worker/gpu_model_runner.py` | `_log_finetuning_batch`: `[batch] prefill=.. ft=.. decode=[kv sizes] \| eager/graph(MODE)`; gated on `print_step_mode`, skips decode-only batches; **graph flag from the real cudagraph dispatch** (not inferred from has_ft) |
| `ft_experiment.py` | rewritten as a real **HTTP server** harness: launches `vllm serve` with finetuning, fires completions every 1s ×N, shuts down |

**Notes / gotchas found:** `python -m vllm.entrypoints.openai.api_server` triggers a
circular import (`from vllm import SamplingParams`) — use the `vllm serve` console
script instead. The decode-only `decode=1` was a single running request generating one
token per step; those batches are no longer logged.

**Verification:** `tests/test_phase1_m1.py` ✅ 3/3, `test_phase1_m2.py` ✅ 5/5,
`test_config_loader.py` ✅ 30/30 (debug section). `ft_experiment.py` run on the 5090
shows the full cycle live (fill→FULL→backward 2s→reopen) with readable `[batch]` lines
and no hash spam when the debug flags are off. Run with `VLLM_USE_FLASHINFER_SAMPLER=0`.

## Phase 3 — Real backward pass ✅ (gradcheck-verified; 8B runtime is the user's to confirm)

### Step P3.1 — per-model backward services + logits/loss/logit-gradient ✅

Split the child backward into **per-model services** and completed the *forward of
the loss* the forward pass deferred (vLLM only materializes last-token logits, so
Phase 2 saved pre-LM-head hidden states + input ids for the LM head to run later).

**Restructure.** The child loop + per-model SFT math moved out of
`backward_process.py` into a new package `deltaserve/bwd_services/`:
- `bwd_services/base.py` — `BackwardService` (model-agnostic recv/dispatch loop,
  CUDA-IPC mappings, hash debug cmds, `process_activations`) + `service_main` (child
  entry point: marks process, binds device, picks service, runs) + `get_service`
  factory (arch string → service class; non-OPT → `NotImplementedError`).
- `bwd_services/opt.py` — `OPTBackwardService.compute_loss_and_grad`.
- `backward_process.py` keeps only the **parent-side** `BackwardProcess` (spawn / pipe
  / weight+buffer IPC) + the shared hashing helpers; it imports `service_main` lazily
  in `start()` to avoid an import cycle.

**Loss + logit gradient.** On each buffer-full signal, *before* the existing
clean+sleep, `OPTBackwardService` now:
1. reconstructs full logits `final_hidden @ lm_head.weight.T` (fp32 LM head;
   `final_hidden` is already post-final-norm), trims the padded vocab to
   `hf_config.vocab_size`;
2. computes next-token CE vs `concat_input_ids` as labels — **shift-by-1 within each
   sample**, full sequence, no prompt masking (matches DeltaServe
   `get_logits_and_targets`/`compute_total_loss`);
3. computes the CE gradient dLoss/dLogits = `softmax − one-hot`, normalized over valid
   tokens (DeltaServe `_logit_backward`), and **stashes it** (`self.last_logit_grad`)
   for the next slice; prints `loss` + grad norm.
Then clean buffers → sleep → send done (`loss` included in the response). No LoRA
backward / optimizer / pause-resume yet.

**Plumbing.** Per-FT-sample token lengths are needed to split the flat buffers and
shift safely (no target crosses a sample boundary). `_build_finetune_mask` now also
produces `self._ft_sample_lens` (req_ids order == buffer-write order); the runner
passes them to `coordinator.record_capture(n, sample_lens)`; the coordinator
accumulates `self.sample_lens` and forwards them via
`notify_buffer_full(n, sleep_s, sample_lens)`, resetting on backward-done. The worker
passes the model arch as the backward `service_name` and sends a `meta` dict
(LM-head weight key, org vocab size, logit scale) with `share_weights`. The LM-head
weight is already in the shared base weights (opt-125m ties embeddings, so the key is
`model.decoder.embed_tokens.weight`).

**Changes:**

| File | Change |
|---|---|
| `deltaserve/bwd_services/__init__.py` | NEW — exports `get_service` / `service_main` / `BackwardService` |
| `deltaserve/bwd_services/base.py` | NEW — model-agnostic loop + child entry point + factory |
| `deltaserve/bwd_services/opt.py` | NEW — `OPTBackwardService` logits/loss/logit-grad |
| `deltaserve/backward_process.py` | removed `_backward_stub_main` (→ base); `service_name` arg + lazy `service_main` spawn; `share_weights(..., meta=)`; `notify_buffer_full(..., sample_lens=)` |
| `deltaserve/coordinator.py` | `sample_lens` accumulation + forward on trigger + reset on done |
| `v1/worker/gpu_model_runner.py` | `_build_finetune_mask` builds `_ft_sample_lens`; `record_capture(n, sample_lens)` |
| `v1/worker/gpu_worker.py` | pass `service_name=arch` to `BackwardProcess`; resolve + send LM-head `meta` |

**Verification:** unit test of `OPTBackwardService.compute_loss_and_grad` (2 samples,
lengths [3,2]) — loss equals manual per-sample CE, grad equals `softmax−onehot`/n,
shapes/finiteness correct. `ft_experiment.py` on the 5090: each backward cycle logs a
finite `loss` + logit-grad norm; the fill→FULL→backward→reopen cycle still runs;
real-request inference unchanged.

### Step P3.2 — pivot to Llama-3: llama3 loss service + backward-useful capture ✅

Moved the active target to **meta-llama/Meta-Llama-3-8B** (opt-125m kept as a renamed
reference path) and made the captured activations correct + useful for the upcoming
per-layer LoRA backward.

**Capture redesign (`accumulate.py`).** Dropped the opt-specific output hooks
(`self_attn.out_proj` / `.fc2` → `attn_out`/`ffn_out`) and replaced them with
**residual-stream layer-input** capture via `register_forward_pre_hook`s, auto-detected
by module name:
- `layers.{i}.input_layernorm` → `layer_in[i]` (residual entering layer i);
- `model.norm` → `final_in` (pre-final-norm residual, = input to layer L).
The fused add-norm means the pre-hook sees `args=(hidden,)` (layer 0) or
`(hidden, residual)` (i>0); residual = `args[0]` or `args[0]+args[1]`, copied
immediately (the op may update `residual` in place). `final_hidden` (post-norm) +
`concat_input_ids` are still captured for the loss. opt has none of these modules, so
nothing registers there — its loss path (post-norm `final_hidden`) is unchanged.

**Loss service (`bwd_services/llama3.py`).** `Llama3BackwardService` reuses the new
shared `base._logit_loss_and_grad` (logits = `final_hidden @ lm_head.T`, per-sample
shift-by-1 CE + logit grad) — identical math to opt; only `lm_head_key` differs
(Llama-3-8B is **untied** → `lm_head.weight`). `OPTBackwardService` now delegates to the
same helper.

**Capture correctness gate.** `Llama3BackwardService.verify_activations` (run each cycle
when `print_activation_hash=true`) asserts (a) `layer_in[0] ≈ embed[ids]` and (b)
`RMSNorm(final_in) ≈ final_hidden`, validating the residual reconstruction the backward
will differentiate through. The worker `meta` now also carries `rms_norm_eps`,
`norm_weight_key`, `embed_weight_key`.

**Changes:**

| File | Change |
|---|---|
| `deltaserve/accumulate.py` | residual-stream pre-hook capture (`layer_in`/`final_in`); dropped `attn_out`/`ffn_out` |
| `deltaserve/bwd_services/base.py` | shared `_logit_loss_and_grad`; `verify_activations` hook + `_verify` flag; `get_service` → `LlamaForCausalLM` |
| `deltaserve/bwd_services/llama3.py` | NEW — `Llama3BackwardService` (loss + `verify_activations`) |
| `deltaserve/bwd_services/opt.py` | delegate to `_logit_loss_and_grad` |
| `deltaserve/backward_process.py` | `activation_hash_report` → `layer_in`/`final_in`/`final_hidden`/`concat_input_ids` |
| `v1/worker/gpu_worker.py` | `meta` += `rms_norm_eps` / `norm_weight_key` / `embed_weight_key` |
| `configs/serving_config_finetuning_{opt,llama3}.yaml` | renamed opt; NEW llama3 (Meta-Llama-3-8B, llama3-toy-lora{,-ft}) |
| `ft_experiment_{opt,llama3}.py` | renamed opt harness; NEW llama3 harness |
| `launch_deltaserve.py`, `tests/test_{config_loader,phase1_m1}.py` | config-name refs updated |

**Verification:** loss-math equivalence re-checked after the refactor (matches manual CE).
`ft_experiment_llama3.py` on the 5090 with `print_activation_hash=true`: per cycle the
`[verify]` lines report `layer_in[0]≈embed` and `RMSNorm(final_in)≈final_hidden` within
bf16 tolerance, and `[backward]` logs a finite CE `loss` + logit-grad norm; the
co-serving cycle runs and completions are coherent. opt harness still runs.

### Step P3.3 — real LoRA backward + optimizer (llama3) ✅

Replaced the simulated sleep with the actual **manual** LoRA SFT backward in the subprocess
(`Llama3BackwardService`), training the FT adapter (`adapters/llama3-toy-lora-ft`, q/k/v/o,
r=16, α=32 ⇒ scaling 2.0) from the captured activations while inference keeps serving. Per
the user: **manual gradient computation (no autograd), reusing the recorded activations +
per-layer forward rematerialization**, AdamW + StepLR like DeltaServe. MLP/embeddings/norms
are frozen — only the 8 LoRA tensors per layer get grads.

**Math** (`bwd_services/llama3.py`, module-level helpers + service): head = per-sample shift
CE over `RMSNorm(final_in)@lm_head.T` (fp32, vocab-chunked) → exact RMSNorm-backward to
`grad_final_in`; per layer `i=L-1…0`: rematerialize the layer forward from `layer_in[i]`
(RMSNorm → q/k/v base+LoRA → NeoX RoPE → per-sample GQA causal attn fp32 softmax → o
base+LoRA → +resid → RMSNorm → SwiGLU MLP → +resid), then hand-derived backward (FFN through
frozen MLP, O grads, GQA softmax-bwd, RoPE-bwd, q/k/v base+LoRA grads, RMSNorm-bwd +
residual). LoRA grads derived directly in **PEFT layout** (`grad_A=grad_Zᵀ@x`,
`grad_B=scaling·gyᵀ@Z`); per-layer grad-clip 1.0; written to fp32 master `.grad`.
`logit_grad` is `[n,vocab]` with **zeroed rows at each sample's last token** (we keep all
`T_i` tokens vs DeltaServe's `T_i−1`; causal attention makes those contribute zero).
Adapted to vLLM: fused `qkv_proj`/`gate_up_proj` sliced; cos/sin rebuilt from
`inv_freq=1/5e5^(2i/128)`; no score clamp.

**Lifecycle** (`bwd_services/base.py`): `process_backward` hook — default (opt) = loss-only,
llama3 overrides with `zero_grad → manual backward → optimizer.step → StepLR on epoch
increment`. `is_trainer=True` services skip the simulated sleep. Epoch plumbed
scheduler→coordinator→`notify_buffer_full(epoch=)`→service.

**Changes:** `config/finetune.py` (`learning_rate`/`weight_decay`/`gamma`); `gpu_worker.py`
(`meta` += model dims + `lora_scaling` from adapter_config + lr/wd/gamma); `coordinator.py`
+ `ft_scheduler.py` + `backward_process.py` (epoch); `bwd_services/{base,llama3}.py` (the
backward + optimizer); NEW `tests/test_llama3_backward.py`.

**Verification:** `tests/test_llama3_backward.py` ✅ — manual grads (head `grad_final_in`
+ all 8 per-layer LoRA tensors + input grad) match `torch.autograd` to **~1e-7** rel-err on
synthetic fp32 shapes. Runtime (user-run): `scripts/ft_experiment_llama3.py` — per-cycle
`[verify]` OK + `[backward] loss=…` decreasing, co-serving cycle intact. opt path unchanged.

### Step P3.4 — precision flag, served-weight publish, epoch flush, gate/up save ✅

Four refinements that finished the trainer; all `tests/test_llama3_backward.py` ✅ **12/12**
(now incl. the saved-`gate_up` path) + CPU smoke tests for the publish + epoch-flush.

- **Backward precision flag** (`finetune.backward_fp32`, default false): the bulk backward
  matmuls (FFN-bwd, LoRA-grad, rope/proj-bwd) run in the **model dtype (bf16)** by default
  (matches DeltaServe's fp16/bf16 bulk + the precision memory), with `true` forcing fp32.
  The load-bearing ops (attention scores/softmax, RMSNorm, LM-head/final-norm, fp32 LoRA
  master) are **always fp32**. Threaded via `meta`; `layer_backward(cdt=…)`.
- **Served-weight publish** (the trained weights actually reach inference). The worker
  pre-`add_lora` + `pin_lora`s the FT adapter into a stable served slot and IPC-shares
  vLLM's served LoRA stacked buffers (`qkv_proj` 3 slices + `o_proj`) with the subprocess
  (`gpu_worker._maybe_share_ft_served_lora`, `backward_process.share_lora_buffers`). After
  `optimizer.step`, `Llama3BackwardService._publish_to_served()` writes the fp32 master into
  those buffers — clamp(±6.5e4) + cast bf16 + ×scaling on B (vLLM bakes α/r into B; applies
  `scale=1`); no transpose (PEFT A `[r,in]` / B `[out,r]` match vLLM's layout). Safe with no
  locking: FT admission is closed for the whole backward, so the adapter is idle until the
  done-reply reopens it. DeltaServe analogue: fp32 home + clamp/cast refresh at adapter load.
- **Epoch-boundary flush** (`coordinator.flush_partial`, called from `FinetuneScheduler.schedule`
  when `store.has_next()` is false): trigger the backward on a **non-full** buffer when the
  current epoch's samples are exhausted, so the epoch's trailing samples are trained before
  the next epoch and StepLR steps at the right boundary. So the backward fires on **buffer
  full OR epoch end**.
- **MLP gate/up activation save** (memory-for-latency). The forward now also captures each
  `mlp.gate_up_proj` output (`[n, 2·inter]`, ~940 MB at n=512) via a post-hook
  (`accumulate.py`); the backward uses it to **skip the gate_up matmul** (the layer's widest,
  frozen matmul) and also **drops the previously-unused `down`/`out` recompute** — so the
  per-layer remat is now attention-only. `layer_forward(saved_gate_up=…)`; opt has no
  `gate_up_proj` so it's llama3-only.

**Phase-3 status:** the co-serving training loop is complete and gradcheck-verified — capture
→ manual fp32 backward → optimizer/StepLR → publish to the served adapter, gated by buffer-full
or epoch-end. The 8B end-to-end run is the user's to confirm. Remaining co-serving polish (the
GPU-yielding `_maybe_pause` contract, backward CUDA graphs, attention batching) is Phase 5.

## Phase 4 — SLO-aware scheduler + estimator 🟡 (code complete; GPU validation pending)

Replaces fixed FT injection with SLO-aware admission. Implemented as ONE **merged** step
estimator (vLLM runs a single mixed prefill+decode batch, so DeltaServe's separate prefill +
decode estimators collapse into one):

    T_step ≈ α·S + β·T_in + γ·T_ft + δ·B_d + ε·K + c   (S = Σnᵢ² ≈ T_in²/P proxy)

- **Estimator** (`deltaserve/estimator.py`): `StepFeatures`, `StepParams` (6 coeffs, eager +
  graph regimes — γ kept in BOTH for future graphed co-serving), `StepExecutionTracker` (ports
  `BatchExecutionTracker`: rolling record, `check_refit` every 256, predicted-vs-actual CSV),
  `MergedExecutionEstimator` (`predict`, `data_fit` lstsq partitioned by `was_graph`,
  `max_next_ft_tokens` quadratic admission solver). Unit-tested CPU-only
  (`tests/test_merged_estimator.py`, 21/21).
- **Online wiring** (`ft_scheduler.py` + `gpu_model_runner.py` + `coordinator.py`): CUDA-event
  step timing → `coord.last_step_s`; `will_use_graph` queried from vLLM's real
  `CudagraphDispatcher` (shared via the coordinator singleton — no mirror); every served step
  stamped (`was_graph`, predicted) at `schedule()` and recorded with measured duration at
  `update_from_output()`; refit every 256 steps; stats CSV on shutdown.
- **Offline profiler** (`profiling_batch_generator.py` + `EngineCore.profile_execution_model`):
  generates shape sweeps (prefill decomposition / decode B×K / coserve / mixed) and runs them
  through the live scheduler at launch (before `run_busy_loop`), seeding the estimator.
  Isolation: `_profiling_mode` suppresses auto-inject, backward child detached, coord reset
  between shapes, synthetic reqs purged. Generator unit-tested (`tests/test_profiling_shapes.py`,
  15/15).
- **Admission gate** (`ft_scheduler._slo_ft_budget`): replaces the fixed
  `coord.next_ft_budget()`; computes the upcoming step's inference composition from
  `self.running` (decode B,K) + `self.waiting` head (prefill T_in,P), predicts `T_current`, then
  `x_ft = min(max_next_ft_tokens(budget_ttft), max_next_ft_tokens(budget_tbt), coord cap)`.
  TTFT + max-TBT gates implemented; avg-TBT deferred (needs per-request last-token tracking).
- **Config** (`config/finetune.py` + `config_loader.py` + YAMLs): `slo.{ttft_slo, avg_tbt_slo,
  max_tbt_slo}`, `finetune.{profile_on_launch, profile_num_repeats, batch_prediction_stats_path}`.

**Phase 4b — async scheduling ENABLED. ✅** `async_scheduling` now defaults ON for co-serving
(`config/vllm.py`; was force-off). `FinetuneScheduler` inherits `AsyncScheduler` (output
placeholders). Under uniproc, async sets `max_concurrent_batches=2` → batch-queue pipelining
(`schedule(N+1)` before `record_capture(N)`). Made safe by **reserve-at-inject** in the
coordinator: `reserved_fill` (admitted, not-yet-saved) + `fill_count` (committed);
`space_remaining = capacity − committed − reserved` so admission can't overflow; `reserve(n)`
returns a **disjoint per-step write offset** (committed+reserved) the runner uses, so two
in-flight steps never overlap their buffer writes; backward triggers (buffer-full + epoch-flush)
fire from `record_capture` only when `reserved==0` (no in-flight saves) → race-free. Per-step
**duration** is stashed on `scheduler_output` (not a single coordinator slot the pipeline would
clobber). Epoch boundary holds the next epoch's admission until the tail flushes, so epochs
don't co-reside in the buffer. (Backward spawning is uniproc-only, so multiproc async isn't a
concern.)

**Phase 4c — control plane + deferred timing + eval harness + fixes.**
- **FT start control plane.** FT admission is **gated off at launch** (`finetune.start_on_launch`,
  default still True but the experiments set it False) so a profiling pass and warmup can run
  with zero FT. A POST `/start_finetuning` endpoint
  (`entrypoints/serve/finetune/api_router.py`, attached only when finetuning is enabled) calls
  `collective_rpc("deltaserve_start_finetuning")` → `coordinator.start_finetuning()`, which flips
  `ft_started`; `next_ft_budget()` returns 0 until then. `auto_benchmark.py` POSTs it after warmup;
  `ft_experiment_llama3.py` POSTs it before the first prompt. Gating is independent of `profiling`
  so the two never conflict.
- **Deferred CUDA-event timing.** Step timing uses a **deferred CUDA-event ring** (no per-step
  sync — read RING=4 steps later, off the hot path) so async pipelining isn't serialized; the
  runner pushes completed (features, duration, was_graph, predicted) samples to a coordinator queue
  and the scheduler drains them into the tracker (decoupled because the duration isn't known at
  `update_from_output` time). Prefill steps still sync once on their end event so the `_maybe_pause`
  backward resume lands after the forward completes; decode steps (the throughput-critical bulk)
  stay fully async.
- **Server-side FT-throughput log.** When `finetune.bwd_log_path` is set, the coordinator writes a
  row (wall clock + cumulative tokens trained) each time the backward completes — the data the
  eval plotter reads for the FT-throughput band.
- **Eval harness** (`eval/`, ported from `DeltaServe/eval/llama3/`): `auto_benchmark.py` launches
  `vllm serve` (base llama3 + inference LoRA `adapters/llama3-toy-lora`; `--co` adds the finetune
  flags + FT LoRA `adapters/llama3-toy-lora-ft`), replays a request timeline from
  `eval/timelines/5090/`, streams `/v1/completions` (ttft = first chunk), POSTs `/start_finetuning`
  after warmup, tees server stdout to `server<suffix>.log`, and emits
  `timeline_results<suffix>.csv` (`idx,t_rel_s,latency_s,status,ttft_s,avg_tbt_s,worst_tbt_s`).
  `auto_plot.py` (csv+numpy+matplotlib, no pandas) renders a 4-panel figure (request timeline /
  E2E latency / inference+FT throughput bands / TTFT-SLO satisfaction), reading the SLO from the
  config YAML. GPU autodetects 5090 vs A100.
- **Fixes shaken out under load:** epoch-flush deadlock at the corpus epoch boundary
  (`try_epoch_flush` no longer requires `admission_open`); FT-partition async leak (the partition
  loop now only retires **this-step** injects — `num_output_placeholders == 0` — never in-flight
  prior-step FT reqs, which had stalled the engine at 256 running / 0 throughput); `has_requests()`
  refined so the busy loop doesn't spin at 100% CPU on a stuck partial buffer; `_trigger_backward`
  no-ops while `profiling`; shutdown IPC guards (`notify_buffer_full`/`poll_response` tolerate a
  dead child); TTFT budget now subtracts the in-flight queue wait (scheduling for step N+1 happens
  while step N's forward runs under async).

**Phase 4d — GPU validation findings: the TTFT stall was the *frontend*, not co-serving. ✅**
Running `eval/auto_benchmark.py --co --loose` showed inference TTFT spiking ~1.4–1.6s in periodic
bursts (≈ every inference burst), which closed FT admission. A top-down instrumentation pass
(since reverted) localized it definitively, ruling out — by measurement — the engine step
(no step >100ms), the GPU/backward (no sync stalls), our SLO-estimator timing, GC, and the
client. The block was a **stop-the-world on the API process's asyncio event loop**: vLLM attaches
per-step **`scheduler_stats`** to the **rank-0 frontend's** output stream, so under
`--api-server-count N` that one frontend receives a ZMQ message *every engine step* (~80/s during
a decode burst) and the recv/`decode`/`process_outputs`/metrics-`record` churn saturates its single
loop — starving HTTP accept + SSE streaming. So the inference-latency problem (and the resulting
FT non-admission) was **not** GPU contention from co-serving, the backward, or our scheduler.

Fixes (all kept; the diagnostic prints/watchdog were removed afterward):
- **`disable_log_stats` auto-defaults ON when `enable_finetuning`** (`engine/arg_utils.py:create_engine_config`
  mutates `self.disable_log_stats=True`; runs before every consumer reads it). Kills the per-step
  stats stream. Our SLO estimator uses its own engine-side CUDA timing, so it's unaffected. Also
  set explicitly in the YAML `engine:` section. **This is the actual TTFT fix.**
- **`--api-server-count` plumbed** (`eval/auto_benchmark.py` flag + YAML `server.api_server_count`,
  CLI overrides): 1 shared EngineCore (all DeltaServe state intact — verified the engine process is
  spawned non-daemon so the backward child can still spawn) + N frontends behind a shared socket,
  to shard frontend output processing.
- **Observability kept lean:** `[engine-recv HH:MM:SS.mmm] #N ADD req=…` (per-request engine arrival,
  via `coord.inf_req_count`) and `[batch … t=+Xs] …` (the batch-shape line now carries a
  since-`start_finetuning` timer, `coord.ft_start_time`). `eval/auto_plot.py` gained avg/p90-TTFT +
  avg-TBT annotations (flagged vs the SLOs) and an inf-only E2E-latency overlay when the no-co run
  exists.

**Phase 4e — co-serving admission/buffer tuning (the levers for the next step). ✅ (mechanism)**
- **`per_step_budget = capacity`** (was `capacity // 2`; set in `gpu_worker.py` — the binding site
  since the worker creates the coordinator singleton first — + `ft_scheduler.py` + coordinator
  default). An idle/FT-only step now fills the buffer in **one eager forward** instead of ~4.
- **Buffer-wedge fix.** The buffer used to stall at a near-full level during idle (e.g. 208/256):
  no untrained sample fit the free space, yet the static-corpus-min trigger didn't fire. Now the
  scheduler hands the coordinator the **peek-next** smallest-untrained length after each injection
  (`store.pop_next()` → `coord.note_injection(len)`), which raises the flush flag
  (`epoch_flush_pending`) when the buffer can't grow (epoch drained OR next sample won't fit /
  would overflow); the existing backward trigger consumes + unsets it. So the partial buffer is
  trained instead of wedging, and idle slack isn't wasted.
- **Pause/resume is fire-and-forget.** The runner no longer does a blocking `event.synchronize()`
  between pausing the backward and resuming it — pause/resume are just `mp.Event` toggles — and it
  only engages when a backward is actually in flight (`coord.pending_backward`). The `_trigger_backward`
  cross-process visibility wait is now scoped to a **capture-completion event** (`coord.capture_done_evt`,
  recorded by the runner after the activation copies) rather than a full-device synchronize.
- **`ft_tokens_admission_constrain_factor`** (config; folds into `FinetuneConfig`). When `> 0` and a
  step carries prefill, FT tokens admitted ≤ `prefill_tokens · factor` (a `min` on top of the SLO +
  buffer caps); `-1` disables. A direct lever for the next step.

**Remaining for Phase 4:** the SLO admission is currently **over-admitting** FT — co-serving still
inflates inference E2E latency more than the SLOs intend (the estimator/gate admits too much FT per
step). Next focus is tuning admission so co-serving inference E2E stays within target (levers:
`ft_tokens_admission_constrain_factor`, the SLO-gate budgets, estimator accuracy under load). Other
follow-ups: avg-TBT gate (needs per-request last-token tracking); **FT loss divergence** in the
loose-co run (training-quality — LR / corpus / publish cadence — separate from the SLO scheduler).

## Phase 6 — Inference pre-emption of FT-only stepping (`forward_interruptible`) 🟡 (code complete; GPU validation pending)

Three-tier system that catches late-arriving inference requests at progressively later
points in the pipeline, all behind ONE config gate `finetune.forward_interruptible`
(default `False` — bit-identical behaviour to today when off, via short-circuit
attribute loads at every check site). Closes all windows where a late HTTP arrival
could land except "after the kernel-launch returned" (unfixable without separate CUDA
streams).

Tier-by-tier in window order (smallest → largest):

- **Tier A — pre-schedule grace window. ✅ (code).**
  When the FT scheduler reports `would_step_be_ft_only()`, the engine main loop does a
  bounded blocking poll on `input_queue` (default 2 ms via `ft_only_admission_grace_ms`)
  before letting `schedule()` commit. If a request arrives in the window, it gets
  admitted and the upcoming batch becomes co-serve. Cost: `X / (X + ft_forward_ms)`
  FT throughput hit during idle (~2.5% at X=2 ms / forward=80 ms). Catch probability
  scales with QPS — `R × X` for uniform Poisson arrivals.

- **Tier B — post-schedule, pre-execute rollback. ✅ (code).**
  After `schedule()` returns, if the batch is FT-only AND `input_queue` is non-empty,
  `_rollback_ft_step(scheduler_output)` undoes all FT-side state (`_free_blocks` for
  the FT requests, `coord.release_reserve(n, samples=)`, `store.release_claimed`,
  `coord.restore_admission(snap)`), drains the queue, and re-schedules once. Bounded
  to one retry.

- **Tier C — mid-forward abort. ✅ (code).**
  The input-socket thread sets `coord.ft_abort_event` on each ADD whenever
  `coord.ft_only_in_flight` is True (cheap gate — no-op outside FT-only forwards).
  Each `accumulate.py` hook checks `event.is_set()` after its copy work and raises
  the `FTAborted` sentinel; the runner catches it inside `execute_model`, zeros the
  partial-write tail at this batch's offset (`accumulator.zero_offset_range`), and
  returns an empty `ModelRunnerOutput(_ft_aborted=True)`. The engine sees the sentinel
  after `future.result()`, calls `_rollback_ft_step`, and skips `update_from_output`.
  Pipeline-depth-2 contamination is handled by an entry-time abort check inside
  `execute_model` (bails before queueing any kernels when the event is already set);
  rollback clears the event so the next FT-only batch starts fresh. Savings ceiling
  is partial (~30–60% of the FT forward, bounded by already-queued GPU kernels).

**Store API change (load-bearing for B + C).** Replaces one-way
`FinetuningStore.confirmed_trained` with a 3-phase API:
- `claim(samples)` — admit time: remove from `len_buckets` / `sorted_lengths`, track
  in `_claimed: set[int]`. `trained` stays False.
- `commit_claimed(samples)` — backward-done time: `trained[idx] = True`, drop from
  `_claimed`. Called via `coord.on_backward_done = store.commit_claimed`, fired from
  `coord.poll_backward` on the success response. **Fixes the pre-existing flaw** where
  samples were marked `trained=True` at admit time, before any backward had actually
  processed their activations — meaning a sample admitted then rolled back would have
  been silently counted as trained.
- `release_claimed(samples)` — rollback time: return samples to the selectable pool.
- `advance_epoch()` now refuses while any sample is `_claimed` (in-flight), so an
  epoch boundary can't silently orphan in-flight FT samples.

**Coordinator changes.** `release_reserve(n, samples=)` symmetric to `reserve`;
`snapshot_admission()` / `restore_admission(snap)` for the admission flags
(deliberately **does NOT** capture/restore `reserved_fill` — that's what
`release_reserve` is for, and restoring a snapshotted `reserved_fill` would clobber a
pipelined intervening commit between snapshot and rollback); `buffer_samples` list
tracking which `FinetuningSample`s contributed to the current activation buffer (so
`on_backward_done` can route the commit to exactly the right samples);
`ft_abort_event` (`threading.Event`) + `ft_only_in_flight` (bool) for tier-C signalling.

**Activation buffer behaviour on abort.** Reservation accounting rolls back cleanly
(`reserved_fill -= n`, `buffer_samples` minus this batch's samples); the partial-write
tail at `[off : off+n]` is zeroed across all hook-target buffers
(`accumulator.zero_offset_range`); KV blocks are freed via `_free_blocks`. `fill_count`
is untouched (it's only bumped in `record_capture`, which the abort path skips), so
the next FT batch reuses the same offset and overwrites any stale bytes that would
have been there. InputBatch state cleans up on the NEXT step's `_update_states` via
the existing "unscheduled" path (same mechanism that handles the normal FT retire).

## Unified-phase FT scheduler — `slo.coserving_admission_phase: both` 🟡 (code; GPU A/B pending)

Opt-in alternative to the default prefill-only FT admission. New scheduler
class `BothPhaseFinetuneScheduler` (`deltaserve/ft_scheduler_both.py`) is a
sibling of `FinetuneScheduler` — selected when
`slo.coserving_admission_phase: both` is set in the YAML (with a startup
soft-fall back to `"prefill"` if `ft_tokens_admission_constrain_factor != -1`,
since the proportional cap is prefill-relative).

**Rationale.** Today's rule "FT only rides prefill-carrying steps" is
DeltaServe-era conservatism — the SLO estimator already predicts decode +
graph/eager regimes, so it can decide per-step whether adding FT fits the
TBT/TTFT budget. An FT sample is structurally just "an inference req with 1
token to generate" (that's literally what `FinetuneInjector` produces). The
load-bearing cost to price in: any batch carrying FT forces eager
(`force_eager=self._ft_has`), so admitting FT onto a CUDA-graphed decode-only
batch pays the eager penalty. The estimator's `will_use_graph` regime split
already models this.

**Implementation.** Two overrides vs the parent `FinetuneScheduler`:
- `_initial_ft_budget(feats, earliest_arrival)` — drops the
  `decode_only → 0` short-circuit. Every step composition runs through the
  SLO budget computation. (The parent class gets a new hook method that
  defaults to today's behaviour, so this is a clean one-line override.)
- `_slo_ft_budget(feats, earliest_arrival)` — when the upcoming step is
  decode-only, scales `max_tbt_slo` by `decode_only_ft_safety_margin`
  (default 0.7) before computing the TBT-derived FT-token cap. Conservative
  margin justified by: (a) eager penalty from losing the CUDA-graph fast
  path can dominate sub-5ms decode-only step time; (b) γ coefficient hasn't
  seen many decode-only + FT samples until online refit accumulates them.

Everything else inherits from `FinetuneScheduler`: injection, coordinator,
backward triggers, async-safety (reserve-at-inject), the
`match_prefill_workload_factor` leaky-bucket gate, the `_unspent_prefill`
counter. The leaky-bucket counter still self-gates on `feats.t_in > 0`, so
long decode-only stretches naturally rate-limit FT (no new credit accrues)
once banked credit is consumed.

**Config.** Two new fields under `slo:` in `FinetuneConfig`:
- `coserving_admission_phase: str = "prefill"` — selects the scheduler.
- `decode_only_ft_safety_margin: float = 0.7` — TBT-budget multiplier on
  decode-only steps.

New YAML: `configs/serving_config_finetuning_llama3_both.yaml` (copy of the
existing llama3 YAML with `slo.coserving_admission_phase: both` +
`slo.decode_only_ft_safety_margin: 0.7`).

**Eval tooling.**
- `eval/auto_benchmark.py` gains `--scheduler {prefill,both}` (default
  `prefill`); the flag picks the corresponding YAML and the output suffix is
  extended to `_co_factor_<X>_phase_<Y>_<mode>` so A/B runs across schedulers
  don't overwrite.
- `eval/auto_plot_schedulers.py` (new) emits two A/B PNGs centered on
  `phase=both`: `both_vs_inf-only` (co-serving overhead vs no-co baseline)
  and `both_vs_prefill` (head-to-head scheduler comparison). Same 4-panel
  layout as `auto_plot.py` (timeline / E2E latency / throughput / TTFT
  satisfaction). Reuses `auto_plot.py` helpers (`parse_bwd_log_csv`,
  `_distribute_to_bins`, etc.). Auto-detects factor + phases from disk.
- `eval/auto_plot.py`: percentile panel dropped (figure now single-row
  4-panel; percentile view lives in `auto_plot_schedulers.py`'s context).

**Verification path (pending GPU validation).**
1. Smoke: launch with the new YAML, confirm log line
   `BothPhaseFinetuneScheduler active | decode_only_safety=0.7 | ...`.
2. Confirm `[batch]` logs eventually show FT tokens admitted on decode-only
   steps (`decode=[...] | ft=N` with `t_in=0`).
3. A/B: run `auto_benchmark.py --co --loose` three times (no-co baseline /
   `--scheduler prefill` / `--scheduler both`), then
   `python eval/auto_plot_schedulers.py --mode loose`. Read the two output
   PNGs for: TTFT SLO satisfaction held, E2E latency curves comparable,
   `phase=both` FT throughput ≥ `phase=prefill` FT throughput (the win
   condition).

**Known follow-ups (documented, not blocking).**
- Profiling pass extension: `profiling_batch_generator.py` shape sweeps
  don't cover `(B_d, K, T_in=0, T_ft>0)` and mixed shapes. γ converges via
  online refit (every 256 steps) but slower than profiling-seeded.
- Eager-penalty differential modeling: today's TBT check uses
  `t_current = predict(..., will_use_graph=False)` for the eager-with-FT
  prediction. A future refinement could compare against `predict(...,
  will_use_graph=True)` (no-FT graph time) to explicitly fight the
  graph-loss tax.
- `match_prefill_workload_factor` on decode-only: counter accumulates only
  on prefill-carrying steps. A future `match_any_workload_factor` could
  count decode tokens too. Out of scope per the "keep
  match_prefill_workload_factor" requirement.

## Phase 6.1 — Slice-based FT activation save ✅ (code)

Per-layer hooks (`input_layernorm`, `model.norm`, `mlp.gate_up_proj`) now use a slice
view `val[start : start + n]` on the fast path instead of `val[mask]` (an
`index_select` kernel + gather allocation). `_build_finetune_mask` also computes
`_ft_start` + `_ft_contiguous` (first/last True positions; contiguous iff
`last_excl - first == n`). `begin_step` + `accumulate_final` accept the new args;
the hooks branch on `_cur_contiguous` and fall back to the mask path silently when
the FT-True positions are interleaved with non-FT (e.g. an FT request lands in a
freed inference slot mid-batch — common is contiguous, but not guaranteed). Same
bytes either way; saves one CUDA kernel + one allocation per hook firing
(~33 per Llama-3 FT forward).

## Phase 5 — Optimizations & assets 🟡 (5.1, 5.2, 5.3, 5.4, 5.5 shipped; multi-TP + dedicated FT pool open)

- **Pause/resume `_maybe_pause` — the GPU-yielding co-serving contract. ✅ IMPLEMENTED (prefill-gated).**
  An `mp.Event` GPU-grant (SET = backward may run; CLEARED = yield) is created in
  `BackwardProcess` (`backward_process.py`, starts SET) and passed to the child via
  `service_main`. `BackwardService._maybe_pause()` (`bwd_services/base.py`) blocks on it (bounded
  `wait(timeout=5)`), called at **every layer boundary** in `Llama3BackwardService.process_backward`
  (`bwd_services/llama3.py:468`). The model runner (`gpu_model_runner.execute_model`) clears the
  grant around any forward that carries **prefill tokens** (`coord.gpu_pause_backward()` /
  `gpu_resume_backward()` via the coordinator) and leaves it set on decode-only steps — so a
  prefill pre-empts the backward within one layer's kernels, while decodes co-run. Prefill is
  detected from the scheduler-stashed `_ft_step_features.t_in > 0`. (opt service is loss-only / no
  per-layer loop, so it doesn't pause.)
- **Backward CUDA graphs** (padded-attention path; port `SFT_service_graph.py`, honoring the
  persistent-buffer rules for LoRA `.grad` / attention `ctx`). ✅ IMPLEMENTED (P5.2 in
  `VLLM_FORK_CHANGES.md`): per-layer FFN graph + single shared padded-attention graph,
  pre-captured at child startup. Flag: `finetune.backward_cuda_graph` (default off). Math is
  bit-identical to the eager path (gradcheck-verified by `tests/test_llama3_backward_graph.py`).
- **Backward latency:** batch/pad the per-sample attention loop into one kernel (the likely
  hotspot at small n); profile with the existing `[backward] remat-forward vs manual-grad` split.
  (Already done: bf16 bulk default, `gate_up` save skips the MLP recompute, dropped unused
  `down`/`out` recompute.)
- **Phase 5.3 — perf polish + admission strategies + bug fixes. ✅ SHIPPED.** Bundled set
  documented in detail in `VLLM_FORK_CHANGES.md` (P5.3 stage row + design note):
  - `match_prefill_workload_factor: float` — leaky-bucket FT admission strategy.
    Accumulates inference-prefill tokens seen but not yet "spent" on FT; admits ONE
    FT sample sized to the next sample's `input_len` (capped by SLO budget) when
    `(counter + t_in) * factor >= next_sample.input_len`. Factor scales how much
    credit each prefill token earns: `1.0` ≡ the original boolean-on behaviour,
    `>1` more aggressive, `<1` more conservative. Mutually exclusive with
    `ft_tokens_admission_constrain_factor`. Config:
    `finetune.match_prefill_workload_factor: <float>`. Default 0.0 (disabled).
  - **Oversized-sample drop at load** (`finetuning_store.py:load()`) — fixes a real
    deadlock when only samples with `input_len > max_saved_finetuning_tokens` remain
    in the pool. Surfaced by `pure_ft_bench.py` on `alpaca_1000.txt`.
  - **Fused AdamW** (`bwd_services/llama3.py` — `torch.optim.AdamW(..., fused=True)`)
    + **persistent grad_qh/kh/vh buffers** in `attn_backward_core` + **single shared
    padded-attn graph** in `Llama3GraphedBackward` (was per-layer; the core has no
    layer-specific weights so 32 captures was wasteful).
  - **One-shot `set_corpus_meta` IPC** replaces sending the corpus total on every
    `notify_buffer_full`. **Per-cycle log line** restructured to one line with
    `epoch=N processed/total tokens` progress.
  - **CUDA syncs** in `_handle_process_activations` coalesced 2 → 1 (timing now via
    CUDA events; cleanup sync drains the GPU before the events are queried).
  - **eval/auto_benchmark** output files tagged `_factor_<X>` (or `_factor_off` for
    `-1`); `eval/auto_plot` 5-panel layout with E2E latency percentile + p99
    highlighted + `--factor X` (auto-detects smallest) + factor in plot title;
    **`eval/pure_ft_bench.py`** new pure-FT (no inference traffic) benchmark.
- **Phase 5.4 — Forward-recompute CUDA graph (per layer). ✅ SHIPPED.** Captures the
  full per-layer forward rematerialization (RMSNorm in_ln + Q/K/V proj + RoPE +
  padded-attention forward + O proj + residual add) as one CUDA graph per layer,
  extending the Phase 5.2 backend from 2 captured regions to **3 per layer** under
  the same `finetune.backward_cuda_graph` flag. The forward graph's outputs ARE the
  FFN-bwd / padded-attn-bwd graphs' input buffers (writes directly to
  `static_resid_mid`, `static_gate`, `static_up`, `static_qh_pad/kh_pad/vh_pad`) —
  eliminates intermediate copy-in steps. Per-sample attention forward is replaced
  by a captureable padded variant mirroring `_padded_attn_core`. Pre-capture cost
  goes from 33 → 65 graphs at child startup (~few hundred ms; one-time). Per-layer
  silent eager fallback on capture/replay failure (`fwd_failed`) or when
  `_attn_fit=False`. Gradcheck parity 111/111 in
  `tests/test_llama3_backward_graph.py` (45 new assertions).
- **Phase 5.5 (a.k.a. F1) — save post-RoPE qh/kh/vh per layer. ✅ SHIPPED.** Opt-in
  via `finetune.save_attn_qkv: bool = False`. When ON, a `forward_pre_hook` on each
  `self_attn.attn` module captures the FT rows of post-RoPE q, k, v (vLLM's
  `self_attn.attn(q, k, v)` call site, args ≡ post-RoPE q, k, v). New per-layer
  buffers `attn_qh/kh/vh[i]` in `FinetuneAccumulator`, threaded through CUDA-IPC
  `share_activations`. The backward (both eager `layer_forward` and the Phase 5.4
  forward graph) short-circuits Q/K/V proj + RoPE entirely, only recomputing
  RMSNorm in_ln (cheap; the Q/K/V LoRA-A backward needs `x_norm1`).
  - **Cost:** +~96 MB at `max_saved_finetuning_tokens=256` on Llama-3-8B
    (`qh [s_max, q_size=4096]` bf16 ≈ 2 MB/layer × 32 = 64 MB; `kh/vh [s_max, kv_size=1024]`
    bf16 ≈ 0.5 MB/layer × 32 each = 16 MB each).
  - **Recovery:** ~13-16 GFLOPs/layer × 32 = ~400-500 GFLOPs eliminated per
    backward — ~5 ms on a 5090.
  - Per-layer silent eager fallback when `saved_gu` is absent for that layer.
    Forward graph variant captured at runner construction based on the
    `save_attn_qkv` flag (mode is fixed per-runner — graph captures the
    appropriate branch once).
  - Parity verified vs the recompute path in `test_layer_forward_saved_qkv_parity`
    + `test_forward_graph_saved_qkv_parity`.
  - **Trade-offs ranked** (`backward-review-issues.md` §F):
    | candidate | GFLOPs saved/layer | +MB | ratio (GFLOPs/MB) |
    |---|---|---|---|
    | **qh/kh/vh post-RoPE (shipped)** | 13-16 | 96 | **5.2** |
    | `mlp_gate_up` (already shipped) | 30 | 469 | 2.0 |
    | `ctx_flat` | 0.7 | 67 | 0.33 |
    | `x_norm1` alone | 0.005 | 67 | 0.002 |
- **Phase 5.6 — save the post-attention residual per layer. ✅ SHIPPED (2026-09-08).**
  Opt-in via `finetune.save_resid_mid: bool = False` (ON in every shipped YAML). A
  `forward_pre_hook` on each `post_attention_layernorm` captures the FT rows of
  `o + residual` — vLLM's fused add-norm is called with `(o_proj_out, residual)`, so it is
  the same `args[0] + args[1]` idiom as the `layer_in` hooks; under TP the o_proj output is
  already the RowParallelLinear-reduced full tensor. New per-layer buffers `resid_mid[i]`
  (`[s_max, hidden]`) in `FinetuneAccumulator`, in `buffers` / `zero_offset_range`, meta key
  `save_resid_mid`, trainer mirror + `saved_resid_mid=` threaded into `layer_forward` and
  `GraphedBackward.forward` for both families.
  - **Backward:** `layer_forward(saved_resid_mid=…)` skips the O projection (base + LoRA
    GEMM) + residual add; `ctx_flat` is still produced for the O-proj backward. In the graph
    runner the saved residual is staged straight into `static_resid_mid` (Graph A's input),
    the family core skips step 5, and — since there is no `o` reduce left — the forward tail
    is captured with the core again even under TP (`_forward_needs_reduce`). Eager fallback
    when the mode is on but a layer's saved residual is absent.
  - **TP win (the motivation):** the forward remat's `o` all-reduce was the only collective
    in the forward recompute → **7 → 6 collectives per layer** (Llama-3-8B: 224 → 192 per
    cycle, all in the backward). Counted per layer, per rank, per path in
    `tests/test_tp_backward_graph_nccl.py` (eager 7; graph 7; graph + saved 6, forward 0).
  - **Cost:** +64 MB on Llama-3-8B, +100 MB on Qwen3-14B at s_max=256 (bf16). Saves the
    ~8.6 GFLOPs/layer O-proj GEMM (~275 GFLOPs/cycle) on top of the reduce.
  - **Gates:** `tests/test_accumulate_hooks.py` 49/49 (hook level: fake vLLM-named decoder,
    contiguous + mask paths, non-zero offset, inert off-step, tier-C zeroing);
    `test_forward_graph_saved_resid_mid_parity` (llama3 graph test → 311/311; resid_mid alone
    and with qkv + ctx = in_ln-only recompute; missing-saved fallback) and
    `test_saved_resid_mid_forward_graph` (qwen3 → 217/217); NCCL layer gate 2680/2680 and
    NCCL trainer gate 212/212 (graph + save == graph exactly).
  - With `save_attn_qkv` + `save_attn_ctx` + `save_resid_mid`, the per-layer forward
    recompute is RMSNorm in_ln alone (the Q/K/V LoRA-A grads need `x_norm1`).
- **Phase 5.7 — LM-head restructure (batched rows, one conversion per chunk per pass).
  ✅ SHIPPED (2026-09-08).** A kernel-level profile of the real `process_backward` on a
  synthetic Qwen3-14B-shaped service (2 GPUs, NCCL, graph on, all saves on, no inference
  contention; `scratchpad/bwd_profile_qwen3.py`) showed the LM head at ~63 ms of a ~168 ms
  cycle: `head_backward` ran `logits_chunked` per SAMPLE, so the bf16 head was converted
  to fp32 nine times per cycle (~40 GB of copy traffic, 29 ms) and the fp32 GEMMs ran with
  ~31 rows each (34 ms, far below peak). Now all predicting rows are gathered into one
  `[n_valid, D]` matrix, each vocab chunk is converted once per pass (logits pass +
  `logit_grad @ W` pass), the shift-by-one targets / `n_valid` normalisation are row
  indexing. **Same math, same fp32 contract** (inputs, accumulation, softmax, CE all
  fp32; only GEMM tile order differs). One caveat found on the prod path: `rmsnorm`
  returns the input dtype, so the rows are upcast explicitly (the CPU gates run fp32 and
  could not see it). **Profile after:** wall 168 → 127 ms (head GEMMs 34 → 15 ms, copy
  kernels 36 → 16 ms); loss on the same synthetic inputs unchanged. Gates: gradcheck
  12/12 + 18/18, both overfit runs identical, NCCL trainer 212/212.
- A dedicated FT activation pool only if vLLM's allocator gets in the way; multi-TP
  correctness (backward per-rank). Profiling pass extension to cover `decode + FT`
  and `decode-only + FT` shapes (currently online-refit only). (eval/analysis
  tooling port: ✅ done in P5.3 above.)

## Phase 7 — Tensor parallelism (TP=2) 🟡 (M0–M4.2 GPU-validated; M5 + M4.3 2-GPU-gated, live A/B pending)

**Goal:** make the co-serving LoRA SFT backward correct and runnable under
`tensor_parallel_size > 1` for Llama-3. Guiding principle, mirroring the original
DeltaServe (`model_rpc.py` inits NCCL per rank and builds per-rank backward services):
**backward-per-rank** — each TP rank runs its own backward child on its own weight shard,
and the only cross-rank traffic is a small set of all-reduces on a **dedicated NCCL group
across the backward children** (they are not in vLLM's inference group).

Design invariant: FT tokens are identical across ranks (same injected prefill), so all
ranks fill their activation buffers in lock-step and fire the backward on the same step.
This keeps admission rank-local and makes the collectives safe — if one rank ran a
backward the other skipped, the all-reduce would deadlock.

`tp_size == 1` leaves every path below inert, so the single-GPU behaviour is unchanged.

### M1 — process launch ✅ (GPU-verified)

vLLM spawns `WorkerProc` with `daemon=True`, and Python forbids a daemonic process from
having children — so no rank could fork its backward child. Fixed by spawning the worker
**non-daemonic** iff `enable_finetuning` (`multiproc_executor.make_worker_process`).

That trade removes Python's automatic reaping, and the pre-existing death-pipe monitor
does **not** cover the gap: it is armed only *after* `WorkerProc.__init__` has run
`init_device()` + `load_model()` (~30 s), and on EOF it merely shuts two message queues
rather than terminating the process. Observed consequence: an EngineCore crash during
init left both workers reparented to PID 1 holding **14.7 GiB each**, which then failed
the *next* run with a misleading "free memory less than gpu_memory_utilization" error.
Fixed by arming `prctl(PR_SET_PDEATHSIG, SIGKILL)` at the top of `worker_main` (before
the model load) and escalating the death-pipe monitor to `os._exit(1)`; the backward
child arms PDEATHSIG too, since SIGKILL bypasses Python's cleanup entirely.

**Verified:** TP=2 server boots, two workers each with their own backward child on their
own GPU; clean Ctrl-C leaves no orphans; and a deliberately crashed EngineCore (bad
`data_path`, which dies *after* the executor is built — the original repro) now returns
both GPUs to idle with zero orphaned workers.

### M2 — shard-aware weights, dims, activation buffers ✅

The weights the worker shares are already this rank's shards; the bug was reading their
dims as full. Now `Hq`, `Hkv`, `inter` are **local** (`full // tp_size`) while
`hidden_size` (all-reduced residual stream) and `vocab` stay full. New module-level
`lora_shard_slice()` slices the disk-loaded (full) FT adapter into each rank's shard,
matching how vLLM shards the served LoRA buffers — q/k/v **B** on output rows, o **A** on
input cols, replicated factors whole — so the publish path needs **no** change. The
vocab-parallel `lm_head` is all-gathered to full at load, removing vocab parallelism from
the backward entirely (cost: ~1.05 GB resident per rank).

**Verified:** `tests/test_llama3_tp_shard.py` 23/23 (shard reconstruction rank0⊕rank1 ==
full, replicated factors whole, tp=1 identity, local-dim tiling of the fused qkv/gate_up).

### M3 — NCCL group + gradient all-reduces ✅

A second NCCL group across the backward children (`tcp://127.0.0.1:$DSERVE_BACKWARD_NCCL_PORT`,
default 29677). **7 all-reduces per layer**, and the set is surgical:

| Where | Tensor | Why |
|---|---|---|
| forward remat | `o` | `o_proj` is row-parallel → partial sum |
| backward | `grad_resid_mid` **minus** `grad_out`, then add back | reduce only the FFN partial; the residual passthrough is already full on every rank and would otherwise be counted `tp_size`× |
| backward | `grad_x` **minus** `grad_resid_mid`, then add back | reduce only the attention-path partial |
| backward | `grad_qA`, `grad_kA`, `grad_vA`, `grad_oB` | replicated factors — grads sum across shards |

`grad_{q,k,v}B` (output-sharded) and `grad_oA` (input-sharded) are already this rank's
correct shard and need no reduce. Reducing `grad_x` naively double-counts the residual;
the gloo test below caught exactly that and drove the surgical form.

**Verified:** `tests/test_llama3_tp_backward_gloo.py` 10/10 — a REAL 2-process gloo
collective shards a layer, runs each rank through the reduces, and asserts `grad_x` ==
reference (and identical across ranks), reduced replicated factors == reference, and
sharded factors concat == reference, all to ~1e-6.

### M4 / M4.1 — publish + control-plane bridge ✅ (GPU-validated)

Publish needed no code change: `_publish_to_served` sizes from `pa.shape`/`pb.shape`, and
M2 made the masters shard-shaped to match vLLM's already-sharded served buffers.

The real work was the control plane. Under TP>1 the scheduler (EngineCore) and the worker
are **different processes with different `get_coordinator()` singletons**, and only the
worker holds the backward IPC handle — so `_trigger_backward` silently no-op'd and no
backward ever fired. Fixed with a relay, all gated on `relay_mode = tp_size > 1`:

- scheduler → workers: `SchedulerOutput.finetune_backward_trigger` (**broadcast** — this
  is what guarantees lock-step, without which the collectives deadlock);
- workers → scheduler: `ModelRunnerOutput.finetune_saved` / `.finetune_backward_done` /
  `.finetune_ft_started`.

**Verified on 2× RTX 5090:** TP=2 and TP=1 track each other **cycle-for-cycle to ~0.01
loss** on identical data (same samples, same order, same hyperparameters, same epoch), and
a self-contained 2-GPU NCCL closed-loop test using the production `layer_forward` /
`layer_backward` / `head_backward` / `lora_shard_slice` and the exact optimizer+clip loop
matched tp=1 to **5 decimal places over 60 steps with zero replicated-master drift**.
TP=1 gradcheck 12/12 unchanged.

### M4.2 — SLO estimator + all-rank backward ack under TP ✅ (GPU-validated 2026-09-01)

**Problem.** The coordinator is a process-wide singleton. Under the multiproc executor the
runner's `push_sample` lands on the worker's coordinator while the scheduler drains its own
(always empty), so the estimator never became ready, `admit_ft_to_step` stayed on the
cold-start buffer-cap path, and `profile_on_launch` had to be off (the pass drained 0
samples). Two more signals were severed the same way: the profiling pass's
`record_timing` gate (never reached the worker) and the cudagraph dispatcher (the
scheduler stamped `was_graph=False` every step). Separately, only `output_rank`'s output
reaches the scheduler, so the backward ack reopened admission on rank 0's child while
rank 1's could still be publishing (the "ack race").

**Fix — extend the per-signal TP relay; estimator / admission / profiling shapes untouched.**
- `v1/outputs.py`: `ModelRunnerOutput.finetune_timing: list[tuple] | None`. The runner
  drains its coordinator queue onto it every `sample_tokens` (bounded queue); the scheduler's
  `update_from_output` pushes each tuple into its coordinator, so the existing `schedule()`
  drain → tracker → refit → validation CSV path runs unchanged. Samples describe a step ~4
  older (ring latency), irrelevant at the 256-step refit cadence.
- `v1/core/sched/output.py`: `SchedulerOutput.finetune_record_timing: bool = True`, stamped
  per step from the scheduler coordinator's flag; the runner's ring stores it (fallback: the
  local flag, so uniproc is unchanged).
- `gpu_model_runner.py`: the ring owner tuple now also carries the CUDA-graph mode the step
  **actually** ran with, which is what gets pushed as `was_graph`.
- Ack race: `coordinator.{relay_backward_outstanding, poll_own_backward_ack, take_relay_ack}`
  + `gpu_model_runner._ft_relay_backward_done`: each rank polls its own child into a stash,
  the ranks MIN-all-reduce a done flag over `get_tp_group().cpu_group` (gloo — no GPU sync;
  only while a backward is outstanding, i.e. a handful of tiny collectives per cycle), and
  every rank takes its ack on the same step, so rank 0 relays "done" only when both children
  have published. `poll_backward_relay` keeps its single-rank semantics.
- Idle steps relay too: `execute_model`'s 0-token early return (the output the scheduler
  sees for the steps `FinetuneScheduler.has_requests` keeps issuing while a backward is
  outstanding) now returns a fresh `ModelRunnerOutput` with the relay fields
  (`_ft_fill_relay_fields`, shared with `sample_tokens`). Found on the first TP timeline
  run: the ack only arrived with the next inference batch, so FT ran exactly one cycle per
  traffic burst and stalled through the idle valleys (~40 tok/s vs continuous on one GPU).
- `bwd_log` header: `coordinator._write_bwd_log_row` writes the header on an empty file
  too (the eval drivers truncate the log at launch); `eval-tp/repair_bwd_log.py` fixes
  logs from before that + applies the window trim.
- `bwd_services/base.py`: the `[backward] nanms` cycle time was a commented-out event read
  (on single GPU too); restored with an event-scoped `end_evt.synchronize()`.
- Both TP YAMLs: `profile_on_launch: true`; the M4.2 caveat comments replaced.

**Formula.** No TP term: TP halves the per-layer GEMMs and adds two all-reduces per layer
whose bytes scale with tokens (absorbed by β, δ) and whose fixed latencies land in `c`; the
backward child's PCIe traffic during FT lands in γ. The original DeltaServe estimator had
no TP term either (its tracker timed the whole rank fan-out from the router process, which
is exactly the single-owner-of-timings property the relay restores). The launch-time
profiling pass is unchanged and, running through the real executor, calibrates TP-specific
coefficients. Rank 0's timing suffices: the per-layer all-reduces keep ranks in lock-step.

**Verified (CPU):** `tests/test_tp_timing_relay.py` 22/22 — worker coord → pickle →
scheduler coord → tracker → estimator ready (both regimes fitted, eager prediction within
4% of the synthetic cost); `finetune_record_timing` survives the broadcast pickle and the
runner-side fallback prefers it; two fake ranks whose children ack at different steps relay
exactly once, on the same step, with rank 0's early ack held rather than lost. Existing
`test_merged_estimator`, `test_phase1_step{1,2}`, `test_config_loader` unchanged.

**Verified (GPU, Qwen3-14B TP=2, 2× RTX 5090).**
- The launch-time profiling pass drained **146 relayed samples** and fitted all three
  regimes (`inf_prefill=69 rmse=0.0011, eager=33 rmse=0.0010, decode_only=44 rmse=0.0008`);
  `[ft-profile] done: 146 samples` — it was 0 before M4.2.
- Admission is SLO-gated: per-step `ft=` varies with load (120 … 256) instead of sitting
  at the 256 cap; `[backward] <ms>ms` prints real cycle times (~630–750 ms eager on the 14B).
- **Idle-step relay (found on the first timeline run).** With the ack only relayed from
  `sample_tokens`, FT completed exactly one cycle per traffic burst — the ack reached the
  scheduler only when the next inference batch ran, so FT stalled through every idle
  valley (~40 tok/s on loose). After relaying on the 0-token early return too, cycles fire
  every ~0.35 s in the valleys and the FT band fills them as on a single GPU.
- Timeline replays (`eval-tp/auto_benchmark_tp.py`, `ttft_slo: 0.4`, `max_tbt_slo 0.05`,
  prefill-only admission, plots in `eval-tp/plots/`):

| Mode | Requests | TTFT p50 / p95 / p99 | TTFT ≤ 0.4 s | Inference tok/s | FT tok/s | Cycles | FT loss |
|---|---|---|---|---|---|---|---|
| loose | 240/240 ok | 77 / 405 / 537 ms | 95.0% | 255 | 467 | 89 | 1.72 → 1.32 |
| tight | 480/480 ok | 79 / 315 / 541 ms | 97.5% | 490 | 206 | 41 | 1.94 → 1.61 |
| nutanix 600–800 | 534/534 ok | 75 / 258 / 509 ms | 98.1% | 112 | 239 | 230 | 1.93 → 1.32 |

  The TTFT tail is the price of continuous FT: each burst's first requests wait behind an
  in-flight FT-only step on the 14B (≈0.3 s eager forward at 256 FT tokens). That is what
  `forward_interruptible` addresses and it is still off under TP — next-session item 3.
- Not yet run: the inference-only baselines (no grey overlay in the plots), the
  `validate_estimator` TP-vs-tp1 residual comparison, and the Llama-3 TP=2 regression
  A/B against the pre-refactor cycle times (the Llama-3 loose replay ran: 55 cycles,
  TTFT p50 ≈ 76 ms).

### M5 — backward CUDA graphs under TP ✅ (2-GPU NCCL gated 2026-09-08; live A/B pending)

The captured regions are per layer: forward remat, FFN-bwd (Graph A), padded-attn-bwd
(Graph B). Of the 7 TP all-reduces per layer only the forward remat's `o` (o-proj partial
sum) fell inside a capture. Landed as designed — **no collective is ever captured**:

- `common/graph.py`: new `static_o [s, D]` (the family core's last write) and a new
  `forward_tail()` (`resid_mid = layer_in + o`; gate/up split of the saved gate||up).
  `_forward_core` captures core + tail together at `tp_size == 1`, so the capture count and
  the tp=1 values are unchanged (llama3 165/165, qwen3 92/92 re-pass). Under TP `forward()`
  replays the core, then `all_reduce(static_o[:n])` — the real rows only; the tail rows are
  zero on every rank, and the shape (`[n, D]`, model dtype) matches the eager
  `layer_forward` reduce so a rank that fell back to eager for a layer stays lock-step —
  then runs the tail eagerly. The eager fallback inside `forward()` now also passes
  `all_reduce` (it silently skipped the o reduce before, but was unreachable under TP).
- `llama3.py` / `qwen3.py`: `graph_forward_core` stops at `static_o`;
  `layer_backward_graphed(..., all_reduce=None)` issues the six backward-side reduces at the
  same points as `layer_backward` (`reduce_partial` after Graph A, `reduce_partial` on
  `grad_x` + the four replicated-factor reduces at the end). **Correction to the design
  note above's item 2: they were NOT already there** — the graphed backward had no reduces
  at all, so dropping the guard alone would have trained silently on un-reduced gradients.
- `common/trainer.py`: the `tp_size > 1` force-eager branch is gone; the per-layer loop
  passes `self._all_reduce` to the graphed backward. The runner reads
  `svc._all_reduce` (set in `_build_state` before the runner is built).
- Invariants kept: lock-step (identical replay sequence on both ranks from the trigger
  broadcast; per-rank fallbacks — `fwd_failed` / `ffn_failed` / `attn_failed`, `_attn_fit`
  — cannot hang because every collective is issued from eager code in the same order and
  shape on both paths); `_maybe_pause` still between Graph A and B; no NCCL inside any
  graph, so `forward_interruptible` tier C cannot desync a capture.

**Gates (the "impossible 2-process variant" turned out to be possible — this box has two
GPUs):**
- `tests/test_tp_backward_graph_nccl.py` — 2 processes × 2 GPUs, real NCCL, per family ×
  {fp32, bf16} model dtype: graph-TP == eager-TP on each rank (cache + `grad_x` + all eight
  LoRA grads, 1e-5), reduced tensors identical across ranks, graph-TP == tp=1 reference
  (1e-4 fp32 / 3e-2 bf16, sharded factors reassembled), including a padded-attention
  OVERFLOW batch (the eager forward fallback must still reduce `o`). **1196/1196.**
- `tests/test_tp_trainer_graph_nccl.py` — the real `LoraSftTrainerService`
  (`_build_state` → NCCL group → `GraphedBackward` with `_all_reduce`;
  `process_backward` → clip → fused AdamW) for 4 training steps on a synthetic model, graph
  vs eager TP2 vs tp=1, both families: losses and masters graph == eager to 1e-7; with the
  clip off, TP2 == tp=1 to 1e-4 and rank 0 == rank 1. **132/132.** It also makes the M4.3
  clip trap reproducible (`--clip on`: TP2 drifts from tp=1 — see Open items).
- tp=1 unchanged: gradcheck 12/12 + 16/16; gloo TP 10/10 × 2; shard 23/23 + 42/42.

**Same session, P5.6 (`save_resid_mid`)** took the forward remat's `o` reduce out
entirely: with it on (both TP YAMLs) the per-layer collective count is 6, all in the
backward, and the forward tail is captured with the core again. Both NCCL gates cover the
mode (layer gate now 2680/2680 incl. per-layer collective counts; trainer gate 212/212).

**Pending:** the live A/B on the real models (Next step item 1): `(graph)` cycle lines,
cycle loss equal to eager cycle-for-cycle, ~5 ms/cycle (dispatch) saved — less than
M4.3's ~20 ms of PCIe traffic, which is still the bigger lever.


### Open items

- **M5 live A/B** on Llama-3 / Qwen3-14B TP=2 (Next step item 1).
- ~~**`set_corpus_meta` never reaches the children under TP**~~ (the child's meter
  printed `N/?`). Fixed 2026-09-08: `FinetuneScheduler` stores the total on
  `coord.corpus_total_tokens`; under relay mode `_trigger_backward` puts it on every
  trigger command (one int) and the worker's `execute_trigger` calls the child's
  `set_corpus_meta` once, ahead of the work signal on the same ordered pipe, so the very
  first cycle line reads `N/total`. `tests/test_tp_timing_relay.py::test_corpus_meta_relay`.
- ~~**Latent: the per-layer `clip_grad_norm_` is rank-asymmetric.**~~ Fixed by M4.3
  (below): the clip runs after the bucketed reduce with a rank-symmetric norm.

### M4.3 — gradient bucketing + comm stream + rank-symmetric clip ✅ (2026-09-08)

**Motivation, measured.** A kernel-level profile of the real `process_backward` on a
synthetic Qwen3-14B shape (2 GPUs, NCCL, graph + all saves, no inference) put the 240
all-reduces per cycle at 37 ms of an ~125 ms cycle, and a microbenchmark gave 0.36 ms per
2.5 MB reduce / 0.045 ms per 160 KB reduce across the two 5090s (no P2P, host-staged
PCIe). 160 of the 240 were the replicated LoRA-factor grads, which nobody reads before
the optimizer. Every reduce was also issued on the compute stream with the child pinned
to one hardware queue (`CUDA_DEVICE_MAX_CONNECTIONS=1`), so no overlap was possible.

**What landed** (`bwd_services/common/tp.py`, `common/trainer.py`, both families,
`backward_process.py`):
- `FactorBucket`: one persistent flat buffer (25 MB on Qwen3-14B, 16 MB on Llama-3-8B,
  compute dtype, outside any graph pool) holding every layer's `qA/kA/vA/oB` grads in
  backward order, so each group of `BUCKET_LAYERS=8` consecutive layers is one contiguous
  slice. The trainer `stash`es a layer's four grads as they are produced and submits the
  group's slice when its last layer is done.
- `CommQueue`: the dedicated CUDA stream + the ordering, in one place. `submit` records
  an event on the compute stream and makes the comm stream wait on it (reduce-after-
  produce), issues the collective, records a completion event; `wait` makes the compute
  stream wait on all of them (consume-after-reduce). Only persistent buffers are
  submitted, rewritten no earlier than the next cycle (no overwrite in flight). CPU /
  gloo: synchronous. Debug: `DSERVE_BWD_COMM_SYNC=1` waits after each submit;
  `DSERVE_BWD_COMM_DELAY=<cycles>` spins the comm stream before each collective so a
  missing wait reads stale data deterministically.
- `clip_layers_symmetric_`: the per-layer clip with `clip_grad_norm_` semantics but the
  norm over the FULL parameter set — replicated grads (identical on every rank) + the
  sharded grads' squared norms summed across ranks with one `[L]` all-reduce — so both
  ranks derive the same coefficient. Runs after `comm.wait()`; tp=1 keeps the original
  per-layer `clip_grad_norm_` unchanged.
- Families: `layer_backward` / `layer_backward_graphed` take `reduce_factors=True`; the
  trainer passes `False` under TP (the default keeps the M3 gloo tests and the
  layer-level NCCL gate unchanged). `bucket_layers` is meta-configurable (tests use 2
  over 3 layers).
- The child no longer sets `CUDA_DEVICE_MAX_CONNECTIONS=1` (DeltaServe's MPS ordering
  aid; the two-stream design carries explicit event dependencies instead).

**Result.** Qwen3-14B TP=2: 240 → 86 collectives per cycle (80 residual + 5 buckets + 1
clip vector); uncontended cycle 123 → 121 ms — the bucket reduces overlap the layer
loop, and the 80 residual reduces (~29 ms) are the TP floor on this box. The main win is
correctness: TP=2 now equals tp=1 with the clip firing.

**Gates.** `tests/test_tp_bucket_gloo.py` 92/92 (CPU, 2-process gloo, real trainer, 3
layers / 2 buckets, clip firing: TP2 masters + losses == tp1 to 1e-4, rank0 == rank1,
12 collectives per cycle). `tests/test_tp_trainer_graph_nccl.py` 220/220 (2 GPUs: graph /
eager / graph+save / graph+sync / graph+delay — the async comm path bit-identical to the
sync and delayed modes, TP2 == tp1 with the clip on, counts 12 / 12 / 9 per cycle).
`tests/test_phase1_step2.py` updated for the removed env var. Every tp=1 gate unchanged.

### forward_interruptible under TP (rank-symmetric tier C) ✅ (2026-09-08)

**Why it was inert.** Tiers A (pre-schedule grace poll) and B (post-schedule rollback) run
in the EngineCore process and were TP-safe from the start. Tier C's abort signal was a
`threading.Event` on the *engine's* coordinator, set by the input thread; under TP the
workers check their *own* coordinators in other processes, so the event never reached
them — and had it reached one rank only, that rank would have left the FT-only forward
while the other waited in the next layer's TP all-reduce forever.

**Design.**
- `coordinator.FtArrivalSignal`: an 8-byte POSIX shared-memory counter. `EngineCore.__init__`
  creates it before the executor spawns the workers (when finetuning + `forward_interruptible`
  + tp>1) and exports its name as `DSERVE_FT_ARRIVAL_SHM`; the input thread bumps it on every
  ADD; `shutdown()` unlinks it. Written through `struct` (no persistent buffer export, which
  would make `SharedMemory.close()` raise at exit).
- `coordinator.FtAbortPoller` (one per worker, built in `gpu_worker` with the TP group's gloo
  `cpu_group`): `entry()` at the start of an FT-only forward, `layer()` at each layer
  boundary — each MAX-all-reduces the rank's local "counter moved" bit, so all ranks take the
  same decision at the same layer; one CPU collective per layer, no GPU sync. `note_served()`
  on inference-bearing batches and `finish()` after the FT-only forward account for arrivals
  that have been scheduled. **Active only between `entry()` and `finish()`**: the accumulator
  hooks are armed for every FT-bearing batch, including co-serving batches with inference
  tokens, and the first live run aborted 291 of those (their inference requests lost the step)
  — `hook_check` is inert when the poller is inactive and runs no collective.
- `accumulate.py`: hooks call `_abort_poll(boundary)`; only the `input_layernorm` pre-hook
  passes `boundary=True` (one collective per layer); tp=1 installs `Event.is_set` for every
  hook as before.
- `ModelRunnerOutput.finetune_aborted`: a real dataclass field for the sentinel (the dynamic
  `_ft_aborted` attribute is lost across the worker → engine hop). The engine's
  `_ft_output_aborted` checks it on the model output and, under `step_with_batch_queue`, on
  the execute_model future too; the runner's `sample_tokens` returns the sentinel (with the
  TP relay fields) when execute_model left a pending abort, since under TP the engine still
  calls it. The runner logs `[ft-abort] tier C: FT-only forward (n tokens) aborted at
  entry|layer k for a late inference arrival`.
- Backward child: `service_main` exits quietly on `KeyboardInterrupt` (server teardown
  forwards SIGINT; the two shutdown tracebacks in every log were that).
- `eval-tp/ft_bench_tp.py`: the inference `ok` counter was broken (tasks were dropped from the
  in-flight set by a done-callback before the loop could read them → "0/N ok" in every run);
  counted in the callback now.

**Two more findings from the tight-trace replays, both fixed the same day.**
- *The burst's first request was slow even with tier C on.* Most burst starts land while
  the **backward** is running, not during an FT-only forward (the FT-only forward is ~60 ms
  of a ~430 ms cycle), so nothing was there to abort — and the prefill itself ran at
  80–200 ms instead of 40 ms. Cause: the `_maybe_pause` grant was re-set right after the
  prefill was *enqueued* ("fire-and-forget", designed for MPS), so without an MPS daemon the
  child resumed while the prefill was still executing and the two contexts time-sliced.
  Fix: `finetune.pause_until_prefill_done` (opt-in; ON in the TP YAMLs) — the runner records
  a CUDA event behind the forward and resumes the child only once it has fired
  (`_ft_maybe_resume_backward`, a non-blocking `query()` on every execute_model incl. idle
  steps and on sample_tokens; the child's 5 s cap bounds any missed resume).
- *Aborts clustered at layers 31–36.* Partly the 4 rps bench's periodicity (FT-only forwards
  start at a fixed phase after each request), partly that the hooks run at CPU launch time
  while the GPU trails. `_LaunchAheadGate` (in both pollers, incl. the new tp=1
  `LocalAbortPoller`) waits on the event of the layer launched two boundaries ago before
  each check, so the decision tracks GPU progress and an abort stops real GPU work.

**Tight-trace A/B (Qwen3-14B TP=2, 8 bursts of 60 requests, TTFT SLO 0.4 s):**

| run | TTFT sat % | p95 | p99 | max | first request of each burst (ms) | FT tok/s |
|---|---|---|---|---|---|---|
| inference-only | 100.0 | 85 ms | 86 ms | 113 ms | 40 41 40 39 38 40 38 41 | — |
| co, interruption off | 99.2 | 97 ms | 377 ms | 577 ms | 92 52 144 53 44 40 458 58 | 277 |
| co, interruption on | 99.2 | 101 ms | 391 ms | 524 ms | 111 115 77 198 84 101 84 72 | ~266 |
| co, off + pause fix | 100.0 | 103 ms | 172 ms | 196 ms | 64 44 144 50 39 42 173 50 | — |
| **co, on + pause fix** | **100.0** | **86 ms** | **91 ms** | **150 ms** | **55 60 67 61 61 60 59 64** | **262** |

Loose and nutanix-600-800 (before these two fixes, graphs + saves + M4.3 + head): TTFT
satisfaction 98.3 / 99.4 % at 642 / 392 FT tok/s (1 Sept: 95.0 / 98.1 at 467 / 239);
inference-only 100 / 100 %; co-serving TTFT p95 306 / 162 ms vs 84 / 76 ms — the tail that
the fixes above address.

**Gates.** `tests/test_ft_abort_tp.py` 10/10 (2-process gloo: quiet forward → no abort and
1 + L collectives per rank; an arrival only rank 0 observes → BOTH ranks abort at the same
layer with equal collective counts; an arrival before the forward → both abort at entry;
`finish` / `note_served` clear it; inactive poller inert with no collective; the sentinel
field survives pickle). `tests/test_accumulate_hooks.py::test_abort_poll_boundary` (raise from
the boundary hook, rows consistent up to the aborting layer). Live (Qwen3-14B TP=2 `ft_bench`,
4 rps): both ranks armed on the same signal; every `[ft-abort]` pair reports the same layer.

### Validation 2026-09-08 (evening) — both families, full stack, no MPS

Stack: graphs (M5) + all three activation saves + head restructure + M4.3 bucketing +
`forward_interruptible` (rank-symmetric tier C) + `pause_until_prefill_done`, no MPS daemon.
Timeline replays via `eval-tp/auto_benchmark_tp.py`, inference-only vs co-serving.

| family / trace | TTFT SLO | inf-only sat % · p50 / p95 | co sat % · p50 / p95 / p99 | TBT p50 inf → co | FT tok/s |
|---|---|---|---|---|---|
| Qwen3-14B loose | 0.4 s | 100 · 72 / 84 ms | 98.3 · 77 / 306 / 439 ms | 25.7 → 26.5 ms | 642 |
| Qwen3-14B tight | 0.4 s | 100 · 74 / 85 ms | 100 · 75 / 86 / 91 ms | 28.6 → 28.9 ms | 262 |
| Qwen3-14B nutanix 600–800 | 0.4 s | 100 · 63 / 76 ms | 99.4 · 73 / 162 / 368 ms | 18.0 → 19.1 ms | 392 |
| Llama-3-8B loose | 0.25 s | 100 · 46 / 53 ms | 99.2 · 98 / 192 / 246 ms | 15.1 → 26.5 ms | 1088 |
| Llama-3-8B tight | 0.25 s | 100 · 47 / 53 ms | 96.9 · 121 / 240 / 295 ms | 15.4 → 36.2 ms | 641 |
| Llama-3-8B nutanix 600–800 | 0.25 s | 100 · 39 / 48 ms | 98.5 · 69 / 193 / 266 ms | 10.6 → 18.9 ms | 814 |

(Qwen3 loose / nutanix were run before the pause fix; tight after. 1 Sept Qwen3 baselines:
95.0 / 98.1 % at 467 / 239 tok/s.) Llama-3's tight run shows the co-serving cost most
clearly: burst-start TTFT 38–65 ms vs 27–28 ms inference-only, but the REST of each
burst runs at 128 ms median TTFT and 36 ms TBT (vs 47 / 15 ms) — Llama-3's SLO gate
(0.25 s) admits FT into almost every prefill of a burst and every such step runs eager.
Zero NaN, zero tracebacks, no stuck backward in any run; losses 5.77 → 1.36–1.62 (Llama-3),
2.93 → 1.4 (Qwen3).

**Known-issues sweep (same evening).** *FT loss divergence in the loose-co run* — not
reproduced in any of today's six co-serving replays (all descend monotonically apart from
epoch-boundary jumps); closed. *Runner `self.requests` leak for FT requests* — fixed: the
scheduler stamps `SchedulerOutput.finetune_retired_req_ids` (retired in `update_from_output`
or rolled back) and `_update_states` drops those entries + input-batch slots. *Dead-child
wedge (C7)* — the coordinator now DISABLES finetuning with an error when the child is dead
(`is_alive()` False after the 5 s warning) or unresponsive for 60 s, in both the tp=1 poll
and the TP relay path, instead of leaving admission silently closed. *`test_config_loader`*
— the one failing check expected the hub model id verbatim; vLLM rewrites it to the local
snapshot path under `HF_HUB_OFFLINE`, the test accepts that form now (32/32). *avg-TBT
admission gate* — still deferred (design item, needs per-request last-token tracking).

### M5 notes — why the collectives stay outside the graphs

**NCCL-in-a-captured-region is not forbidden** — NCCL ≥ 2.9 supports graph
capture, PyTorch's `ProcessGroupNCCL` supports it, and Megatron-LM captures TP all-reduces
inside per-layer graphs routinely. It requires eager PG init + a warmup collective, and
identical capture/replay order on every rank — which DeltaServe's broadcast trigger
already guarantees. Two genuine obstacles remain: (1) `_maybe_pause` must stay *outside*
any captured region, so unlike an FT-only system we can never collapse to one graph per
step; (2) `forward_interruptible` tier-C aborts unilaterally mid-backward, which would
desync the collective and hang both ranks — the abort would have to become a collective
decision at a layer boundary.

**Higher-value work first (call it M4.3):** on this box there is no P2P, so all 224
collectives (~218 MB/rank/cycle) cross PCIe — order ~20 ms, against the ~5 ms of *dispatch*
overhead that graphing removes. Standard FT-system practice (DDP/FSDP gradient bucketing)
applies directly: the 128 replicated-factor grads (4 × 128 KB per layer) are read only by
the optimizer, so they can be deferred into **one flat ~16 MB all-reduce** after the layer
loop — **224 → 97 collectives** — and the 96 that remain on the critical path can overlap
with compute on a separate comm stream. Doing so also forces the per-layer clip to move
after the reduce, which fixes the rank-asymmetry above.

### Assets

- `configs/serving_config_finetuning_llama3_tp2.yaml` — TP=2 co-serving config.
- `eval-tp/launch_deltaserve.py` — HTTP launcher (streams logs, waits on `/health`).
- `eval-tp/auto_benchmark_tp.py` — the timeline (real-workload) benchmark under TP: the
  replay / CSV / trim helpers are imported from `eval/auto_benchmark.py`, the server comes
  from the family-aware launcher (`--family`, `--tp`, `--co` for co-serving vs baseline).
  Modes `--loose` / `--tight` / `--nutanix` / `--nutanix-600-800` (the 600–800 s slice of the
  original Nutanix trace) from `eval/timelines/5090/`. Per-family, per-TP, per-mode output
  names in `eval-tp/output/`; `--kill-stale` pre-flight as in `ft_bench_tp.py`.
- `eval-tp/auto_plot_tp.py` — the 4-panel figure for eval-tp runs (`eval/auto_plot.py`'s
  builder, which gained optional `timeline_csv` / `infonly_csv` / `title` overrides and
  re-bases the FT cumulative counter so warmup tokens no longer spike the first bin).
- `eval-tp/ft_bench_tp.py` — TP co-serving FT bench. Writes **per-family, per-TP** filenames
  (`bwd_log_{family}_tp{N}.csv`, `server_{family}_tp{N}.log`; P8) and **truncates** the bwd log per run (the
  server opens it in append mode, so runs used to silently concatenate). Counts cycles
  from the CSV, not the server log — under TP every rank prints its own `[backward]` line,
  so log-line counting reports `tp_size`× the real count. `--kill-stale` +
  a pre-flight check refuse to launch when a previous run left GPU-resident processes.
- `tests/test_llama3_tp_shard.py`, `tests/test_llama3_tp_backward_gloo.py` (both are
  standalone scripts — run with `python`, not pytest, which is not installed in the env).

### Estimator validation under TP (2026-09-08, night) — the per-step trace

**Tooling.** `finetune.step_trace_path` (new) makes the FT scheduler write one CSV row per
timed step — realized features, the RAW and margined prediction, what the admission loop
reasoned on (`est_*`, `t_baseline`, `t_admit`, `ttft_slack`, `queue_wait`), the CUDA-event
GPU time, the host dispatch time, `pending_bwd` (scheduler) / `paused_bwd` (runner), and
occupancy — plus one row per rolled-back FT-only step. Unlike `validate_estimator` it does
not change the system under test (async scheduling stays on). Cost: ~4 µs per step on the
scheduler thread (measured; `tests/test_step_trace.py`), file I/O on a daemon thread that
flushes once per second; off → one `is None` check. Only the step sequence number rides on
the SchedulerOutput; the runner returns `(seq, t_exec, host_s, paused)` in the existing
timing tuple. `eval-tp/auto_benchmark_tp.py --step-trace` writes
`eval-tp/output/step_trace<suffix>.csv`; `eval-tp/analyze_step_trace.py --family … --tight`
joins it with the results / bwd_log / server log and prints the per-regime accuracy, the
interference splits, admission fidelity, the worst misses and every TTFT violation
(`--plot` for the 3-panel PNG).

**Result (Qwen3-14B TP=2, full stack, tight + nutanix-600-800, both 100 % TTFT).** The
model is accurate for every step the admission gate reasons on: eager (FT-carrying) steps
ratio p50 1.00 / p99 1.06, RMSE 1.3 ms; clean inference prefill 0.99 / 1.16, RMSE 2 ms;
admission's own prediction for the admitted set vs actual p99 1.06, max 1.36. Every
systematic miss has one root cause — **the backward child sharing the GPU (driver
time-slicing, no MPS):**

| steps | tight | nutanix 600–800 |
|---|---|---|
| decode-only, backward in flight (never paused by design) | ratio p50 1.45 · p99 2.9 (n=25) | p50 1.87 · p90 2.72 · p99 9.0 (n=518); 16-ms steps of 100–195 ms |
| inference prefill, backward in flight, child *paused* | p50 1.09 (+3 ms) | p50 1.15 · p90 1.70 · p99 2.26 (+8 ms mean); 1.39 right after the FT-only step that triggered the backward |
| everything else | 1.00 | 0.99 (decode 0.92 — see below) |
| worst-TBT > 50 ms (co vs inference-only) | 115 vs 26 of 480 | 184 vs 15 of 534 |

43 of the 45 decode steps > 50 ms on nutanix had a backward in flight. The contended
samples also enter the online refit: the decode-only RMSE climbs 0.7 → 4.6 ms across
the fit log and the *clean* decode steps end up over-predicted by 8 % (ratio 0.92). The
backward pays too: cycle p50 148 ms but p90 313 ms (uncontended 121). Secondary
findings: (i) admission on this box is **TBT-bound, not TTFT-bound** — every prefill step
that admitted nothing (438 on tight, 239 on nutanix) had decodes present and an eager
co-serving step costs ~50 ms ≈ `max_tbt_slo`, so only the 23-token samples ride
co-serving steps and FT mostly runs as 253-token FT-only steps in idle gaps; (ii) eager
steps are host-blocked (host ≈ GPU, growing with tokens 45 → 63 ms) — launch-queue
back-pressure, i.e. GPU-bound, not a CPU problem; (iii) "wrong schedule" is not a factor:
the realized composition differs from what admission reasoned on in 5 % of admitted
steps (a decode finished in between — conservative), rollbacks 15 / tier-C aborts 14 on
nutanix. Proposed fixes are in the session write-up: tag contended samples (exclude from
the admission fit, fit a contention factor), a TBT-aware pause on decode-only steps (or
MPS), and bounding the child's GPU run-ahead so a pause takes effect within a layer
(today `_maybe_pause` only stops enqueueing; the graphed loop has no host sync, so the
GPU keeps running what was already queued).

**Acted on the same night (Qwen3 only, TTFT SLO 0.4 s).** (1) The estimator got a fourth
regime, `decode_bwd` (`StepFeatures.bwd`, stamped from `coord.pending_backward`): a
decode-only step with a backward in flight predicts with its own coefficients (falls back
to `decode_only` until its first refit); prefill-carrying steps taken with a backward in
flight are excluded from every fit. Clean decode went from ratio 0.92 to 1.00 on nutanix;
`decode_bwd` is centred (p50 0.94-0.99 where it has samples) but keeps a wide spread
(p90 1.3, p99 3.7 — the contention is bimodal). Pausing the child on decode-only steps
is out of scope by decision. (2) The pause: `backward_run_ahead_boundaries` (CUDA-event
ring, default 2) alone made the loose trace WORSE (paused prefills 1.95×, 2.5 s cycles,
5 violations, 97.9 %) — the two children then blocked at different boundaries and the
rank that had issued the next residual all-reduce spun that NCCL kernel on its GPU for
the whole pause. The pause is now rank-symmetric (gloo agreement at every boundary,
`tp.backward_cpu_group()`), and the LM head's vocab-chunk loops are boundaries too.
Result, all three traces, TTFT satisfaction **100 / 100 / 100 %** (loose / tight /
nutanix-600-800) at co-serving TTFT p50/p95/p99 of 75/95/132, 76/94/121, 74/115/178 ms
(inference-only 72/84/86, 74/85/86, 63/76/81); paused prefills with a backward in flight
run 1.07-1.10× p50 / 1.23-1.37× p90 (was 1.09-1.95 p50 / 1.24-2.10 p90); backward
cycle max 456 ms (was 2558 on loose); FT tokens unchanged (30.8k / 13.5k / 77.6k per
window). The residual +6-7 ms on the first prefill after the trigger is the run-ahead
depth plus the agreement latency. Worst-TBT > 50 ms stays at 57 / 60 / 188 requests vs
26 / 1 / 15 inference-only — the decode-only contention, by decision untouched. Gates:
`tests/test_merged_estimator.py` 49/49, `tests/test_tp_trainer_graph_nccl.py --family
qwen3` 111/111 (run-ahead 2 + the agreement path).

---

## Phase 8 — Qwen3 family + backward-service restructure ✅ (Qwen3-14B TP=2 GPU-validated; 0.6B smoke + Llama rope_theta re-check done 2026-09-08)

**Goal.** Add `Qwen3ForCausalLM` (Qwen/Qwen3-14B-Base at TP=2 on the 2× 5090; Qwen/Qwen3-0.6B-Base
single-GPU for smoke tests) as a second co-served + LoRA-finetuned family, and restructure
the backward service so shared logic lives in its own modules and each family has its own
file — with Llama-3 numerically identical and no slower, and upstream-vLLM edits minimal
and generic (no Llama/Qwen branches in vLLM files).

**What Qwen3 adds.** Exactly one op: a per-head `RMSNorm(head_dim)` on q and k (`q_norm` /
`k_norm`, frozen) between the qkv projection and RoPE. Module names, fused qkv/gate_up,
GQA, SwiGLU, the residual layout, the LoRA layer classes and the TP shard geometry are
identical to Llama-3; the norm acts within a head, so the 7 all-reduces per layer are
unchanged. Its backward inserts an fp32 `rmsnorm_backward` between the RoPE backward and
the q/k projection backward and needs the PRE-norm q/k — which the `self_attn.attn`
pre-hook does not see (it captures post-norm post-RoPE). Originally the family therefore
declared `supports_saved_qkv=False` and recomputed Q/K/V; **since 2026-09-08 the q/k save
hooks the `self_attn.q_norm` / `k_norm` INPUTS instead** (`Family.saved_qkv_pre_transform`,
`ft_meta.saved_qkv_pre_transform`, `FinetuneAccumulator(attn_qkv_pre_transform=True)`),
which is exactly the tensor the norm backward needs; the remat skips the Q/K/V GEMM and
re-applies only the elementwise norm + RoPE — eager `layer_forward` and
`graph_forward_core` alike. Gates: `tests/test_qwen3_backward.py` (saved pre-norm qkv
== autograd), `tests/test_qwen3_backward_graph.py::test_saved_qkv_pre_norm` (qkv-only and
all-saves → 475/475), `tests/test_accumulate_hooks.py` (Qwen-style fake with
`q_norm`/`k_norm`; disables itself when the norm modules are missing → 147/147),
`tests/test_tp_backward_graph_nccl.py` `save_all` mode (4020/4020). Qwen3 now matches
Llama-3: with all three saves the per-layer forward recompute is in_ln alone.
Qwen3-0.6B additionally has `Hq*Hd = 2048 ≠ hidden = 1024` and a tied
`lm_head`; both were latent bugs on the Llama-only code and are fixed generically.

### Layout (`dserve-vllm/vllm/deltaserve/bwd_services/`)

| Module | Holds |
|---|---|
| `common/ops.py` | `rope_cos_sin`, `apply_rope`, `rope_backward`, `rmsnorm`, `rmsnorm_backward`, `proj`, `proj_backward` |
| `common/attention.py` | `attn_forward_core` (per-sample GQA loop, extracted from `layer_forward`), `attn_backward_core` |
| `common/ffn.py` | `ffn_forward_tail` (gate/up or saved), `ffn_backward_core` |
| `common/head.py` | `head_backward`, `logits_chunked` |
| `common/tp.py` | `lora_shard_slice`, `init_backward_tp_group`, `reduce_partial` (the residual-aware reduce) |
| `common/family.py` | `Family` dataclass: `layer_forward` / `layer_backward` / `layer_weights` map / `graph_forward_core` / `layer_backward_graphed` / `supports_saved_qkv` |
| `common/trainer.py` | `LoraSftTrainerService`: `_build_state` (family weight map, `meta["lm_head_key"]`), `process_backward`, `_publish_to_served`, DIAG, verify |
| `common/graph.py` | `GraphedBackward` (was `llama3_graph.py`, `git mv`): family forward core delegated; `static_ctx_flat` sized `[s, q_size]` |
| `llama3.py` | full `layer_forward` / `layer_backward` / `graph_forward_core` / `layer_backward_graphed` + `LLAMA3` + `Llama3BackwardService` |
| `qwen3.py` | the same with the q/k-norm insertion; its own `graph_forward_core` (norm between projection and RoPE, writes the runner's `static_q_pre`/`static_k_pre`) so all three graph regions run |
| `registry.py` | arch/alias → `"module:Class"` (lazy import), `is_trainer`, `get_family` |
| `../ft_meta.py` | `build_backward_meta` (the `meta` dict, moved out of the worker), `rope_theta_of`, `read_lora_scaling`, `effective_save_attn_qkv` |

Family functions are bound once in `_build_state` and called directly on the hot path;
`reduce_partial` only runs when `all_reduce is not None`; family modules import lazily
(a Llama child never imports `qwen3.py`).

### Milestones + gates

| Q | Scope | Gate | Status |
|---|---|---|---|
| Q0 | `rope_theta` fix (`ft_meta.rope_theta_of`) | Llama-3 DIAG remat error → bf16 noise; `pure_ft_bench` reproduces the 2.12 reference | ✅ GPU-verified 2026-09-08 (rel err 1e-4…5e-3 per layer, 6e-2 at the last layer, stable across cycles; loss 5.77 → 2.10 by cycle 30 → ~1.0) |
| Q1 | Restructure with Llama-3 bit-parity | all five Llama gates at their counts | ✅ gradcheck 12/12, shard 23/23, gloo 10/10, overfit (99.9% drop), graph parity 165/165 |
| Q2 | Qwen3 family + CPU tests | gradcheck incl. q/k-norm and `q_size≠hidden`; shard at 0.6B/14B geometry; real-gloo TP; overfit with tied head | ✅ 16/16, 42/42, 10/10, overfit 98.5% drop |
| Q3 | Generic vLLM unblocks | Qwen3 + finetuning lands on the v1 runner; publish gate via registry; `meta` via `ft_meta` | ✅ code (`test_phase1_step1.py` 8/8) |
| Q4 | Configs + family-aware eval-tp | `--dry-run` for all three presets prints the right `serve` cmd | ✅ |
| Q5 | GPU ladder | 0.6B smoke → Llama TP=2 regression → Qwen3-14B TP=2 | ❌ user's next run |

**Upstream vLLM edits (both generic).** `config/vllm.py:_get_v2_model_runner_unsupported_features`
appends `"DeltaServe co-serving finetuning"` when `enable_finetuning` (routes every
finetuning run to the v1 runner; an explicit `VLLM_USE_V2_MODEL_RUNNER=1` now errors instead
of silently serving FT-less). `v1/worker/gpu_worker.py`: the served-LoRA publish gate is
`registry.is_trainer(arch)` instead of `arch == "LlamaForCausalLM"`; the inline `meta` dict +
adapter-scaling read became `ft_meta.build_backward_meta` / `read_lora_scaling`; the
accumulator's `save_attn_qkv` goes through `ft_meta.effective_save_attn_qkv` (off only for a
family whose backward cannot consume saved q/k at all) and `ft_meta.saved_qkv_pre_transform`
(which hook stage to capture: post-RoPE at `self_attn.attn`, or the `q_norm`/`k_norm` inputs).

### Verified (CPU)

- `tests/test_llama3_{backward,tp_shard,tp_backward_gloo,train_overfit,backward_graph}.py`
  — 12/12, 23/23, 10/10, pass, 165/165 after the refactor (imports only). The overfit test
  had been broken at HEAD (unpacked `layer_forward`'s dict as a pair) and was repaired.
- `tests/test_qwen3_{backward,tp_shard,tp_backward_gloo,train_overfit}.py` on the shared
  `tests/bwd_harness.py` — 16/16, 42/42, 10/10, pass.
- `tests/test_phase1_step1.py` 8/8 (config plumbing incl. the v2-runner entry).
- `eval-tp/{launch_deltaserve,ft_bench_tp}.py --dry-run` for `llama3`, `qwen3-14b`, `qwen3-0.6b`.

### GPU ladder — status

All four rungs done (2026-09-08): (1) Llama-3 tp=1 DIAG + `pure_ft_bench` — remat error at
bf16 noise, stable across cycles; 863 cycles / 100 s, 2097 FT tok/s, loss 5.77 → 2.10 at
cycle ~30 → ~1.0 (the old stall at ~4.3 is gone); (2) Qwen3-0.6B single-GPU smoke — 654
cycles / 60 s, loss 4.40 → ~0.8 into epoch 1, 240/240 inference OK, after fixing the tied
`lm_head` resolution (the LoRA-wrapped embedding is named `…embed_tokens.base_layer.weight`
on the worker; the child looks keys up by their normalized name, so `lm_head_key` came out
None); (3) Llama-3 TP=2 co-serves loose / tight / nutanix-600-800 with the full stack; (4)
Qwen3-14B TP=2 likewise — tables under Phase 7 "Validation 2026-09-08 (evening)".

### GPU ladder (original plan; expected outcomes)

1. **Llama-3-8B tp=1, DIAG**: `DSERVE_TP_DIAG=1 python eval/pure_ft_bench.py` → the
   `[tpdiag] … remat-vs-captured rel|d|` lines should read ~1e-2 or below at every layer
   (was 0.22 → 0.45); the loss curve should track the 2.12 reference.
2. **Qwen3-0.6B single GPU**: `python eval-tp/ft_bench_tp.py --family qwen3-0.6b --tp 1
   --duration 60` → `Qwen3BackwardService` banner, `[backward]` cycles with finite,
   decreasing loss, inference 200s; with `DSERVE_TP_DIAG=1` the remat error stays small
   (validates the q/k-norm remat, theta=1e6, the tied head, `q_size≠hidden` in one run).
3. **Llama-3 TP=2 regression**: `--family llama3 --tp 2` → same cycle time and loss trend
   as the pre-refactor run (`eval-tp/output/`).
4. **Qwen3-14B TP=2**: `--family qwen3-14b --tp 2 --duration 120 --kill-stale` → both
   ranks print `TP group up`, cycles fire, finite decreasing loss, no hang; record per-rank
   `nvidia-smi` headroom (config starts at `gpu_memory_utilization: 0.80`).

### Follow-ups (scoped, not started)

- ~~**Exact `save_attn_qkv` for Qwen3**~~ — landed 2026-09-08, via the simpler route:
  pre-hooks on `self_attn.q_norm`/`k_norm` save the PRE-norm q/k (same memory as the
  Llama-3 save; no `inv_rms` bookkeeping), v from the attn pre-hook; the remat re-applies
  norm + RoPE. See the Phase 8 section above.
- ~~Qwen3 `graph_forward_core`~~ — landed: `common/graph.py` gained `static_q_pre`/`static_k_pre`
  (+ `cache_views` keys), `qwen3.graph_forward_core` writes them; parity vs eager in
  `tests/test_qwen3_backward_graph.py` (forward cache incl. `q_pre`/`k_pre`, graphed vs
  eager backward grads, the `save_attn_ctx` branch). Graphs under TP landed in M5.
- Broader `deltaserve/` regrouping (scheduling / data / backward subpackages) — deferred
  because it would churn vLLM import strings for no Qwen3 benefit.

---

## Top risks

- **Forward-reimpl fidelity for the backward (open)** — the P3.3 per-layer recompute must
  match vLLM's Llama bit-for-bit enough that gradients are correct; the precision rules + the
  per-layer verify are the mitigations.
- **Pause/resume `_maybe_pause` — wired (prefill-gated). ✅** The backward now yields the GPU to
  any forward carrying prefill tokens via the `mp.Event` grant, bounding TTFT during a backward.
  Remaining latency-under-load tuning (the loose-co spike) and the FT loss divergence are open.
- **Editing vLLM internals** — `Scheduler`/`GPUModelRunner`/model defs aren't stable APIs;
  keep changes localized (tagged `[DeltaServe]`) so rebases stay tractable.
- Resolved earlier: cross-process CUDA IPC (Phase 1 ✓), forward-hook capture vs compile
  (eager invariant ✓), full-logits-vs-hidden-state (hidden-state ✓), FT leaking into
  decode/KV/sampling (`FinetuneScheduler` retire ✓).
