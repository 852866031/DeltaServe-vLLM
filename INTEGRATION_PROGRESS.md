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

**Phase 7 (TP=2) is the active line.** M0–M4.1 are done and GPU-validated on 2× RTX
5090: TP=2 trains equivalently to TP=1 (cycle-for-cycle to ~0.01 loss on identical
data). The next items, in priority order, are in the **Phase 7** section below —
(1) **M4.2**, relaying `push_sample` so SLO-aware admission works under TP at all
(today TP runs with no SLO gate); (2) the backward **ack race** (admission reopens on
one rank's ack); (3) **M4.3** gradient bucketing + a comm stream, which is worth more
than M5 on a box with no P2P and also fixes the latent rank-asymmetric gradient clip.

**Open, and NOT a TP problem:** FT loss stalls around ~4.3 after ~25 cycles where an
earlier `pure_ft` run reached ~2.6 on the same samples — but this **reproduces
identically at TP=1**, so it is a config- or regime-level training-quality issue, not a
tensor-parallelism bug. Ruled out so far: learning rate (lowered to 5e-6 back in May,
before the good run), the data (identical sample sequence and order), code drift (no
commits touched the FT path after Jun 7), and `save_attn_qkv`/`save_attn_ctx`. An
env-gated diagnostic (`DSERVE_TP_DIAG=1` → `Llama3BackwardService._diag_remat_check`)
rematerializes the full forward from `layer_in[0]` and compares it against the captured
activations; it shows the remat diverging from vLLM's real forward (final_in relative
error 0.22 → 0.45 over cycles) **identically at TP=1 and TP=2**. Next step there is to
re-run `eval/pure_ft_bench.py` and confirm the 2.12 reference still reproduces — note
that curve was only 32 cycles over 5.0 s, so it may be a weaker reference than it looks.

Phase 6 (`forward_interruptible` + slice activation save) remains code-complete and
pending GPU validation on a co-serving replay; it is **off** under TP (and conflicts
with M5 — see the Phase 7 notes). Phases 5.2–5.5 have shipped. The unified-phase
scheduler (`coserving_admission_phase: both`) has shipped as code; GPU A/B is pending.
From here on we **prioritize Llama-3**; opt-125m stays at its current stage (loss-only)
as a reference path.

---

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
- A dedicated FT activation pool only if vLLM's allocator gets in the way; multi-TP
  correctness (backward per-rank). Profiling pass extension to cover `decode + FT`
  and `decode-only + FT` shapes (currently online-refit only). (eval/analysis
  tooling port: ✅ done in P5.3 above.)

## Phase 7 — Tensor parallelism (TP=2) 🟡 (M0–M4.1 done + GPU-validated; M4.2 / M5 open)

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

### Open items

- **M4.2 — SLO-aware admission under TP (the biggest functional gap).** The runner pushes
  timing samples to the *worker* coordinator while the scheduler drains its *own*, so the
  estimator never receives them and online refit is dead under TP. `profile_on_launch` is
  therefore **off** in the TP config (with it on, profiling drains 0 samples and fits a
  degenerate estimator that then reads as "ready" and rejects all FT admission). TP
  consequently runs on the cold-start fixed-budget path with **no SLO gate at all** —
  i.e. the project's core value-add is not exercised under TP. Fix: relay `push_sample`.
- **Backward ack race.** `execute_model` collects `ModelRunnerOutput` from a single
  `output_rank`, so admission reopens on **one** rank's ack while the other child may
  still be publishing or zeroing buffers. Bounded by the per-layer NCCL lock-step, but
  real.
- **`set_corpus_meta` never reaches the children under TP** (guarded on
  `backward_process is not None`, which is `None` on the scheduler coord) → the child's
  per-epoch progress meter renders `N/?`.
- **Latent: the per-layer `clip_grad_norm_` is rank-asymmetric.** It is computed over a
  rank-local tensor set (full replicated grads + only *this* rank's half of the sharded
  ones), so each rank derives a slightly different clip scale and applies it to the
  *identical* reduced replicated grads — slowly desynchronizing the replicated masters.
  Dormant today only because the per-layer norms do not exceed `max_norm=1.0` (measured
  divergence when they do: ~0.08%/step). Folding the clip after a bucketed reduce fixes
  it — see M5 notes.
- **M5 — backward CUDA graphs under TP** (see below).

### M5 notes — and why it should not be next

`finetune.backward_cuda_graph` is force-disabled at `tp_size > 1` (silent eager
fallback). **The config comment claiming the blocker is "full-dim capture buffers" is
wrong**: `llama3_graph.py` derives every dimension from the service (`svc.Hq`, `svc.inter`,
…), which M2 already made local, and its `D`-sized buffers are residual-stream tensors
that are legitimately full. The real blocker is that the captured regions have no
all-reduces wired in.

Note that **NCCL-in-a-captured-region is not forbidden** — NCCL ≥ 2.9 supports graph
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
- `eval-tp/ft_bench_tp.py` — TP co-serving FT bench. Writes **per-TP** filenames
  (`bwd_log_tp{N}.csv`, `server_tp{N}.log`) and **truncates** the bwd log per run (the
  server opens it in append mode, so runs used to silently concatenate). Counts cycles
  from the CSV, not the server log — under TP every rank prints its own `[backward]` line,
  so log-line counting reports `tp_size`× the real count. `--kill-stale` +
  a pre-flight check refuse to launch when a previous run left GPU-resident processes.
- `tests/test_llama3_tp_shard.py`, `tests/test_llama3_tp_backward_gloo.py` (both are
  standalone scripts — run with `python`, not pytest, which is not installed in the env).

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
