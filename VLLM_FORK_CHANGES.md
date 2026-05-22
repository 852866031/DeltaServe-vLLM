# DeltaServe fork — changes from plain vLLM

Every change we make to the upstream vLLM tree (`vllm/`), organized by stage.
This is the "what did we touch and why" manifest; `INTEGRATION_PROGRESS.md` is the
"how far along / how verified" tracker. **Keep this in sync** as new changes land.

Scope: the `vllm/` source tree only (changes *from plain vLLM*). Project-root
tooling that *consumes* these (configs, tests, launcher, adapters) is listed at the
bottom for cross-reference but is not itself a vLLM change.

Two kinds of change:
- **NEW** files — net-new code, all under `vllm/deltaserve/` (+ one config dataclass).
  These never existed upstream; documented with **Function** and **Used by**.
- **MODIFIED** upstream files — small, localized edits to existing vLLM code, marked
  with `[DeltaServe]` comments in-source for easy grep/rebase.

---

## Stage index (which files each stage touched)

| Stage | New | Modified |
|---|---|---|
| 1 — config flag + logging + YAML loader | `deltaserve/__init__.py`, `deltaserve/config_loader.py`, `config/finetune.py` | `config/__init__.py`, `config/vllm.py`, `engine/arg_utils.py`, `v1/worker/gpu_worker.py` |
| 2 — backward stub process + MPS | `deltaserve/backward_process.py` | `v1/worker/gpu_worker.py` |
| 3 — share FT-adapter + base weights (CUDA IPC) | — | `config/finetune.py`*, `deltaserve/backward_process.py`*, `v1/worker/gpu_worker.py` |
| P2.1 — finetuning sample store | `deltaserve/finetuning_store.py` | `config/finetune.py`* |
| P2.2 (M1) — FT injection + mask + force-eager | `deltaserve/ft_injector.py`, `deltaserve/ft_scheduler.py` | `v1/request.py`, `v1/core/sched/output.py`, `config/vllm.py`, `v1/worker/gpu_model_runner.py` |
| P2.3 (M2) — activation buffers + per-layer accumulation + hash | `deltaserve/accumulate.py` | `deltaserve/backward_process.py`*, `v1/worker/gpu_worker.py`, `v1/worker/gpu_model_runner.py` |
| P2.4 — co-serving coordinator (fill tracking + admission) | `deltaserve/coordinator.py` | `deltaserve/backward_process.py`*, `deltaserve/accumulate.py`*, `deltaserve/ft_scheduler.py`*, `v1/worker/gpu_worker.py`, `v1/worker/gpu_model_runner.py` |
| P2.5 — HTTP experiment + observability (debug section, batch log, backward sleep) | — | `config/finetune.py`*, `deltaserve/config_loader.py`*, `deltaserve/backward_process.py`*, `deltaserve/coordinator.py`*, `v1/worker/gpu_worker.py`, `v1/worker/gpu_model_runner.py` |
| P3.1 — per-model backward services + logits/loss/logit-gradient | `deltaserve/bwd_services/__init__.py`, `deltaserve/bwd_services/base.py`, `deltaserve/bwd_services/opt.py` | `deltaserve/backward_process.py`*, `deltaserve/coordinator.py`*, `v1/worker/gpu_worker.py`, `v1/worker/gpu_model_runner.py` |
| P3.2 — pivot to Llama-3: llama3 loss service + residual-stream capture | `deltaserve/bwd_services/llama3.py` | `deltaserve/accumulate.py`*, `deltaserve/bwd_services/base.py`*, `deltaserve/bwd_services/opt.py`*, `deltaserve/backward_process.py`*, `v1/worker/gpu_worker.py` |
| P3.3 — real manual LoRA backward + optimizer (llama3) | `tests/test_llama3_backward.py` | `deltaserve/bwd_services/llama3.py`*, `deltaserve/bwd_services/base.py`*, `config/finetune.py`*, `v1/worker/gpu_worker.py`, `deltaserve/coordinator.py`*, `deltaserve/ft_scheduler.py`*, `deltaserve/backward_process.py`* |
| P3.4 — precision flag + served-weight publish + epoch flush + gate/up save | — | `config/finetune.py`* (`backward_fp32`), `deltaserve/bwd_services/{base,llama3}.py`* (cdt, publish), `deltaserve/backward_process.py`* (`share_lora_buffers`), `v1/worker/gpu_worker.py` (`_maybe_share_ft_served_lora`, accumulator `intermediate_size`), `deltaserve/coordinator.py`* (`flush_partial`), `deltaserve/ft_scheduler.py`* (epoch flush), `deltaserve/accumulate.py`* (`mlp_gate_up` capture) |

| P4 — SLO-aware admission + execution-time estimator | `deltaserve/estimator.py`, `deltaserve/profiling_batch_generator.py`, `tests/test_merged_estimator.py`, `tests/test_profiling_shapes.py` | `config/finetune.py`* (SLO + profiling fields), `deltaserve/config_loader.py`* (`slo` section folded into FinetuneConfig), `deltaserve/coordinator.py`* (`last_step_s`, `cudagraph_dispatcher` refs), `deltaserve/ft_scheduler.py`* (estimator/tracker, SLO budget gate, graph predicate via dispatcher, online record+refit, profiling hooks, stats dump), `v1/worker/gpu_model_runner.py` (CUDA-event step timing), `v1/engine/core.py` (`profile_execution_model` + launch call), `configs/serving_config_finetuning_{opt,llama3}.yaml`* (`slo` section) |

| P5.1 — `_maybe_pause` GPU-yield contract (prefill-gated) | — | `deltaserve/backward_process.py`* (`_gpu_grant` mp.Event + `set_pause`), `deltaserve/bwd_services/base.py`* (`service_main` arg + `_maybe_pause`), `deltaserve/bwd_services/llama3.py`* (per-layer `_maybe_pause` call), `deltaserve/coordinator.py`* (`gpu_pause_backward`/`gpu_resume_backward`), `v1/worker/gpu_model_runner.py` (pause around prefill forwards) |
| P4b — async scheduling enabled (reserve-at-inject) | — | `config/vllm.py`* (async default ON for FT, was force-off), `deltaserve/ft_scheduler.py`* (inherit `AsyncScheduler`; reserve scheduled-FT rows + stash per-step write offset; per-step duration read; epoch-flush request + hold admission), `deltaserve/coordinator.py`* (`reserved_fill`, `reserve`, `request_epoch_flush`/`try_epoch_flush`, triggers gated on `reserved==0`), `v1/worker/gpu_model_runner.py` (use stashed write offset; stash per-step `_ft_step_duration`), `v1/engine/core.py`* (profiling `reset_coord` clears new fields) |
| P4c — deferred timing + control plane + fixes | `entrypoints/serve/finetune/api_router.py` (POST `/start_finetuning`) | `deltaserve/coordinator.py`* (`ft_started`+`start_finetuning`, `bwd_log` writer, `_trigger_backward` no-op while profiling, `try_epoch_flush` no longer gated on admission_open — epoch-flush deadlock fix), `deltaserve/ft_scheduler.py`* (deferred CUDA-event timing → coordinator queue drain; `has_requests` refined to not spin on a stuck partial buffer; FT-partition only touches this-step injects — async leak fix; TTFT queue-wait term; `ft_started` gate), `deltaserve/backward_process.py`* (`notify_buffer_full`/`poll_response` tolerate dead child at shutdown), `deltaserve/bwd_services/base.py`* (`_total_tokens_trained` in backward log), `v1/worker/gpu_model_runner.py` (deferred timing ring; `[batch]` log occupancy + admit-state + wall-clock ts; `deltaserve_start_finetuning` worker RPC in `gpu_worker.py`), `config/finetune.py`* (`bwd_log_path`, `start_on_launch`), `entrypoints/serve/__init__.py` (attach finetune router) |

`*` = same file extended in a later stage.

> **P5.1:** the backward yields the GPU at every layer boundary while the main process runs an
> inference **prefill** forward (TTFT-critical); decode-only steps let the backward co-run. The
> grant is an `mp.Event` (SET = may run); the runner clears it around prefill forwards.

> **P4 design notes:** one **merged** 6-param step estimator
> `T ≈ α·S + β·T_in + γ·T_ft + δ·B_d + ε·K + c` (vLLM runs one mixed prefill+decode
> batch, so prefill+decode estimators collapse into one). γ kept in BOTH eager and
> graph regimes (future graphed co-serving). Graph regime is queried from vLLM's real
> `CudagraphDispatcher` via the shared coordinator singleton — no mirror. Offline
> profiling runs synthetic batches through the live scheduler at launch
> (`EngineCore.profile_execution_model`, before `run_busy_loop`). Async scheduling
> remains force-off (P4b/Part E, not yet implemented): the blocker is the activation-
> buffer fill accounting, fixable with reserve-at-inject. See
> `.claude/plans/ok-write-a-plan-optimized-lighthouse.md` for the full plan.

> **Rename (P2.5):** what was `deltaserve/capture.py` / `FinetuneCapture` / `capture_final`
> is now `deltaserve/accumulate.py` / `FinetuneAccumulator` / `accumulate_final`
> ("accumulate", to avoid confusion with CUDA-graph capture). Older entries below use
> the current names.

---

## NEW files

### `vllm/config/finetune.py` — `FinetuneConfig`
**Function:** the single config dataclass for co-serving (analogue of DeltaServe's
`finetune.*` YAML section). Fields: `enable_finetuning` (master gate),
`backward_mps_percentage`, `finetuning_lora_path` (FT adapter), `data_path`,
`num_epochs`, `max_prepare`, `max_saved_finetuning_tokens`, `backward_sleep_seconds`,
the SLO knobs (`ttft_slo` / `avg_tbt_slo` / `max_tbt_slo`, grouped under a `slo:` YAML
section) + **`ft_tokens_admission_constrain_factor`** (P4e: cap FT tokens ≤
`prefill_tokens · factor` per prefill step; `-1` disables), and the debug knobs
`print_weight_hash` / `print_activation_hash` / `print_step_mode` (grouped under a
`debug:` YAML section). `slo` + `debug` keys are folded into FinetuneConfig by the loader.
**Used by:** `config/__init__.py` (export), `config/vllm.py` (attached as
`VllmConfig.finetune_config` + read in `__post_init__`), `engine/arg_utils.py`
(`EngineArgs` field + `--finetune-config` CLI), `deltaserve/config_loader.py`,
`deltaserve/ft_injector.py`, `deltaserve/ft_scheduler.py`, `v1/worker/gpu_worker.py`,
`v1/worker/gpu_model_runner.py`.

### `vllm/deltaserve/__init__.py` — logging helpers
**Function:** package home for all net-new code. Provides `dprint(msg)`
(green in main process / purple in the backward subprocess, TTY-guarded,
`[deltaserve]`-prefixed) and `mark_backward_process()` (flips dprint to purple).
**Used by:** essentially every other deltaserve module + `gpu_worker.py` +
`gpu_model_runner.py`. `mark_backward_process()` is called by `backward_process.py`.

### `vllm/deltaserve/config_loader.py` — YAML → EngineArgs loader
**Function:** reads a DeltaServe-style sectioned YAML and maps it onto vLLM's
`EngineArgs` + `FinetuneConfig`. `finetune` (+ `debug`, folded in) → `FinetuneConfig`;
`server`/`adapters` → returned `extras` dict; every other section → `EngineArgs`
kwargs. Resolves relative path-valued keys to absolute. Functions: `load_yaml_config`,
`split_config`, `build_engine_args`, `engine_args_from_yaml`, `print_loaded_config`.
**Used by:** `scripts/launch_deltaserve.py`, `scripts/ft_experiment_{opt,llama3}.py`,
`tests/test_config_loader.py`, `tests/test_phase1_m1.py`. (Purely a launcher/test-side helper; not imported by the
running engine.)

### `vllm/deltaserve/backward_process.py` — the backward (SFT) subprocess (parent side)
**Function:** spawns and talks to the second GPU process. `BackwardProcess`
(`start`/`ping`/`shutdown`/`share_weights`/`checksum`/`share_activations`/
`hash_activations`/`notify_buffer_full`/`poll_response`) spawns a `daemon` child via
`torch.multiprocessing` (CUDA-IPC reductions), wraps `.start()` with the MPS env so
only the child gets the constrained partition, and shares GPU tensors zero-copy. The
child entry point + recv loop now live in `bwd_services/` (P3.1); `start()` imports
`bwd_services.base.service_main` lazily and spawns it with the model `service_name`.
Also keeps the cross-process hashing helpers
`weight_hash_report`/`activation_hash_report`/`print_hash_report`/`_tensor_hash`/
`_checksum`/`_summarize_weights`. `share_weights(..., meta=)` carries the LM-head key /
vocab size / logit scale; `notify_buffer_full(n, sleep_s, sample_lens)` carries the
per-sample token counts. Hash printing is gated by the
`print_weight_hash`/`print_activation_hash` debug flags (off ⇒ the backward stays quiet).
**Used by:** `v1/worker/gpu_worker.py` (spawns it, shares weights + meta);
`coordinator.py` (signals it); `bwd_services/base.py` (imports the hashing helpers);
`tests/test_phase1_step2.py`, `tests/test_phase1_step3.py`.

### `vllm/deltaserve/bwd_services/` — per-model backward services (child side)
**Function:** the child backward process's recv loop + per-model SFT math (P3.1; split
out of `backward_process.py`). `base.py` — `BackwardService` (model-agnostic loop:
`ready` handshake, ping/shutdown, `share_weights`/`share_activations` IPC mappings,
hash debug cmds, `process_activations`) + the **shared** `_logit_loss_and_grad`
(reconstructs full logits `final_hidden @ lm_head.weight.T` fp32, trims padded vocab;
next-token CE vs `concat_input_ids` shift-by-1 per sample, no prompt masking; CE logit
gradient `softmax − one-hot` normalized) + an optional `verify_activations` hook +
`service_main(conn, mps, dev, service_name)` (child entry: mark process purple, bind
device, build service, run) + `get_service` factory (arch → service; `LlamaForCausalLM`
→ llama3, `OPTForCausalLM` → opt, else `NotImplementedError`). The `process_backward`
hook (default = loss-only via `_logit_loss_and_grad`; `is_trainer` services skip the
simulated sleep) is called by `process_activations` (optionally `verify_activations`
first) before the buffer clean; epoch threaded in from `notify_buffer_full`.
`opt.py` — loss-only (`compute_loss_and_grad` → `_logit_loss_and_grad` on `final_hidden`).
**`llama3.py` (P3.3)** — the real **manual** LoRA SFT backward: on weights-received builds
the fp32 master LoRA params (per layer/proj q/k/v/o A/B), slices the fused base weights,
and builds `AdamW`+`StepLR`; `process_backward` does `zero_grad` → head (`head_backward`)
→ per-layer remat (`layer_forward`) + hand-derived grads (`layer_backward`) → per-layer
clip → `optimizer.step` → `StepLR` on epoch increment. Module-level math helpers
(`rmsnorm`/`rmsnorm_backward`, `apply_rope`/`rope_backward`, `_proj`/`_proj_backward`,
`layer_forward`/`layer_backward`, `head_backward`) are gradchecked in
`tests/test_llama3_backward.py`. `verify_activations` (debug-gated) checks
`layer_in[0]≈embed`, `RMSNorm(final_in)≈final_hidden`.
**P3.4:** `layer_backward(cdt=…)` runs the bulk matmuls in the model dtype (bf16) by default
or fp32 if `meta.backward_fp32` (attention core / RMSNorm / LM-head always fp32);
`layer_forward(saved_gate_up=…)` skips the gate_up matmul using the captured `mlp_gate_up`
(and no longer recomputes the unused `down`/`out`); after `optimizer.step`,
`_publish_to_served()` writes the fp32 master into the IPC-shared vLLM served buffers
(clamp+cast bf16, ×scaling on B, at the pinned slot) so inference uses the trained weights.
**Used by:** `backward_process.py` (`BackwardProcess.start()` spawns `service_main`).
Reference: DeltaServe `models/{llama,llama3}/SFT_service.py`.

### `vllm/deltaserve/finetuning_store.py` — FT sample store
**Function:** loads + tokenizes a corpus (one sample/line) and serves samples by
length-bucketed selection. `FinetuningSample` (tokenized, prefill-only) +
`FinetuningStore` (`load`, `pop_best_under`, `pop_next`, `confirmed_trained`,
`advance_epoch`, `has_next`). Pure Python; port of DeltaServe's `FinetuningManager`
data layer.
**Used by:** `deltaserve/ft_injector.py`; `tests/test_finetuning_store.py`.

### `vllm/deltaserve/ft_injector.py` — builds FT Requests
**Function:** `FinetuneInjector` owns a `FinetuningStore` + the FT `LoRARequest`
(reserved id 1000). `next_ft_requests(token_budget)` greedily packs samples up to a
token budget into vLLM `Request`s (`max_tokens=1`, FT lora, `is_finetuning=True`),
then marks them trained so the next step draws fresh samples.
**Used by:** `deltaserve/ft_scheduler.py`.

### `vllm/deltaserve/accumulate.py` — `FinetuneAccumulator`
**Function:** allocates fixed-size GPU buffers and captures the FT-token rows of the
**residual stream** (P3.2; replaced the earlier opt-specific output capture). Buffers:
per-layer `layer_in[i]` `[max_saved_finetuning_tokens, hidden]` (residual entering layer
i) + `final_in` (pre-final-norm residual) + `final_hidden` (post-norm) + `concat_input_ids`.
Capture points are auto-detected by module name and hooked with `register_forward_pre_hook`:
`layers.{i}.input_layernorm` → `layer_in[i]`, `model.norm` → `final_in`. The fused
add-norm means the pre-hook sees `(hidden,)` (layer 0) or `(hidden, residual)` (i>0), so
the residual = `args[0]` or `args[0]+args[1]` (copied immediately — the op may update
`residual` in place). On an FT step (`begin_step` arms the mask + accumulating offset)
the pre-hooks copy FT-token-only rows at `[offset:offset+n]`; `accumulate_final` saves
`final_hidden` + ids. No-ops off FT steps; buffers live outside any CUDA-graph pool.
Models without the fused pattern (opt) register no pre-hooks → only `final_hidden` +
`concat_input_ids`.
**P3.4 — `mlp_gate_up[i]` capture:** also discovers `layers.{i}.mlp.gate_up_proj` and a
forward (post) hook copies its output (`[n, 2·intermediate]` = gate‖up) per layer (sized via
`intermediate_size`). This lets the backward skip the gate_up matmul (the layer's widest,
frozen matmul). llama3-only (opt has no `gate_up_proj`).
**Used by:** `v1/worker/gpu_worker.py` (allocates it, registers hooks, shares buffers
with the backward process, injects into the runner); `v1/worker/gpu_model_runner.py`
(drives `begin_step`/`accumulate_final`/`end_step` per FT step).

### `vllm/deltaserve/coordinator.py` — `FinetuneCoordinator`
**Function:** process-wide singleton holding the FT activation-buffer fill state +
admission gate. `fill_count` is the write offset / fill level vs `capacity =
max_saved_finetuning_tokens`. `next_ft_budget()` (**`per_step_budget = capacity`** as of
P4e — was `0.5·capacity`; capped by free space, 0 while backward pending) tells the
scheduler how much FT to admit; `current_offset()` + `record_capture(n, sample_lens)`
track accumulation and signal the backward when full — forwarding `sample_lens` via
`notify_buffer_full` (closing admission); `poll_backward()` reopens admission once the
backward reports done (resetting `fill_count` + `sample_lens`).
**P3.4:** carries `current_epoch` (forwarded on `notify_buffer_full(epoch=)`); the backward
fires on buffer-full OR a flush flag (below).
**P4d/P4e:**
- `note_injection(next_sample_len)` — called by the scheduler after each FT injection with
  the **peek-next** smallest-untrained sample length (`store.pop_next()`, `None` if epoch
  drained). Raises the flush flag (`epoch_flush_pending`) when the buffer can't grow (epoch
  drained OR next sample won't fit the free space / would overflow); the backward trigger
  (`record_capture` / `try_epoch_flush`) consumes it and `_trigger_backward` unsets it. Fixes
  the idle buffer-wedge (was stuck at e.g. 208/256 because a static-corpus-min trigger never
  fired). Replaces the old `flush_partial()`.
- `_trigger_backward` waits on a **capture-completion event** (`capture_done_evt`, recorded by
  the runner after the activation copies) instead of a full-device `torch.cuda.synchronize()`.
- `gpu_pause_backward`/`gpu_resume_backward` are plain `mp.Event` toggles (fire-and-forget).
- `inf_req_count` (bumped by EngineCore per ADD; drives the `[engine-recv] #N` log) and
  `ft_start_time` (set by `start_finetuning`; drives the `[batch … t=+Xs]` since-start timer).
**Used by:** `v1/worker/gpu_worker.py` (creates it, sets `backward_process`, injects
into the runner), `v1/worker/gpu_model_runner.py` (offset + `record_capture`),
`deltaserve/ft_scheduler.py` (`next_ft_budget`/`poll_backward`/`flush_partial`).

### `vllm/deltaserve/ft_scheduler.py` — `FinetuneScheduler(Scheduler)`
**Function:** subclass that injects FT Requests into the scheduler queues before
`super().schedule()` (gated on real work present), records scheduled FT ids in
`SchedulerOutput.finetune_req_ids`, cleans up any unscheduled FT, and in
`update_from_output` retires FT via `_free_blocks` *before* the base loop — so they
free KV the same step and never produce an `EngineCoreOutput` (invisible to the
frontend).
**P4 (SLO admission):** `_slo_ft_budget` computes the per-step FT budget from the predicted
step time vs the TTFT/max-TBT SLOs (estimator in `deltaserve/estimator.py`), `min`'d with the
coordinator's buffer-space budget. **P4e:** after the budget, an optional cap
`ft_tokens_admission_constrain_factor` (config) limits FT tokens to `prefill_tokens · factor`
on prefill-carrying steps (`-1` disables); and the scheduler calls `coord.note_injection(pop_next)`
to drive the flush flag.
**Used by:** the running engine — selected by qualname string in
`config/vllm.py.__post_init__` (`scheduler_config.scheduler_cls`) and instantiated by
`EngineCore` (`v1/engine/core.py:132/145`). Not imported directly anywhere.

---

## MODIFIED upstream files

### `vllm/config/__init__.py`
Stage 1. Import + `__all__`-export `FinetuneConfig`.

### `vllm/config/vllm.py`
- Stage 1: import `FinetuneConfig`; add `finetune_config: FinetuneConfig` field to
  `VllmConfig` (default factory).
- Stage P2.2 / P4b: in `__post_init__`, when `finetune_config.enable_finetuning` and no
  `scheduler_cls` set, select `"vllm.deltaserve.ft_scheduler.FinetuneScheduler"`. (P2.2
  originally forced `async_scheduling = False`; **P4b made async the default** for
  co-serving — only set `async_scheduling = True` when it's still `None`, made safe by
  reserve-at-inject buffer accounting.)

### `vllm/engine/arg_utils.py`
Stage 1. `EngineArgs` gets a `finetune_config` field, a `--finetune-config` CLI arg,
and passes it into the constructed `VllmConfig`. (Mirrors the `profiler_config`
pattern; the one V0-adjacent file both engines share.)
Stage P4d: in `create_engine_config`, when `finetune_config.enable_finetuning` and the
user didn't set it, **force `self.disable_log_stats = True`**. vLLM attaches per-step
`scheduler_stats` to the rank-0 API frontend's output stream; processing it every engine
step saturates that frontend's asyncio loop and stalls HTTP accept + SSE streaming (the
real cause of the co-serving TTFT spikes). Mutating `self` here propagates to every
`engine_args.disable_log_stats` read (engine + frontends) since `create_engine_config`
runs first. The SLO estimator uses its own engine-side CUDA timing, unaffected.

### `vllm/v1/request.py`
Stage P2.2. Add `self.is_finetuning = False` to `Request.__init__` (single source of
truth for FT-tagged requests; set by the injector after construction).

### `vllm/v1/core/sched/output.py`
Stage P2.2. Add `SchedulerOutput.finetune_req_ids: set[str]` (default empty); import
`field`. Populated by `FinetuneScheduler`, read by the model runner to build the mask.

### `vllm/v1/worker/gpu_worker.py`
- Stage 1: `dprint` the `enable_finetuning` flag at the end of `init_device()`
  (proves the flag crossed the spawn boundary into the GPU process).
- Stage 2: if `enable_finetuning`, construct + `start()` a `BackwardProcess`
  (`self.backward_process`).
- Stage 3: `_maybe_share_finetuning_weights()` at the end of `load_model()` — shares
  base `named_parameters` (frozen) + fp32 FT-adapter weights via CUDA IPC; weight hash
  compare gated on `print_weight_hash`.
- Stage P2.3 (M2): `_maybe_setup_finetuning_accumulator()` at the end of `load_model()`
  — allocates `FinetuneAccumulator`, registers per-layer hooks, shares the buffers with
  the backward process, and injects the accumulator + backward handles into the runner.
- Stage P2.4/P2.5: also creates the `FinetuneCoordinator` (worker runs before the
  scheduler), sets its `backward_process` + `backward_sleep_s`, and injects it.
- Stage P3.1: passes `service_name = hf_config.architectures[0]` to `BackwardProcess`
  (selects the per-model `bwd_services` child); `_maybe_share_finetuning_weights`
  resolves the LM-head weight key (separate `lm_head.weight`, else the tied
  `*.embed_tokens.weight`) + org `vocab_size` + `logit_scale` into a `meta` dict sent
  with `share_weights`.
- Stage P3.3: `meta` += model dims (hidden/layers/heads/kv-heads/head_dim/intermediate/
  rope_theta), `lora_scaling` (α/r from `adapter_config.json`), and `learning_rate`/
  `weight_decay`/`gamma`; passes `intermediate_size` to the accumulator.
- Stage P3.4: `_maybe_share_ft_served_lora()` (llama3 only) at the end of `load_model()` —
  pre-`add_lora` + `pin_lora` the FT adapter into a stable served slot, gather vLLM's served
  LoRA stacked buffers (`qkv_proj` slices q/k/v + `o_proj`) for that slot, and IPC-share them
  with the backward via `share_lora_buffers` so the trainer publishes updated weights straight
  into the tensors inference reads. `meta` += `backward_fp32`.
- Stage P4e: creates the coordinator with **`per_step_budget = cap`** (full
  `max_saved_finetuning_tokens`; this is the *binding* site since the worker creates the
  singleton before the scheduler). Was `cap // 2`.

### `vllm/v1/worker/gpu_model_runner.py`
Stage P2.2. Add `_build_finetune_mask` (per-token bool mask in `InputBatch.req_ids`
order from `finetune_req_ids`, sets `self._ft_has`/`self._ft_mask_gpu`/`self._ft_num`;
P3.1 also builds `self._ft_sample_lens`, the per-FT-sample token counts in buffer-write
order).
In `execute_model`: pass `force_eager=self._ft_has` to the cudagraph dispatch (+assert
`CUDAGraphMode.NONE` on FT steps) and `skip_compiled=...or self._ft_has` to
`set_forward_context` (so M2 hooks fire).
Stage P2.3 (M2): in `execute_model`, `begin_step` (pre-forward, at the coordinator's
`current_offset()`) + `accumulate_final` / `end_step` (post-forward, before the
last-token gather) drive `FinetuneAccumulator`, then `record_capture(n, _ft_sample_lens)`
(P3.1 adds the sample lengths) advances the coordinator; a one-shot
`_maybe_verify_ft_accumulation` (gated on `print_activation_hash`) hashes the buffers vs
the backward process's mapped view.
Stage P2.5: `_log_finetuning_batch(scheduler_output, cudagraph_mode)` — gated on
`print_step_mode`, prints `[batch … t=+Xs] prefill=.. ft=.. decode=[kv sizes] | eager/graph(...)`
for non-decode-only batches (P4d: `t=+Xs` = elapsed since `start_finetuning`, from
`coord.ft_start_time`); the graph flag comes from the **real** dispatch decision.
Stage P4e: the FT pause/resume around a prefill forward is **fire-and-forget** — it only
engages when a backward is in flight (`coord.pending_backward`) and there is **no** blocking
`event.synchronize()` between `gpu_pause_backward()` and `gpu_resume_backward()`. After the
activation copies it records `coord.capture_done_evt` (a CUDA event) so the backward trigger
can scope its cross-process visibility wait to just the capture, not the whole device.

> Note: `vllm/v1/core/sched/scheduler.py` is **not** modified — we subclass it.

### `vllm/v1/engine/core.py`
Stage P4d: in `EngineCore._handle_client_request` (ADD branch), gated on `print_step_mode`,
bump `coord.inf_req_count` and `dprint` `[engine-recv HH:MM:SS.mmm] #N ADD req=<id>` (per-request
arrival at the engine). No other engine-loop changes.

---

## Project tooling (consumes the above; not a vLLM change)

| Path | Role |
|---|---|
| `configs/serving_config_finetuning_{opt,llama3}.yaml` | sectioned config consumed by the loader |
| `scripts/launch_deltaserve.py` | offline launcher: YAML → `LLM`, serves via inference adapter |
| `scripts/ft_experiment_{opt,llama3}.py` | co-serving harness: launches a real `vllm serve` HTTP server with finetuning, fires a completion every 1s ×N, then shuts it down (server stdout streams the decision logs) |
| `scripts/train_opt125m_lora.py` | trains the opt-125m toy LoRA adapters |
| `scripts/bench_activation_save.py` | microbenchmark: activation-save overhead per FT step |
| `eval/auto_benchmark.py` | launches `vllm serve` (±`--co`), replays a request timeline, streams `/v1/completions` (ttft = first chunk), writes `timeline_results<suffix>.csv`. P4d/e: `--api-server-count N` (or YAML `server.api_server_count`) shards the frontend; reads `server.api_server_count` from the config |
| `eval/auto_plot.py` | 4-panel per-mode figure. P4d: TTFT panel annotates avg/p90 TTFT + avg TBT (flagged vs `slo.{ttft_slo,avg_tbt_slo}`); E2E-latency panel overlays the inf-only (no-co) curve when its results exist |
| `adapters/{opt125m,llama3}-toy-lora{,-ft}/` | the inference + FT adapters |
| `alpaca_1000.txt` | FT corpus |
| `tests/test_config_loader.py`, `test_finetuning_store.py`, `test_phase1_step2.py`, `test_phase1_step3.py`, `test_phase1_m1.py`, `test_phase1_m2.py`, `test_merged_estimator.py`, `test_profiling_shapes.py` | CPU/GPU verification |
