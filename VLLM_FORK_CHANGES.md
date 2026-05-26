# DeltaServe fork — changes from plain vLLM

Every change we make to the vendored vLLM source tree (the `vllm/` Python
package, which lives under `dserve-vllm/vllm/` from the repo root), organized
by stage. This is the "what did we touch and why" manifest;
`INTEGRATION_PROGRESS.md` is the "how far along / how verified" tracker.
**Keep this in sync** as new changes land.

Scope: only the inner `vllm/` Python package (the published `dserve-vllm`
distribution wraps it). Project-root tooling that *consumes* these (configs,
tests, launcher, adapters) is listed at the bottom for cross-reference but is
not itself a vLLM change. Paths below are written relative to the package
root (`dserve-vllm/vllm/<path>` from the repo root).

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
| P4d — admission close tightening (async) | — | `deltaserve/coordinator.py`* (`note_injection` takes `admitted_now`, uses post-admit free space), `deltaserve/ft_scheduler.py`* (computes `admitted_now` from `next_ft_requests` output and passes through) |
| P6 — `forward_interruptible` (A + B + C tiers) inference pre-emption | — | `config/finetune.py`* (`forward_interruptible` master switch + `ft_only_admission_grace_ms`), `deltaserve/coordinator.py`* (`FTAborted` sentinel, `ft_abort_event`/`ft_only_in_flight`, `release_reserve(n, samples=)`, `snapshot_admission`/`restore_admission` — restores ONLY admit/flush flags, NOT reserved_fill; `buffer_samples` + `on_backward_done` hook so the store commits claimed samples only after the backward acks), `deltaserve/finetuning_store.py`* (`claim` / `commit_claimed` / `release_claimed` 3-phase API replaces one-way `confirmed_trained`; `advance_epoch` refuses while `_claimed` non-empty; `has_claimed`), `deltaserve/ft_injector.py`* (calls `claim` at admit, stashes `req._ft_sample`), `deltaserve/ft_scheduler.py`* (registers `coord.on_backward_done = store.commit_claimed`; admission snapshot at top of `schedule`; passes samples to `coord.reserve`; releases unscheduled-FT samples; `_rollback_ft_step(scheduler_output)` helper used by both tiers B and C; clears `ft_abort_event` at end of rollback; `would_step_be_ft_only()` predicate for tier A), `deltaserve/accumulate.py`* (per-hook `is_set()` check raises `FTAborted` after copy work; `zero_offset_range(off, n)` zeros all hook-target buffers on the aborted offset; `_abort_event` wired from `gpu_worker._maybe_setup_finetuning_accumulator`), `v1/worker/gpu_model_runner.py` (`execute_model` arms `_ft_only_run` and wraps `_model_forward` in try/`except FTAborted`; entry-time bail when event already set — handles pipeline-depth-2 contamination; sentinel `ModelRunnerOutput(_ft_aborted=True)` on bail; `accumulator.zero_offset_range` cleanup; `finally` clears `ft_only_in_flight` + `accumulator.end_step()`), `v1/worker/gpu_worker.py`* (wires `accumulator._abort_event = coord.ft_abort_event` when feature on), `v1/engine/core.py`* (`_maybe_ft_only_grace_poll` before `step_fn` — tier A grace window on `input_queue`; `_maybe_rollback_ft_for_late_arrival` after `schedule()` — tier B; sentinel routing in `step_with_batch_queue` skips `sample_tokens` for aborted batches; abort handler after `future.result()` calls `_rollback_ft_step` and skips `update_from_output`; input thread sets `coord.ft_abort_event` on ADD when `ft_only_in_flight` via cached `_ft_coord_handle`) |
| P6.1 — slice-based FT activation save | — | `deltaserve/accumulate.py`* (`_cur_start` / `_cur_contiguous` fields; `begin_step` + `accumulate_final` accept `start` + `contiguous` kwargs; pre/out hooks slice `val[start:start+n]` on the fast path, fall back to `val[mask]` when not contiguous), `v1/worker/gpu_model_runner.py` (`_build_finetune_mask` also computes `_ft_start` + `_ft_contiguous` from first/last True positions; `execute_model` passes them to `accumulator.begin_step` + `accumulator.accumulate_final`) |
| P5.2 — CUDA-graph backward (per-layer FFN + padded-attention) | `deltaserve/bwd_services/llama3_graph.py`, `tests/test_llama3_backward_graph.py` | `config/finetune.py`* (`backward_cuda_graph` master switch + `backward_cuda_graph_attn_{bn_max,l_max}` padded-attn bounds), `deltaserve/bwd_services/llama3.py`* (extract `ffn_backward_core` / `attn_backward_core` from `layer_backward`; instantiate `Llama3GraphedBackward` in `_build_state` when flag set; new `_layer_backward_graphed` composes graph-A + eager O-bwd + `_maybe_pause` + graph-B + eager tail; `process_backward` per-layer loop dispatches to graphed runner when present), `v1/worker/gpu_worker.py` (meta dict forwards `backward_cuda_graph` + bn_max/l_max + `max_saved_finetuning_tokens` to the child) |
| P5.3 — perf polish + admission strategies + bug fixes | `eval/pure_ft_bench.py` | `config/finetune.py`* (new `match_with_prefill_workload` admission strategy), `deltaserve/finetuning_store.py`* (drop samples with `input_len > max_saved_finetuning_tokens` at load — fixes FT-admission deadlock when only oversized samples remain in the pool), `deltaserve/ft_scheduler.py`* (`_unspent_prefill` leaky-bucket counter; admission shaper branches between factor-cap and match-with-prefill-workload; `set_corpus_meta` IPC after `store.load()`), `deltaserve/backward_process.py`* (new `set_corpus_meta(total_tokens_per_epoch)` one-shot IPC method), `deltaserve/bwd_services/base.py`* (per-cycle log line restructured to one line: `[backward] Xms (graph/eager) loss=… total_trained=… n=… epoch=… N/total tokens`; persistent `_grad_qh/kh/vh_buf` allocation; CUDA-event timing replaces wall-clock-after-sync; sync coalesced from 2/cycle → 1/cycle; new `set_corpus_meta` cmd handler), `deltaserve/bwd_services/llama3.py`* (fused AdamW via `torch.optim.AdamW(..., fused=True)`; persistent `grad_qh/kh/vh_buf` plumbed into `attn_backward_core` via opt-in kwargs; `_publish_to_served` docstring expanded with scaling-contract / FT-slot-exclusivity notes), `deltaserve/bwd_services/llama3_graph.py`* (single shared padded-attn graph reused across all L layers — `_attn_graphs` dict collapsed to one `_attn_graph` since the core has no per-layer weights), `eval/auto_benchmark.py` (factor tag `_factor_<X>` appended to output suffix on `--co` runs; `_load_yaml_cfg()` helper shared with `build_server_cmd`), `eval/auto_plot.py` (5-panel layout with new E2E latency percentile panel + p99 highlighted; `--factor` CLI arg with auto-detection of smallest factor; factor in plot title) |
| Eng obs — per-batch lifecycle trace + ms bwd-log timestamps | — | `config/finetune.py`* (`print_scheduler_add`, `print_engine_batch_exec`, `print_engine_batch_done`, `print_engine_req_recv` — independent gates for the per-batch lifecycle prints; `print_step_mode` becomes a convenience master switch for all four), `deltaserve/coordinator.py`* (`_write_bwd_log_row` uses `isoformat(timespec="milliseconds")`), `v1/engine/core.py`* (`_classify_batch_for_log` decode-only / zero-token gate; `_maybe_log_batch_scheduled` after `schedule()` + `_maybe_log_batch_done` after `future.result()`; engine-recv print gate switched to OR of `print_engine_req_recv | print_step_mode`), `v1/worker/gpu_model_runner.py` (`_log_finetuning_batch` gate switched to OR of `print_engine_batch_exec | print_step_mode`) |

`*` = same file extended in a later stage.

> **P5.1:** the backward yields the GPU at every layer boundary while the main process runs an
> inference **prefill** forward (TTFT-critical); decode-only steps let the backward co-run. The
> grant is an `mp.Event` (SET = may run); the runner clears it around prefill forwards.

> **P6 (forward_interruptible):** three-tier inference pre-emption of FT-only stepping, all
> behind one config flag (`finetune.forward_interruptible`, default OFF — zero cost when off via
> short-circuit attribute loads at each hook site).
> **A** — pre-schedule grace: when the next step would be FT-only, briefly block on
> `input_queue` so late HTTP arrivals make it into this step.
> **B** — post-schedule rollback: if `schedule()` produced an FT-only batch and `input_queue`
> is non-empty, undo the FT scheduling (free KV, release reserve + claimed samples, restore
> admission flags) and re-schedule once.
> **C** — mid-forward abort: the input-socket thread sets `coord.ft_abort_event` on each ADD
> while `coord.ft_only_in_flight`; the activation-accumulation hooks check the event after
> their copy work and raise `FTAborted`; the runner catches it, zeros the partial-write tail
> at the aborted offset, returns an empty `ModelRunnerOutput(_ft_aborted=True)`; the engine
> sees the sentinel after `future.result()` and runs `_rollback_ft_step`.
> The 3-phase store API (`claim` / `commit_claimed` / `release_claimed`) is load-bearing — it
> also fixes the pre-existing bookkeeping flaw where samples were marked `trained=True` at
> admit time (before the backward had actually processed them). `advance_epoch` now refuses
> while any sample is claimed in-flight. `snapshot_admission` deliberately does NOT capture
> `reserved_fill` — that's undone via `release_reserve(n)`, and restoring a snapshotted
> `reserved_fill` would clobber an intervening pipelined commit (`record_capture` between
> snapshot and rollback). See `.claude/plans/can-you-make-a-elegant-cherny.md` for the
> end-to-end design + verification path.

> **P6.1 (slice activation save):** when the FT-True positions in the mask form a contiguous
> span (the common case — FT requests admitted at the tail of waiting and not interleaved by
> InputBatch slot reuse), per-layer hooks gather with a slice `val[start:start+n]` (view, no
> kernel, no allocation) instead of `val[mask]` (index_select). Contiguity is **not**
> guaranteed (FT can land in a freed inference slot mid-batch via
> `_register_add_request` + `condense()`), so the mask path is kept as a silent fallback.

> **P5.2 (backward CUDA graph):** mirrors DeltaServe `models/llama/SFT_service_graph.py`.
> Two graphs per layer behind one flag (`finetune.backward_cuda_graph`, default OFF):
> **Graph A** — FFN-backward at fixed `[s_max, D]` (s_max=`max_saved_finetuning_tokens`,
> the same width the activation buffers are pre-allocated at, so the graph is shape-stable
> by construction). **Graph B** — padded-attention backward CORE at `[bn_max, l_max]`,
> scatter (flat → padded) → captured scores/softmax/dQ/dK/dV → gather back to flat.
> **All `L` per-layer graphs are pre-captured up-front in the runner constructor's
> `prepare()`**, paid once at child startup (before the first `share_activations` ack)
> against zero-initialized static buffers — capture only depends on shapes/addresses,
> not values, so replay-time staging produces the correct gradients. The first real
> backward sees only replay cost; no warmup + capture stalls land on a live co-serving
> step. Silent eager fallback per-layer on capture or shape-fit failure (those layers
> get added to `ffn_failed` / `attn_failed`). `_maybe_pause()` is still called once per
> layer — relocated from the layer top to **between Graph A and Graph B** since the
> host-side `mp.Event.wait` can't run inside a captured region. Static IO buffers (g,
> resid_mid, gate, up, qh/kh/vh/grad_ctx pads, grad outputs, bn_idx/pos_idx, masks) are
> allocated OUTSIDE the shared graph pool — the load-bearing rule for avoiding pool-
> aliasing NaN traps (DeltaServe reference lines 111–113 vs the `graph_pool_handle()` at
> 114). LoRA-grad ownership stays with the eager Q/K/V/O proj backwards, so
> `nn.Parameter.grad` lifecycle is unchanged vs eager. **Layer forward-remat
> (`layer_forward`) is NOT graphed** — it runs eagerly between layers, matching the
> reference. Graphing it would require a third per-layer graph with padded-attention
> forward scaffolding; treated as a follow-up if dispatch overhead in the eager remat
> becomes the bottleneck. The graphed and eager paths share `ffn_backward_core` /
> `attn_backward_core`, so gradient values are bit-identical (verified by
> `tests/test_llama3_backward_graph.py`).

> **P5.3 (perf polish + admission strategies + bug fixes):**
> • **`match_with_prefill_workload`** — alternate FT admission strategy
>   (`config/finetune.py`, default False). Maintains a leaky-bucket counter
>   `_unspent_prefill` of inference prefill tokens seen but not yet "spent" on
>   FT. On each prefill-carrying step (t_in > 0): peek the smallest untrained
>   FT sample; if `_unspent_prefill + t_in >= sample.input_len`, admit that ONE
>   sample sized exactly to it (the per-step pack-multiple-samples path is
>   bypassed) and reset the counter; otherwise accumulate `+= t_in` and skip
>   FT this step. Any successful FT admission (any path) resets the counter
>   unconditionally so credit is consumed atomically. Mutually exclusive with
>   `ft_tokens_admission_constrain_factor` — the flag wins when both are set.
>   FT-only/idle steps and decode-only steps follow the existing flow.
> • **Oversized-sample deadlock fix** (`finetuning_store.py:load()`): samples
>   with `input_len > max_saved_finetuning_tokens` are now dropped at load
>   time with a one-line warning. Previously they sat in the selectable pool
>   forever — `has_next()` stayed True so `advance_epoch` never fired, but
>   `pop_best_under(cap)` returned None so the injector returned empty;
>   `note_injection(next_sample_len > effective_space, 0)` closed admission;
>   the empty buffer meant no flush trigger fired; `has_requests()` returned
>   False; engine idled forever. Surfaced by `pure_ft_bench.py` on
>   `alpaca_1000.txt` (3 samples > 256-token cap, 813 tokens dropped, FT
>   cycled cleanly through `num_epochs` afterward).
> • **One-shot `set_corpus_meta` IPC** (`backward_process.py`,
>   `bwd_services/base.py`, `ft_scheduler.py`): the FT corpus token count is
>   constant for the run, so it's sent ONCE from the engine core to the
>   backward child right after `FinetuningStore.load()` completes — rather
>   than bundled into every `notify_buffer_full`. Child stores it on
>   `self._total_tokens_per_epoch` and uses it for the per-epoch progress
>   meter in the per-cycle log.
> • **Per-cycle log line** (`bwd_services/base.py`): now one structured line
>   per backward — `[backward] Xms (graph/eager) loss=… total_trained=…
>   n=… epoch=… N/total tokens` — derived from CUDA events around
>   `process_backward()` so timing is GPU-strict, not wall-clock-after-sync.
>   The `N/total tokens` part is the in-epoch progress (resets at epoch
>   advance). CUDA sync coalesced from 2/cycle (timing + cleanup) to 1
>   (cleanup only; timing reads the events after the cleanup sync, which
>   has already drained the GPU).
> • **Fused AdamW** (`bwd_services/llama3.py`): `torch.optim.AdamW(...,
>   fused=True)`. 256 LoRA tensors (8 per layer × 32 layers) go through a
>   single CUDA fused kernel instead of per-tensor dispatch — ~3-5 ms saved
>   per backward. Numerically identical.
> • **Persistent `grad_qh/kh/vh_buf`** (`bwd_services/llama3.py`): allocated
>   once on the service in `_build_state` at `s_max`, passed through to
>   `attn_backward_core` via opt-in kwargs (the gradcheck test path passes
>   None and gets fresh allocs — unchanged behavior). Eliminates 96
>   zero-fills/backward (32 layers × 3 tensors).
> • **Single shared padded-attention graph** (`bwd_services/llama3_graph.py`):
>   the padded-attn core reads only static IO + masks + indices, NO layer-
>   specific weights — so 32 identical per-layer captures are wasteful. The
>   `_attn_graphs: dict[int, CUDAGraph]` was collapsed to one
>   `_attn_graph: CUDAGraph | None` reused across all L layers, with
>   `attn_failed` becoming a single bool. ~1.5-3 s saved at startup capture +
>   smaller graph-pool footprint.
> • **eval/auto_benchmark `_factor_<X>` suffix** — output files on `--co` runs
>   now carry the FT-admission factor (`_co_factor_1_loose.csv` etc.) so A/B
>   runs across factors don't overwrite each other. `-1` (disabled) renders
>   as `off`. Implemented via `_load_yaml_cfg()` helper shared with
>   `build_server_cmd` to avoid double YAML loads.
> • **eval/auto_plot 5-panel + factor + p99** — second-row E2E latency
>   percentile panel (empirical CDF) with the p99 marker highlighted across
>   series (heavier vertical guide, color-matched horizontal ticks with the
>   p99 value annotated). `--factor X` CLI arg picks which `_factor_<X>`
>   set to plot; if absent, auto-detects from `output/` and picks the
>   smallest (`off` = -1 sorts smallest). Factor appears in the plot title.
> • **`eval/pure_ft_bench.py`** — new pure-FT benchmark script (no inference
>   traffic). Launches the server with FT enabled, POSTs `/start_finetuning`,
>   idles for `--duration` seconds, trims the bwd_log to the post-POST
>   window, and summarizes (cycles, span, total trained, avg tok/s,
>   per-cycle batch_tokens + dt stats). Useful for validating Phase 5
>   graph/perf changes in isolation.

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
section) + the two **mutually exclusive FT admission shapers**: P4e's
**`ft_tokens_admission_constrain_factor`** (cap FT tokens ≤ `prefill_tokens · factor`
per prefill step; `-1` disables) and P5.3's **`match_with_prefill_workload`**
(leaky-bucket — accumulate observed prefill tokens, admit ONE FT sample when
accumulated ≥ next sample's input_len, reset on any FT admit; default False; wins
over the factor when both set). P5.2's **backward CUDA-graph knobs** —
**`backward_cuda_graph`** master switch (default False) + the padded-attention
bounds **`backward_cuda_graph_attn_bn_max`** (default 8) + **`backward_cuda_graph_attn_l_max`**
(default 64). The **`backward_fp32`** bulk-compute precision flag (P3.4, default
False = bf16). The P6 functional knobs `forward_interruptible` (master switch for
inference pre-emption of FT-only stepping — tiers A + B + C) +
`ft_only_admission_grace_ms` (tier-A grace window in ms, default 2.0, 0 disables A
while keeping B and C). The debug knobs `print_weight_hash` / `print_activation_hash`
/ `print_step_mode` plus the per-batch lifecycle gates `print_scheduler_add` /
`print_engine_batch_exec` / `print_engine_batch_done` / `print_engine_req_recv`
(each independently togglable; `print_step_mode` is a convenience master switch
that enables all four). `slo` + `debug` keys are folded into FinetuneConfig by
the loader.
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
| `eval/auto_benchmark.py` | launches `vllm serve` (±`--co`), replays a request timeline, streams `/v1/completions` (ttft = first chunk), writes `timeline_results<suffix>.csv`. P4d/e: `--api-server-count N` (or YAML `server.api_server_count`) shards the frontend; reads `server.api_server_count` from the config. Also writes `bench_meta<suffix>.json` with `t_first_wall_iso` so the plotter can anchor the FT (wall-clock) series at the same t=0 as the inference (monotonic) series — fixes the throughput-panel misalignment where inference and FT peaks appeared in-phase due to the FT log anchoring at first-backward-row instead of benchmark start |
| `eval/auto_plot.py` | 4-panel per-mode figure. P4d: TTFT panel annotates avg/p90 TTFT + avg TBT (flagged vs `slo.{ttft_slo,avg_tbt_slo}`); E2E-latency panel overlays the inf-only (no-co) curve when its results exist. Loads `bench_meta<suffix>.json` (if present) and passes `t0_wall` to `parse_bwd_log_csv` so the FT series is anchored at benchmark t=0; falls back to legacy first-row anchoring when the meta is missing. `--throughput-window` flag tunes the rolling-mean width on the throughput panel (auto-pick in `[5, 60]` seconds when unset) |
| `adapters/{opt125m,llama3}-toy-lora{,-ft}/` | the inference + FT adapters |
| `alpaca_1000.txt` | FT corpus |
| `tests/test_config_loader.py`, `test_finetuning_store.py`, `test_phase1_step2.py`, `test_phase1_step3.py`, `test_phase1_m1.py`, `test_phase1_m2.py`, `test_merged_estimator.py`, `test_profiling_shapes.py` | CPU/GPU verification |
