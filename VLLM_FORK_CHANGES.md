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
| P5.3 — perf polish + admission strategies + bug fixes | `eval/pure_ft_bench.py` | `config/finetune.py`* (new `match_prefill_workload_factor: float` admission strategy), `deltaserve/finetuning_store.py`* (drop samples with `input_len > max_saved_finetuning_tokens` at load — fixes FT-admission deadlock when only oversized samples remain in the pool), `deltaserve/ft_scheduler.py`* (`_unspent_prefill` leaky-bucket counter; admission shaper branches between factor-cap and match-with-prefill-workload; `set_corpus_meta` IPC after `store.load()`), `deltaserve/backward_process.py`* (new `set_corpus_meta(total_tokens_per_epoch)` one-shot IPC method), `deltaserve/bwd_services/base.py`* (per-cycle log line restructured to one line: `[backward] Xms (graph/eager) loss=… total_trained=… n=… epoch=… N/total tokens`; persistent `_grad_qh/kh/vh_buf` allocation; CUDA-event timing replaces wall-clock-after-sync; sync coalesced from 2/cycle → 1/cycle; new `set_corpus_meta` cmd handler), `deltaserve/bwd_services/llama3.py`* (fused AdamW via `torch.optim.AdamW(..., fused=True)`; persistent `grad_qh/kh/vh_buf` plumbed into `attn_backward_core` via opt-in kwargs; `_publish_to_served` docstring expanded with scaling-contract / FT-slot-exclusivity notes), `deltaserve/bwd_services/llama3_graph.py`* (single shared padded-attn graph reused across all L layers — `_attn_graphs` dict collapsed to one `_attn_graph` since the core has no per-layer weights), `eval/auto_benchmark.py` (factor tag `_factor_<X>` appended to output suffix on `--co` runs; `_load_yaml_cfg()` helper shared with `build_server_cmd`), `eval/auto_plot.py` (5-panel layout with new E2E latency percentile panel + p99 highlighted; `--factor` CLI arg with auto-detection of smallest factor; factor in plot title) |
| Eng obs — per-batch lifecycle trace + ms bwd-log timestamps | — | `config/finetune.py`* (`print_scheduler_add`, `print_engine_batch_exec`, `print_engine_batch_done`, `print_engine_req_recv` — independent gates for the per-batch lifecycle prints; `print_step_mode` becomes a convenience master switch for all four), `deltaserve/coordinator.py`* (`_write_bwd_log_row` uses `isoformat(timespec="milliseconds")`), `v1/engine/core.py`* (`_classify_batch_for_log` decode-only / zero-token gate; `_maybe_log_batch_scheduled` after `schedule()` + `_maybe_log_batch_done` after `future.result()`; engine-recv print gate switched to OR of `print_engine_req_recv | print_step_mode`), `v1/worker/gpu_model_runner.py` (`_log_finetuning_batch` gate switched to OR of `print_engine_batch_exec | print_step_mode`) |
| P5.4 — forward-recompute CUDA graph (per-layer; 3rd captured region) | — | `deltaserve/bwd_services/llama3_graph.py`* (new `_forward_core` + `_padded_attn_forward_core` + `_capture_forward` + `stage_forward_inputs` + `forward` + `cache_views` methods on `Llama3GraphedBackward`; new static IO `static_layer_in/cos/sin/saved_gate_up/x_norm1/qh_flat/kh_flat/vh_flat/ctx_flat`; `prepare()` pre-captures L=32 forward graphs; `begin_backward()` stages `cos/sin` once per backward; `fwd_failed: set[int]` mirrors `ffn_failed`), `deltaserve/bwd_services/llama3.py`* (`process_backward` per-layer loop routes through `graph_runner.forward(...)` when `_attn_fit` AND `saved_gu` is present; eager `layer_forward(...)` fallback otherwise; layer_forward arg list unchanged in this stage), `config/finetune.py`* (docstring on `backward_cuda_graph` updated to "forward + FFN-bwd + attn-bwd" coverage), `tests/test_llama3_backward_graph.py` (new `test_forward_graph_parity` + `test_forward_overflow_fallback`: 45 new assertions, parity bit-identical to eager `layer_forward`) |
| P5.5 (F1) — save post-RoPE qh/kh/vh per layer | — | `config/finetune.py`* (new `save_attn_qkv: bool = False` opt-in field), `deltaserve/accumulate.py`* (auto-detect `self_attn.attn` modules per layer; allocate `attn_qh [s_max, q_size]` + `attn_kh/vh [s_max, kv_size]` per layer; new `_make_attn_qkv_pre_hook` reads `args=(q, k, v)` post-RoPE; `zero_offset_range` extended; new `q_size`/`kv_size`/`save_attn_qkv` constructor args), `deltaserve/bwd_services/llama3.py`* (`_build_state` reads `meta["save_attn_qkv"]` → `self.save_attn_qkv`; `layer_forward` accepts `saved_qh/kh/vh` and short-circuits Q/K/V proj + RoPE; `process_backward` extracts saved q/k/v per layer and threads through both paths), `deltaserve/bwd_services/llama3_graph.py`* (`save_attn_qkv` mode flag; new `static_saved_qh/kh/vh` IO; `_forward_core` branches on the mode flag to read from the saved buffers instead of computing Q/K/V/RoPE; `stage_forward_inputs` + `forward` signatures extended with `saved_qh/kh/vh` kwargs), `v1/worker/gpu_worker.py` (derive `q_size = num_heads * head_dim`, `kv_size = num_kv_heads * head_dim`; pass through to `FinetuneAccumulator`; add `save_attn_qkv` to the shared `meta` dict so the backward subprocess knows the mode), `tests/test_llama3_backward_graph.py`* (new `test_layer_forward_saved_qkv_parity` + `test_forward_graph_saved_qkv_parity`: 45 new assertions, parity vs the recompute path) |
| P5.6 — save the post-attention residual per layer (`save_resid_mid`) | `tests/test_accumulate_hooks.py` | `config/finetune.py`* (new `save_resid_mid: bool = False`), `deltaserve/ft_meta.py`* (meta key), `deltaserve/accumulate.py`* (`_POST_LN_SUFFIX`; `post_attention_layernorm` discovery; per-layer `resid_mid` buffers registered with the existing fused add-norm `_make_pre_hook`; `buffers["resid_mid"]`; `zero_offset_range`), `v1/worker/gpu_worker.py`* (passes the flag), `deltaserve/bwd_services/common/trainer.py`* (mirror flag; `saved_resid_mid=` threaded per layer), `deltaserve/bwd_services/{llama3,qwen3}.py`* (`layer_forward(saved_resid_mid=)` skips O-proj + reduce + residual add; `graph_forward_core` skips step 5 in the mode), `deltaserve/bwd_services/common/graph.py`* (`save_resid_mid` mode; staged straight into `static_resid_mid`; `_forward_needs_reduce`; tail captured with the core when no reduce is needed; eager fallback on a missing saved residual), `tests/test_{llama3,qwen3}_backward_graph.py`* (parity tests), `tests/test_tp_{backward,trainer}_graph_nccl.py`* (save mode + per-layer collective counts), `configs/*.yaml`* (`save_resid_mid: true`) |
| UnifiedFT — `slo.coserving_admission_phase: both` scheduler | `deltaserve/ft_scheduler_both.py`, `configs/serving_config_finetuning_llama3_both.yaml`, `eval/auto_plot_schedulers.py` | `config/finetune.py`* (new `coserving_admission_phase: str = "prefill"` + `decode_only_ft_safety_margin: float = 0.7` fields under the `slo:` YAML section), `config/vllm.py`* (branch `scheduler_cls` on the phase in `__post_init__`; soft-fall to `"prefill"` with `logger.warning` when `phase=="both"` AND `ft_tokens_admission_constrain_factor != -1`), `deltaserve/ft_scheduler.py`* (extract the FT admission gate to a new `_initial_ft_budget(feats, earliest_arrival)` hook so subclasses can override the `decode_only → 0` short-circuit cleanly), `eval/auto_benchmark.py`* (new `--scheduler {prefill,both}` CLI arg → maps to the corresponding YAML via `_SCHED_CONFIGS`; new `_phase_tag(cfg)` helper reads `slo.coserving_admission_phase` from the loaded YAML; output suffix scheme extended to `_co_factor_<X>_phase_<Y>_<mode>`; `_load_yaml_cfg(config_path)` + `build_server_cmd(..., config_path)` take an explicit path arg), `eval/auto_plot.py`* (drop the E2E latency percentile panel; figure shrinks to a single-row 4-panel layout) |

| P7 — TP=2 co-serving finetuning (M1-M4.1) | `configs/serving_config_finetuning_llama3_tp2.yaml`, `tests/test_llama3_tp_shard.py`, `tests/test_llama3_tp_backward_gloo.py`, `eval-tp/{launch_deltaserve,ft_bench_tp}.py` | `v1/executor/multiproc_executor.py` (**new file for the fork** — non-daemon workers when finetuning + `PR_SET_PDEATHSIG` + lethal death-pipe monitor), `v1/outputs.py` (**new file for the fork** — worker→scheduler relay fields), `v1/core/sched/output.py` (`finetune_backward_trigger` broadcast), `v1/worker/gpu_worker.py` (tp geometry in `meta`, lm_head all-gather, local accumulator widths, `relay_mode`), `v1/worker/gpu_model_runner.py` (relay stash / `execute_trigger` / ack poll), `deltaserve/bwd_services/llama3.py`* (local dims, `lora_shard_slice`, backward NCCL group, the per-layer all-reduces, graph force-off under TP), `deltaserve/bwd_services/base.py`* (child `PR_SET_PDEATHSIG`), `deltaserve/coordinator.py`* (relay mode), `deltaserve/ft_scheduler.py`* (relay wiring), `deltaserve/backward_process.py`* (daemon-guard message) |
| P8 — Qwen3 family + backward-service restructure | `deltaserve/bwd_services/common/{__init__,ops,attention,ffn,head,tp,family,trainer,graph}.py` (`graph.py` = `git mv` of `llama3_graph.py`), `deltaserve/bwd_services/{registry,qwen3}.py` (qwen3 incl. its `graph_forward_core`; `common/graph.py` gained `static_q_pre`/`static_k_pre`), `deltaserve/ft_meta.py`, `tests/test_qwen3_backward_graph.py`, `configs/serving_config_finetuning_qwen3_{14b_tp2,0.6b}.yaml`, `tests/bwd_harness.py`, `tests/test_qwen3_{backward,tp_shard,tp_backward_gloo,train_overfit}.py`, `scripts/{toy_adapters,init_adapters_qwen3}.py`, `adapters/qwen3-{14b,0.6b}-toy-lora{,-ft}` | `deltaserve/bwd_services/llama3.py`* (now only the Llama-3 layer math + `LLAMA3` + a 5-line service), `deltaserve/bwd_services/base.py`* (`get_service` → registry shim; `is_trainer` class attr), `deltaserve/bwd_services/__init__.py`*, `config/vllm.py`* (finetuning listed as unsupported on the v2 model runner), `v1/worker/gpu_worker.py`* (publish gate via `registry.is_trainer`; `meta` via `ft_meta.build_backward_meta`; `rope_theta` fix; `effective_save_attn_qkv`), `eval-tp/{launch_deltaserve,ft_bench_tp}.py`* (`--family` presets, YAML-derived model/adapter, per-family output names), `tests/test_llama3_*.py`* (import paths; overfit test repaired), `scripts/init_adapters_llama3.py`* (thin wrapper over `toy_adapters.py`) |
| P7 / M4.2 — SLO estimator + all-rank backward ack under TP | `tests/test_tp_timing_relay.py` | `v1/outputs.py`* (`finetune_timing`), `v1/core/sched/output.py`* (`finetune_record_timing`), `v1/worker/gpu_model_runner.py`* (timing ring: per-step record gate + actual `was_graph`; `sample_tokens` relays the timing queue; `_ft_relay_backward_done` MIN-all-reduce over the TP `cpu_group`), `deltaserve/ft_scheduler.py`* (stamp gate; push relayed samples), `deltaserve/coordinator.py`* (`relay_backward_outstanding` / `poll_own_backward_ack` / `take_relay_ack`; bwd-log header also on an empty file), `deltaserve/bwd_services/base.py`* (cycle-time event read restored), `v1/worker/gpu_model_runner.py`* also relays on the idle 0-token early return via `_ft_fill_relay_fields` (FT stalled through idle valleys without it), `configs/serving_config_finetuning_{llama3_tp2,qwen3_14b_tp2}.yaml`* (`profile_on_launch: true`; qwen3 `ttft_slo 0.4`) |
| P7 / M5 — backward CUDA graphs under TP | `tests/test_tp_backward_graph_nccl.py`, `tests/test_tp_trainer_graph_nccl.py` | `deltaserve/bwd_services/common/graph.py`* (new `static_o`; new `forward_tail()` = residual add + gate/up split; `_forward_core` captures family core + tail at tp=1 — capture count and tp=1 values unchanged; under TP `forward()` replays the core, all-reduces `static_o[:n]` eagerly, then runs the tail; the eager fallback inside `forward()` now passes `all_reduce`), `deltaserve/bwd_services/{llama3,qwen3}.py`* (`graph_forward_core` ends at `static_o`; `layer_backward_graphed(..., all_reduce=None)` issues the six backward reduces at the same points as `layer_backward` — they were absent before), `deltaserve/bwd_services/common/trainer.py`* (the `tp_size > 1` force-eager branch removed; the loop passes `self._all_reduce` to the graphed backward), `deltaserve/bwd_services/common/family.py`* (contract docstrings), `config/finetune.py`* (docstring), `configs/serving_config_finetuning_{llama3_tp2,qwen3_14b_tp2}.yaml`* (comments) |
| P7 — corpus-meta relay (`N/?` meter fix) | — | `deltaserve/coordinator.py`* (`corpus_total_tokens`; the relayed trigger command carries `total_tokens_per_epoch`; `execute_trigger` forwards it to the child once via `set_corpus_meta` before `notify_buffer_full`), `deltaserve/ft_scheduler.py`* (stores the total on the coordinator in addition to the tp=1 direct send), `tests/test_tp_timing_relay.py`* (`test_corpus_meta_relay`) |
| P8.1 — exact `save_attn_qkv` for Qwen3 (pre-norm q/k) | — | `deltaserve/bwd_services/common/family.py`* (`saved_qkv_pre_transform`), `deltaserve/bwd_services/qwen3.py`* (`supports_saved_qkv` back to True + `saved_qkv_pre_transform=True`; `layer_forward(saved_qh/kh=pre-norm)` re-applies norm + RoPE and keeps `q_pre`/`k_pre`; `graph_forward_core` skips the Q/K/V GEMM in the mode), `deltaserve/ft_meta.py`* (`saved_qkv_pre_transform(arch)`), `deltaserve/accumulate.py`* (`_Q_NORM_SUFFIX`/`_K_NORM_SUFFIX` discovery; `attn_qkv_pre_transform` ctor flag; `_make_arg_pre_hook`; q/k from the norm inputs, v from the attn pre-hook; self-disables when the norm modules are missing), `v1/worker/gpu_worker.py`* (passes the stage flag), `config/finetune.py`* (docstring), `tests/test_qwen3_backward.py`* (refusal test → exactness test), `tests/test_qwen3_backward_graph.py`* (`test_saved_qkv_pre_norm`), `tests/test_accumulate_hooks.py`* (Qwen-style fake), `tests/test_tp_backward_graph_nccl.py`* (`save_all` mode), `configs/serving_config_finetuning_{qwen3_14b_tp2,qwen3_0.6b,llama3_tp2}.yaml`* (all three saves on) |
| P5.7 — LM-head restructure | — | `deltaserve/bwd_services/common/head.py`* (`head_backward` batches every sample's predicting rows into one fp32 GEMM per vocab chunk and converts each bf16 head chunk once per pass — was per sample; explicit fp32 upcast of the normed rows; `logits_chunked` kept as a helper) |
| P7 / M4.3 — gradient bucketing + comm stream + rank-symmetric clip | `tests/test_tp_bucket_gloo.py` | `deltaserve/bwd_services/common/tp.py`* (`REPLICATED_FACTORS`/`SHARDED_FACTORS`, `BUCKET_LAYERS`, `CommQueue`, `FactorBucket`, `clip_layers_symmetric_`), `deltaserve/bwd_services/common/trainer.py`* (builds the bucket + queue at tp>1; `reduce_factors=False`; stash / submit per group; `wait` → replicated grads to the masters → symmetric clip → step; tp=1 path unchanged), `deltaserve/bwd_services/{llama3,qwen3}.py`* (`reduce_factors=` kwarg on both backward entry points), `deltaserve/backward_process.py`* (no longer sets `CUDA_DEVICE_MAX_CONNECTIONS=1` on the child), `tests/test_tp_trainer_graph_nccl.py`* (rewritten: 5 modes incl. sync/delay bit-identity, clip on, collective counts), `tests/test_phase1_step2.py`* |
| P6 / TP — forward_interruptible tier C rank-symmetric under TP | `tests/test_ft_abort_tp.py` | `deltaserve/coordinator.py`* (`FtArrivalSignal` shm counter, `FtAbortPoller` MAX-reduce over the TP gloo cpu_group, `ft_abort_poller` attr), `v1/engine/core.py`* (creates the signal before the executor + env `DSERVE_FT_ARRIVAL_SHM`; input thread bumps it per ADD; unlink on shutdown; `_ft_output_aborted` helper checks the sentinel field on the model output and the execute future), `v1/outputs.py`* (`finetune_aborted` field), `v1/worker/gpu_model_runner.py`* (poller entry/finish/note_served, `_ft_abort_sentinel` with relay fields, `sample_tokens` pending-abort sentinel, `[ft-abort]` log), `v1/worker/gpu_worker.py`* (wires the poller under TP, the Event at tp=1), `deltaserve/accumulate.py`* (`_abort_poll(boundary)`; only the layer_in pre-hook is a boundary), `deltaserve/bwd_services/base.py`* (quiet KeyboardInterrupt at shutdown), `eval-tp/ft_bench_tp.py`* (ok counter fix), `config/finetune.py`* (`pause_until_prefill_done`), `v1/worker/gpu_model_runner.py`* (also: resume-on-completion via a CUDA event + `_ft_maybe_resume_backward`), `deltaserve/coordinator.py`* (also: `_LaunchAheadGate`, `LocalAbortPoller` for tp=1), `configs/serving_config_finetuning_{qwen3_14b_tp2,llama3_tp2}.yaml`* (`forward_interruptible: true`, 2 ms grace) |

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
> `nn.Parameter.grad` lifecycle is unchanged vs eager. The graphed and eager paths
> share `ffn_backward_core` / `attn_backward_core`, so gradient values are
> bit-identical (verified by `tests/test_llama3_backward_graph.py`). **P5.4 added a
> third per-layer graph for the layer forward recompute** — see the P5.4 design
> note below.

> **P5.3 (perf polish + admission strategies + bug fixes):**
> • **`match_prefill_workload_factor: float`** — alternate FT admission
>   strategy (`config/finetune.py`, default 0.0). Maintains a leaky-bucket
>   counter `_unspent_prefill` of inference prefill tokens seen but not yet
>   "spent" on FT. On each prefill-carrying step (t_in > 0, factor > 0): peek
>   the smallest untrained FT sample; if
>   `(_unspent_prefill + t_in) * factor >= sample.input_len`, admit that ONE
>   sample sized to it (capped by the SLO budget; per-step pack-multiple path
>   bypassed) and reset the counter; otherwise accumulate `+= t_in` and skip
>   FT this step. Factor scales the credit each prefill token earns:
>   `1.0` ≡ the original boolean-on behaviour, `>1` more aggressive,
>   `<1` more conservative, `0` disables (trigger never fires). Any
>   successful FT admission (any path) resets the counter unconditionally
>   so credit is consumed atomically. Mutually exclusive with
>   `ft_tokens_admission_constrain_factor` — the leaky-bucket wins when
>   factor > 0.
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

> **P5.4 (forward-recompute CUDA graph):** extends the P5.2 backend from 2 graphs
> per layer (FFN-bwd, attn-bwd) to **3 graphs per layer** under the same
> `finetune.backward_cuda_graph` flag. The new region is the layer-forward
> rematerialization (RMSNorm in_ln + Q/K/V proj + RoPE + padded-attention forward +
> O proj + residual add). Same shape-stability strategy as Graphs A/B: capture at
> fixed `s_max = max_saved_finetuning_tokens` against zero-initialized static
> buffers, per-layer eager fallback (`fwd_failed`), pre-captured up-front in
> `prepare()` so the first real backward sees only replay cost. **Key insight that
> made it tractable**: the forward graph's OUTPUTS are exactly the inputs Graph A /
> Graph B already read (`static_resid_mid`, `static_gate`, `static_up`,
> `static_qh_pad/kh_pad/vh_pad`). Writing directly into them eliminates the
> intermediate copy-in step. Per-sample attention forward is replaced by a
> captureable padded variant mirroring the existing padded-attn backward
> (`_padded_attn_forward_core`). Pre-capture cost goes from 33 → 65 graphs at child
> startup (~few hundred ms). LoRA `.data` storage references are stable across
> `optimizer.step()` (fused AdamW does in-place updates), so the captured kernels
> see the up-to-date LoRA values at replay time. The scatter from flat `qh/kh/vh`
> to padded uses `index_put_(accumulate=True)` (not `accumulate=False`) so the
> tail-row → (0,0) write inside the captured region adds zero (input tail is
> zero by construction) instead of clobbering the legit sample-0/pos-0 slot.

> **P5.5 (save post-RoPE qh/kh/vh per layer; a.k.a. F1):** opt-in via
> `finetune.save_attn_qkv: bool = False`. When ON, three new per-layer buffers
> (`attn_qh`, `attn_kh`, `attn_vh`) in `FinetuneAccumulator` get populated by a
> `forward_pre_hook` on each `self_attn.attn` module — the hook sees
> `args = (q, k, v)` POST-RoPE (vLLM's `LlamaAttention.forward` calls RoPE then
> `attn(q, k, v)`, so a pre-hook on `attn` is the cleanest hook point). The
> backward (both eager `layer_forward` and the Phase 5.4 forward graph) reads
> from these buffers via the new `saved_qh/kh/vh` params and short-circuits the
> Q/K/V projection + RoPE entirely. RMSNorm in_ln stays a recompute (cheap, ~few
> MFLOPs; the Q/K/V LoRA-A backward needs `x_norm1` for
> `grad_A = grad_Z.t() @ x_norm1`). Memory cost: ~96 MB at s_max=256 on
> Llama-3-8B (qh ~2 MB/layer × 32 = 64 MB; kh/vh ~0.5 MB/layer × 32 each = 32 MB).
> Compute saved: ~13-16 GFLOPs/layer × 32 ≈ 400-500 GFLOPs per backward, ~5 ms on
> a 5090. The Phase 5.4 forward graph's `_forward_core` branches on a mode flag
> set at runner construction (`self.save_attn_qkv`) — the captured graph either
> reads from the saved buffers (skip Q/K/V/RoPE) or computes them (the
> recompute path). Mode is fixed per-runner so each capture records the
> appropriate branch once.

> **UnifiedFT (`coserving_admission_phase: both`):** sibling scheduler that
> admits FT on any step composition (prefill / decode / mixed / idle) under
> the SLO estimator's gate, NOT just prefill-carrying steps. Selected at
> `VllmConfig.__post_init__` based on the new `slo.coserving_admission_phase`
> field. Soft-fall to `"prefill"` (with `logger.warning` at startup) when
> `phase == "both"` AND `ft_tokens_admission_constrain_factor != -1` —
> the proportional cap is prefill-relative and incompatible with decode-only
> admission. Two overrides vs the parent `FinetuneScheduler`:
> (1) `_initial_ft_budget` drops the `decode_only → 0` short-circuit (the
> parent class gets a new hook method that defaults to today's behaviour);
> (2) `_slo_ft_budget` applies `slo.decode_only_ft_safety_margin` (default
> 0.7) as a multiplier on `max_tbt_slo` when the upcoming step is decode-only.
> Justifications for the tighter decode-only margin: the eager penalty from
> losing the CUDA-graph fast path can dominate sub-5ms decode-only step time;
> the estimator γ coefficient hasn't seen many decode-only + FT samples until
> online refit accumulates them. Everything else inherits — same injection,
> coordinator, backward triggers, async-safety,
> `match_prefill_workload_factor`. The leaky-bucket counter still self-gates
> on `feats.t_in > 0`, so long decode-only stretches naturally rate-limit FT
> once banked credit is consumed (no separate "match decode" knob).
> Eval tooling: `auto_benchmark.py --scheduler {prefill,both}` picks the
> corresponding YAML; output suffix extended to
> `_co_factor_<X>_phase_<Y>_<mode>`. `eval/auto_plot_schedulers.py` emits two
> A/B PNGs centered on `phase=both`: `both_vs_inf-only` and
> `both_vs_prefill`. `eval/auto_plot.py` percentile panel dropped (figure
> now single-row 4-panel).

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
per prefill step; `-1` disables) and P5.3's **`match_prefill_workload_factor: float`**
(leaky-bucket — accumulate observed prefill tokens, admit ONE FT sample when
`(counter + t_in) * factor ≥ next_sample.input_len`, reset on any FT admit;
default 0.0; wins over the per-step factor when `> 0`). P5.2's **backward CUDA-graph knobs** —
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

**UnifiedFT addition (`_initial_ft_budget` hook):** the FT-admission gate that
short-circuits `budget=0` on decode-only steps was extracted into a new
`_initial_ft_budget(feats, earliest_arrival)` method. Default body matches
today's behaviour; the sibling `BothPhaseFinetuneScheduler` (next entry)
overrides it to relax the gate. The body of `schedule()` now calls
`self._initial_ft_budget(...)` instead of inlining the `decode_only` check.

### `vllm/deltaserve/ft_scheduler_both.py` — `BothPhaseFinetuneScheduler`
**Function:** sibling of `FinetuneScheduler` selected when
`slo.coserving_admission_phase: both`. Removes the `decode_only → 0`
short-circuit by overriding `_initial_ft_budget` (always returns the SLO
budget regardless of step composition) AND `_slo_ft_budget` (applies
`slo.decode_only_ft_safety_margin` as a multiplier on `max_tbt_slo` when the
upcoming step is decode-only). Inherits everything else — injection,
coordinator wiring, backward triggers, async-safety, the
`match_prefill_workload_factor` leaky-bucket gate (which self-gates on
`feats.t_in > 0` so long decode-only stretches naturally rate-limit FT).
**Used by:** the running engine when the phase is `"both"` — selected by
qualname string in `config/vllm.py.__post_init__`. Soft-fall to
`FinetuneScheduler` (with `logger.warning`) when
`ft_tokens_admission_constrain_factor != -1`.

---

### `vllm/deltaserve/bwd_services/common/` — the family-agnostic LoRA-SFT trainer stack (P8)
**Function:** everything the trained families share, split out of the old 1000-line
`llama3.py`: `ops.py` (RoPE / RMSNorm / LoRA projection + backwards), `attention.py`
(`attn_forward_core` — the per-sample GQA loop, extracted — and `attn_backward_core`),
`ffn.py` (`ffn_forward_tail`, `ffn_backward_core`), `head.py` (`head_backward`,
`logits_chunked`), `tp.py` (`lora_shard_slice`, `init_backward_tp_group`,
`reduce_partial` — the residual-aware "reduce only the partial" helper), `family.py`
(the `Family` record: layer functions, frozen per-layer weight map, graph hooks,
`supports_saved_qkv`), `trainer.py` (`LoraSftTrainerService`: `_build_state` builds
`self.base[i]` from `family.layer_weights` — fused `qkv`/`gate_up` sliced by local widths,
everything else copied verbatim, so Qwen3's `q_norm`/`k_norm` arrive with no family code;
reads `meta["lm_head_key"]` so tied-embedding models work; binds the family's layer
functions once; `process_backward`, `_publish_to_served`, DIAG, verify), `graph.py`
(`GraphedBackward`, the `git mv` of `llama3_graph.py`: the captureable forward is the
family's `graph_forward_core(runner, lw)` — `None` → forward eager, FFN/attn graphs still
captured; `static_ctx_flat` is `[s, q_size]`, not `[s, hidden]`). Pure moves: Llama-3 numerics
are bit-identical (all five Llama gates re-pass); `q_size` replaces the old `D = Hq*Hd`
conflation so `head_dim*num_heads != hidden` models (Qwen3-0.6B) work.
**Used by:** `bwd_services/llama3.py`, `bwd_services/qwen3.py`, `tests/bwd_harness.py`, all `tests/test_{llama3,qwen3}_*.py`.

### `vllm/deltaserve/bwd_services/qwen3.py` — Qwen3 family (P8)
**Function:** the Qwen3 layer math: `layer_forward` / `layer_backward` /
`layer_backward_graphed` composed from `common/*` like Llama-3's, plus the per-head
`q_norm`/`k_norm` (`rmsnorm` on `[n,H,Hd]`) between the projection and RoPE in the forward
and the fp32 `rmsnorm_backward` between the RoPE backward and the q/k projection backward;
the cache carries the pre-norm `q_pre`/`k_pre`. `LAYER_WEIGHTS` = the shared map +
`self_attn.{q,k}_norm.weight`. `graph_forward_core` (the captureable forward with the norm between projection and RoPE; writes the
runner's `static_q_pre`/`static_k_pre` for the backward tail — `common/graph.py` allocates them and
`cache_views` exposes them). `QWEN3 = Family(..., supports_saved_qkv=False)` (the `self_attn.attn`
hook captures post-norm q/k, so the saved-qkv shortcut is not exact → the trainer recomputes);
`Qwen3BackwardService(LoraSftTrainerService)`. Graph parity in `tests/test_qwen3_backward_graph.py`.
No new TP collective (the norm is per-head). Gradchecked in `tests/test_qwen3_backward.py`.
**Used by:** `bwd_services/registry.py` (`Qwen3ForCausalLM` / `qwen3`).

### `vllm/deltaserve/bwd_services/registry.py` — arch → service (P8)
**Function:** `_SERVICES` maps HF architecture strings + aliases to `"module:Class"`
(lazy import — a child only loads its own family); `get_service`, `is_trainer`
(class-attribute query, False for unknown names), `get_family`, `supported_names`.
`base.get_service` is a shim over it (the family modules import `base`).
**Used by:** `bwd_services/base.py`, `v1/worker/gpu_worker.py` (`_maybe_share_ft_served_lora` gate), `deltaserve/ft_meta.py`.

### `vllm/deltaserve/ft_meta.py` — the backward `meta` dict (P8)
**Function:** `build_backward_meta(hf_config, ft_cfg, ...)` — the single producer of the
dict `BackwardProcess.share_weights` ships to the child (model dims, weight keys,
optimizer + backward flags, TP geometry), moved out of the worker; `rope_theta_of`
(**bug fix**: reads `hf_config.rope_parameters["rope_theta"]` first — under transformers
5.x the top-level attribute is gone and the old `getattr(..., 10000.0)` silently fed
theta=10000 to every backward remat; Llama-3 is 5e5, Qwen3 1e6); `read_lora_scaling`
(alpha/r from `adapter_config.json`); `effective_save_attn_qkv` (off when the family cannot
consume the saved q/k, so the accumulator does not allocate them).
**Used by:** `v1/worker/gpu_worker.py`.

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
- Stage P8: `_get_v2_model_runner_unsupported_features` appends
  `"DeltaServe co-serving finetuning"` when `finetune_config.enable_finetuning` — all FT
  hooks live in the v1 runner, so every finetuning run (incl. `Qwen3ForCausalLM`, the one
  default-v2 arch) is routed to v1 via the upstream mechanism, and an explicit
  `VLLM_USE_V2_MODEL_RUNNER=1` fails loudly in `_validate_v2_model_runner`.

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

Stage P7 (TP). Add `SchedulerOutput.finetune_backward_trigger: dict | None`. Under TP>1
the scheduler decides when to fire a backward (it owns the store + `min_sample_len`) but
cannot reach the per-rank backward child, whose IPC handle lives on the worker. The
trigger params `{n, sample_lens, epoch, sleep_s}` ride this struct instead. It must live
here because `SchedulerOutput` is the only object BROADCAST from EngineCore to every
worker each step — and the broadcast is load-bearing: it is what makes both ranks fire
the same step, without which the per-layer NCCL all-reduces deadlock. `None` on ordinary
steps and for tp=1.
- Stage P7 / M4.2: `finetune_record_timing: bool = True` — per-step record gate for the
  runner's timing ring (the profiling pass's warmup/recorded switch), stamped by the
  scheduler so it reaches the worker under TP.

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

- Stage P7 (TP): TP geometry + shard awareness. `_maybe_share_finetuning_weights` reads
  `tp_size`/`tp_rank` from vLLM's TP group and puts them (plus `backward_nccl_port`) in
  the child's `meta`; it also **all-gathers the vocab-parallel `lm_head` to full**
  (`get_tp_group().all_gather`, a collective both ranks enter during `load_model`) so the
  backward's `head_backward` and final norm stay free of vocab parallelism — at the cost
  of ~1.05 GB resident per rank. `_maybe_setup_finetuning_accumulator` sizes the
  `mlp_gate_up` / `attn_qkv` / `attn_ctx` buffers to **local** widths (`inter//tp`,
  `q//tp`, `kv//tp`) to match the TP-sharded module-hook outputs, while the
  residual-stream buffers (`layer_in`, `final_*`) stay full `hidden_size`; it also sets
  `coordinator.relay_mode = tp_size > 1`. Every one of these is an identity at tp=1.
- Stage P8 (generic, no family branches): `_maybe_share_finetuning_weights` builds
  `meta` via `ft_meta.build_backward_meta` (+ `read_lora_scaling`) instead of an inline
  dict — this is also where the `rope_theta` bug fix lands; `_maybe_share_ft_served_lora`
  gates on `bwd_services.is_trainer(arch)` instead of `arch == "LlamaForCausalLM"`;
  `_maybe_setup_finetuning_accumulator` passes
  `save_attn_qkv=ft_meta.effective_save_attn_qkv(arch, ft_cfg)`.

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

- Stage P7 (TP): relay mode (`coord.relay_mode`, worker side). Three edits, all inert at
  tp=1: (1) in the FT capture block, do **not** call `coord.record_capture` — stash
  `(n, sample_lens)` for the relay instead, so the worker never self-triggers a backward;
  (2) call `coord.execute_trigger(cmd)` when a relayed `finetune_backward_trigger` arrives
  — including on the **idle 0-token early-return path**, or a buffer-full/epoch-flush
  trigger issued on an idle step is silently dropped; (3) in `sample_tokens`, attach
  `coord.poll_backward_relay()`, the stashed saved-counts, and `ft_started` to `output`.
  That last one lives in `sample_tokens` because the same `output` object is returned by
  the sync path *and* wrapped by `AsyncGPUModelRunnerOutput`, so setting it there covers
  both.
- Stage P7 / M4.2: the timing ring's owner tuple carries the stamped
  `finetune_record_timing` (fallback: the local coordinator flag) and the CUDA-graph mode the
  step actually ran with (pushed as `was_graph`); `sample_tokens` (relay branch) drains the
  coordinator's timing queue onto `output.finetune_timing` and relays the backward ack via
  `_ft_relay_backward_done` — own-child poll + MIN all-reduce of a done flag over
  `get_tp_group().cpu_group` (gloo, no GPU sync, only while a backward is outstanding) so
  the ack is relayed only once every rank's child is done. The relay fields are filled by
  `_ft_fill_relay_fields` on BOTH `sample_tokens` outputs and `execute_model`'s idle
  (0-token) early return — the latter is what the scheduler sees while a backward is
  outstanding with no inference in flight.

### `vllm/v1/executor/multiproc_executor.py`
Stage P7 (TP). Previously untouched by the fork — TP>1 is the first time the multiproc
executor is on the finetuning path at all.

- `WorkerProc.make_worker_process`: spawn the worker with `daemon=False` **iff**
  `finetune_config.enable_finetuning`. Python forbids a daemonic process from having
  children, and under TP each rank must fork its own backward SFT child, so this is the
  one change without which TP finetuning cannot launch. TP=1 and every non-finetuning run
  keep `daemon=True` → bit-identical to upstream. (`UniProcExecutor` never calls this.)
- `_arm_parent_death_signal()` (new module-level helper) + a call at the very top of
  `WorkerProc.worker_main`: `prctl(PR_SET_PDEATHSIG, SIGKILL)`. `daemon=False` removes
  Python's automatic reaping of workers, and the existing death-pipe monitor does **not**
  cover the gap — it is only armed *after* `WorkerProc.__init__` has run `init_device()`
  + `load_model()` (~30 s), and on EOF it merely shuts two message queues rather than
  terminating the process. Without this, an EngineCore crash during init left workers
  reparented to PID 1 holding ~15 GiB per GPU, which then failed the *next* run with a
  misleading out-of-memory error. Linux-only, finetuning-only; includes a
  `getppid() == 1` check for the case where the parent died before the `prctl` landed.
- `WorkerProc.monitor_death_pipe(..., hard_exit: bool = False)`: on EOF, escalate to
  `os._exit(1)` after shutting the queues. Defaults `False` → upstream behaviour
  unchanged; `worker_main` passes `hard_exit=<finetuning enabled>`.

### `vllm/v1/outputs.py`
Stage P7 (TP). Previously untouched. Adds the worker→scheduler half of the TP relay to
`ModelRunnerOutput` — this struct is the only object flowing back from a worker to
EngineCore each step, and under TP the scheduler-side coordinator owns the store and
admission but not the backward IPC handle:

- `finetune_saved: tuple[int, list[int]] | None` — the `(n, sample_lens)` this rank's
  runner just wrote into the FT activation buffer, so the scheduler can advance its
  buffer accounting and decide the next trigger.
- `finetune_backward_done: dict | None` — the child's ack payload, so the scheduler
  commits the trained samples and reopens admission.
- `finetune_ft_started: bool | None` — mirrors the worker coordinator's `ft_started`.
  `POST /start_finetuning` runs a `collective_rpc` that reaches **workers only**, so
  without this mirror the EngineCore scheduler never learns FT admission was opened.

All three are `None` for tp=1 (relay mode off).
- Stage P7 / M4.2: `finetune_timing: list[tuple] | None` — the worker's completed
  `(StepFeatures, duration_s, was_graph, predicted)` timing samples, drained onto the
  output every `sample_tokens` so the EngineCore scheduler's estimator sees them under TP.

### `vllm/v1/engine/core.py`
Stage P4d: in `EngineCore._handle_client_request` (ADD branch), gated on `print_step_mode`,
bump `coord.inf_req_count` and `dprint` `[engine-recv HH:MM:SS.mmm] #N ADD req=<id>` (per-request
arrival at the engine). No other engine-loop changes.

---

## Project tooling (consumes the above; not a vLLM change)

| Path | Role |
|---|---|
| `configs/serving_config_finetuning_{opt,llama3}.yaml` | sectioned config consumed by the loader |
| `configs/serving_config_finetuning_qwen3_{14b_tp2,0.6b}.yaml` | P8: Qwen3-14B TP=2 co-serving config (bring-up flags as the Llama TP=2 one; `gpu_memory_utilization 0.80`) and the Qwen3-0.6B single-GPU smoke config |
| `scripts/toy_adapters.py`, `scripts/init_adapters_{llama3,qwen3}.py` | P8: shared toy-adapter trainer (rank-16 q/k/v/o LoRA, inference + `-ft` copy) with thin per-family entry points; `init_adapters_qwen3.py --size {14b,0.6b}` |
| `tests/bwd_harness.py`, `tests/test_qwen3_{backward,tp_shard,tp_backward_gloo,train_overfit}.py` | P8: family-parametrized test harness (weights in the `lw` layout, TP sharding, autograd refs, synthetic trainer) + the Qwen3 gates (16/16, 42/42, 10/10, overfit) |
| `eval-tp/auto_benchmark_tp.py` | TP timeline benchmark: `eval/auto_benchmark.py`'s replay / results-CSV / bwd-log-trim helpers + the family-aware launcher (`--family`, `--tp`, `--co`); modes `--loose` / `--tight` / `--nutanix` / `--nutanix-600-800` from `eval/timelines/5090/`; outputs `timeline_results_<family>_tp<N>[_co_factor_<f>_phase_<p>]_<mode>.csv` + `bwd_log` / `bench_meta` / `server` siblings in `eval-tp/output/` |
| `eval-tp/auto_plot_tp.py`, `eval-tp/repair_bwd_log.py` | TP plots via `eval/auto_plot.py`'s figure builder (`<mode>_<family>_tp<N>_co_….png`, one colour across modes); bwd-log header/trim repair for logs written before the empty-file header fix |
| `eval/auto_plot.py`*, `eval/auto_benchmark.py`* | plotter: `make_figure_for_mode(timeline_csv=, infonly_csv=, title=)` overrides, `plot_throughput_curves(ft_on_bottom=True)` (finetune band at the bottom, inference stacked on top — now the default for single-GPU plots too), FT cumulative counter re-based after the window filter (warmup tokens no longer spike the first bin); benchmark: `trim_bwd_log_before` referenced an undefined `cutoff_iso` (NameError at the end of every `--co` run) — fixed |
| `eval-tp/{launch_deltaserve,ft_bench_tp}.py` | P8: `--family {llama3,qwen3-14b,qwen3-0.6b}` presets (default `llama3`); base model from the YAML's `model.model`, inference adapter from `adapters.lora_path_0`, served name = family; outputs `bwd_log_{family}_tp{N}.csv` / `server_{family}_tp{N}.log` |
| `scripts/launch_deltaserve.py` | offline launcher: YAML → `LLM`, serves via inference adapter |
| `scripts/ft_experiment_{opt,llama3}.py` | co-serving harness: launches a real `vllm serve` HTTP server with finetuning, fires a completion every 1s ×N, then shuts it down (server stdout streams the decision logs) |
| `scripts/train_opt125m_lora.py` | trains the opt-125m toy LoRA adapters |
| `scripts/bench_activation_save.py` | microbenchmark: activation-save overhead per FT step |
| `eval/auto_benchmark.py` | launches `vllm serve` (±`--co`), replays a request timeline, streams `/v1/completions` (ttft = first chunk), writes `timeline_results<suffix>.csv`. P4d/e: `--api-server-count N` (or YAML `server.api_server_count`) shards the frontend; reads `server.api_server_count` from the config. Writes `bench_meta<suffix>.json` with `t_first_wall_iso` for the plotter's t=0 anchor. **UnifiedFT**: `--scheduler {prefill,both}` (default `prefill`) selects the corresponding serving YAML via `_SCHED_CONFIGS`; output suffix is `_co_factor_<X>_phase_<Y>_<mode>` so A/B runs don't overwrite |
| `eval/auto_plot.py` | Single-row 4-panel per-mode figure (request timeline / E2E latency vs time / throughput / TTFT satisfaction). P4d: TTFT panel annotates avg/p90 TTFT + avg TBT (flagged vs `slo.{ttft_slo,avg_tbt_slo}`); E2E-latency panel overlays the inf-only (no-co) curve when its results exist. Loads `bench_meta<suffix>.json` (if present) and passes `t0_wall` to `parse_bwd_log_csv` so the FT series anchors at benchmark t=0. `--throughput-window` flag tunes the rolling-mean width on the throughput panel. (Earlier revision had a 5th E2E-latency-percentile panel; dropped at UnifiedFT — the percentile view lives in `auto_plot_schedulers.py`'s context.) |
| `eval/auto_plot_schedulers.py` | A/B comparison plotter for the UnifiedFT scheduler. Emits TWO PNGs centered on `phase=both`: `scheduler_compare_<mode>_factor_<tag>_both_vs_inf-only.png` (co-serving overhead vs no-co baseline) and `scheduler_compare_<mode>_factor_<tag>_both_vs_prefill.png` (head-to-head: `both` vs default `prefill` scheduler). Same 4-panel layout as `auto_plot.py`. Reuses `auto_plot.py` helpers (`parse_bwd_log_csv`, `_distribute_to_bins`, `_ft_per_bin`, `_smooth`, `plot_latency_vs_time`, `plot_request_timeline`, `read_slo`). Multi-series throughput panel uses per-run filled inference bands + hatched FT bands (cycling hatch patterns so overlaps stay readable) + per-run total dashed line, matching `auto_plot.plot_throughput_curves`'s visual identity. Multi-series TTFT-satisfaction panel folds per-run avg/p90 stats into a single combined text box. Prints every input file path as it resolves them so the user can see which runs landed on disk. Graceful degradation when a file is missing (`[plot] skip both_vs_X: phase=Y run not found`) |
| `adapters/{opt125m,llama3}-toy-lora{,-ft}/` | the inference + FT adapters |
| `alpaca_1000.txt` | FT corpus |
| `tests/test_config_loader.py`, `test_finetuning_store.py`, `test_phase1_step2.py`, `test_phase1_step3.py`, `test_phase1_m1.py`, `test_phase1_m2.py`, `test_merged_estimator.py`, `test_profiling_shapes.py` | CPU/GPU verification |
