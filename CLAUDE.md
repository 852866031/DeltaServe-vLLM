# DeltaServe-vLLM — Project Context

## What this project is

DeltaServe co-serves LoRA SFT **finetuning** with **inference** on the same GPUs:
FT samples ride inside inference batches, their activations are saved during the
forward, a per-rank **backward child** trains a dedicated LoRA adapter from them, and
an **SLO-aware scheduler** decides each step how much FT work to admit without
breaking inference TTFT / TBT.

It is a **vLLM general plugin** (`deltaserve-plugin/`, import `deltaserve`) on
**stock, unpatched vLLM** (`vllm-upstream/`, submodule pinned at the v0.34.0 release,
`62d4bc3c7`, on branch `v2-runner-port`; `main` is still at v0.33.0, `4ba787ec9`). vLLM
supplies multi-LoRA batching, paged KV, the scheduler and the multi-process engine;
the plugin adds activation capture, the backward process, the FT-admission
scheduler + estimator, and the interrupt/pause machinery.

**`README.md` is the user-facing reference** — architecture figure, supported
models, the forward/backward/CUDA-graph/interrupt/throttle/optimization tables,
install, launch, experiments, tests. Read it first.

The **`legacy` branch** holds the pre-plugin fork (`dserve-vllm/`), the original
DeltaServe framework (`DeltaServe/`, with its own CLAUDE.md), and every
phase-by-phase design note (`INTEGRATION_PROGRESS.md`, `VLLM_FORK_CHANGES.md`,
`CLEANUP_LOG.md`, …). Use it as read-only history: `git show legacy:<path>`.

## Layout

```
deltaserve-plugin/deltaserve/   ALL our code (see README "Repository layout")
vllm-upstream/                  stock vLLM submodule — NEVER edit
configs/                        template_<model>.yaml (commented starting points); preset/ = the configs behind
                                experiments + tests. Sections: finetune_param / finetune_features / throttles /
                                slo / debug / telemetry + plain vLLM sections (old single `finetune` still loads)
adapters/  data/                toy LoRA adapters; datasets (FT corpora, data/timelines/ request traces)
eval-tp/                        main eval (launcher, timeline replay, compare_runs, cleanup_validate.sh)
eval/  eval-arxiv/              single-GPU eval; long-context probe
scripts/                        adapter builders, convert_config_layout.py, timelines/ generators
tests/                          standalone scripts (python, NOT pytest); tests/run_all.sh
docs/figures/                   overview + architecture figures + generators (make_overview.py, make_architecture.py), result plots
```

**vLLM version and model runner.** vLLM has two model runners: v1
(`gpu_model_runner.py`, gone after v0.33) and v2 (`vllm/v1/worker/gpu`, the only one
from v0.34). The plugin hosts its hooks on either: `runner_common.FtRunnerMixin` holds
the per-step logic, `model_runner.py` + `cudagraph.py` host it on v1,
`model_runner_v2.py` + `cudagraph_v2.py` on v2. `finetune_features.model_runner` =
`auto` (default: v1 where vLLM has it, else v2) | `v1` | `v2`; `DSERVE_MODEL_RUNNER`
overrides it. The v1 host stays until v0.34 is confirmed on the A100 machine, so a
v0.33 env remains a fallback (the same plugin runs on both).

The v2 runner has no `_model_forward` and dispatches before it prepares inputs, so on
v2: mask + FT mode in `gather_batch_req_state`, forward begin on
`step_timing.forward_start()`, `skip_compiled` from a forward pre-hook on the model,
forward end after the stock `execute_model`, retired FT requests in `finish_requests`,
the `has_ft` set as a second `capture()` pass. Its profile run selects `max_loras`
dummy adapters (v1's selects none), so the v2 runner removes the pinned FT adapter
before it instead of after (the first FT request reloads it, on both runners).

**v0.34 port (2026-11-25), validated on the 5090s; A100 verification by the user
pending.** Every test passes on v0.33/v1, v0.33/v2 and v0.34 (55 scripts: both leak
gates, the live adapter-routing run). Qwen3-14B TP=2 timelines (tight-half /
loose-half / production-600-800), all 100 % TTFT SLO:
- v0.33 v1 → v2: FT tok/s 349→372 / 627→630 / 466→463, TTFT p50 within 2 ms.
- v0.33 v2 → v0.34: FT 372→369 / 630→629 / 463→465, TTFT p50 within 4 ms, step times
  within 0.1 ms.
Runs in `eval-tp/output/cleanup/v2port_{v1,v2,v034}`. What v0.34 changed for us:
- **Compile cache key:** `VllmConfig.compile_factors()` replaces `compute_hash()`;
  `DeltaServeConfig` implements both (without `compile_factors` every finetune knob
  would key the cache).
- **Extensible KV cache (default on):** the KV cache is sized ONCE, from the device
  memory free after warmup + graph capture, within `gpu_memory_utilization`. On 14B
  TP=2 at 0.8: 30,720 tokens (v0.33: 50,448), no effect on the three timelines. The
  backward child keeps growing after that measurement (0.8 → 2.25 GiB per GPU; 1.3 →
  5.2 GiB peak with a 1024-token backward), and nothing resizes the cache later: **keep
  `gpu_memory_utilization` below 1 as the child's headroom** (vLLM's startup warning
  suggests dropping it; its own exception list covers our case).
- `VLLM_USE_V2_MODEL_RUNNER=0` is an error on v0.34: tests pin FT-off baselines through
  `runner_select.pin_vllm_runner`.
- New scheduler queue `async_kv_waiting` (KV connector loads): in the FT cleanup list.

**Findings of the same day (v0.34, production timeline, single runs):**
- FT-only forward graphs off: FT-only step 53.7 → 61.7 ms, FT −2.6 %.
- **The backward's attention never used its graph**: the padded-attention budget
  (`backward_cuda_graph_attn_bn_max` 8 × `_l_max` 64) held no real cycle (13 samples per
  256-token cycle, 15-26 per 512), so every cycle's attention ran eager, on both hosts and
  on the old fork too, with one log line at the first overflow. Now: both keys default to
  `auto` (`bwd_budget.py`: replays the epoch's batches — admission is smallest-first — and
  takes the cheapest shape that holds 99 %); a configured budget gets a launch line with
  the share of the data it holds; eager cycles are counted (ack `attn_graph`,
  `backward.csv`, `/finetuning_status`, a log line every 200 cycles while it happens).
  5090, 16 × 48 vs 8 × 64: cycle 147 → 122 ms, FT +6 %.
- **torch 2.14's `torch/_native` layer slows the eager backward on a slow-CPU host**
  (Python dispatch checks before bmm / sum / rmsnorm / ... on CUDA): A100 cloud VM,
  Qwen3-14B TP=4, FT only: 2111 µs/token vs 1801 on torch 2.11, and 1764 with
  `TORCH_DISABLE_NATIVE_JIT=1`. `finetune_features.backward_disable_torch_native`
  (default true) sets it around the backward child's spawn ONLY; never for the serving
  processes (their overridden ops would change numerics). No slowdown was seen on the
  5090 desktop CPU.
- Activation buffer memory, 14B TP=2, all saves on: 2.70 MB per token per GPU (256 →
  0.65 GiB, 1024 → 2.58, 2048 → 5.16; gate_up is 52 % of it). A 1024-token buffer AND
  backward (`max_saved_finetuning_tokens: 1024`) with `gpu_memory_utilization: 0.86`
  runs on the 5090s without OOM: worker 26.6 GiB, child ≤ 5.2 GiB, KV 22,528 tokens;
  FT 462 → 592 tok/s, but backward cycle 147 → 468 ms and TTFT p99 87 → 188 ms (still
  100 % SLO) — the cost of a coarser yield, which a 256-token backward avoids.
- `metrics.final_flush` could write the final window before the last periodic one
  (writer-thread buffer): fixed (`_Writer.drain`), `tests/test_metrics.py` H2.

**Gradient accumulation (branch `grad-accumulation`, 2026-11-25).**
`finetune_param.ft_step_size` = FT tokens per optimizer step = the activation buffer;
`bwd_batch_size` = tokens per backward batch (must divide it). Neither set: both are
`max_saved_finetuning_tokens`, and every path is the one before the feature
(`coordinator.multi_batch` False). With several batches per step:
- the coordinator sends the child whole samples of ≤ `bwd_batch_size` tokens as soon as
  the committed rows hold a batch (`_next_batch`), one at a time; the batch that empties
  a closed buffer is FINAL (clip + update + publish); an empty final batch ends a step
  whose rows were all trained already. The command carries `start` / `first` / `final`
  (also on the TP relay command).
- FT admission stays open while a non-final batch runs (the weights do not change) and
  closes only for the final one (`backward_blocks_admission`); a step carries at most
  one batch's worth (`per_step_budget`), so forward steps keep their size.
- the trainer weights each batch's gradient by its predicting rows and divides by the
  step's total: k batches == one backward of the step (`tests/test_grad_accum.py`).
- a step that carries FT tokens pauses the child like an inference prefill
  (`FtRunnerMixin._ft_pauses_backward`): FT forwards never overlap a backward, so the
  estimator is unchanged (FT steps taken with a batch in flight: pred/actual 1.01).
Invariants it added:
- **Nothing is reserved behind an abortable FT-only step** (`abortable_in_flight`): rows
  are handed out in order, and a tier-C abort gives the aborted step's rows back; a
  later step already in flight keeps ITS rows, leaving a hole of unwritten rows the
  next commit counts as saved (zeros → loss ln(vocab) → NaN after the update).
- **The save op moves `op_rows` (= one backward batch) rows per step**, not the buffer:
  a step with more FT rows must take the hook path (`_build_finetune_mask`), and the
  admission loop caps a step at `per_step_budget`.
- **No zero-fill behind a batch of a multi-batch step** (the forward is writing the
  step's later rows, and the next step's right after the final ack).
- `ft_sizes.py` stays import-free (the child's handover thread reads it; importing
  vLLM's config there made the adapter switch time out).
Qwen3-14B TP=2, production timeline, v0.34, 16 × 48 backward budget, single runs —
256 / 256 → 1024 / 256 (`gpu_memory_utilization` 0.86): 100 % TTFT SLO both; FT 490 →
524 tok/s; TTFT p50 68 → 72, p99 88 → 99 ms; avg TBT p50 20.5 → 23.1 ms (a decode step
overlapping a backward takes 23.7 vs 12.7 ms in both; 40 % of them overlap one instead
of 20 %); loss after 185 s 0.83 → 1.00 (a quarter of the updates); KV 26,624 tokens;
worker 26.8 GiB, child 2.35 GiB. For comparison 1024 / 1024: FT 592, TTFT p99 188 ms.
Both leak gates pass with 1024 / 256 (`DSERVE_M1_FT_SIZES`, `DSERVE_M1_GPU_MEM_UTIL`,
`DSERVE_M1_FT_EXTRA`). Runs: `eval-tp/output/cleanup/ga_1024_256_r2`, `v034_bwd_bn16_l48`.
Known: batches are whole samples, so a 1024-token step is 4 batches of ~247 + a tail of
~38 (one more cycle's fixed cost); steps taken with a paused backward in flight are
still excluded from the estimator's fits (as inference prefills always were).

**Agreed next steps (2026-11-25), in order:**
1. Gradient accumulation, later phases: backward batch size as an estimator feature, a
   larger batch while no inference runs, sizing by the overlapping decode batch; A100
   presets → 2048 / 512 (not runnable here).
2. Backward attention budget: `auto` is done (`bwd_budget.py`, follows `ft_step_size`
   / `bwd_batch_size`). Left as options: several graph shapes / length-bucketed batches.
3. Fill the graph padding: admit FT tokens up to the next captured graph size.
4. Worker-side dummy runs instead of the launch profiling pass.
5. FT graph path on the breakable CUDA graph (FT batches run without a graph there
   now); accept only after measuring abort latency and prefill TTFT under a paused
   backward against the compiled path.
6. Drop the v1 host (`model_runner.py`, `cudagraph.py`, `runner_select`).
Candidates, measure first: one graph set instead of plain + `has_ft`; Llama-3 save
call sites; refusing unvalidated combinations at launch (context / pipeline
parallel); speculative decoding (the user's call). Not doing: FT samples resident on
the GPU; skipping FT rows in the KV write (the attention reads K/V back from the cache).

## Rules

- **vLLM stays unmodified.** If something seems to need a vLLM edit, find a seam
  (subclass, wrapper on a small method, object-level trick) and add it in
  `deltaserve/plugin.py`. Never copy an upstream method body into the plugin.
- **One file per model family** in `bwd_services/` (`llama3.py`, `qwen3.py`); shared
  logic lives in `bwd_services/common/`. Never route a new family through another
  family's file.
- **Host facts live in `CLAUDE.local.md`** (git-ignored, one per checkout): GPUs, the
  conda env (vLLM editable from `vllm-upstream/` + the plugin editable), `HF_HOME`
  (weights are read with `HF_HUB_OFFLINE=1`), output dirs, limits. On a host without
  one, find out (`nvidia-smi`, `conda env list`, where the weights are), write it, and
  tell the user. Size runs to the host: TP and memory settings follow its GPUs.
- **Tests are scripts:** `python tests/test_x.py`, or `tests/run_all.sh <out_dir> [glob]`.
- Long-running servers / benchmarks (`launch_deltaserve.py`, `auto_benchmark_tp.py`,
  `cleanup_validate.sh`): give the user the command and the expected output rather
  than running them yourself, unless asked.
- New features go behind a config flag that defaults to the previous behaviour
  (bit-identical when off). A new key goes into `FINETUNE_SECTIONS`
  (`config_loader.py`) under its section, and into the templates with a comment.
- **Presets reproduce runs:** edit `configs/preset/*.yaml` only together with the
  experiments that use them; user-facing defaults live in `configs/template_*.yaml`.

## How the plugin hooks into vLLM

Co-serving is switched on per launch with `--additional-config '{"deltaserve": {...}}'`
(the launchers build it from a YAML via `deltaserve.config_loader.finetune_cli_args`).
`plugin.register` runs in every vLLM process (and on unpickling the config):

| What | Seam |
|---|---|
| `FinetuneConfig` | parsed from `additional_config` into a `DeltaServeConfig` (compile-cache key = graph-affecting fields only: `compile_factors` / `compute_hash`); `VllmConfig.finetune_config` property |
| worker / scheduler / async defaults | `VllmConfig.__post_init__` wrapper → `engine_config.configure_vllm_config` |
| model runner choice (vLLM ≤ 0.33) | `_get_v2_model_runner_unsupported_features` + `use_v2_model_runner` wrappers, by `finetune_features.model_runner` (`runner_select.py`) |
| backward child, weight + activation + served-LoRA sharing | `worker.DeltaServeWorker` |
| FT mask, save arming, timing ring, backward pause, tier-C abort, relay | `runner_common.FtRunnerMixin`, hosted by `model_runner_v2.DeltaServeModelRunnerV2` (built by `Worker._make_model_runner`) or, on the v1 runner, `model_runner.DeltaServeModelRunner` (class swap) |
| `has_ft` CUDA-graph set | v2: `cudagraph_v2.FtCudaGraphManager` (class swap after `initialize_kv_cache`); v1: `cudagraph.FtCudagraphDispatcher`; `FtBatchDescriptor` keys the set in both |
| graph-path abort | `CUDAGraphWrapper.__call__` wrapper: poll between piecewise segments |
| activation save in graphs | accumulator hooks issue `torch.ops.vllm.dserve_save_rows` (traced by Dynamo) |
| tiers A/B, rollback, rps / recv bookkeeping | `ft_scheduler.FinetuneScheduler` (`BothPhaseFinetuneScheduler` for phase `both`) |
| arrival signal, profiling pass (a failure disables FT, inference serves), estimator save, status utility, input-thread ADD signal | `engine_core.EngineHooks`, via `EngineCore` / `EngineCoreProc` wrappers + `FtInputQueue` |
| system metrics (`telemetry.metrics_dir`) | `metrics.MetricsRecorder` owned by the FinetuneScheduler: tokens from `update_from_output`, step GPU time from the timing-sample drain, backward rows from the ack (child adds gpu/wall ms + memory), memory files from `DeltaServeWorker.compile_or_warm_up_model`; final window on the stop request |
| live dashboard (`dserve-top`) | `live.LiveFeed` inside the recorder (step stream with folded decode runs, per-second series, rolling `steps.*.jsonl.gz`); FT admission codes stamped on `StepFeatures.admit` by `admit_ft_to_step`; `/telemetry/{data,start,stop,memory}` → `FinetuneScheduler.telemetry`; `deltaserve/top.py` (stdlib) polls and serves `top_page.html` |
| trained-adapter checkpoints | `save_adapter` cmd to the child (fire-and-forget; result via files, `adapter_io`); TP shards gathered to rank 0 over the children's gloo group |
| `disable_log_stats` auto-on | `EngineArgs.create_engine_config` wrapper |
| relay data (TP) | attributes on `SchedulerOutput` / `ModelRunnerOutput` (pickled with them) |
| TP workers spawning children | daemon check lifted around the child's `start()`; `PR_SET_PDEATHSIG` |
| HTTP routes | `register_vllm_serve_api_routers` wrapper → `deltaserve/api.py` |

## Load-bearing invariants (break these and it hangs or silently degrades)

- **FT samples are prefill-only**: one forward, no KV past the step, no sampler
  output, retired the same step, invisible to the frontend.
- **Only hidden states are saved**; the LM head runs in the backward (fp32).
- **Precision contract:** fp32 attention scores/softmax, RMSNorm, LM head / final
  norm, LoRA masters; bf16 bulk GEMMs. Downgrading the GQA scores matmul made loss
  plateau.
- **`_maybe_pause` stays outside every captured region** and runs at every backward
  boundary (between the per-layer graphs, inside the LM-head chunk loop). It is how
  the child yields the GPU; `pause_until_prefill_done` holds it until the prefill
  completes on the GPU (needed when no MPS daemon runs).
- **Persistent buffers live outside the CUDA-graph pool** (LoRA grads, attention ctx,
  activation buffers) — pool aliasing produced NaNs in the original.
- **TP lock-step:** the backward trigger is *broadcast* on `SchedulerOutput`; never
  make FT admission or the pause decision rank-local. Children vote over gloo at
  each pause boundary; never add a collective between two boundaries only one rank
  can reach.
- **Tier-C aborts are rank-symmetric** (shm `FtArrivalSignal` + gloo MAX vote per
  layer) and apply only to FT-ONLY forwards — a batch with inference tokens is never
  aborted. The engine reads the abort from `ModelRunnerOutput.finetune_aborted`.
- **The backward ack is relayed only when every rank's child is done**, including on
  idle (0-token) steps.
- **Grad clip under TP goes through `tp.clip_layers_symmetric_`**; bucketed factor
  reduces are ordered only via `tp.CommQueue`.
- **Never reduce `grad_x` wholesale** — reduce only the partial (attention / FFN)
  part; the residual passthrough is already full on every rank.
- **Vocab padding:** use the real local `lm_head.shape[0]`, not `vocab // tp`.
- **An FT request vLLM did not schedule can sit in more than one queue** (skipped
  because the batch already holds `max_loras` adapters: `waiting` /
  `kv_holding_waiting` + the `deferred_waiting` set in vLLM 0.33, `async_kv_waiting`
  in 0.34). The cleanup in
  `_schedule_step` removes it from all of them; left in a queue without its
  `self.requests` entry it kills the engine. `tests/test_ft_lora_limit.py`.
- **`_publish_to_served` stays under `torch.no_grad()`** — otherwise every publish
  grows an autograd graph (150 MiB per backward on Qwen3-14B until the child OOMs).
- **The child's pipe replies are read as backward acks** (`poll_backward` takes any
  message). A new child command must not reply on it — `save_adapter` reports
  through files (`adapter_io.read_save_result`). The adapter-handover commands
  (`save_lora`, `switch_adapter`, `export_lora`) do reply: they are the worker-RPC
  fallback of the handover line and are sent only while no backward is in flight.
- **Child-side collectives follow the command stream:** a save gathers over gloo,
  so every rank must reach it at the same point (manual saves are sent to all
  children at one engine step; periodic ones count *attempted* cycles, identical
  on every rank).
- **RoPE in the backward must equal the served RoPE:** `ft_meta.rope_scaling_of`
  refuses variants `ops.rope_inv_freq` cannot reproduce (Llama-3.1+ is `llama3`).
- **`disable_finetuning` is sticky** — the TP mirror of the workers' `ft_started`
  must not reopen it.

## Status (2026-11)

Plugin on stock vLLM validated against the original fork on Qwen3-14B TP=2 (2× RTX
5090), tight-half / loose-half / production-600-800 at TTFT SLO 0.4 s: 100 % TTFT
satisfaction everywhere, TTFT p99 201→137 / 204→134 / 295→~100 ms, FT throughput
equal or higher (numbers + plots in `docs/results.md`). Llama-3-8B (TP=1/2) and
Qwen3-0.6B are supported (Llama-3.1/3.2/3.3 too: their `llama3` RoPE scaling is
reproduced); A100 TP=4 presets exist for Qwen3-14B/32B.

**Inference correctness gates: `tests/test_phase1_m1.py` (Qwen3-0.6B) and
`tests/test_tp_inference_leak.py` (Qwen3-14B TP=2, the production preset unchanged —
same method, profile `tp2`).** FT runs with a 10×-amplified
FT adapter (a leak would move logits by ~97 nats; a positive control proves it) against
FT-off baselines, eager and CUDA-graph, admission phase `prefill` and `both`, with
staggered arrivals so FT provably co-batches with real prefills and decodes. Measured
drift is bf16 batch-shape noise (≤ 0.19 nats); divergences are exact or near ties.

**Upstream issue (not ours), gone at v0.33.0:** at the pin before it (117afeea4), the v2
model runner + LoRA + CUDA graphs failed at warmup ("CUDA graphs must be captured on a
non-default stream"), plugins disabled or not. At v0.33.0 it warms up (checked
2026-11-25). The tests still pin FT-off baselines to the runner of their FT runs
(`VLLM_USE_V2_MODEL_RUNNER`), since the two differ in bf16 numerics.

**vLLM v0.33.0 port (2026-11-09):** every test passes (incl. the 2-GPU NCCL gates and the
14B TP leak gate), and the TP=2 timeline A/B against the old pin is better on all three
workloads (FT +8 / +22 / +15 %, TTFT p50 −7 to −16 %, 100 % SLO). vLLM 0.33's piecewise
prefill steps spend ~20 ms blocked on the host (2 ms before), inference-only too —
upstream, hidden by async scheduling; profile it if it ever shows in TTFT.

**Adapter checkpoints, status, fail-safes (2026-11-09):** POST /save_adapter +
periodic / epoch saves (PEFT, fp32 masters, TP-merged), GET /finetuning_status,
trainer build errors abort the launch, a failed profiling pass degrades to
inference-only, Llama-3.1/3.2/3.3 RoPE scaling in the backward. Tests:
`test_rope_scaling`, `test_adapter_save`, `test_ft_failsafe` (CPU),
`test_adapter_save_e2e` (GPU).

**Configs + system metrics (2026-11-10):** `configs/template_*.yaml` (commented, one
per model) and `configs/preset/`; sections finetune_param / finetune_features /
telemetry; `telemetry.metrics_dir` records requests.csv / backward.csv /
summary.jsonl / memory_rank*.json, GET /system_metrics, `scripts/metrics_report.py`.
Tests: `test_metrics` (CPU), `test_metrics_e2e`, `test_templates_launch` (GPU).
`dserve-top` (live web dashboard; starts recording at runtime if the config does not):
`test_dserve_top` (CPU), `test_dserve_top_e2e` (GPU).

**Admission margin (2026-11-15):** `slo.admission_margin` = `legacy` (default, the
original ×(1+1.5·RMSE[s]) ≈ no margin) | `relative` (×(1+k·relRMSE)) | `absolute`. On the
stress timeline (`eval-tp/margin_ab.py`, 14B TP=2) FT steps over `max_tbt_slo`: legacy
10.2 % → relative k=1.5 2.0 %, FT −1 %. `relative` is the default; every preset pins
`legacy` (presets reproduce runs). Untested with admission phase `both`.

**Adapter routing + finetuning jobs** (off unless configured; README "Finetuning jobs and adapter routing"): `deltaserve/ft_job.py`
(jobs, stage registry), `adapter_save.py`, `route_middleware.py`, `route_worker.py`,
`adapter_routing/` (index, decision, router, pool, engine_router, index_process,
resolver, engine). Always say "adapter routing" / `adapter_routing`, never bare
"routing". **The scheduler places five kinds of work**: the encode step (a routed
request's adapter is chosen INSIDE the engine, by a step issued at the start of
`schedule()`; the e5 encoder lives in a worker), the forward batch, the backward
batch, indexing (GPU process sharing the encoder's weights while no inference
request is present, else CPU), and the adapter handover at a job's end (save /
publish / switch, triggered on the engine loop, done off it). Invariants:
- **Attaching an adapter to a request recomputes its prefix-cache block hashes**
  (`engine_router.attach_adapter`) — vLLM hashed it before the adapter was known;
  without it a routed request reuses the BASE model's KV. `tests/test_engine_router.py`.
- **A held request (awaiting its encode step, or its cold adapter) is not in the
  scheduler**: it counts in `has_requests`, and aborts must reach `EngineRouter`.
- **The encode step clears the backward grant and does not restore it**; the next
  model step keeps the pause (inference prefill) or releases it
  (`model_runner._ds_route_hold`).
- **Never `EngineCore.add_lora` from the pool** (blocks the loop on every rank):
  vLLM loads lazily, the scheduler arranges a background read (cold start).
Validated on vLLM 0.33, Qwen3-0.6B and Qwen3-14B TP=2 (live test 30/30, both
inference gates); with routing off, performance and GPU memory equal the
pre-merge main. Production trace 534/534, TTFT p50/p95/p99 85/104/108 ms routed
vs 71/99/104 named, FT −4 % (README). `max_loras` ≥ adapters in use at once + 1.
- **A streamed response's start is never made to wait for the routing decision**
  (vLLM submits a streamed request only after the start was sent): it is kept back
  and sent with the first body message (`route_middleware`).
- **The handover talks to the backward children directly** (`handover_channel.py`,
  an engine background thread), never on the engine loop and never on the child's
  command pipe; only while no backward is in flight.
- **Adapter tiers are vLLM's, read through `LoraCacheMirror`** (`pool.sync_tiers`):
  act on a tier only after syncing (a stale one made removal skip `remove_lora`).
`GET /adapters/available` (API server, reads the pool registry) tells a client
what it can use. The encode time given to the TBT check is the worker's own
(the engine's wall time also holds the wait for the step in flight).
**SLO semantics (branch `grad-accumulation`, 2026-11-26).** `avg_tbt_slo` is THE token
target: enforced over a sliding window (`slo.avg_tbt_window_s`, default 1.0 s,
`deltaserve/tbt_window.py`; 0 = off), and TBT attainment is per request against it
(metrics, `dserve-top`, `compare_runs.py --avg-tbt-slo`). `max_tbt_slo` is a per-step
backstop, 0.1 s by default and in every config (a step with a prefill is slow anyway).
`slo.avg_tbt_pause_backward` (default on) + `avg_tbt_headroom` (0.9): a decode-only step
under a running backward is predicted with the overlap, and pauses the child when the
window average would pass headroom × target (the mark rides `SchedulerOutput`, held like
the throttle's pause). `slo.avg_tbt_admission_headroom` (default 1.0): the same fraction
for FT admission. 14B TP=2, v0.34, production timeline at 1× / 2× rate, 256 / 256,
single runs, TTFT 100 % everywhere:
- target 50 ms: 100 % TBT attainment up to 3×, FT 598 / 533 / 482 tok/s.
- target 20 ms (inference only: 100 / 87.8 %): no gate 58 / 37 %; gate 88 / 53 %, FT
  498 / 377 (unchanged by the gate); + admission headroom 0.9: 81 / 68 %, FT 472 / 335.
  A pause hysteresis was tried and dropped (83 / 50 %). What is left of the gap: the
  child's yield latency on the step that asks for the pause, and mixed steps.
Most presets still say `avg_tbt_slo: 0.02`: not reachable at 2× (the user's call).
Postponed: FT suppression when the system is at / over capacity (a milder
long-request workload; the ones tried overload inference alone).
Not done: `validation` stage; adapter load cost from profiling (config constants
now); the "tight" indexing level; recovery when phase 2 of an adapter switch
fails; giving up on a dead indexing process; threshold calibration on a real
adapter pool. `docs/adapter_routing_scheduling.md` is the design walkthrough.

Open: the live backward-graph vs eager A/B under TP on the real models; mixed-batch
forward graphs for Llama-3 (only Qwen3 has the save call sites); relative vs absolute
margin at another `max_tbt_slo` (they coincide at 50 ms).

## Useful commands

```bash
P=<the env's python, from CLAUDE.local.md>
$P eval-tp/launch_deltaserve.py --family qwen3-14b --dry-run          # resolved vllm serve cmd
$P eval-tp/auto_benchmark_tp.py --family qwen3-14b --tp 2 --co --tight-half --kill-stale
eval-tp/cleanup_validate.sh <tag> tight-half loose-half production-600-800   # A/B vs eval-tp/output/cleanup/00_base
$P eval-tp/compare_runs.py --suffix <sfx> --a <dirA> --b <dirB>
tests/run_all.sh /tmp/ds-tests
$P docs/figures/make_overview.py && $P docs/figures/make_architecture.py   # regenerate both figures
```

Env switches: `DSERVE_MODEL_RUNNER=v1|v2` (overrides `finetune_features.model_runner`;
on v0.33 the tests and A/B runs use it), `DSERVE_TEST_FAIL_PROFILING=1` (tests: make the launch profiling pass raise),
`DSERVE_BACKWARD_NCCL_PORT` (backward NCCL group, default 29677),
`DSERVE_TP_DIAG=1` (remat-vs-captured activation check), `DSERVE_BWD_COMM_SYNC=1` /
`DSERVE_BWD_COMM_DELAY=<cycles>` (comm-stream debugging).
