# DeltaServe-vLLM — Project Context

## What this project is

DeltaServe co-serves LoRA SFT **finetuning** with **inference** on the same GPUs:
FT samples ride inside inference batches, their activations are saved during the
forward, a per-rank **backward child** trains a dedicated LoRA adapter from them, and
an **SLO-aware scheduler** decides each step how much FT work to admit without
breaking inference TTFT / TBT.

It is a **vLLM general plugin** (`deltaserve-plugin/`, import `deltaserve`) on
**stock, unpatched vLLM** (`vllm-upstream/`, submodule pinned at the v0.33.0 release,
`4ba787ec9`; branch `vllm-v0.33`, env `deltaserve-v033`). vLLM
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

**vLLM version.** v0.33.0 is the last release with the v1 model runner, where every
runner hook lives (`model_runner.py`, `cudagraph.py`). vLLM `main` removed v1 on
2026-11-07 (vllm-project/vllm#63373), so the next upgrade means re-hosting those hooks
on the v2 runner (`vllm/v1/worker/gpu/`): `prepare_inputs` for the FT mask, a wrapper
around the inline forward for timing / pause / tier-C, and a `CudaGraphManager`
subclass for the `has_ft` set. Scheduler, EngineCore, LoRA and model internals carry
over (checked statically against main).

## Rules

- **vLLM stays unmodified.** If something seems to need a vLLM edit, find a seam
  (subclass, wrapper on a small method, object-level trick) and add it in
  `deltaserve/plugin.py`. Never copy an upstream method body into the plugin.
- **One file per model family** in `bwd_services/` (`llama3.py`, `qwen3.py`); shared
  logic lives in `bwd_services/common/`. Never route a new family through another
  family's file.
- **Env:** conda env `deltaserve-v033` (`/mnt/storage/conda/envs/deltaserve-v033`, vLLM v0.33.0, torch 2.14, no torchaudio; the pre-upgrade env `deltaserve-plugin` stays for the old pin):
  vllm editable from `vllm-upstream/` + the plugin editable. Weights in
  `HF_HOME=/mnt/storage/huggingface`, `HF_HUB_OFFLINE=1`.
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
| `FinetuneConfig` | parsed from `additional_config` into a `DeltaServeConfig` (compile-cache hash = graph-affecting fields only); `VllmConfig.finetune_config` property |
| worker / scheduler / async defaults | `VllmConfig.__post_init__` wrapper → `engine_config.configure_vllm_config` |
| v1 model runner forced | `_get_v2_model_runner_unsupported_features` wrapper (all hooks live in the v1 runner) |
| backward child, weight + activation + served-LoRA sharing | `worker.DeltaServeWorker` |
| FT mask, save arming, timing ring, backward pause, tier-C abort, relay | `model_runner.DeltaServeModelRunner` (adopted by class swap) |
| `has_ft` CUDA-graph set | `cudagraph.FtCudagraphDispatcher` + `FtBatchDescriptor` |
| graph-path abort | `CUDAGraphWrapper.__call__` wrapper: poll between piecewise segments |
| activation save in graphs | accumulator hooks issue `torch.ops.vllm.dserve_save_rows` (traced by Dynamo) |
| tiers A/B, rollback, rps / recv bookkeeping | `ft_scheduler.FinetuneScheduler` (`BothPhaseFinetuneScheduler` for phase `both`) |
| arrival signal, profiling pass (a failure disables FT, inference serves), estimator save, status utility, input-thread ADD signal | `engine_core.EngineHooks`, via `EngineCore` / `EngineCoreProc` wrappers + `FtInputQueue` |
| system metrics (`telemetry.metrics_dir`) | `metrics.MetricsRecorder` owned by the FinetuneScheduler: tokens from `update_from_output`, step GPU time from the timing-sample drain, backward rows from the ack (child adds gpu/wall ms + memory), memory files from `DeltaServeWorker.compile_or_warm_up_model`; final window on the stop request |
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
  completes on the GPU (no MPS daemon on this box).
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
- **The child's pipe replies are read as backward acks** (`poll_backward` takes any
  message). A new child command must not reply on it — `save_adapter` reports
  through files (`adapter_io.read_save_result`).
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
equal or higher (numbers + plots in README "Results"). Llama-3-8B (TP=1/2) and
Qwen3-0.6B are supported (Llama-3.1/3.2/3.3 too: their `llama3` RoPE scaling is
reproduced); A100 TP=4 presets exist for Qwen3-14B/32B.

**Inference correctness gates: `tests/test_phase1_m1.py` (Qwen3-0.6B) and
`tests/test_tp_inference_leak.py` (Qwen3-14B TP=2, the production preset unchanged —
same method, profile `tp2`).** FT runs with a 10×-amplified
FT adapter (a leak would move logits by ~97 nats; a positive control proves it) against
FT-off baselines, eager and CUDA-graph, admission phase `prefill` and `both`, with
staggered arrivals so FT provably co-batches with real prefills and decodes. Measured
drift is bf16 batch-shape noise (≤ 0.19 nats); divergences are exact or near ties.

**Upstream issue (not ours):** at the previous pin (117afeea4), the v2 model runner + LoRA +
CUDA graphs fails at warmup ("CUDA graphs must be captured on a non-default stream"),
plugins disabled or not. Qwen3 defaults to v2 when finetuning is off, so FT-off Qwen3
+ LoRA runs need `VLLM_USE_V2_MODEL_RUNNER=0` (finetuning runs are routed to v1).

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

Open: the live backward-graph vs eager A/B under TP on the real models; a relative
(not seconds-based) admission margin for the estimator; mixed-batch forward graphs
for Llama-3 (only Qwen3 has the save call sites).

## Useful commands

```bash
P=/mnt/storage/conda/envs/deltaserve-v033/bin/python
$P eval-tp/launch_deltaserve.py --family qwen3-14b --dry-run          # resolved vllm serve cmd
$P eval-tp/auto_benchmark_tp.py --family qwen3-14b --tp 2 --co --tight-half --kill-stale
eval-tp/cleanup_validate.sh <tag> tight-half loose-half production-600-800   # A/B vs eval-tp/output/cleanup/00_base
$P eval-tp/compare_runs.py --suffix <sfx> --a <dirA> --b <dirB>
tests/run_all.sh /tmp/ds-tests
$P docs/figures/make_overview.py && $P docs/figures/make_architecture.py   # regenerate both figures
```

Env switches: `DSERVE_TEST_FAIL_PROFILING=1` (tests: make the launch profiling pass raise),
`DSERVE_BACKWARD_NCCL_PORT` (backward NCCL group, default 29677),
`DSERVE_TP_DIAG=1` (remat-vs-captured activation check), `DSERVE_BWD_COMM_SYNC=1` /
`DSERVE_BWD_COMM_DELAY=<cycles>` (comm-stream debugging).
