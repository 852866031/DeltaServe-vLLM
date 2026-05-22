# DeltaServe — Revised Evaluation Plan

> Target: re-submission as a co-serving paper for LLM inference + LoRA SFT.
> This plan supersedes `osdi_eval.md`. Key changes from the OSDI version:
> S-LoRA dropped, dataset switched to Alpaca, primary hardware moved to a
> 4-GPU multi-tenant setting, **LLMStation added as the direct competitor**.
> The eval keeps our **loose / tight + Company X** workload framing — we do
> not mirror LLMStation's RPS-sweep or TPOT-sweep figures. Our story is
> driven by *workload shape*, not by a steady-rate sweep.

---

## 0. What's different from the OSDI plan

| Item | OSDI version | Now |
|---|---|---|
| Primary baseline | S-LoRA | **LLMStation** + **vLLM-3GPU + PEFT-1GPU split pool**. S-LoRA dropped — pre-graph era, not apples-to-apples. |
| FT dataset | Emotion (~20-tok samples) | **Alpaca-1000** (real instruction-tuning lengths). Already the live path (`serving_config_finetuning_alpaca.yaml`). |
| FT throughput unit | tokens/s | **tokens/s** (kept). Backward batching is token-bounded; tokens/s is the natural unit. We convert LLMStation's samples/s numbers to tokens/s using their reported avg seq length when citing their results. |
| Primary hardware | 1× RTX 5090 + 4× A100 | **4× A100 for the head-to-head** (LLMStation does not run on 5090). **4× RTX 5090 used for mechanism studies** that don't require LLMStation. |
| Workload framing | Synthetic burst patterns + Company X | **loose / tight + Company X** (`timeline_loose.csv`, `timeline_tight.csv` — already wired into `auto_benchmark.py --loose / --tight`). Current code already shows we beat LLMStation on FT throughput in the loose case. |
| SLO definition | Per-experiment (100 ms, 0.1 s, 0.3 s) | **P99 TTFT and P99 TPOT, dual constraint, percentile-based.** Targets fixed across plots. |
| Single-GPU deep dive | Headline (Fig 3) | Demoted to ablation context. The contested deployment is multi-GPU. |
| Model | Llama-7B (llama1) | **Llama-3-8B** primary (active dev path; GQA; matches `eval/llama3/` tooling and `packed_kv` allocator path). |

---

## 1. Guiding questions

1. **Q1 — Head-to-head on our workloads.** Across loose, tight, and the
   Company X trace, how do DeltaServe, LLMStation, and the split-pool
   baseline compare on PEFT throughput at the same dual P99 TTFT/TPOT
   SLO?
2. **Q2 — Where DeltaServe wins, and why.** The loose workload — long
   inference valleys with bursty peaks — is where our prefill-fusion +
   forward/backward CUDA-graph capture compound into a measurable FT
   throughput advantage over LLMStation. Establish this and explain it
   with utilization counters and FT-tokens-during-valley measurements.
3. **Q3 — Tight-workload SLO behavior.** Under high-pressure tight
   traffic, does DeltaServe still hold the dual SLO (yielding FT as
   needed)? Does LLMStation? Does the split-pool baseline?
4. **Q4 — Mechanism contributions.** How much does each DeltaServe
   mechanism contribute to the headline result: CUDA graph capture
   (forward + backward, treated as one mechanism) and the two-regime
   SLO estimator? (The unified memory pool gets its own dedicated
   experiment — see Q7 — rather than being squashed into a bar.)
5. **Q5 — Predictor accuracy.** Is the two-regime SLO estimator accurate
   enough across batch composition and hardware to drive admission
   correctly?
6. **Q6 — Cold-start sensitivity.** How much does offline profiling
   buy us at startup vs. relying purely on live refit? Quantifies a
   mechanism the OSDI version didn't have.
7. **Q7 — Memory manager dynamics.** Under a workload that swings
   between inference-heavy peaks and FT-heavy valleys, does the
   unified pool's page-type partitioning (KV / adapter / activation)
   track those shifts? How sensitive is overall throughput to the
   activation budget knob, and how does that compare to a statically
   partitioned baseline?

---

## 2. Common setup

### 2.1 Hardware
- **4 × NVIDIA A100 (40 GB)** — **head-to-head platform**. All
  three-system comparisons (DeltaServe vs LLMStation vs split-pool)
  run here. LLMStation's release does not run on RTX 5090, so this is
  the only platform where the head-to-head is feasible.
- **4 × NVIDIA RTX 5090 (32 GB)** — **mechanism-study platform**.
  Used for experiments where LLMStation isn't required: predictor
  cross-hardware fit, memory-manager dynamics (32 GB makes the
  unified pool the binding constraint and exposes page-type
  repartitioning visibly), and any ablation that benefits from
  showing a consumer-GPU number.
- Both servers have intra-node NVLink. ≥ 256 GB host RAM. MPS daemon
  active on both.

### 2.2 Models
- **Llama-3-8B** (only model). LoRA rank **r=16**, α=32, dropout=0.05,
  applied to all attention projections.

### 2.3 Workloads
Three traces, kept consistent across every experiment. **Inference
arrivals are pre-recorded traces, not steady RPS sweeps** — this is
deliberate; we want to show robustness on workload *shape*, not a
synthetic dial.

| Trace | Source | Pattern | Where we expect to win |
|---|---|---|---|
| **loose** | `timeline_loose.csv` | Long valleys + sharp peaks; ample co-serving slack | Yes — current results already show DeltaServe > LLMStation on FT throughput here. **This is the headline workload.** |
| **tight** | `timeline_tight.csv` | Dense peaks, minimal slack | SLO compliance > FT throughput. We expect DeltaServe to hold dual SLO and yield FT. |
| **Company X** | 20-min production trace | Multi-scale bursts, irregular | Realism check; the trace LLMStation can't claim to have seen. |

PEFT inputs across all traces: **Alpaca-1000** (`load_alpaca.py` →
`_p95.txt`), pinned so `attn_l_max` is sized to avoid the monolithic
attention-bwd fallback.

### 2.4 Baselines (3 systems compared)

**Multi-GPU model is replication, not tensor parallelism.** DeltaServe
runs **one independent server replica per GPU** (each replica holds
the full model + LoRA adapter). The benchmark client shards requests
by `request_id % num_gpus == gpu_index`. This applies to all three
systems and the corresponding configurations are stated below.

1. **DeltaServe (full)** — 4 single-GPU replicas, all mechanisms on:
   `--enable-cuda-graph --enable-prefill-cuda-graph
   --enable-bwd-cuda-graph --packed-kv --alpaca`.
2. **LLMStation** — author release, configured as 4 single-GPU
   replicas (no TP). MPS thread % and deferral bound tuned per
   workload as in their paper. Direct competitor.
3. **vLLM (3 GPU) + PEFT (1 GPU) split pool** — what production
   actually deploys. **3 single-GPU vLLM replicas** with the same
   `req_id % 3` hash routing for the inference share; torchtune/PEFT
   on the 4th GPU.

> Notes:
> - **DeltaServe-Inf** (FT disabled) retained for ablation context only.
> - **chunked-training / FineInfer** intentionally skipped. They were
>   LLMStation's strawmen, not ours; if a reviewer asks we can add a
>   single-cell comparison row to Tab 1.

### 2.5 SLOs
- **P99 TTFT SLO = 500 ms.**
- **P99 TPOT SLO = 50 ms** (Llama-3-8B, LLMStation default).
- Both enforced **simultaneously**. PEFT throughput is reported subject
  to the dual constraint. All TTFT/TPOT/E2E numbers also reported as
  CDFs and P50/P99 in tables.

### 2.6 Metrics
- **PEFT throughput:** tokens/s. (When citing LLMStation paper numbers,
  convert their samples/s using the reported avg input length so the
  comparison is in our unit.)
- **Latency:** P99 TTFT, P99 TPOT, P50 E2E.
- **Utilization:** raw NSight Compute counters (SM Active %, Compute
  Warps in Flight, DRAM R/W BW). nvidia-smi util dropped as primary
  signal.
- **Allocator occupancy:** used-pages over time, 1 Hz CSV from the
  existing `_OccupancyTracker`.
- **System overheads:** scheduler decision time per iteration;
  predictor inference time.

---

## 3. Experiments

### Exp 1 — Headline head-to-head: loose, tight, Company X (Q1)
**The main result figure. Single platform: 4×A100. One row per
workload, three systems per panel.**

- Three rows: loose / tight / Company X.
- Four columns per row, mirroring the existing `plot_loose_tight.py`
  layout but with three system traces overlaid:
  1. Inference arrival pattern (shared across systems; one trace).
  2. P99 TTFT over 30-s windows, dashed at 500 ms.
  3. Per-request E2E latency scatter.
  4. Cumulative FT tokens over time (with tokens/s annotation).
- **Driver:** extend `auto_benchmark.py` to emit a per-system suffix and
  add an `--llmstation` / `--vllm-split` mode that launches the
  appropriate baseline harness. Then a new `compare_systems_plot.py`
  reads the per-system CSVs and produces the overlay.
- **Headline numbers** reported as **Tab 1** (one row per
  workload-system, six metrics: P99 TTFT, P99 TPOT, P50 E2E, total FT
  tokens, FT tokens/s, avg SM Active %).

**Expected story:**
- Loose: DeltaServe ≫ LLMStation on FT tokens/s (this is the result
  current data already supports). Both meet TTFT/TPOT.
- Tight: DeltaServe and LLMStation comparable on PEFT (or DeltaServe
  slightly behind, intentionally yielding); DeltaServe holds dual SLO,
  the split-pool baseline doesn't (only 3 GPUs serve inference under
  the same arrivals). Whether LLMStation holds is the open question.
- Company X: DeltaServe > LLMStation on PEFT, both hold SLO.

### Exp 2 — Loose-workload deep dive: *why* we win (Q2)
**The "innovation showcase" plot. Anchored on the loose workload from
Exp 1 because that's where the gap is largest and most explainable.**

Single 4-panel figure, one platform (4×A100), all three systems:
1. Inference arrival pattern.
2. **Cumulative FT tokens over time** (the headline trace) — the
   gap opens up during long valleys.
3. **SM Active %** time series (NSight Compute) — DeltaServe should
   stay high through valleys while LLMStation oscillates because its
   PEFT path has more launch overhead per kernel.
4. **Per-iteration kernel-launch count** for the FT path — pulled from
   our profiler vs. LLMStation profiling. Quantifies the graph-capture
   advantage directly.

Companion text in the paper attributes the gap to two design choices:
- Forward FT fused into **prefill** (vs LLMStation's decode-fusion):
  during loose valleys, decode batches are small/empty and
  decode-fusion has nothing to ride on; prefill-fusion still admits
  FT tokens on the next inference burst.
- **Full-graph capture of forward + backward**: removes per-iteration
  kernel-launch overhead that LLMStation's PyTorch-Autograd-based PEFT
  path pays at every layer.

### Exp 3 — Tight-workload SLO behavior (Q3)
**Single-platform figure, 4×A100, three systems, the tight workload from
Exp 1.**

Two-panel layout:
1. **TTFT CDF and TPOT CDF**, 500 ms / 50 ms reference lines. Goal:
   show DeltaServe at ≥ P99 compliance, and read off where LLMStation
   and split-pool fall.
2. **Cumulative FT tokens vs. time** (so the reader sees DeltaServe
   yields FT during peaks rather than violating SLO).

Reported numbers fold into Tab 1; this figure is for the visual story
of *yielding* under pressure.

### Exp 4 — Mechanism ablations (Q4)
**4×A100, Llama-3-8B, loose workload (the regime where each mechanism
is most exercised).**

Single bar plot: FT tokens/s subject to dual P99 SLO, one bar per
configuration. Two ablations — packed_kv is *not* among them; it's a
layout-level GQA optimization, not a headline mechanism, and Exp 7
covers the memory-manager story directly.

- **Full DeltaServe** (all mechanisms on).
- **A1. All CUDA graphs off** — disable `enable_decode_cuda_graph`,
  `enable_prefill_cuda_graph`, *and* `enable_bwd_cuda_graph` together.
  One bar instead of three: graphs are the same idea applied to
  forward and backward; reviewers don't need fwd-only vs bwd-only,
  only "graphs vs no graphs".
- **A2. Two-regime estimator → single-regime** (patch `predict_*` to
  always use `_eager_params`). Tests whether the regime split is
  load-bearing or just bookkeeping.

Each ablation expected to lose ≥10% on FT tokens/s or inflate P99 TTFT
past the SLO floor (then the bar collapses to 0).

### Exp 5 — SLO estimator accuracy + scheduler overhead (Q5)
**Three panels (predictor) + one inline number (scheduler timing).**

Predictor accuracy panels:
- 4×5090 — predicted vs. measured prefill latency, with `_graph_params`
  and `_eager_params` fits overlaid.
- 4×A100 — same.
- 30-s slice from the Company X replay on 4×A100 — predicted vs.
  actual prefill duration over time.

Layout sweep: the OSDI `(#inf_reqs, #ft_samples)` set, plus
graph-bucket-aligned variants so we hit `_graph_params` for some shapes
and `_eager_params` for others. Reported metrics: relative error and
R² per regime. Compare side-by-side to LLMStation's predictor R²
(0.73 / 0.69 on held-out hardware).

Scheduler overhead: a single inline number (one sentence, not a panel).
Measure per-iteration decision time inside `_co_serving_step` over a
1-minute window of the loose workload. LLMStation reports < 1 µs cached
/ 18 µs uncached — we should be in the same ballpark and just say so.
A bar chart isn't earned here.

### Exp 6 — Cold-start: value of offline profiling
**4×A100, Llama-3-8B, Company X trace replay. Single 2-panel figure.**

Motivation: the system's two-regime predictor and `GraphEligibility`
mirror are seeded by `profiling_batch_generator.py` *before* live
serving. The OSDI version had nothing equivalent. The natural reviewer
question is "do you actually need the offline profiler, or would live
refit catch up fast enough?" — answer with data.

Two configurations replayed against the same trace:
- **DeltaServe-warm:** standard startup (offline profiling runs, then
  serving begins).
- **DeltaServe-cold:** offline profiling skipped; predictor starts
  with default params (or the cold-start fallback in `data_fit`).
  Live refit fires every 256 batches, as today.

Two panels:
1. **P99 TTFT over rolling 30-s windows for the first 5 minutes**, both
   configurations overlaid. Cold-start is expected to over-admit FT
   while the predictor is uncalibrated, blowing TTFT until the first
   refits land.
2. **Cumulative FT tokens** for the same window. Cold-start may
   actually *appear* ahead in early FT throughput — that's the whole
   problem (it's stealing from the SLO budget it can't yet measure).

Reported: time-to-SLO-compliance (first 30-s window where P99 TTFT
falls under 500 ms) for each configuration. We expect warm to start in
compliance and cold to take 1–3 minutes.

### Exp 7 — Memory manager dynamics
**4×5090, Llama-3-8B. Two-panel figure.**

Motivation: the actually-novel piece of our memory manager isn't the
GQA-packed KV layout (that's a layout-level trick) — it's the
**unified pool** that holds KV pages, LoRA adapter weights, and FT
activation buffers in one allocator and repartitions across page
types as workload shifts. Static-partition allocators (separate KV /
activation / adapter pools sized at startup) can't do that without
manual retuning per workload. 5090 picked here because 32 GB HBM
makes pool dynamics the binding constraint and exposes the
repartitioning visibly; on A100-40 GB the same trace barely moves
the page-type mix.

#### Panel A — Pool occupancy over time, page-type breakdown
- Workload: one full Company X trace replay (longest, most varied).
- Source: extend the `_OccupancyTracker` CSV to log page counts per
  type (`page_type_map` already distinguishes KV / adapter /
  activation / free — just emit the 4-vector at each sample).
- Plot: stacked-area chart. x = time; y = pages; layers KV (bottom),
  adapter, activation, free (top to total).
- Story: KV grows during inference peaks and shrinks during valleys;
  activation grows during valleys as more FT batches commit. Annotate
  one peak→valley transition so the swap is readable at a glance.
  This is the figure that makes "unified pool, dynamic page-type
  partitioning" a thing the reviewer can *see*.

#### Panel B — Activation-budget sensitivity, with a static-partition counterfactual
- Knob: `cfg.memory.max_finetuning_tokens` — the activation buffer
  budget. Bigger value → bigger FT batches admitted, smaller value →
  more KV pages free for inference.
- Sweep 4–6 points (e.g., `{2k, 4k, 8k, 16k, 32k}`) at fixed total
  pool size, on each of the three workloads (loose / tight / Company
  X) so the curves can be compared.
- Plot: x-axis = `max_finetuning_tokens`; left y-axis = FT tokens/s
  subject to dual SLO; one curve per workload.
- Static-partition counterfactual: for each workload, mark the
  hindsight-optimal static budget (peak of its curve). Then show
  that **no single static value** is within X% of optimal on *all
  three* workloads — a static-partition allocator forced to commit
  at startup either wastes capacity or violates SLO under workload
  drift. The unified pool is what makes the system robust without
  retuning.

Companion paragraph notes that packed_kv is one available pool layout
inside the unified manager (default for GQA models, not a paper
contribution).

### Exp 8 — Convergence sanity (optional)
**Train an Alpaca LoRA to a fixed step count under DeltaServe and under
torchtune (split-pool baseline). Final loss + a small downstream eval
score (MMLU subset or AlpacaEval). One small table, no figure.**

Not a quality argument — a one-line refutation of "but does the
interrupted backward actually train?". LLMStation doesn't do this
either; matching is sufficient.

---

## 4. Plot/table inventory

| # | Artifact | Platform | Driver |
|---|---|---|---|
| Fig 1 | Exp 1 — head-to-head (loose / tight / Company X × 3 systems × 4 panels) | 4×A100 | extend `auto_benchmark.py` + new `compare_systems_plot.py` |
| Fig 2 | Exp 2 — loose deep dive (4 panels) | 4×A100 | `compare_systems_plot.py` w/ NSight overlays |
| Fig 3 | Exp 3 — tight SLO behavior (CDFs + FT-tokens timeline) | 4×A100 | new `tight_cdf_plot.py` |
| Fig 4 | Exp 4 — ablation bars (3 bars) | 4×A100 | new `ablation_plot.py` |
| Fig 5 | Exp 5 — predictor accuracy (3 panels) | 4×5090 + 4×A100 | extend tracker dump + plot script |
| Fig 6 | Exp 6 — cold-start (warm vs cold, 2 panels) | 4×A100 | new `cold_start_plot.py` |
| Fig 7 | Exp 7 — memory manager dynamics (occupancy time series + activation-budget sweep) | 4×5090 | extend `_OccupancyTracker` page-type logging + new `mem_dynamics_plot.py` |
| Tab 1 | Exp 1 — headline numbers (workload × system) | 4×A100 | derive from CSVs |
| Tab 2 | Exp 8 — convergence (optional) | — | manual |

---

## 5. Execution order

1. **Get LLMStation running on 4×A100** with Llama-3-8B end-to-end.
   Single biggest schedule risk; verify on a known LLMStation workload
   before integrating. (No 5090 port — confirmed unsupported.)
2. **Get the vLLM-3GPU + PEFT-1GPU baseline running** on 4×A100.
3. **Exp 1 (loose + tight + Company X) on 4×A100** — the headline. If
   the loose-workload FT-tokens/s advantage doesn't survive the dual-
   SLO reframing, the rest of the paper has to be re-narrativized
   before continuing.
4. **Exp 2 (loose deep dive)** — uses Exp 1's CSVs + adds NSight
   captures. Only re-runs the system whose counters we want.
5. **Exp 3 (tight SLO).** Same Exp 1 CSVs; mostly just re-plotting.
6. **Exp 4 (ablations) on 4×A100, Exp 5 (predictor) on both, Exp 6
   (cold-start) on 4×A100, Exp 7 (memory manager dynamics) on
   4×5090.** Parallelizable across the two boxes — A100 runs the
   trace replays, 5090 runs the occupancy logging + budget sweep
   concurrently.
7. **Exp 8 (convergence)** if there's slack.

---

## 6. Risks and contingencies

- **LLMStation does not run on RTX 5090** (confirmed). Hence the
  head-to-head is A100-only, and 5090 is reserved for mechanism
  studies. We say so explicitly in the eval section rather than
  pretending we tried.
- **Replica-mode comparison fairness.** Because we run all systems as
  per-GPU replicas (no TP) with hash-routed requests, the per-GPU
  load distribution depends on the hash function. Mitigation: use a
  fixed-seed `request_id` and verify per-GPU request counts are
  within ±2% across the three systems on every run. Document the
  routing scheme in the paper to head off "but TP would be different"
  reviewer questions.
- **Loose-workload PEFT advantage doesn't survive the dual-SLO
  reframing.** Existing data is at our prior SLO settings; the dual
  P99 500 ms / 50 ms reframe is stricter. Mitigation: run Exp 1 on
  4×A100 first as a load-bearing calibration before committing the
  rest.
- **NSight Compute counter capture under MPS** is fragile. Mitigation:
  if NSight refuses under our MPS setup, fall back to NVML SM-active
  sampling (`pynvml.nvmlDeviceGetUtilizationRates`) — coarser but
  works under MPS.
- **Cold-start experiment (Exp 6)** depends on having a clean way to
  bypass `profiling_batch_generator`. Mitigation: a config flag
  (`finetune.skip_offline_profiling=true`, ~1-line plumbing) gating
  the call site in `manager.estimate_finetuning_overhead`. Verify the
  cold-start path actually exercises `data_fit`'s fallback before
  committing to the experiment.
- **Memory-dynamics experiment (Exp 7) needs visible page-type
  variance.** If the 5090 pool is sized so generously that "free"
  pages dominate the stacked area, the repartitioning story doesn't
  read. Mitigation: pre-check the occupancy CSV; if free dominates,
  shrink `unified_mem_manager_max_size_gb` so the working set is the
  binding constraint, and document this in the figure caption. (This
  is the same mitigation as for the activation-budget panel — same
  knob.)
- **Convergence experiment (Exp 8)** is new. If it doesn't fit the
  schedule, drop it and add one paragraph noting the omission is
  symmetric with LLMStation's.

---

## 7. What we will explicitly *not* claim

- We do not claim better PEFT *quality* than LLMStation. Exp 8 is a
  sanity check, not a quality argument.
- We do not claim multi-adapter serving — codebase serves one adapter
  at a time.
- We do not claim multi-node — all hardware is single-node.
- We do not claim Llama-70B — fits LLMStation's 8-H100 setup, but
  not our per-GPU-replica testbed where each GPU must hold the full
  model.
- We do not claim a tensor-parallel deployment. Multi-GPU = N replicas
  with hash-routed requests. Cross-GPU TP is left to future work.
- We do not run an RPS-sweep or TPOT-sweep figure mirroring
  LLMStation's. Our story is workload-shape (loose/tight/real), not
  steady-rate sensitivity.
