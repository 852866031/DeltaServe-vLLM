# LLMStation — Evaluation Section Summary

> Source: *Resource Multiplexing in Tuning and Serving Large Language Models* (LLMStation), USENIX ATC '25 — He et al., ETH Zürich + NUS.
> This document preserves the writing order of Section 6 of the paper and captures every concrete fact (workloads, hardware, configs, numbers, figure references, and qualitative findings). It is intended as a side-by-side reference for re-imagining the DeltaServe evaluation plan.

---

## 6. Evaluation — Framing

LLMStation is evaluated on both synthetic and real-world workloads. The evaluation answers **three guiding questions**, in the order presented:

1. How does LLMStation compare to other specialized frameworks and out-of-the-box solutions for **co-execution of PEFT and LLM inference**?
2. What are the **throughput–latency tradeoffs** in LLMStation?
3. What are the **overheads** of LLMStation's scheduler and Autograd engine?

(A fourth de facto question, **adapter access distribution / adapter size sensitivity**, is addressed in §6.5.)

---

## 6.1 Experimental Setup

### Hardware
- A server with **4 × RTX 3090** GPUs.
- Servers with **4 × H100** GPUs (used singly and in pairs to build an 8-H100 setup).
- RAM: **256 GB – 512 GB** per server.
- All servers have **intra-node NVLink**.

### Models and adapters
- **Llama series** [Touvron et al.] is the only model family.
- **LoRA** is the only PEFT method.
- Three model sizes: **Llama-3.1-8B**, **Llama-2-13B**, **Llama-3.1-70B**.
- Three LoRA ranks: **r = 8, 16, 32**.

**Table 1 (paper) — model and GPU mapping:**

| Model | GPUs | # Layers | TP degree |
|---|---|---|---|
| Llama-3.1-8B | 2 × RTX 3090 | 32 | 2 |
| Llama-2-13B | 4 × RTX 3090 | 32 | 4 |
| Llama-3.1-70B | 2 × (4 × H100) = 8 H100 | 80 | 4 |

### Datasets and traces
- **Inference inputs:** ShareGPT (real-world).
- **PEFT inputs:** Alpaca (real-world).
- **Inference arrival pattern:**
  - Synthetic workloads = steady fixed RPS.
  - Real-world workloads = **BurstGPT** trace.

### Implementation
- ≈ **3,000 lines of code**.
- **Autograd engine:** modified PyTorch Autograd using **C++ stackless coroutines**.
- **Inference engine + memory manager + cache manager:** built atop **vLLM**.
- **Fusion engine:** built atop **FineInfer**.
- **Parallelism strategies:** based on **Nanotron**.
- Correctness: fine-tuning results are aligned with **Hugging Face PEFT**.

### Baselines (4 systems compared)
- **vLLM + torchtune** — out-of-the-box solution. vLLM handles serving (with Punica-style adapter batching + paged KV); torchtune handles fine-tuning. Co-located via Nvidia MPS with `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` tuning.
- **FineInfer** — temporal multiplexing baseline (Deferred Continuous Batching).
- **chunked-training** — based on FlexLLM-style chunked PEFT.
- **LLMStation** — the proposed system.

The paper notes that all baselines were **carefully tuned** (deferral bound for FineInfer, MPS thread % for vLLM+torchtune, chunk size for chunked-training).

### Metrics and SLOs
- Headline metric: **PEFT throughput in samples/s** subject to two SLO constraints **simultaneously**:
  - **P99 TTFT SLO = 500 ms** (all configurations, default).
  - **P99 TPOT SLO = 50 ms** for Llama-3.1-8B and **80 ms** for larger models (default).
- For latency studies: report **P99 TTFT and P99 TPOT** at varying PEFT-throughput targets.

---

## 6.2 End-to-End Comparison

### 6.2.1 Synthetic Workloads with Varying Request Rates (Figure 9i)
Three sub-experiments, one per model. Inference RPS is swept along the x-axis; y-axis is PEFT throughput (samples/s).

**(a) Llama-8B on 2 × RTX 3090 — TTFT 500 ms, TPOT 50 ms.** RPS sweep: 0.5, 1, 2, 3, 4, 5.
**(b) Llama-13B on 4 × RTX 3090 — TTFT 500 ms, TPOT 80 ms.** RPS sweep: 0.25, 0.5, 1, 2, 3, 4.
**(c) Llama-70B on 8 × H100 — TTFT 500 ms, TPOT 80 ms.** RPS sweep: 2, 4, 6, 8, 10, 12.

**Key low-rate observation (Llama-8B and Llama-13B, RPS ≤ 0.5):**
LLMStation outperforms baselines by:
- **2.38× – 8.17×** vs. FineInfer
- **2.53× – 14.77×** vs. vLLM + torchtune
- **1.57× – 2.18×** vs. chunked-training

**Diagnosis (per baseline):**
- **FineInfer:** temporal multiplexing → cannot co-execute decode with PEFT, leaves compute idle.
- **vLLM + torchtune:** must cap MPS thread % at launch → underutilizes GPU when no inference is running.
- **chunked-training:** does NOT suffer from the above two; its slowdown is due to **inherent additional data movement overhead** (loading the full LLM from HBM→SRAM 2N times for N chunks).

**Behavior as RPS rises:**
- FineInfer's PEFT throughput **drops to zero quickly** because it cannot guarantee TTFT.
- The other systems converge as RPS increases, until all drop to zero when TPOT can no longer be guaranteed.

**Llama-70B (8 × H100) — deployment-flexibility narrative:**
- FineInfer: 4-way TP per server, deployed on each server independently (because of cross-server bandwidth + high per-sample PEFT latency).
- vLLM + torchtune: cannot share the base model, and one server's GPU memory only fits one stack — so vLLM and torchtune are placed on **different servers**.
- LLMStation and chunked-training: deployed in the same way as vLLM + torchtune for fair comparison.
- **Result** when inference fits in 1 server: LLMStation beats FineInfer / vLLM+torchtune / chunked-training by up to **1.77× / 1.80× / 1.38×**.
- When inference saturates a single server and needs both: FineInfer and vLLM+torchtune **drop to zero PEFT throughput**; LLMStation can keep doing PEFT on the other server's spare capacity until both servers saturate.

### 6.2.2 Real-World Workloads with Varying SLO Targets (Figure 9ii)
Trace = **BurstGPT slice** with average **4.15 req/s**. Four sub-experiments:

**(a) Llama-8B / 2 × RTX 3090 / TTFT 500 ms, TPOT swept** at 40, 60, 80, 100, 120, 140 ms.
**(b) Llama-13B / 4 × RTX 3090 / TTFT 500 ms, TPOT swept** at 60, 80, 100, 120, 140, 160 ms.
**(c) Llama-70B / 8 × H100 / TTFT 500 ms, TPOT swept** at 60, 80, 100, 120, 140, 160 ms.
**(d) Llama-70B / 8 × H100 / TPOT 80 ms, TTFT swept** at 175, 200, 225, 250, 275, 300 ms.

**Qualitative findings:**
- Clustered request arrivals in real workloads let *more decode steps batch together* → more headroom for PEFT under LLMStation.
- vLLM + torchtune and FineInfer **perform worse on real-world fluctuating load than on synthetic steady load**: vLLM+torchtune must use stricter MPS limits to avoid SLO violations under bursts; FineInfer sometimes always violates TTFT.
- As P99 TPOT SLO loosens, **LLMStation and chunked-training converge** because chunked-training can use a larger chunk size.
- Under **strict P99 TTFT**, LLMStation's gain narrows: prefill can only run after the current decode step ends, so the number of co-scheduled PEFT tasklets is bounded by remaining time in **prefill**, not decode.

### 6.2.3 Case Study of GPU Utilization (Figure 9iii)
- Setup: a **3-minute slice** from Figure 9ii(a) replayed on **Llama-70B / 8 × H100**, P99 TTFT = 500 ms, P99 TPOT = 80 ms.
- Four utilization signals tracked over time:
  - **SM Active (%)**
  - **Compute Warps in Flight**
  - **DRAM Read Bandwidth**
  - **DRAM Write Bandwidth**

**Per-system observations:**
- **LLMStation:** high SM Active, high Compute Warps in Flight, high DRAM Write Bandwidth most of the time.
- **FineInfer:** high utilization only when inference load is low (cannot run PEFT during inference).
- **vLLM + torchtune:** cannot use half of GPU resources even at low inference load — base model not shared.
- **chunked-training:** high SM Active and DRAM Read Bandwidth but **low Compute Warps in Flight** → reflects its data-movement overhead.

---

## 6.3 Throughput–Latency Tradeoff

In this section SLOs are **not enforced**. Instead, the experiment fixes a target **PEFT throughput** and measures the resulting P99 TTFT and P99 TPOT. The "Inference only" line is included as a floor.

### Llama-3.1-8B / 2 × RTX 3090 (Figure 10)
- Real workload: another BurstGPT slice with **avg 0.88 req/s**.
- PEFT throughput sweep: 0.4, 0.8, 1.2, 1.6 samples/s.
- **LLMStation vs. baselines (best-case):**
  - **P99 TTFT:** up to **33.13× / 52.64× / 1.4×** lower than FineInfer / vLLM+torchtune / chunked-training.
  - **P99 TPOT:** up to **2.29× / 362.49× / 1.29×** lower.
- FineInfer's TTFT is consistently worst (its scheduler defers prefill to maximize PEFT). At PEFT = 1.6, FineInfer's P99 TTFT exceeds **8 seconds**.
- However, FineInfer's P99 TPOT can be up to **36% lower** than LLMStation when PEFT throughput is low — the price LLMStation pays for early prefill admission.
- vLLM + torchtune is dominated on both axes (under-utilization at low load + stronger MPS clamps at high load).
- chunked-training mirrors LLMStation's trend but is slightly worse, due to data-movement overhead.

### Llama-3.1-70B / 8 × H100 (Figure 11)
- Same real workload as §6.2.
- PEFT throughput sweep: 2.0, 2.4, 2.8, 3.2 samples/s.
- **LLMStation vs. baselines (best-case):**
  - **P99 TTFT:** up to **180.48× / 1.23×** lower than FineInfer / chunked-training.
  - **P99 TPOT:** up to **14.22× / 1.24×** lower.
- FineInfer's P99 TPOT is up to **28% lower** than LLMStation when PEFT throughput is low; but its P99 TTFT exceeds **25 seconds** at high PEFT throughput.
- vLLM + torchtune is **omitted from this plot**: when the targeted PEFT throughput exceeds what 4×H100 can produce, all 8 H100s have to run torchtune → no GPU left for inference.

---

## 6.4 LLMStation Autograd Engine and Scheduler Overheads

### Autograd engine (Figure 12)
- Llama-3.1-8B / Llama-2-13B / Llama-70B have **32 / 40 / 80 layers** respectively.
- Methodology: **selectively suspend and resume after each layer** to bound context-switch overhead.
- Two sequence-length panels: **512** and **1024** tokens.
- # suspends swept: 0, 8, 16, 32, 40, 80.

**Reported overhead:**
- **Single-GPU PEFT (Llama-8B on 1 × RTX 3090):** less than **0.5%** additional latency — credited to lightweight C++ stackless coroutines.
- **Multi-GPU PEFT:** up to **18%** additional latency — attributed to extra inter-GPU and inter-process synchronization plus variability amplification.

### Scheduler
- Hardware: **AMD EPYC 7313 16-Core**.
- Average per-iteration overhead of the planner + latency predictor: **18 µs**.
- When profiling results / cached runtime records exist (no predictor call needed): **< 1 µs** per iteration.
- **Latency predictor generalization test:**
  - Trained on profiling data from RTX 3090 + H100 servers.
  - Tested on **A100** servers (held-out hardware).
  - Models: Llama-3.1-8B and Llama-2-13B with LoRA rank = 8.
  - Reported **R² scores: 0.73** (8B) and **0.69** (13B). The paper explicitly notes that the impact of prediction error vanishes after the first few iterations because actual co-execution latencies get cached in a nested index keyed by (hardware, model, adapter) → (decode batch size, PEFT input length).

---

## 6.5 Impact of Access Distribution and Sizes of Adapters (Figure 13)

Setup: Llama-3.1-8B + LoRA on **2 × RTX 3090**, same real-world workload as §6.3.

### Adapter access distribution (left panel)
- LoRA rank fixed at **32**, **8 LoRA adapters** in total.
- Three access patterns:
  1. **None** — no LoRA adapter accessed by inference.
  2. **Skewed** — 50% of adapters are accessed uniformly.
  3. **Uniform** — all 8 adapters accessed uniformly.
- **Result:** LLMStation outperforms FineInfer / vLLM+torchtune / chunked-training by up to **2.98× / 2.41× / 1.74×**.
- Throughput drops *slightly* going None → Skewed.
- All variants drop to **zero PEFT throughput in Uniform** because inference-only already violates the P99 TTFT SLO.

### Adapter size (right panel)
- Number of adapters fixed at **8**, accessed-percentage at **50%** (i.e., Skewed).
- LoRA rank swept: **8, 16, 32**.
- **Result:** LLMStation outperforms FineInfer / vLLM+torchtune / chunked-training by up to **2.98× / 2.41× / 1.73×**.
- All variants degrade slightly as rank grows from 8 to 32.
- Behavior matches Punica's prior observation: at higher rank, throughput is more sensitive to access distribution (Punica reports up to 2.5× higher latency in their *distinct* (= LLMStation's *Uniform*) distribution at rank 32).

---

## Headline Numbers Cited in Conclusion (for cross-checking)
The paper claims: **1.38× – 14.77× higher PEFT throughput** vs. heavily tuned baselines while meeting inference latency SLOs, with negligible scheduler/autograd overheads.

---

## Notes for the Re-imagined Evaluation Plan (DeltaServe perspective)

These are *implicit* properties of LLMStation's evaluation that are useful when designing DeltaServe's revised evaluation, especially since LLMStation is now the most direct concurrent competitor:

- **LLMStation reports PEFT in samples/s, not tokens/s.** DeltaServe currently reports tokens/s. A direct head-to-head needs a unit-aligned metric; samples × avg-seq-length is one way, but Alpaca samples are not the same length as DeltaServe's Emotion samples (~20 tokens). Standardizing the FT dataset would avoid this confusion.
- **LLMStation does not measure FT convergence quality either** — same gap as DeltaServe. Both papers report only throughput.
- **Headline speedups are vs. tuned baselines, including chunked-training (FlexLLM-style)**. DeltaServe explicitly does *not* compare against LLMStation or chunked-training; that is the obvious missing comparison.
- **Hardware overlap is partial:** LLMStation uses RTX 3090 + H100 + A100 (predictor generalization). DeltaServe uses RTX 5090 + A100. Adding H100 to DeltaServe and/or RTX 5090 to a head-to-head with LLMStation would be defensible.
- **Models evaluated:** LLMStation uses 8B, 13B, 70B Llamas with multiple TP degrees. DeltaServe only uses Llama-7B. A model-size sweep is a clear delta.
- **SLO definitions used by LLMStation:** P99 TTFT 500 ms, P99 TPOT 50/80 ms — explicit, percentile-based, dual-constraint. DeltaServe's SLOs are configured per-experiment (100 ms / 0.1 s / 0.3 s) and the dual-constraint compliance metric is presented only in the §4.4 figure. Aligning to P99-of-both-SLOs would make the comparison legible.
- **LLMStation runs ablations on adapter access distribution and adapter rank.** DeltaServe runs neither. Both are cheap to add and directly relevant to "multi-tenant LoRA serving" claims.
- **Both autograd and scheduler overheads are micro-benchmarked** in LLMStation (Figure 12, scheduler timing, predictor R²). DeltaServe's SLO-estimator accuracy plot (Figure 5) is the analog of the predictor R²; a parallel autograd / scheduler microbenchmark is missing in DeltaServe.
- **LLMStation reports raw GPU counters (SM Active, Compute Warps in Flight, DRAM R/W bandwidth)**, not just nvidia-smi utilization. DeltaServe reports a single utilization number; switching to per-counter time series would directly counter reviewer questions about *what kind* of utilization is being reclaimed.
- **LLMStation's "design philosophy" comparison point:** in DeltaServe's Related Work, LLMStation is described as *fusing forward FT into decode*, while DeltaServe fuses forward FT into prefill. A side-by-side experiment that splits the FT-forward-merge target (decode vs. prefill) on the same workload would empirically substantiate that claim, which is currently only argued in prose.
