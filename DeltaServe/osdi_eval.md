# DeltaServe — Evaluation Section Summary

> Source: *DeltaServe: Efficient Co-Serving of LoRA-based LLM Inference and Fine-Tuning Using SLO-Aware Scheduling* (Section 4 + relevant figures).
> This document preserves the writing order of the original paper and captures every concrete fact (workloads, hardware, configs, numbers, figure references, and qualitative findings) so it can be used as a reference when re-imagining the evaluation plan.

---

## 4. Evaluation — Framing

The evaluation is structured around **four guiding questions**, in the order they appear in the paper:

1. **Workload sensitivity (single GPU).** How do different inference workloads — synthetic bursty traces *and* a long, irregular production trace — influence DeltaServe's capacity to preserve SLOs while admitting fine-tuning work? *(addressed in §4.2)*
2. **Multi-GPU scaling.** Can DeltaServe preserve inference SLOs and sustain fine-tuning progress when scaled to multiple GPUs? *(addressed in §4.3)*
3. **SLO estimator accuracy.** Can the SLO estimator reliably predict latency across batch compositions and hardware so the scheduler can make correct admission decisions? *(addressed in §4.4)*
4. **Scheduler adaptation.** How does the SLO-aware scheduler adapt its decisions to maintain SLOs as load increases? *(addressed in §4.4)*

---

## 4.1 Experimental Setup

### Model
- **Llama-7B** is the only model evaluated.
- The authors implement **customized forward and backward pipelines** to support mixed inference + fine-tuning execution.

### Hardware
Two configurations:
- **Single GPU:** NVIDIA **RTX 5090** (32 GB).
- **Multi-GPU:** **4 × NVIDIA A100 (40 GB)**.

### Baselines
Three systems are compared:
- **S-LoRA** — primary baseline; DeltaServe extends it. Uses FCFS with a starvation-prevention mechanism: a decoding counter resets after each prefill and increments after each decode step. When the counter reaches a threshold, S-LoRA tries to form a new prefill batch; if one exists, it pauses the active decode batch, issues prefill, and merges results into the ongoing batch.
- **DeltaServe** — full system with all scheduling, batching, and execution optimizations for co-serving.
- **DeltaServe-Inf** — variant of DeltaServe with fine-tuning **disabled**; serves only inference. Used to isolate the cost of the SLO-aware scheduler from the cost of co-serving.

### Fine-tuning Workload
- A dedicated **LoRA adapter** on Llama-7B.
- Adapter config: **rank r = 16**, scaling factor **α = 32**, **dropout = 0.05**, applied to **all attention projection matrices**.
- Dataset: **Emotion classification** [Saravia et al., 2018], 20,000 labeled samples, 6 categories, average input length ≈ **20 tokens** — chosen as representative of *lightweight adaptation* workloads.
- Optimizer: **AdamW**, learning rate **1e-3**, weight decay **0.01**.
- **Backward batch size = 256 tokens** — activations accumulate until 256 fine-tuning tokens are processed before a backward pass is triggered.

### Inference Workloads
Three workload patterns total:

| Workload | Type | Pattern |
|---|---|---|
| **burst-light** | synthetic | 2-second peaks at 8 RPS alternating with 4-second valleys at 0–1 RPS — *long low-load intervals → ample co-serving opportunity*. |
| **burst-dense** | synthetic | Inverted: 4-second peaks, 2-second valleys — *fewer idle windows, higher scheduling pressure*. |
| **Company X production trace** | real | 20 minutes; irregular spikes, multi-scale bursts, several near-zero-traffic intervals — *realistic stress test*. |

For multi-GPU experiments the **same three patterns** are reused, with request rates **scaled proportionally** to the four-GPU setting.

Plot conventions used throughout §4.2 / §4.3:
- x-axis = wall-clock time.
- left y-axis (purple line) = total inference tokens submitted per second.
- right y-axis (grey bars) = incoming RPS.

---

## 4.2 Co-serving Results: Single GPU (RTX 5090)

S-LoRA is the primary baseline so that any difference is attributable to (a) the SLO-aware scheduler and (b) support for mixed FT + inference execution.

**Figure 3** is organized as a 4-row × 3-column grid:
- Row 1: arrival pattern + total fine-tuning tokens processed.
- Row 2: TTFT (percentile) + average TBT.
- Row 3: end-to-end request latency (percentile).
- Row 4: GPU utilization over time.
- Columns: burst-light | burst-dense | Company X.
- Dashed horizontal lines in Row 2 mark configured TTFT SLO targets.

### Burst-light (column 1)
- **S-LoRA:** sharp tail-TTFT inflation; 95th–100th percentile requests inflate dramatically; **TTFT SLO is broken at the 50th percentile**. Cause: requests arrive near the end of each peak and queue behind long decode phases of prior requests.
- **DeltaServe:** flatter TTFT curve, **always below SLO**. Cause: the SLO-aware scheduler proactively issues new prefill batches whenever the predicted latency budget is at risk, so requests are not stalled behind long decodes.
- **TBT trade-off:** DeltaServe adds ≈ **+2 ms** average TBT — expected, because more aggressive prefill launches produce larger merged decode batches and thus modestly longer per-step decode time.
- **End-to-end latency:** average **+1.8%** vs. S-LoRA, mainly from longer prefills when FT samples are mixed in. Tail latency is *substantially better* due to SLO-guarded scheduling that prevents starvation.
- **Fine-tuning processed:** **16,834 tokens** in the 1-minute window → **306.1 tokens/s**.
- **GPU utilization:** S-LoRA shows pronounced drops during low-RPS intervals; DeltaServe fills them with FT. **+25.3%** average utilization.

### Burst-dense (column 2)
- The denser pattern amplifies S-LoRA's queuing: its TTFT curve overtakes DeltaServe's much earlier — **roughly 60% of requests miss the TTFT SLO** under S-LoRA.
- **End-to-end latency:** DeltaServe **+3.4%** vs. S-LoRA on average; tails are again better.
- **Fine-tuning processed:** **10,481 tokens** → **183.9 tokens/s** (lower than burst-light because idle windows are shorter).
- **GPU utilization:** **+9.0%** average — smaller gain than burst-light because fewer idle cycles are available to harvest.

### Company X production trace (column 3) — 20 minutes
- The trace differs in *both duration and intensity*: higher peaks, longer bursts, and longer valleys than the synthetic patterns.
- **S-LoRA:** longer high-load periods amplify queuing → larger tails in TTFT and end-to-end latency.
- **DeltaServe:** longer valleys are exploited for FT.
- **Fine-tuning processed:** **267,922 tokens** → **223.5 tokens/s** average.
- **GPU utilization:** DeltaServe reduces utilization valleys present in S-LoRA's curve and raises average utilization by **+22.2%**.

### Single-GPU summary numbers (paper claims)
| Workload | Avg TTFT Δ | Avg TBT Δ | E2E latency Δ | FT tokens (total) | FT tokens/s | GPU util Δ |
|---|---|---|---|---|---|---|
| burst-light | −0.047 s (−40.3%) | +0.002 s | +0.016 s (+1.8%) | 16,834 | 306.1 | +25.3% |
| burst-dense | −0.053 s (−42.7%) | +0.002 s | +0.032 s (+3.4%) | 10,481 | 183.9 | +9.0% |
| Company X | −0.120 s (−61.0%) | +0.001 s | +0.046 s (+4.6%) | 267,922 | 223.5 | +22.2% |

---

## 4.3 Co-serving Results: Multi-GPU (4 × A100)

The baseline reflects **how production systems actually deploy today**: a hard split between inference and FT GPUs. Concretely:
- **Baseline:** 3 A100s run S-LoRA (inference only) + 1 A100 dedicated to fine-tuning. No coordination between the two pools.
- **DeltaServe:** all 4 A100s participate in **both** inference and fine-tuning via coordinated scheduling.

**Figure 4** mirrors Figure 3's layout but adds a row:
- Row 1: arrival pattern.
- Row 2: TTFT + avg TBT.
- Row 3: **end-to-end latency over time** (each point = one request, x-aligned to its arrival).
- Row 4: **cumulative fine-tuning progress over time** (total FT tokens on the 4-GPU cluster).
- Row 5: average GPU utilization across all 4 GPUs.

Because the baseline only allocates **3 of the 4 GPUs to inference**, the same proportional traffic places a heavier load on the baseline's inference pool. As a result, baseline **SLO satisfaction drops from ≈40% (single-GPU setting) to ≈10%** in both synthetic workloads.

### Burst-light
- DeltaServe maintains a flat TTFT distribution below the SLO threshold; baseline shows a steep tail.
- With ample SLO headroom, DeltaServe admits more FT samples.
- **FT throughput: +71.7%** vs. the dedicated 1-GPU baseline FT pool.
- **Avg E2E latency:** **−6.8%** (−0.192 s) vs. baseline.
- **Avg GPU utilization:** **+12.8%**.

### Burst-dense
- Long peaks, minimal slack → DeltaServe **suppresses FT** to preserve TTFT.
- **FT throughput: −68.7%** vs. baseline (the baseline's dedicated FT GPU is unaffected by inference pressure, so it keeps making progress; DeltaServe deliberately yields capacity).
- **Avg E2E latency: −16.4%** (−0.658 s) — a key result: DeltaServe spends the freed compute on inference and dramatically reduces latency.
- **Behavioral observation about latency timeline shape:** Under high request intensity, DeltaServe pulls newly arrived requests into prefill as early as possible. This makes merged decode batches grow quickly during a peak, so the **earliest-arriving** requests in the peak see the highest latency. The baseline admits requests *later*, so the **last-arriving** requests of each burst suffer the most. Net effect: DeltaServe collapses the tail and improves SLO compliance under dense load.
- **Avg GPU utilization: +6.0%**.

### Company X production trace
- DeltaServe maintains stable TTFT; baseline again develops a pronounced tail.
- **Avg E2E latency: +2.1%** (+0.060 s) — a small overhead from opportunistic FT admission, but TTFT compliance is maintained.
- **FT throughput: +34.1%** vs. baseline.
- **Avg GPU utilization: +13.8%**.

### Multi-GPU summary numbers
| Workload | Avg TTFT Δ | Avg TBT Δ | Avg latency Δ | FT tokens Δ | GPU util Δ |
|---|---|---|---|---|---|
| burst-light | −0.198 s (−51.4%) | +0.001 s | −0.192 s (−6.8%) | +71.7% | +12.8% |
| burst-dense | −0.274 s (−61.8%) | −0.001 s | −0.658 s (−16.4%) | −68.7% | +6.0% |
| Company X | −0.107 s (−40.8%) | +0.003 s | +0.060 s (+2.1%) | +34.1% | +13.8% |

---

## 4.4 Ablation Study

Two ablations are presented, in this order: SLO estimator accuracy, then scheduler behavior under fixed-rate load.

### 4.4.1 Offline Profiling — SLO Estimator Accuracy

Goal: validate that the analytical latency model in the SLO estimator is accurate enough for the scheduler to make correct admission decisions.

**Figure 5** (three panels):
- Top (RTX 5090) and middle (A100): predicted vs. measured prefill and decode latency across mixed-batch layouts of the form `(#inf_reqs, #ft_samples)`.
  - Layouts swept: `(2,1)`, `(2,2)`, `(4,4)`, `(10,0)`, `(16,0)`, `(16,2)`, `(17,0)`, `(20,0)`, `(20,4)`, `(30,0)`, `(40,4)`, `(50,0)`, `(60,0)`, `(60,4)`.
  - Each request uses a **50-token prompt**.
  - Solid curves = measured; dashed curves = predicted.
- Bottom: a **30-second slice** from the Company X production trace on the RTX 5090 — predicted vs. actual prefill duration.

**Reported errors:**
| Setting | Prefill error | Decode error |
|---|---|---|
| RTX 5090 (synthetic batch sweep) | **1.66%** | +0.002 s |
| A100 (synthetic batch sweep) | **3.40%** | +0.004 s |
| Company X 30-second slice (RTX 5090) | **avg 4.69%** | — |

**Takeaway in the paper:** the analytical model generalizes across hardware *and* remains accurate under dynamic real workloads, enabling reliable fine-grained scheduling.

### 4.4.2 SLO-Aware Scheduler Behavior under Fixed-Rate Load

**Figure 6** evaluates DeltaServe (co-serving and inference-only) against S-LoRA at **fixed inference rates of 4, 8, and 12 RPS**.
- SLO targets: **TTFT = 100 ms** and **avg-TBT = 30 ms**.
- Left panel: P99 SLO satisfaction (must satisfy *both* TTFT and avg-TBT).
- Right panel: average end-to-end latency, with FT throughput overlaid as a line.

Results across loads:
- **SLO satisfaction:** DeltaServe meets SLOs in *both* modes at all three RPS values; **S-LoRA only complies at 4 RPS**.
- **Latency progression:**
  - **4 RPS:** DeltaServe-Inf's latency is *slightly higher* than S-LoRA's, because the scheduler admits new requests into prefill earlier and creates larger merged decode batches (same effect as in §4.3). With ample SLO headroom, the co-serving mode admits substantial FT work — this is where the gap between the co-serving and inference-only bars is largest, and FT throughput peaks at **228 tokens/s**.
  - **8 RPS:** queuing starts; the SLO-aware scheduler's starvation-avoidance logic now drives DeltaServe's inference latency *below* S-LoRA's.
  - **12 RPS:** rising pressure forces larger decode batches to preserve TTFT, so DeltaServe-Inf's latency again exceeds S-LoRA's. FT throughput drops further; the gap between co-serving and inference-only bars narrows.

**Interpretation in the paper:** the figure illustrates how the scheduler adapts its batching and admission decisions to balance inference responsiveness against fine-tuning progress as load increases.

---

## Headline Numbers Cited in the Conclusion (for cross-checking)
The conclusion summarizes the evaluation as: TTFT and TBT targets met with **negligible end-to-end latency overhead**, **up to 306 tokens/s** of fine-tuning throughput, and average GPU utilization raised by **up to +25.3%**.

---

## Notes for the Re-imagined Evaluation Plan

The following are *implicit* properties of the existing evaluation that may be relevant when revising:

- **Only one model is evaluated** (Llama-7B). No model-size sweep, no MoE, no quantized/larger backbones.
- **Only one fine-tuning task** (Emotion classification, ~20-token inputs). No long-context FT, no instruction-tuning-scale data, no varied LoRA ranks.
- **Backward batch size is fixed at 256 tokens** — no sensitivity study on this knob, even though it directly governs how often heavy backward passes occur.
- **Hardware is limited** to RTX 5090 (single) and 4×A100 40 GB. No H100, no heterogeneous mix, no inter-node multi-GPU.
- **The multi-GPU baseline is a "split-pool" baseline** (3 inference + 1 FT) rather than a co-serving competitor — concurrent systems LLMStation [12] and FlexLLM [25] are discussed in Related Work but **not empirically compared**.
- **SLO targets vary across experiments** (100 ms TTFT in §4.4 vs. visually-set thresholds in §4.2/§4.3 figures, 0.1 s on synthetic, 0.3 s on Company X). A consolidated SLO-sensitivity sweep is absent.
- **No isolation experiments** for individual mechanisms (e.g., paged activations vs. naive activation buffers; pause/resume backward vs. uncoordinated backward; effect of the SLO estimator vs. a static admission policy).
- **No fine-tuning quality / convergence results** — only throughput in tokens/s. Whether the deferred, interrupted backward passes affect convergence speed or final accuracy is not measured.
- **Algorithm overhead is claimed to be < 0.1 ms per iteration** but is not directly measured in a figure; this could be made into an explicit microbenchmark.
- **GPU utilization is reported as a single percentage** but the underlying signal (SM occupancy vs. nvidia-smi util vs. memory bandwidth utilization) is not specified.
