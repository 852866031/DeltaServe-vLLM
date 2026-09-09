# Weekly report — week of September 1

DeltaServe on vLLM: co-serving LoRA finetuning next to inference on two RTX 5090s
(tensor parallelism across the two GPUs). This week closed the remaining gaps in the
backward pass under tensor parallelism, cut the backward's own cost by about a third,
made inference pre-emption of finetuning work with two GPUs, and validated the whole
stack on both model families (Llama-3-8B and Qwen3-14B) against recorded request traces.

## 1. CUDA graphs for the backward pass, now with tensor parallelism

**Background.** The finetuning backward pass runs in a separate process on each GPU. It
rematerializes each transformer layer's forward from saved activations, then computes the
LoRA gradients by hand. Before this week the backward could run as a set of CUDA graphs on
a single GPU (a graph replays a pre-recorded sequence of kernels with one launch, removing
per-kernel dispatch cost), but with two GPUs it was forced back to eager execution.

**Why two GPUs were harder.** Under tensor parallelism each GPU holds half of every
attention head and half of the feed-forward block, and the two halves have to be added
together at several points in every layer with an all-reduce over NCCL. A collective inside
a captured graph is possible but dangerous here: the backward must yield the GPU to
inference at every layer boundary (a host-side wait that cannot be captured), and a
one-sided abort with a collective in flight would deadlock both GPUs. So the rule became
**no collective is ever inside a graph**.

**What we did.** Six of the seven all-reduces per layer already sat in eager code between
the captured regions. The seventh, the output-projection sum of the forward
rematerialization, sat inside the forward graph. We split that graph at the output
projection: the captured part ends at the partial sum, the all-reduce runs eagerly, and a
short eager tail finishes the layer. On a single GPU the tail is captured together with the
core, so single-GPU behaviour is bit-identical to before. The graphed backward also had
to be taught the six backward-side reductions; they were missing, so enabling graphs under
TP without that would have trained on un-reduced gradients.

**Structure.** Three captured regions per layer: the forward rematerialization, the
feed-forward backward, and one shared padded-attention backward used by every layer.
Llama-3-8B: 65 graphs, captured once at startup in about 0.3 s. The GPU-yield point stays
between the two backward replays of each layer, once per layer, unchanged from the eager
path.

**Verification.** Two new tests run on both GPUs with real NCCL: a layer-level parity test
(graph equals eager on each GPU to 1e-5, equals the single-GPU reference, identical across
GPUs, including the fallback path when a batch overflows the padded attention budget;
4020 checks) and a trainer-level test that runs the real training loop for several steps
and compares losses and adapter weights across graph, eager, and single-GPU (220 checks).

## 2. What the backward actually costs, and what we removed

**Cross-GPU traffic.** Per layer the backward needed 7 all-reduces: 1 in the forward
rematerialization and 6 in the backward (two residual-stream gradients, four replicated
LoRA-factor gradients). On Qwen3-14B (40 layers) that is 280 collectives per cycle, about
205 MB per GPU. This box has no peer-to-peer link between the GPUs, so every collective
crosses PCIe through host memory: measured 0.36 ms per 2.5 MB reduce and 0.045 ms per
160 KB reduce.

**Activation saving instead of recompute.** The backward used to recompute most of each
layer's forward from the saved layer input. We now save three more per-layer tensors during
the inference forward, each behind a config flag:

| saved tensor | what the backward no longer recomputes | memory (Qwen3-14B, 256 tokens) |
|---|---|---|
| post-RoPE q/k/v (Llama-3) or pre-norm q/k + v (Qwen3) | the Q/K/V projection GEMM and RoPE | ~48 MB per GPU |
| attention context | the attention forward (scores, softmax, AV) | ~16 MB per GPU |
| post-attention residual | the output-projection GEMM, the residual add, **and the forward's only all-reduce** | ~100 MB |

With all three on, the per-layer forward recompute is a single RMSNorm, and the
cross-GPU traffic drops to 6 collectives per layer, all in the backward. Qwen3 needed its
own variant: its per-head q/k normalisation sits between the projection and RoPE, so the
values captured after attention are not what the norm's backward needs. Capturing q and k
at the input of the norm modules instead makes the shortcut exact for Qwen3 too.

**A kernel-level profile changed the priorities.** Profiling one real backward cycle on a
Qwen3-14B-shaped model (two GPUs, no inference running) showed where the ~165 ms went:

| component | ms | note |
|---|---|---|
| LM head | 63 | per-sample GEMMs with ~31 rows each, and the bf16 head converted to fp32 nine times per cycle |
| NCCL all-reduces | 37 | 240 collectives |
| feed-forward and attention backward GEMMs | 22 | |
| elementwise / reductions | 17 | |
| copies | 10 | staging saved activations |

The LM head was the largest single item and mostly waste: the same fp32 result is produced
by batching all samples' rows into one GEMM per vocabulary chunk and converting each chunk
once per pass. Same math, same fp32 precision (verified to 1e-7 against the old
implementation), and the cycle dropped from 165 ms to 123 ms. Together with graphs and the
saves, the uncontended Qwen3-14B cycle went from ~194 ms to ~121 ms this week.

## 3. GPU-to-GPU transfer: bucketing and a communication stream

**Before.** All 240 reductions per cycle were issued inline, on the same CUDA stream as
the compute, at the point where each gradient was produced. 160 of them were the
replicated LoRA-factor gradients, which nobody reads until the optimizer step at the end
of the cycle. Each reduce is a rendezvous between the two GPUs, and the backward process
was also started with the CUDA setting that pins every stream onto a single hardware queue
(inherited from the original DeltaServe, where it served MPS ordering), so no overlap
between communication and compute was possible even in principle.

There was also a correctness problem hiding in the same place: gradient clipping ran
per layer on each GPU over the tensors that GPU held, replicated factors in full but only
its own half of the sharded ones. The two GPUs derived slightly different clip scales and
applied them to gradients that are supposed to be identical, so their replicated adapter
weights drifted apart whenever a layer's gradient norm exceeded the clip threshold.

**After.** The factor gradients are copied into one persistent flat buffer as they are
produced and reduced once per group of eight layers on a dedicated communication stream,
overlapping the next group's compute; 240 collectives become 86 (80 residual reduces, 5
buckets, 1 for clipping). Clipping runs after the bucketed reduce with a norm that sums
the sharded halves across GPUs in one tiny collective, so both GPUs scale identically. The
single-queue setting is gone.

**Ordering correctness** lives in one helper: every submit waits for the producing
compute, every consumer waits for the completion event, only persistent buffers are
submitted. It is tested by running the same training with a synchronous mode (wait after
every reduce) and a delay-injection mode (the communication stream spins before each
collective, so any missing wait reads stale data every time) and requiring bit-identical
results, plus a CPU test where the clip fires and the two-GPU run must match the
single-GPU run. The measured time saving is small, about 2 ms per cycle, because the 80
residual reduces stay on the critical path; that is the floor for tensor parallelism on
this hardware. The real gain is the correctness fix.

## 4. Inference pre-empting finetuning-only steps

**What it is.** When no inference request is waiting, the scheduler builds steps made only
of finetuning samples. An inference request that arrives during such a step would
otherwise wait for it to finish. Pre-emption lets the request cut in at three points: a
short wait on the input queue before an FT-only step is scheduled, a rollback if a
request landed between scheduling and dispatch, and an abort in the middle of the forward.
The mid-forward abort is an event that the engine's input thread sets on arrival and that
the per-layer activation hooks check; the hook raises, the runner returns an empty result,
and the scheduler restores its bookkeeping so the samples are retried later.

**Single GPU.** Everything runs in one process, so a shared in-memory event is enough.

**Two GPUs: the challenges.** The GPU workers are separate processes from the engine, so
the event never reached them; the feature was silently inert. Worse, even with a signal,
each GPU checking it independently would be wrong: an FT-only forward runs a collective
every layer, so if one GPU leaves at layer k while the other continues, the other blocks
forever in the next collective. The abort has to be a joint decision. Further, the
"aborted" marker on the result was a dynamic attribute that did not survive the
worker-to-engine hop, and the engine's async pipeline calls a second sampling entry point
that also had to know about the abort.

**What we built.** The engine publishes an arrival counter in POSIX shared memory,
created before the workers are spawned. Each worker compares the counter against its
value at forward start, once at entry and once per layer boundary, and MAX-all-reduces
its local bit over the TP group's CPU (gloo) group before acting, so both GPUs always take
the same decision at the same layer; one small CPU collective per layer, no GPU sync. The
first live run exposed a mistake: the hooks are armed for every batch that carries
finetuning samples, including mixed batches with inference tokens, and those were being
aborted too, taking their inference requests' step with them. The decision is now active
only for FT-only forwards. Two secondary findings were also fixed: the CPU launches layers
far ahead of the GPU, so a hook-time abort only saved the un-launched tail (the poller now
bounds the launch-ahead to two layers), and, most importantly, most burst-start requests
were slow for a different reason entirely.

**The pause finding.** On the dense trace, the first request of each burst usually
arrived while the backward was running, not during an FT-only forward. Its prefill took
80 to 200 ms instead of 40 ms because the "yield the GPU to prefill" grant was re-set right
after the prefill was enqueued, not when it finished, so without MPS the backward resumed
immediately and time-sliced against the prefill. The runner now records a CUDA event
behind the prefill and resumes the backward only when that event has completed.

**Result on the dense trace (Qwen3-14B, 8 bursts of 60 requests):**

| run | TTFT satisfaction | p99 | max | first request of each burst (ms) | FT tok/s |
|---|---|---|---|---|---|
| inference only | 100 % | 86 ms | 113 ms | 40 41 40 39 38 40 38 41 | — |
| co-serving, before | 99.2 % | 377 ms | 577 ms | 92 52 144 53 44 40 458 58 | 277 |
| co-serving, after | 100 % | 91 ms | 150 ms | 55 60 67 61 61 60 59 64 | 262 |

## 5. Validation and a discovery about MPS

Both families now co-serve the loose, dense, and Nutanix traces with the full stack:

| | TTFT satisfaction (loose / dense / Nutanix) | FT tok/s |
|---|---|---|
| Qwen3-14B TP=2 | 98.3 / 100 / 99.4 % (1 Sept: 95.0 / — / 98.1) | 642 / 262 / 392 (was 467 / — / 239) |
| Llama-3-8B TP=2 | 99.2 / 96.9 / 98.5 % (SLO 0.25 s) | 1088 / 641 / 814 |

Also verified: the Llama-3 RoPE base-frequency fix from last week (the backward's
rematerialization now matches the served model to bf16 noise and the loss no longer
stalls), and the Qwen3-0.6B single-GPU path, which needed a fix for models whose LM head
is tied to the embedding.

One finding worth knowing: the backward process has always been launched with an MPS
thread-percentage variable, but no MPS control daemon runs on this machine, so the
variable was inert and the backward was simply time-sliced against inference by the
driver. We made "no MPS" the explicit default and rely on the pause contract instead. The
consequence is that a backward cycle under inference load takes roughly twice its
uncontended time; that trade-off is now understood rather than assumed.

## 6. Next

The SLO estimator's coefficients are fitted with no backward running; with the pause now
holding through the whole prefill, the contention pattern has changed. The next step is a
two-GPU versus single-GPU residual comparison of the estimator to see whether its
step-time model needs a tensor-parallel term.
