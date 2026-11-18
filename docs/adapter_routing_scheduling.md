# Adapter routing under one scheduler

How DeltaServe serves a request whose LoRA adapter is chosen by the system
("adapter routing"), and how every piece of extra work this needs is decided by the
SLO-aware scheduler at step granularity.

This is a design walkthrough. Timings marked *measured* come from Qwen3-14B TP=2 on
2× RTX 5090; the scheduling behaviour describes the target design, parts of which are
still being implemented. Figure sources (`.drawio`) are in
[`figures/scheduling/`](figures/scheduling/).

## 1. Request path: what adapter routing costs

![Request path](figures/scheduling/01-request-path.png)

A **named** request already says which adapter it wants. A **routed** request leaves
the choice to the system, which adds two steps between entering the engine and the
prefill:

- **Encoding** — the prompt is embedded by a small encoder (e5-large-v2) on the GPU.
  Measured ≈ 3.5 ms + 0.014 ms × total tokens, 4–17 ms for typical batches.
- **Pick adapter** — the embedding is compared against the **indexed adapter pool**
  (one routing entry per adapter). No match, or an encode failure, falls back to the
  base model.

Everything else is shared. The API server tokenizes (for a routed request it also
runs the encoder's tokenizer) and forwards at once; the arrival time is stamped when
the request enters the engine, so the time spent choosing the adapter counts against
TTFT. The chosen adapter is reported in the response headers.

**Load adapter** applies to both paths and depends on where the adapter is: a GPU
slot (0 ms), host memory (≈ 4 ms, inside the step), or disk (the request is held one
step while the adapter is read in the background).

## 2. DeltaServe before adapter routing

![DeltaServe overview](figures/scheduling/02-deltaserve-overview.png)

DeltaServe co-serves LoRA finetuning with inference on the same GPUs. FT samples ride
inside the inference batch; their activations are saved during the forward; a
backward process trains the adapter from them and pauses whenever inference needs
the GPU. The SLO-aware scheduler admits FT samples only while TTFT / TBT stay within
the SLO.

Here the scheduler has two things to decide: what goes into the forward batch, and
when the backward may run.

## 3. Adapter lifecycle: from training to being routed to

![Adapter lifecycle](figures/scheduling/03-adapter-life.png)

Adapter routing closes a loop. A finetuning job's samples go through the forward
batch (1), their activations are saved (2), and the backward trains the adapter (3).
When the job finishes, the **adapter handover** saves and publishes it (4);
**indexing** computes its routing entry (5) and adds it to the adapter pool. From
then on it is a candidate for routed requests (6), which are served with it (7).

The loop introduces three kinds of work that did not exist in section 2: encoding,
indexing, and the adapter handover.

## 4. DeltaServe with adapter routing

![Scheduling overview](figures/scheduling/04-scheduling-overview.png)

The same picture with the loop added. Inputs are on the left (inference requests,
the adapter pool, finetuning jobs), responses on the right, and everything in between
is work the scheduler decides. Orange marks what is new compared with section 2:
the adapter pool, encoding, indexing and the adapter handover.

The principle: **anything that occupies the GPU or the engine loop is decided by the
scheduler.** Nothing takes GPU time on its own initiative.

## 5. Schedulable objects

![Schedulable objects](figures/scheduling/05-schedulable-objects.png)

Reading the work off the overview gives five schedulable objects:

| | Object | What it is | When it may run |
|---|---|---|---|
| ① | Encoding | embed routed prompts, pick adapters | its own step, before the batch; capped by total tokens |
| ② | Forward engine batch | decode + prefill + FT samples; adapter load at step start | every step |
| ③ | Backward batch | one backward pass of the adapter being trained | any time except encode steps and steps with inference prefill |
| ④ | Indexing | compute a new adapter's routing entry | on the GPU only when no inference request is present; otherwise on the CPU |
| ⑤ | Adapter handover | save, publish, switch to the next job | off the engine loop; FT admission closes when it starts |

## 6. Where everything runs

![GPU layout](figures/scheduling/06-gpu-layout.png)

With *n* GPUs under tensor parallelism, each GPU holds one worker and its backward
child. The encoder lives in the worker of **one** rank, so an encode is an ordinary
worker call and can never overlap a model step. Indexing is a separate process that
shares the encoder's weights with that worker when it runs on the GPU, and has a
CPU process to fall back to. Adapters sit in one of three places — GPU slot, host
memory, disk — which is what the load cost in section 1 depends on.

## 7. One scheduling step

![One scheduling step](figures/scheduling/07-one-scheduling-step.png)

A step, in order: encode the routed requests that are waiting, pick their adapters,
load what the batch needs, run the forward batch, then let the backward use what is
left. The backward is paused only for the two step kinds that are latency-critical
for a new request — an encode step and a step containing inference prefill — and
resumes as soon as the prefill completes. A decode-only step does not pause it.

Two more decisions sit in the scheduler but are not part of building the batch, and
are taken only when the state changes: whether indexing may use the GPU, and when to
start an adapter handover. Indexing and the handover then run on their own, outside
the engine loop.

## 8. Scheduler decisions

![Scheduler decisions](figures/scheduling/08-scheduler-decisions.png)

The decisions in the order the scheduler takes them, with the batch contents shown
alongside as they grow:

1. **Adapter routing** — routed requests in the queue get an encode step and an
   adapter; named requests skip it.
2. **Adapter location** — on disk: the request goes back to the queue for one step
   while the adapter is read in the background (if there is no other work, it is
   scheduled directly and loaded in-step instead). In host memory: load into a GPU
   slot at step start. In a GPU slot: nothing to do.
3. **Backward pause / resume** — pause for an encode step or a prefill step, resume
   otherwise.
4. **FT admission** — add FT samples only while the predicted TTFT / TBT stay within
   the SLO; closed while an adapter handover is in progress.
5. **Indexing** — allowed on the GPU only when no inference request is present; the
   scheduler notifies the indexing process only when this changes.
6. **Adapter handover** — when a job finishes, close FT admission and start the
   handover off the engine loop.

## 9. Inference first

![Inference first](figures/scheduling/09-inference-first.png)

Every object other than the forward batch yields to inference. The backward yields
at each of its boundaries, indexing yields between encoder layers and moves to the
CPU, and the handover does its GPU part (copying weights to host memory) in the
backward child and everything else on the CPU. The estimator is told what adapter
routing cost: the encode time is already inside TTFT because the arrival is stamped
before it, it is added explicitly to TBT, and the predicted adapter load time is
added to the step time.

## 10. Adapter pool

![Adapter pool](figures/scheduling/10-adapter-pool.png)

The pool tracks each adapter's routing entry and where the adapter currently is.
Load cost by location (*measured*):

| Location | Cost | How it is scheduled |
|---|---|---|
| GPU slot | 0 ms | — |
| host memory | ≈ 4 ms | loaded at step start |
| disk, other work pending | ≈ 12 ms read in the background, then ≈ 6 ms | request held one step |
| disk, nothing else to do | ≈ 18 ms | loaded in-step |

GPU slots follow vLLM's own LRU; the pool mirrors it and pins the adapters the
current batch needs rather than choosing victims itself. Adapters in use stay in host
memory, so the disk path is taken only on a cold start.
