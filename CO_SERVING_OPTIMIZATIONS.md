# Co-serving Optimizations Summary

Living catalogue of optimizations layered on top of the DeltaServe co-serving
infrastructure as ported to vLLM. Grouped by what they target. File paths and
line numbers cite the in-tree implementation; companion docs are
`INTEGRATION_PROGRESS.md` (per-phase progress + verification) and
`VLLM_FORK_CHANGES.md` (every change vs upstream vLLM).

---

## 1. Activation saves — memory-for-compute

What we capture during the FT forward to skip recompute in the backward.

| Save | Shape (Llama-3-8B, s_max=256) | Bytes | What it lets us skip in backward | Where |
|---|---|---|---|---|
| `layer_in[i]` × L | `[s_max, D=4096]` bf16 | 2 MB × 32 = **64 MB** | Recomputing prior layers — backward starts from the residual entering layer i | `deltaserve/accumulate.py` |
| `final_in` | `[s_max, D]` bf16 | **2 MB** | Re-doing the layer-L→pre-final-norm path; needed by `head_backward` | `deltaserve/accumulate.py` |
| `final_hidden` | `[s_max, D]` bf16 | **2 MB** | Re-doing the final-norm forward | `deltaserve/accumulate.py` |
| `concat_input_ids` | `[s_max]` int64 | **2 KB** | Re-emitting CE targets (shift-by-1 per sample) | `deltaserve/accumulate.py` |
| `mlp_gate_up[i]` × L | `[s_max, 2·inter=28672]` bf16 | 14 MB × 32 = **448 MB** | post_ln RMSNorm + gate + up matmuls (the FFN forward) | `deltaserve/accumulate.py` |
| `attn_qh[i]` × L *(F1)* | `[s_max, q_size=4096]` bf16 | 2 MB × 32 = **64 MB** | Q proj + RoPE per layer | `deltaserve/accumulate.py` |
| `attn_kh[i]` × L *(F1)* | `[s_max, kv_size=1024]` bf16 | 0.5 MB × 32 = **16 MB** | K proj + RoPE per layer | `deltaserve/accumulate.py` |
| `attn_vh[i]` × L *(F1)* | `[s_max, kv_size]` bf16 | 0.5 MB × 32 = **16 MB** | V proj per layer (no RoPE on V) | `deltaserve/accumulate.py` |

**Total**: ~612 MB at s_max=256 (~516 MB pre-F1).

What we still recompute per layer in the backward:
- in_ln RMSNorm (cheap; needed for Q/K/V LoRA-A `grad_A = grad_Z.t() @ x_norm1`)
- Attention forward (scores/softmax/`att @ v`) — the meaty per-sample op
- O projection (base + LoRA)
- Residual add (`resid_mid = layer_in + o`)

What we never compute in the backward:
- post_ln RMSNorm + gate/up matmuls (saved as `mlp_gate_up`)
- Q/K/V projections + RoPE (saved as `attn_qh/kh/vh`)
- The down matmul / layer output (chain-rule gradient from next layer feeds in)

---

## 2. CUDA graphs for the backward — dispatch-overhead elimination

Three regions captured at child startup, all `[s_max, ...]` shape-stable.
Per-layer eager fallback on capture/replay failure or padded-attn budget
overflow. Gradient values bit-identical to eager (gradcheck 111/111 +
12/12 in `tests/test_llama3_backward_graph.py` and
`tests/test_llama3_backward.py`).

| Graph | Count | What it captures | Where |
|---|---|---|---|
| Forward recompute (per layer) | L=32 | RMSNorm in_ln + (Q/K/V/RoPE or saved-qkv read) + padded-attention forward + O proj + residual. Writes outputs directly into the inputs Graph A/B already read. | `bwd_services/llama3_graph.py` `_forward_core` |
| FFN backward (per layer) | L=32 | `silu/sigmoid + down/gate/up GEMMs + post_ln rmsnorm-bwd + residual add`. Per-layer because down/gate/up weights vary. | `bwd_services/llama3_graph.py` `_capture_ffn` |
| Padded attention backward (shared) | **1** | Per-sample GQA scores/softmax/dQ/dK/dV core. Single graph reused across all L layers — no per-layer weights in the core. ~31× less startup vs DeltaServe's per-layer attn captures. | `bwd_services/llama3_graph.py` `_padded_attn_core` |

Pre-captured at child startup in `prepare()` against zero-initialized static
buffers (capture only depends on shapes/addresses, not values — replay-time
staging produces correct gradients).

Key invariants:
- **Static IO outside the graph pool** — `torch.zeros(...)` allocated in the
  default caching allocator before `graph_pool_handle()`. Load-bearing for
  avoiding pool-aliasing NaN on persistent LoRA grads.
- **`_maybe_pause` between Graph A and Graph B** — once per layer
  (relocated from the eager path's at-layer-start; `mp.Event.wait` can't
  run inside a captured region).
- **`save_attn_qkv` mode** — when on, the captured forward graph reads from
  `static_saved_qh/kh/vh` and skips Q/K/V proj + RoPE entirely.

---

## 3. Defer LM-head compute to the backward

vLLM V1 only materializes logits for sampled positions, so the inference
forward gives us nothing useful for FT loss.

- Save **pre-LM-head hidden states** (`final_hidden`) + input ids in the
  forward (single bf16 write per token, negligible cost).
- Reconstruct full logits inside the backward via `final_hidden @ lm_w.T`
  (fp32; LM-head precision is load-bearing).
- **Chunked over vocab** (`_VOCAB_CHUNK=16384` in
  `bwd_services/llama3.py:_logits_chunked`) so the fp32 LM-head temporary
  is bounded — important on 128K-vocab Llama-3.
- Per-sample shift-by-1 CE; gradient = `softmax − one-hot`, normalized over
  valid tokens; matches DeltaServe's `_logit_backward` math.

---

## 4. FT admission rules

Two mutually exclusive strategies + a global SLO gate underneath both.

### Per-step admission strategies (mutually exclusive)

- **`ft_tokens_admission_constrain_factor`** (proportional cap, default `-1` = off):
  when `> 0` and the step carries prefill, `ft_tokens ≤ inference_prefill_tokens · factor`
  (a `min` on top of the SLO + buffer caps). Direct knob; no effect on prefill-free
  steps (idle / FT-only fills).
- **`match_with_prefill_workload`** (leaky-bucket, default `False`): scheduler
  maintains `_unspent_prefill`. On every prefill-carrying step:
  - if `_unspent_prefill + t_in ≥ peek_next_ft_sample.input_len` → admit that
    one FT sample, reset counter to 0.
  - else → don't admit, accumulate `_unspent_prefill += t_in`.
  Any successful FT admission resets the counter (atomic, no double-spend).

### SLO gate (always on under both)

- 6-param merged step-time estimator (`bwd_services/estimator.py`):
  ```
  T_step ≈ α·S + β·T_in + γ·T_ft + δ·B_d + ε·K + c
  ```
  where `S ≈ T_in² / P` is a prefill-quadratic proxy. Eager + graph regimes
  tracked separately.
- Admission picks
  `x_ft = min(max_next_ft_tokens(ttft_budget), max_next_ft_tokens(max_tbt_budget), coord cap)`.
- **Offline profiling pass at launch** (`profiling_batch_generator.py`,
  `EngineCore.profile_execution_model`) — shape sweeps run through the live
  scheduler before serving begins, seeds estimator.
- **Online refit every 256 steps** with predicted-vs-actual CSV.
- **Deferred CUDA-event timing ring** (RING=4 steps) — async-pipelining safe.

---

## 5. Backward-side compute optimizations

Beyond CUDA graphs, the per-cycle work the backward does:

- **Fused AdamW** (`bwd_services/llama3.py:_build_state`,
  `torch.optim.AdamW(..., fused=True)`) — single CUDA kernel for all 256
  LoRA tensors (8 per layer × 32 layers) instead of per-tensor dispatch.
- **Persistent `grad_qh/kh/vh` buffers** at s_max
  (`bwd_services/llama3.py`) — saves 96 zero-fill kernel launches per
  backward (32 layers × 3 tensors). Passed via
  `attn_backward_core(..., grad_qh_buf=...)`.
- **bf16 bulk default** (`backward_fp32: false`) for FFN-bwd / LoRA-grad /
  rope-proj-bwd. fp32 stays load-bearing for scores/softmax, RMSNorm
  internals, LM head, LoRA master. `backward_fp32: true` for strictly
  more accurate grads at ~2× backward matmul cost.
- **Sync coalescing** (`bwd_services/base.py:_handle_process_activations`) —
  2 `torch.cuda.synchronize()` per cycle → 1. Backward timing via CUDA
  events (`enable_timing=True`) instead of wall-clock-after-sync. The
  remaining sync covers both the backward (so events become queryable)
  AND the buffer zero-fill.
- **Per-cycle backward log restructured** to one line:
  `[backward] Xms (graph/eager) loss=… total_trained=… n=… epoch=… N/total tokens`.

---

## 6. Co-serving GPU sharing primitives

- **`_maybe_pause` prefill-gated GPU yield** (`bwd_services/base.py`,
  `bwd_services/llama3.py`) — backward yields the GPU when any forward
  carries prefill tokens (TTFT-bounding), runs concurrently with
  decode-only steps. Called at every layer boundary in the backward
  (load-bearing co-serving contract).
- **Fire-and-forget pause/resume** — `mp.Event` toggles only, no blocking
  `event.synchronize()` between pause and resume. The pause-resume
  cross-process visibility wait is scoped to `coord.capture_done_evt`
  (set by the runner after activation copies) rather than a full device
  sync.
- **MPS partition** for the backward child
  (`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`) — gives the backward a constrained
  slice so inference keeps the rest of the GPU.

---

## 7. FT-injection hot-path optimizations

- **Slice-based activation save** (Phase 6.1,
  `deltaserve/accumulate.py:_make_pre_hook`) — per-layer hooks use
  `val[start:start+n]` (a view, no CUDA kernel, no allocation) on the
  contiguous fast path instead of `val[mask]` (index_select kernel +
  gather allocation). Silent fallback to mask when FT positions are
  interleaved with non-FT (rare — happens when an FT request lands in a
  freed inference slot mid-batch). Saves ~33 CUDA kernels per FT forward
  on Llama-3.
- **Eager FT-step gate** (`v1/worker/gpu_model_runner.py`,
  `_build_finetune_mask` + `force_eager=self._ft_has`) — any batch carrying
  FT tokens runs eager (`skip_compiled`). Foundational invariant that
  lets activation-saving hooks fire correctly without pool-aliasing NaNs.

---

## 8. Async scheduling safety (Phase 4b)

- **`async_scheduling: true` by default** under co-serving
  (`config/vllm.py.__post_init__`; was force-off). Allows batch-queue
  pipelining (`schedule(N+1)` before `record_capture(N)`) under uniproc
  (`max_concurrent_batches=2`).
- **Reserve-at-inject** (`deltaserve/coordinator.py`) — `coord.reserve(n)`
  returns a disjoint per-step write offset (`committed + reserved`);
  `space_remaining = capacity − committed − reserved`. Two in-flight
  steps can never overlap their buffer writes; admission can't overflow.
- **Per-step duration stashed on `scheduler_output`** rather than a
  coordinator slot (which the pipeline would clobber).
- **Backward triggers gated on `reserved==0`** so buffer-full / epoch-flush
  fires only when no save is in flight — race-free.

---

## 9. Buffer / admission lifecycle

- **`per_step_budget = capacity`** (`gpu_worker.py`, was `capacity / 2`) —
  an idle / FT-only step can fill the buffer in one eager forward instead
  of ~4.
- **Peek-next epoch-flush trigger** (`deltaserve/coordinator.py:note_injection`) —
  raises a flush flag when the buffer can't grow (epoch drained OR next
  sample won't fit). Trains the partial buffer instead of wedging at
  e.g. 208/256 forever.
- **Oversized-sample drop at load** (`deltaserve/finetuning_store.py:load`) —
  samples with `input_len > max_saved_finetuning_tokens` are dropped with
  a warning during load. Prevents the FT-admission deadlock that would
  otherwise occur when only oversized samples remain.

---

## 10. Inference pre-emption of FT-only stepping (Phase 6 — `forward_interruptible`)

All behind one config flag (`finetune.forward_interruptible`, default
`False` → bit-identical behaviour when off). Three tiers catching late
HTTP arrivals at progressively later windows:

- **Tier A — pre-schedule grace poll** (`v1/engine/core.py`): when the FT
  scheduler reports `would_step_be_ft_only()`, the engine main loop does
  a bounded blocking poll on `input_queue` (default 2 ms,
  `ft_only_admission_grace_ms`) before letting `schedule()` commit.
- **Tier B — post-schedule rollback** (`deltaserve/ft_scheduler.py:_rollback_ft_step`):
  if `schedule()` produced FT-only AND an arrival landed since, undo
  all FT-side state (`_free_blocks`, `coord.release_reserve`,
  `store.release_claimed`, `coord.restore_admission`) and re-schedule
  once.
- **Tier C — mid-forward abort** (`deltaserve/accumulate.py` hook check):
  input-socket thread sets `coord.ft_abort_event` on each ADD while an
  FT-only batch is in flight; hooks raise `FTAborted` mid-forward;
  runner zeros the partial-write tail (`accumulator.zero_offset_range`)
  and returns an empty `ModelRunnerOutput(_ft_aborted=True)`.

Required the **3-phase store API** (`deltaserve/finetuning_store.py`):
`claim(samples)` at admit → `commit_claimed(samples)` at backward-done →
`release_claimed(samples)` at rollback. Replaces the one-way
`confirmed_trained=True` and **fixes a pre-existing bookkeeping bug**
where samples were marked trained at admit time, before any backward
had actually processed them.

---

## 11. Control plane + frontend stall fix (Phase 4c/4d)

The largest TTFT-under-co-serving wins actually came from here — not from
anything in the engine:

- **`/start_finetuning` POST endpoint** (`entrypoints/serve/finetune/api_router.py`) —
  FT admission gated off at launch (`finetune.start_on_launch: false`)
  until POST'd. Lets profiling + warmup run with zero FT. Eval harness
  POSTs after replay starts.
- **`disable_log_stats` auto-defaults ON when `enable_finetuning`**
  (`engine/arg_utils.py:create_engine_config`) — **this was the actual
  TTFT spike fix.** vLLM attaches per-step `scheduler_stats` to the
  rank-0 frontend's output stream, saturating that one asyncio loop and
  stalling HTTP accept + SSE streaming during decode bursts (~80 msg/s).
  Auto-defaulting `disable_log_stats=True` kills the per-step stats
  stream; the SLO estimator uses its own engine-side CUDA timing, so
  it's unaffected.
- **`--api-server-count N` plumbing** (`eval/auto_benchmark.py` flag +
  YAML `server.api_server_count`) — 1 shared EngineCore + N frontends
  behind a shared socket, shards frontend output processing.

---

## 12. Cross-process IPC efficiency

- **CUDA-IPC zero-copy** for: base model weights, FT adapter (fp32),
  per-layer activation buffers, served LoRA stacked buffers. The trainer
  writes inference's served weights directly — no copy, no read-modify-write
  race (FT admission is closed for the entire backward cycle).
- **One-shot `set_corpus_meta` IPC** (Phase 5.3) — replaces per-
  `notify_buffer_full` corpus-total send. Single IPC at startup after
  `FinetuningStore.load()`.
- **`torch.multiprocessing`** spawn context (registers CUDA-IPC reductions)
  — required for sharing CUDA tensors across the spawned backward child.

---

## 13. Served-LoRA hot-publish (Phase 3.4)

- After each `optimizer.step()`, the trained fp32 master is written into
  vLLM's served LoRA stacked buffers (`bwd_services/llama3.py:_publish_to_served`)
  — the exact tensors inference reads via punica kernels.
- Clamp(±6.5e4) + cast to served dtype; `B * scaling` (vLLM punica
  hardcodes `scale=1`, not `α/r` — we bake the scaling into B at publish
  time).
- Safe with no locking: FT admission is closed for the whole backward,
  so the adapter is idle until the done-reply reopens it.
- The FT adapter is pre-loaded into a **stable served slot** at startup
  (`gpu_worker._maybe_share_ft_served_lora`) — no inference request can
  land on this slot.

---

## 14. Eval tooling

- `eval/auto_benchmark.py` replays request timelines from
  `eval/timelines/5090/`, streams `/v1/completions`. Outputs tagged
  `_factor_<X>` (or `_factor_off` for `-1`) so A/B runs don't overwrite.
- `eval/auto_plot.py` — 5-panel layout (timeline / E2E latency /
  throughput bands / TTFT-SLO satisfaction / **E2E latency percentile
  with p99 highlighted**). `--factor X` arg with smallest-factor
  autodetect; factor appears in plot title.
- `eval/pure_ft_bench.py` — pure-FT (no inference traffic) benchmark.
  Launches the server, POSTs `/start_finetuning`, idles for `--duration`,
  trims + summarizes the bwd_log. Isolated backward-throughput
  measurement.

---

## Layered story

The optimizations work as layers — none individually delivers the current
behaviour:

1. **Activation saves** (memory-for-compute) shrink what the backward needs
   to recompute.
2. **CUDA graphs** eliminate dispatch overhead in the remaining backward
   compute.
3. **Backward-side compute** (fused AdamW, persistent buffers, sync
   coalescing, bf16 bulk, chunked LM head) trims per-cycle work.
4. **GPU sharing primitives** (`_maybe_pause`, MPS, fire-and-forget)
   bound inference latency during the backward.
5. **Async + reserve-at-inject** lets the inference batch queue pipeline
   through co-serving.
6. **SLO-aware admission** + the two strategy knobs decide how much FT
   to admit per step.
7. **`forward_interruptible`** catches late inference arrivals at three
   points to pre-empt FT-only steps.
8. **Control plane + frontend fixes** (`disable_log_stats`,
   `api-server-count`, `/start_finetuning`) — where the largest
   TTFT-under-co-serving wins actually came from.

## Pointers

- Per-phase progress + verification: `INTEGRATION_PROGRESS.md`
- Every change vs upstream vLLM: `VLLM_FORK_CHANGES.md`
- Project context, design constraints, precision rules: `CLAUDE.md`
- DeltaServe reference architecture (read-only): `DeltaServe/CLAUDE.md`
