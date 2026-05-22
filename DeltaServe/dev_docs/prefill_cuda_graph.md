# Prefill CUDA Graph

CUDA Graph optimization for the **prefill** stage of inference, extending the
existing decode CUDA graph. Captures the full prefill forward pass (embeddings,
32 transformer layers with LoRA, attention, FFN, and last-token logit
projection) and replays it for subsequent requests that fall into the same
token-length bucket.

Branch: `feature/cuda-graph-prefill`
Related: `docs/cuda_graph_report.md` (decode CUDA graph)

---

## Scope (MVP)

Only activates when **all** of the following hold:

- `--enable-cuda-graph` is passed
- `batch_size == 1` (single request per prefill call)
- No fine-tuning (no `finetune_mask`, no `ref_mask`)
- No interruption event (`prefill_interrupt_event is None`)

Otherwise the regular `_context_forward` path is used. Batch size > 1 and
fine-tuning mode are intentionally excluded from this MVP because they require
handling per-request Python loops and CPU-side synchronization points that
would need additional work to make graph-safe.

---

## How it works

### Cache key

```
key = (batch_size, total_token_num_bucket)
total_token_num_bucket = ceil(total_token_num, 128)
```

With `batch_size == 1` only, the practical cache shape is
`(1, 128), (1, 256), (1, 384), ...`. Each unique bucket allocates its own
static buffers and captures its own graph the first time it is hit.

### Static buffers (per bucket)

During `prefill_capture`, the runner allocates T_bucket-sized static tensors
for every dynamic-shape tensor in the forward pass, clones them into the
engine state, runs one warmup iteration on a side stream, captures the graph,
and replays it once so the CAPTURE caller receives valid output:

| Buffer                      | Shape                                  | Purpose                                       |
|-----------------------------|----------------------------------------|-----------------------------------------------|
| `input_ids`                 | `[T_bucket]` int64                     | Padded token ids (padding = 0)                |
| `position_cos`, `position_sin` | `[T_bucket, head_dim]` fp16         | Padded rotary embedding lookups               |
| `batch_req_bins`            | `[T_bucket]` long                      | LoRA adapter id per token (pinned to engine)  |
| `prefill_mem_index_key/value` | `[T_bucket]` long                    | KV cache slot indices (dedicated pad slots)   |
| `prefill_key_buffer`, `prefill_value_buffer` | `[T_bucket, kv_heads, head_dim]` fp16 | Per-token K/V scratch before pool write |
| `delta[0..2]`               | `[T_bucket, max_lora_dim]` fp16        | LoRA shrink/expand scratch                    |
| `att_m_buffers[layer]`      | `[tp_q_head_num, max_total_tokens]` fp16 | Attention score buffer                      |
| `finetune_mask`             | `[T_bucket]` bool                      | All zeros (inference-only guard)              |
| `b_start_loc`, `b_seq_len`  | `[batch_size]`                         | Cloned as-is (small)                          |
| `b_loc_key`, `b_loc_value`  | same as original                       | Cloned for stable addresses                   |

### Padding the KV cache

A subtle correctness issue: prefill writes K/V to the KV cache pool at
positions listed in `prefill_mem_index_key/value`. If padding tokens reused
real token slots, padding K/V would overwrite real K/V, corrupting subsequent
decode reads.

Fix: when `use_prefill_cg` is true, `_prefill` allocates **T_bucket** KV slots
(not `total_token_num`). The first `total_token_num` slots hold real K/V; the
remaining `T_bucket - total_token_num` slots are dedicated padding-scratch
slots that get overwritten with padding K/V but are never read (attention is
gated by `b_seq_len`).

### Graph-friendly post-infer

The production `post_infer.token_forward_with_finetune_outputs` contains a
per-request Python loop and `.item()` calls — these break graph capture. For
inference-only prefill we use a new path, `_post_infer_inference_only`, which
produces last-token logits via pure GPU ops (`cumsum`, gather, norm, matmul).

### Graph-friendly branches

The body of `_context_forward` contains four `torch.any(infer_state.finetune_mask)`
checks that each trigger a `.item()` sync. These are gated behind
`infer_state._skip_finetune_checks`, which the prefill-CUDA-graph wrapper sets
to `True`.

The `mark_cost_time` decorator on `pre_infer.context_forward` and the
`_context_attention`/`_context_ffn` templates unconditionally calls
`torch.cuda.synchronize()`. This was patched to skip synchronize when
`torch.cuda.is_current_stream_capturing()` is true.

### Warmup

Prefill uses **1** warmup iteration (decode uses 3). Empirically, more warmup
iterations on the side stream leave residual state that makes the first
capture produce wrong logits. One iteration is enough to warm up cuBLAS
workspace and Triton autotune.

---

## Usage

```bash
# Same flag enables decode AND prefill CUDA graph
python -m dserve.server.api_server \
    --model meta-llama/Meta-Llama-3-8B \
    --lora /path/to/adapter \
    --enable-cuda-graph \
    ... (other args)
```

For the in-tree launcher:

```bash
python eval/llama3/launch_llama3.py --enable-cuda-graph
```

### Environment

`dserve._kernels` requires Torch libraries on the loader path:

```bash
export LD_LIBRARY_PATH=/path/to/conda/env/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH
```

### Debug modes

`DSERVE_PREFILL_CG_DEBUG=pad_only` runs the padded forward path without graph
capture/replay. Useful to isolate "is the bug in my padding, or in the
capture machinery?"

---

## Performance (LLaMA-3-8B, single request, 16 output tokens)

### TTFT (time to first token)

| Prompt size (tokens)   | Baseline | CUDA Graph CAPTURE | CUDA Graph REPLAY | Speedup (REPLAY) |
|------------------------|---------:|-------------------:|------------------:|-----------------:|
| ~6 (bucket 128)        |   0.044s |             0.178s |            0.022s |           ~2.0x  |
| ~50 (bucket 128)       |   0.039s |              —     |            0.022s |           ~1.8x  |
| ~120 (bucket 128)      |   0.043s |              —     |            0.023s |           ~1.9x  |
| ~200 (bucket 256)      |   0.046s |              —     |            0.022s |           ~2.1x  |
| ~300 (bucket 384)      |   0.039s |              —     |            0.023s |           ~1.7x  |
| ~450 (bucket 512)      |   0.049s |             0.163s |            0.024s |           ~2.0x  |

CAPTURE cost is paid once per bucket the first time it is seen, then amortized
across all subsequent REPLAYs in that bucket.

### Correctness (10 prompts of varying length, greedy decode)

- **7 / 10** produce output **identical** to baseline
- **2 / 10** semantically equivalent but pick different tokens where top-2
  logits are close (same numerical-sensitivity behavior as the decode CUDA
  graph, caused by cuBLAS workspace warm-up on the capture stream)
- **1 / 10** (largest bucket CAPTURE) has one-token drift at the start but
  remains coherent

Both of the "different" cases reproduce on baseline with slight input
perturbations; they do not indicate a correctness bug in the graph path.

---

## Known limitations and risks

### `max_input_len` is frozen in the captured graph

`context_attention_fwd` is called with `max_input_len = int(total_token_num)`
at capture time. Inside the kernel this is a scalar Python int that gets baked
into the graph. It also determines:

- Grid size: `(batch, head, cdiv(max_input_len, BLOCK))`, BLOCK=128
- Internal scratch: `tmp = torch.empty((batch, head, max_input_len + 256))`

During REPLAY with a longer prompt in the same bucket, `b_seq_len` is updated
(so the attention mask is correct), but the captured grid / scratch size is
not. In practice with BLOCK=128 and a 128-token bucket, `cdiv(anything_in_bucket, 128) == 1`,
so the grid is identical. For larger buckets the grid does change with the
capture-time length.

**Empirical**: across prompt lengths 6–450 tokens with buckets 128/256/384/512,
no correctness regression was observed. But this was not rigorously proven.
A safer long-term fix is to set `max_input_len` to the bucket size at capture,
so the kernel always covers the whole bucket and `b_seq_len` handles
truncation.

### Bucket cache growth

One graph per unique `(batch_size, total_token_num_bucket)`. With
batch_size=1 and 128-token buckets, the cache is bounded by
`ceil(max_seq_len / 128)` entries. Each capture costs roughly 150–200 ms and
holds all of the static buffers above in GPU memory for the life of the
server. For very long-context models you may want a larger bucket size.

### Not yet supported

- `batch_size > 1`: requires padding Q across batched requests and per-request
  LoRA adapter handling that is currently keyed to exact batch size
- Fine-tuning mode: the `torch.any(finetune_mask)` branches do real work
  (activation saving) that cannot be captured as a fixed graph
- Reference alignment path (`ref_mask != None`)
- Interruptible prefill (`prefill_interrupt_event` is checked repeatedly)

---

## File map

| File | What changed |
|------|--------------|
| `dserve/common/cuda_graph_runner.py` | Added `prefill_capture`, `prefill_replay`, `get_prefill_cache_key`, `has_prefill_graph`, separate `_prefill_cache` dict |
| `dserve/models/peft/lora_unordered_batch_mixed.py` | `_prefill` branches into CUDA graph path; allocates T_bucket KV slots; new `_prefill_with_cuda_graph`, `_post_infer_inference_only`; `_context_forward` honors `_skip_finetune_checks` flag |
| `dserve/utils/infer_utils.py` | `mark_cost_time` skips `torch.cuda.synchronize()` during graph capture |

---

## Debugging flow that got us here

The same methodology as the decode CUDA graph:

1. **Pure baseline** (no `--enable-cuda-graph`) — verify model/adapter pipeline works and capture reference output.
2. **`DSERVE_PREFILL_CG_DEBUG=pad_only`** — run the padded forward eagerly, no graph capture. This isolates "is padding correct?" from "is graph machinery correct?"
3. **Full graph** (`--enable-cuda-graph` only) — run capture + replay.

Three bugs surfaced and were fixed in order:

1. `mark_cost_time`'s `torch.cuda.synchronize()` raised
   `operation not permitted when stream is capturing`.
2. Padding token K/V written into the last real KV slot corrupted subsequent
   decode reads → fixed by allocating T_bucket dedicated slots.
3. With 3 warmup iterations (the decode default), the first CAPTURE produced
   garbage logits → reduced to 1 warmup iteration.
