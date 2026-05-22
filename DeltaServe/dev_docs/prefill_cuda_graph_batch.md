# Prefill CUDA Graph — Batched (128-token TTFT)

The full-graph prefill path now supports `batch_size > 1` via batch padding
to bs_bucket ∈ {1, 2, 4, 8, 16, 32, 64}. Padded request slots have
`b_seq_len = 0`, so attention treats them as no-ops; the captured graph runs
as if there were `bs_bucket` real requests. Output logits are bs_bucket-sized
and sliced to the real `batch_size` before returning to the caller.

This document records measured TTFT at **~128 tokens per request** (the small
end of the prefill workload, where kernel-launch overhead dominates and
CUDA graph yields the largest relative speedup).

---

## Test setup

- Model: Meta-LLaMA-3-8B + llama3-toy-lora (rank=16)
- Prompts: ~10–13 tokens each, 8 distinct natural-language prompts. All fall
  in the `T_bucket = 128` token bucket.
- Concurrency: N requests fired simultaneously via threads.
- For each batch size we run a warmup pass to trigger CUDA graph capture,
  then a measured pass that is REPLAY-only.
- `--enable-cuda-graph` toggles the full-graph prefill path.

Server flags (both runs identical apart from `--enable-cuda-graph`):

```
--max_total_token_num 25000
--enable_unified_mem_manager --unified_mem_manager_max_size 6
```

---

## TTFT @ 128-token bucket

| batch_size | Baseline TTFT (s) | Full-graph CUDA TTFT (s) | TTFT speedup |
|:----------:|------------------:|-------------------------:|-------------:|
| 1          | 0.0341            | **0.0191**               | **1.79x**    |
| 2          | 0.0345            | **0.0227**               | **1.52x**    |
| 4          | 0.0348            | **0.0209**               | **1.67x**    |
| 8          | 0.0327            | **0.0195**               | **1.68x**    |

## Wall-clock for the whole batch

| batch_size | Baseline wall (s) | Full-graph wall (s) | Wall speedup |
|:----------:|------------------:|--------------------:|-------------:|
| 1          | 0.216             | **0.098**           | **2.20x**    |
| 2          | 0.218             | **0.105**           | **2.08x**    |
| 4          | 0.219             | **0.106**           | **2.07x**    |
| 8          | 0.221             | **0.109**           | **2.03x**    |

## Throughput (req/s)

| batch_size | Baseline | Full-graph CUDA | Throughput speedup |
|:----------:|---------:|----------------:|-------------------:|
| 1          |     4.6  |   **10.2**      | **2.2x**           |
| 2          |     9.2  |   **19.0**      | **2.1x**           |
| 4          |    18.3  |   **37.7**      | **2.1x**           |
| 8          |    36.2  |   **73.4**      | **2.0x**           |

---

## Notes

- TTFT speedup (1.5–1.8x) is consistently lower than wall-clock speedup
  (2.0–2.2x) — graph replay collapses many small kernel launches into a
  single submission, and the saving compounds across the whole batch.
- The capture cost is amortized across all subsequent replays in the same
  bucket; first request hitting a new bucket pays ~0.15–0.2 s.
- This benchmark covers the small-prompt regime. At ~512 tokens/request the
  picture changes: kernel-launch overhead is a smaller fraction of work,
  and the prefill scheduler can fragment large batches into mini-batches
  that interact poorly with the CUDA graph KV-slot padding (see notes in
  `prefill_cuda_graph.md`). For the 128-token regime documented here the
  CUDA graph path is uniformly faster than baseline.
