# eval-arxiv — long-context (arxiv) co-serving probe

Question from Ramya Prabhu (2026-09-10): *what happens with a dense model on long
context — run arxiv on DeltaServe*. Her trace columns `prompt` / `output_tokens` come
from the HuggingFace dataset **`ccdv/arxiv-summarization`** (`article` → prompt,
tokenized `abstract` length → output_tokens); the raw dataset has no such columns.
The median arxiv article is ~8k tokens.

Model: **Qwen3-14B-Base, TP=2** on the two RTX 5090s (32k native context).
Llama-3-8B cannot take these prompts (8192 max positions); Llama-3.1-8B would serve
them but its `llama3` RoPE scaling is not implemented in the backward remat.
YAML: `configs/serving_config_finetuning_qwen3_14b_tp2_arxiv.yaml`
(`max_model_len 16384`, `ttft_slo 3.0`; everything else as the TP=2 YAML).

## Files

| file | what |
|---|---|
| `fetch_arxiv.py` | downloads one parquet shard of the `document/test` split from the Hub (no `datasets` lib) and keeps the N longest articles as JSONL under `data/` (git-ignored). Needs `pyarrow` (base miniconda python has it). |
| `launch_arxiv.py` | brings the server up from the arxiv YAML via `eval-tp/launch_deltaserve.build_server_cmd`: `--co [--start-finetuning]` for co-serving, otherwise inference-only on the **v1 runner** (both modes on the same runner). `--max-num-batched-tokens` overrides the prefill chunk. |
| `trace_bench.py` | **the replay**: Poisson arrivals at `--rps`, `--n` requests, prompts of `--len ± --jitter` tokens cut from distinct articles (round-robin reuse only if the pool runs out), `max_tokens` = abstract length (`ignore_eos`), streamed; per request: arrival, TTFT, E2E, mean / max inter-token gap. Same `--seed` → identical arrivals and prompts across modes. |
| `prefill_bench.py` | the earlier idle-server probe (TTFT vs prompt length, `--gap`, `--conc`); its raw outputs were not kept, numbers below. |
| `plot_arxiv.py` | `plots/trace_timeline_{co,inf}.png` (requests, tokens/s, step times + backward cycles) and `plots/trace_compare.png` (per-request TTFT / E2E A-B). Blue = inference, orange = finetuning / co-serving, grey = inference-only baseline. |
| `output/`, `plots/` | the 0.4 rps trace run: `trace_{co,inf}.csv` (+`_meta.json`), `server_trace_{co,inf}.log`, `bwd_log_trace_co.csv`, `step_trace_trace_co.csv`, the PNGs. |

```bash
/home/jiaxuan/miniconda3/bin/python eval-arxiv/fetch_arxiv.py --out eval-arxiv/data/long_articles.jsonl
python eval-arxiv/launch_arxiv.py --co --start-finetuning        # or without flags: inference-only
python eval-arxiv/trace_bench.py --data eval-arxiv/data/long_articles.jsonl --rps 0.4 --n 20 --len 8192 --warmup --out eval-arxiv/output/trace_co.csv
python eval-arxiv/plot_arxiv.py
```

## 0.4 rps trace — 20 requests, 8k ± 512 prompt tokens, 113–430 output tokens (2026-09-10)

Same Poisson schedule (seed 0, last arrival at 57.5 s) and the same 20 articles in both
modes and both configs. `gpu_memory_utilization 0.80` throughout.

**Current config** (`max_num_batched_tokens: 4096`, `ttft_slo: 2`) — `output/`, `plots/`:

| | inference only | co-serving |
|---|---|---|
| TTFT p50 / p90 / max | 1.67 / 2.58 / 3.02 s | 1.69 / 4.93 / 5.07 s |
| E2E p50 / max | 11.3 / 18.7 s | 12.7 / 22.6 s |
| max TBT per request (p50 / max) | 782 / 785 ms | 781 / 781 ms |
| KV cache | ~72k tokens | 56,768 tokens |
| FT throughput | — | 113 tok/s (25 backward cycles in 61 s) |

**Previous config** (`max_num_batched_tokens: 2048`, `ttft_slo: 3`) — `output/chunk2048_slo3/`, `plots/chunk2048_slo3/`:

| | inference only | co-serving |
|---|---|---|
| TTFT p50 / p90 / max | 1.98 / 2.91 / 3.17 s | 1.98 / 3.70 / 4.59 s |
| E2E p50 / max | 11.4 / 19.3 s | 11.8 / 21.2 s |
| max TBT per request (p50 / max) | 405 / 405 ms | 403 / 405 ms |
| KV cache | 71,984 tokens | 59,776 tokens |
| FT throughput | — | 107 tok/s (27 backward cycles in 61 s) |

What the traces show:

- **Chunk size trades TTFT against TBT, in both modes.** 4096-token chunks: an 8k prefill
  is 2 steps, TTFT p50 1.98 → 1.67 s, but every decoding request now sees one ~780 ms
  token gap (one chunk) instead of ~400 ms. Prefill compute is unchanged.
- **TTFT is queueing behind other prefills.** One 8k prefill is 1.5 s; at 0.4 rps a new
  arrival finds another prefill in flight about half the time.
- **The co-serving TTFT tail is KV capacity, not contention — in both configs.** The slow
  requests (8, 9, 10, 16 here; 9, 10 before) waited with 6–7 resident requests holding
  ~51–59k KV tokens against a 56.8–59.8k-token cache, one request in `waiting` and no
  prefill chunk scheduled; the step trace shows zero FT tokens, no backward and no pause
  in those windows. The backward children cost ~12–15k KV tokens at 0.80 utilisation → one
  fewer resident 8k request than the inference-only server. `gpu_memory_utilization ~0.88`
  or a smaller CUDA-graph capture set (the pool reserves 5 GiB/GPU for decode batches
  ≤ 512 that never occur here) removes it.
- **FT rides no inference prefill in either config** (1 of 40 chunks carried 6 FT tokens
  at 4096). 93 % of steps are decode-only and `coserving_admission_phase: prefill` denies
  them; prefill chunks fill the step budget. FT runs only while the system is empty
  (~48–53 s and after 61 s) → ~110 FT tok/s. Leaving budget for FT (e.g. 2304 = 2048 + 256)
  or the `both` scheduler are the levers; neither has been run yet.
- The worker's `[batch …] FT OPEN buf=0/256` field is always 0 under TP: the worker
  coordinator's buffer is not maintained in relay mode (the scheduler's is). Cosmetic.

## Earlier idle-server prefill probe (raw outputs not kept)

Median TTFT with 8 output tokens, distinct article per request, 4 samples per length:
2048 → 0.378 s, 4096 → 0.756 s, 8192 → 1.542 s, 12288 → 2.366 s, 16000 → 3.167 s
(~0.19 ms/token, compute-bound; an 8192-token chunk saves 1–2 %; co-serving +1–2 %;
four concurrent 8k prefills serialize: 1.6 / 3.5 / 5.0 / 6.2 s).

## Bug found and fixed on the way (`deltaserve/ft_scheduler.py`)

The first co-serving runs of the probe had FT dying ~15 s into the workload with
`[ft-sched] FT exhausted … admission_open=False epoch_flush_pending=True` at epoch 0.
Sequence: a tier-C abort of an FT-only step is rolled back correctly; the re-schedule
then admits ~250 FT tokens next to the 2048-token inference chunk, `note_injection`
closes admission ("next sample won't fit"), but the base scheduler drops the FT requests
(the chunk took the whole token budget) and only their samples were released — the flags
stayed closed with an empty buffer, and nothing can reopen admission without a backward.
Three fixes:

1. `_current_step_features`: a *running* request still prefilling (chunked prefill) is
   counted as its next chunk, not as one decode token (chunks 2–4 of an 8k prompt looked
   decode-only → phase gate → 0 FT, and the prefill was hidden from the SLO prediction).
2. `admit_ft_to_step`: FT admission is capped by the step's remaining token budget.
3. `schedule()`: the admission flags are re-derived from the FT tokens that actually
   scheduled (restore the snapshot, re-run `note_injection`), mirroring `_rollback_ft_step`.

Verified live (0 exhaustions over 19 rollbacks on the sweep; the 0.4 rps trace above
ran with FT alive throughout). `tests/test_tp_timing_relay.py` 26/26 and
`tests/test_step_trace.py` 28/28 pass.
