# Results

Qwen3-14B-Base, TP=2 on 2× RTX 5090, co-serving the shipped preset
(`configs/preset/serving_config_finetuning_qwen3_14b_tp2.yaml`), TTFT SLO 0.4 s. Three
inference workloads replayed from recorded traces:

| Workload | Requests | Span | Rate |
|---|---|---|---|
| tight-half | 240 | 22 s | ~11 req/s |
| loose-half | 120 | 20 s | ~6 req/s |
| production-600-800 | 534 | 184 s | ~2.9 req/s (production trace slice) |

The baseline is the original patched-fork implementation. The final version is
this plugin on stock vLLM. Each figure is the mean of two runs.

| Workload | TTFT p99, baseline → final | FT throughput, baseline → final | TTFT SLO satisfaction |
|---|---|---|---|
| tight-half | 201 → **137 ms** | 297 → 295 tok/s | 100 % → 100 % |
| loose-half | 204 → **134 ms** | 550 → **579** tok/s | 100 % → 100 % |
| production-600-800 | 295 → **~100 ms** | 391 → **401** tok/s | 100 % → 100 % |

![baseline vs final summary](figures/cleanup_compare_summary.png)
![baseline vs final distributions](figures/cleanup_compare_distributions.png)

Reproduce with `eval-tp/cleanup_validate.sh` (see [Running experiments](../README.md#running-experiments)).

**vLLM v0.33.0 (current pin) vs the previous pin (117afeea4)**, same plugin and preset:

| Workload | TTFT p50 | TTFT p99 | FT throughput | TTFT SLO satisfaction |
|---|---|---|---|---|
| tight-half | 110.5 → **102.9 ms** | 138 → 143 ms | 295 → **360 tok/s (+22 %)** | 100 % → 100 % |
| loose-half | 104.2 → **87.8 ms** | 134 → 135 ms | 579 → **626 tok/s (+8 %)** | 100 % → 100 % |
| production-600-800 | 76.1 → **68.9 ms** | 99.8 → **89.3 ms** | 401 → **460 tok/s (+15 %)** | 100 % → 100 % |

vLLM itself got faster (decode step 18.3 → 14.9 ms, prefill 40 → 36 ms) and its CUDA
graphs use 6.8 → 4.0 GiB, leaving ~15 % more KV cache. One regression, upstream and not
FT-specific: the host dispatch of piecewise prefill steps grew 2 → ~20 ms (inference-only
prefills too); async scheduling hides it.

**vLLM v0.34.0 + the 2026-11 changes vs v0.33.0** (Qwen3-14B TP=2, 2× RTX 5090, TTFT SLO
0.4 s, single runs; 100 % of the requests within the TTFT SLO in every cell). Three
columns after the baseline, because the SLO settings changed too:
- **same settings**: the new code with the old SLO settings (`max_tbt_slo: 0.05`, the
  average TBT not enforced) — what the port, the backward attention budget and the
  backward child without torch's `_native` layer are worth on their own;
- **default preset**: the settings as they are now (`avg_tbt_slo: 0.035` kept over a
  1 s window, `max_tbt_slo: 0.1` as a per-step backstop, the backward gate);
- **step-1024 preset** (`…_tp2_step1024.yaml`): one 1024-token backward per step,
  256 FT tokens per mixed step.

| Workload | Metric | v0.33 | v0.34, same settings | default preset | step-1024 preset |
|---|---|---|---|---|---|
| tight-half | FT tok/s | 349 | 415 (+19 %) | 404 (+16 %) | **481 (+38 %)** |
| | TTFT p50 / p99 | 101 / 138 ms | 98 / 142 | 92 / 128 | 70 / 167 |
| | average TBT p50 | 35.8 ms | 34.9 | 33.3 | 28.4 |
| | requests within 35 ms average TBT | 45 % | 51 % | 77 % | 100 % |
| loose-half | FT tok/s | 627 | 709 (+13 %) | 723 (+15 %) | **922 (+47 %)** |
| | TTFT p50 / p99 | 86 / 129 ms | 84 / 117 | 89 / 119 | 66 / 170 |
| | average TBT p50 | 27.4 ms | 28.2 | 27.7 | 23.9 |
| | requests within 35 ms average TBT | 100 % | 100 % | 100 % | 100 % |
| production-600-800 | FT tok/s | 466 | 484 (+4 %) | 597 (+28 %) | **751 (+61 %)** |
| | TTFT p50 / p99 | 69 / 89 ms | 69 / 85 | 72 / 112 | 92 / 127 |
| | average TBT p50 | 20.3 ms | 20.3 | 25.8 | 23.6 |
| | requests within 35 ms average TBT | 100 % | 100 % | 99.8 % | 99.6 % |

With the same settings nothing got worse beyond run-to-run noise (tight-half TTFT p99
138 → 142 ms) and FT is 4–19 % higher: the backward cycle went from 148 to 123 ms once
its attention actually replayed its graph. The default preset buys more FT on the
production timeline with the room the new SLO settings give it — TTFT p99 89 → 112 ms
and the average TBT 20 → 26 ms, both far inside their targets. The step-1024 preset
moves the TTFT tail further (up to 170 ms) for 38–61 % more FT and a lower average TBT.
Runs: `eval-tp/output/cleanup/{v2port_v1, ga_final_oldslo, ga_final_default,
ga_final_step1024b}`.
