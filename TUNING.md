# Tuning inference against finetuning

How to move a DeltaServe deployment along its one trade: **inference latency
(TTFT, time between tokens) against finetuning throughput (FT tokens / s)**.
Every number is measured on Qwen3-14B TP=2, 2× RTX 5090, vLLM 0.34, the three
5090 timelines (tight-half, loose-half, production-600-800), single runs,
2026-11-26. Keys are written `section.key` as in the serving YAML; each key's
full description is its docstring in `deltaserve-plugin/deltaserve/config.py`.

Read this before changing an SLO, a throttle, a size or the MPS setting, and
before explaining a latency or throughput number. Update it when a measurement
changes what it says.

## 1. What decides the trade

FT uses the GPU in three ways, and each has its own guard:

| FT work | What it costs inference | What limits it |
|---|---|---|
| FT tokens in a step that also carries inference (**mixed step**) | that step is longer: the requests prefilled in it get their first token later, the requests decoding in it wait | the SLO admission checks (§3), the per-step cap (§5) |
| FT tokens in a step with no inference (**FT-only step**) | a request arriving meanwhile waits for it, or aborts it (tier C) | `forward_interruptible`, `ft_only_tokens_per_step` |
| the **backward** (a separate process) | a decode step that overlaps it is slower; a prefill pauses it | the pause contract (always), the backward gate (§3), MPS (§6) |

Two facts decide most tuning questions:

- **The admission spends whatever room the targets leave.** Loosen a target, or
  make inference cheaper (MPS), and FT rises until the measured latency is back
  near the target. A higher TTFT p50 after such a change is the admission using
  the room, not a regression. To get the latency back, tighten the target or
  add headroom — not the other way round.
- **Attainment is counted per request against `slo.ttft_slo` and
  `slo.avg_tbt_slo`.** A window average steered exactly onto the target leaves
  about half the requests above it: the headroom keys (§3) exist for that.

## 2. Start here

| Goal | Change | Measured |
|---|---|---|
| A safe default | `configs/preset/serving_config_finetuning_qwen3_14b_tp2.yaml` (256-token buffer, `ttft_slo: 0.4`, `avg_tbt_slo: 0.035`) | FT 407 / 721 / 595 tok/s (tight / loose / production), 100 % TTFT, 85 / 100 / 99.6 % of requests within the TBT target |
| Lower latency at about the same FT | run an MPS daemon, `finetune_features.backward_mps_percentage: 50` (`…_tp2_mps.yaml`, §6) | production: TTFT p50 73 → 67 ms, average TBT 25.9 → 19.4 ms, FT 595 → 600; tight-half: TBT attainment 85 → 100 %, FT 407 → 433; loose-half: TTFT p50 87 → 71 ms, FT 721 → 640 (the cap also holds while inference is idle) |
| Much more FT, latency tail allowed to move | one large backward: `…_tp2_step1024.yaml` (§5) | FT 481 / 922 / 751 (+18 to +28 % over the default), TTFT p99 up to 170 ms, TBT attainment 100 / 100 / 99.6 % |
| More FT under MPS | MPS daemon with `backward_mps_percentage: 0` | production FT 595 → 675, average TBT 20.8 ms, TTFT p50 73 → 82 ms |
| Requests miss the TBT target near capacity | `slo.overload_hold_s: 1.0`, then `slo.avg_tbt_admission_headroom: 0.9` (§3) | at capacity 92.5 → 98.3 %, just over it 69 → 83 % (inference alone: 100 / 84 %), FT −5 to −8 % |
| Requests queue / TTFT collapses in bursts | check `GPU KV cache size` in the launch log first (§7) | 30,720 → 63,488 tokens (`gpu_memory_utilization` 0.80 → 0.88) took TTFT attainment 82 → 96 % at an unchanged FT rate |
| FT must stop above a known request rate | `throttles.rps_throttle_*` (§4) | — |

## 3. The SLO targets and their margins (`slo`)

| Key | Default | Effect of raising it |
|---|---|---|
| `ttft_slo` | 1.0 (0.4 in the 5090 presets) | more FT in mixed steps; TTFT moves towards it. Admission keeps the oldest waiting request within 0.9 × this |
| `avg_tbt_slo` | 0.1 (0.035 in the presets) | THE token-latency target: more FT next to decoding requests, more backward overlap. 20 ms is out of reach at twice the production rate (inference alone: 88 %) |
| `avg_tbt_window_s` | 1.0 | the window the average is kept over. 0 = `avg_tbt_slo` is not enforced at all, only `max_tbt_slo` per step |
| `max_tbt_slo` | 0.1 | a per-step backstop, not a target: a step with a prefill is slow anyway. With the settings before 2026-11-26 (0.05 here, no window, no backward gate) the same code gives FT 484 instead of 595 on production, TTFT p99 85 instead of 112 ms |
| `avg_tbt_pause_backward` | true | a decode step that would take the window average past `avg_tbt_headroom` × target pauses the backward. Off: the backward overlaps decode freely (at a 20 ms target: 88 → 58 % attainment, FT unchanged) |
| `avg_tbt_headroom` | 0.9 | lower = the backward gives way earlier. 0.8 in the step-1024 preset |
| `avg_tbt_admission_headroom` | 1.0 | below 1, FT admission steers the window to that fraction. 0.9 helps at / over capacity (at capacity 92.5 → 98.3 %, just over it 69 → 79 %, FT −2 %) and can cost attainment and FT where there was room (20 ms target at the normal rate: 88 → 81 %, FT −5 %) |
| `overload_hold_s` | 0 (1.0 in the templates) | FT stands down — no admission, backward paused — until this long after the window average was above the target or a request had waited 0.9 × `ttft_slo`. 1 s: attainment back to the inference-only level for −5 to −8 % FT. 5 s: −29 % FT at the normal rate and nothing more |
| `admission_margin` / `_k` | `relative`, 1.5 (presets pin `legacy`) | inflates every prediction by the estimator's error: fewer FT steps over `max_tbt_slo` (10 % → 2 %), FT −1 % |

What the estimator can and cannot predict (actual / predicted): prefill, mixed
and FT-only steps within 0.96–1.08; decode-only 0.82–1.00; **a decode step
under a running backward: 0.3–2.2** — without MPS it is either a clean step or
twice one, and the flag it is selected by (`pending_backward`) is true for
about 40 % of steps the child is not actually working in. The backward gate
leans on that prediction; MPS shrinks the problem (§6).

## 4. Throttles (`throttles`)

These close FT by a rule instead of by a prediction. They are for a workload you
know; the SLO checks of §3 adapt on their own.

| Key | Default | Use |
|---|---|---|
| `coserving_admission_phase` | `prefill` | `both` also admits FT to decode-only and idle steps: more FT, more pressure on TBT (`slo.decode_only_ft_safety_margin` scales `max_tbt_slo` for those steps) |
| `ft_tokens_admission_constrain_factor` | −1 (off) | at most `factor × inference prefill tokens` of FT per mixed step: keeps FT from dwarfing the prefill it rides with (0.25–0.5 in the A100 presets) |
| `match_prefill_workload_factor` | 0 (off) | a leaky bucket: FT is admitted only as inference prefill "pays" for it. Ties FT to the inference load |
| `rps_throttle_enable`, `_close_rps`, `_open_rps`, `_window_s`, `_close_time` | off | admission closes above `close_rps` and reopens below `open_rps`. The blunt tool for a rate you know overloads. It does not replace KV capacity (§7): on the overloaded workload it changed nothing |
| `fwd_token_throttle_enable`, `fwd_token_throttle` | off | also pause the backward for decode-only steps whose inference token count reaches the threshold (1 = every step with inference). Trades FT for TBT without a prediction; the backward gate of §3 does the same with one |

## 5. Sizes (`finetune_param`): where the FT throughput is

| Key | What it is | Guidance |
|---|---|---|
| `max_saved_finetuning_tokens` (or `ft_step_size`) | the activation buffer = one optimizer step | 2.70 MB per token per GPU. 256 → 1024 is the largest single FT gain, if the memory is there |
| `bwd_batch_size` | tokens per backward (default: the whole step) | ONE backward of the step is cheapest per token (FT-only: 1877 vs 1276 tok/s for 1024 vs 4 × 256). Split it only when the child's memory must stay small |
| `ft_tokens_per_step` | cap of a mixed step | this, not the backward, is what protects TTFT with a large buffer: 256 → TTFT p99 125 ms, 512 → 138, uncapped → 185 (FT 795 / 860 / 860) |
| `ft_only_tokens_per_step` | cap of an FT-only step | leave it at the buffer size: such a step is aborted when a request arrives (FT-only 1659 → 1877) |
| `bwd_whole_batches` | with several backwards per step | on: no remainder backward (FT 589 → 624 for 1024 / 256) |
| `bwd_batch_size_idle` | a larger backward only while no inference request is around | measured below one large backward at all times (FT 708 vs 795, KV 6k vs 17k tokens): off |

The step-1024 preset is `max_saved_finetuning_tokens: 1024`,
`ft_tokens_per_step: 256`, `ft_only_tokens_per_step: 1024`, plus
`avg_tbt_headroom: 0.8`, `avg_tbt_admission_headroom: 0.9`,
`overload_hold_s: 1.0` — without those three the larger backward pushes FT to
the limit of the TBT target (loose-half: 77.5 % of requests within it).

A larger step also means fewer optimizer updates per token: the loss after the
same wall time is higher (0.83 → 1.00 for 256 → 1024 on production). That is a
training question, not a serving one, but it is part of the choice.

## 6. MPS

Without an MPS daemon the inference worker and the backward child are two CUDA
contexts the driver time-slices. With one they run concurrently.

| | no MPS | MPS, child uncapped | MPS, child at 50 % |
|---|---|---|---|
| decode step overlapping a backward (clean: 12.7 ms) | 23.8 ms (1.87×) | 15.6 ms (1.19×) | 16.0 ms (1.22×) |
| prefill under a paused backward | 29.6 ms | 30.0 ms | 30.0 ms |
| production: FT tok/s | 595 | 675 | 605 |
| production: TTFT p50 / p99 | 72.9 / 112.5 ms | 82.4 / 118.3 | 66.6 / 110.9 |
| production: average TBT p50 | 25.9 ms | 20.8 | 19.6 |
| tight-half: FT tok/s | 407 | 509 | 424 |
| tight-half: requests within the TBT target | 85 % | 100 % | 100 % |
| loose-half: FT tok/s | 721 | — | 640 |
| loose-half: TTFT p50 / p99 | 87.3 / 125.0 ms | — | 70.8 / 112.4 |
| backward cycle | 123 ms | 132 | 163–200 |

- Start it with `scripts/mps.sh start` (user level, no root) and give the
  server the two exports it prints. `backward_mps_percentage` is the child's
  share of the SMs (`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`, set for the child
  only); it does nothing without a daemon.
- **50** lowers TTFT and TBT on all three timelines and keeps FT on two; on
  loose-half FT drops 11 %: the cap is a hard limit on the child, also while no
  inference runs, and its backward cycle goes from 123 to 163–200 ms. A higher
  percentage gives that back (not measured between 50 and uncapped). **0**
  (uncapped) under a daemon turns the cheaper overlap into FT: the admission puts FT into more prefill
  steps (41 → 50 % of them), which is where the higher TTFT p50 comes from —
  the pause still works (the prefill row above).
- MPS is a property of the host, not of the config: every CUDA process that has
  the pipe directory exported goes through the daemon. The presets without
  `_mps` assume none.
- The A100 presets with `_mps` use 70; nothing above was measured there.

## 7. Before blaming FT

- **KV cache.** On vLLM 0.34 it is what is left of `gpu_memory_utilization`
  after the CUDA graphs and the FT buffers, sized once at launch: 63,488 tokens
  for the default preset at 0.88, 30,720 at 0.80, 17,408 for the step-1024
  preset, 89,088 inference-only. When the running requests' contexts reach it,
  requests queue for KV and TTFT collapses whatever the FT rate. The backward
  child needs the rest of the GPU (2.25 GiB for a 256-token backward, 5.2 GiB
  for 1024): do not take `gpu_memory_utilization` to 1.
- **The system may be over capacity alone.** Run the same trace without `--co`:
  the production timeline with prompts × 4 and outputs × 3 gives 84 % TBT
  attainment with no FT at all.
- **Look at the step trace** (`--step-trace`): step kinds, `pending_bwd` /
  `paused_bwd`, predicted against actual, and the admission codes (`S` SLO,
  `F` buffer, `C` step budget, `O` overload hold, `T` throttle) say which guard
  is binding.

## 8. Tried, no gain (all off by default)

`backward_cuda_graph_attn_shapes` (several attention graph shapes),
`fill_graph_padding`, `ft_breakable_cudagraph`, `profile_extra_shapes`,
`bwd_batch_size_idle`, a pause hysteresis for the backward gate (removed), a
5 s overload hold. Their docstrings carry the numbers; do not spend time on
them again without a new reason.
