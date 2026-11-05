# Cleanup / restructure log

Goal: optimize and clean up DeltaServe-vLLM, and shrink the changes to upstream vLLM,
with **no performance or feature loss** relative to the starting point.

- **Start point:** `origin/main` @ `6c4dc5cbc` (== `origin/tp4-a100-slo-tuning`), branch `cleanup`.
- **Upstream base:** `vllm-project/vllm` @ `117afeea4665367a3066c1df58d4082d07fcc946`
  (`0.21.1rc1.dev123`), checked out read-only at `vllm-upstream/` for diffing.
- **Order:** B (move our logic behind vLLM's seams inside `dserve-vllm/`, imports unchanged),
  then A (extract a separate `deltaserve` plugin package + small patch on stock vLLM).
  Per edit site, prefer: official seam > wrapping a smaller method > object-level trick >
  one-line hook call in the patch. Never copy a whole upstream method into the plugin.

## Rollback

Every step is one commit on `cleanup`; `git revert <sha>` (or `git reset --hard <sha>` of the
previous step) undoes it. The files a step modified are also copied, pre-change, into
`.cleanup_backups/<step>/` (locally excluded from git).

## Validation protocol

- **Tests:** `tests/run_all.sh eval-tp/output/cleanup/<step>/tests` — every standalone test
  script (CPU, gloo, and the 2-GPU NCCL gates).
- **Quick perf (every step that touches runtime code):**
  `eval-tp/cleanup_validate.sh <step>` — Qwen3-14B TP=2 co-serving replay of the first half
  (t < 26 s) of the tight (240 req) and loose (120 req) timelines, with the step trace, then
  `compare_runs.py` against `00_base`. Config: the preset
  `configs/serving_config_finetuning_qwen3_14b_tp2.yaml` (TTFT SLO 0.4 s).
- **Big perf (after a batch of steps):** `eval-tp/cleanup_validate.sh <step> nutanix-600-800`.
- **Metrics watched:** TTFT satisfaction, TTFT p50/p95/p99, requests with worst TBT > 50 ms,
  FT tok/s, backward cycles + cycle p50, loss first/last, mixed/FT-only step p50, KV-cache
  tokens, graph memory.

## Decisions

- Drop: the OPT loss-only path (`bwd_services/opt.py`, OPT config / adapters / scripts / tests,
  and the `final_hidden` activation save only it consumes). The uncommitted `memprof`
  instrumentation was dropped before the branch was cut (backup:
  `.cleanup_backups/00_drop_memprof/`, incl. `eval/memprof/REPORT.md` + data).
- Keep: `max_prepare`; every `print_*` + `print_step_mode`; `validate_estimator` +
  `estimator_validation_path`; every flag the A100 configs use.
- YAML sections: `finetune` (workload + perf/behaviour switches), `throttles` (admission phase,
  constrain / match factors, `rps_throttle_*`, `fwd_token_throttle*`), `slo` (SLOs, profiling,
  safety margin, estimator validation / state), `debug` (`print_*` + trace / log paths).
  The loader still accepts a key in an old section, with a deprecation warning; the repo's
  configs are converted and the old ones kept in `configs/legacy/`.

## Steps

| # | Step | Commit | Tests | Quick perf | Notes |
|---|---|---|---|---|---|
| 00 | Baseline: branch, half timelines, `--out-dir`, runners | `12b94aece` | 24/29 (5 stale, pre-existing) | tight-half 483 FT tok/s, TTFT 100 %, p50 116 ms · loose-half 860 tok/s, 100 %, p50 113 ms; repeat run within ±2 % FT, ±3 % TTFT p50, ±6 % p99 | baseline `00_base`, repeat `00_base_r2` |
| 01 | Drop the OPT loss-only path; `final_hidden` save only for the hash check | `9b55401f3` | (with 02) | (with 02) | fixed the 5 stale tests; `ipc` protocol-only service for the IPC tests |
| 02 | YAML sections finetune / throttles / slo / debug | `23beb40e9` | 27/29 → 02b | tight 470 / loose 872 FT tok/s (−2.6 / +1.3 %), TTFT 100 / 100 %, step + cycle times identical; p99 tail 266 / 245 ms is within the baseline's own tail spread (worst 288 ms) | 23 configs converted, originals in `configs/legacy/` load identically |
| 02b | Tests follow 01 | `5c1da6017` | | | |
| 03 | `DeltaServeWorker` subclass (worker_cls); engine defaults out of `VllmConfig` | `ee05384de` | (with 04) | (with 04) | `gpu_worker.py` stock |
| 04 | `DeltaServeModelRunner` + `FtCudagraphDispatcher`; save op off `ForwardContext` | `80ab83411` | all pass | tight 482 / loose 872 FT tok/s, TTFT 100 / 100 %, p50 116 / 110, p99 220 / 256 ms; every step / cycle timing equal | `gpu_model_runner.py`, `cudagraph_dispatcher.py`, `forward_context.py` stock |
| 05 | EngineCore back to stock + 6 hooks; tiers A/B + abort roll-back in the scheduler | `d7cb8a5ef` | (with 07) | (with 07) | `engine/core.py` 562 → 24 lines; fixes a dropped relayed ack on TP aborts |
| 06 | Relay data as `SchedulerOutput` / `ModelRunnerOutput` attributes; `Request.is_finetuning` attribute | `bb2239853`, `d7f917c76` | (with 07) | (with 07) | `outputs.py`, `sched/output.py`, `request.py` stock |
| 07 | TP workers stay daemonic; the child spawn lifts the daemon check | `1dc691ff0` | M1 flaky (see below) | tight 481 / loose 884 FT tok/s; tight TTFT **98.75 %, p99 406 ms** (one burst behind a 156 ms unpaused prefill) | `multiproc_executor.py` stock; `tests/test_tp_no_orphans.py` added |
| 08 | Save-point hooks issue the graph save op; TP trigger-race fix; M1 near-tie tolerance | `9c0791a9a`, `781ebdd89` | 28/28 (incl. orphan gate 5/5) | tight 483 / 492 FT tok/s, TTFT 100 / 100 %, p99 222 / 200 ms, worst 229 / 228 ms (base 288 / 265); loose 879, p99 163 ms; 0 unpaused slow prefills (base 2 per run) | `qwen3.py`, `qwen2.py` stock |
| 09 | `finetune.graph_ft_only_batches` (FT-only steps on the has_ft graphs, abort between segments) | `d30fb1ac4` | M1 / M2 pass | vs 08: FT-only step 67 → 59 ms (tight), 66 → 57 ms (loose); FT +0.2 / +3 %; TTFT 100 / 100 % | aborts land mid-forward (segment 25) on both ranks |
| 10 | **nutanix-600-800** — baseline code vs current vs current + FT-only graphs | | | TTFT 100 / 100 / 100 %, p99 300 / 223 / 240 ms; FT 486 / 489 / 512 tok/s; avg TBT, E2E equal; worst-TBT p90 120 / 94 / 118 ms | FT-only graphs enabled in the Qwen3-14B TP=2 preset |

| A | **Plugin**: `deltaserve-plugin/` on stock vLLM (`vllm-upstream/` @117afeea4), new env `deltaserve-plugin` | `d832ceb35` … | 28/28 + step1 17/17 (incl. GPU) | vs the fork with the same config: tight 494 / loose 908 FT tok/s (484 / 905), TTFT 100 / 100 %; nutanix 99.81 % (one request at 419 ms — the same burst request the baseline served at 388 ms), FT 508 | vLLM unmodified |
| 11 | `finetune.pause_on_arrival` (child yields when a request arrives during a cycle), ON in the preset | `7d9ee67c5` | `test_pause_on_arrival` 12/12; TP gates pass | nutanix ×2: TTFT 100 / 100 %, p95 166 → 90 / 90 ms, p99 270 → 100 / 99 ms, FT −1.7 / −1.4 %; tight p99 171 → 138, loose 177 → 134 ms (FT −0.6 / −1.3 %) | removes the decode-contention tail |

## Findings along the way

- **Displaced decorators (step 04).** The in-tree runner edits had inserted new
  methods directly under upstream's `@torch.inference_mode()` lines, so
  `execute_model` / `sample_tokens` lost their decorators and two DeltaServe
  helpers gained them. Harmless (the worker wraps both calls in
  `inference_mode`), and gone now that the runner file is stock.
- **Dead per-step work.** `EngineCore._maybe_log_batch_scheduled` classifies every
  non-decode batch under `print_scheduler_add` and then prints nothing (its
  `dprint` is commented out).
- **Whole-buffer zeroing per backward cycle.** The child zeroes every activation
  buffer in full after each cycle (~700 MB of memsets on Qwen3-14B at
  `s_max=256`), though only rows `[:n]` were written.
- **The TP trigger race was behind the step-07 tail.** Every run, the baseline
  included, has 1–3 inference prefills at 114–156 ms (vs ~40 ms) with no backward
  pending: the scheduler's trigger rides step N+1 while step N's prefill is still on
  the GPU, and the fresh child time-slices against it. In 07 it hit the start of a
  burst (5 requests 0.34–0.50 s). Fixed in 08: the runner records each inference
  prefill's completion event, and a trigger that arrives before it fired starts the
  child paused (resumed by the deferred-resume poll).
- **M1 is flaky by construction.** FT admission timing changes the batch
  composition and bf16 results depend on it; the same code passed and failed on
  consecutive runs (divergence at a near-tie, "," vs "."). The test now accepts a
  divergence only at a near-tie (logprob gap < 0.25 nats).
- **`print_scheduler_add` prints nothing** (its line was commented out in May);
  the flag is kept, the per-step work behind it is gone.
- **FT-only steps are host-bound** (step trace: 62.6 ms host of a 66 ms step).
  `finetune.graph_ft_only_batches` (off by default) replays them from the has_ft
  graphs, with the tier-C abort polled between piecewise segments.
- **The remaining TTFT tail was decode contention right after a cycle starts.**
  Decode-only steps share the GPU with the child by design and ran 180–330 ms
  (vs 17) at the start of a backward cycle; a request arriving then queued behind
  one or two of them — the same nutanix burst request at 388 ms (baseline) and
  419 ms (plugin run). `pause_on_arrival` makes the child yield at its next
  boundary when a request arrives; the tail is gone (p99 ~100 ms).
- **FT tok/s in the table above is over-counted, equally for every run.**
  `compare_runs.py` divided all backward tokens in the log by the request window,
  including cycles from the 5 s idle lead-in and after the last request (~1/3 too
  high on the half timelines, ~1/5 on nutanix). Fixed to count in-window cycles
  only. In-window, baseline → final: tight 297 → 295, loose 550 → 579 (+5 %),
  nutanix 391 → 401 tok/s (+2.5 %). Relative comparisons in the table hold.
  Figures: `eval-tp/plots/cleanup_compare_{summary,distributions}.png`
  (`eval-tp/plot_cleanup_compare.py`).
