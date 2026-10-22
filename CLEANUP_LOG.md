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
| 00 | Baseline: branch, half timelines, `--out-dir`, runners | | | | |
