#!/usr/bin/env bash
# Perf validation for the cleanup / restructure work (CLEANUP_LOG.md).
#
#   eval-tp/cleanup_validate.sh <tag> [mode ...]          (default modes: tight-half loose-half)
#   BASE=00_base eval-tp/cleanup_validate.sh 05_worker nutanix-600-800
#   CO=0 eval-tp/cleanup_validate.sh 00_base_inf tight-half loose-half   (inference-only)
#
# Runs the Qwen3-14B TP=2 co-serving timeline replay for each mode, sequentially,
# into eval-tp/output/cleanup/<tag>/, then prints compare_runs.py against
# eval-tp/output/cleanup/$BASE (default 00_base) when that run exists.
set -u
R=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-/mnt/storage/conda/envs/dserve-vllm/bin/python}
TAG=${1:?usage: cleanup_validate.sh <tag> [mode ...]}; shift
MODES=("$@"); [ ${#MODES[@]} -eq 0 ] && MODES=(tight-half loose-half)
BASE=${BASE:-00_base}
CO=${CO:-1}
FAMILY=${FAMILY:-qwen3-14b}
OUT=$R/eval-tp/output/cleanup/$TAG
mkdir -p "$OUT"
COFLAG=(); [ "$CO" = 1 ] && COFLAG=(--co --step-trace)
for m in "${MODES[@]}"; do
  echo "[validate] $TAG / $m  ($(date +%T))"
  "$PY" "$R/eval-tp/auto_benchmark_tp.py" --family "$FAMILY" --tp 2 "--$m" \
      "${COFLAG[@]}" --kill-stale --out-dir "$OUT" > "$OUT/bench_$m.stdout" 2>&1
  echo "[validate] $TAG / $m  rc=$?  ($(date +%T))"
done
[ "$CO" = 1 ] || exit 0
for m in "${MODES[@]}"; do
  sfx=$(ls "$OUT" | sed -n "s/^timeline_results\(.*_$m\)\.csv$/\1/p" | head -1)
  [ -n "$sfx" ] || { echo "[validate] no results for $m"; continue; }
  if [ -f "$R/eval-tp/output/cleanup/$BASE/timeline_results$sfx.csv" ] && [ "$BASE" != "$TAG" ]; then
    "$PY" "$R/eval-tp/compare_runs.py" --suffix "$sfx" --a "$R/eval-tp/output/cleanup/$BASE" \
        --b "$OUT" --label-a "$BASE" --label-b "$TAG"
  else
    "$PY" "$R/eval-tp/compare_runs.py" --suffix "$sfx" --a "$OUT" --b "$OUT" \
        --label-a "$TAG" --label-b "$TAG"
  fi
done
