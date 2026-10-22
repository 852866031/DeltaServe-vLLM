#!/usr/bin/env bash
# Run every standalone test script (tests are plain `python` scripts, not pytest)
# and write a one-line-per-test summary.
#
#   tests/run_all.sh <out_dir> [test_name_glob]
#
# <out_dir>/summary.txt   PASS/FAIL, exit code, seconds, last result line
# <out_dir>/<test>.log    full output of each test
set -u
R=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-/mnt/storage/conda/envs/dserve-vllm/bin/python}
OUT=${1:?usage: tests/run_all.sh <out_dir> [glob]}
GLOB=${2:-test_*.py}
mkdir -p "$OUT"
: > "$OUT/summary.txt"
for t in "$R"/tests/$GLOB; do
  n=$(basename "$t" .py)
  s=$(date +%s)
  timeout 1800 "$PY" "$t" > "$OUT/$n.log" 2>&1
  rc=$?
  d=$(( $(date +%s) - s ))
  last=$(grep -v '^\s*$' "$OUT/$n.log" | tail -1 | cut -c1-150)
  st=PASS; [ $rc -ne 0 ] && st=FAIL
  printf '%-4s rc=%-3s %5ss  %-34s %s\n' "$st" "$rc" "$d" "$n" "$last" | tee -a "$OUT/summary.txt"
done
