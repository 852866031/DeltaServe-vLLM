#!/usr/bin/env bash
# A user-level NVIDIA MPS control daemon for co-serving (no root needed).
#
#   scripts/mps.sh start      start the daemon, print the two exports
#   scripts/mps.sh status
#   scripts/mps.sh stop
#
# Under MPS the inference worker and the backward child run concurrently
# instead of being time-sliced by the driver (a decode step overlapping a
# backward: 1.87x -> 1.2x a clean one on Qwen3-14B TP=2). Every process that
# should use it — the server and its launcher — needs the same
# CUDA_MPS_PIPE_DIRECTORY; `eval "$(scripts/mps.sh env)"` sets it in a shell.
# The pipe directory is a unix socket path: keep it short (< ~100 characters).
set -u
DIR=${DSERVE_MPS_DIR:-/tmp/dserve-mps-$(id -u)}
export CUDA_MPS_PIPE_DIRECTORY=$DIR/pipe CUDA_MPS_LOG_DIRECTORY=$DIR/log
running() { pgrep -u "$(id -u)" -x nvidia-cuda-mps > /dev/null; }   # comm is cut to 15 characters
case "${1:-status}" in
  start)
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    running || setsid nohup nvidia-cuda-mps-control -d > "$DIR/start.out" 2>&1
    sleep 1
    running || { echo "MPS daemon did not start (see $DIR/start.out)" >&2; exit 1; }
    echo "export CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY"
    echo "export CUDA_MPS_LOG_DIRECTORY=$CUDA_MPS_LOG_DIRECTORY" ;;
  env)
    echo "export CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY"
    echo "export CUDA_MPS_LOG_DIRECTORY=$CUDA_MPS_LOG_DIRECTORY" ;;
  stop)
    running && echo quit | nvidia-cuda-mps-control
    sleep 1; running && { echo "still running" >&2; exit 1; }; echo "stopped" ;;
  status)
    if running; then echo "running ($CUDA_MPS_PIPE_DIRECTORY)"; else echo "not running"; fi ;;
  *) echo "usage: $0 start|env|status|stop" >&2; exit 2 ;;
esac
