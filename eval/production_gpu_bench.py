#!/usr/bin/env python3
"""production_gpu_bench.py — run auto_benchmark (inference-only, --co OFF) while
sampling GPU utilization, so we can recreate the "Production Inference Workload
Trace" + "GPU Utilization Under Production Workload" figure.

It does two things at once:

  1. Spawns ``eval/auto_benchmark.py`` as a subprocess **without ``--co``**
     (pure inference replay of the timeline). All extra CLI args are forwarded
     verbatim, and ``--real-timestamp`` is added so the results CSV + bench_meta
     carry a wall clock the plotter can anchor to.
  2. Samples whole-device GPU utilization (+ memory) on a background thread via
     NVML (``pynvml``; falls back to ``nvidia-smi``), timestamping each sample
     with ``datetime.now().isoformat(...)`` — the SAME clock/format as
     ``bench_meta``'s ``t_first_wall_iso`` — so the plotter aligns the GPU trace
     to the benchmark's recording t=0.

Everything this script generates lands in ``eval/production_trace/``:
  - gpu_util<suffix>.csv     (columns: ``timestamp,util_pct,mem_used_mb``)
  - bench_meta<suffix>.json  (copied from auto_benchmark's eval/output/ so the
                              plotter's inputs are self-contained)
  - timeline_results<suffix>.csv (copied, for reference)
(auto_benchmark still writes its originals under eval/output/.)

Then plot with ``eval/plot_production.py``.

Run with the project env's python (needs auto_benchmark's deps; pynvml ships
with vLLM):

  /mnt/storage/conda/envs/dserve-vllm/bin/python eval/production_gpu_bench.py --nutanix
  python eval/production_gpu_bench.py --nutanix --timeline-gpu 5090
  python eval/production_gpu_bench.py --loose --gpu-index 0 --interval-ms 250
"""
import argparse
import csv
import datetime
import os
import shutil
import subprocess
import sys
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
AUTO_OUTPUT_DIR = os.path.join(_HERE, "output")          # auto_benchmark's own dir
PROD_DIR = os.path.join(_HERE, "production_trace")        # everything we generate
AUTO_BENCH = os.path.join(_HERE, "auto_benchmark.py")

# Mode flags auto_benchmark understands → used only to name the output files so
# they line up with timeline_results<suffix>.csv (suffix = "_<mode>" with no --co).
_MODE_FLAGS = ("--nutanix", "--loose", "--tight")


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="milliseconds")


class GpuSampler(threading.Thread):
    """Background whole-device GPU utilization sampler. NVML primary, nvidia-smi
    fallback. Appends (timestamp_iso, util_pct, mem_used_mb) at a fixed rate."""

    def __init__(self, gpu_index: int, interval_s: float):
        super().__init__(daemon=True)
        self.gpu_index = int(gpu_index)
        self.interval_s = float(interval_s)
        self._stop_evt = threading.Event()
        self.rows: list[tuple[str, float, float]] = []
        self._backend = None
        self._nvml = None
        self._handle = None

    def _init_backend(self) -> str:
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            return "pynvml"
        except Exception as e:  # noqa: BLE001
            print(f"[prod-bench] pynvml unavailable ({e}); using nvidia-smi",
                  flush=True)
            return "nvidia-smi"

    def _sample_nvml(self) -> tuple[float, float]:
        u = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
        m = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
        return float(u.gpu), float(m.used) / (1024.0 * 1024.0)

    def _sample_smi(self) -> tuple[float, float]:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits",
             "-i", str(self.gpu_index)],
            text=True).strip().splitlines()[0]
        util_s, mem_s = (p.strip() for p in out.split(","))
        return float(util_s), float(mem_s)

    def run(self) -> None:
        self._backend = self._init_backend()
        sample = self._sample_nvml if self._backend == "pynvml" else self._sample_smi
        # Prime once so the first real sample isn't a cold reading.
        try:
            sample()
        except Exception:  # noqa: BLE001
            pass
        while not self._stop_evt.is_set():
            ts = _now_iso()
            try:
                util, mem = sample()
                self.rows.append((ts, util, mem))
            except Exception as e:  # noqa: BLE001
                # Don't let a transient query error kill the sampler.
                print(f"[prod-bench] sample error: {e}", flush=True)
            self._stop_evt.wait(self.interval_s)
        if self._backend == "pynvml" and self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass

    def stop(self) -> None:
        self._stop_evt.set()


def _detect_mode(forwarded: list[str]) -> str | None:
    for f in forwarded:
        if f in _MODE_FLAGS:
            return f.lstrip("-")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu-index", type=int, default=0,
                    help="GPU device index to sample. Default 0.")
    ap.add_argument("--interval-ms", type=float, default=250.0,
                    help="GPU sampling interval in ms. Default 250.")
    ap.add_argument("--gpu-util-out", default=None,
                    help="GPU-util CSV path. Default: "
                         "eval/production_trace/gpu_util<suffix>.csv.")
    ap.add_argument("--python", default=sys.executable,
                    help="Interpreter for the auto_benchmark subprocess. "
                         "Default: this interpreter (run this script with the "
                         "project env's python).")
    # Everything else is forwarded to auto_benchmark verbatim.
    args, forwarded = ap.parse_known_args()

    if "--co" in forwarded:
        ap.error("--co must NOT be passed: this harness measures the "
                 "inference-only (co-off) production workload. Remove --co.")

    # Default to the nutanix trace (matches the reference figure) if no mode flag.
    if _detect_mode(forwarded) is None:
        forwarded = ["--nutanix"] + forwarded
        print("[prod-bench] no mode flag given — defaulting to --nutanix",
              flush=True)
    mode = _detect_mode(forwarded)
    suffix = f"_{mode}" if mode else ""

    # Default the timeline GPU dir to 5090 if the caller didn't choose one
    # (handles both "--timeline-gpu 5090" and "--timeline-gpu=5090").
    if not any(a == "--timeline-gpu" or a.startswith("--timeline-gpu=")
               for a in forwarded):
        forwarded = forwarded + ["--timeline-gpu", "5090"]
        print("[prod-bench] no --timeline-gpu given — defaulting to 5090",
              flush=True)

    # auto_benchmark writes wall-clock timestamps + bench_meta we anchor to.
    if "--real-timestamp" not in forwarded:
        forwarded = forwarded + ["--real-timestamp"]

    os.makedirs(PROD_DIR, exist_ok=True)
    gpu_out = args.gpu_util_out or os.path.join(PROD_DIR, f"gpu_util{suffix}.csv")

    cmd = [args.python, AUTO_BENCH] + forwarded
    print(f"[prod-bench] sampling GPU {args.gpu_index} every "
          f"{args.interval_ms:.0f} ms → {gpu_out}", flush=True)
    print(f"[prod-bench] launching: {' '.join(cmd)}", flush=True)

    sampler = GpuSampler(args.gpu_index, args.interval_ms / 1000.0)
    sampler.start()
    rc = 1
    try:
        rc = subprocess.call(cmd)
    except KeyboardInterrupt:
        print("[prod-bench] interrupted — stopping sampler + benchmark", flush=True)
    finally:
        sampler.stop()
        try:
            sampler.join(timeout=10.0)
        except Exception as e:  # noqa: BLE001 — never lose the samples on a join hiccup
            print(f"[prod-bench] sampler join warning: {e}", flush=True)

    # Write OUTSIDE the try/finally so the collected samples are persisted even
    # if the benchmark or the join above misbehaved.
    with open(gpu_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "util_pct", "mem_used_mb"])
        w.writerows(sampler.rows)
    print(f"[prod-bench] wrote {len(sampler.rows)} GPU samples → {gpu_out}",
          flush=True)

    # Colocate auto_benchmark's anchor + results in production_trace/ so the
    # plotter's inputs are all in one dir.
    for name in (f"bench_meta{suffix}.json", f"timeline_results{suffix}.csv"):
        src = os.path.join(AUTO_OUTPUT_DIR, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(PROD_DIR, name))
            print(f"[prod-bench] copied {name} → production_trace/", flush=True)

    print(f"[prod-bench] auto_benchmark exit code: {rc}", flush=True)
    if rc != 0:
        print("[prod-bench] WARNING: benchmark returned non-zero — the GPU CSV "
              "may be incomplete.", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
