import threading
import time
from contextlib import contextmanager
from typing import Dict, Tuple, Optional, List

import pynvml
import nvtx


class GPUProfiler:
    """
    Lightweight real-time GPU monitor + NVTX job annotator.

    - Background thread samples NVML every `interval` seconds.
    - `snapshot()` returns the latest utilization and memory.
    - NVTX ranges: start_annotation(note) -> job_id; stop_annotation(job_id).
    - Also supports `with profiler.annotate(note): ...` context manager.

    Notes:
      - gpu_util (%) is NVML SM utilization with ~500ms granularity.
      - mem_util (%) is NVML memory controller utilization.
      - mem_used_gb is the current device memory usage.
    """

    def __init__(self, interval: float = 0.2, device_index: int = 0):
        self.interval = float(interval)
        self.device_index = int(device_index)

        # NVTX session scope
        nvtx.push_range("session")

        # NVML
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)

        # Realtime state
        self._state_lock = threading.Lock()
        self._state: Dict[str, float] = {
            "ts": 0.0,
            "gpu_util": 0.0,
            "mem_util": 0.0,
            "mem_used_gb": 0.0,
        }

        # Active annotations: job_id -> (note, nvtx_range_obj)
        self.annotations: Dict[int, Tuple[str, object]] = {}
        self.annotation_lock = threading.Lock()
        self.next_job_id = 0
        self.job_id_lock = threading.Lock()

        # Background sampler
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def snapshot(self) -> Dict[str, object]:
        """Thread-safe read of the latest metrics."""
        with self._state_lock:
            return dict(self._state)
        
    def start_annotation(self, note: str) -> int:
        with self.job_id_lock:
            job_id = self.next_job_id
            self.next_job_id += 1

        tag = f"job_{job_id}:{note}"
        nvtx_range = nvtx.start_range(message=tag, color="blue")

        with self.annotation_lock:
            self.annotations[job_id] = (note, nvtx_range)
        return job_id

    def stop_annotation(self, job_id: int):
        with self.annotation_lock:
            entry = self.annotations.pop(job_id, None)
        if entry is not None:
            _, nvtx_range = entry
            nvtx.end_range(nvtx_range)
    
    def mark_annotation(self, note: str) -> int:
        with self.job_id_lock:
            job_id = self.next_job_id
            self.next_job_id += 1

        tag = f"job_{job_id}:{note}:start"
        nvtx.mark(message=tag, color="blue")
        return job_id

    @contextmanager
    def annotate(self, note: str):
        """Usage: with profiler.annotate('prefill'): ..."""
        jid = self.start_annotation(note)
        try:
            yield jid
        finally:
            self.stop_annotation(jid)

    # ---------------- REALTIME SAMPLER ----------------
    def _run(self):
        # simple loop sampling NVML periodically
        while not self._stop_evt.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)

                # Build list of active tags (non-blocking-ish)
                with self.annotation_lock:
                    active = [f"job_{jid}:{note}" for jid, (note, _) in self.annotations.items()]

                now = time.time()
                with self._state_lock:
                    self._state = {
                        "ts": now,
                        "gpu_util": float(util.gpu),            # %
                        "mem_util": float(util.memory),         # %
                        "mem_used_gb": float(mem.used) / (1024 ** 3),
                        "active_annotations": active,           # optional, for observability
                    }
            except Exception as e:
                # Keep sampler resilient; you can also log this
                with self._state_lock:
                    self._state["error"] = str(e)

            # Sleep last; if interrupted, we exit fast
            self._stop_evt.wait(self.interval)

    