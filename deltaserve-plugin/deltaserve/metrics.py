# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""System metrics for a co-serving run (``telemetry.metrics_dir``).

Off unless ``metrics_dir`` is set (one ``is None`` check per hook). When on,
the EngineCore records, with a few dict updates per step and every file
write on a daemon thread:

``requests.csv``   one row per finished inference request: TTFT, inter-token
                   gaps (mean / max), end-to-end latency, prompt / output
                   tokens. Times are taken in the EngineCore (arrival = the
                   frontend's stamp, token = the step that produced it), so
                   they exclude HTTP / detokenization time.
``backward.csv``   one row per LoRA backward cycle: GPU time (CUDA events),
                   wall time (incl. pauses while inference had the GPU),
                   tokens, samples, loss, the child's GPU memory.
``summary.jsonl``  one JSON line every ``metrics_interval_s``: over that
                   window — TTFT (of the requests whose first token came out
                   in it) / TBT percentiles + SLO attainment,
                   inference and FT token throughput, forward GPU time by step
                   kind (inference-only / mixed / FT-only) with the mixed
                   steps' time split by tokens, backward GPU time, and the
                   latest memory figures; plus run totals.
``memory_rank<r>.json``  written by each worker after warm-up: weights,
                   KV cache, FT activation buffers, CUDA graphs / other,
                   per GPU.

GET /system_metrics returns the current window + totals.

Forward step GPU times come from the runner's CUDA-event ring (the same
samples the SLO estimator fits on); the backward child runs on the same GPU,
time-sliced, so forward and backward GPU seconds can overlap — the shares are
of each side's own busy time, not a partition of wall time.
"""

from __future__ import annotations

import csv
import json
import math
import os
import queue
import threading
import time

_REQ_FIELDS = ("finish_unix_time", "request_id", "prompt_tokens",
               "output_tokens", "ttft_s", "tbt_mean_s", "tbt_max_s", "e2e_s",
               "finish_reason")
_BWD_FIELDS = ("unix_time", "cycle", "epoch", "tokens", "samples", "loss",
               "gpu_ms", "wall_ms", "train_tokens_per_gpu_s",
               "child_mem_alloc_gib", "child_mem_peak_gib",
               "child_mem_reserved_gib")
GIB = float(1 << 30)


def _pct(sorted_vals: list[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    i = min(len(sorted_vals) - 1, max(0, math.ceil(q / 100 * len(sorted_vals)) - 1))
    return sorted_vals[i]


class LogHistogram:
    """Fixed log-spaced buckets (2 % wide, 0.1 ms … ~1 h): O(1) insert and a
    bounded percentile cost however many values a window collects. Count,
    sum, max and the at-or-below-threshold count stay exact."""

    LO, RATIO, N = 1e-4, 1.02, 900

    def __init__(self, threshold: float | None = None) -> None:
        self.counts = [0] * (self.N + 1)
        self.n = 0
        self.total = 0.0
        self.max = 0.0
        self.threshold = threshold
        self.at_or_below = 0
        self._lg = math.log(self.RATIO)

    def add(self, x: float) -> None:
        self.n += 1
        self.total += x
        if x > self.max:
            self.max = x
        if self.threshold is not None and x <= self.threshold:
            self.at_or_below += 1
        i = 0 if x <= self.LO else min(self.N, int(math.log(x / self.LO) / self._lg) + 1)
        self.counts[i] += 1

    def pct(self, q: float) -> float | None:
        """Upper edge of the bucket holding the q-th percentile (≤ 2 % high),
        capped at the exact max."""
        if not self.n:
            return None
        need = max(1, math.ceil(q / 100 * self.n))
        run = 0
        for i, c in enumerate(self.counts):
            run += c
            if run >= need:
                return min(self.max, self.LO * self.RATIO ** i)
        return self.max

    def mean(self) -> float | None:
        return self.total / self.n if self.n else None


class _Writer:
    """Appends CSV rows / JSON lines on a daemon thread."""

    def __init__(self, root: str) -> None:
        os.makedirs(root, exist_ok=True)
        self.root = root
        self._q: queue.Queue = queue.Queue()
        self._files: dict = {}
        self._t = threading.Thread(target=self._run, name="ds-metrics",
                                   daemon=True)
        self._t.start()

    def row(self, name: str, fields: tuple, values: tuple) -> None:
        self._q.put(("csv", name, fields, values))

    def line(self, name: str, obj: dict) -> None:
        self._q.put(("json", name, None, obj))

    def _open(self, name: str, fields):
        f = self._files.get(name)
        if f is None:
            path = os.path.join(self.root, name)
            new = not os.path.exists(path) or os.path.getsize(path) == 0
            f = open(path, "a", newline="")
            if fields is not None and new:
                csv.writer(f).writerow(fields)
            self._files[name] = f
        return f

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                break
            kind, name, fields, payload = item
            try:
                f = self._open(name, fields)
                if kind == "csv":
                    csv.writer(f).writerow(payload)
                else:
                    f.write(json.dumps(payload) + "\n")
                if self._q.empty():
                    f.flush()
            except Exception:  # noqa: BLE001 — metrics never break serving
                pass
        for f in self._files.values():
            try:
                f.close()
            except Exception:  # noqa: BLE001
                pass

    def close(self, timeout: float = 5.0) -> None:
        self._q.put(None)
        self._t.join(timeout)


class _Window:
    """What one summary interval accumulates."""

    def __init__(self, t0: float, max_tbt_slo: float | None = None) -> None:
        self.t0 = t0
        self.ttft: list[float] = []        # one per first token in the window: exact
        self.gaps = LogHistogram(max_tbt_slo)   # one per token: bucketed
        self.e2e: list[float] = []
        self.requests = 0
        self.prompt_tokens = 0
        self.output_tokens = 0
        # forward steps: kind -> [count, gpu_s]
        self.steps = {"inference_only": [0, 0.0], "mixed": [0, 0.0],
                      "ft_only": [0, 0.0]}
        self.mixed_inf_gpu_s = 0.0      # mixed-step time split by tokens
        self.mixed_ft_gpu_s = 0.0
        self.ft_fwd_tokens = 0
        self.inf_prefill_tokens = 0
        self.backwards = 0
        self.bwd_gpu_s = 0.0
        self.bwd_wall_s = 0.0
        self.bwd_tokens = 0
        self.losses: list[float] = []


class MetricsRecorder:
    """EngineCore-side recorder (owned by the FinetuneScheduler)."""

    def __init__(self, root: str, interval_s: float = 10.0,
                 ttft_slo: float | None = None, max_tbt_slo: float | None = None,
                 clock=time.time, start_ticker: bool = True) -> None:
        self.root = root
        self.interval_s = max(0.5, float(interval_s))
        self.ttft_slo = ttft_slo
        self.max_tbt_slo = max_tbt_slo
        self._clock = clock
        self._w = _Writer(root)
        now = clock()
        self._win = _Window(now, max_tbt_slo)
        self._t_start = now
        self._live: dict[str, list] = {}   # req_id -> [arrival, first, last, n, gap_sum, gap_max, prompt]
        self._last: dict | None = None
        self._child_mem: dict = {}
        self._cycle = 0
        self.enabled = True
        # run totals
        # requests = finished; first_tokens = requests that produced one (the
        # TTFT population); ttft_slo_met counts against first_tokens.
        self.tot = {"requests": 0, "first_tokens": 0, "prompt_tokens": 0,
                    "output_tokens": 0, "ttft_slo_met": 0, "backwards": 0, "trained_tokens": 0,
                    "ft_fwd_tokens": 0, "fwd_gpu_s": 0.0, "bwd_gpu_s": 0.0}
        # The window is touched by the engine loop and by the ticker that
        # closes windows while the engine idles (no steps → no flush otherwise).
        self._lock = threading.Lock()
        self._closed = False
        # Windows start when serving does (``start_serving``, after the launch
        # profiling pass); until then the ticker writes nothing.
        self._serving = False
        self._ticker = None
        if start_ticker:
            self._ticker = threading.Thread(target=self._tick, name="ds-metrics-tick",
                                            daemon=True)
            self._ticker.start()

    def start_serving(self) -> None:
        """Serving begins: restart the clock (the profiling pass is over)."""
        with self._lock:
            now = self._clock()
            self._win = _Window(now, self.max_tbt_slo)
            self._t_start = now
            self._serving = True

    def _tick(self) -> None:
        while not self._closed:
            time.sleep(min(1.0, self.interval_s / 4))
            if not self._serving:
                continue
            try:
                self.maybe_flush()
            except Exception:  # noqa: BLE001
                pass

    # -- inference requests ---------------------------------------------------

    def on_outputs(self, outputs, requests: dict, now: float | None = None) -> None:
        """``outputs``: the dict super().update_from_output returned
        (client index -> EngineCoreOutputs); ``requests``: scheduler.requests
        (finished ones may already be gone)."""
        if not outputs:
            return
        with self._lock:
            self._on_outputs(outputs, requests,
                             self._clock() if now is None else now)

    def _on_outputs(self, outputs, requests, now) -> None:
        live = self._live
        for eco in outputs.values():
            for o in getattr(eco, "outputs", ()) or ():
                n_new = len(o.new_token_ids or ())
                st = live.get(o.request_id)
                if st is None:
                    if not n_new:
                        continue
                    req = requests.get(o.request_id)
                    arrival = getattr(req, "arrival_time", None) or now
                    prompt = int(getattr(req, "num_prompt_tokens", 0) or 0)
                    st = live[o.request_id] = [arrival, now, now, 0, 0.0, 0.0, prompt]
                    st[3] = n_new
                    self._first_token(max(0.0, now - arrival))
                elif n_new:
                    gap = now - st[2]
                    st[2] = now
                    st[3] += n_new
                    st[4] += gap
                    if gap > st[5]:
                        st[5] = gap
                    self._win.gaps.add(gap)
                if o.finish_reason is not None:
                    self._finish(o.request_id, live.pop(o.request_id),
                                 str(getattr(o.finish_reason, "name", o.finish_reason)),
                                 now)

    def _first_token(self, ttft: float) -> None:
        """A request's TTFT counts in the window where its first token came
        out (not where the request finished, which can be many windows
        later) — so the live view and each window show TTFT as it happens."""
        self._win.ttft.append(ttft)
        t = self.tot
        t["first_tokens"] += 1
        if self.ttft_slo and ttft <= self.ttft_slo:
            t["ttft_slo_met"] += 1

    def _finish(self, rid, st, reason, now) -> None:
        arrival, first, last, n, gap_sum, gap_max, prompt = st
        ttft = max(0.0, first - arrival)
        e2e = max(0.0, last - arrival)
        n_gaps = max(0, n - 1)
        tbt_mean = gap_sum / n_gaps if n_gaps else None
        w = self._win
        w.requests += 1
        w.e2e.append(e2e)
        w.prompt_tokens += prompt
        w.output_tokens += n
        t = self.tot
        t["requests"] += 1
        t["prompt_tokens"] += prompt
        t["output_tokens"] += n
        self._w.row("requests.csv", _REQ_FIELDS, (
            round(now, 3), rid, prompt, n, round(ttft, 5),
            "" if tbt_mean is None else round(tbt_mean, 5),
            round(gap_max, 5) if n_gaps else "", round(e2e, 5), reason))

    # -- forward steps --------------------------------------------------------

    def on_step(self, feats, gpu_s: float) -> None:
        """One timed forward step (the runner's CUDA-event sample)."""
        if gpu_s <= 0:
            return
        with self._lock:
            self._on_step(feats, gpu_s)

    def _on_step(self, feats, gpu_s: float) -> None:
        t_in, t_ft, b_d = float(feats.t_in), float(feats.t_ft), int(feats.b_d)
        inf_prefill = max(0.0, t_in - t_ft)
        w = self._win
        if t_ft <= 0:
            kind = "inference_only"
        elif inf_prefill > 0 or b_d > 0:
            kind = "mixed"
            inf_tok = inf_prefill + b_d
            share = inf_tok / (inf_tok + t_ft)
            w.mixed_inf_gpu_s += gpu_s * share
            w.mixed_ft_gpu_s += gpu_s * (1 - share)
        else:
            kind = "ft_only"
        if kind == "inference_only" and t_in <= 0 and b_d == 0:
            return                                   # empty step
        s = w.steps[kind]
        s[0] += 1
        s[1] += gpu_s
        w.ft_fwd_tokens += int(t_ft)
        w.inf_prefill_tokens += int(inf_prefill)
        self.tot["ft_fwd_tokens"] += int(t_ft)
        self.tot["fwd_gpu_s"] += gpu_s

    # -- backward cycles ------------------------------------------------------

    def on_backward(self, resp: dict, epoch: int = 0) -> None:
        if not isinstance(resp, dict) or resp.get("event") != "activations_processed":
            return
        with self._lock:
            self._on_backward(resp, epoch)

    def _on_backward(self, resp: dict, epoch: int) -> None:
        self._cycle += 1
        n = int(resp.get("n", 0) or 0)
        gpu_ms = resp.get("gpu_ms")
        wall_ms = resp.get("wall_ms")
        loss = resp.get("loss")
        mem = resp.get("mem") or {}
        if mem:
            self._child_mem = mem
        w = self._win
        w.backwards += 1
        w.bwd_tokens += n
        if gpu_ms is not None:
            w.bwd_gpu_s += gpu_ms / 1000.0
            self.tot["bwd_gpu_s"] += gpu_ms / 1000.0
        if wall_ms is not None:
            w.bwd_wall_s += wall_ms / 1000.0
        if loss is not None:
            try:
                w.losses.append(float(loss))
            except (TypeError, ValueError):
                pass
        self.tot["backwards"] += 1
        self.tot["trained_tokens"] += n

        def r(x, k=3):
            return "" if x is None else round(float(x), k)
        self._w.row("backward.csv", _BWD_FIELDS, (
            round(self._clock(), 3), self._cycle, int(epoch), n,
            int(resp.get("samples", 0) or 0), r(loss, 6), r(gpu_ms), r(wall_ms),
            r(n / (gpu_ms / 1000.0), 1) if gpu_ms else "",
            r(mem.get("alloc", 0) / GIB) if mem else "",
            r(mem.get("peak", 0) / GIB) if mem else "",
            r(mem.get("reserved", 0) / GIB) if mem else ""))

    # -- summaries ------------------------------------------------------------

    def memory(self) -> dict:
        """Worker files (per rank, written after warm-up) + the child's latest."""
        out = {}
        try:
            for name in sorted(os.listdir(self.root)):
                if name.startswith("memory_rank") and name.endswith(".json"):
                    with open(os.path.join(self.root, name)) as f:
                        out[name[len("memory_"):-len(".json")]] = json.load(f)
        except OSError:
            pass
        if self._child_mem:
            out["backward_child_rank0_gib"] = {
                k: round(v / GIB, 3) for k, v in self._child_mem.items()}
        return out

    def snapshot(self, now: float | None = None) -> dict:
        now = self._clock() if now is None else now
        w = self._win
        dt = max(1e-9, now - w.t0)
        ttft = sorted(w.ttft)
        gaps = w.gaps
        fwd_gpu = sum(s[1] for s in w.steps.values())
        inf_gpu = w.steps["inference_only"][1] + w.mixed_inf_gpu_s
        ft_gpu = w.steps["ft_only"][1] + w.mixed_ft_gpu_s

        def rd(x, k=4):
            return None if x is None else round(x, k)
        slo = None
        if self.ttft_slo and ttft:
            slo = round(sum(1 for x in ttft if x <= self.ttft_slo) / len(ttft), 4)
        tbt_slo = None
        if self.max_tbt_slo and gaps.n:
            tbt_slo = round(gaps.at_or_below / gaps.n, 4)
        return {
            "unix_time": round(now, 3),
            "window_s": round(dt, 3),
            "inference": {
                "requests_finished": w.requests,
                "first_tokens": len(ttft),       # TTFTs below: requests whose first token came out in this window
                "ttft_s": {"p50": rd(_pct(ttft, 50)), "p90": rd(_pct(ttft, 90)),
                           "p99": rd(_pct(ttft, 99)), "max": rd(ttft[-1] if ttft else None),
                           "mean": rd(sum(ttft) / len(ttft) if ttft else None)},
                # inter-token gaps; percentiles from 2 %-wide buckets
                "tbt_s": {"p50": rd(gaps.pct(50)), "p90": rd(gaps.pct(90)),
                          "p99": rd(gaps.pct(99)), "max": rd(gaps.max if gaps.n else None),
                          "mean": rd(gaps.mean()), "tokens": gaps.n},
                "ttft_slo": self.ttft_slo, "ttft_slo_attainment": slo,
                "max_tbt_slo": self.max_tbt_slo, "tbt_slo_attainment": tbt_slo,
                "prompt_tokens_per_s": round(w.prompt_tokens / dt, 1),
                "output_tokens_per_s": round(w.output_tokens / dt, 1),
                "in_flight": len(self._live),
            },
            "finetune": {
                "backwards": w.backwards,
                "trained_tokens_per_s": round(w.bwd_tokens / dt, 1),
                "ft_forward_tokens_per_s": round(w.ft_fwd_tokens / dt, 1),
                "backward_gpu_ms_mean": rd(w.bwd_gpu_s * 1000 / w.backwards
                                           if w.backwards else None, 2),
                "backward_wall_ms_mean": rd(w.bwd_wall_s * 1000 / w.backwards
                                            if w.backwards else None, 2),
                "loss_mean": rd(sum(w.losses) / len(w.losses) if w.losses else None, 6),
            },
            "gpu_time_s": {
                "forward": {k: {"steps": v[0], "gpu_s": round(v[1], 4)}
                            for k, v in w.steps.items()},
                "forward_total": round(fwd_gpu, 4),
                "forward_inference_share": rd(inf_gpu / fwd_gpu if fwd_gpu else None),
                "forward_ft_share": rd(ft_gpu / fwd_gpu if fwd_gpu else None),
                "backward": round(w.bwd_gpu_s, 4),
                "forward_busy_fraction": rd(fwd_gpu / dt),
                "backward_busy_fraction": rd(w.bwd_gpu_s / dt),
            },
            "memory": self.memory(),
            "totals": {**{k: (round(v, 3) if isinstance(v, float) else v)
                          for k, v in self.tot.items()},
                       "uptime_s": round(now - self._t_start, 1)},
        }

    def maybe_flush(self, now: float | None = None, force: bool = False) -> None:
        now = self._clock() if now is None else now
        if not force and now - self._win.t0 < self.interval_s:
            return                                   # the common case: no lock
        with self._lock:
            if not force and now - self._win.t0 < self.interval_s:
                return
            snap = self.snapshot(now)
            self._last = snap
            self._win = _Window(now, self.max_tbt_slo)
        self._w.line("summary.jsonl", snap)

    def current(self) -> dict:
        """GET /system_metrics: the last full window and the one in progress."""
        with self._lock:
            cur = self.snapshot()
        return {"metrics_dir": self.root, "interval_s": self.interval_s,
                "last_window": self._last, "current_window": cur}

    def final_flush(self) -> None:
        """At the first sign of shutdown (vLLM's default stop kills the
        EngineCore right after): write the window in progress synchronously,
        not through the writer thread."""
        if self._closed:
            return
        with self._lock:
            snap = self.snapshot()
            self._last = snap
            self._win = _Window(self._clock(), self.max_tbt_slo)
        try:
            with open(os.path.join(self.root, "summary.jsonl"), "a") as f:
                f.write(json.dumps({**snap, "final": True}) + "\n")
        except OSError:
            pass

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.maybe_flush(force=True)
        finally:
            self._closed = True
            self._w.close()


def write_memory_snapshot(root: str, rank: int, model_runner, accumulator=None,
                          extra: dict | None = None) -> dict:
    """Worker side, after warm-up: this rank's GPU memory by owner."""
    import torch

    def nbytes(ts):
        return sum(t.numel() * t.element_size() for t in ts)
    model = model_runner.get_model()
    weights = nbytes(p for p in model.parameters())
    # Per-layer KV views can share one allocation: count each storage once.
    seen, kv = set(), 0
    stack = list(getattr(model_runner, "kv_caches", None) or [])
    while stack:
        t = stack.pop()
        if isinstance(t, (list, tuple)):
            stack.extend(t)
            continue
        st = t.untyped_storage()
        if st.data_ptr() not in seen:
            seen.add(st.data_ptr())
            kv += st.nbytes()
    acts = 0
    if accumulator is not None:
        for v in accumulator.buffers.values():
            acts += nbytes(v if isinstance(v, list) else [v])
    alloc = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    free, total = torch.cuda.mem_get_info()
    snap = {
        "unix_time": round(time.time(), 3),
        "gib": {k: round(v / GIB, 3) for k, v in {
            "model_weights": weights,
            "kv_cache": kv,
            "ft_activation_buffers": acts,
            "other_allocated (lm_head copy, LoRA slots, temporaries)":
                max(0, alloc - weights - kv - acts),
            "torch_reserved_not_allocated (incl. CUDA graph pools)":
                max(0, reserved - alloc),
            "inference_process_reserved": reserved,
            "device_total": total,
            "device_used_all_processes": total - free,
            **(extra or {}),
        }.items()},
    }
    os.makedirs(root, exist_ok=True)
    tmp = os.path.join(root, f".memory_rank{rank}.json.tmp")
    with open(tmp, "w") as f:
        json.dump(snap, f, indent=2)
    os.replace(tmp, os.path.join(root, f"memory_rank{rank}.json"))
    return snap
