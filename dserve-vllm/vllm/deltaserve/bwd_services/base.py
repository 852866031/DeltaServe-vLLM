# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Model-agnostic backward-service loop + child entry point.

``service_main`` is the spawned child process's target (set by
``BackwardProcess.start()``). It marks the process, binds the CUDA device,
constructs the per-model ``BackwardService`` selected by ``service_name``, and
runs its recv/dispatch loop.

``BackwardService`` owns the pipe protocol shared by all models: the ``ready``
handshake, ping/shutdown, the CUDA-IPC ``share_weights`` / ``share_activations``
mappings (held to keep the IPC views alive), the hash debug commands, and
``process_activations`` (the buffer-full signal). On that signal it runs the
subclass's ``process_backward`` (the LoRA backward + optimizer step, see
``common/trainer.py``) BEFORE cleaning the buffers, then sends the done-response
that reopens FT admission.
"""

import math
import os
import time

import torch

from vllm.deltaserve import dprint, mark_backward_process
from vllm.deltaserve.backward_process import (
    _MAX_CONNECTIONS_ENV,
    _MPS_PERCENTAGE_ENV,
    _checksum,
    _summarize_weights,
    activation_hash_report,
    print_hash_report,
    weight_hash_report,
)


def get_service(service_name: str):
    """Return the BackwardService subclass for a model architecture string
    (thin shim over ``bwd_services.registry``; imported lazily because the
    family modules import this module)."""
    from vllm.deltaserve.bwd_services.registry import get_service as _get

    return _get(service_name)


def _arm_parent_death_signal() -> None:
    """[DeltaServe] Phase 7 / M1: SIGKILL this backward child when its parent
    worker dies.

    The child is spawned ``daemon=True``, so Python reaps it on a *clean* worker
    exit — but SIGKILL (which is how an orphaned or wedged worker gets cleaned up)
    bypasses Python's exit handlers entirely, stranding this process with its
    share of GPU memory (~650 MB/rank) and, under TP, a rank of the backward
    NCCL group. Ask the kernel to kill us with the parent instead. Linux-only;
    a no-op elsewhere."""
    import ctypes
    import signal
    import sys

    if not sys.platform.startswith("linux"):
        return
    try:
        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(
            PR_SET_PDEATHSIG, signal.SIGKILL)
    except Exception as e:  # noqa: BLE001 — never block the child on this
        dprint(f"[backward] could not arm PR_SET_PDEATHSIG: {e}")
        return
    if os.getppid() == 1:
        dprint("[backward] parent worker already gone at startup; exiting")
        os._exit(1)


def service_main(conn, mps_percentage: int, device_index: int,
                 service_name: str, gpu_grant=None) -> None:
    """Child-process entry point. Runs in the spawned backward process.

    `gpu_grant` is the shared mp.Event for the `_maybe_pause` GPU-yield contract
    (SET = may run, CLEARED = yield to an inference prefill)."""
    mark_backward_process()  # make this process's dprint output purple
    _arm_parent_death_signal()
    if torch.cuda.is_available():
        torch.cuda.set_device(device_index)
    svc = get_service(service_name)(device_index)
    svc._gpu_grant = gpu_grant
    try:
        svc.run(conn)
    except KeyboardInterrupt:
        # Server teardown forwards SIGINT to the process group while a cycle
        # may be in flight; exit quietly instead of printing a traceback.
        dprint("[backward] child interrupted by shutdown; exiting")


class BackwardService:
    """Base class: the model-agnostic backward recv/dispatch loop."""

    def __init__(self, device_index: int) -> None:
        self.device_index = int(device_index)
        # Hold onto shared weights/activations so the IPC mappings stay alive.
        self.shared: dict = {"base": None, "ft": None, "meta": {}}
        self.activations: dict | None = None
        # Run verify_activations each cycle (set from share_activations print_hash).
        self._verify = False
        # Current FT epoch (for per-epoch StepLR in trainer subclasses).
        self.current_epoch = 0
        # Cumulative valid FT tokens trained since process start (for logging).
        self._total_tokens_trained = 0
        # vLLM's served LoRA stacked buffers (IPC-shared) the trainer publishes
        # the trained weights into; {"slot": int, "layers": {i: {proj: {a,b}}}}.
        self.lora_buffers: dict | None = None
        # [Phase 5] Shared GPU-yield event (set by service_main). SET = may run;
        # CLEARED = the main process is running an inference prefill → yield.
        self._gpu_grant = None
        # [pause] Run-ahead bound (see FinetuneConfig.backward_run_ahead_
        # boundaries): a ring of CUDA events, one per _maybe_pause boundary;
        # resolved from meta on first use (-1 = not yet).
        self._run_ahead: int = -1
        self._boundary_events: list = []
        self._boundary_pos: int = 0
        # [pause] Under TP the pause must be rank-symmetric (set by the
        # trainer to the children's gloo group): if one child blocks at
        # boundary i while the other has already issued the next residual
        # all-reduce, that NCCL kernel spins on the GPU waiting for its peer
        # for the whole pause — measured on Qwen3-14B TP=2 as every prefill
        # of a burst running 2× slower with the child "paused".
        self._pause_group = None
        # Tag for the per-cycle one-line log: trainer subclasses set this to
        # "graph" or "eager" depending on which backward path they took.
        self._last_mode: str = "eager"
        # Per-cycle log progress meter — tokens trained in the CURRENT epoch
        # (resets to 0 when ``_handle_process_activations`` sees an epoch
        # higher than the last one). The total is set ONCE by the parent via
        # the ``set_corpus_meta`` IPC command right after FinetuningStore.load
        # completes; stays constant for the run's lifetime.
        self._epoch_processed_tokens: int = 0
        self._cur_epoch_seen: int = 0
        self._total_tokens_per_epoch: int = 0

    def _maybe_pause(self) -> None:
        """GPU-yield contract: block at a layer boundary while the main process
        is running an inference prefill (grant cleared), so prefill pre-empts the
        backward within one layer's kernels. No-op (instant) when the grant is
        set, which is the steady state. Bounded wait so a missed re-set can't
        hang the backward forever.

        The grant wait only stops *enqueueing*; the GPU still runs whatever this
        process had already queued. With graph replays the CPU launches a whole
        cycle in a few ms, so without a bound a "paused" child kept the GPU busy
        for the rest of its cycle. ``backward_run_ahead_boundaries`` (meta)
        therefore keeps the CPU at most N boundaries ahead: record an event
        here, wait for the one recorded N boundaries ago."""
        depth = self._run_ahead
        if depth < 0:
            meta = (self.shared or {}).get("meta") or {}
            depth = int(meta.get("backward_run_ahead_boundaries", 0) or 0)
            if not torch.cuda.is_available():
                depth = 0   # CPU (gloo tests): no device queue to bound
            self._run_ahead = depth
            if depth > 0:
                self._boundary_events = [
                    torch.cuda.Event() for _ in range(depth + 1)]
                self._boundary_pos = 0
        if depth > 0:
            ring = self._boundary_events
            pos = self._boundary_pos
            ring[pos].record()
            self._boundary_pos = (pos + 1) % len(ring)
            # The slot we are about to overwrite next holds the event recorded
            # ``depth`` boundaries ago — wait until the GPU has passed it.
            ring[self._boundary_pos].synchronize()
        g = self._gpu_grant
        if g is None:
            return
        grp = self._pause_group
        if grp is None:
            if not g.is_set():
                g.wait(timeout=5.0)
            return
        # Rank-symmetric (tp>1): every rank reaches every boundary, so the
        # ranks agree with a MAX all-reduce whether ANY grant is cleared and
        # block together; while blocked they poll their own grant and agree
        # with a MIN all-reduce that ALL are set again before any proceeds.
        # The loop exit depends only on the reduced value (the deadline is
        # folded into it), so no rank can leave the loop alone.
        import torch.distributed as dist
        flag = torch.tensor([0 if g.is_set() else 1], dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=grp)
        if not int(flag.item()):
            return
        deadline = time.monotonic() + 5.0
        while True:
            mine = g.wait(timeout=0.002) or time.monotonic() > deadline
            flag = torch.tensor([1 if mine else 0], dtype=torch.int32)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=grp)
            if int(flag.item()):
                return

    # -- per-model hooks ---------------------------------------------------

    def process_backward(self, activations: dict, sample_lens: list[int],
                         n: int, epoch: int):
        """One backward cycle over the first ``n`` buffered rows (LoRA backward
        + optimizer/scheduler step). Returns (loss: float, n_valid: int)."""
        raise NotImplementedError

    def verify_activations(self, activations: dict, n: int) -> None:
        """Optional per-model check that the captured activations are correct.
        Default no-op; subclasses log assertions (gated on the debug flag)."""
        return None

    # -- loop --------------------------------------------------------------

    def run(self, conn) -> None:
        pid = os.getpid()
        inherited_mps = os.environ.get(_MPS_PERCENTAGE_ENV)
        inherited_max_conn = os.environ.get(_MAX_CONNECTIONS_ENV)
        dprint(
            f"[backward] {type(self).__name__} started pid={pid} "
            f"device={self.device_index} inherited "
            f"{_MPS_PERCENTAGE_ENV}={inherited_mps} "
            f"{_MAX_CONNECTIONS_ENV}={inherited_max_conn}"
        )
        # Announce readiness, echoing what we inherited so the parent can confirm
        # the MPS partition was applied to the child only.
        conn.send({
            "event": "ready",
            "pid": pid,
            "mps_percentage": inherited_mps,
            "max_connections": inherited_max_conn,
        })
        try:
            while True:
                try:
                    msg = conn.recv()
                except EOFError:
                    # Parent closed the pipe (e.g. crashed) — exit.
                    break
                cmd = msg.get("cmd") if isinstance(msg, dict) else None
                if cmd == "shutdown":
                    dprint(f"[backward] pid={pid} shutting down")
                    conn.send({"event": "bye", "pid": pid})
                    break
                elif cmd == "ping":
                    conn.send({"event": "pong", "pid": pid,
                               "data": msg.get("data")})
                elif cmd == "share_weights":
                    self._handle_share_weights(conn, msg)
                elif cmd == "checksum":
                    which = msg.get("which", "ft")
                    conn.send({
                        "event": "checksum",
                        "which": which,
                        "value": _checksum(self.shared.get(which) or {}),
                    })
                elif cmd == "share_activations":
                    self._handle_share_activations(conn, msg)
                elif cmd == "share_lora_buffers":
                    self.lora_buffers = msg.get("payload")
                    nl = len((self.lora_buffers or {}).get("layers", {}))
                    dprint(f"[backward] received served LoRA buffers: {nl} layers "
                           f"(slot {(self.lora_buffers or {}).get('slot')})")
                    conn.send({"event": "lora_buffers_received", "num_layers": nl})
                elif cmd == "hash_activations":
                    report = activation_hash_report(self.activations or {},
                                                    msg.get("n", 0))
                    print_hash_report(report, "child")
                    conn.send({"event": "activation_hashes",
                               "hash_report": report})
                elif cmd == "set_corpus_meta":
                    # One-shot: corpus total tokens per epoch. Stored on self
                    # and used by ``_handle_process_activations`` to render
                    # the per-epoch progress meter. Fire-and-forget — no ack.
                    self._total_tokens_per_epoch = int(
                        msg.get("total_tokens_per_epoch", 0))
                    dprint(f"[backward] corpus meta: "
                           f"total_tokens_per_epoch={self._total_tokens_per_epoch}")
                elif cmd == "process_activations":
                    self._handle_process_activations(conn, msg)
                else:
                    conn.send({"event": "error", "msg": f"unknown cmd {cmd!r}"})
        finally:
            conn.close()

    # -- handlers ----------------------------------------------------------

    def _handle_share_weights(self, conn, msg) -> None:
        self.shared["base"] = msg.get("base") or {}
        self.shared["ft"] = msg.get("ft") or {}
        self.shared["meta"] = msg.get("meta") or {}
        resp = {
            "event": "weights_received",
            "base_num": len(self.shared["base"]),
            "ft_num": len(self.shared["ft"]),
            "hash_report": {},
        }
        # Only hash + print when the debug flag is on (else stay quiet).
        if msg.get("print_hash"):
            report = weight_hash_report(self.shared["base"], self.shared["ft"])
            print_hash_report(report, "child")
            resp["hash_report"] = report
            resp.update(_summarize_weights(self.shared["base"], self.shared["ft"]))
        conn.send(resp)

    def _handle_share_activations(self, conn, msg) -> None:
        self.activations = msg.get("buffers")
        self._verify = bool(msg.get("print_hash"))
        nlayers = len(self.activations.get("layer_in", []))
        if msg.get("print_hash"):
            dprint(
                f"[backward] received activation buffers: layer_in x {nlayers} "
                f"+ final_in + final_hidden + concat_input_ids"
            )
        conn.send({"event": "activations_received", "num_layers": nlayers})

    def _handle_process_activations(self, conn, msg) -> None:
        """Buffer-full signal: run the backward, then clean the buffers."""
        n = int(msg.get("n", 0))
        sample_lens = msg.get("sample_lens") or []
        epoch = int(msg.get("epoch", 0))
        # Announce the start of a backward cycle. Reports the per-cycle
        # token count + sample count + epoch, plus the current pause-
        # gate state (``gpu_grant`` SET = bwd can run; CLEAR = paused,
        # the first ``_maybe_pause`` boundary inside ``process_backward``
        # will block until something flips the event). When the trace
        # shows a long gap between this "start" line and the next
        # [backward] line, the gate state here tells you whether the
        # gap is a real backward (gate=SET) or just waiting for an
        # inference pause / throttle to release (gate=CLEAR).
        _grant_set = (self._gpu_grant.is_set()
                      if self._gpu_grant is not None else None)
        _gate = ("SET" if _grant_set
                 else ("CLEAR" if _grant_set is False else "n/a"))
        dprint(
            f"[backward] start: n={n} samples={len(sample_lens)} "
            f"epoch={epoch} gpu_grant={_gate}"
            + ("  (will block until released)"
               if _grant_set is False else "")
        )

        # Reset the per-epoch processed-tokens counter when the parent reports
        # a new epoch — must happen BEFORE we add this cycle's ``n``, so the
        # cycle's training counts toward the new epoch's progress (matches
        # the parent-side semantics: ``self.current_epoch`` reflects the
        # store's epoch AT THE TIME OF TRIGGER, so a flush-fired backward
        # for epoch N+1 carries the N+1 tag).
        if epoch > self._cur_epoch_seen:
            self._epoch_processed_tokens = 0
            self._cur_epoch_seen = epoch

        loss = None
        n_valid = 0
        start_evt = end_evt = None
        cpu_elapsed_ms: float | None = None
        # Run the backward BEFORE cleaning the buffers (clean zeros them). Skip
        # cleanly when there is nothing to compute (e.g. save_activations off leaves
        # the buffers zeroed, or weights/state not built).
        can_compute = (
            n > 0 and self.activations is not None
            and self.activations.get("concat_input_ids") is not None
            and bool(sample_lens)
        )
        if can_compute:
            try:
                if self._verify:
                    self.verify_activations(self.activations, n)
                # Time the backward with CUDA events instead of a wall-clock-
                # after-sync — this lets us defer the host sync to end-of-cycle
                # (after buffer zeroing), coalescing two syncs into one. The
                # event-derived timing is GPU-strict; the previous wall-clock
                # path included host overhead between perf_counter calls.
                if torch.cuda.is_available():
                    start_evt = torch.cuda.Event(enable_timing=True)
                    end_evt = torch.cuda.Event(enable_timing=True)
                    start_evt.record()
                    loss, n_valid = self.process_backward(
                        self.activations, sample_lens, n, epoch)
                    end_evt.record()
                else:
                    t0 = time.perf_counter()
                    loss, n_valid = self.process_backward(
                        self.activations, sample_lens, n, epoch)
                    cpu_elapsed_ms = (time.perf_counter() - t0) * 1000.0
                # FT throughput metric: total FT token-rows PROCESSED (raw
                # ``n`` = sum of this cycle's sample lengths), NOT the CE-valid
                # target count (``n_valid`` = n − num_samples from the
                # shift-by-1). ``n`` is the activation rows actually forwarded +
                # backpropped, which is the throughput number we report.
                self._total_tokens_trained += int(n)
                # ``n`` is the raw batch-row count (sum of seq_lens in this
                # cycle), which matches the units of total_tokens_in_memory.
                self._epoch_processed_tokens += int(n)
            except Exception as e:  # noqa: BLE001 — keep the cycle alive on error
                import traceback
                dprint(f"[backward] backward failed: {e}")
                traceback.print_exc()
                # Drop the events so we don't query an incomplete pair below.
                start_evt = end_evt = None
        else:
            dprint(
                f"[backward] skipped (n={n}, "
                f"no data / save_activations off)"
            )

        # Clean the shared buffers. The single torch.cuda.synchronize() below covers BOTH the backward (via
        # the recorded events) and the zero-fill — the previous code had a sync
        # right after process_backward + another sync after the zero-loop.
        if self.activations:
            for value in self.activations.values():
                for t in (value if isinstance(value, list) else [value]):
                    t.zero_()
            #torch.cuda.synchronize()

        # Emit the per-cycle log with the GPU-strict cycle time. Waiting on the
        # END event alone (not a device-wide synchronize) is enough to make the
        # pair queryable; one event-scoped wait per cycle, off the layer loop.
        if can_compute and loss is not None:
            if start_evt is not None and end_evt is not None:
                end_evt.synchronize()
                total_ms = start_evt.elapsed_time(end_evt)
            elif cpu_elapsed_ms is not None:
                total_ms = cpu_elapsed_ms
            else:
                total_ms = float("nan")
            # Normalise loss to a float for both the format string and the
            # non-finite check below — `loss` may be a 0-d tensor or a
            # Python float depending on the subclass.
            try:
                loss_f = loss.item() if hasattr(loss, "item") else float(loss)
            except Exception:
                loss_f = float("nan")
            # Per-epoch progress meter: tokens trained in THIS epoch over the
            # corpus total. ``?`` when the parent hasn't yet sent ``set_corpus_meta``
            # (shouldn't happen on the live path — scheduler sends it BEFORE
            # opening FT admission — but defensive).
            if self._total_tokens_per_epoch > 0:
                progress = (f"{self._epoch_processed_tokens}/"
                            f"{self._total_tokens_per_epoch}")
            else:
                progress = f"{self._epoch_processed_tokens}/?"
            dprint(
                f"[backward] {total_ms:.1f}ms ({self._last_mode}) "
                f"loss={loss_f:.6f} "
                f"total_trained={self._total_tokens_trained} n={n} "
                f"epoch={epoch} {progress} tokens"
            )
            # Loud warning on non-finite loss: training has diverged, the
            # published LoRA weights are now NaN-tainted, and subsequent
            # inference through this adapter will produce garbage. The cycle
            # itself continues so the user can still see follow-up backwards
            # (and the NaN pattern) — we deliberately do NOT kill the child.
            if not math.isfinite(loss_f):
                dprint(
                    f"[backward] !!! NON-FINITE LOSS ({loss_f}) at "
                    f"epoch={epoch} n={n} — training diverged; the FT "
                    f"adapter published to vLLM is now NaN-tainted and "
                    f"inference through it will return garbage. Lower the "
                    f"learning rate or disable FT to recover."
                )

        conn.send({"event": "activations_processed", "n": n, "loss": loss})
        #dprint("[backward] cleaned; sent done response")
