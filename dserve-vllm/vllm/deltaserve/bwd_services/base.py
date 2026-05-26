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
``process_activations`` (the buffer-full signal). On that signal it computes the
loss + logit gradient via the model-specific ``compute_loss_and_grad`` BEFORE
cleaning the buffers, then keeps the existing clean + sleep + done-response so
the co-serving cycle/timing is preserved. The real LoRA backward + optimizer is
a later Phase-3 slice.
"""

import math
import os
import time

import torch
import torch.nn.functional as F

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
    """Return the BackwardService subclass for a model architecture string."""
    if service_name in ("LlamaForCausalLM", "llama3", "llama"):
        from vllm.deltaserve.bwd_services.llama3 import Llama3BackwardService

        return Llama3BackwardService
    if service_name in ("OPTForCausalLM", "opt"):
        from vllm.deltaserve.bwd_services.opt import OPTBackwardService

        return OPTBackwardService
    raise NotImplementedError(
        f"[deltaserve] no backward service for {service_name!r}; supported: "
        f"LlamaForCausalLM (llama3), OPTForCausalLM (opt-125m)"
    )


def service_main(conn, mps_percentage: int, device_index: int,
                 service_name: str, gpu_grant=None) -> None:
    """Child-process entry point. Runs in the spawned backward process.

    `gpu_grant` is the shared mp.Event for the `_maybe_pause` GPU-yield contract
    (SET = may run, CLEARED = yield to an inference prefill)."""
    mark_backward_process()  # make this process's dprint output purple
    if torch.cuda.is_available():
        torch.cuda.set_device(device_index)
    svc = get_service(service_name)(device_index)
    svc._gpu_grant = gpu_grant
    svc.run(conn)


class BackwardService:
    """Base class: the model-agnostic backward recv/dispatch loop."""

    def __init__(self, device_index: int) -> None:
        self.device_index = int(device_index)
        # Hold onto shared weights/activations so the IPC mappings stay alive.
        self.shared: dict = {"base": None, "ft": None, "meta": {}}
        self.activations: dict | None = None
        # dLoss/dLogits from the most recent backward cycle — stashed for the
        # next slice (LoRA backward + optimizer). Not piped to the parent.
        self.last_logit_grad: torch.Tensor | None = None
        # Run verify_activations each cycle (set from share_activations print_hash).
        self._verify = False
        # Current FT epoch (for per-epoch StepLR in trainer subclasses).
        self.current_epoch = 0
        # Cumulative valid FT tokens trained since process start (for logging).
        self._total_tokens_trained = 0
        # True for services that actually train (optimizer.step) — they own the
        # GPU for the whole pass, so the lifecycle skips the simulated sleep.
        self.is_trainer = False
        # vLLM's served LoRA stacked buffers (IPC-shared) the trainer publishes
        # the trained weights into; {"slot": int, "layers": {i: {proj: {a,b}}}}.
        self.lora_buffers: dict | None = None
        # [Phase 5] Shared GPU-yield event (set by service_main). SET = may run;
        # CLEARED = the main process is running an inference prefill → yield.
        self._gpu_grant = None
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
        hang the backward forever."""
        g = self._gpu_grant
        if g is not None and not g.is_set():
            g.wait(timeout=5.0)

    # -- per-model hooks ---------------------------------------------------

    def compute_loss_and_grad(self, activations: dict, sample_lens: list[int],
                              n: int):
        """Return (loss: float, n_valid: int, logit_grad: Tensor) for n rows.

        Reconstructs full logits from the saved pre-LM-head hidden states + the
        shared LM-head weight, computes next-token cross-entropy against
        ``concat_input_ids`` (shift-by-1 per sample), and the CE gradient w.r.t.
        the logits. Overridden per model (typically via ``_logit_loss_and_grad``)."""
        raise NotImplementedError

    def process_backward(self, activations: dict, sample_lens: list[int],
                         n: int, epoch: int):
        """One backward cycle. Default = loss-only (no optimizer): compute the loss
        + logit gradient and stash the latter. Trainer subclasses (llama3) override
        to run the full manual LoRA backward + optimizer/scheduler step. Returns
        (loss: float, n_valid: int)."""
        loss, n_valid, grad = self.compute_loss_and_grad(activations, sample_lens, n)
        self.last_logit_grad = grad
        return loss, n_valid

    def verify_activations(self, activations: dict, n: int) -> None:
        """Optional per-model check that the captured activations are correct.
        Default no-op; subclasses log assertions (gated on the debug flag)."""
        return None

    # -- shared loss helper ------------------------------------------------

    def _logit_loss_and_grad(self, final_hidden, ids, sample_lens: list[int]):
        """Logits = final_hidden @ lm_head.T (fp32, trimmed); per-sample shift-by-1
        next-token CE loss + CE logit gradient. Shared by all model services.

        ``final_hidden`` is the post-final-norm hidden states ([n, hidden]);
        ``ids`` the matching input ids ([n]); ``sample_lens`` the per-sample token
        counts (summing to n). Returns (loss: float, n_valid: int, grad: Tensor).
        """
        meta = self.shared["meta"]
        lm_head_key = meta.get("lm_head_key")
        base = self.shared["base"] or {}
        if lm_head_key is None or lm_head_key not in base:
            raise KeyError(
                f"LM-head weight key {lm_head_key!r} not in shared base weights")

        # Logits = final_hidden @ lm_head.T, in **fp32** — DeltaServe's precision
        # rule (SFT_service.py:264 logits.float(), :320 lm_head_weight_.float()):
        # a bf16 matmul + softmax over the ~128k vocab loses real precision and
        # corrupts the loss/grad. To avoid materializing the whole [vocab, D] LM
        # head in fp32 at once (~2 GiB for Llama-3 — and the backward process only
        # has the GPU memory the inference engine's gpu_memory_utilization left
        # free), compute the fp32 logits in vocab chunks, upcasting one weight
        # slice at a time. Exact fp32 numerics, bounded peak memory.
        weight = base[lm_head_key]                              # [vocab_pad, D]
        hidden = final_hidden.float()                          # [n, D] fp32
        vocab_size = meta.get("vocab_size")
        vocab = int(vocab_size) if vocab_size is not None else weight.shape[0]
        logits = hidden.new_empty((hidden.shape[0], vocab))    # fp32 [n, vocab]
        chunk = 16384
        for c in range(0, vocab, chunk):
            e = min(c + chunk, vocab)
            logits[:, c:e] = hidden @ weight[c:e].float().t()

        logit_scale = float(meta.get("logit_scale", 1.0) or 1.0)
        if logit_scale != 1.0:
            logits = logits * logit_scale

        ids = ids.long()                                        # [n]

        # Per-sample shift-by-1: position i predicts token i+1 within the sample.
        preds: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        start = 0
        for length in sample_lens:
            length = int(length)
            if length >= 2:
                end = start + length
                preds.append(logits[start:end - 1])             # [L-1, vocab]
                targets.append(ids[start + 1:end])              # [L-1]
            start += length

        if not preds:
            return 0.0, 0, logits.new_zeros((0, logits.shape[-1]))

        pred_cat = torch.cat(preds, dim=0)                      # [N_valid, vocab]
        tgt_cat = torch.cat(targets, dim=0)                     # [N_valid]
        n_valid = int(tgt_cat.shape[0])

        loss = F.cross_entropy(pred_cat, tgt_cat, reduction="sum") / n_valid

        # CE gradient w.r.t. logits (softmax − one-hot), normalized over tokens.
        probs = torch.softmax(pred_cat, dim=-1)                 # [N_valid, vocab]
        probs[torch.arange(n_valid, device=probs.device), tgt_cat] -= 1.0
        probs /= n_valid

        return float(loss.item()), n_valid, probs

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
        """Buffer-full signal: run the backward (loss-only or full train), then
        clean the buffers (+ sleep only for the loss-only/simulated path)."""
        n = int(msg.get("n", 0))
        sleep_s = float(msg.get("sleep_s", 2.0))
        sample_lens = msg.get("sample_lens") or []
        epoch = int(msg.get("epoch", 0))

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
            and self.activations.get("final_hidden") is not None
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
                self._total_tokens_trained += int(n_valid)
                # ``n`` is the raw batch-row count (sum of seq_lens in this
                # cycle), which matches the units of total_tokens_in_memory.
                # ``n_valid`` would undercount by one per sample (CE shift-by-1).
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

        # Clean the shared buffers. Trainers did the real GPU work already; the
        # loss-only/simulated path sleeps to make the co-serving cycle observable.
        # The single torch.cuda.synchronize() below covers BOTH the backward (via
        # the recorded events) and the zero-fill — the previous code had a sync
        # right after process_backward + another sync after the zero-loop.
        if self.activations:
            for value in self.activations.values():
                for t in (value if isinstance(value, list) else [value]):
                    t.zero_()
            torch.cuda.synchronize()

        # Emit the per-cycle log AFTER the sync so the events are queryable.
        if can_compute and loss is not None:
            if start_evt is not None and end_evt is not None:
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

        if not self.is_trainer:
            time.sleep(sleep_s)
        conn.send({"event": "activations_processed", "n": n, "loss": loss})
        #dprint("[backward] cleaned; sent done response")
