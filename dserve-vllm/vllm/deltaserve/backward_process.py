# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The DeltaServe backward (SFT) process — parent-side handle + shared helpers.

This module owns the *parent* side of the second GPU process DeltaServe runs
alongside inference: spawning it (with the child-only MPS partition), the pipe
protocol, and the CUDA-IPC sharing of base/FT weights + activation buffers. The
*child* side (the recv loop + the per-model SFT math) lives in
``vllm/deltaserve/bwd_services/`` — ``base.py`` holds the model-agnostic loop and
``opt.py`` etc. hold the per-model compute. ``start()`` imports the child entry
point lazily so this module can stay free of a bwd_services import cycle.

The analogue in DeltaServe is the backward service spawned from
``model_rpc.py:150-178`` (the GPU-owning process). In vLLM that process is the
``Worker`` (in-process under the single-GPU ``UniProcExecutor``).

The hashing helpers below are shared by both sides (and imported parent-side by
``gpu_worker.py`` / ``gpu_model_runner.py``) to verify the IPC-shared view.

MPS env wrapping (``CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`` /
``CUDA_DEVICE_MAX_CONNECTIONS``) is applied only around ``.start()`` so *only*
the child inherits the constrained MPS partition; the parent (inference) keeps
the full GPU. This mirrors DeltaServe exactly.
"""

import hashlib
import os
import re

# torch.multiprocessing (not plain multiprocessing) so that CUDA tensors sent
# over the pipe are reduced to CUDA-IPC handles — i.e. shared zero-copy with the
# child instead of being copied. This is the mechanism the plan flags as the
# Phase-1 #1 risk; we use it to share the FT adapter + base weights.
import torch.multiprocessing as mp

from vllm.deltaserve import dprint

_MPS_PERCENTAGE_ENV = "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
_MAX_CONNECTIONS_ENV = "CUDA_DEVICE_MAX_CONNECTIONS"


# --- weight hashing (shared by parent + child to verify the IPC-shared view) ---

def _tensor_hash(t) -> str:
    """Deterministic content hash of a tensor (fp32 bytes), dtype-independent."""
    b = t.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha1(b).hexdigest()[:16]


def _layer_index(key: str) -> int | None:
    m = re.search(r"layers?\.(\d+)\.", key)
    return int(m.group(1)) if m else None


def _first_last_by_layer(state: dict, substrings, exclude=()):
    """Return ((lo_layer, lo_key, lo_t), (hi_layer, hi_key, hi_t)) for the
    lowest/highest transformer layer whose key contains all `substrings` and
    none of `exclude`. Returns (None, None) if nothing matches."""
    items = []
    for key, tensor in state.items():
        if all(s in key for s in substrings) and not any(e in key for e in exclude):
            li = _layer_index(key)
            if li is not None:
                items.append((li, key, tensor))
    if not items:
        return None, None
    items.sort(key=lambda x: x[0])
    return items[0], items[-1]


def weight_hash_report(base_state: dict, ft_state: dict) -> dict:
    """Hash the first/last-layer FFN (base) and first/last-layer q_proj LoRA-A
    (adapter). Returns {label: (layer, name, hash)}. Same logic both sides, so
    matching hashes prove the parent and child see identical bytes.

    Base FFN: match fc1 + weight but exclude 'lora'/'bias' — vLLM wraps the
    layer under enable_lora, so the base weight is named '...fc1.base_layer.weight'.
    """
    report: dict = {}
    lo, hi = _first_last_by_layer(base_state, ("fc1", "weight"),
                                  exclude=("lora", "bias"))
    if lo:
        report["base_ffn_fc1_first"] = (lo[0], lo[1], _tensor_hash(lo[2]))
    if hi:
        report["base_ffn_fc1_last"] = (hi[0], hi[1], _tensor_hash(hi[2]))
    lo, hi = _first_last_by_layer(ft_state, ("q_proj", "lora_A"))
    if lo:
        report["ft_q_loraA_first"] = (lo[0], lo[1], _tensor_hash(lo[2]))
    if hi:
        report["ft_q_loraA_last"] = (hi[0], hi[1], _tensor_hash(hi[2]))
    return report


def activation_hash_report(buffers: dict, n: int) -> dict:
    """Hash the captured FT activation buffers (first n rows). Same logic on
    parent + child, so matching hashes prove the backward process sees the same
    bytes the worker wrote (zero-copy IPC). `buffers` holds the per-layer list
    `layer_in` plus `final_in`, `final_hidden`, and `concat_input_ids`.
    """
    report: dict = {}
    if n <= 0 or not buffers:
        return report
    for lkey in ("layer_in", "mlp_gate_up"):
        layers = buffers.get(lkey) or []
        if layers:
            report[f"{lkey}_first"] = _tensor_hash(layers[0][:n])
            report[f"{lkey}_last"] = _tensor_hash(layers[-1][:n])
    for key in ("final_in", "final_hidden", "concat_input_ids"):
        buf = buffers.get(key)
        if buf is not None:
            report[key] = _tensor_hash(buf[:n])
    return report


def print_hash_report(report: dict, who: str) -> None:
    for label, value in report.items():
        if isinstance(value, tuple):  # weight report: (layer, name, hash)
            layer, name, h = value
            dprint(f"[hash:{who}] {label:>20s}  L{layer:<2d} {h}  ({name})")
        else:  # activation report: plain hash string
            dprint(f"[hash:{who}] {label:>20s}  {value}")


def _checksum(state: dict) -> float:
    """fp64 sum over all tensor values — cheap content fingerprint."""
    total = 0.0
    for t in state.values():
        total += float(t.detach().double().sum().item())
    return total


def _summarize_weights(base: dict, ft: dict) -> dict:
    return {
        "base_num": len(base),
        "base_numel": int(sum(t.numel() for t in base.values())),
        "ft_num": len(ft),
        "ft_numel": int(sum(t.numel() for t in ft.values())),
        "ft_checksum": _checksum(ft),
    }


class BackwardProcess:
    """Parent-side handle: spawns and talks to the backward stub process."""

    def __init__(self, mps_percentage: int = 10, device_index: int = 0,
                 service_name: str = "OPTForCausalLM"):
        self.mps_percentage = int(mps_percentage)
        self.device_index = int(device_index)
        # Selects the per-model child service in bwd_services/ (e.g. the model's
        # HF architecture string "OPTForCausalLM").
        self.service_name = str(service_name)
        # spawn (not fork): CUDA-safe and matches vLLM's start method, so the
        # later CUDA-IPC buffer sharing (Step 3) works the same way.
        self._ctx = mp.get_context("spawn")
        self._proc = None
        self._conn = None
        # [Phase 5] GPU-yield event ("_maybe_pause" contract). Convention:
        # SET = the backward child may use the GPU; CLEARED = it must yield
        # (the main process is running an inference PREFILL forward). The child
        # waits on it at every layer boundary, so a prefill pre-empts the
        # backward within one layer's worth of kernels. Starts SET (the backward
        # runs by default; only prefill forwards pause it).
        self._gpu_grant = self._ctx.Event()
        self._gpu_grant.set()
        # Keep references to shared tensors alive on the parent side: CUDA IPC
        # requires the producer to keep the memory alive until the child maps it.
        self._shared_refs: list = []

    def start(self) -> dict:
        """Spawn the child with MPS env applied to the child only.

        Returns the child's 'ready' handshake dict.
        """
        if mp.current_process().daemon:
            raise RuntimeError(
                "[deltaserve] cannot spawn the backward process from a daemonic "
                "process. Under the multiproc executor (TP>1) vLLM workers are "
                "daemonic by default and Python forbids them from having "
                "children. Phase 7/M1 makes the worker NON-daemonic when "
                "enable_finetuning is set (see multiproc_executor.make_worker_"
                "process). If you hit this, that gating did not take effect — "
                "verify finetune_config.enable_finetuning reached the executor."
            )
        dprint(
            f"[backward] BackwardProcess.start on {mp.current_process().name} "
            f"(daemon={mp.current_process().daemon}) → child device_index="
            f"{self.device_index} service={self.service_name}"
        )

        # Lazy import to avoid a module-load cycle: bwd_services.base imports the
        # hashing helpers from this module.
        from vllm.deltaserve.bwd_services.base import service_main

        parent_conn, child_conn = self._ctx.Pipe()
        self._conn = parent_conn

        # --- MPS env wrapping: child-only constrained partition ---
        prev_mps = os.environ.get(_MPS_PERCENTAGE_ENV)
        prev_max_conn = os.environ.get(_MAX_CONNECTIONS_ENV)
        os.environ[_MPS_PERCENTAGE_ENV] = str(self.mps_percentage)
        os.environ[_MAX_CONNECTIONS_ENV] = "1"
        try:
            self._proc = self._ctx.Process(
                target=service_main,
                args=(child_conn, self.mps_percentage, self.device_index,
                      self.service_name, self._gpu_grant),
                name="DeltaServeBackward",
                daemon=True,
            )
            dprint(
                f"[backward] spawning child ({self.service_name}) with "
                f"{_MPS_PERCENTAGE_ENV}={self.mps_percentage} "
                f"{_MAX_CONNECTIONS_ENV}=1"
            )
            self._proc.start()
        finally:
            # Restore parent env immediately so inference keeps the full GPU.
            _restore_env(_MPS_PERCENTAGE_ENV, prev_mps)
            _restore_env(_MAX_CONNECTIONS_ENV, prev_max_conn)

        # Parent doesn't use the child's end of the pipe.
        child_conn.close()

        ready = self._conn.recv()
        dprint(f"[backward] child ready pid={self._proc.pid} {ready}")
        return ready

    def ping(self, data=None, timeout: float = 5.0) -> dict:
        """Round-trip a message to the child (used by tests / sanity checks)."""
        self._conn.send({"cmd": "ping", "data": data})
        if not self._conn.poll(timeout):
            raise TimeoutError("[deltaserve] backward ping timed out")
        return self._conn.recv()

    def share_weights(self, base_state: dict, ft_state: dict,
                      print_hash: bool = False, meta: dict | None = None,
                      timeout: float = 120.0) -> dict:
        """Share the base + FT-adapter GPU tensors with the child via CUDA IPC.

        The dicts must contain CUDA tensors. They are sent over the pipe; the
        torch.multiprocessing context reduces them to IPC handles so the child
        maps the *same* memory (zero-copy). ``meta`` carries small CPU-side
        model info the child needs to reconstruct logits (LM-head weight key,
        org vocab size, logit scale). Returns the child's summary.
        """
        # Hold references so the producer-side memory stays alive (CUDA IPC).
        self._shared_refs.extend(base_state.values())
        self._shared_refs.extend(ft_state.values())
        self._conn.send({"cmd": "share_weights", "base": base_state,
                         "ft": ft_state, "print_hash": bool(print_hash),
                         "meta": meta or {}})
        if not self._conn.poll(timeout):
            raise TimeoutError("[deltaserve] share_weights timed out")
        return self._conn.recv()

    def checksum(self, which: str = "ft", timeout: float = 30.0) -> float:
        """Ask the child to re-checksum its mapped view (zero-copy proof)."""
        self._conn.send({"cmd": "checksum", "which": which})
        if not self._conn.poll(timeout):
            raise TimeoutError("[deltaserve] checksum timed out")
        return self._conn.recv()["value"]

    def share_activations(self, buffers: dict, print_hash: bool = False,
                          timeout: float = 120.0) -> dict:
        """Share the pre-allocated activation buffers with the child via CUDA IPC.

        Sent ONCE; the child maps the same memory, so subsequent in-place writes
        by the worker's accumulation hooks are visible to the child zero-copy.
        `buffers` = {"attn_out": [t,...], "ffn_out": [t,...], "final_hidden": t,
        "concat_input_ids": t}.
        """
        for value in buffers.values():
            if isinstance(value, list):
                self._shared_refs.extend(value)
            else:
                self._shared_refs.append(value)
        self._conn.send({"cmd": "share_activations", "buffers": buffers,
                         "print_hash": bool(print_hash)})
        if not self._conn.poll(timeout):
            raise TimeoutError("[deltaserve] share_activations timed out")
        return self._conn.recv()

    def share_lora_buffers(self, payload: dict, timeout: float = 120.0) -> dict:
        """Share vLLM's served LoRA stacked buffers (CUDA-IPC) with the child so the
        trainer can publish updated weights into the exact tensors inference reads.

        ``payload`` = {"slot": int, "layers": {i: {proj: {"a": Tensor, "b": Tensor}}}}.
        Sent ONCE; the child writes the slot's rows after each optimizer step. Held in
        ``_shared_refs`` so the producer-side memory stays mapped.
        """
        for layer in (payload.get("layers") or {}).values():
            for ab in layer.values():
                self._shared_refs.extend(ab.values())
        self._conn.send({"cmd": "share_lora_buffers", "payload": payload})
        if not self._conn.poll(timeout):
            raise TimeoutError("[deltaserve] share_lora_buffers timed out")
        return self._conn.recv()

    def hash_activations(self, n: int, timeout: float = 30.0) -> dict:
        """Ask the child to hash the first n rows of its mapped activation view."""
        self._conn.send({"cmd": "hash_activations", "n": int(n)})
        if not self._conn.poll(timeout):
            raise TimeoutError("[deltaserve] hash_activations timed out")
        return self._conn.recv()["hash_report"]

    def set_corpus_meta(self, total_tokens_per_epoch: int) -> None:
        """One-shot: tell the child the FT corpus total token count
        (one full pass = one epoch). Sent once after FinetuningStore.load()
        completes; the child uses it to render the per-epoch progress meter
        in its per-cycle ``[backward]`` log. Fire-and-forget — the value is
        only used cooperatively in logging, so a missed message just prints
        ``N/?`` until the next backward (and the next backward can't fire
        before this message lands because the scheduler calls this BEFORE
        opening FT admission)."""
        if self._conn is None or self._proc is None or not self._proc.is_alive():
            return
        try:
            self._conn.send({"cmd": "set_corpus_meta",
                             "total_tokens_per_epoch": int(total_tokens_per_epoch)})
        except (BrokenPipeError, OSError, EOFError, ValueError) as e:
            dprint(f"[backward] set_corpus_meta skipped (child gone): {e}")

    def notify_buffer_full(self, n: int, sleep_s: float = 2.0,
                           sample_lens: list[int] | None = None,
                           epoch: int = 0) -> None:
        """Signal the child that the activation buffer is full (ASYNC, no wait).

        The child computes the loss + logit gradient over the captured rows,
        then cleans the buffer + sleeps `sleep_s`, then sends a response that the
        coordinator picks up via poll_response(). ``sample_lens`` are the
        per-FT-sample token counts (in buffer-write order) so the child can split
        the flat buffers into samples and shift-by-1 for next-token targets.
        """
        # Tolerate a torn-down child (e.g. during engine shutdown a final FT
        # batch can fill the buffer after the backward process is already gone):
        # a broken/closed pipe just means "no backward to notify".
        if self._conn is None or self._proc is None or not self._proc.is_alive():
            return
        try:
            self._conn.send({
                "cmd": "process_activations", "n": int(n),
                "sleep_s": float(sleep_s),
                "sample_lens": list(sample_lens or []), "epoch": int(epoch),
            })
        except (BrokenPipeError, OSError, EOFError, ValueError) as e:
            dprint(f"[backward] notify_buffer_full skipped (child gone): {e}")

    def set_pause(self, paused: bool) -> None:
        """[Phase 5] Yield/return the GPU to/from the backward child. `paused=True`
        (clear the grant) makes the child block at its next layer boundary;
        `paused=False` (set the grant) lets it resume. Called by the main process
        around inference prefill forwards. Cheap CPU-only event toggle."""
        if self._gpu_grant is None:
            return
        if paused:
            self._gpu_grant.clear()
        else:
            self._gpu_grant.set()

    def poll_response(self):
        """Non-blocking: return the child's pending message, or None.

        Tolerates a torn-down child (engine shutdown): a closed pipe makes
        poll(0) report readable but recv() raises EOFError — treat that as
        'no response' rather than crashing the engine loop."""
        if self._conn is None:
            return None
        try:
            if self._conn.poll(0):
                return self._conn.recv()
        except (EOFError, OSError, BrokenPipeError):
            return None
        return None

    def shutdown(self, timeout: float = 5.0) -> int | None:
        """Ask the child to exit, then join. Returns the child exit code."""
        if self._proc is None:
            return None
        try:
            self._conn.send({"cmd": "shutdown"})
            self._proc.join(timeout)
        except Exception:
            pass
        if self._proc.is_alive():
            dprint("[backward] child did not exit in time; terminating")
            self._proc.terminate()
            self._proc.join()
        exitcode = self._proc.exitcode
        dprint(f"[backward] shutdown complete exitcode={exitcode}")
        try:
            self._conn.close()
        except Exception:
            pass
        return exitcode

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()


def _restore_env(key: str, prev_value: str | None) -> None:
    if prev_value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = prev_value
