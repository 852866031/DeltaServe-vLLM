# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-token activation accumulation for finetuning tokens.

(Named "accumulate", not "capture", to avoid confusion with CUDA-graph capture.)

During an FT step (forced eager), we accumulate the FT-token rows of the
**residual stream** into pre-allocated, fixed-size GPU buffers at an accumulating
offset, plus the FT-token pre-LM-head hidden states + their input ids:

- ``layer_in[i]``  — the residual stream *entering* transformer layer i (the input
  to ``layers.{i}.input_layernorm``). This is exactly what a per-layer manual /
  autograd backward recomputes one layer from, so it is the backward-useful capture.
- ``final_in``     — the residual stream entering the final norm (input to
  ``model.norm``); = the residual output of the last layer.
- ``final_hidden`` — the post-final-norm hidden states (model forward output); used
  by the backward process to reconstruct logits for the loss.
- ``concat_input_ids`` — the FT-token input ids (labels, shifted per-sample downstream).
- ``mlp_gate_up[i]`` — the ``mlp.gate_up_proj`` output ([n, 2*intermediate] = gate||up)
  per layer. The MLP is frozen + the widest matmul in the layer, so saving its
  pre-activations lets the backward skip recomputing the gate_up matmul (memory-for-
  compute: ~940 MB at n=512, but removes the layer's biggest recompute). Captured with
  a forward (post) hook; absent on models without ``mlp.gate_up_proj`` (e.g. opt).

The residual-stream values are captured with **forward_pre_hooks**: vLLM's Llama
uses the fused add-norm pattern, so ``input_layernorm`` / ``model.norm`` are called
with ``(hidden, residual)`` (or just ``(hidden,)`` for layer 0). The pre-hook sees
those positional args; the residual entering the module = ``args[0]`` (len 1) or
``args[0] + args[1]`` (len 2). The fused op may update ``residual`` in place, so the
hook computes the sum and copies the masked rows immediately.

Capture is gated on the runner's finetune_mask (set via begin_step) so hooks are
no-ops off FT steps. Buffers live outside any CUDA-graph pool and are shared
zero-copy with the backward process.

Models without the fused add-norm pattern (e.g. opt) have no ``input_layernorm`` /
``model.norm`` modules, so no layer-input hooks register and only
``final_hidden`` + ``concat_input_ids`` are captured (the opt loss path).
"""

import re

import torch

from vllm.deltaserve import dprint

_LAYER_IN_SUFFIX = "input_layernorm"  # residual stream entering each layer
_FINAL_NORM_SUFFIX = ".norm"          # final norm (input = pre-final-norm residual)
_MLP_GATEUP_SUFFIX = "mlp.gate_up_proj"  # MLP pre-activation (gate||up) output
_SELF_ATTN_ATTN_SUFFIX = "self_attn.attn"  # post-RoPE q/k/v: hook sees args=(q,k,v)


class FinetuneAccumulator:
    def __init__(self, model, max_saved: int, hidden_size: int, device,
                 dtype, intermediate_size: int | None = None,
                 q_size: int | None = None,
                 kv_size: int | None = None,
                 save_attn_qkv: bool = False,
                 save_attn_ctx: bool = False) -> None:
        self.max_saved = int(max_saved)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size) if intermediate_size else 0
        self.q_size = int(q_size) if q_size else 0
        self.kv_size = int(kv_size) if kv_size else 0
        self.device = device
        self.dtype = dtype
        # Opt-in: save post-RoPE q/k/v per layer to skip Q/K/V proj + RoPE
        # recompute in the backward. Only meaningful when the attn dims are
        # known; otherwise treated as disabled (the hook target's args layout
        # is model-specific and we don't infer it without dims).
        self._save_attn_qkv = (bool(save_attn_qkv)
                               and self.q_size > 0 and self.kv_size > 0)
        # Opt-in: save the attention context output (= o_proj input) per layer
        # to skip the attention-forward recompute in the backward. Same
        # ``self_attn.attn`` module as the qkv save, but a POST hook reading the
        # output; width = q_size (= num_heads · head_dim).
        self._save_attn_ctx = bool(save_attn_ctx) and self.q_size > 0

        # Per-step state (set by begin_step, read by hooks).
        self._active = False
        self._cur_mask = None
        self._cur_n = 0
        self._cur_offset = 0  # accumulating write offset into the buffers
        # Slice fast-path: when the FT-token region is a contiguous span in
        # the flat batch layout (the common case — FT requests at the end of
        # waiting → at the tail of req_ids), hooks gather rows with a slice
        # view instead of a boolean-mask index_select. Saves one CUDA kernel
        # + one allocation per hook firing (~33 per Llama-3 FT forward).
        # Falls back to the mask path when ``_cur_contiguous`` is False
        # (e.g. an FT request landed in a freed inference slot mid-batch).
        self._cur_start = 0
        self._cur_contiguous = False
        self._handles: list = []
        # [forward_interruptible / tier C] When set, each hook raises
        # ``FTAborted`` after its copy work — bails the model forward at the
        # current layer boundary so the engine can run inference instead.
        # Wired to ``FinetuneCoordinator.ft_abort_event`` at runner setup;
        # None disables the check (when forward_interruptible is off or no
        # coordinator). One ``Event.is_set()`` call per hook firing — that's
        # a C-level atomic bool load, the cheapest cross-thread signal
        # available.
        self._abort_event = None

        # Discover capture points by module name:
        #   layers.{i}.input_layernorm -> layer_in[i]   (auto-detected; absent on opt)
        #   model.norm                 -> final_in
        #   layers.{i}.mlp.gate_up_proj -> mlp_gate_up[i] (frozen MLP pre-activation)
        #   layers.{i}.self_attn.attn  -> attn_qh/kh/vh[i] (post-RoPE q,k,v: feature-gated)
        self._layer_in_modules: dict[int, torch.nn.Module] = {}
        self._gate_up_modules: dict[int, torch.nn.Module] = {}
        self._self_attn_attn_modules: dict[int, torch.nn.Module] = {}
        self._final_norm_module: torch.nn.Module | None = None
        for name, mod in model.named_modules():
            if name.endswith(_LAYER_IN_SUFFIX):
                m = re.search(r"layers\.(\d+)\.", name)
                if m is not None:
                    self._layer_in_modules[int(m.group(1))] = mod
            elif name.endswith(_MLP_GATEUP_SUFFIX):
                m = re.search(r"layers\.(\d+)\.", name)
                if m is not None:
                    self._gate_up_modules[int(m.group(1))] = mod
            elif name.endswith(_SELF_ATTN_ATTN_SUFFIX):
                m = re.search(r"layers\.(\d+)\.", name)
                if m is not None:
                    self._self_attn_attn_modules[int(m.group(1))] = mod
            elif name.endswith(_FINAL_NORM_SUFFIX) and ".layers." not in name:
                self._final_norm_module = mod
        self.num_layers = (max(self._layer_in_modules) + 1
                           if self._layer_in_modules else 0)
        # Save the MLP gate||up only if we found the modules AND know its width.
        self._save_gate_up = bool(self._gate_up_modules) and self.intermediate_size > 0
        # Save post-RoPE q/k/v only if requested, dims known, AND modules found.
        self._save_attn_qkv = (self._save_attn_qkv
                               and bool(self._self_attn_attn_modules))
        # Save attention context (o_proj input) only if requested AND modules found.
        self._save_attn_ctx = (self._save_attn_ctx
                               and bool(self._self_attn_attn_modules))

        # Pre-allocated buffers (plain torch.zeros — outside any CUDA-graph pool).
        def _buf(width=None):
            return torch.zeros(self.max_saved, width or self.hidden_size,
                               device=device, dtype=dtype)

        self.layer_in = [_buf() for _ in range(self.num_layers)]
        self.final_in = _buf() if self._final_norm_module is not None else None
        self.final_hidden = _buf()
        self.concat_input_ids = torch.zeros(self.max_saved, device=device,
                                            dtype=torch.int64)
        self.mlp_gate_up = ([_buf(2 * self.intermediate_size)
                             for _ in range(self.num_layers)]
                            if self._save_gate_up else [])
        # Per-layer post-RoPE q/k/v buffers (only when save_attn_qkv is on).
        # Stored as flat [s_max, q_size] / [s_max, kv_size] — same layout the
        # vLLM attention call sees just before self_attn.attn(q,k,v). The
        # backward reshapes to [n, H, Hd] on the fly.
        self.attn_qh = ([_buf(self.q_size)
                         for _ in range(self.num_layers)]
                        if self._save_attn_qkv else [])
        self.attn_kh = ([_buf(self.kv_size)
                         for _ in range(self.num_layers)]
                        if self._save_attn_qkv else [])
        self.attn_vh = ([_buf(self.kv_size)
                         for _ in range(self.num_layers)]
                        if self._save_attn_qkv else [])
        # Per-layer attention context output (= o_proj input). Flat
        # [s_max, q_size] — same row order as the rest of the batch. Only
        # allocated when save_attn_ctx is on.
        self.attn_ctx = ([_buf(self.q_size)
                          for _ in range(self.num_layers)]
                         if self._save_attn_ctx else [])
        self.buffers = {
            "final_hidden": self.final_hidden,
            "concat_input_ids": self.concat_input_ids,
        }
        if self.layer_in:
            self.buffers["layer_in"] = self.layer_in
        if self.final_in is not None:
            self.buffers["final_in"] = self.final_in
        if self.mlp_gate_up:
            self.buffers["mlp_gate_up"] = self.mlp_gate_up
        if self.attn_qh:
            self.buffers["attn_qh"] = self.attn_qh
            self.buffers["attn_kh"] = self.attn_kh
            self.buffers["attn_vh"] = self.attn_vh
        if self.attn_ctx:
            self.buffers["attn_ctx"] = self.attn_ctx

    def register_hooks(self) -> None:
        for layer, mod in self._layer_in_modules.items():
            self._handles.append(
                mod.register_forward_pre_hook(
                    self._make_pre_hook(self.layer_in[layer])))
        if self._final_norm_module is not None:
            self._handles.append(
                self._final_norm_module.register_forward_pre_hook(
                    self._make_pre_hook(self.final_in)))
        if self._save_gate_up:
            for layer, mod in self._gate_up_modules.items():
                self._handles.append(
                    mod.register_forward_hook(
                        self._make_out_hook(self.mlp_gate_up[layer])))
        if self._save_attn_qkv:
            for layer, mod in self._self_attn_attn_modules.items():
                self._handles.append(
                    mod.register_forward_pre_hook(
                        self._make_attn_qkv_pre_hook(
                            self.attn_qh[layer],
                            self.attn_kh[layer],
                            self.attn_vh[layer])))
        if self._save_attn_ctx:
            for layer, mod in self._self_attn_attn_modules.items():
                self._handles.append(
                    mod.register_forward_hook(
                        self._make_attn_ctx_out_hook(self.attn_ctx[layer])))
        dprint(
            f"[accumulate] residual-stream pre-hooks on "
            f"{len(self._layer_in_modules)} input_layernorm + "
            f"{int(self._final_norm_module is not None)} final norm; "
            f"gate_up post-hooks on {len(self._gate_up_modules) if self._save_gate_up else 0}; "
            f"attn-qkv pre-hooks on {len(self._self_attn_attn_modules) if self._save_attn_qkv else 0}; "
            f"attn-ctx post-hooks on {len(self._self_attn_attn_modules) if self._save_attn_ctx else 0}"
            f"; buffers [{self.max_saved}, {self.hidden_size}] x {self.num_layers} "
            f"layers (+ final_in/final_hidden"
            f"{'/mlp_gate_up' if self._save_gate_up else ''}"
            f"{'/attn_qh+kh+vh' if self._save_attn_qkv else ''}"
            f"{'/attn_ctx' if self._save_attn_ctx else ''})"
        )

    def _make_pre_hook(self, buf):
        def pre_hook(module, args):
            if not self._active or self._cur_n == 0:
                return
            # Fused add-norm: args = (hidden,) for layer 0, else (hidden, residual).
            # Residual stream entering the module = hidden (+ residual). Compute +
            # copy immediately — the fused op may overwrite `residual` in place.
            if len(args) >= 2 and args[1] is not None:
                val = args[0] + args[1]
            else:
                val = args[0]
            # Fast path: FT rows are a contiguous span [_cur_start,
            # _cur_start + _cur_n). Slice is a view; no kernel, no alloc.
            # Fallback: boolean-mask index_select (interleaved FT in batch).
            if self._cur_contiguous:
                rows = val[self._cur_start:self._cur_start + self._cur_n]
            else:
                rows = val[self._cur_mask]
            off = self._cur_offset
            n = min(rows.shape[0], self.max_saved - off)
            if n > 0:
                buf[off:off + n].copy_(rows[:n].to(self.dtype))
            # [forward_interruptible / tier C] After the copy (so the partial
            # buffer state is at least consistent up to this layer), check
            # the abort signal and bail at this layer boundary. Raising from
            # a forward hook unwinds the model.forward() call via Python
            # exception — execute_model catches FTAborted and rolls back.
            _evt = self._abort_event
            if _evt is not None and _evt.is_set():
                from vllm.deltaserve.coordinator import FTAborted
                raise FTAborted()

        return pre_hook

    def _make_attn_qkv_pre_hook(self, qh_buf, kh_buf, vh_buf):
        """Hook fires on ``self_attn.attn(q, k, v)`` — q, k are POST-RoPE, v
        is straight from the qkv_proj split. q/k/v are flat ``[n_total, ...]``
        in the same row order as the rest of the batch, so we gather FT rows
        via the same contiguous-or-mask path as the residual-stream hooks."""
        def pre_hook(module, args):
            if not self._active or self._cur_n == 0:
                return
            # args = (q, k, v); q/k post-RoPE, all flat in batch-row order.
            q, k, v = args[0], args[1], args[2]
            if self._cur_contiguous:
                qrows = q[self._cur_start:self._cur_start + self._cur_n]
                krows = k[self._cur_start:self._cur_start + self._cur_n]
                vrows = v[self._cur_start:self._cur_start + self._cur_n]
            else:
                qrows = q[self._cur_mask]
                krows = k[self._cur_mask]
                vrows = v[self._cur_mask]
            off = self._cur_offset
            n = min(qrows.shape[0], self.max_saved - off)
            if n > 0:
                qh_buf[off:off + n].copy_(qrows[:n].to(self.dtype))
                kh_buf[off:off + n].copy_(krows[:n].to(self.dtype))
                vh_buf[off:off + n].copy_(vrows[:n].to(self.dtype))
            # tier-C abort (same idiom as the other hooks).
            _evt = self._abort_event
            if _evt is not None and _evt.is_set():
                from vllm.deltaserve.coordinator import FTAborted
                raise FTAborted()

        return pre_hook

    def _make_attn_ctx_out_hook(self, ctx_buf):
        """Hook fires AFTER ``self_attn.attn(q, k, v)`` — its output is the
        attention context (the o_proj input), flat ``[n_total, q_size]`` in
        batch-row order. Saved so the backward can skip the attention-forward
        recompute. Flattened defensively in case the op returns [n, H, Hd]."""
        def out_hook(module, inputs, output):
            if not self._active or self._cur_n == 0:
                return
            out = output[0] if isinstance(output, tuple) else output
            if self._cur_contiguous:
                rows = out[self._cur_start:self._cur_start + self._cur_n]
            else:
                rows = out[self._cur_mask]
            off = self._cur_offset
            n = min(rows.shape[0], self.max_saved - off)
            if n > 0:
                ctx_buf[off:off + n].copy_(
                    rows[:n].reshape(n, -1).to(self.dtype))
            # tier-C abort (same idiom as the other hooks).
            _evt = self._abort_event
            if _evt is not None and _evt.is_set():
                from vllm.deltaserve.coordinator import FTAborted
                raise FTAborted()

        return out_hook

    def _make_out_hook(self, buf):
        def out_hook(module, inputs, output):
            if not self._active or self._cur_n == 0:
                return
            out = output[0] if isinstance(output, tuple) else output
            if self._cur_contiguous:
                rows = out[self._cur_start:self._cur_start + self._cur_n]
            else:
                rows = out[self._cur_mask]
            off = self._cur_offset
            n = min(rows.shape[0], self.max_saved - off)
            if n > 0:
                buf[off:off + n].copy_(rows[:n].to(self.dtype))
            # [forward_interruptible / tier C] See _make_pre_hook for the
            # rationale. Bailing from a post-hook means the model still has
            # work after gate_up_proj in this layer (down_proj + residual
            # add + the next input_layernorm) — those will run before the
            # next layer's pre_hook gets a chance to raise. Negligible
            # extra cost; the pre_hook is the load-bearing check.
            _evt = self._abort_event
            if _evt is not None and _evt.is_set():
                from vllm.deltaserve.coordinator import FTAborted
                raise FTAborted()

        return out_hook

    def begin_step(self, mask_gpu, num_ft: int, offset: int = 0,
                   start: int = 0, contiguous: bool = False) -> None:
        """Arm the per-step accumulation state read by the hooks.

        ``start`` / ``contiguous`` are the slice fast-path metadata. When
        ``contiguous`` is True the hooks use ``val[start:start+num_ft]``
        (a view, no kernel); otherwise they fall back to ``val[mask_gpu]``.
        Default ``contiguous=False`` keeps callers that didn't pass the new
        args on the mask path.
        """
        self._cur_mask = mask_gpu
        self._cur_n = int(num_ft)
        self._cur_offset = int(offset)
        self._cur_start = int(start)
        self._cur_contiguous = bool(contiguous)
        self._active = True

    def accumulate_final(self, hidden_states, input_ids_flat, mask_gpu,
                         num_ft: int, offset: int = 0,
                         start: int = 0, contiguous: bool = False) -> None:
        """Save FT-token pre-LM-head hidden states + their input ids. Slice
        fast path mirrors the per-layer hooks: when ``contiguous`` is True,
        gather via ``[start:start+num_ft]`` (view); else mask-gather."""
        if num_ft == 0:
            return
        if contiguous:
            rows = hidden_states[start:start + num_ft]
            ids = input_ids_flat[start:start + num_ft]
        else:
            rows = hidden_states[mask_gpu]
            ids = input_ids_flat[mask_gpu]
        n = min(rows.shape[0], self.max_saved - offset)
        if n <= 0:
            return
        self.final_hidden[offset:offset + n].copy_(rows[:n].to(self.dtype))
        self.concat_input_ids[offset:offset + n].copy_(ids[:n].to(torch.int64))

    def end_step(self) -> None:
        self._active = False
        self._cur_mask = None
        self._cur_n = 0
        self._cur_offset = 0
        self._cur_start = 0
        self._cur_contiguous = False

    def zero_offset_range(self, off: int, n: int) -> None:
        """[forward_interruptible / tier C] Zero the [off:off+n] slice of
        every hook-target buffer so the partial bytes from an aborted
        forward don't leak into debugging / hashing flows. Safe to call
        from the runner's abort path: ``fill_count`` already gates the
        backward from reading these rows (so this is a hygiene-only step,
        not correctness), but it makes the buffer state easy to reason
        about and keeps an accidental re-read deterministic.

        Cost: one GPU memset per buffer (32 layer_in + maybe final_in +
        maybe 32 mlp_gate_up + concat_input_ids + final_hidden), each over
        n rows of hidden-size — sub-millisecond in aggregate at typical
        sizes."""
        if n <= 0 or off < 0:
            return
        n = min(n, self.max_saved - off)
        if n <= 0:
            return
        for buf in self.layer_in:
            buf[off:off + n].zero_()
        if self.final_in is not None:
            self.final_in[off:off + n].zero_()
        self.final_hidden[off:off + n].zero_()
        self.concat_input_ids[off:off + n].zero_()
        for buf in self.mlp_gate_up:
            buf[off:off + n].zero_()
        for buf in self.attn_qh:
            buf[off:off + n].zero_()
        for buf in self.attn_kh:
            buf[off:off + n].zero_()
        for buf in self.attn_vh:
            buf[off:off + n].zero_()
        for buf in self.attn_ctx:
            buf[off:off + n].zero_()
