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


class FinetuneAccumulator:
    def __init__(self, model, max_saved: int, hidden_size: int, device,
                 dtype, intermediate_size: int | None = None) -> None:
        self.max_saved = int(max_saved)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size) if intermediate_size else 0
        self.device = device
        self.dtype = dtype

        # Per-step state (set by begin_step, read by hooks).
        self._active = False
        self._cur_mask = None
        self._cur_n = 0
        self._cur_offset = 0  # accumulating write offset into the buffers
        self._handles: list = []

        # Discover capture points by module name:
        #   layers.{i}.input_layernorm -> layer_in[i]   (auto-detected; absent on opt)
        #   model.norm                 -> final_in
        #   layers.{i}.mlp.gate_up_proj -> mlp_gate_up[i] (frozen MLP pre-activation)
        self._layer_in_modules: dict[int, torch.nn.Module] = {}
        self._gate_up_modules: dict[int, torch.nn.Module] = {}
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
            elif name.endswith(_FINAL_NORM_SUFFIX) and ".layers." not in name:
                self._final_norm_module = mod
        self.num_layers = (max(self._layer_in_modules) + 1
                           if self._layer_in_modules else 0)
        # Save the MLP gate||up only if we found the modules AND know its width.
        self._save_gate_up = bool(self._gate_up_modules) and self.intermediate_size > 0

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
        dprint(
            f"[accumulate] residual-stream pre-hooks on "
            f"{len(self._layer_in_modules)} input_layernorm + "
            f"{int(self._final_norm_module is not None)} final norm; "
            f"gate_up post-hooks on {len(self._gate_up_modules) if self._save_gate_up else 0}"
            f"; buffers [{self.max_saved}, {self.hidden_size}] x {self.num_layers} "
            f"layers (+ final_in/final_hidden"
            f"{'/mlp_gate_up' if self._save_gate_up else ''})"
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
            rows = val[self._cur_mask]
            off = self._cur_offset
            n = min(rows.shape[0], self.max_saved - off)
            if n > 0:
                buf[off:off + n].copy_(rows[:n].to(self.dtype))

        return pre_hook

    def _make_out_hook(self, buf):
        def out_hook(module, inputs, output):
            if not self._active or self._cur_n == 0:
                return
            out = output[0] if isinstance(output, tuple) else output
            rows = out[self._cur_mask]
            off = self._cur_offset
            n = min(rows.shape[0], self.max_saved - off)
            if n > 0:
                buf[off:off + n].copy_(rows[:n].to(self.dtype))

        return out_hook

    def begin_step(self, mask_gpu, num_ft: int, offset: int = 0) -> None:
        self._cur_mask = mask_gpu
        self._cur_n = int(num_ft)
        self._cur_offset = int(offset)
        self._active = True

    def accumulate_final(self, hidden_states, input_ids_flat, mask_gpu,
                         num_ft: int, offset: int = 0) -> None:
        if num_ft == 0:
            return
        rows = hidden_states[mask_gpu]
        n = min(rows.shape[0], self.max_saved - offset)
        if n <= 0:
            return
        self.final_hidden[offset:offset + n].copy_(rows[:n].to(self.dtype))
        ids = input_ids_flat[mask_gpu]
        self.concat_input_ids[offset:offset + n].copy_(ids[:n].to(torch.int64))

    def end_step(self) -> None:
        self._active = False
        self._cur_mask = None
        self._cur_n = 0
        self._cur_offset = 0
