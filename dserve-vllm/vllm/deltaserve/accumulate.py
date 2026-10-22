# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-token activation accumulation for finetuning tokens.

(Named "accumulate", not "capture", to avoid confusion with CUDA-graph capture.)

During an FT step (forced eager), we accumulate the FT-token rows of the
**residual stream** into pre-allocated, fixed-size GPU buffers at an accumulating
offset, plus the FT-token input ids:

- ``layer_in[i]``  — the residual stream *entering* transformer layer i (the input
  to ``layers.{i}.input_layernorm``). This is exactly what a per-layer manual /
  autograd backward recomputes one layer from, so it is the backward-useful capture.
- ``final_in``     — the residual stream entering the final norm (input to
  ``model.norm``); = the residual output of the last layer.
- ``final_hidden`` — the post-final-norm hidden states (model forward output). Only
  allocated with ``save_final_hidden`` (the ``print_activation_hash`` debug check
  compares it against RMSNorm(final_in)); the trainer itself reads ``final_in``.
- ``concat_input_ids`` — the FT-token input ids (labels, shifted per-sample downstream).
- ``mlp_gate_up[i]`` — the ``mlp.gate_up_proj`` output ([n, 2*intermediate] = gate||up)
  per layer. The MLP is frozen + the widest matmul in the layer, so saving its
  pre-activations lets the backward skip recomputing the gate_up matmul (memory-for-
  compute: ~940 MB at n=512, but removes the layer's biggest recompute). Captured with
  a forward (post) hook; absent on models without ``mlp.gate_up_proj``.

The residual-stream values are captured with **forward_pre_hooks**: vLLM's Llama
uses the fused add-norm pattern, so ``input_layernorm`` / ``model.norm`` are called
with ``(hidden, residual)`` (or just ``(hidden,)`` for layer 0). The pre-hook sees
those positional args; the residual entering the module = ``args[0]`` (len 1) or
``args[0] + args[1]`` (len 2). The fused op may update ``residual`` in place, so the
hook computes the sum and copies the masked rows immediately.

Capture is gated on the runner's finetune_mask (set via begin_step) so hooks are
no-ops off FT steps. Buffers live outside any CUDA-graph pool and are shared
zero-copy with the backward process.
"""

import re

import torch

from vllm.deltaserve import dprint
from vllm.deltaserve.ft_save_op import MARKER_ATTR

_LAYER_IN_SUFFIX = "input_layernorm"  # residual stream entering each layer
_FINAL_NORM_SUFFIX = ".norm"          # final norm (input = pre-final-norm residual)
_MLP_GATEUP_SUFFIX = "mlp.gate_up_proj"  # MLP pre-activation (gate||up) output
# Post-attention residual (= layer_in + o_proj out = the FFN input): the fused
# add-norm is called with (o, residual), so a pre-hook sums the args exactly
# like the input_layernorm hooks do. Feature-gated (save_resid_mid).
_POST_LN_SUFFIX = "post_attention_layernorm"
_SELF_ATTN_ATTN_SUFFIX = "self_attn.attn"  # post-RoPE q/k/v: hook sees args=(q,k,v)
# Per-head q/k transform modules (Qwen3's q_norm / k_norm). With
# ``attn_qkv_pre_transform`` the q/k saves hook the INPUT of these modules —
# the raw projection outputs the transform's backward needs — instead of the
# post-transform values ``self_attn.attn`` sees.
_Q_NORM_SUFFIX = "self_attn.q_norm"
_K_NORM_SUFFIX = "self_attn.k_norm"


class FinetuneAccumulator:
    def __init__(self, model, max_saved: int, hidden_size: int, device,
                 dtype, intermediate_size: int | None = None,
                 q_size: int | None = None,
                 kv_size: int | None = None,
                 save_attn_qkv: bool = False,
                 save_attn_ctx: bool = False,
                 save_resid_mid: bool = False,
                 attn_qkv_pre_transform: bool = False,
                 save_final_hidden: bool = False) -> None:
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
        # Where q/k are captured for save_attn_qkv: the ``self_attn.attn``
        # pre-hook (post-RoPE; Llama-3) or the ``self_attn.{q,k}_norm``
        # pre-hooks (raw projection outputs; Qwen3). v always comes from the
        # attn pre-hook. Decided per family by ``ft_meta.saved_qkv_pre_transform``.
        self._attn_qkv_pre_transform = bool(attn_qkv_pre_transform)
        # Opt-in: save the attention context output (= o_proj input) per layer
        # to skip the attention-forward recompute in the backward. Same
        # ``self_attn.attn`` module as the qkv save, but a POST hook reading the
        # output; width = q_size (= num_heads · head_dim).
        self._save_attn_ctx = bool(save_attn_ctx) and self.q_size > 0
        # Opt-in: save the post-attention residual (resid_mid) per layer so
        # the backward skips the O-proj recompute — and, under TP, its
        # all-reduce (the value is captured AFTER vLLM's row-parallel reduce,
        # so it is the full residual on every rank). Full hidden width.
        self._save_resid_mid = bool(save_resid_mid)

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
        # The check the hooks actually call: ``_abort_poll(boundary) -> bool``.
        # tp=1: reads the Event (every hook). TP: ``FtAbortPoller.hook_check``
        # — the rank-symmetric collective runs only from the layer-boundary
        # hook (``boundary=True``, the input_layernorm pre-hook); the other
        # hooks in the layer return False without a collective. None = off.
        self._abort_poll = None

        # Discover capture points by module name:
        #   layers.{i}.input_layernorm -> layer_in[i]
        #   model.norm                 -> final_in
        #   layers.{i}.mlp.gate_up_proj -> mlp_gate_up[i] (frozen MLP pre-activation)
        #   layers.{i}.self_attn.attn  -> attn_qh/kh/vh[i] (post-RoPE q,k,v: feature-gated)
        self._layer_in_modules: dict[int, torch.nn.Module] = {}
        self._gate_up_modules: dict[int, torch.nn.Module] = {}
        self._self_attn_attn_modules: dict[int, torch.nn.Module] = {}
        self._post_ln_modules: dict[int, torch.nn.Module] = {}
        self._q_norm_modules: dict[int, torch.nn.Module] = {}
        self._k_norm_modules: dict[int, torch.nn.Module] = {}
        self._final_norm_module: torch.nn.Module | None = None
        for name, mod in model.named_modules():
            if name.endswith(_LAYER_IN_SUFFIX):
                m = re.search(r"layers\.(\d+)\.", name)
                if m is not None:
                    self._layer_in_modules[int(m.group(1))] = mod
            elif name.endswith(_POST_LN_SUFFIX):
                m = re.search(r"layers\.(\d+)\.", name)
                if m is not None:
                    self._post_ln_modules[int(m.group(1))] = mod
            elif name.endswith(_MLP_GATEUP_SUFFIX):
                m = re.search(r"layers\.(\d+)\.", name)
                if m is not None:
                    self._gate_up_modules[int(m.group(1))] = mod
            elif name.endswith(_SELF_ATTN_ATTN_SUFFIX):
                m = re.search(r"layers\.(\d+)\.", name)
                if m is not None:
                    self._self_attn_attn_modules[int(m.group(1))] = mod
            elif name.endswith(_Q_NORM_SUFFIX):
                m = re.search(r"layers\.(\d+)\.", name)
                if m is not None:
                    self._q_norm_modules[int(m.group(1))] = mod
            elif name.endswith(_K_NORM_SUFFIX):
                m = re.search(r"layers\.(\d+)\.", name)
                if m is not None:
                    self._k_norm_modules[int(m.group(1))] = mod
            elif name.endswith(_FINAL_NORM_SUFFIX) and ".layers." not in name:
                self._final_norm_module = mod
        self.num_layers = (max(self._layer_in_modules) + 1
                           if self._layer_in_modules else 0)
        # Save the MLP gate||up only if we found the modules AND know its width.
        self._save_gate_up = bool(self._gate_up_modules) and self.intermediate_size > 0
        # Save q/k/v only if requested, dims known, AND modules found — in
        # pre-transform mode the q/k norm modules must exist for every layer
        # that has an attn module (else the save would not be exact: fall back
        # to the recompute path rather than capture the wrong stage).
        self._save_attn_qkv = (self._save_attn_qkv
                               and bool(self._self_attn_attn_modules))
        if self._save_attn_qkv and self._attn_qkv_pre_transform:
            have = set(self._q_norm_modules) & set(self._k_norm_modules)
            if have != set(self._self_attn_attn_modules):
                dprint("[accumulate] save_attn_qkv (pre-transform) requested but "
                       f"q_norm/k_norm found on {len(have)} of "
                       f"{len(self._self_attn_attn_modules)} layers — disabling "
                       "the q/k/v save (backward recomputes Q/K/V)")
                self._save_attn_qkv = False
        # Save attention context (o_proj input) only if requested AND modules found.
        self._save_attn_ctx = (self._save_attn_ctx
                               and bool(self._self_attn_attn_modules))
        # Save the post-attention residual only if requested AND modules found.
        self._save_resid_mid = (self._save_resid_mid
                                and bool(self._post_ln_modules))

        # Pre-allocated buffers (plain torch.zeros — outside any CUDA-graph pool).
        # [mixed-fwd-cuda-graph] One scratch row past max_saved: the
        # fixed-shape save op writes the unused index slots there. Every
        # consumer slices ``[:n]`` / ``[off:off+n]`` with n ≤ max_saved, so
        # the extra row is invisible to the backward child and the hooks.
        def _buf(width=None):
            return torch.zeros(self.max_saved + 1, width or self.hidden_size,
                               device=device, dtype=dtype)

        self.layer_in = [_buf() for _ in range(self.num_layers)]
        self.final_in = _buf() if self._final_norm_module is not None else None
        self.final_hidden = _buf() if save_final_hidden else None
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
        # Per-layer post-attention residual (= FFN input). Full hidden width,
        # like layer_in. Only allocated when save_resid_mid is on.
        self.resid_mid = ([_buf() for _ in range(self.num_layers)]
                          if self._save_resid_mid else [])
        self.buffers = {"concat_input_ids": self.concat_input_ids}
        if self.final_hidden is not None:
            self.buffers["final_hidden"] = self.final_hidden
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
        if self.resid_mid:
            self.buffers["resid_mid"] = self.resid_mid

        # [mixed-fwd-cuda-graph] Persistent index tensors for the save op
        # (fixed addresses — the has_ft graphs bake them in), a pinned ring
        # for the per-step host→device refresh, and the slot → buffer table.
        # ``install_save_points`` fills the table and marks the modules.
        self.ft_src = torch.zeros(self.max_saved, device=device,
                                  dtype=torch.int64)
        self.ft_dst = torch.full((self.max_saved,), self.max_saved,
                                 device=device, dtype=torch.int64)
        self._idx_pin = torch.empty((4, 2, self.max_saved), dtype=torch.int64,
                                    pin_memory=(torch.device(device).type == "cuda"))
        self._idx_pin_pos = 0
        self.op_bufs: list = []
        self.op_ready = False
        self._hooked_modules: list = []
        self._op_marker = None

    # ─── [mixed-fwd-cuda-graph] graph-capturable save path ──────────────
    # Slot layout per layer i: 7*i + {0: layer_in, 1: resid_mid, 2: gate_up,
    # 3: q, 4: k, 5: v, 6: ctx}; final_in = 7*L. A disabled save leaves its
    # slot None and the op returns without work for it.
    _OP_NAMES = ("layer_in", "resid_mid", "gate_up", "q", "k", "v", "ctx")

    def install_save_points(self, model) -> bool:
        """Arm the graph-capturable save path: build the slot → buffer table
        and give every hooked module the marker buffer the op declares as
        mutated. The hooks themselves issue the op (in their inactive branch,
        which is the one Dynamo traces into the compiled forward), so the model
        code carries no call sites. Returns False — mixed batches keep the eager
        hook path — when a save point's module is missing."""
        L = self.num_layers
        if not L or len(self._layer_in_modules) != L or (
                self._final_norm_module is None):
            dprint("[accumulate] incomplete save points — mixed FT batches "
                   "stay on the eager hook path")
            return False
        bufs: list = [None] * (7 * L + 1)
        for i in range(L):
            bufs[7 * i + 0] = self.layer_in[i]
            if self.resid_mid:
                bufs[7 * i + 1] = self.resid_mid[i]
            if self.mlp_gate_up:
                bufs[7 * i + 2] = self.mlp_gate_up[i]
            if self.attn_qh:
                bufs[7 * i + 3] = self.attn_qh[i]
                bufs[7 * i + 4] = self.attn_kh[i]
                bufs[7 * i + 5] = self.attn_vh[i]
            if self.attn_ctx:
                bufs[7 * i + 6] = self.attn_ctx[i]
        if self.final_in is not None:
            bufs[7 * L] = self.final_in
        self.op_bufs = bufs
        from vllm.deltaserve.ft_save_op import MARKER_ATTR

        marker = torch.zeros(1, device=self.device, dtype=torch.int32)
        self._op_marker = marker
        for mod in self._hooked_modules:
            if not hasattr(mod, MARKER_ATTR):
                mod.register_buffer(MARKER_ATTR, marker, persistent=False)
        self.op_ready = True
        dprint(f"[accumulate] graph save op armed on {L} layers (+ final norm) "
               f"— mixed FT batches may replay CUDA graphs")
        return True

    def capture_state(self):
        """The save-op state (``ft_save_op.active_state``) for capturing /
        running a has_ft graph: the persistent index tensors as they are (scratch-pointing
        at capture) and the slot table."""
        from vllm.deltaserve.ft_save_op import FtSaveState

        return FtSaveState(src=self.ft_src, dst=self.ft_dst, bufs=self.op_bufs)

    def arm_op_step(self, positions, num_ft: int, offset: int):
        """Refresh the index tensors for this step: ``positions`` are the FT
        rows in the flat batch (any order / interleaving), written to buffer
        rows ``offset .. offset+n``; unused slots gather row 0 into the
        scratch row. One pinned staging slot + one non-blocking H2D copy on
        the current stream (ordered after the previous step's replay).
        Returns the save-op state for the step."""
        n = min(int(num_ft), self.max_saved - int(offset))
        slot = self._idx_pin_pos
        self._idx_pin_pos = (slot + 1) % self._idx_pin.shape[0]
        stage = self._idx_pin[slot]
        stage[0].zero_()
        stage[1].fill_(self.max_saved)
        if n > 0:
            stage[0, :n] = torch.as_tensor(positions[:n], dtype=torch.int64)
            stage[1, :n] = torch.arange(offset, offset + n, dtype=torch.int64)
        self.ft_src.copy_(stage[0], non_blocking=True)
        self.ft_dst.copy_(stage[1], non_blocking=True)
        return self.capture_state()

    def register_hooks(self) -> None:
        """One hook per save point. On an eager FT step (``begin_step`` armed)
        a hook copies the FT rows itself; otherwise — in particular while
        Dynamo traces the compiled forward — it issues the graph-capturable
        save op for its slot (``install_save_points``), which is a no-op unless
        a save state is active (mixed FT step / has_ft capture)."""
        L = self.num_layers
        hooked = self._hooked_modules = []

        def pre(mod, hook):
            hooked.append(mod)
            self._handles.append(mod.register_forward_pre_hook(hook))

        def post(mod, hook):
            hooked.append(mod)
            self._handles.append(mod.register_forward_hook(hook))

        for layer, mod in self._layer_in_modules.items():
            pre(mod, self._make_pre_hook(self.layer_in[layer], 7 * layer + 0,
                                         boundary=True))
        if self._final_norm_module is not None:
            pre(self._final_norm_module, self._make_pre_hook(self.final_in, 7 * L))
        if self._save_gate_up:
            for layer, mod in self._gate_up_modules.items():
                post(mod, self._make_out_hook(self.mlp_gate_up[layer],
                                              7 * layer + 2))
        if self._save_attn_qkv and not self._attn_qkv_pre_transform:
            for layer, mod in self._self_attn_attn_modules.items():
                pre(mod, self._make_attn_qkv_pre_hook(
                    self.attn_qh[layer], self.attn_kh[layer],
                    self.attn_vh[layer], 7 * layer + 3))
        elif self._save_attn_qkv:
            # Pre-transform: q from q_norm's input, k from k_norm's input
            # (raw projection outputs, [n, H, Hd] views → flattened), v from
            # the attn pre-hook's third arg (no transform on v).
            for layer, mod in self._q_norm_modules.items():
                pre(mod, self._make_arg_pre_hook(self.attn_qh[layer], 0,
                                                 7 * layer + 3))
            for layer, mod in self._k_norm_modules.items():
                pre(mod, self._make_arg_pre_hook(self.attn_kh[layer], 0,
                                                 7 * layer + 4))
            for layer, mod in self._self_attn_attn_modules.items():
                pre(mod, self._make_arg_pre_hook(self.attn_vh[layer], 2,
                                                 7 * layer + 5))
        if self._save_attn_ctx:
            for layer, mod in self._self_attn_attn_modules.items():
                post(mod, self._make_attn_ctx_out_hook(self.attn_ctx[layer],
                                                       7 * layer + 6))
        if self._save_resid_mid:
            # post_attention_layernorm(o, residual): the residual entering it
            # is o + residual = resid_mid — the same pre-hook as layer_in.
            for layer, mod in self._post_ln_modules.items():
                pre(mod, self._make_pre_hook(self.resid_mid[layer], 7 * layer + 1))
        dprint(
            f"[accumulate] residual-stream pre-hooks on "
            f"{len(self._layer_in_modules)} input_layernorm + "
            f"{int(self._final_norm_module is not None)} final norm; "
            f"gate_up post-hooks on {len(self._gate_up_modules) if self._save_gate_up else 0}; "
            f"attn-qkv pre-hooks on {len(self._self_attn_attn_modules) if self._save_attn_qkv else 0}"
            f"{' (q/k pre-transform)' if self._save_attn_qkv and self._attn_qkv_pre_transform else ''}; "
            f"attn-ctx post-hooks on {len(self._self_attn_attn_modules) if self._save_attn_ctx else 0}; "
            f"resid-mid pre-hooks on {len(self._post_ln_modules) if self._save_resid_mid else 0}"
            f"; buffers [{self.max_saved}, {self.hidden_size}] x {self.num_layers} "
            f"layers (+ final_in/final_hidden"
            f"{'/mlp_gate_up' if self._save_gate_up else ''}"
            f"{'/attn_qh+kh+vh' if self._save_attn_qkv else ''}"
            f"{'/attn_ctx' if self._save_attn_ctx else ''}"
            f"{'/resid_mid' if self._save_resid_mid else ''})"
        )

    def _op(self, module, slot: int, x, r=None) -> None:
        """The inactive-branch save: the graph-capturable op for ``slot``."""
        if self.op_ready:
            torch.ops.vllm.dserve_save_rows(x, r, getattr(module, MARKER_ATTR),
                                            slot)

    def _make_pre_hook(self, buf, slot: int, boundary: bool = False):
        def pre_hook(module, args):
            if not self._active:
                self._op(module, slot, args[0],
                         args[1] if len(args) >= 2 else None)
                return
            if self._cur_n == 0:
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
            _poll = self._abort_poll
            if _poll is not None and _poll(boundary):
                from vllm.deltaserve.coordinator import FTAborted
                raise FTAborted()

        return pre_hook

    def _make_attn_qkv_pre_hook(self, qh_buf, kh_buf, vh_buf, slot: int):
        """Hook fires on ``self_attn.attn(q, k, v)`` — q, k are POST-RoPE, v
        is straight from the qkv_proj split. q/k/v are flat ``[n_total, ...]``
        in the same row order as the rest of the batch, so we gather FT rows
        via the same contiguous-or-mask path as the residual-stream hooks.
        ``slot`` is q's; k and v follow."""
        def pre_hook(module, args):
            if not self._active:
                for j in range(3):
                    self._op(module, slot + j, args[j])
                return
            if self._cur_n == 0:
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
            _poll = self._abort_poll
            if _poll is not None and _poll(False):
                from vllm.deltaserve.coordinator import FTAborted
                raise FTAborted()

        return pre_hook

    def _make_arg_pre_hook(self, buf, arg_index: int, slot: int):
        """Pre-hook saving the FT rows of ``args[arg_index]`` (flattened to
        ``[n, -1]``) — the pre-transform q/k (``q_norm``/``k_norm`` input,
        a ``[n, H, Hd]`` view) and v (``self_attn.attn`` third arg)."""
        def pre_hook(module, args):
            if not self._active:
                t = args[arg_index]
                self._op(module, slot, t.reshape(t.shape[0], -1))
                return
            if self._cur_n == 0:
                return
            t = args[arg_index]
            if self._cur_contiguous:
                rows = t[self._cur_start:self._cur_start + self._cur_n]
            else:
                rows = t[self._cur_mask]
            off = self._cur_offset
            n = min(rows.shape[0], self.max_saved - off)
            if n > 0:
                buf[off:off + n].copy_(rows[:n].reshape(n, -1).to(self.dtype))
            _poll = self._abort_poll
            if _poll is not None and _poll(False):
                from vllm.deltaserve.coordinator import FTAborted
                raise FTAborted()

        return pre_hook

    def _make_attn_ctx_out_hook(self, ctx_buf, slot: int):
        """Hook fires AFTER ``self_attn.attn(q, k, v)`` — its output is the
        attention context (the o_proj input), flat ``[n_total, q_size]`` in
        batch-row order. Saved so the backward can skip the attention-forward
        recompute. Flattened defensively in case the op returns [n, H, Hd]."""
        def out_hook(module, inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            if not self._active:
                self._op(module, slot, out.reshape(out.shape[0], -1))
                return
            if self._cur_n == 0:
                return
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
            _poll = self._abort_poll
            if _poll is not None and _poll(False):
                from vllm.deltaserve.coordinator import FTAborted
                raise FTAborted()

        return out_hook

    def _make_out_hook(self, buf, slot: int):
        def out_hook(module, inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            if not self._active:
                self._op(module, slot, out)
                return
            if self._cur_n == 0:
                return
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
            _poll = self._abort_poll
            if _poll is not None and _poll(False):
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
            # [mixed-fwd-cuda-graph] Under a CUDA graph the model output (and
            # the persistent input-id buffer) is PADDED to the capture size;
            # the mask covers the unpadded batch. Slice to the mask's length
            # before masking (a no-op on eager steps).
            n_mask = int(mask_gpu.shape[0])
            rows = hidden_states[:n_mask][mask_gpu]
            ids = input_ids_flat[:n_mask][mask_gpu]
        n = min(rows.shape[0], self.max_saved - offset)
        if n <= 0:
            return
        if self.final_hidden is not None:
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
        if self.final_hidden is not None:
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
        for buf in self.resid_mid:
            buf[off:off + n].zero_()
