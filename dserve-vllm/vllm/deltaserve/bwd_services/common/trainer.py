# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""``LoraSftTrainerService`` — the family-agnostic LoRA-SFT backward service.

Owns everything a trained family shares: slicing the shared base weights and
the FT adapter into per-layer views (per this rank's TP shard), the fp32 LoRA
masters + fused AdamW, the backward NCCL group, the per-cycle backward
(``head_backward`` → reverse layer loop → clip → step), the CUDA-graph runner,
and the publish of the trained masters into vLLM's served LoRA buffers.

The per-layer math is the family's: a subclass sets ``family`` to a
``Family`` record (``bwd_services/llama3.py``, ``bwd_services/qwen3.py``) and
inherits the whole loop. Family functions are bound to instance attributes once
in ``_build_state`` and called directly on the hot path — no dispatch per call.
"""

from __future__ import annotations

import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.deltaserve import dprint
from vllm.deltaserve.bwd_services.base import BackwardService
from vllm.deltaserve.bwd_services.common.family import LORA_PROJS, Family
from vllm.deltaserve.bwd_services.common.head import head_backward
from vllm.deltaserve.bwd_services.common.ops import rmsnorm, rope_cos_sin
from vllm.deltaserve.bwd_services.common.tp import (
    BUCKET_LAYERS,
    DEFAULT_BACKWARD_NCCL_PORT,
    REPLICATED_FACTORS,
    CommQueue,
    FactorBucket,
    clip_layers_symmetric_,
    init_backward_tp_group,
    lora_shard_slice,
)

_TOL = 5e-2  # bf16 round-trip tolerance for the capture/forward-fidelity checks

# The PEFT adapter keys the trainer turns into fp32 masters.
_LORA_KEY_RE = re.compile(
    r"layers\.(\d+)\.self_attn\.([qkvo])_proj\.lora_([AB])\.weight")


class LoraSftTrainerService(BackwardService):
    """Backward child that trains the FT LoRA adapter for one model family."""

    family: Family = None  # set by the per-family subclass
    is_trainer = True

    def __init__(self, device_index: int) -> None:
        super().__init__(device_index)
        assert self.family is not None, (
            f"{type(self).__name__} must set the ``family`` class attribute")
        self._built = False
        self.lora: dict = {}            # layer -> {proj -> {"A":Param,"B":Param}}
        self.base: dict = {}            # layer -> {q,k,v,o,gate,up,down,in_ln,post_ln,…}
        self.optimizer = None
        self.scheduler = None
        # CUDA-graph runner (Phase 5). Attached in _build_state when
        # meta["backward_cuda_graph"] is True; None means all-eager.
        self.graph_runner = None
        # Persistent grad_qh/kh/vh buffers (allocated in _build_state once
        # dims + cdt are known). Reused across layers and backwards by the
        # eager attn_backward_core path via its grad_*_buf kwargs.
        self._grad_qh_buf = None
        self._grad_kh_buf = None
        self._grad_vh_buf = None
        # [DeltaServe] Phase 7 / M3: TP shard geometry + the backward-only
        # all-reduce. tp_size=1 (single-GPU) leaves _all_reduce None → no comm.
        self.tp_size = 1
        self.tp_rank = 0
        self._all_reduce = None
        # [DeltaServe] Phase 7 / M4.3: under TP the replicated LoRA-factor
        # grads are bucketed and reduced per group of layers on a comm stream
        # (instead of 4 inline reduces per layer), and the per-layer clip runs
        # after that reduce with a rank-symmetric norm. None at tp_size == 1.
        self._comm: CommQueue | None = None
        self._bucket: FactorBucket | None = None
        self._diag_count = 0
        # Family layer functions, bound once in _build_state.
        self._layer_forward = self.family.layer_forward
        self._layer_backward = self.family.layer_backward

    # -- build master params + optimizer once weights are received -----------

    def _handle_share_weights(self, conn, msg) -> None:
        super()._handle_share_weights(conn, msg)
        try:
            self._build_state()
            self._built = True
        except Exception as e:  # noqa: BLE001 — surface but keep the service alive
            import traceback
            dprint(f"[backward] {self.family.name} state build failed: {e}")
            traceback.print_exc()

    def _build_state(self) -> None:
        meta = self.shared["meta"]
        base = self.shared["base"] or {}
        ft = self.shared["ft"] or {}
        fam = self.family
        # [DeltaServe] Phase 7 / M2: TP shard geometry. Under TP>1 the base
        # weights shared by the worker are this rank's SHARDS: qkv/gate_up are
        # column-parallel (sharded on output → local q/k/v/gate/up widths),
        # o/down are row-parallel (sharded on input → used as-is). So the head
        # counts and intermediate size the backward slices by are LOCAL =
        # full // tp_size. hidden_size (residual stream) and vocab stay FULL —
        # the residual is all-reduced and the worker all-gathers lm_head to full
        # before sharing. tp_size=1 → local == full → single-GPU behaviour.
        self.tp_size = int(meta.get("tp_size", 1))
        self.tp_rank = int(meta.get("tp_rank", 0))
        # [DeltaServe] Phase 7 / M3: stand up a NCCL group ACROSS the backward
        # children (one per rank) — they are NOT in vLLM's inference NCCL group.
        # This carries the per-layer gradient/activation all-reduces.
        self._all_reduce = init_backward_tp_group(
            self.tp_size, self.tp_rank,
            int(meta.get("backward_nccl_port", DEFAULT_BACKWARD_NCCL_PORT)))
        Hq_full = int(meta["num_attention_heads"])
        Hkv_full = int(meta["num_key_value_heads"])
        inter_full = int(meta["intermediate_size"])
        assert Hq_full % self.tp_size == 0 and Hkv_full % self.tp_size == 0 \
            and inter_full % self.tp_size == 0, (
                f"[backward] TP finetuning needs Hq({Hq_full}) / Hkv({Hkv_full})"
                f" / inter({inter_full}) divisible by tp_size({self.tp_size})")
        self.D = int(meta["hidden_size"])          # full (residual stream)
        self.L = int(meta["num_hidden_layers"])
        self.Hq = Hq_full // self.tp_size          # local q heads
        self.Hkv = Hkv_full // self.tp_size        # local kv heads
        self.Hd = int(meta["head_dim"])            # full (per-head, unsharded)
        self.kv_size = self.Hkv * self.Hd          # local
        self.q_size = self.Hq * self.Hd            # local
        self.inter = inter_full // self.tp_size    # local
        self.theta = float(meta["rope_theta"])
        self.eps = float(meta["rms_norm_eps"])
        self.scaling = float(meta.get("lora_scaling", 1.0))
        self.vocab = int(meta["vocab_size"])
        self.dims = (self.Hq, self.Hkv, self.Hd, self.kv_size)
        # Mirror the worker-side ``save_attn_qkv`` flag so the optional
        # graph runner (constructed below) and the eager per-layer loop both
        # see it. Drives whether ``layer_forward`` / the captured forward
        # graph short-circuits Q/K/V proj + RoPE using the saved buffers.
        self.save_attn_qkv = bool(meta.get("save_attn_qkv", False))
        if self.save_attn_qkv and not fam.supports_saved_qkv:
            dprint(f"[backward] save_attn_qkv requested but the {fam.name} "
                   "backward needs the pre-transform q/k; recomputing Q/K/V "
                   "from x_norm1 instead (saved buffers ignored)")
            self.save_attn_qkv = False
        # Mirror the worker-side ``save_attn_ctx`` flag — drives whether
        # ``layer_forward`` / the captured forward graph skips the attention
        # forward (scores/softmax/AV) using the saved ctx buffer.
        self.save_attn_ctx = bool(meta.get("save_attn_ctx", False))
        # Mirror the worker-side ``save_resid_mid`` flag — drives whether the
        # forward remat skips the O projection (+ its TP all-reduce) and reads
        # the saved post-attention residual instead.
        self.save_resid_mid = bool(meta.get("save_resid_mid", False))

        # With enable_lora, vLLM wraps the projections, so the frozen base weight is
        # named e.g. "...qkv_proj.base_layer.weight". Normalize by stripping the
        # ".base_layer" infix so we can look weights up by their logical names
        # (norms/lm_head aren't wrapped, so they pass through unchanged).
        bn = {k.replace(".base_layer.", "."): v for k, v in base.items()}

        def bw(name):
            if name not in bn:
                raise KeyError(
                    f"base weight {name!r} not found (have e.g. "
                    f"{[k for k in list(bn)[:4]]} … {len(bn)} keys)")
            return bn[name]

        # The worker names the LM head / final norm it shared (tied-embedding
        # models expose the head as ``model.embed_tokens.weight``).
        self.lm_w = bw(meta.get("lm_head_key") or "lm_head.weight")
        self.norm_w = bw(meta.get("norm_weight_key") or "model.norm.weight")
        # Bulk backward compute dtype: model dtype (bf16) by default, fp32 if the
        # config flag is set. Attention core / RMSNorm-bwd / LM-head stay fp32.
        self.base_dtype = self.lm_w.dtype
        self.bwd_dtype = torch.float32 if meta.get("backward_fp32") else self.base_dtype

        # Per-layer views of the frozen base weights, per the family's map.
        # Fused ([out,in]) tensors are sliced once into q/k/v and gate/up by
        # this rank's local widths; every other entry is used as-is.
        for i in range(self.L):
            p = f"model.layers.{i}."
            lw: dict = {}
            for key, rel in fam.layer_weights.items():
                t = bw(p + rel)
                if key == "qkv":
                    lw["q"] = t[:self.q_size]
                    lw["k"] = t[self.q_size:self.q_size + self.kv_size]
                    lw["v"] = t[self.q_size + self.kv_size:]
                elif key == "gate_up":
                    lw["gate"] = t[:self.inter]
                    lw["up"] = t[self.inter:]
                else:
                    lw[key] = t
            self.base[i] = lw

        # FT adapter -> per-layer/proj fp32 master nn.Parameters (child-owned clone).
        # [DeltaServe] Phase 7 / M2: the adapter is loaded FULL from disk by the
        # worker, but under TP each rank owns only a shard, matching how vLLM
        # shards the served LoRA buffers we publish into (M4):
        #   q/k/v (column-parallel): B [out_full, r] sharded on output rows;
        #                            A [r, hidden]   replicated (full).
        #   o     (row-parallel):    A [r, in_full] sharded on input cols;
        #                            B [hidden, r]   replicated (full).
        # The head-partition is contiguous per rank (vLLM's QKV/o layout), so the
        # shard is a single contiguous slice at offset tp_rank * local_width.
        # tp_size=1 → lora_shard_slice is identity → single-GPU masters unchanged.
        params: list[nn.Parameter] = []
        for key, t in ft.items():
            m = _LORA_KEY_RE.search(key)
            if m is None:
                continue
            layer, proj, ab = int(m.group(1)), m.group(2), m.group(3)
            shard = lora_shard_slice(proj, ab, t, self.tp_rank, self.tp_size,
                                     self.q_size, self.kv_size)
            # fp32 master, contiguous so the fused AdamW + IPC publish are happy.
            param = nn.Parameter(shard.detach().clone().float().contiguous())
            self.lora.setdefault(layer, {}).setdefault(proj, {})[ab] = param
            params.append(param)

        # ``fused=True`` uses PyTorch's CUDA-fused AdamW kernel — one launch
        # for all LoRA tensors (8 per layer) instead of per-tensor dispatch.
        # Numerically identical to the default AdamW; cuts a few ms per
        # backward. Requires CUDA tensors (all our LoRA masters are on the
        # worker device).
        self.optimizer = torch.optim.AdamW(
            params, lr=float(meta["learning_rate"]), betas=(0.9, 0.999),
            weight_decay=float(meta["weight_decay"]), fused=True)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=float(meta["gamma"]))
        dprint(
            f"[backward] {fam.name} built: {self.L} layers, {len(params)} LoRA "
            f"tensors, scaling={self.scaling}, lr={meta['learning_rate']}, "
            f"wd={meta['weight_decay']}, gamma={meta['gamma']}"
        )

        # Persistent grad_qh/kh/vh buffers for the eager attn_backward_core path
        # (passed via attn_backward_core's optional grad_*_buf kwargs). Sized at
        # s_max so they fit any backward, sliced to [:n] per call. Saves 3·L
        # zero-fill kernel launches per backward. The graphed path has its own
        # static buffers in GraphedBackward.
        s_max = int(meta.get("max_saved_finetuning_tokens", 0))
        if s_max > 0:
            dev = self.lm_w.device
            cdt = self.bwd_dtype
            self._grad_qh_buf = torch.zeros((s_max, self.Hq, self.Hd),
                                            dtype=cdt, device=dev)
            self._grad_kh_buf = torch.zeros((s_max, self.Hkv, self.Hd),
                                            dtype=cdt, device=dev)
            self._grad_vh_buf = torch.zeros((s_max, self.Hkv, self.Hd),
                                            dtype=cdt, device=dev)
        else:
            self._grad_qh_buf = self._grad_kh_buf = self._grad_vh_buf = None

        if self.tp_size > 1 and self.lora:
            dev = self.lm_w.device
            self._comm = CommQueue(self._all_reduce, dev)
            self._bucket = FactorBucket(
                self.lora, self.L, self.bwd_dtype, dev,
                group_layers=int(meta.get("bucket_layers", BUCKET_LAYERS)))
            dprint(f"[backward] M4.3 factor-grad bucketing: {self._bucket.num_groups} "
                   f"bucket(s) for {len(self._bucket.order)} layers "
                   f"({self._bucket.flat.numel() * self._bucket.flat.element_size() / 2**20:.1f} MB), "
                   f"comm stream {'on' if self._comm.stream is not None else 'off (CPU)'}"
                   f"{' [SYNC mode]' if self._comm.sync_mode else ''}"
                   f"{f' [delay {self._comm.delay_cycles} cycles]' if self._comm.delay_cycles else ''}")

        self._maybe_build_graph_runner(meta)

    def _maybe_build_graph_runner(self, meta) -> None:
        """[Phase 5] Optional CUDA-graph runner. Per-layer forward + FFN-bwd
        graphs and one shared padded-attn-bwd graph, pre-captured at startup.
        Falls back per-layer on capture or shape-fit failure (gradient values
        unchanged in either case)."""
        if not meta.get("backward_cuda_graph"):
            return
        # [DeltaServe] Phase 7 / M5: the graph path is TP-safe — no collective
        # is captured. The runner splits the forward graph at the o-proj
        # output and reduces ``static_o`` eagerly (reading ``self._all_reduce``),
        # and the family's ``layer_backward_graphed`` issues the six backward
        # reduces from eager code around the two backward replays.
        if self.family.layer_backward_graphed is None:
            dprint(f"[backward] backward_cuda_graph requested but the "
                   f"{self.family.name} family has no graphed backward; "
                   "keeping eager")
            return
        s_max = int(meta.get("max_saved_finetuning_tokens", 0))
        bn_max = int(meta.get("backward_cuda_graph_attn_bn_max", 8))
        l_max = int(meta.get("backward_cuda_graph_attn_l_max", 64))
        if s_max <= 0:
            dprint("[backward] backward_cuda_graph requested but "
                   "max_saved_finetuning_tokens not provided; keeping eager")
            return
        from vllm.deltaserve.bwd_services.common.graph import GraphedBackward

        try:
            self.graph_runner = GraphedBackward(
                self, s_max=s_max, bn_max=bn_max, l_max=l_max)
        except Exception as e:  # noqa: BLE001
            import traceback
            dprint(f"[backward] graph runner init failed: {e}; "
                   f"falling back to eager backward")
            traceback.print_exc()
            self.graph_runner = None

    def _layer_weights(self, i: int) -> dict:
        """Base (compute dtype) + fp32 LoRA params for layer i, as one dict."""
        lw = dict(self.base[i])
        ld = self.lora.get(i, {})
        for proj in LORA_PROJS:
            lw[proj + "A"] = ld.get(proj, {}).get("A")
            lw[proj + "B"] = ld.get(proj, {}).get("B")
        return lw

    @torch.no_grad()
    def _diag_remat_check(self, activations, n, seq_lens, b_start, cos, sin):
        """[TP DIAG, temporary] Rematerialize the FULL forward from layer_in[0]
        (the embedding output — model-independent) using the CURRENT LoRA masters,
        and compare against the activations vLLM's REAL forward captured. If the
        backward's model view matches the SERVED model these agree to ~bf16 noise;
        a growing divergence localizes where publish/serve disagrees with training."""
        li = activations.get("layer_in")
        if not li:
            return
        gu = activations.get("mlp_gate_up")
        x = li[0][:n]
        rows = []
        for i in range(self.L):
            lw = self._layer_weights(i)
            c = self._layer_forward(x, lw, self.scaling, cos, sin, seq_lens,
                                    b_start, self.dims, self.eps,
                                    all_reduce=self._all_reduce)
            dg = 0.0
            if gu:
                rg = gu[i][:n].float()
                inter = rg.shape[-1] // 2
                num = (c["gate"].float() - rg[:, :inter]).abs().max().item()
                dg = num / (rg[:, :inter].abs().max().item() + 1e-6)
            h = F.silu(c["gate"]) * c["up"]
            ffn = F.linear(h, lw["down"])
            if self._all_reduce is not None:
                ffn = self._all_reduce(ffn)
            x = c["resid_mid"] + ffn
            ref = (li[i + 1][:n] if i + 1 < self.L
                   else activations["final_in"][:n]).float()
            d = (x.float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
            rows.append((i, d, dg))
        head = " ".join(f"L{i}={d:.1e}/gu{g:.1e}" for i, d, g in rows[:3])
        tail = " ".join(f"L{i}={d:.1e}" for i, d, _ in rows[-2:])
        dprint(f"[tpdiag] rank={self.tp_rank} remat-vs-captured rel|d|: "
               f"{head} ... {tail}")

    # -- the real backward (overrides base.process_backward) ------------------

    def process_backward(self, activations, sample_lens, n, epoch):
        self._maybe_pause()
        if not self._built:
            raise RuntimeError(
                f"{self.family.name} backward state not built (weights not shared)")
        # Mode tag for the one-line cycle log (set in BackwardService dispatcher).
        self._last_mode = "graph" if self.graph_runner is not None else "eager"
        seq_lens = [int(s) for s in sample_lens]
        b_start, acc = [], 0
        for s in seq_lens:
            b_start.append(acc)
            acc += s
        device = activations["final_in"].device
        positions = torch.cat([
            torch.arange(s, device=device) for s in seq_lens]) if seq_lens else \
            torch.zeros(0, device=device)
        cos, sin = rope_cos_sin(positions, self.Hd, self.theta)
        ids = activations["concat_input_ids"][:n]
        if os.environ.get("DSERVE_TP_DIAG") and self._diag_count < 5:
            self._diag_count += 1
            self._diag_remat_check(activations, n, seq_lens, b_start,
                                   cos, sin)

        self.optimizer.zero_grad(set_to_none=True)

        # Head: loss + grad w.r.t. final_in (pre-final-norm residual = layer_in[L]).
        loss, n_valid, g = head_backward(
            activations["final_in"][:n], self.lm_w, self.norm_w, self.eps,
            ids, seq_lens, b_start, self.vocab)

        # Saved MLP pre-activations (gate||up) per layer, if captured in the forward
        # — lets the remat skip the gate_up matmul (the layer's biggest recompute).
        saved_gu = activations.get("mlp_gate_up")
        # Saved post-RoPE q/k/v per layer (opt-in via finetune.save_attn_qkv) —
        # when present AND exact for this family, lets the remat skip the
        # second-biggest layer recompute: Q/K/V projection + RoPE.
        if self.save_attn_qkv:
            saved_qh_all = activations.get("attn_qh")
            saved_kh_all = activations.get("attn_kh")
            saved_vh_all = activations.get("attn_vh")
        else:
            saved_qh_all = saved_kh_all = saved_vh_all = None
        # Saved attention context (o_proj input) per layer (opt-in via
        # finetune.save_attn_ctx) — lets the remat skip the attention forward
        # (scores/softmax/AV). None when the feature is off.
        saved_ctx_all = activations.get("attn_ctx")
        # Saved post-attention residual per layer (opt-in via
        # finetune.save_resid_mid) — lets the remat skip the O projection and,
        # under TP, the forward's only all-reduce. None when the feature is off.
        saved_rm_all = activations.get("resid_mid") if self.save_resid_mid else None

        # If the graphed runner is attached, build per-backward scatter indices
        # + key-pad mask once (re-used across all L layers' padded-attention
        # replays). Eager fallback for batches that overflow the padded budget.
        runner = self.graph_runner
        if runner is not None:
            runner.begin_backward(n, seq_lens, b_start)

        layer_forward = self._layer_forward
        layer_backward = self._layer_backward
        layer_backward_graphed = self.family.layer_backward_graphed
        # M4.3: under TP the families leave the replicated factor grads
        # un-reduced; they go to the bucket below.
        bucket, comm = self._bucket, self._comm
        reduce_factors = bucket is None

        # Per-layer manual backward, chaining the input gradient down the stack.
        # Graph runner takes the forward+backward fast path when:
        #   - runner is attached, AND
        #   - the padded-attention budget fits this backward (decided once
        #     in begin_backward → ``runner._attn_fit``), AND
        #   - this layer's saved gate||up is available (production default).
        # Otherwise the layer runs the eager ``layer_forward`` + the
        # appropriate backward path.
        for i in reversed(range(self.L)):
            if runner is None:
                # Eager path. Pause at the layer boundary before the layer's
                # compute — original yield cadence.
                self._maybe_pause()
            lw = self._layer_weights(i)
            x = activations["layer_in"][i][:n]
            gu = saved_gu[i][:n] if saved_gu else None
            qh_i = saved_qh_all[i][:n] if saved_qh_all else None
            kh_i = saved_kh_all[i][:n] if saved_kh_all else None
            vh_i = saved_vh_all[i][:n] if saved_vh_all else None
            ctx_i = saved_ctx_all[i][:n] if saved_ctx_all else None
            rm_i = saved_rm_all[i][:n] if saved_rm_all else None
            with torch.no_grad():
                if runner is not None and runner._attn_fit and gu is not None:
                    # Graphed forward: writes cache straight into the
                    # static buffers Graph A / Graph B already read.
                    cache = runner.forward(
                        i, lw, x, gu, n,
                        saved_qh=qh_i, saved_kh=kh_i, saved_vh=vh_i,
                        saved_ctx=ctx_i, saved_resid_mid=rm_i)
                else:
                    cache = layer_forward(
                        x, lw, self.scaling, cos, sin,
                        seq_lens, b_start, self.dims, self.eps,
                        saved_gate_up=gu,
                        saved_qh=qh_i, saved_kh=kh_i, saved_vh=vh_i,
                        saved_ctx=ctx_i, saved_resid_mid=rm_i,
                        all_reduce=self._all_reduce)  # M3: o_proj reduce (TP)
                if runner is None:
                    grad_x, grads = layer_backward(
                        g, cache, lw, self.scaling, cos, sin,
                        seq_lens, b_start, self.dims, self.eps,
                        cdt=self.bwd_dtype,
                        grad_qh_buf=self._grad_qh_buf,
                        grad_kh_buf=self._grad_kh_buf,
                        grad_vh_buf=self._grad_vh_buf,
                        all_reduce=self._all_reduce,  # M3: grad reduces (TP)
                        reduce_factors=reduce_factors)
                else:
                    grad_x, grads = layer_backward_graphed(
                        runner, self._maybe_pause, i, g, cache, lw,
                        self.scaling, cos, sin, seq_lens, b_start,
                        self.dims, self.eps, self.bwd_dtype,
                        all_reduce=self._all_reduce,  # M5: grad reduces (TP)
                        reduce_factors=reduce_factors)
            ld = self.lora.get(i, {})
            if bucket is None:
                # tp=1: write grads to the fp32 masters (cast up from the bulk
                # compute dtype; PEFT layout matches), then per-layer clip to
                # 1.0 (DeltaServe). Unchanged single-GPU path.
                layer_params = []
                for proj in LORA_PROJS:
                    pa, pb = ld.get(proj, {}).get("A"), ld.get(proj, {}).get("B")
                    if pa is not None:
                        pa.grad = grads[proj + "A"].float()
                        pb.grad = grads[proj + "B"].float()
                        layer_params += [pa, pb]
                if layer_params:
                    torch.nn.utils.clip_grad_norm_(layer_params, max_norm=1.0)
            else:
                # [M4.3] Sharded factor grads are already this rank's shard →
                # to the masters now. Replicated ones → the bucket; a group's
                # slice is reduced on the comm stream as soon as its last
                # layer is done, overlapping the next group's compute.
                for proj in LORA_PROJS:
                    pa, pb = ld.get(proj, {}).get("A"), ld.get(proj, {}).get("B")
                    if pa is None:
                        continue
                    if proj == "o":
                        pa.grad = grads["oA"].float()      # sharded (input cols)
                    else:
                        pb.grad = grads[proj + "B"].float()  # sharded (output rows)
                bucket.stash(i, grads)
                sl = bucket.group_done(i)
                if sl is not None:
                    comm.submit(sl)
            g = grad_x

        if bucket is not None:
            # [M4.3] All buckets landed → replicated grads to the masters,
            # then the rank-symmetric per-layer clip (one [L] all-reduce of
            # the sharded factors' squared norms), then the step.
            comm.wait()
            for i, ld in self.lora.items():
                for key in REPLICATED_FACTORS:
                    prm = ld.get(key[0], {}).get(key[1])
                    if prm is not None:
                        prm.grad = bucket.view(i, key).float()
            clip_layers_symmetric_(self.lora, self.L, self._all_reduce, max_norm=1.0)

        self.optimizer.step()
        if epoch > self.current_epoch:
            self.scheduler.step()
            self.current_epoch = epoch

        # Publish the updated fp32 master into vLLM's served LoRA buffers (the exact
        # tensors inference reads). Safe with no locking: FT admission is closed for
        # the whole backward, so the adapter is idle until the done-reply reopens it.
        self._publish_to_served()

        return loss, n_valid

    @torch.no_grad()
    def _publish_to_served(self) -> None:
        """Write the trained fp32 master into vLLM's served LoRA stacked buffers
        (DeltaServe's load-time refresh): per (layer, proj), clamp + cast to the
        served dtype, with α/r baked into B (vLLM applies scale=1). No transpose —
        PEFT A [r,in] / B [out,r] match vLLM's stacked layout.

        The scaling convention is load-bearing: vLLM's punica forward calls
        ``add_lora_linear(..., 1.0, ...)`` (see
        ``vllm/lora/layers/base_linear.py:226``) — the third positional arg
        ``scale`` is hardcoded to 1.0, NOT taken from the adapter's stored
        ``LoRALayerWeights.scaling`` (alpha/r). We multiply ``pb * self.scaling``
        here so the net inference effect is
        ``(x @ A) @ (B·s) · 1.0 = s · (x · A · B)`` — the correct PEFT LoRA
        forward with one scaling application. If vLLM ever switches that
        callsite to pass the per-adapter scaling instead of 1.0, this publish
        MUST drop the ``* self.scaling`` multiplication or inference will see
        ``scaling²`` and silently degrade output quality.

        Locking: relies on the FT adapter being pre-loaded into a dedicated
        served-LoRA slot at startup (see ``_maybe_share_ft_served_lora`` in
        ``v1/worker/gpu_worker.py``). No inference request can land on this
        slot, so the in-place write is race-free with concurrent inference.
        FT admission is closed for the entire backward cycle (coordinator
        keeps ``pending_backward=True`` until the child acks), so even FT
        requests can't read the slot mid-publish."""
        if not self.lora_buffers:
            return
        slot = int(self.lora_buffers["slot"])
        for i, projd in self.lora_buffers["layers"].items():
            ld = self.lora.get(int(i), {})
            for proj, buf in projd.items():
                pa = ld.get(proj, {}).get("A")
                pb = ld.get(proj, {}).get("B")
                if pa is None:
                    continue
                a_buf, b_buf = buf["a"], buf["b"]
                r, in_dim = pa.shape          # PEFT A [r, in]
                out_dim = pb.shape[0]          # PEFT B [out, r]
                a_buf[slot, 0, :r, :in_dim].copy_(
                    pa.clamp(-6.5e4, 6.5e4).to(a_buf.dtype))
                b_buf[slot, 0, :out_dim, :r].copy_(
                    (pb * self.scaling).clamp(-6.5e4, 6.5e4).to(b_buf.dtype))

    # -- capture correctness checks (debug-gated) -----------------------------

    def verify_activations(self, activations: dict, n: int) -> None:
        meta = self.shared["meta"]
        base = self.shared["base"] or {}
        ids = activations["concat_input_ids"][:n].long()
        layer_in = activations.get("layer_in") or []
        embed_w = base.get(meta.get("embed_weight_key"))
        if layer_in and embed_w is not None:
            d = (layer_in[0][:n].float() - embed_w[ids].float()).abs().max().item()
            scale = embed_w[ids].float().abs().max().item() + 1e-6
            dprint(f"[verify] layer_in[0] vs embed[ids]: max|Δ|={d:.3e} "
                   f"rel={d / scale:.3e} {'OK' if d / scale < _TOL else 'FAIL'}")
        final_in = activations.get("final_in")
        final_hidden = activations.get("final_hidden")
        norm_w = base.get(meta.get("norm_weight_key"))
        if final_in is not None and final_hidden is not None and norm_w is not None:
            h = rmsnorm(final_in[:n], norm_w, float(meta.get("rms_norm_eps", 1e-5)))
            fh = final_hidden[:n].float()
            d = (h.float() - fh).abs().max().item()
            scale = fh.abs().max().item() + 1e-6
            dprint(f"[verify] RMSNorm(final_in) vs final_hidden: max|Δ|={d:.3e} "
                   f"rel={d / scale:.3e} {'OK' if d / scale < _TOL else 'FAIL'}")
