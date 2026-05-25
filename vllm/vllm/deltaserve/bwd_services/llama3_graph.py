# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CUDA-graph capture/replay for the Llama-3 LoRA SFT backward (Phase 5).

Ports DeltaServe ``models/llama/SFT_service_graph.py`` onto our backward
service. Two graphs per layer:

  * **FFN graph** — the FFN-block backward (rmsnorm post_ln-bwd + silu/sigmoid
    + down/gate/up GEMMs + residual). Shape-stable at the activation-buffer
    width (``s_max = max_saved_finetuning_tokens``).
  * **Padded-attention graph** — the per-sample scores/softmax/dQ/dK/dV core,
    rewritten against a padded ``[bn_max, l_max, H_*, Hd]`` view of the
    cached qh/kh/vh + grad_ctx (scatter in, compute, gather out). Bounds come
    from ``finetune.backward_cuda_graph_attn_{bn_max,l_max}``.

Everything outside these two regions stays eager (forward remat, O-proj bwd,
RoPE bwd, Q/K/V-proj bwd, in_ln rmsnorm bwd, optimizer/publish). ``_maybe_pause``
is the caller's responsibility — they invoke it BETWEEN our two replays each
layer so the GPU-yield contract is preserved at the same per-layer cadence as
the all-eager path.

Capture-failure / shape-fit-failure handling per layer is silent eager fallback:
the runner adds the layer id to ``ffn_failed`` / ``attn_failed`` and forwards
to ``ffn_backward_core`` / ``attn_backward_core`` from ``llama3.py`` for the
rest of the run. Gradient values are bit-identical to eager (same math, just
replayed under fixed shapes).

Persistent static IO buffers live OUTSIDE the shared graph pool — the
DeltaServe reference's load-bearing rule (``SFT_service_graph.py`` lines
111–113 vs ``graph_pool_handle()`` at line 114) for avoiding pool-aliasing
NaN traps. We don't capture LoRA-grad writes here (Q/K/V/O proj backwards stay
eager, so ``nn.Parameter.grad`` ownership doesn't change vs the eager path).
"""

from __future__ import annotations

import math
import time

import torch

from vllm.deltaserve import dprint
from vllm.deltaserve.bwd_services.llama3 import (
    attn_backward_core,
    ffn_backward_core,
)


class Llama3GraphedBackward:
    """Per-layer FFN + padded-attention CUDA graphs for the Llama-3 SFT backward.

    Holds one ``torch.cuda.CUDAGraph`` per layer per region. Static IO buffers
    are allocated once at construction and reused across layers (layers run
    sequentially in ``process_backward``). Graph capture is lazy on first
    ``ffn_backward`` / ``attn_backward`` call for that layer."""

    # 2 eager warmup iterations before capture (prime Triton/cuBLAS autotuners).
    _N_WARMUP = 2

    def __init__(self, svc, s_max: int, bn_max: int, l_max: int) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "[backward_cuda_graph] CUDA is not available in this process")

        self.svc = svc
        self.s_max = int(s_max)
        self.bn_max = int(bn_max)
        self.l_max = int(l_max)
        self.device = torch.device(f"cuda:{svc.device_index}")
        self.cdt = svc.bwd_dtype
        self.model_dtype = svc.base_dtype
        self.D = int(svc.D)
        self.inter = int(svc.inter)
        self.Hq = int(svc.Hq)
        self.Hkv = int(svc.Hkv)
        self.Hd = int(svc.Hd)
        self.kv_repeat = self.Hq // self.Hkv
        self.eps = float(svc.eps)
        self.scale = 1.0 / math.sqrt(self.Hd)

        # Per-layer FFN graphs (each binds layer-specific frozen weights:
        # lw["down"], lw["gate"], lw["up"], lw["post_ln"]). Capture/replay
        # failure for a given layer is recorded in ``ffn_failed`` and that
        # layer falls back to eager.
        self._ffn_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.ffn_failed: set[int] = set()
        # Padded-attention graph: ONE captured graph reused for all layers.
        # ``_padded_attn_core`` reads only static IO + masks + indices — NO
        # layer-specific weights — so 32 identical per-layer captures would
        # be wasted memory and startup time. If capture or replay fails, all
        # layers fall back to the eager ``attn_backward_core``.
        self._attn_graph: torch.cuda.CUDAGraph | None = None
        self.attn_failed: bool = False

        # Single graph pool for both regions. Static IO is allocated BEFORE
        # any capture call so it lives in the default caching allocator, not
        # the pool — load-bearing for avoiding LoRA-grad NaN under pool reuse
        # (DeltaServe SFT_service_graph.py lines 111–113 / 114 comment).
        self._graph_pool = torch.cuda.graph_pool_handle()
        self._stream = torch.cuda.Stream(device=self.device)

        self._alloc_static_buffers()

        # Per-backward state (refilled in ``begin_backward``).
        self._cur_seq_lens: list[int] = []
        self._cur_b_start: list[int] = []
        self._cur_n: int = 0
        # Whether the current backward fits the padded-attention budget.
        self._attn_fit: bool = False
        self._fallback_warned: bool = False

        dprint(
            f"[bwd-graph] runner ready: s_max={self.s_max}, "
            f"bn_max={self.bn_max}, l_max={self.l_max}, L={int(svc.L)}, "
            f"model_dtype={self.model_dtype}, cdt={self.cdt}")

        # Eagerly capture every per-layer graph up-front so the first real
        # backward pays only replay cost (no warmup + capture stalls landing on
        # a live co-serving step). The static buffers are zero-initialized; the
        # captured kernel launches only depend on shapes and tensor addresses,
        # not values — replay populates the buffers with real data and produces
        # correct outputs. Mirrors DeltaServe ``GraphedBackwardRunner.prepare``.
        self.prepare()

    # ---------------------------------------------------------------- buffers

    def _alloc_static_buffers(self) -> None:
        """Persistent static IO for both graphs. Outside the graph pool."""
        s, D, inter = self.s_max, self.D, self.inter
        Hq, Hkv, Hd = self.Hq, self.Hkv, self.Hd
        bn, lm = self.bn_max, self.l_max
        dev, mdt, cdt = self.device, self.model_dtype, self.cdt

        # ---- FFN graph IO ----
        # Inputs: chained gradient + cached forward intermediates the FFN
        # backward reads. The FFN-bwd math casts the cached tensors to cdt
        # internally, so we keep them in model dtype to match what the eager
        # ``layer_forward`` produces (no dtype conversion in the copy-in path).
        self.static_g = torch.zeros((s, D), dtype=cdt, device=dev)
        self.static_resid_mid = torch.zeros((s, D), dtype=mdt, device=dev)
        self.static_gate = torch.zeros((s, inter), dtype=mdt, device=dev)
        self.static_up = torch.zeros((s, inter), dtype=mdt, device=dev)
        # Output: grad to resid_mid, in cdt. Caller reads [:n] as a view (the
        # next layer's replay overwrites this, but by then we've consumed it
        # in the eager O-projection backward + the final ``+ grad_resid_mid``).
        self.static_grad_resid_mid = torch.zeros((s, D), dtype=cdt, device=dev)

        # ---- Padded-attention graph IO ----
        # Inputs (scatter target): qh/kh/vh + grad_ctx at [bn_max, l_max, ...].
        # qh/kh/vh come from the forward cache (model dtype); grad_ctx comes
        # from the upstream _proj_backward in cdt, so it stays in cdt to match
        # what eager attn_backward_core would see (no bf16 round-trip on the
        # backward gradient when cdt=fp32).
        self.static_qh_pad = torch.zeros((bn, lm, Hq, Hd), dtype=mdt, device=dev)
        self.static_kh_pad = torch.zeros((bn, lm, Hkv, Hd), dtype=mdt, device=dev)
        self.static_vh_pad = torch.zeros((bn, lm, Hkv, Hd), dtype=mdt, device=dev)
        self.static_grad_ctx_pad = torch.zeros((bn, lm, Hq, Hd), dtype=cdt,
                                               device=dev)
        # Scatter / gather indices: every flat row k carries (bn_idx[k], pos_idx[k])
        # = its (sample, within-sample-position). Tail rows (k ≥ n) point at
        # (0,0) and the scatter uses ``index_put_(accumulate=True)`` — combined
        # with the s_max-shaped flat input rows already zeroed at the tail, this
        # adds zero to slot (0,0) and is a no-op (matches DeltaServe's tail
        # sentinel pattern). The corresponding gather rows are read into the
        # tail of the flat output and ignored by the caller (only [:n] is used).
        self.static_bn_idx = torch.zeros((s,), dtype=torch.long, device=dev)
        self.static_pos_idx = torch.zeros((s,), dtype=torch.long, device=dev)
        # Masks: causal is static (triu, one-time), key-pad varies per backward.
        self.static_causal_mask = torch.triu(
            torch.ones((lm, lm), dtype=torch.bool, device=dev), diagonal=1)
        # True = padded (mask out); refilled per-backward.
        self.static_key_pad_mask = torch.ones((bn, lm), dtype=torch.bool, device=dev)
        # Outputs (flat layout, cdt): grad_qh/kh/vh — what attn_backward_core returns.
        self.static_grad_qh = torch.zeros((s, Hq, Hd), dtype=cdt, device=dev)
        self.static_grad_kh = torch.zeros((s, Hkv, Hd), dtype=cdt, device=dev)
        self.static_grad_vh = torch.zeros((s, Hkv, Hd), dtype=cdt, device=dev)

    # ---------------------------------------------------------------- prepare

    def prepare(self) -> None:
        """Pre-capture all per-layer FFN + padded-attention graphs with dummy
        (zero) static-buffer inputs. Capture only depends on shapes and tensor
        addresses, so values don't matter — replay-time staging produces the
        right gradients. Any per-layer capture failure is silently recorded in
        ``ffn_failed`` / ``attn_failed`` (that layer falls back to eager).

        Cost: roughly ``L * (warmup + capture)`` per region, paid once at child
        startup before the first ``share_activations`` ack — outside the
        co-serving loop, so no inference TTFT impact."""
        L = int(self.svc.L)
        t0 = time.perf_counter()

        # FFN: one graph per layer (different lw["down"]/gate/up addresses).
        for i in range(L):
            try:
                self._capture_ffn(i, self.svc._layer_weights(i))
            except Exception as e:  # noqa: BLE001
                self.ffn_failed.add(i)
                dprint(f"[bwd-graph] FFN layer {i} startup capture failed: "
                       f"{e}; will fall back to eager at runtime")

        # Padded-attention: ONE captured graph reused across all L layers
        # (the core has no layer-dependent weights). Arm a fitting dummy
        # batch so scatter indices and key-pad mask are valid for capture
        # (kernels still see zero values, fine — only shapes/addresses are
        # captured).
        dummy_l = min(self.l_max, self.s_max)
        self.begin_backward(dummy_l, [dummy_l], [0])
        try:
            self._capture_attn()
        except Exception as e:  # noqa: BLE001
            self.attn_failed = True
            dprint(f"[bwd-graph] attn startup capture failed: {e}; "
                   f"all layers will fall back to eager at runtime")
        # Reset per-backward state — the real first backward calls
        # begin_backward again with its own seq_lens.
        self._cur_n = 0
        self._cur_seq_lens = []
        self._cur_b_start = []
        self._attn_fit = False
        # Zero the suppress-once warning flag in case the dummy batch's
        # log message printed; we want the real overflow warning at runtime.
        self._fallback_warned = False

        captured_ffn = L - len(self.ffn_failed)
        attn_status = "0/1" if self.attn_failed else "1/1"
        elapsed = (time.perf_counter() - t0) * 1000.0
        dprint(
            f"[bwd-graph] pre-captured FFN {captured_ffn}/{L} + "
            f"attn {attn_status} in {elapsed:.0f}ms")

    # ----------------------------------------------------------- per-backward

    def begin_backward(self, n: int, seq_lens: list[int],
                       b_start: list[int]) -> None:
        """Per-backward state setup. Decides if the padded-attention path can
        run for this batch and (if so) builds scatter indices + key-pad mask.

        Called once per ``process_backward`` before the per-layer loop."""
        self._cur_n = int(n)
        self._cur_seq_lens = list(seq_lens)
        self._cur_b_start = list(b_start)

        bn = len(seq_lens)
        max_l = max(seq_lens) if bn > 0 else 0
        self._attn_fit = (
            n <= self.s_max
            and bn <= self.bn_max
            and max_l <= self.l_max
        )

        if not self._attn_fit:
            if not self._fallback_warned:
                self._fallback_warned = True
                dprint(
                    f"[bwd-graph] padded-attention overflow — falling back to "
                    f"eager attn for this and similar batches "
                    f"(bn={bn}/{self.bn_max}, max_l={max_l}/{self.l_max}, "
                    f"n={n}/{self.s_max}). Tune "
                    f"finetune.backward_cuda_graph_attn_{{bn_max,l_max}} if "
                    f"this is frequent.")
            return

        # Build bn_idx / pos_idx on CPU once, ship to GPU in one copy each.
        bn_cpu, pos_cpu = [], []
        for i, ln in enumerate(seq_lens):
            bn_cpu.extend([i] * int(ln))
            pos_cpu.extend(range(int(ln)))
        pad = self.s_max - n
        if pad > 0:
            bn_cpu.extend([0] * pad)
            pos_cpu.extend([0] * pad)
        self.static_bn_idx.copy_(
            torch.tensor(bn_cpu, dtype=torch.long, device="cpu"),
            non_blocking=True)
        self.static_pos_idx.copy_(
            torch.tensor(pos_cpu, dtype=torch.long, device="cpu"),
            non_blocking=True)

        # key_pad_mask: True at padded positions, False at real ones.
        self.static_key_pad_mask.fill_(True)
        if n > 0:
            valid_bn = self.static_bn_idx[:n]
            valid_pos = self.static_pos_idx[:n]
            self.static_key_pad_mask[valid_bn, valid_pos] = False

    # ------------------------------------------------------------- FFN region

    def _capture_ffn(self, layer_id: int, lw: dict) -> None:
        """Capture one per-layer FFN-backward graph. Static buffers must
        already be initialized to the values the warmup iterations will see
        (caller stages them via ``ffn_backward``'s copy-in path)."""
        # Warmup to prime cuBLAS/Triton autotuning before capture.
        for _ in range(self._N_WARMUP):
            _ = ffn_backward_core(
                self.static_g, {
                    "resid_mid": self.static_resid_mid,
                    "gate": self.static_gate,
                    "up": self.static_up,
                }, lw, self.eps, self.cdt)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self._graph_pool, stream=self._stream):
            out = ffn_backward_core(
                self.static_g, {
                    "resid_mid": self.static_resid_mid,
                    "gate": self.static_gate,
                    "up": self.static_up,
                }, lw, self.eps, self.cdt)
            self.static_grad_resid_mid.copy_(out)
        self._ffn_graphs[layer_id] = g

    def ffn_backward(self, layer_id: int, g_actual: torch.Tensor,
                     cache: dict, lw: dict) -> torch.Tensor:
        """Graphed FFN-backward replay. Returns a VIEW of the static output
        (caller consumes it before the next layer's replay; no clone)."""
        n = g_actual.shape[0]

        if layer_id in self.ffn_failed:
            return ffn_backward_core(g_actual, cache, lw, self.eps, self.cdt)

        # Stage inputs into static buffers (zero the tail so the captured math
        # operates on a fully-defined [s_max, ...] tensor with junk-free pads).
        self._stage_ffn_inputs(g_actual, cache, n)

        if layer_id not in self._ffn_graphs:
            try:
                self._capture_ffn(layer_id, lw)
            except Exception as e:  # noqa: BLE001 — silent fall-through
                self.ffn_failed.add(layer_id)
                dprint(f"[bwd-graph] FFN capture failed for layer {layer_id}: "
                       f"{e}; falling back to eager for this layer")
                return ffn_backward_core(g_actual, cache, lw, self.eps, self.cdt)

        try:
            self._ffn_graphs[layer_id].replay()
        except Exception as e:  # noqa: BLE001
            self.ffn_failed.add(layer_id)
            dprint(f"[bwd-graph] FFN replay failed for layer {layer_id}: "
                   f"{e}; falling back to eager for this layer")
            return ffn_backward_core(g_actual, cache, lw, self.eps, self.cdt)

        return self.static_grad_resid_mid[:n]

    def _stage_ffn_inputs(self, g_actual: torch.Tensor, cache: dict,
                          n: int) -> None:
        s = self.s_max
        if g_actual.dtype != self.cdt:
            g_actual = g_actual.to(self.cdt)
        self.static_g[:n].copy_(g_actual)
        if n < s:
            self.static_g[n:].zero_()
        self.static_resid_mid[:n].copy_(cache["resid_mid"])
        if n < s:
            self.static_resid_mid[n:].zero_()
        self.static_gate[:n].copy_(cache["gate"])
        if n < s:
            self.static_gate[n:].zero_()
        self.static_up[:n].copy_(cache["up"])
        if n < s:
            self.static_up[n:].zero_()

    # ----------------------------------------------------- padded-attention

    def _padded_attn_core(self) -> None:
        """The shape-stable padded-attention backward CORE. Every tensor shape
        depends only on (bn_max, l_max, Hq, Hkv, Hd) — captureable as one CUDA
        graph per layer. Reads static_{qh,kh,vh,grad_ctx}_pad + masks/indices,
        writes static_grad_{qh,kh,vh} (flat layout via gather)."""
        bn, lm = self.bn_max, self.l_max
        Hq, Hkv, Hd = self.Hq, self.Hkv, self.Hd
        kv_repeat = self.kv_repeat

        # Combined mask: causal OR key-pad → [bn, 1, l, l] (broadcasts over heads).
        attn_mask = (self.static_causal_mask.unsqueeze(0).unsqueeze(0)
                     | self.static_key_pad_mask.unsqueeze(1).unsqueeze(1))

        # GQA: expand kh/vh from Hkv to Hq via repeat_interleave on the head dim.
        if kv_repeat != 1:
            kh_rep = self.static_kh_pad.repeat_interleave(kv_repeat, dim=2)
            vh_rep = self.static_vh_pad.repeat_interleave(kv_repeat, dim=2)
        else:
            kh_rep = self.static_kh_pad
            vh_rep = self.static_vh_pad

        # → [bn, Hq, l, Hd] fp32 (scores/softmax/dQ/dK/dV always fp32 — load-
        # bearing GQA precision rule).
        q_att = self.static_qh_pad.permute(0, 2, 1, 3).contiguous().float()
        k_att = kh_rep.permute(0, 2, 1, 3).contiguous().float()
        v_att = vh_rep.permute(0, 2, 1, 3).contiguous().float()
        grad_ctx_att = (self.static_grad_ctx_pad
                        .permute(0, 2, 1, 3).contiguous().float())

        scores = (q_att @ k_att.transpose(-1, -2)) * self.scale
        scores = scores.masked_fill(attn_mask, -1e9)
        att = torch.softmax(scores, dim=-1)
        # Fully-masked rows (entirely padded samples) → softmax = NaN; map to 0.
        att = torch.nan_to_num(att, nan=0.0)

        grad_att = grad_ctx_att @ v_att.transpose(-1, -2)        # [bn, Hq, l, l]
        grad_v_att = att.transpose(-1, -2) @ grad_ctx_att        # [bn, Hq, l, Hd]
        sm = (grad_att * att).sum(-1, keepdim=True)
        grad_scores = (att * (grad_att - sm)).masked_fill(attn_mask, 0.0)
        grad_q_att = (grad_scores @ k_att) * self.scale          # [bn, Hq, l, Hd]
        grad_k_att = (grad_scores.transpose(-1, -2) @ q_att) * self.scale

        # Back to [bn, l, H, Hd], reduce repeat for kh/vh, cast to cdt.
        grad_q_pad = grad_q_att.permute(0, 2, 1, 3).contiguous().to(self.cdt)
        grad_k_pad_rep = grad_k_att.permute(0, 2, 1, 3).contiguous()
        grad_v_pad_rep = grad_v_att.permute(0, 2, 1, 3).contiguous()
        if kv_repeat != 1:
            grad_k_pad = grad_k_pad_rep.view(
                bn, lm, Hkv, kv_repeat, Hd).sum(3).to(self.cdt)
            grad_v_pad = grad_v_pad_rep.view(
                bn, lm, Hkv, kv_repeat, Hd).sum(3).to(self.cdt)
        else:
            grad_k_pad = grad_k_pad_rep.to(self.cdt)
            grad_v_pad = grad_v_pad_rep.to(self.cdt)

        # Gather padded → flat: grad_*_pad[bn_idx[k], pos_idx[k]] → static_grad_*[k]
        # Tail rows (k ≥ n) gather from (0,0); caller only reads [:n].
        self.static_grad_qh.copy_(grad_q_pad[self.static_bn_idx, self.static_pos_idx])
        self.static_grad_kh.copy_(grad_k_pad[self.static_bn_idx, self.static_pos_idx])
        self.static_grad_vh.copy_(grad_v_pad[self.static_bn_idx, self.static_pos_idx])

    def _capture_attn(self) -> None:
        """Capture the shared padded-attention graph (one for all layers).
        Inputs must be staged in the static_*_pad buffers + indices +
        key_pad_mask before this. ``_padded_attn_core`` reads no layer-
        specific weights, so the single captured graph replays correctly
        for every layer's attn-backward."""
        for _ in range(self._N_WARMUP):
            self._padded_attn_core()
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self._graph_pool, stream=self._stream):
            self._padded_attn_core()
        self._attn_graph = g

    def _stage_attn_inputs(self, qh: torch.Tensor, kh: torch.Tensor,
                           vh: torch.Tensor, grad_ctx: torch.Tensor,
                           n: int) -> None:
        # Scatter flat [n, H_*, Hd] → padded [bn_max, l_max, H_*, Hd] via the
        # pre-built (bn_idx, pos_idx) at [:n]. Tail of static_*_pad is left as
        # whatever the previous layer wrote (the captured graph reads it but
        # the gather drops those rows). Zero the pads instead for cleanliness
        # and to guarantee fully-masked rows produce zero gradients regardless
        # of garbage in the padded buffers.
        self.static_qh_pad.zero_()
        self.static_kh_pad.zero_()
        self.static_vh_pad.zero_()
        self.static_grad_ctx_pad.zero_()
        if n > 0:
            valid_bn = self.static_bn_idx[:n]
            valid_pos = self.static_pos_idx[:n]
            self.static_qh_pad[valid_bn, valid_pos] = qh.to(self.model_dtype)
            self.static_kh_pad[valid_bn, valid_pos] = kh.to(self.model_dtype)
            self.static_vh_pad[valid_bn, valid_pos] = vh.to(self.model_dtype)
            self.static_grad_ctx_pad[valid_bn, valid_pos] = grad_ctx.to(self.cdt)

    def attn_backward(self, layer_id: int, qh: torch.Tensor, kh: torch.Tensor,
                      vh: torch.Tensor, grad_ctx: torch.Tensor,
                      seq_lens: list[int], b_start: list[int],
                      dims: tuple) -> tuple[torch.Tensor, torch.Tensor,
                                            torch.Tensor]:
        """Graphed padded-attention replay. Returns (grad_qh, grad_kh, grad_vh)
        as VIEWS of the static flat output buffers (consumed in this layer's
        iteration before the next layer overwrites them).

        Silent eager fallback if (a) the batch overflows the padded budget
        (decided once per backward in ``begin_backward``), or (b) the shared
        attn graph capture/replay raised at some point. ``layer_id`` is kept
        in the signature for the failure-log message but not used to select
        a graph (the same graph replays for every layer — see ``_capture_attn``
        docstring)."""
        n = qh.shape[0]

        if not self._attn_fit or self.attn_failed:
            return attn_backward_core(qh, kh, vh, grad_ctx,
                                      seq_lens, b_start, dims, self.cdt,
                                      grad_qh_buf=self.static_grad_qh,
                                      grad_kh_buf=self.static_grad_kh,
                                      grad_vh_buf=self.static_grad_vh)

        self._stage_attn_inputs(qh, kh, vh, grad_ctx, n)

        if self._attn_graph is None:
            try:
                self._capture_attn()
            except Exception as e:  # noqa: BLE001
                self.attn_failed = True
                dprint(f"[bwd-graph] attn capture failed at layer {layer_id}: "
                       f"{e}; all layers will fall back to eager")
                return attn_backward_core(qh, kh, vh, grad_ctx,
                                          seq_lens, b_start, dims, self.cdt,
                                          grad_qh_buf=self.static_grad_qh,
                                          grad_kh_buf=self.static_grad_kh,
                                          grad_vh_buf=self.static_grad_vh)

        try:
            self._attn_graph.replay()
        except Exception as e:  # noqa: BLE001
            self.attn_failed = True
            dprint(f"[bwd-graph] attn replay failed at layer {layer_id}: "
                   f"{e}; all layers will fall back to eager")
            return attn_backward_core(qh, kh, vh, grad_ctx,
                                      seq_lens, b_start, dims, self.cdt,
                                      grad_qh_buf=self.static_grad_qh,
                                      grad_kh_buf=self.static_grad_kh,
                                      grad_vh_buf=self.static_grad_vh)

        return (self.static_grad_qh[:n],
                self.static_grad_kh[:n],
                self.static_grad_vh[:n])
