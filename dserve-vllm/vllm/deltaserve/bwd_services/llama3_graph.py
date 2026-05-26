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
    _proj,
    apply_rope,
    attn_backward_core,
    ffn_backward_core,
    layer_forward,
    rmsnorm,
    rope_cos_sin,
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
        # Per-layer forward (rematerialization) graphs. Must be per-layer
        # because Q/K/V/O base + LoRA weights are layer-specific (same reason
        # FFN-bwd is per-layer). Captured against ``static_layer_in`` +
        # ``static_cos/sin`` + ``static_saved_gate_up`` reading layer-i
        # weights; writes into static_resid_mid/gate/up (= Graph A inputs)
        # + static_qh_pad/kh_pad/vh_pad (= Graph B inputs) + the flat
        # static_x_norm1/qh_flat/kh_flat/vh_flat/ctx_flat needed by the
        # eager backward tail. Per-layer eager fallback (``fwd_failed``).
        self._fwd_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.fwd_failed: set[int] = set()

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

        # ---- Forward-recompute graph IO ----
        # Inputs (per-layer): residual-stream input layer_in[i] in model dtype.
        # Outputs go into static buffers reused by the rest of the backward:
        #   - static_resid_mid / static_gate / static_up are ALSO Graph A's
        #     inputs (the FFN-bwd reads them directly — no copy needed between
        #     F-graph and A-graph).
        #   - static_qh_pad / kh_pad / vh_pad are ALSO Graph B's inputs (the
        #     forward scatters directly into the padded layout — no
        #     _stage_attn_inputs hop between F-graph and B-graph).
        # Flat duplicates are kept for the eager backward tail (RoPE-bwd,
        # Q/K/V-proj-bwd, in_ln rmsnorm-bwd) which still needs flat views.
        self.static_layer_in = torch.zeros((s, D), dtype=mdt, device=dev)
        # cos/sin shared across the L layers' forward graphs — only depend
        # on per-backward positions, so staged ONCE per backward in
        # ``begin_backward``. fp32 (matches rope_cos_sin's output dtype).
        self.static_cos = torch.zeros((s, Hd // 2), dtype=torch.float32, device=dev)
        self.static_sin = torch.zeros((s, Hd // 2), dtype=torch.float32, device=dev)
        # Saved MLP gate||up captured in the forward (one buffer per layer
        # in the accumulator; we stage the current layer's into one static
        # buffer per backward call). Skipping the gate_up matmul is the
        # production path (see Llama3BackwardService.process_backward).
        self.static_saved_gate_up = torch.zeros((s, 2 * inter), dtype=mdt, device=dev)
        # Outputs (flat, model dtype): the rest of the cache the eager tail
        # consumes. qh/kh/vh flat are written alongside the padded scatter.
        self.static_x_norm1 = torch.zeros((s, D), dtype=mdt, device=dev)
        self.static_qh_flat = torch.zeros((s, Hq, Hd), dtype=mdt, device=dev)
        self.static_kh_flat = torch.zeros((s, Hkv, Hd), dtype=mdt, device=dev)
        self.static_vh_flat = torch.zeros((s, Hkv, Hd), dtype=mdt, device=dev)
        self.static_ctx_flat = torch.zeros((s, D), dtype=mdt, device=dev)

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

        # Padded-attention + forward: arm a fitting dummy batch so scatter
        # indices, key-pad mask, AND cos/sin are valid for capture (kernels
        # still see zero values for everything else — only shapes/addresses
        # are captured).
        dummy_l = min(self.l_max, self.s_max)
        self.begin_backward(dummy_l, [dummy_l], [0])

        try:
            self._capture_attn()
        except Exception as e:  # noqa: BLE001
            self.attn_failed = True
            dprint(f"[bwd-graph] attn startup capture failed: {e}; "
                   f"all layers will fall back to eager at runtime")

        # Forward: one graph per layer (Q/K/V/O base + LoRA addresses vary).
        # Same dummy-batch arming as attn — cos/sin already staged by
        # begin_backward above. Stage zero layer_in / saved_gate_up via
        # the static buffers' default zero state (already zero from
        # _alloc_static_buffers).
        for i in range(L):
            try:
                self._capture_forward(i, self.svc._layer_weights(i))
            except Exception as e:  # noqa: BLE001
                self.fwd_failed.add(i)
                dprint(f"[bwd-graph] forward layer {i} startup capture failed: "
                       f"{e}; will fall back to eager at runtime")

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
        captured_fwd = L - len(self.fwd_failed)
        attn_status = "0/1" if self.attn_failed else "1/1"
        elapsed = (time.perf_counter() - t0) * 1000.0
        dprint(
            f"[bwd-graph] pre-captured forward {captured_fwd}/{L} + "
            f"FFN {captured_ffn}/{L} + attn {attn_status} in {elapsed:.0f}ms")

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

        # Stage cos/sin ALWAYS (used by both the graphed forward and the
        # eager forward fallback when the padded-attention budget overflows
        # — the fallback in ``Llama3GraphedBackward.forward`` reads
        # ``static_cos/sin[:n]``). Build positions on CPU = concat(arange(s)
        # for s in seq_lens), pad with zeros — RoPE(zeros, cos=any, sin=any)
        # = zeros because the q/k inputs are already zero in the tail.
        # We size exactly to s_max here; in production n ≤ s_max so the full
        # sequence fits, and any pathological n > s_max (only seen in the
        # gradcheck overflow test) just truncates — that path doesn't use
        # the forward graph fallback, only ``attn_backward``.
        if n > 0:
            pos_cpu: list[int] = []
            for s in seq_lens:
                if len(pos_cpu) >= self.s_max:
                    break
                want = min(int(s), self.s_max - len(pos_cpu))
                pos_cpu.extend(range(want))
            pos_cpu.extend([0] * (self.s_max - len(pos_cpu)))
            pos_t = torch.tensor(pos_cpu, dtype=torch.long, device=self.device)
            cos, sin = rope_cos_sin(pos_t, self.Hd, float(self.svc.theta))
            self.static_cos.copy_(cos)
            self.static_sin.copy_(sin)
        else:
            self.static_cos.zero_()
            self.static_sin.zero_()

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

    # ------------------------------------------------------ forward recompute

    def _padded_attn_forward_core(self) -> None:
        """Captureable padded-attention FORWARD. Reads
        ``static_qh_pad/kh_pad/vh_pad`` + masks (just scattered by the upstream
        Q/K/V/RoPE region of the same forward graph), writes the flat
        ``static_ctx_flat`` via padded-compute + gather.

        Math mirrors the per-sample loop in ``layer_forward`` (lines 175-191
        of llama3.py): GQA repeat_interleave, fp32 scores/softmax, causal +
        key-pad mask, ``att @ v`` cast back to model dtype. Output layout
        is the flat ``[n, Hq*Hd]`` that the eager O-proj backward
        consumes."""
        bn, lm = self.bn_max, self.l_max
        Hq, Hkv, Hd = self.Hq, self.Hkv, self.Hd
        D = Hq * Hd
        kv_repeat = self.kv_repeat
        mdt = self.model_dtype

        # Same masking layout as the backward graph: broadcast over heads.
        attn_mask = (self.static_causal_mask.unsqueeze(0).unsqueeze(0)
                     | self.static_key_pad_mask.unsqueeze(1).unsqueeze(1))

        if kv_repeat != 1:
            kh_rep = self.static_kh_pad.repeat_interleave(kv_repeat, dim=2)
            vh_rep = self.static_vh_pad.repeat_interleave(kv_repeat, dim=2)
        else:
            kh_rep = self.static_kh_pad
            vh_rep = self.static_vh_pad

        # [bn, Hq, l, Hd] fp32 — scores/softmax always fp32 (GQA precision rule).
        q_att = self.static_qh_pad.permute(0, 2, 1, 3).contiguous().float()
        k_att = kh_rep.permute(0, 2, 1, 3).contiguous().float()
        v_att = vh_rep.permute(0, 2, 1, 3).contiguous().float()

        scores = (q_att @ k_att.transpose(-1, -2)) * self.scale
        scores = scores.masked_fill(attn_mask, -1e9)
        att = torch.softmax(scores, dim=-1)
        # Fully-masked rows (entirely padded samples) → softmax = NaN; map to 0.
        att = torch.nan_to_num(att, nan=0.0)
        ctx_att = att @ v_att                       # [bn, Hq, l, Hd] fp32

        # Back to [bn, l, Hq, Hd], cast to model dtype.
        ctx_pad = ctx_att.permute(0, 2, 1, 3).contiguous().to(mdt)
        # Gather padded → flat [s_max, D]. Tail rows gather from (0,0); caller
        # only reads [:n].
        gathered = ctx_pad[self.static_bn_idx, self.static_pos_idx]   # [s, Hq, Hd]
        self.static_ctx_flat.copy_(gathered.reshape(self.s_max, D))

    def _forward_core(self, lw: dict) -> None:
        """The captureable layer-forward body (no eager helpers, no python loop
        over samples). Reads ``static_layer_in`` + ``static_cos/sin`` +
        ``static_saved_gate_up``; writes:

          - ``static_x_norm1`` (eager Q/K/V LoRA-A bwd input)
          - ``static_qh_flat / kh_flat / vh_flat`` (eager RoPE-bwd input)
          - ``static_qh_pad  / kh_pad  / vh_pad`` (Graph B input — scattered)
          - ``static_ctx_flat`` (eager O-proj bwd input)
          - ``static_resid_mid`` (Graph A input)
          - ``static_gate / static_up`` (Graph A input — sliced from saved_gate_up)

        Mirrors ``layer_forward`` with ``saved_gate_up != None``; differs only
        in being shape-stable at the fixed ``s_max`` slab and writing into
        static buffers in place."""
        s, D, inter = self.s_max, self.D, self.inter
        Hq, Hkv, Hd = self.Hq, self.Hkv, self.Hd
        kv_size = Hkv * Hd
        scaling = float(self.svc.scaling)
        mdt = self.model_dtype

        # 1) RMSNorm(in_ln) on [s_max, D]
        x_norm1 = rmsnorm(self.static_layer_in, lw["in_ln"], self.eps)
        self.static_x_norm1.copy_(x_norm1)

        # 2) Q/K/V projections (base + LoRA) — LoRA `.data` refs are stable.
        q = _proj(x_norm1, lw["q"], lw["qA"], lw["qB"], scaling)          # [s, D]
        k = _proj(x_norm1, lw["k"], lw["kA"], lw["kB"], scaling)          # [s, kv]
        v = _proj(x_norm1, lw["v"], lw["vA"], lw["vB"], scaling)

        # 3) RoPE on q, k; pack v.
        qh = apply_rope(q.view(s, Hq, Hd), self.static_cos, self.static_sin)
        kh = apply_rope(k.view(s, Hkv, Hd), self.static_cos, self.static_sin)
        vh = v.view(s, Hkv, Hd)
        # Flat writes (model dtype) — read by the eager RoPE-bwd tail.
        self.static_qh_flat.copy_(qh)
        self.static_kh_flat.copy_(kh)
        self.static_vh_flat.copy_(vh)
        # Scatter into padded layout (Graph B inputs). All s_max rows write,
        # so we MUST use accumulate=True: tail rows (k ≥ n) have
        # (bn_idx, pos_idx) = (0, 0), which coincides with the legit
        # (sample 0, position 0) slot — without accumulate they'd overwrite
        # it (last-write-wins). With accumulate=True, the legit row adds
        # qh[real_0] and tail rows add 0 (input tail is zero because
        # rmsnorm(0)·W = 0 and 0·W^T = 0). Pre-zeroed in
        # ``stage_forward_inputs`` so accumulate starts from a clean slab.
        self.static_qh_pad.index_put_(
            (self.static_bn_idx, self.static_pos_idx), qh, accumulate=True)
        self.static_kh_pad.index_put_(
            (self.static_bn_idx, self.static_pos_idx), kh, accumulate=True)
        self.static_vh_pad.index_put_(
            (self.static_bn_idx, self.static_pos_idx), vh, accumulate=True)

        # 4) Padded-attention forward → static_ctx_flat.
        self._padded_attn_forward_core()

        # 5) O projection (base + LoRA) → resid_mid = layer_in + o.
        o = _proj(self.static_ctx_flat, lw["o"], lw["oA"], lw["oB"], scaling)
        self.static_resid_mid.copy_(self.static_layer_in + o)

        # 6) Slice saved gate||up into separate gate/up. Skip the gate_up
        #    matmul (the "saved_gate_up != None" path in eager layer_forward).
        self.static_gate.copy_(self.static_saved_gate_up[:, :inter])
        self.static_up.copy_(self.static_saved_gate_up[:, inter:])

    def _capture_forward(self, layer_id: int, lw: dict) -> None:
        """Capture the per-layer forward-recompute graph. Static buffers must
        already be initialized (we stage zeros up-front in ``prepare``; real
        backwards stage real values into them via ``stage_forward_inputs``)."""
        for _ in range(self._N_WARMUP):
            self._forward_core(lw)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self._graph_pool, stream=self._stream):
            self._forward_core(lw)
        self._fwd_graphs[layer_id] = g

    def stage_forward_inputs(self, layer_in: torch.Tensor,
                             saved_gate_up: torch.Tensor, n: int) -> None:
        """Copy this layer's inputs into the static slabs (zero the tail).
        ``cos/sin`` are staged once per backward in ``begin_backward``; the
        ``static_*_pad`` buffers are zeroed here so the in-graph scatter
        produces zero rows at the tail.

        Cheap: 2 ``copy_`` + 1 ``zero_`` per layer (plus the pad zero_,
        amortized across both the forward write and the Graph B input)."""
        s = self.s_max
        if layer_in.dtype != self.model_dtype:
            layer_in = layer_in.to(self.model_dtype)
        self.static_layer_in[:n].copy_(layer_in)
        if n < s:
            self.static_layer_in[n:].zero_()
        if saved_gate_up.dtype != self.model_dtype:
            saved_gate_up = saved_gate_up.to(self.model_dtype)
        self.static_saved_gate_up[:n].copy_(saved_gate_up)
        if n < s:
            self.static_saved_gate_up[n:].zero_()
        # Zero the padded q/k/v staging targets before the scatter so tail
        # writes (to slot (0,0)) don't leave stale data from a prior layer.
        self.static_qh_pad.zero_()
        self.static_kh_pad.zero_()
        self.static_vh_pad.zero_()

    def forward(self, layer_id: int, lw: dict, layer_in: torch.Tensor,
                saved_gate_up: torch.Tensor, n: int) -> dict:
        """Graphed layer forward-recompute. Returns the cache dict of static
        views the downstream backward consumes — same shape contract as the
        eager ``layer_forward``.

        Silent eager fallback when (a) the padded budget didn't fit
        (``_attn_fit`` False), or (b) this layer's capture/replay raised
        (added to ``fwd_failed``). Eager fallback calls ``layer_forward(...)``
        and returns its dict directly."""
        if not self._attn_fit or layer_id in self.fwd_failed:
            return layer_forward(layer_in, lw, self.svc.scaling,
                                 self.static_cos[:n], self.static_sin[:n],
                                 self._cur_seq_lens, self._cur_b_start,
                                 (self.Hq, self.Hkv, self.Hd, self.Hkv * self.Hd),
                                 self.eps, saved_gate_up=saved_gate_up)

        self.stage_forward_inputs(layer_in, saved_gate_up, n)

        if layer_id not in self._fwd_graphs:
            try:
                self._capture_forward(layer_id, lw)
            except Exception as e:  # noqa: BLE001
                self.fwd_failed.add(layer_id)
                dprint(f"[bwd-graph] forward capture failed for layer "
                       f"{layer_id}: {e}; falling back to eager forward")
                return layer_forward(layer_in, lw, self.svc.scaling,
                                     self.static_cos[:n], self.static_sin[:n],
                                     self._cur_seq_lens, self._cur_b_start,
                                     (self.Hq, self.Hkv, self.Hd,
                                      self.Hkv * self.Hd),
                                     self.eps, saved_gate_up=saved_gate_up)

        try:
            self._fwd_graphs[layer_id].replay()
        except Exception as e:  # noqa: BLE001
            self.fwd_failed.add(layer_id)
            dprint(f"[bwd-graph] forward replay failed for layer "
                   f"{layer_id}: {e}; falling back to eager forward")
            return layer_forward(layer_in, lw, self.svc.scaling,
                                 self.static_cos[:n], self.static_sin[:n],
                                 self._cur_seq_lens, self._cur_b_start,
                                 (self.Hq, self.Hkv, self.Hd,
                                  self.Hkv * self.Hd),
                                 self.eps, saved_gate_up=saved_gate_up)

        return self.cache_views(n)

    def cache_views(self, n: int) -> dict:
        """View dict over the static forward outputs at [:n]. Same keys/shapes
        as ``layer_forward``'s return so the downstream ``layer_backward`` /
        ``_layer_backward_graphed`` reads it transparently. The caller must
        consume this before the NEXT layer's forward replay overwrites the
        buffers (the per-layer loop already does this — each iteration's
        backward fully consumes its cache before the next iteration's forward
        runs)."""
        # ``x`` = the layer's input (= layer_in[i] view) — used by the eager
        # in_ln rmsnorm backward. Same slab the forward read from.
        return {
            "x": self.static_layer_in[:n],
            "x_norm1": self.static_x_norm1[:n],
            "qh": self.static_qh_flat[:n],
            "kh": self.static_kh_flat[:n],
            "vh": self.static_vh_flat[:n],
            "ctx_flat": self.static_ctx_flat[:n],
            "resid_mid": self.static_resid_mid[:n],
            "gate": self.static_gate[:n],
            "up": self.static_up[:n],
        }
