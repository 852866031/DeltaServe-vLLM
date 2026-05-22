# llama3_backward_service.py
# Subclass of your LlamaSFTBackwardService that overrides ONLY attention path for Llama-3 GQA
#
# Assumptions you stated:
# - You keep the SAME packed LoRA tensor shape as llama1: w_combined_leaf: [2, 4r, Hq, Hd]
# - KV are "padded to D" (e.g. D=4096, D_kv=1024 valid, rest zeros)
# - Therefore adapter receive/pack format stays identical; only attention math needs GQA head mapping.
#
# What changes:
# - In forward: k_/v_ are computed as [S, D] (padded) but ONLY first D_kv are valid.
# - For attention matmuls: reshape kv using Hkv heads (not Hq), and repeat kv heads to Hq.
# - In backward: reduce repeated-head gradients back into Hkv (sum across replication factor)
# - For K/V LoRA + base K/V matmul backprop: ONLY use first D_kv columns of gk/gv when multiplying by w_k/w_v^T
#   because your w_k/w_v in llama3 should be [D, D_kv] (typical) OR if you store padded [D, D], then slice too.

import hashlib
import math
from dserve.models.llama.SFT_service import LlamaSFTBackwardService
import torch
from dserve.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from dserve.models.llama.triton_kernel.rmsnorm import rmsnorm_backward, rmsnorm_forward

# import your base service
# from .llama1_backward_service import LlamaSFTBackwardService


def tensor_hash(t: torch.Tensor, algo="sha256") -> str:
    h = hashlib.new(algo)
    h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def _rmsnorm_pt(x: torch.Tensor, w: torch.Tensor, eps: float):
        # x: [S, D], w: [D]
        # Do rsqrt in fp32 for stability, output in x.dtype
        x_f = x.float()
        var = (x_f * x_f).mean(dim=-1, keepdim=True)
        inv = torch.rsqrt(var + eps)
        y = (x_f * inv).to(dtype=x.dtype)
        return y * w.to(dtype=x.dtype)

def _rotary_apply_pt(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    x:  [S, H, Hd]
    cos/sin: [S, Hd/2]
    returns rotated x (no in-place, autograd-safe)
    """
    S, H, Hd = x.shape
    Dh = Hd // 2
    x1 = x[..., :Dh]
    x2 = x[..., Dh:]
    cos = cos[:, None, :].to(dtype=x.dtype)  # [S,1,Dh]
    sin = sin[:, None, :].to(dtype=x.dtype)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat([y1, y2], dim=-1)


class Llama3SFTBackwardService(LlamaSFTBackwardService):
    """
    Only overrides _backpop_attention for GQA.
    Everything else (loss, FFN backward, etc.) is inherited.
    """

    # ATTN_BN_MAX, ATTN_L_MAX, USE_GRAPHED_ATTENTION are now instance attributes
    # set by the base __init__ from cfg.cuda_graph.

    # Tracks whether the one-time startup info line for the padded attention
    # backward has been printed. Class-level so all instances share it.
    _padded_attn_info_printed: bool = False

    # Tracks whether we've already logged the "padded attention overflow →
    # monolithic fallback" warning. One-shot so a skewed dataset doesn't
    # spam the console.
    _padded_attn_overflow_warned: bool = False

    def __init__(self, network_config, *args, **kwargs):
        super().__init__(network_config, *args, **kwargs)
        # Llama3/GQA params
        # (use the keys you actually have in your network_config; adjust if names differ)
        self.num_heads_q_ = int(network_config.get("num_attention_heads", network_config.get("n_head")))
        self.num_heads_kv_ = int(network_config.get("num_key_value_heads", network_config.get("n_kv_head", self.num_heads_q_)))
        assert self.num_heads_q_ % self.num_heads_kv_ == 0, "GQA requires Hq % Hkv == 0"
        self.kv_repeat_ = self.num_heads_q_ // self.num_heads_kv_
        self.head_dim_ = self.embed_dim_ // self.num_heads_q_
        assert self.embed_dim_ == self.num_heads_q_ * self.head_dim_

    def receive_adapter(self, adapter_dict):
        super().receive_adapter(adapter_dict)
        for layer in self.adapter_weights.lora_weights:
            layer.requires_grad = True
        return
        

    @torch.no_grad()
    def _backpop_attention(
        self,
        last_layer_input: torch.Tensor,         # [S, D]
        grad_ffn_input: torch.Tensor,           # [S, D]  dL/d(out)
        layer_weight,
        layer_id: int,
    ):
        device = last_layer_input.device
        Hq = int(self.num_heads_q_)
        Hkv = int(self.num_heads_kv_)
        Hd = int(self.head_dim_)
        D = int(self.embed_dim_)
        D_kv = Hkv * Hd
        assert D == Hq * Hd
        assert Hq % Hkv == 0
        kv_repeat = Hq // Hkv

        # CPU-side per-request metadata, precomputed in get_logits_and_targets.
        # All Python-side loop bounds and slice indices below read from these
        # lists, so no D→H syncs are needed inside this function.
        seq_lens_py = self.activations.seq_lens_py
        b_start_py = self.activations.b_start_py

        # ----- positions -----
        position_ids = torch.cat(
            [torch.arange(0, ln, device=device, dtype=torch.long) for ln in seq_lens_py]
        )
        cos = self.model_weights._cos_cached.index_select(0, position_ids)  # [S, Hd/2]
        sin = self.model_weights._sin_cached.index_select(0, position_ids)

        # ----- weights -----
        w_q = layer_weight.q_weight_          # [D, D]
        w_k = layer_weight.k_weight_          # [D, D_kv] (typical) or padded [D,D]
        w_v = layer_weight.v_weight_          # [D, D_kv] (typical) or padded [D,D]
        w_o = layer_weight.o_weight_          # [D, D]
        w_attn_norm = layer_weight.att_norm_weight_  # [D]

        # ----- packed LoRA param (optimizer leaf) -----
        w_combined_leaf = self.adapter_weights.lora_weights[layer_id]  # [2, 4r, Hq, Hd]
        w_fp = w_combined_leaf.to(torch.float16)  # compute in fp16
        r = w_fp.shape[1] // 4
        assert w_fp.shape[2] == Hq and w_fp.shape[3] == Hd

        # Unpack A/B views (autograd-free; we will build grads explicitly)
        qA = w_fp[0, 0:r].reshape(r, -1).T      # [D, r]
        qB = w_fp[1, 0:r].reshape(-1, r).T      # [r, D]
        kA = w_fp[0, r:2*r].reshape(r, -1).T
        kB = w_fp[1, r:2*r].reshape(-1, r).T
        vA = w_fp[0, 2*r:3*r].reshape(r, -1).T
        vB = w_fp[1, 2*r:3*r].reshape(-1, r).T
        oA = w_fp[0, 3*r:4*r].reshape(r, -1).T
        oB = w_fp[1, 3*r:4*r].reshape(-1, r).T

        scale_lora = self.adapter_weights.scaling

        def rotary_fwd(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
            # x [S,H,Hd], cos/sin [S,Hd/2] ; do out-of-place for clarity
            S, H, Hd_ = x.shape
            Dh = Hd_ // 2
            x1 = x[..., :Dh]
            x2 = x[..., Dh:]
            cos_ = cos[:, None, :].to(dtype=x.dtype)
            sin_ = sin[:, None, :].to(dtype=x.dtype)
            y1 = x1 * cos_ - x2 * sin_
            y2 = x2 * cos_ + x1 * sin_
            return torch.cat([y1, y2], dim=-1)

        def rotary_bwd(g: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
            # inverse linear transform
            S, H, Hd_ = g.shape
            Dh = Hd_ // 2
            g1 = g[..., :Dh]
            g2 = g[..., Dh:]
            cos_ = cos[:, None, :].to(dtype=g.dtype)
            sin_ = sin[:, None, :].to(dtype=g.dtype)
            dx1 = g1 * cos_ + g2 * sin_
            dx2 = -g1 * sin_ + g2 * cos_
            return torch.cat([dx1, dx2], dim=-1)

        def proj_lora_fwd(X: torch.Tensor, A: torch.Tensor, B: torch.Tensor):
            # all fp16 here
            return (X @ A @ B) * scale_lora

        # ---------- forward (manual, save what we need) ----------
        x_prev = last_layer_input.to(torch.float16)  # [S,D]
        x_norm = rmsnorm_forward(x_prev, w_attn_norm, eps=self.eps_)                 # fp16
        X = x_norm
        S_total = X.shape[0]

        # Base projections
        q_base = X @ w_q.to(dtype=X.dtype)  # [S,D]

        k_base_raw = X @ w_k.to(dtype=X.dtype)  # [S, D_kv] or [S,D]
        v_base_raw = X @ w_v.to(dtype=X.dtype)

        # Infer whether base is padded
        if k_base_raw.shape[1] == D:
            k_base = k_base_raw[:, :D_kv]
            v_base = v_base_raw[:, :D_kv]
        else:
            k_base = k_base_raw
            v_base = v_base_raw
            assert k_base.shape[1] == D_kv

        # LoRA projections (full D for all four)
        q_lora = proj_lora_fwd(X, qA, qB)          # [S,D]
        k_lora_full = proj_lora_fwd(X, kA, kB)     # [S,D] (KV padded region ~0 by your padding)
        v_lora_full = proj_lora_fwd(X, vA, vB)     # [S,D]
        o_lora = None  # computed later

        q = q_base + q_lora                        # [S,D]
        k = k_base + k_lora_full[:, :D_kv]         # [S,D_kv]
        v = v_base + v_lora_full[:, :D_kv]         # [S,D_kv]

        qh = q.view(S_total, Hq, Hd)
        kh = k.view(S_total, Hkv, Hd)
        vh = v.view(S_total, Hkv, Hd)

        qh_rot = rotary_fwd(qh, cos, sin)          # [S,Hq,Hd]
        kh_rot = rotary_fwd(kh, cos, sin)          # [S,Hkv,Hd]

        # per-request offsets (CPU-side, no syncs)
        Bn = len(seq_lens_py)

        # Attention forward (store ctx only; att is recomputed in backward)
        ctx = torch.empty((S_total, Hq, Hd), device=device, dtype=qh_rot.dtype)
        scale = 1.0 / math.sqrt(Hd)

        for i in range(Bn):
            st = b_start_py[i]
            ln = seq_lens_py[i]

            q_blk = qh_rot[st:st+ln].transpose(0, 1)      # [Hq,L,Hd]
            k_blk = kh_rot[st:st+ln].transpose(0, 1)      # [Hkv,L,Hd]
            v_blk = vh[st:st+ln].transpose(0, 1)          # [Hkv,L,Hd]  (note: no rotary on v)

            if kv_repeat != 1:
                k_rep = k_blk.repeat_interleave(kv_repeat, dim=0)  # [Hq,L,Hd]
                v_rep = v_blk.repeat_interleave(kv_repeat, dim=0)
            else:
                k_rep, v_rep = k_blk, v_blk

            # Compute scores in fp32 from the start so the forward att
            # matches the backward's recomputed att_f exactly. The old code
            # did the matmul in fp16 then upcast, which produced different
            # attention weights than the fp32 backward recomputation —
            # causing systematically wrong gradients (loss wouldn't decrease).
            scores = (q_blk.float() @ k_rep.float().transpose(-1, -2)) * scale  # [Hq,L,L] fp32
            mask = torch.triu(torch.ones((ln, ln), device=device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask.unsqueeze(0), -1e9)
            scores = scores.clamp(-80.0, 80.0)
            att = torch.softmax(scores, dim=-1).to(dtype=q_blk.dtype)  # back to fp16

            ctx_blk = (att @ v_rep).transpose(0, 1)                 # [L,Hq,Hd]
            ctx[st:st+ln] = ctx_blk

        ctx_flat = ctx.reshape(S_total, D)                           # [S,D]

        # O forward
        o_base = ctx_flat @ w_o.to(dtype=ctx_flat.dtype)             # [S,D]
        Zo = ctx_flat @ oA                                           # [S,r]
        o_lora = (Zo @ oB) * scale_lora                              # [S,D]
        o_total = o_base + o_lora                                    # [S,D]

        out = x_prev + o_total                                       # [S,D]

        # ---------- backward (manual) ----------
        grad_out = grad_ffn_input.to(dtype=out.dtype)                # [S,D]
        grad_x_prev_resid = grad_out                                 # residual path
        grad_o_total = grad_out

        # Back through O
        G = grad_o_total * scale_lora                     # [S,D]
        tmp = G @ oB.t()                                  # [S,r]

        grad_oA = ctx_flat.t() @ tmp                      # [D,r]
        grad_oB = (ctx_flat @ oA).t() @ G                 # [r,D]
        grad_ctx_from_lora_o = tmp @ oA.t()               # [S,D]

        grad_ctx_flat = (grad_o_total @ w_o.t()) + grad_ctx_from_lora_o  # [S,D]

        # Attention backward
        grad_qh_rot = torch.zeros_like(qh_rot)                       # [S,Hq,Hd]
        grad_kh_rot = torch.zeros_like(kh_rot)                       # [S,Hkv,Hd]
        grad_vh = torch.zeros_like(vh)                               # [S,Hkv,Hd]

        for i in range(Bn):
            st = b_start_py[i]
            ln = seq_lens_py[i]

            # g_ctx in q-head space
            g_ctx_blk = grad_ctx_flat[st:st+ln].view(ln, Hq, Hd).transpose(0, 1)  # [Hq, L, Hd]

            # q in q-head space, k/v in kv-head space (rotary already applied to q/k)
            q_blk = qh_rot[st:st+ln].transpose(0, 1)      # [Hq, L, Hd]
            k_blk = kh_rot[st:st+ln].transpose(0, 1)      # [Hkv, L, Hd]
            v_blk = vh[st:st+ln].transpose(0, 1)          # [Hkv, L, Hd]

            # expand kv heads to q heads
            if kv_repeat != 1:
                k_rep = k_blk.repeat_interleave(kv_repeat, dim=0)    # [Hq, L, Hd]
                v_rep = v_blk.repeat_interleave(kv_repeat, dim=0)    # [Hq, L, Hd]
            else:
                k_rep, v_rep = k_blk, v_blk

            mask = torch.triu(
                torch.ones((ln, ln), device=device, dtype=torch.bool),
                diagonal=1
            )  # [L, L]

            # ---- do everything in fp32 (matches your intent) ----
            q_f = q_blk.float()
            k_f = k_rep.float()
            v_f = v_rep.float()
            g_f = g_ctx_blk.float()

            # IMPORTANT: keep pre-clamp scores for clamp backward mask
            scores_pre_f = (q_f @ k_f.transpose(-1, -2)) * scale        # [Hq, L, L]
            scores_pre_f = scores_pre_f.masked_fill(mask.unsqueeze(0), -1e9)

            scores_clamped_f = scores_pre_f.clamp(-80.0, 80.0)
            att_f = torch.softmax(scores_clamped_f, dim=-1)             # [Hq, L, L]

            # dV_rep, dAtt (fp32)
            dV_rep_f = att_f.transpose(-1, -2) @ g_f                    # [Hq, L, Hd]
            dAtt_f   = g_f @ v_f.transpose(-1, -2)                      # [Hq, L, L]

            # softmax backward wrt scores_clamped_f (fp32)
            s = (dAtt_f * att_f).sum(dim=-1, keepdim=True)
            dScores_clamped_f = (dAtt_f - s) * att_f
            dScores_clamped_f = dScores_clamped_f.masked_fill(mask.unsqueeze(0), 0.0)

            # ---- CLAMP BACKWARD (this is the missing piece) ----
            # derivative of clamp: 1 inside (-80,80), 0 outside
            clamp_ok = (scores_pre_f > -80.0) & (scores_pre_f < 80.0)
            dScores_pre_f = dScores_clamped_f * clamp_ok

            # backprop to q/k (still fp32)
            dQ_f     = (dScores_pre_f @ k_f) * scale
            dK_rep_f = (dScores_pre_f.transpose(-1, -2) @ q_f) * scale

            # cast once at end
            dQ     = dQ_f.to(dtype=q_blk.dtype)
            dK_rep = dK_rep_f.to(dtype=q_blk.dtype)
            dV_rep = dV_rep_f.to(dtype=q_blk.dtype)

            # write dQ into q-head grads
            grad_qh_rot[st:st+ln] += dQ.transpose(0, 1)                 # [L, Hq, Hd]

            # reduce dK/dV back to kv heads
            if kv_repeat != 1:
                dK_kv = dK_rep.view(Hkv, kv_repeat, ln, Hd).sum(dim=1)  # [Hkv, L, Hd]
                dV_kv = dV_rep.view(Hkv, kv_repeat, ln, Hd).sum(dim=1)
            else:
                dK_kv, dV_kv = dK_rep, dV_rep

            grad_kh_rot[st:st+ln] += dK_kv.transpose(0, 1)              # [L, Hkv, Hd]
            grad_vh[st:st+ln]     += dV_kv.transpose(0, 1)              # [L, Hkv, Hd]

        # Rotary backward
        grad_qh = rotary_bwd(grad_qh_rot, cos, sin)                    # [S,Hq,Hd]
        grad_kh = rotary_bwd(grad_kh_rot, cos, sin)                    # [S,Hkv,Hd]

        gq_flat = grad_qh.reshape(S_total, D)                          # [S,D]
        gk_flat = grad_kh.reshape(S_total, D_kv)                       # [S,D_kv]
        gv_flat = grad_vh.reshape(S_total, D_kv)                       # [S,D_kv]

        # Back to X through base projections
        grad_X_from_q = gq_flat @ w_q.t()

        if w_k.shape[1] == D_kv:
            grad_X_from_k = gk_flat @ w_k.t()
            grad_X_from_v = gv_flat @ w_v.t()
        else:
            grad_X_from_k = gk_flat @ w_k[:, :D_kv].t()
            grad_X_from_v = gv_flat @ w_v[:, :D_kv].t()

        # LoRA Q grads
        G = gq_flat * scale_lora
        tmp = G @ qB.t()              # [S,r]
        grad_qA = X.t() @ tmp         # [D,r]
        grad_qB = (X @ qA).t() @ G    # [r,D]
        grad_X_from_lora_q = tmp @ qA.t()

        # For padded LoRA K/V: build padded grads in D space for B/A math
        gk_pad = torch.zeros((S_total, D), device=device, dtype=X.dtype)
        gv_pad = torch.zeros((S_total, D), device=device, dtype=X.dtype)
        gk_pad[:, :D_kv] = gk_flat.to(dtype=X.dtype)
        gv_pad[:, :D_kv] = gv_flat.to(dtype=X.dtype)

        # LoRA K grads
        G = gk_pad * scale_lora
        tmp = G @ kB.t()
        grad_kA = X.t() @ tmp
        grad_kB = (X @ kA).t() @ G
        grad_X_from_lora_k = tmp @ kA.t()

        # LoRA V grads
        G = gv_pad * scale_lora
        tmp = G @ vB.t()
        grad_vA = X.t() @ tmp
        grad_vB = (X @ vA).t() @ G
        grad_X_from_lora_v = tmp @ vA.t()

        # Total grad wrt X (x_norm)
        grad_X = (grad_X_from_q + grad_X_from_k + grad_X_from_v +
                grad_X_from_lora_q + grad_X_from_lora_k + grad_X_from_lora_v)

        # RMSNorm backward to x_prev
        grad_from_norm = rmsnorm_backward(x_prev, grad_X, w_attn_norm, eps=self.eps_)
        grad_x_prev = grad_from_norm + grad_x_prev_resid

        # ----- pack LoRA grads back to combined grad tensor -----
        # Build a fp16 grad tensor matching w_fp layout, then cast to fp32 for optimizer
        grad_combined = torch.zeros_like(w_fp)  # fp16

        def pack_G(G, transpose_first: bool):
            if transpose_first:
                G = G.t()  # [r,D]
            return G.reshape(r, Hq, Hd)

        # A-side (stored in [0, ...]) expects [D,r] -> transpose_first=True
        grad_combined[0, 0:r]       = pack_G(grad_qA.to(dtype=w_fp.dtype), True)
        grad_combined[0, r:2*r]     = pack_G(grad_kA.to(dtype=w_fp.dtype), True)
        grad_combined[0, 2*r:3*r]   = pack_G(grad_vA.to(dtype=w_fp.dtype), True)
        grad_combined[0, 3*r:4*r]   = pack_G(grad_oA.to(dtype=w_fp.dtype), True)

        # B-side (stored in [1, ...]) is [r,D] already -> transpose_first=False
        grad_combined[1, 0:r]       = pack_G(grad_qB.to(dtype=w_fp.dtype), False)
        grad_combined[1, r:2*r]     = pack_G(grad_kB.to(dtype=w_fp.dtype), False)
        grad_combined[1, 2*r:3*r]   = pack_G(grad_vB.to(dtype=w_fp.dtype), False)
        grad_combined[1, 3*r:4*r]   = pack_G(grad_oB.to(dtype=w_fp.dtype), False)

        # ----- clip in fp32, assign fp32 grad to optimizer leaf (branchless) -----
        max_norm = 1.0
        g32 = grad_combined.float()
        gn = g32.norm()
        clip_scale = torch.clamp(max_norm / (gn + 1e-6), max=1.0)
        g32.mul_(clip_scale)

        self.adapter_weights.lora_weights[layer_id].grad = g32
        return grad_x_prev.to(dtype=torch.float16)

    # ======================================================================
    # Padded single-graph attention backward
    #
    # A drop-in alternative to _backpop_attention whose internal shapes are
    # all fixed at (ATTN_BN_MAX, ATTN_L_MAX). Enabled by setting
    # LlamaSFTBackwardService.USE_GRAPHED_ATTENTION = True on the base class
    # (dispatch happens in _lora_context_backward).
    #
    # The math follows the monolithic version exactly; the only structural
    # difference is that the per-request attention loop is replaced with a
    # single batched attention call over [Bn_max, Hq, L_max, L_max] shapes,
    # plus a causal + key-padding mask. The intent is that the whole
    # function can eventually be captured as one CUDA graph per layer —
    # all internal shapes depend only on the class constants.
    #
    # Organization:
    #   Eager prep:    build scatter indices, write real inputs into padded
    #                  buffers, populate masks / cos / sin.
    #   Shape-stable:  the compute block from RMSNorm-fwd through grad-pack
    #                  + clip + .grad assignment. Every tensor shape here
    #                  depends only on (Bn_max, L_max, D, Hq, Hkv, Hd, r).
    #   Eager post:    gather padded grad_x_prev back into flat [S_total, D].
    # ======================================================================
    @torch.no_grad()
    def _backpop_attention_padded(self, last_layer_input, grad_ffn_input, layer_weight, layer_id):
        """Thin eager wrapper. Allocates a fresh ctx, fills it from the flat
        inputs, calls the shape-stable core, and returns the sliced flat
        output. The runner path bypasses this wrapper entirely and calls
        _backpop_attention_padded_core directly with a persistent ctx in
        GraphedBackwardRunner._attn_padded_replay.

        If the current batch doesn't fit the (ATTN_BN_MAX, ATTN_L_MAX)
        budget, this gracefully falls back to the monolithic
        _backpop_attention (variable-shape per-request) so the backward
        still completes correctly. A one-time log is emitted so the user
        knows tuning is needed."""
        self._maybe_print_padded_attn_info()

        Bn_max = int(self.ATTN_BN_MAX)
        L_max = int(self.ATTN_L_MAX)
        seq_lens_py = self.activations.seq_lens_py
        Bn = len(seq_lens_py)
        overflow_bn = Bn > Bn_max
        overflow_l = Bn > 0 and max(seq_lens_py) > L_max
        if overflow_bn or overflow_l:
            if not Llama3SFTBackwardService._padded_attn_overflow_warned:
                Llama3SFTBackwardService._padded_attn_overflow_warned = True
                max_l = max(seq_lens_py) if Bn > 0 else 0
                print(
                    f"\033[35m[BWD-GRAPH]: Padded attention overflow — falling back "
                    f"to monolithic _backpop_attention. Batch (Bn={Bn}, max_L={max_l}) "
                    f"exceeds (ATTN_BN_MAX={Bn_max}, ATTN_L_MAX={L_max}). Tune the "
                    f"class constants in Llama3SFTBackwardService if this is "
                    f"frequent.\033[0m"
                )
            return self._backpop_attention(last_layer_input, grad_ffn_input, layer_weight, layer_id)

        ctx = self._alloc_fresh_attn_ctx(last_layer_input.device)
        self._fill_attn_ctx_eager(ctx, last_layer_input, grad_ffn_input)
        self._backpop_attention_padded_core(ctx, layer_weight, layer_id)

        s_actual = last_layer_input.shape[0]
        return ctx.flat_grad_x_prev_out[:s_actual]

    def _maybe_print_padded_attn_info(self):
        """One-time startup log for the padded attention path. Fires from
        the eager wrapper only; the runner emits its own memory line for
        the captured path in prepare()."""
        if Llama3SFTBackwardService._padded_attn_info_printed:
            return
        Llama3SFTBackwardService._padded_attn_info_printed = True
        Bn_max = int(self.ATTN_BN_MAX)
        L_max = int(self.ATTN_L_MAX)
        Hq = int(self.num_heads_q_)
        Hkv = int(self.num_heads_kv_)
        Hd = int(self.head_dim_)
        D = int(self.embed_dim_)
        MB = 1024.0 * 1024.0
        fp16_bytes = 2
        fp32_bytes = 4
        x_prev_mb = Bn_max * L_max * D * fp16_bytes / MB
        grad_out_mb = Bn_max * L_max * D * fp16_bytes / MB
        cos_sin_mb = 2 * Bn_max * L_max * (Hd // 2) * fp16_bytes / MB
        scores_mb = Bn_max * Hq * L_max * L_max * fp32_bytes / MB
        att_mb = Bn_max * Hq * L_max * L_max * fp32_bytes / MB
        qkv_fp32_mb = 3 * Bn_max * Hq * L_max * Hd * fp32_bytes / MB
        total_est_mb = x_prev_mb + grad_out_mb + cos_sin_mb + scores_mb + att_mb + qkv_fp32_mb
        print(
            f"\033[35m[BWD-GRAPH]: Padded attention enabled. "
            f"Bn_max={Bn_max}, L_max={L_max}, "
            f"Hq={Hq}, Hkv={Hkv}, Hd={Hd}, D={D}\033[0m"
        )
        print(
            f"\033[35m[BWD-GRAPH]: Padded attention peak scratch estimate "
            f"(per-layer, re-used across layers via caching allocator):\n"
            f"    x_prev/grad_out pad: {x_prev_mb + grad_out_mb:6.1f} MB  "
            f"cos/sin pad: {cos_sin_mb:5.1f} MB\n"
            f"    fp32 scores: {scores_mb:6.1f} MB  "
            f"fp32 att:    {att_mb:6.1f} MB  "
            f"fp32 q/k/v temps: {qkv_fp32_mb:6.1f} MB\n"
            f"    total peak: ~{total_est_mb:6.1f} MB\033[0m"
        )

    def _alloc_fresh_attn_ctx(self, device):
        """Allocate a fresh SimpleNamespace ctx holding every fixed-size
        buffer _backpop_attention_padded_core reads from or writes to.
        Used only by the eager wrapper; GraphedBackwardRunner allocates
        persistent buffers directly in prepare()."""
        from types import SimpleNamespace
        Bn_max = int(self.ATTN_BN_MAX)
        L_max = int(self.ATTN_L_MAX)
        D = int(self.embed_dim_)
        Hd = int(self.head_dim_)
        # s_max for the eager path is Bn_max * L_max; any real batch that
        # passes the wrapper's assertions fits inside this.
        s_max = Bn_max * L_max

        ctx = SimpleNamespace()
        ctx.flat_in_padded       = torch.zeros((s_max, D),                device=device, dtype=torch.float16)
        ctx.grad_flat_in_padded  = torch.zeros((s_max, D),                device=device, dtype=torch.float16)
        ctx.bn_idx               = torch.zeros((s_max,),                  device=device, dtype=torch.long)
        ctx.pos_idx              = torch.zeros((s_max,),                  device=device, dtype=torch.long)
        ctx.pad_x_prev           = torch.zeros((Bn_max, L_max, D),        device=device, dtype=torch.float16)
        ctx.pad_grad_out         = torch.zeros((Bn_max, L_max, D),        device=device, dtype=torch.float16)
        ctx.pad_cos              = torch.zeros((Bn_max, L_max, Hd // 2),  device=device, dtype=self.model_weights._cos_cached.dtype)
        ctx.pad_sin              = torch.zeros((Bn_max, L_max, Hd // 2),  device=device, dtype=self.model_weights._sin_cached.dtype)
        ctx.key_pad_mask         = torch.ones ((Bn_max, L_max),           device=device, dtype=torch.bool)
        ctx.causal_mask          = torch.triu(
            torch.ones((L_max, L_max), device=device, dtype=torch.bool),
            diagonal=1,
        )
        ctx.flat_grad_x_prev_out = torch.zeros((s_max, D),                device=device, dtype=torch.float16)
        # ctx.lora_grad_out intentionally left unset — the core's fallback
        # branch Python-assigns the leaf's .grad directly for the eager path.
        return ctx

    def _fill_attn_ctx_per_backward(self, ctx):
        """Everything that depends on `seq_lens_py` but not on the per-layer
        inputs. Called ONCE per backward (not per layer) — the runner
        invokes this at the top of `run()`, the eager wrapper invokes it
        via `_fill_attn_ctx_eager`.

        Builds scatter indices, zeroes the flat input tails, and populates
        pad_cos / pad_sin / key_pad_mask. The `flat_in_padded` / `grad_flat_in_padded`
        tails stay zero for the rest of the backward, so the per-layer path
        only needs to overwrite `[:s_actual]` each call."""
        device = ctx.flat_in_padded.device
        seq_lens_py = self.activations.seq_lens_py
        Bn = len(seq_lens_py)
        s_max = ctx.flat_in_padded.shape[0]
        s_actual = sum(seq_lens_py) if Bn > 0 else 0

        # Build bn_idx / pos_idx on CPU, ship to GPU in one copy each.
        bn_idx_cpu = []
        pos_idx_cpu = []
        for i in range(Bn):
            ln = seq_lens_py[i]
            bn_idx_cpu.extend([i] * ln)
            pos_idx_cpu.extend(range(ln))
        # Pad tail with (0, 0). Harmless because flat_in_padded[tail] is
        # zero and the core's scatter uses index_put_(accumulate=True), so
        # repeated zero-adds to slot (0, 0) are no-ops.
        pad_len = s_max - s_actual
        if pad_len > 0:
            bn_idx_cpu.extend([0] * pad_len)
            pos_idx_cpu.extend([0] * pad_len)
        ctx.bn_idx.copy_(torch.tensor(bn_idx_cpu, dtype=torch.long, device=device))
        ctx.pos_idx.copy_(torch.tensor(pos_idx_cpu, dtype=torch.long, device=device))

        # Zero the flat input tails ONCE. The per-layer path only touches
        # [:s_actual] afterwards, so the tail remains zero throughout the
        # backward.
        ctx.flat_in_padded.zero_()
        ctx.grad_flat_in_padded.zero_()

        # Fill pad_cos / pad_sin from _cos_cached using the valid prefix
        # of bn_idx/pos_idx (variable-shape scatter, fine eagerly).
        ctx.pad_cos.zero_()
        ctx.pad_sin.zero_()
        if s_actual > 0:
            valid_bn = ctx.bn_idx[:s_actual]
            valid_pos = ctx.pos_idx[:s_actual]
            ctx.pad_cos[valid_bn, valid_pos] = self.model_weights._cos_cached[valid_pos]
            ctx.pad_sin[valid_bn, valid_pos] = self.model_weights._sin_cached[valid_pos]

        # Fill key_pad_mask: True at padded positions, False at real ones.
        ctx.key_pad_mask.fill_(True)
        if s_actual > 0:
            valid_bn = ctx.bn_idx[:s_actual]
            valid_pos = ctx.pos_idx[:s_actual]
            ctx.key_pad_mask[valid_bn, valid_pos] = False

    def _fill_attn_ctx_per_layer(self, ctx, last_layer_input, grad_ffn_input):
        """Per-layer update: just copies this layer's flat inputs into the
        persistent padded buffers. Assumes `_fill_attn_ctx_per_backward`
        has already zeroed the tails and built the scatter indices."""
        s_actual = last_layer_input.shape[0]
        ctx.flat_in_padded[:s_actual].copy_(last_layer_input.to(torch.float16))
        ctx.grad_flat_in_padded[:s_actual].copy_(grad_ffn_input.to(torch.float16))

    def _fill_attn_ctx_eager(self, ctx, last_layer_input, grad_ffn_input):
        """Full fill — used only by the eager wrapper `_backpop_attention_padded`,
        which allocates a fresh ctx every call. Calls both sub-steps."""
        self._fill_attn_ctx_per_backward(ctx)
        self._fill_attn_ctx_per_layer(ctx, last_layer_input, grad_ffn_input)

    @torch.no_grad()
    def _backpop_attention_padded_core(self, ctx, layer_weight, layer_id):
        """Shape-stable core of the padded attention backward.

        Every tensor inside this method has a shape that depends only on
        class constants (ATTN_BN_MAX, ATTN_L_MAX, D, Hq, Hkv, Hd, r) or on
        the fixed shapes of inputs attached to ctx. This makes the whole
        function graph-capturable as a single CUDA graph per layer.

        Reads from ctx (all stable .data_ptr() across replays):
          flat_in_padded       [S_max, D] fp16  — real data at :s_actual, zeros at s_actual:
          grad_flat_in_padded  [S_max, D] fp16  — same pattern
          bn_idx               [S_max]    int64 — valid mapping at :s_actual, zeros at s_actual:
          pos_idx              [S_max]    int64 — same
          pad_x_prev           [Bn_max, L_max, D]        fp16  — scratch, zeroed + scattered
          pad_grad_out         [Bn_max, L_max, D]        fp16  — scratch, zeroed + scattered
          pad_cos / pad_sin    [Bn_max, L_max, Hd/2]     — pre-filled by caller
          key_pad_mask         [Bn_max, L_max]           bool — pre-filled by caller
          causal_mask          [L_max, L_max]            bool — static, caller allocates once

        Writes:
          flat_grad_x_prev_out [S_max, D] fp16 — the gathered flat grad output
          (eager path) self.adapter_weights.lora_weights[layer_id].grad = g32
          (graph path, ctx.lora_grad_out set) ctx.lora_grad_out.copy_(g32)
        """
        Hq = int(self.num_heads_q_)
        Hkv = int(self.num_heads_kv_)
        Hd = int(self.head_dim_)
        D = int(self.embed_dim_)
        D_kv = Hkv * Hd
        assert D == Hq * Hd
        assert Hq % Hkv == 0
        kv_repeat = Hq // Hkv

        Bn_max = int(self.ATTN_BN_MAX)
        L_max = int(self.ATTN_L_MAX)
        N = Bn_max * L_max

        # ----- Internal scatter: flat inputs → padded buffers -----
        # index_put_(accumulate=True) handles the tail "sentinel" writes
        # cleanly: tail k's have (bn_idx[k], pos_idx[k]) = (0, 0) and
        # flat_in_padded[k] = 0, so they add zero to pad_x_prev[0, 0]
        # after the unique valid write for k=0 lands there.
        ctx.pad_x_prev.zero_()
        ctx.pad_x_prev.index_put_(
            (ctx.bn_idx, ctx.pos_idx), ctx.flat_in_padded, accumulate=True
        )
        ctx.pad_grad_out.zero_()
        ctx.pad_grad_out.index_put_(
            (ctx.bn_idx, ctx.pos_idx), ctx.grad_flat_in_padded, accumulate=True
        )

        # Combined attention mask — built inside core so it's captured;
        # shape-stable bool op.
        attn_mask = ctx.causal_mask.unsqueeze(0).unsqueeze(0) | ctx.key_pad_mask.unsqueeze(1).unsqueeze(1)
        # shape: [Bn_max, 1, L_max, L_max]

        # ----- Weights -----
        w_q = layer_weight.q_weight_
        w_k = layer_weight.k_weight_
        w_v = layer_weight.v_weight_
        w_o = layer_weight.o_weight_
        w_attn_norm = layer_weight.att_norm_weight_

        w_combined_leaf = self.adapter_weights.lora_weights[layer_id]
        w_fp = w_combined_leaf.to(torch.float16)
        r = w_fp.shape[1] // 4
        assert w_fp.shape[2] == Hq and w_fp.shape[3] == Hd

        qA = w_fp[0, 0:r].reshape(r, -1).T      # [D, r]
        qB = w_fp[1, 0:r].reshape(-1, r).T      # [r, D]
        kA = w_fp[0, r:2*r].reshape(r, -1).T
        kB = w_fp[1, r:2*r].reshape(-1, r).T
        vA = w_fp[0, 2*r:3*r].reshape(r, -1).T
        vB = w_fp[1, 2*r:3*r].reshape(-1, r).T
        oA = w_fp[0, 3*r:4*r].reshape(r, -1).T
        oB = w_fp[1, 3*r:4*r].reshape(-1, r).T

        scale_lora = self.adapter_weights.scaling
        scale = 1.0 / math.sqrt(Hd)

        def rotary_fwd_padded(x, cos, sin):
            Dh = Hd // 2
            x1 = x[..., :Dh]
            x2 = x[..., Dh:]
            cos_b = cos.unsqueeze(2).to(x.dtype)
            sin_b = sin.unsqueeze(2).to(x.dtype)
            y1 = x1 * cos_b - x2 * sin_b
            y2 = x2 * cos_b + x1 * sin_b
            return torch.cat([y1, y2], dim=-1)

        def rotary_bwd_padded(g, cos, sin):
            Dh = Hd // 2
            g1 = g[..., :Dh]
            g2 = g[..., Dh:]
            cos_b = cos.unsqueeze(2).to(g.dtype)
            sin_b = sin.unsqueeze(2).to(g.dtype)
            dx1 = g1 * cos_b + g2 * sin_b
            dx2 = -g1 * sin_b + g2 * cos_b
            return torch.cat([dx1, dx2], dim=-1)

        def proj_lora(X, A, B):
            return (X @ A @ B) * scale_lora

        # ----- Forward: RMSNorm, Q/K/V, LoRA, Rotary -----
        x_prev_flat = ctx.pad_x_prev.view(N, D)
        x_norm_flat = rmsnorm_forward(x_prev_flat, w_attn_norm, eps=self.eps_)
        X = x_norm_flat

        q_base = X @ w_q.to(dtype=X.dtype)
        k_base_raw = X @ w_k.to(dtype=X.dtype)
        v_base_raw = X @ w_v.to(dtype=X.dtype)

        if k_base_raw.shape[1] == D:
            k_base = k_base_raw[:, :D_kv]
            v_base = v_base_raw[:, :D_kv]
        else:
            k_base = k_base_raw
            v_base = v_base_raw
            assert k_base.shape[1] == D_kv

        q_lora = proj_lora(X, qA, qB)
        k_lora_full = proj_lora(X, kA, kB)
        v_lora_full = proj_lora(X, vA, vB)

        q = q_base + q_lora                        # [N, D]
        k = k_base + k_lora_full[:, :D_kv]         # [N, D_kv]
        v = v_base + v_lora_full[:, :D_kv]         # [N, D_kv]

        qh = q.view(Bn_max, L_max, Hq, Hd)
        kh = k.view(Bn_max, L_max, Hkv, Hd)
        vh = v.view(Bn_max, L_max, Hkv, Hd)

        qh_rot = rotary_fwd_padded(qh, ctx.pad_cos, ctx.pad_sin)
        kh_rot = rotary_fwd_padded(kh, ctx.pad_cos, ctx.pad_sin)

        if kv_repeat != 1:
            kh_rep = kh_rot.repeat_interleave(kv_repeat, dim=2)
            vh_rep = vh.repeat_interleave(kv_repeat, dim=2)
        else:
            kh_rep = kh_rot
            vh_rep = vh

        q_att = qh_rot.permute(0, 2, 1, 3).contiguous()
        k_att = kh_rep.permute(0, 2, 1, 3).contiguous()
        v_att = vh_rep.permute(0, 2, 1, 3).contiguous()

        # ----- Attention forward -----
        q_f = q_att.float()
        k_f = k_att.float()
        v_f = v_att.float()

        scores_pre_f = (q_f @ k_f.transpose(-1, -2)) * scale
        scores_pre_f = scores_pre_f.masked_fill(attn_mask, -1e9)
        scores_clamped_f = scores_pre_f.clamp(-80.0, 80.0)
        att_f = torch.softmax(scores_clamped_f, dim=-1)
        # Fully-masked rows (entirely padded requests) → softmax = nan; map to 0.
        att_f = torch.nan_to_num(att_f, nan=0.0)

        ctx_f = att_f @ v_f
        ctx_t = ctx_f.to(dtype=q_att.dtype)
        ctx_t = ctx_t.permute(0, 2, 1, 3).contiguous()
        ctx_flat = ctx_t.view(N, D)

        # ----- O projection (+LoRA) -----
        o_base = ctx_flat @ w_o.to(dtype=ctx_flat.dtype)
        Zo = ctx_flat @ oA
        o_lora = (Zo @ oB) * scale_lora
        o_total = o_base + o_lora

        # ----- Backward begins -----
        grad_out_flat = ctx.pad_grad_out.view(N, D)
        grad_x_prev_resid = grad_out_flat
        grad_o_total = grad_out_flat

        G = grad_o_total * scale_lora
        tmp = G @ oB.t()
        grad_oA = ctx_flat.t() @ tmp
        grad_oB = (ctx_flat @ oA).t() @ G
        grad_ctx_from_lora_o = tmp @ oA.t()
        grad_ctx_flat = (grad_o_total @ w_o.t()) + grad_ctx_from_lora_o

        grad_ctx = (grad_ctx_flat.view(Bn_max, L_max, Hq, Hd)
                                  .permute(0, 2, 1, 3)
                                  .contiguous())
        g_f = grad_ctx.float()

        # ----- Attention backward -----
        dV_rep_f = att_f.transpose(-1, -2) @ g_f
        dAtt_f = g_f @ v_f.transpose(-1, -2)

        s_sum = (dAtt_f * att_f).sum(dim=-1, keepdim=True)
        dScores_clamped_f = (dAtt_f - s_sum) * att_f
        dScores_clamped_f = dScores_clamped_f.masked_fill(attn_mask, 0.0)

        clamp_ok = (scores_pre_f > -80.0) & (scores_pre_f < 80.0)
        dScores_pre_f = dScores_clamped_f * clamp_ok

        dQ_f = (dScores_pre_f @ k_f) * scale
        dK_rep_f = (dScores_pre_f.transpose(-1, -2) @ q_f) * scale

        dQ = dQ_f.to(dtype=q_att.dtype)
        dK_rep = dK_rep_f.to(dtype=q_att.dtype)
        dV_rep = dV_rep_f.to(dtype=q_att.dtype)

        grad_qh_rot = dQ.permute(0, 2, 1, 3).contiguous()
        grad_kh_rep = dK_rep.permute(0, 2, 1, 3).contiguous()
        grad_vh_rep = dV_rep.permute(0, 2, 1, 3).contiguous()

        if kv_repeat != 1:
            grad_kh_rot = grad_kh_rep.view(Bn_max, L_max, Hkv, kv_repeat, Hd).sum(dim=3)
            grad_vh = grad_vh_rep.view(Bn_max, L_max, Hkv, kv_repeat, Hd).sum(dim=3)
        else:
            grad_kh_rot = grad_kh_rep
            grad_vh = grad_vh_rep

        grad_qh = rotary_bwd_padded(grad_qh_rot, ctx.pad_cos, ctx.pad_sin)
        grad_kh = rotary_bwd_padded(grad_kh_rot, ctx.pad_cos, ctx.pad_sin)

        gq_flat = grad_qh.reshape(N, D)
        gk_flat = grad_kh.reshape(N, D_kv)
        gv_flat = grad_vh.reshape(N, D_kv)

        # ----- Base Q/K/V backward -----
        grad_X_from_q = gq_flat @ w_q.t()
        if w_k.shape[1] == D_kv:
            grad_X_from_k = gk_flat @ w_k.t()
            grad_X_from_v = gv_flat @ w_v.t()
        else:
            grad_X_from_k = gk_flat @ w_k[:, :D_kv].t()
            grad_X_from_v = gv_flat @ w_v[:, :D_kv].t()

        # ----- LoRA Q backward -----
        G = gq_flat * scale_lora
        tmp = G @ qB.t()
        grad_qA = X.t() @ tmp
        grad_qB = (X @ qA).t() @ G
        grad_X_from_lora_q = tmp @ qA.t()

        # LoRA K/V need D-space padding to match monolithic convention.
        gk_pad_D = torch.zeros((N, D), device=X.device, dtype=X.dtype)
        gv_pad_D = torch.zeros((N, D), device=X.device, dtype=X.dtype)
        gk_pad_D[:, :D_kv] = gk_flat.to(dtype=X.dtype)
        gv_pad_D[:, :D_kv] = gv_flat.to(dtype=X.dtype)

        G = gk_pad_D * scale_lora
        tmp = G @ kB.t()
        grad_kA = X.t() @ tmp
        grad_kB = (X @ kA).t() @ G
        grad_X_from_lora_k = tmp @ kA.t()

        G = gv_pad_D * scale_lora
        tmp = G @ vB.t()
        grad_vA = X.t() @ tmp
        grad_vB = (X @ vA).t() @ G
        grad_X_from_lora_v = tmp @ vA.t()

        grad_X = (grad_X_from_q + grad_X_from_k + grad_X_from_v +
                  grad_X_from_lora_q + grad_X_from_lora_k + grad_X_from_lora_v)

        # ----- RMSNorm backward -----
        grad_from_norm = rmsnorm_backward(x_prev_flat, grad_X, w_attn_norm, eps=self.eps_)
        grad_x_prev_pad_flat = grad_from_norm + grad_x_prev_resid   # [N, D]

        # ----- Pack LoRA grads -----
        grad_combined = torch.zeros_like(w_fp)

        def pack_G(G, transpose_first):
            if transpose_first:
                G = G.t()
            return G.reshape(r, Hq, Hd)

        grad_combined[0, 0:r]     = pack_G(grad_qA.to(dtype=w_fp.dtype), True)
        grad_combined[0, r:2*r]   = pack_G(grad_kA.to(dtype=w_fp.dtype), True)
        grad_combined[0, 2*r:3*r] = pack_G(grad_vA.to(dtype=w_fp.dtype), True)
        grad_combined[0, 3*r:4*r] = pack_G(grad_oA.to(dtype=w_fp.dtype), True)

        grad_combined[1, 0:r]     = pack_G(grad_qB.to(dtype=w_fp.dtype), False)
        grad_combined[1, r:2*r]   = pack_G(grad_kB.to(dtype=w_fp.dtype), False)
        grad_combined[1, 2*r:3*r] = pack_G(grad_vB.to(dtype=w_fp.dtype), False)
        grad_combined[1, 3*r:4*r] = pack_G(grad_oB.to(dtype=w_fp.dtype), False)

        # Branchless grad clip in fp32.
        max_norm = 1.0
        g32 = grad_combined.float()
        gn = g32.norm()
        clip_scale = torch.clamp(max_norm / (gn + 1e-6), max=1.0)
        g32.mul_(clip_scale)

        # LoRA grad output.
        #
        # Graph path: ctx.lora_grad_out is a runner-allocated fp32 buffer
        # OUTSIDE the graph memory pool. The captured `.copy_` kernel writes
        # fresh g32 data into it every replay, and optimizer.step() reads
        # from the same stable address — eliminating the pool-aliasing bug
        # that corrupted LoRA weights in the previous capture attempt.
        #
        # Eager path: no ctx.lora_grad_out; fall back to the original Python
        # assignment so existing eager behavior is unchanged.
        if getattr(ctx, "lora_grad_out", None) is not None:
            ctx.lora_grad_out.copy_(g32)
        else:
            self.adapter_weights.lora_weights[layer_id].grad = g32

        # ----- Internal gather: padded grad_x_prev → flat_grad_x_prev_out -----
        # Fixed-size advanced indexing gather. Valid positions pull the
        # right per-request grad; sentinel tail positions read pad_x_prev[0, 0]
        # into the tail slots of flat_grad_x_prev_out, which the eager caller
        # discards by slicing [:s_actual].
        grad_x_prev_pad = grad_x_prev_pad_flat.view(Bn_max, L_max, D)
        ctx.flat_grad_x_prev_out.copy_(grad_x_prev_pad[ctx.bn_idx, ctx.pos_idx])

    # -----------------------------
    # Drop-in replacement
    # -----------------------------
    def _backpop_attention_autograd(
        self,
        last_layer_input: torch.Tensor,         # [S, D]
        grad_ffn_input: torch.Tensor,           # [S, D]  (this is dL/d(output_of_attn_residual))
        layer_weight,
        layer_id: int,
        batch_seq_lens: torch.Tensor,           # [B]
    ):
        device = last_layer_input.device

        Hq = int(self.num_heads_q_)
        Hkv = int(self.num_heads_kv_)
        Hd = int(self.head_dim_)
        D = int(self.embed_dim_)
        D_kv = Hkv * Hd
        assert D == Hq * Hd

        # ----- positions -----
        # (same semantics as your llama1 code)
        position_ids = torch.cat(
            [torch.arange(0, int(batch_seq_lens[i]), device=device) for i in range(len(batch_seq_lens))]
        )
        cos = self.model_weights._cos_cached.index_select(0, position_ids)  # [S, Hd/2]
        sin = self.model_weights._sin_cached.index_select(0, position_ids)  # [S, Hd/2]

        # ----- weights -----
        w_q = layer_weight.q_weight_          # [D, D] (likely fp16)
        w_k = layer_weight.k_weight_          # [D, D_kv] (likely)
        w_v = layer_weight.v_weight_          # [D, D_kv]
        w_o = layer_weight.o_weight_          # [D, D]
        w_attn_norm = layer_weight.att_norm_weight_  # [D]

        # ----- packed LoRA param -----
        # IMPORTANT: keep it as the optimizer param; do NOT detach/clone here.
        w_combined_leaf = self.adapter_weights.lora_weights[layer_id]
        w_combined_leaf.requires_grad_(True)

        # We will rebuild the forward with autograd enabled.
        # Make last_layer_input require grad so we can return grad to previous layer.
        x_prev = last_layer_input.detach().requires_grad_(True)

        scale_lora = self.adapter_weights.scaling  # scalar or 1-element tensor; assume broadcastable

        # Unpack r/Hq/Hd from packed tensor
        # w_combined_leaf: [2, 4r, Hq, Hd]
        w_fp = w_combined_leaf
        r = w_fp.shape[1] // 4
        assert w_fp.shape[2] == Hq and w_fp.shape[3] == Hd

        # Build A/B in the same way your llama1 service does (but autograd-safe)
        # qA: [D, r], qB: [r, D], etc. (all are views -> grads flow to w_combined_leaf)
        qA = w_fp[0, 0:r].reshape(r, -1).T
        qB = w_fp[1, 0:r].reshape(-1, r).T
        kA = w_fp[0, r:2*r].reshape(r, -1).T
        kB = w_fp[1, r:2*r].reshape(-1, r).T
        vA = w_fp[0, 2*r:3*r].reshape(r, -1).T
        vB = w_fp[1, 2*r:3*r].reshape(-1, r).T
        oA = w_fp[0, 3*r:4*r].reshape(r, -1).T
        oB = w_fp[1, 3*r:4*r].reshape(-1, r).T

        def proj_lora(X, A, B):
            # Fix your dtype error: A/B may be fp32 while X is fp16
            A_ = A.to(dtype=X.dtype)
            B_ = B.to(dtype=X.dtype)
            return (X @ A_ @ B_) * scale_lora

        # ----- forward (autograd-traceable) -----
        x_norm = _rmsnorm_pt(x_prev, w_attn_norm, eps=self.eps_)   # [S, D]
        X = x_norm

        # Base projections
        q_base = X @ w_q.to(dtype=X.dtype)                         # [S, D]
        k_base = X @ w_k.to(dtype=X.dtype)                         # [S, D_kv]
        v_base = X @ w_v.to(dtype=X.dtype)                         # [S, D_kv]

        # LoRA projections (full D for all four)
        q_lora = proj_lora(X, qA, qB)                              # [S, D]
        k_lora_full = proj_lora(X, kA, kB)                         # [S, D] (KV padded region is zeros by your construction)
        v_lora_full = proj_lora(X, vA, vB)                         # [S, D]

        # Apply padding rule: only first D_kv are valid for KV
        q = q_base + q_lora                                        # [S, D]
        k = k_base + k_lora_full[:, :D_kv]                         # [S, D_kv]
        v = v_base + v_lora_full[:, :D_kv]                         # [S, D_kv]

        # Reshape and rotary
        qh = q.view(-1, Hq, Hd)
        kh = k.view(-1, Hkv, Hd)
        vh = v.view(-1, Hkv, Hd)

        qh = _rotary_apply_pt(qh, cos, sin)
        kh = _rotary_apply_pt(kh, cos, sin)

        # Causal attention per request
        S_total = qh.shape[0]
        Bn = batch_seq_lens.shape[0]
        scale = 1.0 / math.sqrt(Hd)

        b_start = torch.cat(
            [torch.tensor([0], device=device), batch_seq_lens.cumsum(dim=0)[:-1]],
            dim=0
        )

        ctx = torch.empty((S_total, Hq, Hd), device=device, dtype=qh.dtype)

        kv_repeat = Hq // Hkv
        for i in range(Bn):
            st = int(b_start[i])
            ln = int(batch_seq_lens[i])

            q_blk = qh[st:st+ln].transpose(0, 1)        # [Hq, L, Hd]
            k_blk = kh[st:st+ln].transpose(0, 1)        # [Hkv, L, Hd]
            v_blk = vh[st:st+ln].transpose(0, 1)        # [Hkv, L, Hd]

            if kv_repeat != 1:
                k_rep = k_blk.repeat_interleave(kv_repeat, dim=0)  # [Hq, L, Hd]
                v_rep = v_blk.repeat_interleave(kv_repeat, dim=0)  # [Hq, L, Hd]
            else:
                k_rep, v_rep = k_blk, v_blk

            scores = (q_blk @ k_rep.transpose(-1, -2)) * scale   # fp16/bf16
            scores = scores.float()                             # <-- upcast BEFORE masked_fill
            mask = torch.triu(
                torch.ones((ln, ln), device=device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(mask.unsqueeze(0), -1e9) # safe now (fp32)
            # optional clamp
            scores = scores.clamp(-80.0, 80.0)
            att = torch.softmax(scores, dim=-1).to(q_blk.dtype)  # cast back if you want fp16 att

            ctx_blk = (att @ v_rep).transpose(0, 1)                # [L, Hq, Hd]
            ctx[st:st+ln] = ctx_blk

        ctx_flat = ctx.reshape(S_total, D)                         # [S, D]

        # O projection (+LoRA)
        o_base = ctx_flat @ w_o.to(dtype=ctx_flat.dtype)           # [S, D]
        o_lora = proj_lora(ctx_flat, oA, oB)                       # [S, D]
        o_total = o_base + o_lora                                  # [S, D]

        out = x_prev + o_total                                     # [S, D]

        # ----- vector-Jacobian product to match your incoming grad_ffn_input -----
        # grad_ffn_input is dL/d(out). Build a scalar whose gradient equals that VJP.
        g = grad_ffn_input.to(dtype=out.dtype)
        vjp = (out * g).sum()

        # Compute grads
        grad_x_prev, grad_w = torch.autograd.grad(
            vjp,
            [x_prev, w_combined_leaf],
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )

        max_norm = 1.0
        g32 = grad_w.float()
        gn = g32.norm()
        if gn > max_norm:
            g32.mul_(max_norm / (gn + 1e-6))

        # IMPORTANT: assign to the parameter the optimizer actually steps
        self.adapter_weights.lora_weights[layer_id].grad = g32
        # Return grad to previous layer (match your existing dtype expectations)
        return grad_x_prev.to(dtype=torch.float16)
    