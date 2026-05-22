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
        batch_seq_lens: torch.Tensor,           # [B]
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

        # ----- positions -----
        position_ids = torch.cat(
            [torch.arange(0, int(batch_seq_lens[i]), device=device) for i in range(len(batch_seq_lens))]
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

        # per-request offsets
        Bn = batch_seq_lens.shape[0]
        b_start = torch.cat(
            [torch.tensor([0], device=device), batch_seq_lens.cumsum(dim=0)[:-1]],
            dim=0
        )

        # Attention forward (store ctx only; att is recomputed in backward)
        ctx = torch.empty((S_total, Hq, Hd), device=device, dtype=qh_rot.dtype)
        scale = 1.0 / math.sqrt(Hd)

        for i in range(Bn):
            st = int(b_start[i])
            ln = int(batch_seq_lens[i])

            q_blk = qh_rot[st:st+ln].transpose(0, 1)      # [Hq,L,Hd]
            k_blk = kh_rot[st:st+ln].transpose(0, 1)      # [Hkv,L,Hd]
            v_blk = vh[st:st+ln].transpose(0, 1)          # [Hkv,L,Hd]  (note: no rotary on v)

            if kv_repeat != 1:
                k_rep = k_blk.repeat_interleave(kv_repeat, dim=0)  # [Hq,L,Hd]
                v_rep = v_blk.repeat_interleave(kv_repeat, dim=0)
            else:
                k_rep, v_rep = k_blk, v_blk

            scores = (q_blk @ k_rep.transpose(-1, -2)) * scale      # [Hq,L,L] fp16
            scores = scores.float()                                  # fp32 for stability
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
            st = int(b_start[i])
            ln = int(batch_seq_lens[i])

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

        # ----- clip in fp32, assign fp32 grad to optimizer leaf -----
        max_norm = 1.0
        g32 = grad_combined.float()
        gn = g32.norm()
        if gn > max_norm:
            g32.mul_(max_norm / (gn + 1e-6))

        self.adapter_weights.lora_weights[layer_id].grad = g32
        return grad_x_prev.to(dtype=torch.float16)

    @torch.no_grad()
    def _backpop_attention_1(
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

            scores = (q_blk @ k_rep.transpose(-1, -2)) * scale      # [Hq,L,L] fp16
            scores = scores.float()                                  # fp32 for stability
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
    