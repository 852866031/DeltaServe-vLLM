
import torch.nn.functional as F
from dserve.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
import torch
import math
from dserve._kernels import dispatch_bgmv

import torch

@torch.no_grad()
def rotary_emb_fwd_pt(q: torch.Tensor,
                      cos: torch.Tensor,
                      sin: torch.Tensor) -> None:
    T, H, D = q.shape
    assert D % 2 == 0, "head_dim must be even"
    Dh = D // 2
    assert cos.shape == (T, Dh) and sin.shape == (T, Dh), \
        f"cos/sin expected {(T, Dh)}, got {cos.shape}/{sin.shape}"
    q_flat = q.reshape(T * H, D)
    q_even = q_flat[:, :Dh] 
    q_odd  = q_flat[:, Dh:]
    cos_exp = cos[:, None, :].expand(T, H, Dh).reshape(T * H, Dh)
    sin_exp = sin[:, None, :].expand(T, H, Dh).reshape(T * H, Dh)
    q_even_orig = q_even.clone()
    q_even.mul_(cos_exp).addcmul_(q_odd, sin_exp, value=-1.0)
    q_odd.mul_(cos_exp).addcmul_(q_even_orig, sin_exp)

@torch.no_grad()
def dispatch_bgmv_pt_exact(
    x: torch.Tensor,   # [N, Din] or [N, r]
    w: torch.Tensor,   # [size, H, Hd]
    start_indices: torch.Tensor,
    lora_ranks: torch.Tensor,
    loc_indices: torch.Tensor,
    indices: torch.Tensor,
    qkvo: int,
    lora_scales: torch.Tensor,
    first_launch: bool
):
    results = []
    N = x.size(0)

    for i in range(N):
        a_id = indices[i].item()
        R = lora_ranks[a_id].item() // 4
        start = start_indices[a_id].item()
        scale = lora_scales[a_id].item()
        rows = loc_indices[start + qkvo * R : start + (qkvo + 1) * R]

        w_block = w.index_select(0, rows)  # [R, H, Hd]
        if first_launch:
            # A-side: Aᵀ = w_block.view(R, H).T
            A = w_block.reshape(R, -1).T  # [Din, R]
            result = x[i:i+1] @ A  # [1, R]
        else:
            # B-side: Bᵀ = w_block.view(H, R).T
            B = w_block.reshape(-1, R).T  # [R, D]
            result = x[i:i+1] @ B * scale  # [1, D]

        results.append(result)

    return torch.cat(results, dim=0)  # [N, R] or [N, D]

@torch.no_grad()
def dispatch_bgmv_pt(
        x: torch.Tensor,   # [N, Din]
        w: torch.Tensor,   # [size, H, Hd]
        start_indices: torch.Tensor,
        lora_ranks: torch.Tensor,
        loc_indices: torch.Tensor,
        indices: torch.Tensor,
        qkvo: int,
        lora_scales: torch.Tensor,
        first_launch: bool
):
    device, dtype = x.device, x.dtype
    N, Din = x.shape
    a_id = indices[0].item()
    r = int(lora_ranks[a_id]) // 4  # Real rank per Q/K/V/O
    scale = lora_scales[a_id].item()
    start_idx = start_indices[a_id].item()
    outer_row_idx = loc_indices[start_idx + r * qkvo : start_idx + r * (qkvo + 1)].to(device)
    block = w.index_select(0, outer_row_idx)  # [r, H, Hd]
    H, Hd = block.shape[1:]

    if first_launch:
        A_raw = block.reshape(r, -1).T 
        out = torch.matmul(x, A_raw)  # [N, r]
    else:
        B_raw = block.reshape(-1, r).T 
        out = torch.matmul(x, B_raw)  # [N, D]
        out.mul_(scale)

    return out.to(dtype)

def compare_tensors(name, a, b):
    diff = (a - b).float()
    print(f"[{name}] shape={list(a.shape)}")
    print(f"  L2 error : {diff.norm():.4f}")
    print(f"  Mean |Δ|  : {diff.abs().mean().item():.6f}")
    print(f"  Max  |Δ|  : {diff.abs().max().item():.6f}")
    print()

def run_unit_test(D=1024, r=16, N=4, H=4, qkvo=0, scale=1.0):
    """
    D: hidden dim (e.g., 64)
    r: LoRA rank (per Q/K/V/O), total = 4r
    N: batch size
    H: num_heads
    """
    device = torch.device("cuda")
    dtype = torch.float16

    x = torch.randn(N, D, dtype=dtype, device=device)
    wA = torch.randn(r * 4, H, D // H, dtype=dtype, device=device)
    wB = torch.randn(r * 4, H, D // H, dtype=dtype, device=device)

    delta = torch.zeros(N, r, dtype=dtype, device=device)
    y = torch.zeros(N, D, dtype=dtype, device=device)

    start = torch.tensor([0], dtype=torch.long, device=device)
    ranks = torch.tensor([r * 4], dtype=torch.long, device=device)
    locs = torch.arange(r * 4, dtype=torch.long, device=device)
    idxs = torch.zeros(N, dtype=torch.long, device=device)
    scales = torch.tensor([scale], dtype=dtype, device=device)

    # CUDA Shrink
    dispatch_bgmv(delta, x, wA, start, ranks, locs, idxs, qkvo, scales)
    delta_ref = delta.clone()

    # CUDA Expand
    dispatch_bgmv(y, delta, wB, start, ranks, locs, idxs, qkvo, scales)
    y_ref = y.clone()

    # PyTorch Shrink
    delta_pt = dispatch_bgmv_pt(x, wA, start, ranks, locs, idxs, qkvo, scales, first_launch=True)

    # PyTorch Expand
    y_pt = dispatch_bgmv_pt(delta_pt, wB, start, ranks, locs, idxs, qkvo, scales, first_launch=False)

    # PyTorch exact Shrink
    delta_pt_2 = dispatch_bgmv_pt_exact(x, wA, start, ranks, locs, idxs, qkvo, scales, first_launch=True)

    # PyTorch exact Expand
    y_pt_2 = dispatch_bgmv_pt_exact(delta_pt, wB, start, ranks, locs, idxs, qkvo, scales, first_launch=False)

    compare_tensors("delta diff", delta_pt, delta_ref)
    compare_tensors("final y diff", y_pt, y_ref)

    compare_tensors("delta diff exact", delta_pt_2, delta_ref)
    compare_tensors("final y diff exact", y_pt_2, y_ref)

if __name__ == "__main__":
    run_unit_test()
