import triton
import triton.language as tl
import torch


@triton.jit
def copy_rows_kernel(dst_ptr, rows_ptr, gpu_idx_ptr,
                     N, H, D, stride_dst_H, stride_dst_D,
                     stride_row_H, stride_row_D,
                     BLOCK_H: tl.constexpr,
                     BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= N:
        return

    # Compute source and destination base addresses
    dst_index = tl.load(gpu_idx_ptr + pid)  # scalar
    h_offsets = tl.arange(0, BLOCK_H)
    d_offsets = tl.arange(0, BLOCK_D)

    for h in range(0, H, BLOCK_H):
        for d in range(0, D, BLOCK_D):
            h_mask = h_offsets + h < H
            d_mask = d_offsets + d < D

            row_ptr = rows_ptr + pid * stride_row_H * H + (h + h_offsets[:, None]) * stride_row_H + (d + d_offsets[None, :]) * stride_row_D
            dst_row_ptr = dst_ptr + dst_index * stride_dst_H * H + (h + h_offsets[:, None]) * stride_dst_H + (d + d_offsets[None, :]) * stride_dst_D

            val = tl.load(row_ptr, mask=h_mask[:, None] & d_mask[None, :])
            tl.store(dst_row_ptr, val, mask=h_mask[:, None] & d_mask[None, :])

def triton_copy_rows(dst: torch.Tensor, rows: torch.Tensor, gpu_idx: torch.Tensor):
    assert dst.ndim == 3 and rows.ndim == 3
    assert rows.shape[0] == gpu_idx.shape[0]
    N, H, D = rows.shape
    BLOCK_H = 8
    BLOCK_D = 32

    dst_ptr = dst.contiguous()
    rows_ptr = rows.contiguous()
    gpu_idx = gpu_idx.contiguous()

    grid = (N,)

    copy_rows_kernel[grid](
        dst_ptr, rows_ptr, gpu_idx,
        N=N, H=H, D=D,
        stride_dst_H=dst.stride(1),
        stride_dst_D=dst.stride(2),
        stride_row_H=rows.stride(1),
        stride_row_D=rows.stride(2),
        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D,
    )