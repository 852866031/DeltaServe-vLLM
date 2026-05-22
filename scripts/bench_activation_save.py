#!/usr/bin/env python
"""Microbenchmark: per-step cost of SAVING FT activations vs NOT saving.

The eager forward is identical whether or not we save activations, so the
overhead of "save" is exactly the per-step save block the accumulation hooks
run: for each transformer layer, a masked gather of the attention-output and
FFN-output rows + a copy into the shared buffer, plus the final hidden states
and target ids. This times that block in isolation (CUDA events), at the real
shapes, so the number is the added latency per FT step in the eager case.

For context it also times an "eager forward proxy" — the dominant per-layer
matmuls (attention qkv/o + MLP) at the same token count — and prints the save
overhead as a % of that. The proxy omits attention softmax / norms, so the real
forward is larger and the true % is LOWER than printed (this is an upper bound).

    python bench_activation_save.py --preset opt125m
    python bench_activation_save.py --preset llama3-8b
"""

import argparse

import torch

# (layers, hidden, [per-layer matmuls as (k_in, n_out)])
_PRESETS = {
    # OPT-125m: qkv 768->2304, o 768->768, fc1 768->3072, fc2 3072->768
    "opt125m": (12, 768, [(768, 2304), (768, 768), (768, 3072), (3072, 768)]),
    # Llama-3-8B: GQA qkv 4096->6144, o 4096->4096, gate/up 4096->14336, down 14336->4096
    "llama3-8b": (32, 4096, [(4096, 6144), (4096, 4096),
                             (4096, 14336), (4096, 14336), (14336, 4096)]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(_PRESETS), default="opt125m")
    ap.add_argument("--total-tokens", type=int, default=262)
    ap.add_argument("--ft-tokens", type=int, default=256)
    ap.add_argument("--max-saved", type=int, default=512)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    args = ap.parse_args()

    dev = "cuda"
    dt = torch.float16
    L, H, matmuls = _PRESETS[args.preset]
    total, n = args.total_tokens, args.ft_tokens
    off = 0

    # --- activation-save block (what FinetuneAccumulator does per FT step) ---
    attn_out = [torch.randn(total, H, device=dev, dtype=dt) for _ in range(L)]
    ffn_out = [torch.randn(total, H, device=dev, dtype=dt) for _ in range(L)]
    hidden = torch.randn(total, H, device=dev, dtype=dt)
    input_ids = torch.randint(0, 50000, (total,), device=dev, dtype=torch.long)
    mask = torch.zeros(total, dtype=torch.bool, device=dev)
    mask[:n] = True
    attn_buf = [torch.zeros(args.max_saved, H, device=dev, dtype=dt) for _ in range(L)]
    ffn_buf = [torch.zeros(args.max_saved, H, device=dev, dtype=dt) for _ in range(L)]
    final_buf = torch.zeros(args.max_saved, H, device=dev, dtype=dt)
    ids_buf = torch.zeros(args.max_saved, device=dev, dtype=torch.long)

    # precomputed integer FT-row indices (in reality built once/step from the
    # CPU mask we already construct in _build_finetune_mask, then H2D'd once)
    ft_idx = mask.nonzero(as_tuple=True)[0].to(torch.int64)

    def save_current():
        # CURRENT: boolean-mask gather (out[mask] -> nonzero() -> device SYNC) + copy
        for layer in range(L):
            r = attn_out[layer][mask]
            attn_buf[layer][off:off + n].copy_(r[:n].to(dt))
            r = ffn_out[layer][mask]
            ffn_buf[layer][off:off + n].copy_(r[:n].to(dt))
        r = hidden[mask]
        final_buf[off:off + n].copy_(r[:n].to(dt))
        ids_buf[off:off + n].copy_(input_ids[mask][:n].to(torch.int64))

    def save_index_select():
        # OPT A: precomputed int indices + index_select straight into the buffer
        # (one kernel/layer, no intermediate, no per-layer sync)
        idx = mask.nonzero(as_tuple=True)[0]  # once per step (1 sync)
        for layer in range(L):
            torch.index_select(attn_out[layer], 0, idx, out=attn_buf[layer][off:off + n])
            torch.index_select(ffn_out[layer], 0, idx, out=ffn_buf[layer][off:off + n])
        torch.index_select(hidden, 0, idx, out=final_buf[off:off + n])
        torch.index_select(input_ids, 0, idx, out=ids_buf[off:off + n])

    def save_slice():
        # OPT B (best): FT rows laid out contiguously -> plain slice copy, no
        # gather/index/sync at all
        for layer in range(L):
            attn_buf[layer][off:off + n].copy_(attn_out[layer][:n])
            ffn_buf[layer][off:off + n].copy_(ffn_out[layer][:n])
        final_buf[off:off + n].copy_(hidden[:n])
        ids_buf[off:off + n].copy_(input_ids[:n])

    save_step = save_current  # default timed below; all three timed in __main__

    # --- eager forward proxy (dominant per-layer matmuls) ---
    inputs = [torch.randn(total, k, device=dev, dtype=dt) for (k, _) in matmuls]
    weights = [torch.randn(k, nn, device=dev, dtype=dt) for (k, nn) in matmuls]

    def forward_proxy():
        for _ in range(L):
            for x, w in zip(inputs, weights):
                _ = x @ w

    def timeit(fn):
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / args.iters  # ms/iter

    cur_ms = timeit(save_current)
    idx_ms = timeit(save_index_select)
    slice_ms = timeit(save_slice)
    fwd_ms = timeit(forward_proxy)

    print(f"preset={args.preset}  layers={L} hidden={H}  "
          f"total_tokens={total} ft_tokens={n}")
    print(f"  forward proxy (matmuls only)        = {fwd_ms * 1000:8.1f} us")
    for name, ms in [("current  (boolean out[mask])", cur_ms),
                     ("opt A    (index_select->buf)", idx_ms),
                     ("opt B    (contiguous slice)", slice_ms)]:
        print(f"  {name:34s} = {ms * 1000:8.1f} us  "
              f"| {100 * ms / fwd_ms:5.1f}% of fwd  "
              f"| {cur_ms / ms:4.1f}x vs current")


if __name__ == "__main__":
    main()
