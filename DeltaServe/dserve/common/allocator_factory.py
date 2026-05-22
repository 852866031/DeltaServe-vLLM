import torch

from dserve.common.unified_mem_allocator import UnifiedMemoryAllocator
from dserve.common.packed_kv_mem_allocator import PackedKVMemoryAllocator


_REGISTRY = {
    "unified":   UnifiedMemoryAllocator,
    "packed_kv": PackedKVMemoryAllocator,
}

_AUTO = "auto"


def get_allocator_class(name: str):
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown allocator '{name}'. choices: {sorted(_REGISTRY) + [_AUTO]}"
        )
    return _REGISTRY[name]


def _resolve_auto(head_num, num_kv_heads) -> str:
    """Pick an allocator based on the model's GQA factor F = head_num / num_kv_heads.

    F > 1 (GQA, e.g. Llama-3 with 32 q-heads / 8 kv-heads) → packed_kv saves
    ~F× memory in the KV portion of the pool by packing F sub-slots per page.
    F == 1 (MHA, e.g. Llama-1/2) → packed_kv is byte-for-byte identical to
    unified; pick unified to avoid unnecessary subclass code paths.
    """
    if num_kv_heads is None or num_kv_heads >= head_num:
        return "unified"
    return "packed_kv"


def make_allocator(name: str, *, head_num, head_dim, vocab_size, layer_num,
                   max_pool_size, dtype=torch.float16, device="cuda",
                   log_path=None, max_finetuning_tokens=1024,
                   num_kv_heads=None):
    """Build the allocator selected by `name`. `name` may be "auto" — in
    which case the GQA factor decides (see `_resolve_auto`). Subclass-
    specific kwargs are forwarded only when relevant (e.g. num_kv_heads
    for packed_kv)."""
    if name == _AUTO:
        resolved = _resolve_auto(head_num, num_kv_heads)
        print(
            f"[allocator_factory] memory.allocator='auto' → selected "
            f"'{resolved}' (head_num={head_num}, num_kv_heads={num_kv_heads}, "
            f"F={head_num // (num_kv_heads or head_num)})"
        )
        name = resolved
    cls = get_allocator_class(name)
    extra = {}
    if cls is PackedKVMemoryAllocator:
        if num_kv_heads is None:
            num_kv_heads = head_num   # MHA fallback (F = 1)
        extra["num_kv_heads"] = num_kv_heads
    return cls(
        head_num=head_num,
        head_dim=head_dim,
        vocab_size=vocab_size,
        layer_num=layer_num,
        max_pool_size=max_pool_size,
        dtype=dtype,
        device=device,
        log_path=log_path,
        max_finetuning_tokens=max_finetuning_tokens,
        **extra,
    )
