# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The sizes of a finetuning step, read from any config object: FT tokens
per optimizer step (``ft_step_size``: the activation buffer's capacity) and per
backward batch (``bwd_batch_size``). Kept free of imports — the backward child's
handover thread and the tests' stand-in configs use it without loading vLLM's
config machinery."""


def step_tokens_of(cfg) -> int:
    """``cfg.step_tokens`` (FT tokens per optimizer step = the activation
    buffer), also for a config object that only carries the single-size key."""
    v = getattr(cfg, "step_tokens", None)
    return int(v if v is not None else cfg.max_saved_finetuning_tokens)


def bwd_tokens_of(cfg) -> int:
    """``cfg.bwd_tokens`` (FT tokens per backward batch), with the same
    fallback."""
    v = getattr(cfg, "bwd_tokens", None)
    return int(v if v is not None else step_tokens_of(cfg))


def fwd_tokens_of(cfg) -> int:
    """FT tokens ONE forward step may carry (``ft_tokens_per_step``): at most
    a backward batch, which is also the default."""
    b = bwd_tokens_of(cfg)
    v = getattr(cfg, "ft_tokens_per_step", None)
    return int(b if v is None else max(1, min(int(v), b)))


def ft_only_tokens_of(cfg) -> int:
    """FT tokens a step WITHOUT inference tokens may carry
    (``ft_only_tokens_per_step``): between ``ft_tokens_per_step`` (the
    default) and a backward batch."""
    f, b = fwd_tokens_of(cfg), max(bwd_tokens_of(cfg), idle_bwd_tokens_of(cfg))
    v = getattr(cfg, "ft_only_tokens_per_step", None)
    return int(f if v is None else max(f, min(int(v), b)))


def idle_bwd_tokens_of(cfg) -> int:
    """FT tokens per backward batch while no inference request is in the
    system (``bwd_batch_size_idle``): between a backward batch (= off, the
    default) and the step."""
    b, st = bwd_tokens_of(cfg), step_tokens_of(cfg)
    v = getattr(cfg, "bwd_batch_size_idle", None)
    return int(b if v is None else max(b, min(int(v), st)))


def buffer_tokens_of(cfg) -> int:
    """Rows of the activation buffer (``ft_buffer_size``): the step (the
    default), or fewer — whole backward batches, at least one — when the
    buffer is a ring of batch slots."""
    b, st = bwd_tokens_of(cfg), step_tokens_of(cfg)
    v = getattr(cfg, "ft_buffer_size", None)
    if v is None:
        return int(st)
    return int(max(b, min(int(v) // b * b, st)))


def ring_slots_of(cfg) -> int:
    """Batch slots of the ring (0 = the buffer holds the step: no ring)."""
    buf, st = buffer_tokens_of(cfg), step_tokens_of(cfg)
    return 0 if buf >= st else buf // bwd_tokens_of(cfg)
