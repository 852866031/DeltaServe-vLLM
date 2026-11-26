# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The padded-attention budget of the backward CUDA graph, chosen from the
finetuning dataset (``finetune_features.backward_cuda_graph_attn_bn_max`` /
``_l_max`` = ``auto``).

The backward's attention graph is captured for one padded shape: at most
``bn_max`` samples per backward, each at most ``l_max`` tokens. A backward that
does not fit runs its attention eager — correct, but back to several thousand
small Python-issued kernels per cycle (and, since torch 2.14, the
``torch/_native`` dispatch checks on each). Too small a budget therefore makes
the graph worthless without saying so (8 x 64 against 13 samples of ~19 tokens
did, for every cycle); too large a one pays for padding, quadratically in
``l_max``.

The corpus is tokenized at launch and admission takes the smallest untrained
sample first, so the batches of an epoch are known well enough to choose: this
module replays that packing and picks the cheapest shape that holds the wanted
share of the batches, within a bound on the scores tensor. What actually runs
also depends on what each step may admit and on aborted steps, so the choice
carries a margin and the eager fallback stays.
"""

from __future__ import annotations

import math

from deltaserve.ft_sizes import bwd_tokens_of, idle_bwd_tokens_of, step_tokens_of

AUTO = "auto"
# Fallback when there is no dataset to look at (the values before ``auto``).
DEFAULT_BN_MAX, DEFAULT_L_MAX = 8, 64


def _pack(lengths, cap: int) -> list[list[int]]:
    """Whole samples in order, groups of at most ``cap`` tokens."""
    groups: list[list[int]] = []
    cur: list[int] = []
    used = 0
    for n in lengths:
        if cur and used + n > cap:
            groups.append(cur)
            cur, used = [], 0
        cur.append(n)
        used += n
    if cur:
        groups.append(cur)
    return groups


def simulate_batches(lengths, batch_tokens: int,
                     step_tokens: int | None = None,
                     whole_batches: bool = False) -> list[tuple[int, int]]:
    """One epoch packed the way it is trained — the admission loop takes the
    SMALLEST untrained sample first (``FinetuningStore.pop_next``) until the
    step's buffer (``step_tokens``) is full, and a backward batch is whole
    samples in buffer order up to ``batch_tokens`` — as ``(samples, longest
    sample)`` per backward batch. So an epoch runs from batches of many short
    samples to batches of few long ones, and with several batches per step the
    last batch of a step is a partial one. Samples longer than a batch are
    skipped (the store drops them at load)."""
    cap = int(batch_tokens)
    if whole_batches:                 # the step ends on a batch boundary
        step_tokens = None
    step = max(cap, int(step_tokens or cap))
    order = sorted(int(x) for x in lengths if 0 < int(x) <= cap)
    return [(len(b), max(b)) for st in _pack(order, step) for b in _pack(st, cap)]


def choose_budget(lengths, batch_tokens: int, *, step_tokens: int | None = None,
                  coverage: float = 0.99,
                  margin: float = 1.15, local_heads: int = 1,
                  max_scores_mb: float = 256.0) -> dict:
    """The cheapest ``(bn_max, l_max)`` — cost ``bn · l²``, what the padded
    attention computes — under which at least ``coverage`` of the simulated
    batches fit, with ``margin`` on the sample count and the scores tensor
    (``bn · heads · l · l`` fp32) within ``max_scores_mb``. Returns the choice
    and what it was based on."""
    batches = simulate_batches(lengths, batch_tokens, step_tokens)
    info = {"batches": len(batches), "batch_tokens": int(batch_tokens),
            "samples": sum(b for b, _ in batches)}
    if not batches:
        return {**info, "bn_max": DEFAULT_BN_MAX, "l_max": DEFAULT_L_MAX,
                "coverage": 0.0, "reason": "no samples"}
    limit = max_scores_mb * (1 << 20)

    def scores_bytes(bn: int, l: int) -> float:
        return 4.0 * bn * max(1, int(local_heads)) * l * l

    def with_margin(bn: int) -> int:
        return int(math.ceil(bn * margin))

    need = coverage * len(batches)
    best = None           # (cost, bn, l, covered)
    fallback = None       # the best coverage reachable within the memory bound
    for l in sorted({longest for _, longest in batches}):
        fits_l = sorted(bn for bn, longest in batches if longest <= l)
        for i, bn in enumerate(fits_l):
            if i + 1 < len(fits_l) and fits_l[i + 1] == bn:
                continue                      # take the last of equal counts
            covered = i + 1
            bn_m = with_margin(bn)
            if scores_bytes(bn_m, l) > limit:
                break
            cand = (bn_m * l * l, bn_m, l, covered)
            if fallback is None or (covered, -cand[0]) > (fallback[3], -fallback[0]):
                fallback = cand
            if covered >= need and (best is None or cand[0] < best[0]):
                best = cand
                break                         # larger bn only costs more at this l
    chosen = best or fallback
    if chosen is None:
        # even one sample of the shortest length does not fit the bound
        bn, l = min(batches, key=lambda b: b[0] * b[1] * b[1])
        chosen = (bn * l * l, bn, l, 0)
        info["reason"] = "scores bound too small for any batch"
    _, bn_max, l_max, covered = chosen
    real = sum(min(b, bn_max) * lo for b, lo in batches) / len(batches)
    return {**info, "bn_max": int(bn_max), "l_max": int(l_max),
            "coverage": covered / len(batches),
            "scores_mb": scores_bytes(bn_max, l_max) / (1 << 20),
            "padded_tokens": int(bn_max * l_max),
            "bn_seen": (min(b for b, _ in batches), max(b for b, _ in batches)),
            "l_seen": (min(lo for _, lo in batches), max(lo for _, lo in batches)),
            "mean_longest_x_bn": real}


def choose_shapes(batches, bn_max: int, l_max: int, k: int,
                  margin: float = 1.15) -> list[tuple[int, int]]:
    """``k - 1`` more attention shapes INSIDE the budget ``bn_max x l_max``
    (``backward_cuda_graph_attn_shapes``), so that a backward pays for the
    padding of the smallest shape that holds it. An epoch runs from batches of
    many short samples to batches of few long ones, so the batches the budget
    holds are cut by their longest sample into ``k`` runs, each with the shape
    of its largest batch; the cut that minimises the padded work
    (Σ ``bn · l²`` over the batches) is found exactly. Returns the extra
    shapes (the budget itself keeps the last run and everything else)."""
    k = int(k)
    fit = sorted(((lo, bn) for bn, lo in batches if bn <= bn_max and lo <= l_max))
    if k <= 1 or len(fit) < 2:
        return []
    # distinct longest lengths: (l, count, max bn among batches up to here)
    groups: list[list[int]] = []
    for lo, bn in fit:
        if groups and groups[-1][0] == lo:
            groups[-1][1] += 1
            groups[-1][2] = max(groups[-1][2], bn)
        else:
            groups.append([lo, 1, bn])
    n = len(groups)
    k = min(k, n)

    def shape(i: int, j: int) -> tuple[int, int, int]:
        """Groups i..j as one run: (bn with margin, l, batches)."""
        bn = max(g[2] for g in groups[i:j + 1])
        cnt = sum(g[1] for g in groups[i:j + 1])
        return min(bn_max, int(math.ceil(bn * margin))), groups[j][0], cnt

    def cost(i: int, j: int, last: bool) -> float:
        bn, lo, cnt = shape(i, j)
        if last:                      # the budget takes the last run
            bn, lo = bn_max, l_max
        return float(cnt) * bn * lo * lo

    INF = float("inf")
    # best[c][j]: groups 0..j in c runs (the run ending at n-1 is the budget)
    best = [[INF] * n for _ in range(k + 1)]
    cut = [[-1] * n for _ in range(k + 1)]
    for j in range(n):
        best[1][j] = cost(0, j, j == n - 1)
    for c in range(2, k + 1):
        for j in range(c - 1, n):
            for i in range(c - 1, j + 1):
                v = best[c - 1][i - 1] + cost(i, j, j == n - 1)
                if v < best[c][j]:
                    best[c][j], cut[c][j] = v, i
    c = min(range(1, k + 1), key=lambda x: best[x][n - 1])
    runs: list[tuple[int, int]] = []          # (first group, last group)
    j = n - 1
    while c >= 1:
        i = cut[c][j] if c > 1 else 0
        runs.append((i, j))
        j, c = i - 1, c - 1
    out = {shape(i, j)[:2] for i, j in runs if j != n - 1}
    out.discard((int(bn_max), int(l_max)))
    return sorted(out, key=lambda x: x[0] * x[1] * x[1])


def extra_shapes(ft_cfg, bn_max: int, l_max: int, tokenizer_name=None, *,
                 lengths=None) -> list[tuple[int, int]]:
    """The extra attention shapes for a resolved budget ([] with
    ``backward_cuda_graph_attn_shapes`` <= 1 or without a dataset)."""
    k = int(getattr(ft_cfg, "backward_cuda_graph_attn_shapes", 1) or 1)
    if k <= 1:
        return []
    try:
        if lengths is None:
            lengths = dataset_lengths(ft_cfg, tokenizer_name)
    except Exception:  # noqa: BLE001 — only a speed-up
        return []
    if not lengths:
        return []
    batches = simulate_batches(lengths, batch_tokens_of(ft_cfg), _step_of(ft_cfg))
    return choose_shapes(batches, int(bn_max), int(l_max), k)


def batch_tokens_of(ft_cfg) -> int:
    """FT tokens per backward batch (``bwd_batch_size``)."""
    return bwd_tokens_of(ft_cfg)


def _step_of(ft_cfg) -> int | None:
    """The step size the packing follows (None with ``bwd_whole_batches``: a
    step then ends on a batch boundary, every batch is a full one)."""
    if getattr(ft_cfg, "bwd_whole_batches", False):
        return None
    return step_tokens_of(ft_cfg)


def coverage_of(lengths, batch_tokens: int, bn_max: int, l_max: int,
                step_tokens: int | None = None) -> float:
    """Share of the simulated backward batches a given budget holds."""
    batches = simulate_batches(lengths, batch_tokens, step_tokens)
    if not batches:
        return 1.0
    return sum(1 for bn, lo in batches if bn <= bn_max and lo <= l_max) / len(batches)


_LENGTHS: dict = {}


def dataset_lengths(ft_cfg, tokenizer_name: str | None) -> list[int]:
    key = (getattr(ft_cfg, "data_path", None), tokenizer_name,
           batch_tokens_of(ft_cfg), getattr(ft_cfg, "max_prepare", None))
    if key not in _LENGTHS:
        _LENGTHS[key] = _dataset_lengths(ft_cfg, tokenizer_name)
    return _LENGTHS[key]


def _dataset_lengths(ft_cfg, tokenizer_name: str | None) -> list[int]:
    """Token length of every sample the store would train on (tokenized like
    the scheduler's store does; empty when there is no corpus or tokenizer)."""
    path = getattr(ft_cfg, "data_path", None)
    if not path or not tokenizer_name:
        return []
    from transformers import AutoTokenizer

    from deltaserve.finetuning_store import FinetuningStore

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    store = FinetuningStore(
        path, lambda text: tok(text)["input_ids"],
        max_saved_finetuning_tokens=batch_tokens_of(ft_cfg),
        max_prepare=getattr(ft_cfg, "max_prepare", None))
    store.load()
    return [int(s.input_len) for s in store.samples]


def _is_auto(v) -> bool:
    return isinstance(v, str) and v.strip().lower() == AUTO


def resolve_idle(ft_cfg, tokenizer_name: str | None = None, *, lengths=None,
                 local_heads: int = 1) -> tuple[int, int]:
    """The attention budget of the SECOND graph set (``bwd_batch_size_idle``):
    always chosen from the dataset, for batches of the idle size (the
    built-in values scaled by the size ratio without one)."""
    big, small = idle_bwd_tokens_of(ft_cfg), batch_tokens_of(ft_cfg)
    try:
        if lengths is None:
            lengths = dataset_lengths(ft_cfg, tokenizer_name)
    except Exception:  # noqa: BLE001 — the budget is only a speed-up
        lengths = []
    if not lengths:
        return DEFAULT_BN_MAX * max(1, big // max(1, small)), DEFAULT_L_MAX
    info = choose_budget(lengths, big, local_heads=local_heads)
    return int(info["bn_max"]), int(info["l_max"])


def resolve(ft_cfg, tokenizer_name: str | None = None, *, lengths=None,
            local_heads: int = 1) -> tuple[int, int, dict | None]:
    """``(bn_max, l_max, info)`` for the backward child. A number in the config
    is taken as it is; ``auto`` on either key is chosen from the dataset
    (``info`` then says how; None when nothing was chosen)."""
    bn = getattr(ft_cfg, "backward_cuda_graph_attn_bn_max", DEFAULT_BN_MAX)
    l = getattr(ft_cfg, "backward_cuda_graph_attn_l_max", DEFAULT_L_MAX)
    if not (_is_auto(bn) or _is_auto(l)):
        info = None
        try:
            if lengths is None:
                lengths = dataset_lengths(ft_cfg, tokenizer_name)
            if lengths:
                info = {"configured": True, "batch_tokens": batch_tokens_of(ft_cfg),
                        "coverage": coverage_of(lengths, batch_tokens_of(ft_cfg),
                                                int(bn), int(l),
                                                _step_of(ft_cfg)),
                        "suggest": choose_budget(lengths, batch_tokens_of(ft_cfg),
                                                 step_tokens=_step_of(ft_cfg),
                                                 local_heads=local_heads)}
        except Exception:  # noqa: BLE001 — only a diagnostic
            info = None
        return int(bn), int(l), info
    if lengths is None:
        try:
            lengths = dataset_lengths(ft_cfg, tokenizer_name)
        except Exception as e:  # noqa: BLE001 — the budget is only a speed-up
            lengths = []
            err = f"{type(e).__name__}: {e}"
        else:
            err = None
    else:
        err = None
    if not lengths:
        info = {"reason": err or "no dataset to choose from", "batches": 0}
        return (DEFAULT_BN_MAX if _is_auto(bn) else int(bn),
                DEFAULT_L_MAX if _is_auto(l) else int(l), info)
    info = choose_budget(lengths, batch_tokens_of(ft_cfg),
                         step_tokens=_step_of(ft_cfg), local_heads=local_heads)
    return (info["bn_max"] if _is_auto(bn) else int(bn),
            info["l_max"] if _is_auto(l) else int(l), info)


def describe(bn: int, l: int, info: dict | None) -> str:
    if info is None:
        return f"backward attention graph budget: {bn} samples x {l} tokens (configured)"
    if info.get("configured"):
        cov, sug = info["coverage"], info["suggest"]
        msg = (f"backward attention graph budget: {bn} samples x {l} tokens "
               f"(configured): {100 * cov:.0f} % of the dataset's backward batches fit")
        if cov < 0.9:
            msg += (f" — !!! the others run their attention EAGER (slow). Use "
                    f"{sug['bn_max']} x {sug['l_max']} "
                    f"({100 * sug['coverage']:.0f} %), or 'auto'")
        return msg
    if not info.get("batches"):
        return (f"backward attention graph budget: {bn} x {l} (auto: "
                f"{info.get('reason', 'no data')} — the built-in values)")
    return (f"backward attention graph budget: {bn} samples x {l} tokens (auto: "
            f"{100 * info['coverage']:.1f} % of {info['batches']} simulated "
            f"batches of <= {info['batch_tokens']} tokens fit; they hold "
            f"{info['bn_seen'][0]}-{info['bn_seen'][1]} samples, longest "
            f"{info['l_seen'][0]}-{info['l_seen'][1]} tokens; scores "
            f"{info['scores_mb']:.1f} MB)")
