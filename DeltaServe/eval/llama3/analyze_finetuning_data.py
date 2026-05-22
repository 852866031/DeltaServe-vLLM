"""Analyze a fine-tuning data file to pick ATTN_BN_MAX / ATTN_L_MAX.

Reads a plain-text file with one training sample per line, tokenizes each
with the configured model's tokenizer, and prints:

  * Per-sample token count distribution (min / median / mean / max / p95 / p99)
  * Worst-case packing: given a token budget B, how many samples of the
    SHORTEST observed length could fit (this is what drives ATTN_BN_MAX)
  * Best-case packing: how many samples of the MAX observed length fit
  * Pad waste projection: for a chosen (Bn_max, L_max), what fraction of
    the padded budget is wasted on sentinel positions vs real tokens

Usage:

    python eval/llama3/analyze_finetuning_data.py path/to/data.txt
    python eval/llama3/analyze_finetuning_data.py path/to/data.txt \
        --tokenizer meta-llama/Meta-Llama-3-8B \
        --budget 512 \
        --bn-max 6 --l-max 256
"""

import argparse
import os
import statistics
import sys
from typing import List, Sequence


def load_samples(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    # Drop blank lines but keep samples with leading/trailing spaces.
    samples = [ln for ln in lines if ln.strip()]
    return samples


def tokenize(samples: Sequence[str], tokenizer_name: str) -> List[int]:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    lens: List[int] = []
    for s in samples:
        ids = tokenizer.encode(s, add_special_tokens=True)
        lens.append(len(ids))
    return lens


def percentile(sorted_values: Sequence[int], p: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    k = (len(sorted_values) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = k - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def bucketize(values: Sequence[int], edges: Sequence[int]) -> List[int]:
    """Return count of values within each [edges[i], edges[i+1]) bucket,
    plus a trailing [edges[-1], inf) bucket."""
    counts = [0] * len(edges)
    for v in values:
        for i in range(len(edges) - 1, -1, -1):
            if v >= edges[i]:
                counts[i] += 1
                break
    return counts


def print_distribution(lens: List[int]) -> None:
    n = len(lens)
    if n == 0:
        print("  (no samples)")
        return
    s = sorted(lens)
    total = sum(s)
    print(f"  samples           : {n}")
    print(f"  total tokens      : {total}")
    print(f"  min               : {s[0]}")
    print(f"  p25               : {percentile(s, 25):.1f}")
    print(f"  median            : {percentile(s, 50):.1f}")
    print(f"  mean              : {statistics.mean(s):.1f}")
    print(f"  p75               : {percentile(s, 75):.1f}")
    print(f"  p90               : {percentile(s, 90):.1f}")
    print(f"  p95               : {percentile(s, 95):.1f}")
    print(f"  p99               : {percentile(s, 99):.1f}")
    print(f"  max               : {s[-1]}")

    print()
    print("  length histogram:")
    edges = [0, 16, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 2048]
    counts = bucketize(s, edges)
    max_count = max(counts) if counts else 1
    for i, e in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else "inf"
        bar = "#" * max(0, int(40 * counts[i] / max_count)) if max_count > 0 else ""
        print(f"    [{e:>4}, {str(hi):>5}): {counts[i]:>6}  {bar}")


def greedy_pack_max_bn(sorted_asc: Sequence[int], budget: int) -> (int, int):
    """Greedy-pack DISTINCT samples starting from the shortest.

    Returns (max_bn, total_tokens_used). Since each sample is consumed at
    most once per batch, the answer is bounded by the dataset size and by
    the prefix-sum cutoff on the sorted-ascending lengths."""
    total = 0
    for i, ln in enumerate(sorted_asc):
        if total + ln > budget:
            return i, total
        total += ln
    return len(sorted_asc), total


def print_packing(lens: List[int], budget: int) -> None:
    if not lens:
        return
    s = sorted(lens)
    n = len(s)
    shortest = s[0]
    longest = s[-1]
    median = percentile(s, 50)
    p95 = percentile(s, 95)

    # Worst case: greedy-pack DISTINCT samples starting from the shortest
    # until we blow the budget. Cannot reuse samples, so the count is
    # bounded both by prefix-sum and by the dataset size.
    worst_case_bn, worst_case_tokens = greedy_pack_max_bn(s, budget)

    # Typical: pretend every sample in the batch has the median length.
    # Bn is bounded by both `budget // median` and dataset size.
    typical_bn = max(1, min(n, budget // max(1, int(median))))

    # Best case (fewest concurrent requests): one longest sample alone.
    best_case_bn = 1 if longest <= budget else 0

    print(f"  Token budget        : {budget}")
    print(f"  Dataset size        : {n} distinct samples")
    print(f"  Longest single req  : {longest}  "
          f"({'fits' if longest <= budget else 'DOES NOT FIT — exceeds budget'})")
    print(f"  Shortest single req : {shortest}")
    print(f"  Median sample       : {int(median)}")
    print(f"  p95 sample          : {int(p95)}")
    print()
    print(f"  Requests per batch at {budget}-token budget (distinct samples only):")
    print(f"    worst case  (greedy-pack shortest distinct samples)")
    print(f"                          : {worst_case_bn} reqs, {worst_case_tokens} tokens used")
    print(f"    typical     (all at median length)         : ~{typical_bn} reqs")
    print(f"    best case   (one longest sample)           : ~{best_case_bn} reqs")
    print()
    print(f"  → ATTN_BN_MAX must be at least {worst_case_bn} to avoid the eager fallback")
    print(f"    in the worst case (small-sample-dominated batch).")


def print_padding_waste(lens: List[int], bn_max: int, l_max: int, budget: int) -> None:
    """For a given (Bn_max, L_max), estimate compute blowup of the padded
    attention path vs the monolithic per-request path. Assumes distinct
    samples per batch (scheduler can't reuse a sample within one backward)."""
    if not lens:
        return
    s = sorted(lens)
    n = len(s)
    median = max(1, int(percentile(s, 50)))
    p95 = max(1, int(percentile(s, 95)))
    max_len = s[-1]

    # Padded shape cost is always constant:
    padded_linear = bn_max * l_max                # dim-linear ops
    padded_quad = bn_max * (l_max ** 2)           # attention quadratic ops

    # Real-workload Bn_real when every sample in the batch has length L_real.
    # Capped by (a) Bn_max (the graphed path would fall back past this),
    # (b) len(dataset) — no duplicate samples in one batch, and
    # (c) budget // L_real — token-budget packing.
    def bn_real_for(L_real: int) -> int:
        return max(1, min(bn_max, n, budget // max(1, L_real)))

    def safe_ratio(num, den):
        if den == 0:
            return float("inf")
        return num / den

    print(f"  Assumed config: ATTN_BN_MAX={bn_max}, ATTN_L_MAX={l_max}")
    print(f"  Padded linear ops : {padded_linear:,}  (Bn_max * L_max)")
    print(f"  Padded quad ops   : {padded_quad:,}  (Bn_max * L_max^2)")
    print()
    print(f"  Compute blowup vs. a real batch packed to {budget} tokens:")
    print(f"  (Bn_real is capped by min(ATTN_BN_MAX, dataset size, budget/L))")
    print(f"    {'assumption':<24}{'Bn_real':>8}{'linear×':>12}{'quad×':>12}{'~atten×':>12}")
    for name, L_real in (
        ("median length", median),
        ("p95 length",    p95),
        ("max length",    max_len),
    ):
        bn_real = bn_real_for(L_real)
        lin = bn_real * L_real
        quad = bn_real * (L_real ** 2)
        lin_ratio = safe_ratio(padded_linear, lin)
        quad_ratio = safe_ratio(padded_quad, quad)
        # Blended heuristic: attention backward time is ~50/50 linear and
        # quadratic in Llama-style models.
        atten_ratio = 0.5 * lin_ratio + 0.5 * quad_ratio
        print(f"    {name:<24}{bn_real:>8}{lin_ratio:>11.1f}×{quad_ratio:>11.1f}×{atten_ratio:>11.1f}×")

    # Extra row: the actual worst-case greedy pack (distinct shortest
    # samples) gives the highest Bn_real and therefore the most favorable
    # padded-vs-real ratio — this is the scenario the graphed path is
    # designed to handle.
    worst_bn, worst_tokens = greedy_pack_max_bn(s, budget)
    if worst_bn > 0:
        avg_len_in_worst = worst_tokens / worst_bn
        # For the blowup, approximate the real quad work as
        # sum(L_i^2) ≈ Bn * (avg L)^2. This slightly overestimates on
        # skewed distributions but is close enough for a rule of thumb.
        lin = worst_tokens
        quad = worst_bn * (avg_len_in_worst ** 2)
        lin_ratio = safe_ratio(padded_linear, lin)
        quad_ratio = safe_ratio(padded_quad, quad)
        atten_ratio = 0.5 * lin_ratio + 0.5 * quad_ratio
        label = "worst-case distinct pack"
        print(f"    {label:<24}{worst_bn:>8}{lin_ratio:>11.1f}×{quad_ratio:>11.1f}×{atten_ratio:>11.1f}×")

    print()
    print("  Interpretation: `atten×` < 1.5 is cheap. 1.5–3 is OK if graph capture")
    print("  savings cover it. >3 is almost certainly slower than monolithic eager.")


def suggest_l_max(lens: List[int], bn_max: int, budget: int) -> None:
    """Suggest an ATTN_L_MAX that matches the p95 sequence length rounded
    up to a friendly size."""
    if not lens:
        return
    p95 = int(percentile(sorted(lens), 95))
    p99 = int(percentile(sorted(lens), 99))
    max_len = max(lens)

    def round_up_to_power_of_2(x: int) -> int:
        p = 1
        while p < x:
            p <<= 1
        return p

    print(f"  p95 length               : {p95}")
    print(f"  p99 length               : {p99}")
    print(f"  max length               : {max_len}")
    print()
    print("  Suggested ATTN_L_MAX choices (round up to nearest power of 2):")
    for name, target in (("p95", p95), ("p99", p99), ("max", max_len)):
        rounded = round_up_to_power_of_2(target)
        print(f"    cover {name:>3} ({target:>4}) → ATTN_L_MAX = {rounded}")
    print()
    print("  Trade-off reminder: doubling ATTN_L_MAX ~4× attention compute.")
    print("  Prefer the tightest value that still covers your real workload.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_file",
        help="Path to a text file with one fine-tuning sample per line.",
    )
    parser.add_argument(
        "--tokenizer",
        default="meta-llama/Meta-Llama-3-8B",
        help="HF tokenizer name or local model dir (default: Meta-Llama-3-8B).",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=256,
        help=(
            "Token budget per backward batch (maps to finetuning_config's "
            "max_saved_finetuning_tokens). Default: 256."
        ),
    )
    parser.add_argument(
        "--bn-max",
        type=int,
        default=8,
        help="ATTN_BN_MAX to evaluate (default: 8).",
    )
    parser.add_argument(
        "--l-max",
        type=int,
        default=256,
        help="ATTN_L_MAX to evaluate (default: 256).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data_file):
        print(f"Error: file not found: {args.data_file}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading samples from {args.data_file} ...")
    samples = load_samples(args.data_file)
    print(f"Loaded {len(samples)} non-empty samples.")
    if not samples:
        sys.exit(0)

    print(f"Tokenizing with '{args.tokenizer}' (this may download the tokenizer)...")
    try:
        lens = tokenize(samples, args.tokenizer)
    except Exception as e:
        print(f"Tokenizer load/encode failed: {e}", file=sys.stderr)
        print("Try passing --tokenizer with a local model directory.", file=sys.stderr)
        sys.exit(1)

    print()
    print("=" * 60)
    print("Token count distribution")
    print("=" * 60)
    print_distribution(lens)

    print()
    print("=" * 60)
    print(f"Batch packing at token budget = {args.budget}")
    print("=" * 60)
    print_packing(lens, args.budget)

    print()
    print("=" * 60)
    print(f"Padding waste for ATTN_BN_MAX={args.bn_max}, ATTN_L_MAX={args.l_max}")
    print("=" * 60)
    print_padding_waste(lens, args.bn_max, args.l_max, args.budget)

    print()
    print("=" * 60)
    print("ATTN_L_MAX suggestions")
    print("=" * 60)
    suggest_l_max(lens, args.bn_max, args.budget)


if __name__ == "__main__":
    main()
