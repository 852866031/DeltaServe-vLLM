"""Drop the longest N% of samples from a fine-tuning data file.

Tokenizes every line, computes the --percentile cutoff (default p95), and
writes out only the samples whose token count is at or below the cutoff.
The filtered file is what you feed the fine-tuning service once you've
tightened ATTN_L_MAX to match the p95 — any sample longer than that would
otherwise trigger the monolithic fallback in _backpop_attention_padded.

Also sanity-checks the filtered set against --bn-max for a given --budget
and warns if the worst-case distinct-sample pack would still exceed
ATTN_BN_MAX (you'd see monolithic fallbacks on small-sample-dominated
batches).

Usage:

    python eval/llama3/keep_p95.py data.txt data_p95.txt
    python eval/llama3/keep_p95.py data.txt data_p95.txt \
        --tokenizer meta-llama/Meta-Llama-3-8B \
        --percentile 95 \
        --bn-max 8 --budget 256
"""

import argparse
import os
import sys
from typing import List, Sequence, Tuple


def load_samples(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def tokenize(samples: Sequence[str], tokenizer_name: str) -> List[int]:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    return [len(tokenizer.encode(s, add_special_tokens=True)) for s in samples]


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


def greedy_pack_max_bn(sorted_asc: Sequence[int], budget: int) -> Tuple[int, int]:
    """Greedy-pack distinct samples starting from the shortest until the
    next one would blow the budget. Returns (num_samples, total_tokens)."""
    total = 0
    for i, ln in enumerate(sorted_asc):
        if total + ln > budget:
            return i, total
        total += ln
    return len(sorted_asc), total


def summarize(lens: Sequence[int], label: str) -> None:
    if not lens:
        print(f"  {label}: empty")
        return
    s = sorted(lens)
    n = len(s)
    print(f"  {label}:")
    print(f"    count  : {n}")
    print(f"    tokens : total={sum(s)}, min={s[0]}, median={int(percentile(s, 50))}, "
          f"mean={sum(s)/n:.1f}, p95={int(percentile(s, 95))}, max={s[-1]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Input text file, one sample per line.")
    parser.add_argument("output", help="Output text file for the filtered samples.")
    parser.add_argument(
        "--tokenizer",
        default="meta-llama/Meta-Llama-3-8B",
        help="HF tokenizer name or local model dir (default: Meta-Llama-3-8B).",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=95.0,
        help="Percentile cutoff (default: 95 — keeps samples whose token count "
             "is <= the 95th percentile length; drops the longest 5%%).",
    )
    parser.add_argument(
        "--bn-max",
        type=int,
        default=8,
        help="ATTN_BN_MAX to sanity-check the filtered set against (default: 8).",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=256,
        help="Token budget per backward batch for the BN sanity check "
             "(default: 256 — matches max_saved_finetuning_tokens).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if os.path.exists(args.output) and not args.force:
        print(
            f"Error: output file already exists: {args.output} "
            f"(pass --force to overwrite)",
            file=sys.stderr,
        )
        sys.exit(1)
    if not (0.0 < args.percentile < 100.0):
        print(
            f"Error: --percentile must be in (0, 100), got {args.percentile}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading samples from {args.input} ...")
    samples = load_samples(args.input)
    print(f"Loaded {len(samples)} non-empty samples.")
    if not samples:
        print("Nothing to filter.", file=sys.stderr)
        sys.exit(0)

    print(f"Tokenizing with '{args.tokenizer}' ...")
    try:
        lens = tokenize(samples, args.tokenizer)
    except Exception as e:
        print(f"Tokenizer load/encode failed: {e}", file=sys.stderr)
        sys.exit(1)

    s_sorted = sorted(lens)
    cutoff = int(percentile(s_sorted, args.percentile))
    print()
    print(f"p{args.percentile:.0f} token-count cutoff: {cutoff}")
    print(f"Keeping samples with token count <= {cutoff}.")

    kept_samples: List[str] = []
    kept_lens: List[int] = []
    dropped = 0
    for sample, ln in zip(samples, lens):
        if ln <= cutoff:
            kept_samples.append(sample)
            kept_lens.append(ln)
        else:
            dropped += 1

    print()
    print("Summary:")
    summarize(lens, "original ")
    summarize(kept_lens, "filtered ")
    print(f"  dropped    : {dropped} sample(s) "
          f"({dropped / max(1, len(samples)) * 100.0:.2f}%)")

    # BN_MAX sanity check on the filtered set.
    print()
    print(f"ATTN_BN_MAX sanity check (bn_max={args.bn_max}, budget={args.budget}):")
    if not kept_lens:
        print("  (nothing to check)")
    else:
        filtered_sorted = sorted(kept_lens)
        worst_bn, worst_tokens = greedy_pack_max_bn(filtered_sorted, args.budget)
        print(f"  worst-case distinct pack at {args.budget}-token budget: "
              f"{worst_bn} reqs, {worst_tokens} tokens used")
        if worst_bn > args.bn_max:
            print(
                f"  WARNING: worst-case Bn={worst_bn} exceeds ATTN_BN_MAX={args.bn_max}. "
                f"Small-sample-dominated batches will still hit the monolithic "
                f"fallback. Consider either (a) bumping ATTN_BN_MAX to >= {worst_bn}, "
                f"or (b) also dropping the shortest samples so the 9th-shortest "
                f"cumulative sum exceeds the budget."
            )
        else:
            print(f"  OK: worst-case Bn={worst_bn} <= ATTN_BN_MAX={args.bn_max}.")

    # Write output.
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in kept_samples:
            f.write(sample + "\n")
    print()
    print(f"Wrote {len(kept_samples)} samples to {args.output}.")


if __name__ == "__main__":
    main()
