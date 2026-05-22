"""load_alpaca.py — render Stanford Alpaca into the line-based format
DeltaServe's finetuning loader expects.

Pulls tatsu-lab/alpaca from HuggingFace, applies coarse char-length filters
to drop empty/very-long samples, randomly subsamples N (default 1000), and
writes one rendered prompt per line to alpaca_<N>.txt next to this script.

The finetuning_store loader (dserve/server/router/finetuning_store.py) is
line-based — one .strip()-ed line per sample — so newlines inside the
rendered prompt are flattened to single spaces here. The "### Instruction:"
/ "### Response:" markers still cue the model.

After running this, point finetune.data_path at the produced file and run
keep_p95.py to size attn_l_max precisely against the new distribution.

Example:
    python eval/llama3/data/load_alpaca.py
    python eval/llama3/data/load_alpaca.py --n 2000 --max-chars 800
"""

import argparse
import random
from pathlib import Path

from datasets import load_dataset


PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)


def render(sample: dict) -> str:
    instr = (sample.get("instruction") or "").strip()
    inp = (sample.get("input") or "").strip()
    out = (sample.get("output") or "").strip()
    if inp:
        text = PROMPT_WITH_INPUT.format(instruction=instr, input=inp, output=out)
    else:
        text = PROMPT_NO_INPUT.format(instruction=instr, output=out)
    # Flatten so the line-based loader sees one sample per line. Whitespace
    # collapse also strips the double-newlines from the template — the
    # "### X:" markers carry the structure on their own.
    return " ".join(text.split())


def passes_filter(sample: dict, min_chars: int, max_chars: int,
                  min_output_chars: int) -> bool:
    instr = (sample.get("instruction") or "").strip()
    out = (sample.get("output") or "").strip()
    if not instr or not out:
        return False
    if len(out) < min_output_chars:
        return False
    inp = (sample.get("input") or "").strip()
    total = len(instr) + len(inp) + len(out)
    if total < min_chars or total > max_chars:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000,
                        help="Number of samples to keep (default: 1000).")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for the random subsample (default: 42).")
    parser.add_argument("--min-chars", type=int, default=40,
                        help="Drop samples whose total instr+input+output "
                             "char length is below this (default: 40).")
    parser.add_argument("--max-chars", type=int, default=1200,
                        help="Drop samples whose total instr+input+output "
                             "char length exceeds this (default: 1200, ~300 "
                             "Llama-3 tokens). Tighten this to keep "
                             "attn_l_max small.")
    parser.add_argument("--min-output-chars", type=int, default=20,
                        help="Drop samples whose output is shorter than "
                             "this (default: 20). Filters out trivial "
                             "yes/no responses.")
    parser.add_argument("--out", type=str, default=None,
                        help="Output path. Default: alpaca_<n>.txt next to "
                             "this script.")
    args = parser.parse_args()

    print("Loading tatsu-lab/alpaca ...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    print(f"Loaded {len(ds)} raw samples.")

    print(f"Filtering (min_chars={args.min_chars}, max_chars={args.max_chars}, "
          f"min_output_chars={args.min_output_chars}) ...")
    kept_idx = [
        i for i, s in enumerate(ds)
        if passes_filter(s, args.min_chars, args.max_chars,
                         args.min_output_chars)
    ]
    print(f"Kept {len(kept_idx)} / {len(ds)} after filtering "
          f"({len(kept_idx) / len(ds) * 100:.1f}%).")

    if len(kept_idx) < args.n:
        raise SystemExit(
            f"Only {len(kept_idx)} samples pass the filter, fewer than "
            f"requested --n {args.n}. Loosen the filter or lower --n."
        )

    rng = random.Random(args.seed)
    chosen = rng.sample(kept_idx, args.n)
    chosen.sort()

    out_path = Path(args.out) if args.out else \
        Path(__file__).parent / f"alpaca_{args.n}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing {args.n} samples to {out_path} ...")
    with open(out_path, "w", encoding="utf-8") as f:
        for i in chosen:
            f.write(render(ds[i]) + "\n")
    print("Done.")


if __name__ == "__main__":
    main()
