#!/usr/bin/env python3
"""margin_stress_gen.py — the request timeline behind the admission-margin A/B
(eval-tp/margin_ab.py).

The SLO admission loop fills each prefill step with finetuning tokens until the
PREDICTED step time reaches ``max_tbt_slo`` (when decodes are running) or the
TTFT deadline. Whatever the estimator under-predicts then lands as a step over
the limit. This timeline makes that visible:

- decodes are almost always running (outputs 64-256 tokens), so the max-TBT
  check is active on nearly every step;
- prompts are log-uniform in 16-160 tokens, so prefill steps span several-fold
  in time and the estimator's error has to hold across shapes, not at one point
  (the stock timelines are all 50 / 50);
- arrivals are Poisson with a rate that alternates 1.5 and 3 req/s every 15 s, so
  the decode population (and the step shape) keeps moving.

    python scripts/timelines/margin_stress_gen.py      # → data/timelines/5090/timeline_margin_stress.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_ROOT / "data/timelines/5090/timeline_margin_stress.csv"))
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--start", type=float, default=3.0, help="first arrival (s)")
    ap.add_argument("--rates", default="1.5,3", help="req/s, alternated every --period")
    ap.add_argument("--period", type=float, default=15.0)
    ap.add_argument("--prompt", default="16,160", help="log-uniform prompt length range")
    ap.add_argument("--output", default="64,256", help="uniform output length range")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rates = [float(x) for x in a.rates.split(",")]
    p_lo, p_hi = (int(x) for x in a.prompt.split(","))
    o_lo, o_hi = (int(x) for x in a.output.split(","))
    rows, t = [], a.start
    while True:
        rate = rates[int((t - a.start) // a.period) % len(rates)]
        t += rng.expovariate(rate)
        if t >= a.start + a.duration:
            break
        p = int(round(math.exp(rng.uniform(math.log(p_lo), math.log(p_hi)))))
        rows.append((round(t, 6), p, rng.randint(o_lo, o_hi)))
    per_sec: dict[int, int] = {}
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp_s", "prompt_length", "max_new_tokens", "second", "index_in_second"])
        for ts, p, o in rows:
            s = int(ts)
            i = per_sec.get(s, 0)
            per_sec[s] = i + 1
            w.writerow([ts, p, o, s, i])
    print(f"{len(rows)} requests over {a.duration:.0f} s → {a.out}")


if __name__ == "__main__":
    main()
