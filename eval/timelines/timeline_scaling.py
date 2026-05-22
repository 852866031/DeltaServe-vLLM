"""timeline_scaling.py — rescale a request timeline's RPS.

Reads a `timeline_*.csv` (the same schema auto_benchmark consumes:
timestamp_s, prompt_length, max_new_tokens, second, index_in_second)
and writes a rescaled copy plus a 2-row PNG for visual comparison.

Semantics
---------
SCALE_FACTOR < 1 ⇒ lower RPS by *dropping* requests. Original
    timestamps are preserved verbatim; we keep a uniformly random
    `round(N * SCALE_FACTOR)`-sized subset (seeded for reproducibility).
    This preserves the burst pattern exactly — bursts appear in the
    same wall-clock windows, just sparser.

SCALE_FACTOR > 1 ⇒ higher RPS by *compressing* the time axis.
    Every timestamp is multiplied by 1/SCALE_FACTOR; the request count
    and per-request shape are preserved. Note: this also stretches the
    pre-roll before the first event, so bursts visually shift toward
    t=0 (smaller for early events, larger for late ones).

SCALE_FACTOR == 1 ⇒ no-op (still writes a copy + plot for sanity).

Configuration is hardcoded below. To run a different scale / file, edit
the constants in the CONFIG block and re-run.
"""

import csv
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ─── CONFIG ───────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent

# Which timeline to rescale. The default is the 5090 `live` schedule
# because it's the one auto_benchmark picks up by default; swap to
# A100/ or another mode if you want a different starting point.
INPUT_CSV = _HERE / "A100" / "timeline_nutanix.csv"

# RPS multiplier. Examples:
#   2.0  → twice as many requests per second (same N reqs over half the time)
#   0.5  → half as many requests per second (same N reqs over twice the time)
#   1.0  → no-op (still writes a copy + a plot so you can sanity-check the path)
SCALE_FACTOR = 0.25

# Output paths are derived from INPUT_CSV + SCALE_FACTOR so multiple
# scalings of the same source don't clobber each other.
def _scale_tag(scale: float) -> str:
    """e.g. 1.5 → 'x1p5', 0.5 → 'x0p5', 2.0 → 'x2', 1.0 → 'x1'."""
    s = f"{scale:g}".replace(".", "p")
    return f"x{s}"


_TAG = _scale_tag(SCALE_FACTOR)
OUTPUT_CSV = INPUT_CSV.with_name(f"{INPUT_CSV.stem}_{_TAG}{INPUT_CSV.suffix}")
PLOT_PATH = INPUT_CSV.with_name(f"{INPUT_CSV.stem}_{_TAG}.png")

# Per-second binning width for the bar chart in the plot. The CSVs use
# 1s buckets in their `second` column; keep the same here.
BIN_WIDTH_S = 1.0

# Seed for the random subset selection in the downscale path. Fixed so
# repeated runs with the same input + scale produce the same kept set.
DROP_SEED = 0xD5E27E
# ──────────────────────────────────────────────────────────────────────


def load_timeline(path: Path):
    """Return a list of dicts with the original schema preserved.
    `timestamp_s`, `prompt_length`, `max_new_tokens` are typed; the
    `second` / `index_in_second` columns are recomputed by the rescaler
    so we don't bother typing them here."""
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"timestamp_s", "prompt_length", "max_new_tokens"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"[timeline_scaling] {path}: missing columns {sorted(missing)}")
        for r in reader:
            rows.append({
                "timestamp_s": float(r["timestamp_s"]),
                "prompt_length": int(float(r["prompt_length"])),
                "max_new_tokens": int(float(r["max_new_tokens"])),
            })
    rows.sort(key=lambda x: x["timestamp_s"])
    return rows


def _recompute_buckets(rows):
    """In-place: refresh the `second` and `index_in_second` columns from
    each row's current `timestamp_s`. Rows must already be sorted by
    `timestamp_s`."""
    sec_counter = {}
    for r in rows:
        sec = int(r["timestamp_s"])
        r["second"] = sec
        r["index_in_second"] = sec_counter.get(sec, 0)
        sec_counter[sec] = r["index_in_second"] + 1


def rescale(rows, scale: float):
    """Dispatch on `scale`:
      * scale < 1 → drop a uniformly random subset (timestamps untouched).
      * scale > 1 → compress timestamps by 1/scale (count untouched).
      * scale == 1 → return a fresh copy unchanged.
    `second` / `index_in_second` are recomputed in every case so the
    output CSV stays internally consistent."""
    if scale <= 0:
        sys.exit(f"[timeline_scaling] SCALE_FACTOR must be > 0, got {scale}")

    if scale < 1.0:
        scaled = _rescale_drop(rows, scale)
    elif scale > 1.0:
        scaled = _rescale_compress(rows, scale)
    else:
        scaled = [dict(r) for r in rows]

    scaled.sort(key=lambda x: x["timestamp_s"])
    _recompute_buckets(scaled)
    return scaled


def _rescale_drop(rows, scale: float):
    """Down-scale by random subsampling. Keeps original timestamps so
    the burst pattern in wall-clock time stays put — RPS drops because
    fewer requests land in each window, not because the windows move."""
    n = len(rows)
    keep_n = int(round(n * scale))
    if keep_n >= n:
        return [dict(r) for r in rows]
    if keep_n <= 0:
        return []
    rng = random.Random(DROP_SEED)
    indices = list(range(n))
    rng.shuffle(indices)
    kept = sorted(indices[:keep_n])
    return [dict(rows[i]) for i in kept]


def _rescale_compress(rows, scale: float):
    """Up-scale by compressing the time axis. Same N requests, each at
    timestamp / scale (rounded to 3 decimals so the CSV stays readable
    and the in-memory value matches what we write)."""
    inv = 1.0 / scale
    return [
        {
            "timestamp_s": round(r["timestamp_s"] * inv, 3),
            "prompt_length": r["prompt_length"],
            "max_new_tokens": r["max_new_tokens"],
        }
        for r in rows
    ]


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["timestamp_s", "prompt_length", "max_new_tokens",
                  "second", "index_in_second"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})


def _per_second_counts(rows, bin_width: float = BIN_WIDTH_S):
    """Histogram-style request count per bin. Returns (bin_centers, counts).
    Uses fixed bin_width=1s by default to match the CSV's `second` column."""
    if not rows:
        return np.array([]), np.array([])
    ts = np.array([r["timestamp_s"] for r in rows], dtype=float)
    t_max = float(ts.max()) + bin_width
    edges = np.arange(0.0, t_max + bin_width, bin_width)
    counts, _ = np.histogram(ts, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, counts


def _stats_blurb(rows) -> str:
    if not rows:
        return "0 requests"
    n = len(rows)
    span = max(r["timestamp_s"] for r in rows) - min(r["timestamp_s"] for r in rows)
    span = max(span, 1e-9)
    return f"N={n} | span={span:.1f}s | mean RPS={n / span:.2f}"


def plot_comparison(orig, scaled, out_path: Path, scale: float):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

    for ax, rows, label, color in (
        (axes[0], orig,   "original",            "tab:blue"),
        (axes[1], scaled, f"scaled ×{scale:g}",  "tab:orange"),
    ):
        centers, counts = _per_second_counts(rows)
        ax.bar(centers, counts, width=BIN_WIDTH_S * 0.95,
               color=color, alpha=0.55, edgecolor="none",
               label="reqs / s")
        if counts.size:
            ax.plot(centers, counts, color=color, linewidth=1.0, alpha=0.9)
        ax.set_ylabel("requests / second")
        ax.set_title(f"{label}  ({_stats_blurb(rows)})", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    # Share an x-range that covers both so the visual scaling is honest.
    max_t = 0.0
    for rows in (orig, scaled):
        if rows:
            max_t = max(max_t, max(r["timestamp_s"] for r in rows))
    for ax in axes:
        ax.set_xlim(0, max_t * 1.02)
    axes[1].set_xlabel("time (s)")

    fig.suptitle(
        f"{INPUT_CSV.name} — RPS rescale by ×{scale:g}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    if not INPUT_CSV.exists():
        sys.exit(f"[timeline_scaling] input not found: {INPUT_CSV}")
    orig = load_timeline(INPUT_CSV)
    print(f"[timeline_scaling] loaded {len(orig)} rows from {INPUT_CSV}")

    scaled = rescale(orig, SCALE_FACTOR)
    write_csv(scaled, OUTPUT_CSV)
    print(f"[timeline_scaling] wrote scaled timeline → {OUTPUT_CSV}")
    print(f"[timeline_scaling]   original: {_stats_blurb(orig)}")
    print(f"[timeline_scaling]   scaled : {_stats_blurb(scaled)}")

    plot_comparison(orig, scaled, PLOT_PATH, SCALE_FACTOR)
    print(f"[timeline_scaling] wrote comparison plot → {PLOT_PATH}")


if __name__ == "__main__":
    main()
