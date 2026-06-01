"""timeline_slice.py — cut a sub-window out of a request timeline.

Reads a `timeline_*.csv` (the same schema auto_benchmark consumes:
timestamp_s, prompt_length, max_new_tokens, second, index_in_second),
keeps the rows whose `timestamp_s` falls in a [START, END) window, and
writes a new timeline **rebased so the original START maps to t=0**.

Example
-------
Slice the default 5090 nutanix trace between 600s and 800s::

    python timeline_slice.py --start 600 --end 800

→ writes `timelines/5090/timeline_nutanix_original_600-800.csv` with only
the requests originally fired in [600, 800), with every `timestamp_s`
shifted down by 600 so the first window second is t=0. A request that
fired at t=605.0 in the original lands at t=5.0 in the slice — the
pre-roll gap is preserved, not collapsed.

Why rebase to 0? `auto_benchmark.py` anchors its replay to the first
row's timestamp (`base_ts = timeline_rows[0].timestamp_s`), and the SLO
plotting anchors wall-clock t=0 to the recording start. Emitting a slice
that already starts near 0 keeps those offsets honest and makes the
sliced run directly comparable to a full run.

Interval convention
--------------------
The window is half-open `[START, END)` by default — START is included,
END is excluded — so tiling slices ([0,200), [200,400), …) partition
the timeline with no double-counted request. Pass `--inclusive-end` to
include a request landing exactly on END.

Resolving the input
-------------------
Either name a stock timeline with `--timeline {nutanix,loose,tight,live}`
(+ `--gpu`, auto-detected like auto_benchmark), or point straight at a
file with `--input PATH`. `--output PATH` overrides the derived name.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


# ----------------------------------------------------------------------
# GPU autodetect — mirrors auto_benchmark.detect_gpu_subdir so a bare
# `--timeline nutanix` resolves to the same timelines/<gpu>/ subdir the
# benchmark would pick up.
# ----------------------------------------------------------------------
def detect_gpu_subdir() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True, timeout=2.0,
        )
        name = (out.strip().splitlines() or [""])[0].upper()
        if "A100" in name:
            return "A100"
        if "5090" in name:
            return "5090"
    except Exception:
        pass
    return "5090"


def load_timeline(path: Path):
    """Return rows as dicts with the typed schema preserved. `second` /
    `index_in_second` are recomputed by the slicer, so we don't type
    them here. Rows come back sorted by `timestamp_s`."""
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"timestamp_s", "prompt_length", "max_new_tokens"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"[timeline_slice] {path}: missing columns {sorted(missing)}")
        for r in reader:
            rows.append({
                "timestamp_s": float(r["timestamp_s"]),
                "prompt_length": int(float(r["prompt_length"])),
                "max_new_tokens": int(float(r["max_new_tokens"])),
            })
    rows.sort(key=lambda x: x["timestamp_s"])
    return rows


def slice_timeline(rows, start: float, end: float,
                   inclusive_end: bool = False, rebase_to: float = 0.0):
    """Keep rows in the [start, end) window (closed at end if
    `inclusive_end`) and shift `timestamp_s` so `start` maps to
    `rebase_to` (default 0.0). The pre-roll gap between `start` and the
    first kept request is preserved, since we rebase by the window edge,
    not by the first row."""
    if end <= start:
        sys.exit(f"[timeline_slice] --end ({end}) must be greater than "
                 f"--start ({start})")

    def _in_window(t: float) -> bool:
        if t < start:
            return False
        return t <= end if inclusive_end else t < end

    shift = start - rebase_to
    sliced = [
        {
            "timestamp_s": round(r["timestamp_s"] - shift, 3),
            "prompt_length": r["prompt_length"],
            "max_new_tokens": r["max_new_tokens"],
        }
        for r in rows
        if _in_window(r["timestamp_s"])
    ]
    sliced.sort(key=lambda x: x["timestamp_s"])
    _recompute_buckets(sliced)
    return sliced


def _recompute_buckets(rows):
    """In-place: refresh `second` and `index_in_second` from each row's
    (already rebased) `timestamp_s`. Rows must be sorted by
    `timestamp_s`."""
    sec_counter = {}
    for r in rows:
        sec = int(r["timestamp_s"])
        r["second"] = sec
        r["index_in_second"] = sec_counter.get(sec, 0)
        sec_counter[sec] = r["index_in_second"] + 1


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["timestamp_s", "prompt_length", "max_new_tokens",
                  "second", "index_in_second"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})


def _stats_blurb(rows) -> str:
    if not rows:
        return "0 requests"
    n = len(rows)
    span = max(r["timestamp_s"] for r in rows) - min(r["timestamp_s"] for r in rows)
    span = max(span, 1e-9)
    return f"N={n} | span={span:.1f}s | mean RPS={n / span:.2f}"


def _win_tag(start: float, end: float) -> str:
    """e.g. 600,800 → '600-800'; 47.5,200.25 → '47p5-200p25'."""
    def _fmt(x: float) -> str:
        return (f"{x:g}").replace(".", "p")
    return f"{_fmt(start)}-{_fmt(end)}"


def _resolve_input(args) -> Path:
    if args.input:
        return Path(args.input).expanduser().resolve()
    gpu = args.gpu or detect_gpu_subdir()
    return _HERE / gpu / f"timeline_{args.timeline}.csv"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Slice a [start, end) window out of a request timeline "
                    "and rebase it so the original start becomes t=0.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--timeline", default="nutanix_original",
                     help="Stock timeline name under timelines/<gpu>/ "
                          "(nutanix_original, nutanix, loose, tight, live). "
                          "Default: nutanix_original (the full untrimmed trace).")
    src.add_argument("--input", default=None,
                     help="Direct path to a timeline CSV (overrides "
                          "--timeline/--gpu).")
    ap.add_argument("--gpu", default=None, choices=["5090", "A100"],
                    help="timelines/<gpu>/ subdir for --timeline. "
                         "Default: auto-detected.")
    ap.add_argument("--start", type=float, required=True,
                    help="Window start (seconds, inclusive) in the "
                         "ORIGINAL timeline.")
    ap.add_argument("--end", type=float, required=True,
                    help="Window end (seconds, exclusive) in the "
                         "ORIGINAL timeline.")
    ap.add_argument("--inclusive-end", action="store_true",
                    help="Include a request landing exactly on --end "
                         "(default: half-open [start, end)).")
    ap.add_argument("--rebase-to", type=float, default=0.0,
                    help="New timestamp that --start maps to. Default 0.0 "
                         "(original start → t=0).")
    ap.add_argument("--output", default=None,
                    help="Output CSV path. Default: alongside the input, "
                         "named <stem>_<start>-<end>.csv.")
    args = ap.parse_args()

    in_csv = _resolve_input(args)
    if not in_csv.exists():
        sys.exit(f"[timeline_slice] input not found: {in_csv}")

    if args.output:
        out_csv = Path(args.output).expanduser().resolve()
    else:
        tag = _win_tag(args.start, args.end)
        out_csv = in_csv.with_name(f"{in_csv.stem}_{tag}{in_csv.suffix}")

    orig = load_timeline(in_csv)
    print(f"[timeline_slice] loaded {len(orig)} rows from {in_csv}")

    sliced = slice_timeline(orig, args.start, args.end,
                            inclusive_end=args.inclusive_end,
                            rebase_to=args.rebase_to)
    if not sliced:
        edge = "]" if args.inclusive_end else ")"
        print(f"[timeline_slice] WARNING: window [{args.start}, {args.end}{edge} "
              f"is empty — no rows written.")

    write_csv(sliced, out_csv)
    edge = "]" if args.inclusive_end else ")"
    print(f"[timeline_slice] window [{args.start}, {args.end}{edge}  "
          f"→ rebased so {args.start}s = t={args.rebase_to:g}s")
    print(f"[timeline_slice]   original: {_stats_blurb(orig)}")
    print(f"[timeline_slice]   slice   : {_stats_blurb(sliced)}")
    print(f"[timeline_slice] wrote sliced timeline → {out_csv}")


if __name__ == "__main__":
    main()
