#!/usr/bin/env python3
"""repair_bwd_log.py — add the missing header to a bwd_log written by a server
that started before the empty-file header fix, then apply the timeline-window
trim auto_benchmark_tp.py would have applied (cutoff = t_first_wall − base_ts).

    python eval-tp/repair_bwd_log.py eval-tp/output/bwd_log_<stem>.csv
"""
import csv
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from auto_benchmark import load_timeline_csv, trim_bwd_log_before  # noqa: E402

HEADER = ["timestamp", "epoch", "batch_idx", "batch_tokens", "batch_loss",
          "total_processed_tokens"]


def repair(path: Path) -> None:
    lines = path.read_text().splitlines()
    if lines and lines[0].startswith("timestamp"):
        print(f"[repair] {path.name}: header present")
    else:
        path.write_text(",".join(HEADER) + "\n" + "\n".join(lines) + ("\n" if lines else ""))
        print(f"[repair] {path.name}: header added ({len(lines)} rows)")
    meta = path.with_name(path.name.replace("bwd_log", "bench_meta")).with_suffix(".json")
    if not meta.exists():
        print(f"[repair] no {meta.name}; not trimming")
        return
    m = json.loads(meta.read_text())
    rows = load_timeline_csv(m["timeline_csv"])
    base_ts = rows[0].timestamp_s if rows else 0.0
    cutoff = datetime.datetime.fromisoformat(m["t_first_wall_iso"]) - datetime.timedelta(seconds=base_ts)
    trim_bwd_log_before(str(path), cutoff)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        repair(Path(arg))
