"""Average TBT over a sliding window (deltaserve/tbt_window.py) and its switch.

    python tests/test_tbt_window.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "deltaserve-plugin"))

FAILED = []
N = [0]


def check(name, ok):
    N[0] += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        FAILED.append(name)


def close(a, b):
    return abs(a - b) < 1e-9


def test_average():
    print("The window's average")
    from deltaserve.tbt_window import TbtWindow

    w = TbtWindow(1.0)
    check("empty: no average, a hypothetical step is its own average",
          w.average(0.0) is None and close(w.average_with(0.0, 4, 0.03), 0.03))
    w.step_done(10.00, 4, True)
    check("the first step has no gap before it", w.average(10.0) is None)
    w.step_done(10.02, 4, True)      # 4 tokens, 20 ms after the previous
    w.step_done(10.06, 2, True)      # 2 tokens, 40 ms
    check("every token of the window, averaged (4×20 + 2×40) / 6",
          close(w.average(10.06), (4 * 0.02 + 2 * 0.04) / 6))
    check("a step being built is added to it",
          close(w.average_with(10.06, 6, 0.05),
                (4 * 0.02 + 2 * 0.04 + 6 * 0.05) / 12))
    check("a step without decode requests does not change it",
          close(w.average_with(10.06, 0, 0.5), (4 * 0.02 + 2 * 0.04) / 6))
    check("steps older than the window are dropped",
          close(w.average(11.03), 0.04) and w.average(11.07) is None)


def test_gaps():
    print("What counts as a token gap")
    from deltaserve.tbt_window import TbtWindow

    w = TbtWindow(1.0)
    w.step_done(0.00, 2, True)
    w.step_done(0.02, 2, True)
    w.step_done(0.05, 0, True)       # FT-only step, decoders waiting
    check("a step the decode requests waited through is carried: the step "
          "being built is charged with it",
          close(w.average_with(0.05, 2, 0.02), (2 * 0.02 + 2 * 0.05) / 4))
    w.step_done(0.07, 2, True)
    check("... and so is the next decode step (30 + 20 ms)",
          close(w.average(0.07), (2 * 0.02 + 2 * 0.05) / 4))
    w.step_done(0.10, 0, False)      # FT-only, nothing running
    w.step_done(0.13, 3, True)
    check("a step with no inference request running is not a token gap",
          close(w.average(0.13), (2 * 0.02 + 2 * 0.05 + 3 * 0.03) / 7))
    w.step_done(5.0, 3, True)
    check("after an idle engine the gap is dropped, not counted as 5 s",
          w.average(5.0) is None)
    w.step_done(5.02, 3, True)
    check("and counting resumes", close(w.average(5.02), 0.02))


def test_switch():
    print("The switch")
    from deltaserve.config import FinetuneConfig
    from deltaserve.config_loader import FINETUNE_SECTIONS

    check("off by default (avg_tbt_window_s = 0)",
          FinetuneConfig().avg_tbt_window_s == 0.0)
    check("read from the slo: section of a serving YAML",
          "avg_tbt_window_s" in FINETUNE_SECTIONS["slo"])
    a = FinetuneConfig(enable_finetuning=True)
    b = FinetuneConfig(enable_finetuning=True, avg_tbt_window_s=1.0)
    check("not part of the compile-cache hash",
          a.compute_hash() == b.compute_hash())


if __name__ == "__main__":
    test_average()
    test_gaps()
    test_switch()
    if FAILED:
        print(f"\n{len(FAILED)} of {N[0]} checks FAILED")
        sys.exit(1)
    print(f"\nall {N[0]} checks passed")
