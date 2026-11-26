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

    check("on by default: avg_tbt_slo is kept over a 1 s window, and the "
          "per-step bound is the 100 ms backstop",
          FinetuneConfig().avg_tbt_window_s == 1.0
          and FinetuneConfig().max_tbt_slo == 0.1)
    check("read from the slo: section of a serving YAML",
          "avg_tbt_window_s" in FINETUNE_SECTIONS["slo"])
    a = FinetuneConfig(enable_finetuning=True)
    b = FinetuneConfig(enable_finetuning=True, avg_tbt_window_s=1.0)
    check("not part of the compile-cache hash",
          a.compute_hash() == b.compute_hash())


def test_backward_gate():
    print("The backward gate (avg_tbt_pause_backward)")
    import types

    from deltaserve.config import FinetuneConfig
    from deltaserve.config_loader import FINETUNE_SECTIONS
    from deltaserve.estimator import StepFeatures
    from deltaserve.ft_scheduler import FinetuneScheduler
    from deltaserve.runner_common import FtRunnerMixin
    from deltaserve.tbt_window import TbtWindow

    cfg = FinetuneConfig()
    check("on by default, steering to 90 % of the target",
          cfg.avg_tbt_pause_backward is True and cfg.avg_tbt_headroom == 0.9
          and {"avg_tbt_pause_backward", "avg_tbt_headroom"}
          <= set(FINETUNE_SECTIONS["slo"]))

    # -- the scheduler's decision -------------------------------------------
    gate = FinetuneScheduler._backward_would_break_avg_tbt
    CLEAN, OVERLAP = 0.013, 0.024         # a decode step alone / with a backward

    def sched(avg_slo, window=True, on=True, ready=True):
        w = TbtWindow(1.0) if window else None
        est = types.SimpleNamespace(
            is_ready=ready,
            predict=lambda f: OVERLAP if f.bwd else CLEAN)
        return types.SimpleNamespace(
            _tbt_pause_backward=on, _tbt_window=w, _estimator=est,
            _avg_tbt_slo=avg_slo, _tbt_headroom=0.9,
            _tbt_gate_stats={"paused": 0, "overlapped": 0})

    def decode(bwd=True, b_d=8, t_in=0, t_ft=0):
        return StepFeatures(t_in=t_in, t_ft=t_ft, b_d=b_d, bwd=bwd)

    import time as _t
    s = sched(0.02)
    check("target 20 ms: a 24 ms overlapped step would break it -> pause",
          gate(s, decode()) and s._tbt_gate_stats["paused"] == 1)
    s = sched(0.05)
    check("target 50 ms: the same step fits -> the backward keeps running",
          not gate(s, decode()) and s._tbt_gate_stats["overlapped"] == 1)
    now = _t.monotonic()
    s2 = sched(0.02)
    for i in range(12):                   # a stretch of clean steps: avg 13 ms
        s2._tbt_window.step_done(now - 0.2 + CLEAN * i, 8, True)
    check("target 20 ms with a window of clean steps: the slack lets one "
          "overlapped step through",
          not gate(s2, decode()))
    cfg2 = FinetuneConfig()
    check("the admission headroom is off by default",
          cfg2.avg_tbt_admission_headroom == 1.0
          and "avg_tbt_admission_headroom" in FINETUNE_SECTIONS["slo"])
    check("no backward in flight: nothing to decide",
          not gate(sched(0.02), decode(bwd=False)))
    check("a step with an inference prefill or FT tokens is not its business "
          "(they pause the child anyway)",
          not gate(sched(0.02), decode(t_in=40))
          and not gate(sched(0.02), decode(t_in=40, t_ft=40)))
    check("no decode request: nothing to protect",
          not gate(sched(0.02), decode(b_d=0)))
    check("switched off / no window / estimator not fitted: never",
          not gate(sched(0.02, on=False), decode())
          and not gate(sched(0.02, window=False), decode())
          and not gate(sched(0.02, ready=False), decode()))

    # -- the runner: a held pause, released by the first step that lets go ---
    class R(FtRunnerMixin):
        pass

    r = R()
    r._ds_init()
    r.vllm_config = types.SimpleNamespace(
        finetune_config=FinetuneConfig(enable_finetuning=True))
    log = []
    co = types.SimpleNamespace(
        pending_backward=True, multi_batch=False, fwd_throttle_active=False,
        relay_mode=False,
        gpu_pause_backward=lambda: log.append("pause"),
        gpu_resume_backward=lambda: log.append("resume"))

    def out(pause):
        o = types.SimpleNamespace(
            _ft_step_features=types.SimpleNamespace(t_in=0, t_ft=0, b_d=8))
        if pause:
            o.finetune_pause_backward = True
        return o

    check("a plain decode step leaves the backward running",
          r._ft_pause_decision(out(False), co) == (False, False) and log == [])
    check("a step carrying the gate's mark pauses it, held like the throttle",
          r._ft_pause_decision(out(True), co) == (True, True)
          and log == ["pause"] and r._throttle_held
          and co.fwd_throttle_active)
    r._ft_before_step(out(True), co)
    r._ft_pause_decision(out(True), co)
    check("consecutive marked steps: no resume in between",
          "resume" not in log)
    r._ft_before_step(out(False), co)
    check("the first step without the mark releases it",
          log[-1] == "resume" and not r._throttle_held
          and not co.fwd_throttle_active)
    log.clear()
    co.pending_backward = False
    check("no backward in flight: the mark is inert",
          r._ft_pause_decision(out(True), co) == (False, False) and log == [])


if __name__ == "__main__":
    test_average()
    test_gaps()
    test_switch()
    test_backward_gate()
    if FAILED:
        print(f"\n{len(FAILED)} of {N[0]} checks FAILED")
        sys.exit(1)
    print(f"\nall {N[0]} checks passed")
