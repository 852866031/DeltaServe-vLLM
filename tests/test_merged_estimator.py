#!/usr/bin/env python
"""Test the merged execution-time estimator (vllm.deltaserve.estimator).

CPU-only, no model: synthesizes step durations from known coefficients, fits,
and asserts coefficient recovery + admission-solver behaviour. No GPU.

    python tests/test_merged_estimator.py

Run as a file (not `python -c`) so `vllm` resolves to the installed package.
"""

import sys

import numpy as np

from vllm.deltaserve.estimator import (
    MergedExecutionEstimator,
    StepExecutionTracker,
    StepFeatures,
)

_passed = 0
_failed = 0
_TTY = sys.stdout.isatty()
_GREEN = "\033[92m" if _TTY else ""
_RED = "\033[91m" if _TTY else ""
_RESET = "\033[0m" if _TTY else ""


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  {_GREEN}[PASS]{_RESET} {name}")
    else:
        _failed += 1
        print(f"  {_RED}[FAIL]{_RESET} {name}")


# Ground-truth coefficients: (alpha, beta, gamma, delta, epsilon, c)
TRUE = (1.0e-6, 2.0e-4, 5.0e-4, 1.0e-3, 3.0e-5, 0.02)


def true_time(f: StepFeatures) -> float:
    a, b, g, d, e, c = TRUE
    return a * f.s + b * f.t_in + g * f.t_ft + d * f.b_d + e * f.k + c


def make_eager_features(rng) -> StepFeatures:
    """Diverse eager steps: pure prefill, pure decode, mixed, co-serve."""
    kind = rng.integers(0, 4)
    if kind == 0:  # pure prefill, varied decomposition (identifies alpha vs beta)
        p = int(rng.integers(1, 8))
        lens = [int(rng.integers(16, 512)) for _ in range(p)]
        return StepFeatures(t_in=sum(lens), p=p, prefill_lens=lens)
    if kind == 1:  # pure decode, varied B_d and K independently
        b_d = int(rng.integers(1, 64))
        k = float(b_d * rng.integers(8, 2048))
        return StepFeatures(b_d=b_d, k=k)
    if kind == 2:  # mixed prefill + decode
        p = int(rng.integers(1, 4))
        lens = [int(rng.integers(16, 256)) for _ in range(p)]
        b_d = int(rng.integers(1, 48))
        k = float(b_d * rng.integers(8, 1024))
        return StepFeatures(t_in=sum(lens), p=p, prefill_lens=lens, b_d=b_d, k=k)
    # co-serve: prefill incl. FT tokens (identifies gamma)
    p = int(rng.integers(1, 4))
    lens = [int(rng.integers(16, 256)) for _ in range(p)]
    t_ft = float(rng.integers(8, 256))
    return StepFeatures(t_in=sum(lens) + t_ft, p=p + 1,
                        prefill_lens=lens + [int(t_ft)], t_ft=t_ft,
                        b_d=int(rng.integers(0, 16)),
                        k=float(rng.integers(0, 4096)))


def test_recovery():
    print("test: coefficient recovery (eager)")
    rng = np.random.default_rng(0)
    tracker = StepExecutionTracker()
    est = MergedExecutionEstimator()
    for _ in range(400):
        f = make_eager_features(rng)
        tracker.add(f, true_time(f), was_graph=False)
    est.data_fit(tracker)
    check("eager regime fitted", est._eager.is_fitted)
    check("estimator ready", est.is_ready)
    got = (est._eager.alpha, est._eager.beta, est._eager.gamma,
           est._eager.delta, est._eager.epsilon, est._eager.c)
    for nm, t, g in zip("alpha beta gamma delta epsilon c".split(), TRUE, got):
        rel = abs(g - t) / (abs(t) + 1e-12)
        check(f"{nm}: fit {g:.3e} vs true {t:.3e} (rel {rel:.1%})", rel < 0.05)
    check("eager rmse tiny (noise-free fit)", (est.eager_rmse or 1) < 1e-6)


def test_prediction():
    print("test: prediction matches truth")
    rng = np.random.default_rng(1)
    tracker = StepExecutionTracker()
    est = MergedExecutionEstimator()
    for _ in range(400):
        f = make_eager_features(rng)
        tracker.add(f, true_time(f), was_graph=False)
    est.data_fit(tracker)
    # RMSE-margin makes predictions pessimistic (>= truth) but close.
    f = StepFeatures(t_in=512, p=2, prefill_lens=[256, 256], b_d=8, k=4096)
    pred = est.predict(f, will_use_graph=False)
    truth = true_time(f)
    check(f"pred {pred:.4f} ≈ truth {truth:.4f}", abs(pred - truth) / truth < 0.05)


def test_unfitted_safe():
    print("test: cold-start is safe")
    est = MergedExecutionEstimator()
    check("not ready before fit", not est.is_ready)
    check("predict returns 0.0 unfitted", est.predict(StepFeatures(t_in=100, p=1)) == 0.0)
    check("admission returns 0 unfitted", est.max_next_ft_tokens(1.0) == 0)


def test_admission_monotonic():
    print("test: admission solver monotonic + SLO-bounded")
    rng = np.random.default_rng(2)
    tracker = StepExecutionTracker()
    est = MergedExecutionEstimator()
    for _ in range(400):
        f = make_eager_features(rng)
        tracker.add(f, true_time(f), was_graph=False)
    est.data_fit(tracker)

    x_small = est.max_next_ft_tokens(0.01)
    x_big = est.max_next_ft_tokens(0.10)
    check("more budget admits more tokens", x_big > x_small)
    check("zero budget admits nothing", est.max_next_ft_tokens(0.0) == 0)
    check("negative budget admits nothing", est.max_next_ft_tokens(-1.0) == 0)

    # The admitted x must satisfy the quadratic it solved: ΔT(x) ≤ budget.
    budget = 0.05
    x = est.max_next_ft_tokens(budget)
    a, b, g = est._eager.alpha, est._eager.beta, est._eager.gamma
    dt_x = a * x * x + (b + g) * x
    dt_x1 = a * (x + 1) ** 2 + (b + g) * (x + 1)
    check(f"ΔT({x})={dt_x:.4f} ≤ budget {budget}", dt_x <= budget + 1e-9)
    check(f"ΔT({x + 1})={dt_x1:.4f} > budget (tight)", dt_x1 > budget - 1e-9)


def test_two_regimes():
    print("test: eager and graph fit independently; FT forces eager")
    rng = np.random.default_rng(3)
    tracker = StepExecutionTracker()
    est = MergedExecutionEstimator()
    # Graph steps are ~2x faster (different constant) and never carry FT.
    for _ in range(300):
        f = make_eager_features(rng)
        tracker.add(f, true_time(f), was_graph=False)
        if not f.has_ft:
            tracker.add(f, 0.5 * true_time(f), was_graph=True)
    est.data_fit(tracker)
    check("both regimes fitted", est._eager.is_fitted and est._graph.is_fitted)
    f = StepFeatures(t_in=512, p=2, prefill_lens=[256, 256])
    pe = est.predict(f, will_use_graph=False)
    pg = est.predict(f, will_use_graph=True)
    check(f"graph faster than eager ({pg:.4f} < {pe:.4f})", pg < pe)


def test_refit_cadence():
    print("test: refit cadence")
    tracker = StepExecutionTracker()
    fires = 0
    for i in range(1, 600):
        tracker.add(StepFeatures(t_in=10, p=1), 0.01, was_graph=False)
        if tracker.check_refit():
            fires += 1
    check("refit fires at 256 and 512", fires == 2)


def main():
    test_recovery()
    test_prediction()
    test_unfitted_safe()
    test_admission_monotonic()
    test_two_regimes()
    test_refit_cadence()
    print()
    total = _passed + _failed
    color = _GREEN if _failed == 0 else _RED
    print(f"{color}{_passed}/{total} checks passed{_RESET}")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
