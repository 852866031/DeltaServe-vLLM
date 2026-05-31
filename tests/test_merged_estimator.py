#!/usr/bin/env python
"""Test the three-regime composition-based execution-time estimator
(vllm.deltaserve.estimator).

CPU-only, no model: synthesizes step durations from known per-regime
coefficients, fits, and asserts coefficient recovery + per-regime prediction
quality + selection routing. No GPU.

    python tests/test_merged_estimator.py

Run as a file (not `python -c`) so `vllm` resolves to the installed package.
"""

import sys

import numpy as np

from vllm.deltaserve.estimator import (
    REGIME_DECODE_ONLY,
    REGIME_EAGER,
    REGIME_INF_PREFILL,
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


# Per-regime ground-truth coefficients. Each regime has its own set; the
# redesign drops the single-pooled-fit model entirely.
TRUE_INF_PREFILL = dict(alpha=1.0e-6, beta=2.0e-4, delta=8.0e-4, epsilon=2.0e-5,
                        c=0.015)
TRUE_EAGER = dict(alpha=1.5e-6, beta=2.5e-4, gamma=5.0e-4, delta=1.0e-3,
                  epsilon=3.0e-5, c=0.025)
TRUE_DECODE_ONLY = dict(delta=6.0e-4, epsilon=1.5e-5, c=0.005)


def true_time_inf_prefill(f: StepFeatures) -> float:
    p = TRUE_INF_PREFILL
    return (p["alpha"] * f.s + p["beta"] * f.t_in + p["delta"] * f.b_d
            + p["epsilon"] * f.k + p["c"])


def true_time_eager(f: StepFeatures) -> float:
    p = TRUE_EAGER
    return (p["alpha"] * f.s + p["beta"] * f.t_in + p["gamma"] * f.t_ft
            + p["delta"] * f.b_d + p["epsilon"] * f.k + p["c"])


def true_time_decode_only(f: StepFeatures) -> float:
    p = TRUE_DECODE_ONLY
    return p["delta"] * f.b_d + p["epsilon"] * f.k + p["c"]


def true_time_for(f: StepFeatures) -> float:
    """Route each feature to the right truth function based on its
    composition-derived regime."""
    r = f.regime()
    if r == REGIME_EAGER:
        return true_time_eager(f)
    elif r == REGIME_INF_PREFILL:
        return true_time_inf_prefill(f)
    else:
        return true_time_decode_only(f)


# ─── Feature synthesizers, one per regime ────────────────────────────────

def make_inf_prefill_features(rng) -> StepFeatures:
    """Prefill-carrying step with no FT. Mix of pure-prefill and
    prefill+decode shapes for α/β/δ/ε identification."""
    kind = rng.integers(0, 2)
    p = int(rng.integers(1, 8))
    lens = [int(rng.integers(16, 512)) for _ in range(p)]
    if kind == 0:  # pure prefill
        return StepFeatures(t_in=sum(lens), p=p, prefill_lens=lens)
    # prefill + decode (mixed)
    b_d = int(rng.integers(1, 48))
    k = float(b_d * rng.integers(8, 1024))
    return StepFeatures(t_in=sum(lens), p=p, prefill_lens=lens,
                        b_d=b_d, k=k)


def make_eager_features(rng) -> StepFeatures:
    """Any-composition step WITH FT. Mix of FT-only, FT+inf-prefill,
    FT+decode, FT+mixed for the full 6-coef identification."""
    p_inf = int(rng.integers(0, 4))
    inf_lens = [int(rng.integers(16, 256)) for _ in range(p_inf)]
    t_ft = float(rng.integers(8, 256))
    b_d = int(rng.integers(0, 32))
    k = float(b_d * rng.integers(8, 1024)) if b_d > 0 else 0.0
    # Subset convention: T_in includes FT tokens; prefill_lens includes both.
    all_lens = inf_lens + [int(t_ft)]
    return StepFeatures(t_in=sum(inf_lens) + t_ft, p=p_inf + 1,
                        prefill_lens=all_lens, t_ft=t_ft, b_d=b_d, k=k)


def make_decode_only_features(rng) -> StepFeatures:
    """Decode-only step (T_in=T_ft=0, B_d>0). δ/ε/c identification."""
    b_d = int(rng.integers(1, 64))
    k = float(b_d * rng.integers(8, 2048))
    return StepFeatures(b_d=b_d, k=k)


# ─── Tests ───────────────────────────────────────────────────────────────

def test_recovery():
    print("test: per-regime coefficient recovery")
    rng = np.random.default_rng(0)
    tracker = StepExecutionTracker()
    est = MergedExecutionEstimator()
    # Synthesize a balanced mix: 200 per regime.
    for _ in range(200):
        f = make_inf_prefill_features(rng)
        tracker.add(f, true_time_inf_prefill(f), was_graph=True)
    for _ in range(200):
        f = make_eager_features(rng)
        tracker.add(f, true_time_eager(f), was_graph=False)
    for _ in range(200):
        f = make_decode_only_features(rng)
        tracker.add(f, true_time_decode_only(f), was_graph=True)
    est.data_fit(tracker)

    check("inf_prefill regime fitted",
          est._params[REGIME_INF_PREFILL].is_fitted)
    check("eager regime fitted", est._params[REGIME_EAGER].is_fitted)
    check("decode_only regime fitted",
          est._params[REGIME_DECODE_ONLY].is_fitted)
    check("estimator ready", est.is_ready)

    # INF_PREFILL: recover α, β, δ, ε, c (γ is zeroed by partition).
    inf_p = est._params[REGIME_INF_PREFILL]
    for nm, t, g in [
        ("inf_prefill α", TRUE_INF_PREFILL["alpha"], inf_p.alpha),
        ("inf_prefill β", TRUE_INF_PREFILL["beta"], inf_p.beta),
        ("inf_prefill δ", TRUE_INF_PREFILL["delta"], inf_p.delta),
        ("inf_prefill ε", TRUE_INF_PREFILL["epsilon"], inf_p.epsilon),
        ("inf_prefill c", TRUE_INF_PREFILL["c"], inf_p.c),
    ]:
        rel = abs(g - t) / (abs(t) + 1e-12)
        check(f"{nm}: fit {g:.3e} vs true {t:.3e} (rel {rel:.1%})", rel < 0.05)
    check("inf_prefill γ ≡ 0 (partition def)", inf_p.gamma == 0.0)

    # EAGER: recover all six coefficients.
    eg_p = est._params[REGIME_EAGER]
    for nm, t, g in [
        ("eager α", TRUE_EAGER["alpha"], eg_p.alpha),
        ("eager β", TRUE_EAGER["beta"], eg_p.beta),
        ("eager γ", TRUE_EAGER["gamma"], eg_p.gamma),
        ("eager δ", TRUE_EAGER["delta"], eg_p.delta),
        ("eager ε", TRUE_EAGER["epsilon"], eg_p.epsilon),
        ("eager c", TRUE_EAGER["c"], eg_p.c),
    ]:
        rel = abs(g - t) / (abs(t) + 1e-12)
        check(f"{nm}: fit {g:.3e} vs true {t:.3e} (rel {rel:.1%})", rel < 0.05)

    # DECODE_ONLY: recover δ, ε, c (α, β, γ are zeroed by partition).
    do_p = est._params[REGIME_DECODE_ONLY]
    for nm, t, g in [
        ("decode_only δ", TRUE_DECODE_ONLY["delta"], do_p.delta),
        ("decode_only ε", TRUE_DECODE_ONLY["epsilon"], do_p.epsilon),
        ("decode_only c", TRUE_DECODE_ONLY["c"], do_p.c),
    ]:
        rel = abs(g - t) / (abs(t) + 1e-12)
        check(f"{nm}: fit {g:.3e} vs true {t:.3e} (rel {rel:.1%})", rel < 0.05)
    check("decode_only α ≡ 0", do_p.alpha == 0.0)
    check("decode_only β ≡ 0", do_p.beta == 0.0)
    check("decode_only γ ≡ 0", do_p.gamma == 0.0)


def _populate(est, rng, n_per=200):
    tracker = StepExecutionTracker()
    for _ in range(n_per):
        f = make_inf_prefill_features(rng)
        tracker.add(f, true_time_inf_prefill(f), was_graph=True)
    for _ in range(n_per):
        f = make_eager_features(rng)
        tracker.add(f, true_time_eager(f), was_graph=False)
    for _ in range(n_per):
        f = make_decode_only_features(rng)
        tracker.add(f, true_time_decode_only(f), was_graph=True)
    est.data_fit(tracker)


def test_prediction():
    print("test: prediction matches per-regime truth")
    rng = np.random.default_rng(1)
    est = MergedExecutionEstimator()
    _populate(est, rng)

    # Composition-derived selection: each test feature routes to the right
    # regime automatically.
    f_inf = StepFeatures(t_in=512, p=2, prefill_lens=[256, 256], b_d=8,
                         k=4096)
    f_eager = StepFeatures(t_in=512 + 64, p=3,
                           prefill_lens=[256, 256, 64], t_ft=64.0, b_d=8,
                           k=4096)
    f_decode = StepFeatures(b_d=8, k=4096)

    for label, f, truth_fn in [
        ("inf_prefill", f_inf, true_time_inf_prefill),
        ("eager", f_eager, true_time_eager),
        ("decode_only", f_decode, true_time_decode_only),
    ]:
        pred = est.predict(f)
        truth = truth_fn(f)
        # RMSE-margin makes predictions slightly pessimistic; within 5% of
        # truth is the bound the previous test used.
        check(f"{label}: pred {pred:.4f} ≈ truth {truth:.4f}",
              abs(pred - truth) / max(truth, 1e-6) < 0.05)


def test_unfitted_safe():
    print("test: cold-start is safe")
    est = MergedExecutionEstimator()
    check("not ready before fit", not est.is_ready)
    check("predict returns 0.0 unfitted",
          est.predict(StepFeatures(t_in=100, p=1)) == 0.0)
    # max_next_ft_tokens has been deleted as part of the redesign — admission
    # is iterative in the scheduler. Verify the method is gone.
    check("max_next_ft_tokens removed",
          not hasattr(est, "max_next_ft_tokens"))


def test_selection_routing():
    print("test: composition-based regime selection")
    rng = np.random.default_rng(2)
    est = MergedExecutionEstimator()
    _populate(est, rng)

    # _select returns (regime_name, params). Verify routing.
    r, _ = est._select(StepFeatures(t_in=100, p=1, prefill_lens=[100]))
    check(f"INF_PREFILL routing: {r}", r == REGIME_INF_PREFILL)
    r, _ = est._select(StepFeatures(t_in=100, p=2, prefill_lens=[64, 36],
                                    t_ft=36.0))
    check(f"EAGER routing (FT > 0): {r}", r == REGIME_EAGER)
    r, _ = est._select(StepFeatures(b_d=4, k=512))
    check(f"DECODE_ONLY routing: {r}", r == REGIME_DECODE_ONLY)

    # Explicit regime override: admission loop passes regime="eager" for
    # hypothetical-with-FT predictions even when baseline features look
    # like decode-only.
    r, _ = est._select(StepFeatures(b_d=4, k=512), regime=REGIME_EAGER)
    check(f"explicit regime override: {r}", r == REGIME_EAGER)


def test_decode_only_regime_isolated_fit():
    """Synthesize only DECODE_ONLY data; verify the decode_only fit succeeds
    and the other regimes stay unfitted."""
    print("test: decode_only regime fits in isolation")
    rng = np.random.default_rng(3)
    tracker = StepExecutionTracker()
    est = MergedExecutionEstimator()
    for _ in range(50):
        f = make_decode_only_features(rng)
        tracker.add(f, true_time_decode_only(f), was_graph=True)
    est.data_fit(tracker)
    check("decode_only fitted",
          est._params[REGIME_DECODE_ONLY].is_fitted)
    check("inf_prefill unfitted (no data)",
          not est._params[REGIME_INF_PREFILL].is_fitted)
    check("eager unfitted (no data)",
          not est._params[REGIME_EAGER].is_fitted)


def test_eager_regime_with_ft_only_decode():
    """Synthesize the new EAGER shape (T_in=0, T_ft>0, B_d>0) — required by
    the unified-phase scheduler. Verify the fit converges and prediction is
    reasonable on this shape."""
    print("test: eager regime fits (T_in=0, T_ft>0, B_d>0) shape")
    rng = np.random.default_rng(4)
    tracker = StepExecutionTracker()
    est = MergedExecutionEstimator()
    # Mix: half regular eager (with inf prefill), half FT-on-decode only.
    for _ in range(150):
        f = make_eager_features(rng)
        tracker.add(f, true_time_eager(f), was_graph=False)
    for _ in range(150):
        # FT-on-decode-only: T_in = T_ft (subset convention), no inf prefill.
        t_ft = float(rng.integers(8, 128))
        b_d = int(rng.integers(1, 16))
        k = float(b_d * rng.integers(8, 1024))
        f = StepFeatures(t_in=t_ft, p=1, prefill_lens=[int(t_ft)],
                         t_ft=t_ft, b_d=b_d, k=k)
        tracker.add(f, true_time_eager(f), was_graph=False)
    est.data_fit(tracker)
    check("eager fitted on mixed shapes",
          est._params[REGIME_EAGER].is_fitted)
    # Predict on a held-out FT-on-decode-only shape.
    f_test = StepFeatures(t_in=64.0, p=1, prefill_lens=[64], t_ft=64.0,
                          b_d=4, k=2048)
    pred = est.predict(f_test)
    truth = true_time_eager(f_test)
    check(f"FT-on-decode-only pred {pred:.4f} ≈ truth {truth:.4f}",
          abs(pred - truth) / max(truth, 1e-6) < 0.10)


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
    test_selection_routing()
    test_decode_only_regime_isolated_fit()
    test_eager_regime_with_ft_only_decode()
    test_refit_cadence()
    print()
    total = _passed + _failed
    color = _GREEN if _failed == 0 else _RED
    print(f"{color}{_passed}/{total} checks passed{_RESET}")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
