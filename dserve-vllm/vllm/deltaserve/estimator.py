# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Three-regime composition-based execution-time estimator (Phase 4).

vLLM V1 runs ONE flattened batch per step that mixes (chunked) prefill and
decode. The full step-time formula is:

    T_step ≈ α·S + β·T_in + γ·T_ft + δ·B_d + ε·K + c

where, for the prefill part of the step (which INCLUDES injected FT tokens —
subset convention: ``T_ft ⊆ T_in``):
  S    = Σ nᵢ²       prefill self-attention work (proxy T_in²/P when only the
                     total + sample count are known; exact if per-request token
                     counts are passed)
  T_in = Σ nᵢ        total prefill tokens (FT tokens are a subset)
  T_ft               FT-prefill tokens; γ is the extra activation-save overhead
                     paid by FT rows on top of β·T_in
and for the decode part:
  B_d                number of decode requests
  K                  total decode KV tokens

**Three coefficient sets**, partitioned by step COMPOSITION (which maps 1:1
to vLLM's CUDA-graph runtime mode under the default
``cudagraph_mode=FULL_AND_PIECEWISE``):

  * ``INF_PREFILL`` — ``T_ft == 0 AND T_in > 0``, runtime mode PIECEWISE.
    Design matrix [S, T_in, B_d, K, 1] (5 cols; γ ≡ 0).
  * ``EAGER``       — ``T_ft > 0``, runtime mode NONE (FT forces eager).
    Design matrix [S, T_in, T_ft, B_d, K, 1] (full 6 cols).
  * ``DECODE_ONLY`` — ``T_in == 0 AND T_ft == 0 AND B_d > 0``, runtime mode FULL.
    Design matrix [B_d, K, 1] (3 cols; prefill terms ≡ 0).

Selection is composition-derived: `predict()` picks the right regime from the
input features. The optional `regime=` override lets callers force a specific
regime (e.g. the iterative admission loop forces EAGER for per-sample
predictions because adding any FT forces the step eager).

There is NO closed-form admission solve; admission is iterative — the FT
scheduler tries adding samples one-by-one, calls `predict(..., regime=EAGER)`
on each hypothetical, and stops when the SLO would be violated. See
``SLO_ESTIMATOR_REDESIGN.md`` §2 for the full algorithm.

Ports + extends DeltaServe's ``router/tracker.py``
(PrefillExecutionEstimator + DecodeExecutionEstimator + BatchExecutionTracker),
collapsed into a unified three-regime model that natively handles vLLM's
continuous-batching mixed steps.
"""

from __future__ import annotations

import csv
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from vllm.deltaserve import dprint

# Minimum recorded steps per regime before a (re)fit is attempted. The largest
# regime has 6 free parameters; require a small margin for a stable lstsq.
MIN_FIT_SAMPLES = 8
# Live refit cadence (in recorded steps).
REFIT_EVERY = 256

# Regime keys — composition-derived, map 1:1 to vLLM runtime CUDA-graph mode.
REGIME_INF_PREFILL = "inf_prefill"
REGIME_EAGER = "eager"
REGIME_DECODE_ONLY = "decode_only"
REGIMES = (REGIME_INF_PREFILL, REGIME_EAGER, REGIME_DECODE_ONLY)


@dataclass
class StepFeatures:
    """Batch-composition features for one scheduler step.

    Subset convention: ``t_ft`` is a subset of ``t_in`` (FT prefill tokens
    are counted in both). ``β`` (T_in coefficient) is per-prefill-token cost
    paid by any prefill row; ``γ`` (T_ft coefficient) is the extra cost paid
    specifically by FT rows from the activation-capture hooks. Total cost per
    FT token = ``β + γ``.
    """

    t_in: float = 0.0          # total prefill tokens (FT subset included)
    p: int = 0                 # number of prefill samples
    t_ft: float = 0.0          # FT prefill tokens (subset of t_in)
    b_d: int = 0               # number of decode requests
    k: float = 0.0             # total decode KV tokens
    # Exact per-request prefill token counts (inference + FT). When present, S
    # is computed exactly; otherwise the T_in²/P proxy is used.
    prefill_lens: Sequence[int] | None = None

    @property
    def s(self) -> float:
        """Σ nᵢ² — prefill attention work."""
        if self.prefill_lens:
            return float(sum(int(n) * int(n) for n in self.prefill_lens))
        if self.p > 0:
            return float(self.t_in) * float(self.t_in) / float(self.p)
        return 0.0

    @property
    def has_ft(self) -> bool:
        return self.t_ft > 0

    def row(self) -> list[float]:
        """Full design-matrix row [S, T_in, T_ft, B_d, K, 1]. Per-regime
        ``data_fit`` uses subsets of this layout via ``_row_for_regime``."""
        return [self.s, float(self.t_in), float(self.t_ft),
                float(self.b_d), float(self.k), 1.0]

    def regime(self) -> str:
        """Composition-derived regime classifier. Mutually exclusive."""
        if self.t_ft > 0:
            return REGIME_EAGER
        elif self.t_in > 0:
            return REGIME_INF_PREFILL
        else:
            # decode-only OR pure-idle. Pure-idle (b_d == 0) won't be
            # predicted in practice (no inference cost to compute), but it
            # routes here for completeness.
            return REGIME_DECODE_ONLY


@dataclass
class StepParams:
    """Fitted coefficients for one regime. Unused columns are 0.0 (not None)
    so ``eval()`` is correct for any regime — the per-regime fit zeros out
    coefficients for columns excluded from its reduced design matrix.

    The ``is_fitted`` flag tracks whether a fit has actually run; cold-start
    (no fit yet) leaves all coefficients None to distinguish from a genuine
    zero fit."""

    alpha: float | None = None   # S
    beta: float | None = None    # T_in
    gamma: float | None = None   # T_ft
    delta: float | None = None   # B_d
    epsilon: float | None = None # K
    c: float | None = None       # constant

    @property
    def is_fitted(self) -> bool:
        # All non-None ⇒ a fit ran. (Unused-column coefficients are filled
        # with 0.0 at fit time; only None means cold-start.)
        return all(v is not None for v in
                   (self.alpha, self.beta, self.gamma,
                    self.delta, self.epsilon, self.c))

    def eval(self, f: StepFeatures) -> float:
        return (self.alpha * f.s + self.beta * f.t_in + self.gamma * f.t_ft
                + self.delta * f.b_d + self.epsilon * f.k + self.c)


class StepExecutionTracker:
    """Rolling record of every served step: features, measured + predicted
    duration, and the regime (was_graph) that was in effect at dispatch.

    ``was_graph`` is stored verbatim for observability (sanity-check that
    composition regimes correlate with actual runtime mode) but is NOT used
    by the estimator's regime selection (which is composition-derived).
    Ports ``BatchExecutionTracker``.
    """

    def __init__(self, max_steps: int = 10240) -> None:
        self.max_steps = max_steps
        self.features: list[StepFeatures] = []
        self.durations: list[float] = []
        self.predicted: list[float | None] = []
        self.was_graph: list[bool | None] = []
        self.timestamps: list[float] = []
        self._last_refit_size = 0

    def add(self, features: StepFeatures, duration: float,
            predicted: float | None = None,
            was_graph: bool | None = None) -> None:
        self.features.append(features)
        self.durations.append(float(duration))
        self.predicted.append(predicted)
        self.was_graph.append(was_graph)
        self.timestamps.append(time.time())
        if len(self.durations) > self.max_steps:
            self._drop(0)

    def _drop(self, i: int) -> None:
        del self.features[i]
        del self.durations[i]
        del self.predicted[i]
        del self.was_graph[i]
        del self.timestamps[i]

    def size(self) -> int:
        return len(self.durations)

    def check_refit(self) -> bool:
        """True once every REFIT_EVERY new recorded steps."""
        n = self.size()
        if n > 0 and n % REFIT_EVERY == 0 and n > self._last_refit_size:
            self._last_refit_size = n
            return True
        return False

    def write_prediction_stats_csv(self, csv_path: str | None) -> None:
        """Dump predicted-vs-actual per step for offline analysis. Includes
        both the composition-derived regime label AND the recorded
        ``was_graph`` flag for cross-checking."""
        if csv_path is None:
            dprint("[estimator] batch_prediction_stats_path is null — "
                   "skipping prediction-stats CSV dump.")
            return
        parent = os.path.dirname(csv_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        rows = []
        for i in range(self.size()):
            pred = self.predicted[i]
            if pred is None:
                continue
            f = self.features[i]
            rows.append({
                "timestamp": self.timestamps[i],
                "step_index": i,
                "regime": f.regime(),
                "was_graph": self.was_graph[i],
                "t_in": f.t_in, "p": f.p, "t_ft": f.t_ft,
                "b_d": f.b_d, "k": f.k, "s": f.s,
                "execution_duration": self.durations[i],
                "predicted_duration": pred,
            })
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "timestamp", "step_index", "regime", "was_graph",
                "t_in", "p", "t_ft", "b_d", "k", "s",
                "execution_duration", "predicted_duration"])
            writer.writeheader()
            writer.writerows(rows)
        dprint(f"[estimator] wrote {len(rows)} prediction rows → {csv_path}")


class MergedExecutionEstimator:
    """Three-regime composition-based step-time model.

    Each regime fits independently on its own subset of recorded steps using a
    reduced design matrix tailored to its non-zero feature columns. The result
    is per-regime per-coefficient `StepParams` (with zeros for unused
    columns), selected at predict time by step composition.

    Admission is iterative, not closed-form — see the FT scheduler's
    ``admit_ft_to_step`` for the per-sample loop that uses ``predict(...,
    regime="eager")`` for hypothetical-with-FT predictions."""

    def __init__(self) -> None:
        self._params: dict[str, StepParams] = {r: StepParams() for r in REGIMES}
        self._rmse: dict[str, float | None] = {r: None for r in REGIMES}
        self._warned_unfitted = False

    @property
    def is_ready(self) -> bool:
        return any(p.is_fitted for p in self._params.values())

    @property
    def fit_rmse(self) -> float | None:
        """Headline RMSE — pick whichever regime is fitted (EAGER preferred
        since it's the most-exercised regime under steady-state co-serving)."""
        for r in (REGIME_EAGER, REGIME_INF_PREFILL, REGIME_DECODE_ONLY):
            if self._rmse[r] is not None:
                return self._rmse[r]
        return None

    # ─── Legacy properties (kept for backward-compat with existing log lines)
    @property
    def eager_rmse(self) -> float | None:
        return self._rmse[REGIME_EAGER]

    @property
    def graph_rmse(self) -> float | None:
        # Best legacy mapping: the original "graph" regime contained the data
        # the new DECODE_ONLY regime captures most cleanly.
        return self._rmse[REGIME_DECODE_ONLY]

    # ---------------- prediction ----------------
    def _select(self, features: StepFeatures,
                regime: str | None = None) -> tuple[str, StepParams]:
        """Pick a fitted ``StepParams`` for prediction. If ``regime`` is given,
        use that regime when fitted; otherwise derive from composition. Cold-
        start fallback: any fitted regime in priority order."""
        if regime is None:
            regime = features.regime()
        p = self._params[regime]
        if p.is_fitted:
            return regime, p
        # Cold-start fallback — try fitted regimes in priority order.
        for r in (REGIME_EAGER, REGIME_INF_PREFILL, REGIME_DECODE_ONLY):
            if self._params[r].is_fitted:
                return r, self._params[r]
        return regime, p  # still unfitted; predict returns 0

    def predict(self, features: StepFeatures,
                regime: str | None = None) -> float:
        """Predict step time. ``regime`` overrides composition-derived
        selection — used by the iterative admission loop, which forces
        ``regime="eager"`` for hypothetical-with-FT predictions (because
        adding FT always forces eager)."""
        if not self.is_ready:
            if not self._warned_unfitted:
                dprint("[estimator] predict() called before any fit; "
                       "returning 0.0")
                self._warned_unfitted = True
            return 0.0
        selected_regime, p = self._select(features, regime=regime)
        pred = p.eval(features)
        rmse = self._rmse[selected_regime]
        if rmse:
            pred *= 1.0 + 1.5 * rmse   # pessimistic safety margin (DeltaServe)
        return max(0.0, float(pred))

    # ---------------- fitting ----------------
    def data_fit(self,
                 tracker: StepExecutionTracker) -> dict[str, StepParams]:
        """Refit all three regimes from the tracker, partitioned by the
        composition predicate. Each regime uses a reduced design matrix:
        unused columns are excluded from the lstsq solve, and the
        corresponding ``StepParams`` coefficients are filled with 0.0 so
        ``StepParams.eval`` works uniformly."""
        buckets: dict[str, tuple[list[list[float]], list[float]]] = {
            r: ([], []) for r in REGIMES
        }
        for f, dur in zip(tracker.features, tracker.durations):
            r = f.regime()
            X, y = buckets[r]
            X.append(self._row_for_regime(f, r))
            y.append(dur)

        msg_parts: list[str] = []
        for r in REGIMES:
            X, y = buckets[r]
            self._params[r], self._rmse[r] = self._fit_regime(
                X, y, self._params[r], r)
            rmse_str = (f"{self._rmse[r]:.4f}" if self._rmse[r] is not None
                        else "n/a")
            msg_parts.append(f"{r}={len(y)}(rmse={rmse_str})")
        dprint(f"[estimator] fit: " + ", ".join(msg_parts))
        return self._params

    @staticmethod
    def _row_for_regime(f: StepFeatures, regime: str) -> list[float]:
        """Per-regime reduced design-matrix row. Each regime's row contains
        only the columns whose corresponding coefficient can be identified
        from its data (others are ≡ 0 within the regime by construction)."""
        if regime == REGIME_INF_PREFILL:
            # T_ft ≡ 0 within this regime → γ column dropped.
            return [f.s, f.t_in, f.b_d, f.k, 1.0]            # 5 cols
        elif regime == REGIME_EAGER:
            # Full 6-column formula.
            return [f.s, f.t_in, f.t_ft, f.b_d, f.k, 1.0]    # 6 cols
        else:  # REGIME_DECODE_ONLY
            # T_in ≡ T_ft ≡ 0 within this regime → α, β, γ columns dropped.
            return [f.b_d, f.k, 1.0]                          # 3 cols

    @staticmethod
    def _scatter_coefs(coef: np.ndarray, regime: str) -> StepParams:
        """Scatter a regime's fitted coefficient vector back into the
        canonical ``StepParams`` layout. Columns not in the regime's design
        matrix are zeroed (the formula's contribution from those columns is
        identically zero within the regime by partition definition)."""
        if regime == REGIME_INF_PREFILL:
            # Columns: [S, T_in, B_d, K, 1] → α, β, δ, ε, c. γ ≡ 0.
            alpha, beta, delta, epsilon, c = (float(v) for v in coef)
            gamma = 0.0
        elif regime == REGIME_EAGER:
            # Columns: full [S, T_in, T_ft, B_d, K, 1].
            alpha, beta, gamma, delta, epsilon, c = (float(v) for v in coef)
        else:  # REGIME_DECODE_ONLY
            # Columns: [B_d, K, 1] → δ, ε, c. α, β, γ ≡ 0.
            delta, epsilon, c = (float(v) for v in coef)
            alpha = beta = gamma = 0.0
        return StepParams(alpha=alpha, beta=beta, gamma=gamma,
                          delta=delta, epsilon=epsilon, c=c)

    @classmethod
    def _fit_regime(cls, X: list[list[float]], y: list[float],
                    prev: StepParams,
                    regime: str) -> tuple[StepParams, float | None]:
        if len(y) < MIN_FIT_SAMPLES:
            if len(y) > 0:
                dprint(f"[estimator] {regime} regime has {len(y)} samples "
                       f"(<{MIN_FIT_SAMPLES}); keeping previous params")
            return prev, None
        A = np.asarray(X, dtype=float)
        b = np.asarray(y, dtype=float)
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        params = cls._scatter_coefs(coef, regime)
        preds = A @ coef
        rmse = float(np.sqrt(np.mean((preds - b) ** 2)))
        return params, rmse
