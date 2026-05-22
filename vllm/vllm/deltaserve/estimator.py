# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Merged execution-time estimator for SLO-aware FT admission (Phase 4).

vLLM V1 runs ONE flattened batch per step that mixes (chunked) prefill and
decode. So a single linear model predicts a step's execution time:

    T_step ≈ α·S + β·T_in + γ·T_ft + δ·B_d + ε·K + c

where, for the prefill part of the step (which INCLUDES injected FT tokens):
  S    = Σ nᵢ²       prefill self-attention work (proxy T_in²/P when only the
                     total + sample count are known; exact if per-request token
                     counts are passed)
  T_in = Σ nᵢ        total prefill tokens (FT tokens are a subset)
  T_ft               FT tokens; γ·T_ft is the activation-saving overhead a
                     co-serve step pays over an inference-only one
and for the decode part:
  B_d                number of decode requests
  K                  total decode KV tokens

Two parameter sets are fitted, selected by whether the step will run as a CUDA
graph (queried from vLLM's CudagraphDispatcher, never mirrored). Both keep the
full 6 coefficients — including γ — so a future graphed co-serving batch is
covered. There is no separate "capture" regime: vLLM pre-captures graphs at
init, so live steps are either graph-replay or eager.

Admission: adding a fresh FT request of x tokens perturbs only the prefill/FT
features (S += x², T_in += x, T_ft += x; decode unchanged), so its marginal
step-time cost is ΔT(x) = α·x² + (β+γ)·x, independent of the decode load. The
SLO budget (deadline minus the current step's predicted time) bounds x via the
quadratic root.

This is a port of DeltaServe's ``router/tracker.py`` (PrefillExecutionEstimator
+ DecodeExecutionEstimator + BatchExecutionTracker), collapsed into one model.
"""

from __future__ import annotations

import csv
import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from vllm.deltaserve import dprint

# Minimum recorded steps per regime before a (re)fit is attempted. The model
# has 6 free parameters; require a small margin over that for a stable lstsq.
MIN_FIT_SAMPLES = 8
# Live refit cadence (in recorded steps).
REFIT_EVERY = 256


@dataclass
class StepFeatures:
    """Batch-composition features for one scheduler step."""

    t_in: float = 0.0          # total prefill tokens (includes FT)
    p: int = 0                 # number of prefill samples
    t_ft: float = 0.0          # FT tokens (subset of t_in)
    b_d: int = 0               # number of decode requests
    k: float = 0.0             # total decode KV tokens
    # Exact per-request prefill token counts. When present, S is computed
    # exactly; otherwise the T_in²/P proxy is used.
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
        """Design-matrix row [S, T_in, T_ft, B_d, K, 1]."""
        return [self.s, float(self.t_in), float(self.t_ft),
                float(self.b_d), float(self.k), 1.0]


@dataclass
class StepParams:
    """Fitted coefficients for one regime. None ⇒ not yet fitted."""

    alpha: float | None = None   # S
    beta: float | None = None    # T_in
    gamma: float | None = None   # T_ft
    delta: float | None = None   # B_d
    epsilon: float | None = None # K
    c: float | None = None       # constant

    @property
    def is_fitted(self) -> bool:
        return all(v is not None for v in
                   (self.alpha, self.beta, self.gamma,
                    self.delta, self.epsilon, self.c))

    def eval(self, f: StepFeatures) -> float:
        return (self.alpha * f.s + self.beta * f.t_in + self.gamma * f.t_ft
                + self.delta * f.b_d + self.epsilon * f.k + self.c)


class StepExecutionTracker:
    """Rolling record of every served step: features, measured + predicted
    duration, and the regime (was_graph) that was in effect at dispatch.

    ``was_graph`` is stored verbatim so a later refit cannot reclassify a
    historical sample after a bucket's graph state changes (cf. DeltaServe
    ``tracker.py:51-57``). Ports ``BatchExecutionTracker``.
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
        """Dump predicted-vs-actual per step for offline analysis."""
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
                "was_graph": self.was_graph[i],
                "t_in": f.t_in, "p": f.p, "t_ft": f.t_ft,
                "b_d": f.b_d, "k": f.k, "s": f.s,
                "execution_duration": self.durations[i],
                "predicted_duration": pred,
            })
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "timestamp", "step_index", "was_graph", "t_in", "p", "t_ft",
                "b_d", "k", "s", "execution_duration", "predicted_duration"])
            writer.writeheader()
            writer.writerows(rows)
        dprint(f"[estimator] wrote {len(rows)} prediction rows → {csv_path}")


class MergedExecutionEstimator:
    """Two-regime (eager / graph) merged step-time model + admission solver."""

    def __init__(self) -> None:
        self._eager = StepParams()
        self._graph = StepParams()
        self.eager_rmse: float | None = None
        self.graph_rmse: float | None = None
        self._warned_unfitted = False

    @property
    def is_ready(self) -> bool:
        return self._eager.is_fitted or self._graph.is_fitted

    @property
    def fit_rmse(self) -> float | None:
        return self.eager_rmse if self.eager_rmse is not None else self.graph_rmse

    # ---------------- prediction ----------------
    def _select(self, will_use_graph: bool) -> StepParams:
        if will_use_graph and self._graph.is_fitted:
            return self._graph
        if not will_use_graph and self._eager.is_fitted:
            return self._eager
        # Cold-start fallback: whichever regime is fitted.
        return self._eager if self._eager.is_fitted else self._graph

    def predict(self, features: StepFeatures,
                will_use_graph: bool = False) -> float:
        if not self.is_ready:
            if not self._warned_unfitted:
                dprint("[estimator] predict() called before any fit; "
                       "returning 0.0")
                self._warned_unfitted = True
            return 0.0
        p = self._select(will_use_graph)
        pred = p.eval(features)
        rmse = (self.graph_rmse if (will_use_graph and self._graph.is_fitted)
                else self.eager_rmse)
        if rmse:
            pred *= 1.0 + 1.5 * rmse   # pessimistic safety margin (DeltaServe)
        return max(0.0, float(pred))

    # ---------------- admission ----------------
    def max_next_ft_tokens(self, budget_s: float) -> int:
        """Largest x (FT tokens for the next FT request) with
        α·x² + (β+γ)·x ≤ budget_s. Uses eager params (FT forces eager).

        Returns 0 if the model is unfitted or the budget is non-positive.
        """
        p = self._eager if self._eager.is_fitted else self._graph
        if not p.is_fitted:
            return 0
        if budget_s <= 0:
            return 0
        a2 = float(p.alpha)
        a1 = float(p.beta + p.gamma)
        if a2 <= 0:
            # Degenerate quadratic → linear bound.
            if a1 <= 0:
                return 10 ** 12
            return max(0, int(math.floor(budget_s / a1)))
        disc = a1 * a1 + 4.0 * a2 * budget_s
        if disc < 0:
            return 0
        x = (-a1 + math.sqrt(disc)) / (2.0 * a2)
        return max(0, int(math.floor(x)))

    # ---------------- fitting ----------------
    def data_fit(self, tracker: StepExecutionTracker) -> tuple[StepParams,
                                                               StepParams]:
        """Refit both regimes from the tracker, partitioned by the recorded
        was_graph stamp (None ⇒ eager). lstsq over [S,T_in,T_ft,B_d,K,1]."""
        eager_X, eager_y = [], []
        graph_X, graph_y = [], []
        for f, dur, wg in zip(tracker.features, tracker.durations,
                              tracker.was_graph):
            # A step with FT tokens always ran eager (forced).
            is_graph = bool(wg) and not f.has_ft
            if is_graph:
                graph_X.append(f.row())
                graph_y.append(dur)
            else:
                eager_X.append(f.row())
                eager_y.append(dur)

        self._eager, self.eager_rmse = self._fit_regime(
            eager_X, eager_y, self._eager, "eager")
        self._graph, self.graph_rmse = self._fit_regime(
            graph_X, graph_y, self._graph, "graph")
        dprint(f"[estimator] fit: eager={len(eager_y)} (rmse={self.eager_rmse}), "
               f"graph={len(graph_y)} (rmse={self.graph_rmse})")
        return self._eager, self._graph

    @staticmethod
    def _fit_regime(X: list[list[float]], y: list[float],
                    prev: StepParams, name: str) -> tuple[StepParams,
                                                          float | None]:
        if len(y) < MIN_FIT_SAMPLES:
            if len(y) > 0:
                dprint(f"[estimator] {name} regime has {len(y)} samples "
                       f"(<{MIN_FIT_SAMPLES}); keeping previous params")
            return prev, None
        A = np.asarray(X, dtype=float)
        b = np.asarray(y, dtype=float)
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        alpha, beta, gamma, delta, epsilon, c = (float(v) for v in coef)
        params = StepParams(alpha=alpha, beta=beta, gamma=gamma,
                            delta=delta, epsilon=epsilon, c=c)
        preds = A @ coef
        rmse = float(np.sqrt(np.mean((preds - b) ** 2)))
        return params, rmse
