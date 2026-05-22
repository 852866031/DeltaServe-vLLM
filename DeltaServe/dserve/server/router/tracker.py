from dataclasses import dataclass
import math
import os
from typing import Iterable, List, Optional, Sequence, Tuple
import time
import numpy as np
from dserve.server.io_struct import Req
from enum import Enum
import csv
import json
from pathlib import Path

EPS = 1e-9

class BatchExecutionType(Enum):
    PREFILL = 0
    DECODE = 1


class BatchExecutionTracker():
    def __init__(self, max_batches = 10240) -> None:
        self.max_batches = max_batches
        self.inference_tokens_list = []
        self.finetuning_tokens_list = []
        self.execution_type_list = []
        self.execution_duration_list = []
        self.predicted_duration_list = []
        self.was_graph_list: List[Optional[bool]] = []
        self.timestamp_list = []
        self.last_refit_count = 0
    
    def check_refit(self) -> bool:
        if self.size()%256 == 0 and self.size() > self.last_refit_count:
            self.last_refit_count = self.size()
            return True
        return False

    def _enforce_max_size(self) -> None:
        if self.max_batches is not None and len(self.execution_type_list) > self.max_batches:
            self.drop_batch_stats(0)

    def add_batch_stats(
        self,
        inference_tokens: Sequence[List[int]],        # per-request inference tokens
        finetuning_tokens: Sequence[List[int]],    # per-request FT tokens
        execution_type: BatchExecutionType,
        execution_duration: float,
        predicted_duration: Optional[float] = None,
        was_graph: Optional[bool] = None,
    ) -> None:
        """`was_graph` is the regime that was *actually* in effect when this
        batch ran (mirror state at dispatch time). Stored verbatim so that
        `data_fit` doesn't reclassify the sample later — once a bucket is
        captured the mirror flips True for all later runs of that shape, but
        the historical sample's regime never changes. None means "unknown,
        fall back to mirror lookup at fit time" (used by offline profiling
        where the mirror isn't seeded yet)."""
        self.timestamp_list.append(time.time())
        self.inference_tokens_list.append(inference_tokens)
        self.finetuning_tokens_list.append(finetuning_tokens)
        self.execution_type_list.append(execution_type)
        self.execution_duration_list.append(execution_duration)
        self.predicted_duration_list.append(predicted_duration)
        self.was_graph_list.append(was_graph)

    def drop_batch_stats(self, index: int) -> None:
        """Drop the batch statistics at the specified index."""
        if index < 0 or index >= len(self.execution_type_list):
            raise IndexError("Index out of range")

        del self.inference_tokens_list[index]
        del self.finetuning_tokens_list[index]
        del self.execution_type_list[index]
        del self.execution_duration_list[index]
        del self.predicted_duration_list[index]
        del self.was_graph_list[index]
    
    def size(self) -> int:
        """Return the number of recorded batches."""
        return len(self.execution_type_list)

    def print_batch_prediction_stats(self) -> None:
        prefill_batch_layouts = []
        prefill_execution_durations = []
        prefill_predicted_durations = []
        decode_batch_layouts = []
        decode_execution_durations = []
        decode_predicted_durations = []
        for i in range(self.size()):
            if self.predicted_duration_list[i] is None:
                continue
            batch_type = self.execution_type_list[i]
            if batch_type == BatchExecutionType.PREFILL:
                prefill_batch_layouts.append((self.inference_tokens_list[i], self.finetuning_tokens_list[i]))
                prefill_execution_durations.append(self.execution_duration_list[i])
                prefill_predicted_durations.append(self.predicted_duration_list[i])
            else:
                decode_batch_layouts.append((self.inference_tokens_list[i], self.finetuning_tokens_list[i]))
                decode_execution_durations.append(self.execution_duration_list[i])
                decode_predicted_durations.append(self.predicted_duration_list[i])
        print("--- Prefill Batch Prediction Stats ---")
        #print as ready to copy paste python list code
        print("Batch Layouts:", prefill_batch_layouts)
        print("Execution Durations:", prefill_execution_durations)
        print("Predicted Durations:", prefill_predicted_durations)
        print("--- Decode Batch Prediction Stats ---")
        #print as ready to copy paste python list code
        print("Batch Layouts:", decode_batch_layouts)
        print("Execution Durations:", decode_execution_durations)
        print("Predicted Durations:", decode_predicted_durations)
    
    def write_batch_prediction_stats_to_csv(
        self,
        csv_path: Optional[str] = None,
    ):
        # Resolve path: explicit arg wins; else read scheduler.batch_prediction_stats_path
        # from the active config (default: output/scheduler/batch_prediction_stats.csv).
        # `None` from the YAML disables the dump entirely.
        if csv_path is None:
            try:
                from dserve.common.configs.config import get_active_config
                csv_path = get_active_config().scheduler.batch_prediction_stats_path
            except Exception:
                csv_path = "output/scheduler/batch_prediction_stats.csv"
        if csv_path is None:
            print("[BatchExecutionTracker] scheduler.batch_prediction_stats_path "
                  "is null — skipping prediction-stats CSV dump.")
            return
        # Anchor relative paths to the working directory's project root so
        # absolute and relative configurations both land somewhere sensible.
        csv_path = str(csv_path)
        parent_dir = os.path.dirname(csv_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        rows = []
        for i in range(self.size()):
            pred = self.predicted_duration_list[i]
            if pred is None:
                continue

            batch_type_enum = self.execution_type_list[i]
            batch_type = (
                "prefill" if batch_type_enum == BatchExecutionType.PREFILL else "decode"
            )

            rows.append({
                "timestamp": self.timestamp_list[i],
                "batch_index": i,
                "batch_type": batch_type,
                "inference_tokens": json.dumps(self.inference_tokens_list[i]),
                "finetuning_tokens": json.dumps(self.finetuning_tokens_list[i]),
                "execution_duration": self.execution_duration_list[i],
                "predicted_duration": pred,
            })

        # Write single CSV
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "batch_index",
                    "batch_type",
                    "inference_tokens",
                    "finetuning_tokens",
                    "execution_duration",
                    "predicted_duration",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        print(f"[BatchExecutionTracker] Wrote {len(rows)} rows → {csv_path}")
                
        
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Optional
import numpy as np
import time
import math


@dataclass
class PrefillParams:
    """Fitted parameters for the prefill execution model."""
    alpha: Optional[float] = None  # coefficient for sum(n_i^2)
    beta: Optional[float] = None   # coefficient for T_in = sum(n_i)
    gamma: Optional[float] = None  # extra per-token cost for FT
    c: Optional[float] = None      # constant overhead per batch


class PrefillExecutionEstimator:
    """
    Execution time model (prefill), per regime:

        T_prefill ≈ α * Σ n_i² + β * T_in + γ * T_ft + c

    Two regimes are fit separately because the runtime is qualitatively
    different:

      * graph regime (`_graph_params`) — inference-only batches whose
        (bs_bucket, T_bucket) is captured. Replay time is near-constant
        per bucket; γ is irrelevant (FT batches always go eager). α/β
        absorb whatever weak shape dependence remains.
      * eager regime (`_eager_params`) — everything else: uncaptured
        inference-only buckets AND all co-serving batches. Full
        α/β/γ/c. γ captures the activation-saving overhead from
        co-serving.

    Caller is expected to know which regime the predicted batch will
    run in (via GraphEligibility) and pass `will_use_graph=...` to
    `predict_inference`. `predict_coserving` is unconditionally eager
    because the runtime gates the prefill graph off for any batch with
    FT tokens (`lora_unordered_batch_mixed.py`, `not has_ft`).
    """

    def __init__(self) -> None:
        self._graph_params = PrefillParams()
        self._eager_params = PrefillParams()
        self.graph_fit_rmse: Optional[float] = None
        self.eager_fit_rmse: Optional[float] = None

    @property
    def fit_rmse(self) -> Optional[float]:
        """Back-compat: legacy callers read .fit_rmse. Eager is always
        present once any data exists; report it as the headline."""
        return self.eager_fit_rmse if self.eager_fit_rmse is not None else self.graph_fit_rmse

    @fit_rmse.setter
    def fit_rmse(self, value):
        # Setters at verify_* sites — preserve existing semantics by
        # bumping the eager rmse (the verify path is eager-default).
        self.eager_fit_rmse = value

    @staticmethod
    def _as_np(x: Iterable[float]) -> np.ndarray:
        return np.asarray(list(x), dtype=float)

    @staticmethod
    def _linfit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return coef

    # ======================================================================
    # Fitting from explicit batch stats
    # ======================================================================
    def fit(
        self,
        inference_only_tokens: Sequence[List[int]],      # per-batch: [n1, n2, ...]
        inference_only_times: Sequence[float],
        coserving_inf_tokens: Sequence[List[int]],       # per-batch: [n1, n2, ...]
        coserving_ft_tokens: Sequence[List[int]],        # per-batch: [m1, m2, ...]
        coserving_times: Sequence[float],
    ) -> PrefillParams:
        """
        Fit α, β, γ, c from:
          - inference-only batches
          - co-serving batches (inference + FT)

        All token lists are flat 1D lists of per-request total token counts.
        """

        sum_n2_list: List[float] = []
        T_in_list: List[float] = []
        T_ft_list: List[float] = []
        T_measured: List[float] = []

        # ---------- inference-only batches ----------
        for token_list, T in zip(inference_only_tokens, inference_only_times):
            n_inf = np.asarray(token_list, dtype=float)
            if n_inf.size == 0:
                continue

            n_total = n_inf
            sum_n2 = float(np.sum(n_total ** 2))
            T_in = float(np.sum(n_total))
            T_ft = 0.0

            sum_n2_list.append(sum_n2)
            T_in_list.append(T_in)
            T_ft_list.append(T_ft)
            T_measured.append(float(T))

        # ---------- co-serving batches ----------
        for inf_list, ft_list, T in zip(
            coserving_inf_tokens,
            coserving_ft_tokens,
            coserving_times,
        ):
            n_inf = np.asarray(inf_list, dtype=float)
            n_ft = np.asarray(ft_list, dtype=float)

            if n_inf.size == 0 and n_ft.size == 0:
                continue

            # All requests in the batch (inference + FT)
            n_total = np.concatenate([n_inf, n_ft])

            sum_n2 = float(np.sum(n_total ** 2))
            T_in = float(np.sum(n_total))
            T_ft = float(np.sum(n_ft))

            sum_n2_list.append(sum_n2)
            T_in_list.append(T_in)
            T_ft_list.append(T_ft)
            T_measured.append(float(T))

        if len(T_measured) < 4:
            raise ValueError("Not enough batches to fit PrefillExecutionEstimator (need ≥4).")

        S = self._as_np(sum_n2_list)
        Tin = self._as_np(T_in_list)
        Tft = self._as_np(T_ft_list)
        T = self._as_np(T_measured)

        X = np.column_stack([S, Tin, Tft, np.ones_like(T)])
        alpha, beta, gamma, c = self._linfit(X, T)

        # Legacy `fit()` doesn't have eligibility info, so it can't partition
        # by regime. Treat the whole dataset as eager — that's the safe regime
        # since γ is identifiable here (co-serve batches are always eager).
        self._eager_params = PrefillParams(
            alpha=float(alpha),
            beta=float(beta),
            gamma=float(gamma),
            c=float(c),
        )

        preds = X @ np.array([alpha, beta, gamma, c])
        self.eager_fit_rmse = float(np.sqrt(np.mean((preds - T) ** 2)))
        return self._eager_params

    # ======================================================================
    # Prediction API
    # ======================================================================
    def _select_params(self, will_use_graph: bool) -> PrefillParams:
        """Pick params for the requested regime, falling back if the
        regime has no fit yet (cold-start safety)."""
        if will_use_graph and self._graph_params.alpha is not None:
            return self._graph_params
        if not will_use_graph and self._eager_params.alpha is not None:
            return self._eager_params
        # Fallback: whatever's available.
        return self._eager_params if self._eager_params.alpha is not None else self._graph_params

    def _eval_params(self, p: PrefillParams, S: float, Tin: float) -> Optional[float]:
        """Evaluate the linear model with the given params; returns None
        when the regime hasn't been fitted yet."""
        if any(v is None for v in (p.alpha, p.beta, p.c)):
            return None
        return float(p.alpha * S + p.beta * Tin + p.c)

    def predict_inference(
        self,
        token_list: List[int],
        will_use_graph: bool = False,
        will_capture: bool = False,
    ) -> float:
        """Predict prefill time for an inference-only batch.

        Three regimes:
          * `will_capture=True` → first-touch CUDA-graph capture path.
            Cost is structural: 2 eager forwards (side-stream warmup +
            in-capture forward) + 1 graph replay
            (`cuda_graph_runner.prefill_capture()` lines 459-471). Both
            regimes are evaluated and combined analytically — no
            third-regime fit is required.
          * `will_use_graph=True` (and not capturing) → graph regime.
          * else → pure eager regime.

        `will_capture` takes precedence over `will_use_graph` since the
        capture path runs the eager pieces anyway.
        """
        n = np.asarray(token_list, dtype=float)
        if n.size == 0:
            return 0.0

        S = float(np.sum(n ** 2))
        Tin = float(np.sum(n))

        if will_capture:
            eager_pred = self._eval_params(self._eager_params, S, Tin)
            graph_pred = self._eval_params(self._graph_params, S, Tin)
            if eager_pred is None and graph_pred is None:
                print("Model not fitted yet")
                return 0.5
            # Cold-start safety: if a regime is unfitted, substitute the
            # other so the structural sum is still meaningful.
            if eager_pred is None:
                eager_pred = graph_pred
            if graph_pred is None:
                graph_pred = eager_pred
            pred = 2.0 * eager_pred + graph_pred
            # Eager components dominate; bump by eager rmse.
            if self.eager_fit_rmse:
                pred *= 1 + 1.5 * self.eager_fit_rmse
            return float(pred)

        p = self._select_params(will_use_graph)
        if any(v is None for v in (p.alpha, p.beta, p.c)):
            print("Model not fitted yet")
            return 0.5

        pred = p.alpha * S + p.beta * Tin + p.c

        rmse = self.graph_fit_rmse if (will_use_graph and self._graph_params.alpha is not None) else self.eager_fit_rmse
        if rmse:
            pred *= 1 + 1.5 * rmse
        return float(pred)

    def predict_coserving(
        self,
        inference_tokens: List[int],
        finetuning_tokens: List[int],
    ) -> float:
        """
        Predict prefill time for a co-serving batch. Unconditionally
        uses eager-regime params — the runtime gates the prefill graph
        off for any batch with FT tokens.
        """
        p = self._eager_params
        if any(v is None for v in (p.alpha, p.beta, p.gamma, p.c)):
            print("Model not fitted yet")
            return 0.5

        n_inf = np.asarray(inference_tokens, dtype=float)
        n_ft = np.asarray(finetuning_tokens, dtype=float)

        if n_inf.size == 0 and n_ft.size == 0:
            return 0.0

        n_total = np.concatenate([n_inf, n_ft])

        S = float(np.sum(n_total ** 2))
        Tin = float(np.sum(n_total))
        Tft = float(np.sum(n_ft))

        pred = p.alpha * S + p.beta * Tin + p.gamma * Tft + p.c
        if self.eager_fit_rmse:
            pred *= 1 + 1.5 * self.eager_fit_rmse
        return float(pred)

    # ======================================================================
    # Verification helpers
    # ======================================================================
    def verify_inference(self, token_list: List[int], actual_time: float) -> float:
        pred = self.predict_inference(token_list)
        err = abs(pred - actual_time) / max(actual_time, 1e-9)
        print(f"[verify_inference] pred {pred:.3f}s vs actual {actual_time:.3f}s (err {err:.2%})")
        self.fit_rmse = max(self.fit_rmse or 0.0, err)
        return err

    def verify_coserving(
        self,
        inference_tokens: List[int],
        finetuning_tokens: List[int],
        actual_time: float,
    ) -> float:
        pred = self.predict_coserving(inference_tokens, finetuning_tokens)
        err = abs(pred - actual_time) / max(actual_time, 1e-9)
        print(f"[verify_coserving] pred {pred:.3f}s vs actual {actual_time:.3f}s (err {err:.2%})")
        self.fit_rmse = max(self.fit_rmse or 0.0, err)
        return err

    # ======================================================================
    # SLO-based FT admission
    # ======================================================================
    def max_next_ft_tokens(
        self,
        inf_tokens: List[int],     # current inference requests
        ft_tokens: List[int],      # current FT requests
        earliest_req_time: Optional[float],
        ttft: float,
        *,
        ttft_unit: str = "s",
        now: Optional[float] = None,
    ) -> int:
        """
        Compute the maximum FT tokens x allowed for the *next FT request*
        so that TTFT SLO is not violated.

        Current batch:
          - inference requests: inf_tokens
          - FT requests:        ft_tokens
        New FT request: adds x tokens (as its own request).
        Unconditionally uses eager-regime params — adding any FT token
        forces the runtime onto the eager prefill path.
        """

        p = self._eager_params
        if any(v is None for v in (p.alpha, p.beta, p.gamma, p.c)):
            raise ValueError("PrefillExecutionEstimator (eager) not fitted yet.")

        # No SLO → effectively unlimited
        if earliest_req_time is None:
            return 10**12

        if now is None:
            now = time.time()

        # Normalize TTFT
        if ttft_unit == "s":
            ttft_s = float(ttft)
        elif ttft_unit == "ms":
            ttft_s = float(ttft) / 1000.0
        else:
            raise ValueError("ttft_unit must be 's' or 'ms'.")

        deadline = float(earliest_req_time) + ttft_s
        rem_time = deadline - now
        if rem_time <= 0:
            return 0

        # Safety margin (can incorporate RMSE if you want)
        safety = 1.0
        time_budget = rem_time / safety

        # Current batch stats
        n_inf = np.asarray(inf_tokens, dtype=float)
        n_ft = np.asarray(ft_tokens, dtype=float)
        n_total = np.concatenate([n_inf, n_ft]) if (n_inf.size or n_ft.size) else np.zeros(0, dtype=float)

        S_curr = float(np.sum(n_total ** 2))
        T_in_curr = float(np.sum(n_total))
        T_ft_curr = float(np.sum(n_ft))

        const_term = (
            p.alpha * S_curr +
            p.beta * T_in_curr +
            p.gamma * T_ft_curr +
            p.c
        )

        rhs = time_budget - const_term
        if rhs <= 0:
            return 0

        # New FT request with x tokens:
        #   - contributes n_total = x
        #   - contributes T_ft    = x
        #
        # Incremental delay:
        #   ΔT(x) = α * x² + (β + γ) * x
        #
        # Solve: α x² + (β + γ) x - rhs ≤ 0
        a2 = float(p.alpha)
        a1 = float(p.beta + p.gamma)

        if a2 <= 0:
            # Degenerate → linear bound
            x_cont = rhs / max(a1, 1e-12)
            return max(0, int(math.floor(x_cont)))

        disc = a1 * a1 + 4.0 * a2 * rhs
        if disc < 0:
            return 0

        x_root = (-a1 + math.sqrt(disc)) / (2.0 * a2)
        x_max = max(0, math.floor(x_root))
        return int(x_max)

    def data_fit(self, tracker: "BatchExecutionTracker", eligibility=None):
        """
        Fit the prefill estimator from tracked PREFILL batches, partitioned
        by regime:

          * graph regime: inference-only batches whose
            (bs_bucket, T_bucket) is in the eligibility mirror.
          * eager regime: everything else (uncaptured inf-only + all
            co-serving). γ only matters here.

        If `eligibility` is None or unfitted, all batches go into the
        eager regime (back-compat for callers that haven't been wired
        to graph-aware scheduling yet).

        Returns (eager_params, graph_params). Each may have all-None
        fields if its regime didn't have enough samples (≥4 required).
        """
        # Per-regime feature lists
        eager_S, eager_Tin, eager_Tft, eager_T = [], [], [], []
        graph_S, graph_Tin, graph_T = [], [], []

        n_total_seen = 0

        for inf_tokens_per_batch, ft_tokens_per_batch, exec_type, duration, stamped_was_graph in zip(
            tracker.inference_tokens_list,
            tracker.finetuning_tokens_list,
            tracker.execution_type_list,
            tracker.execution_duration_list,
            tracker.was_graph_list,
        ):
            if exec_type != BatchExecutionType.PREFILL:
                continue

            n_inf = np.asarray(inf_tokens_per_batch, dtype=float)
            n_ft  = np.asarray(ft_tokens_per_batch, dtype=float)

            if n_inf.size == 0 and n_ft.size == 0:
                continue

            has_ft = n_ft.size > 0 and float(np.sum(n_ft)) > 0
            n_total = (
                np.concatenate([n_inf, n_ft])
                if (n_inf.size and n_ft.size)
                else (n_inf if n_inf.size else n_ft)
            )

            sum_n2 = float(np.sum(n_total ** 2))
            T_in = float(np.sum(n_total))
            T_ft = float(np.sum(n_ft)) if n_ft.size else 0.0

            # Decide regime: prefer the at-execution-time stamp; only fall
            # back to the current mirror for legacy / offline-profiling
            # samples that didn't record one.
            if has_ft:
                is_graph = False
            elif stamped_was_graph is not None:
                is_graph = bool(stamped_was_graph)
            elif eligibility is not None:
                bs = int(n_inf.size)
                total = int(np.sum(n_inf))
                is_graph = bool(eligibility.will_prefill_use_graph(False, bs, total))
            else:
                is_graph = False

            if is_graph:
                graph_S.append(sum_n2)
                graph_Tin.append(T_in)
                graph_T.append(float(duration))
            else:
                eager_S.append(sum_n2)
                eager_Tin.append(T_in)
                eager_Tft.append(T_ft)
                eager_T.append(float(duration))
            n_total_seen += 1

        if n_total_seen < 4:
            raise ValueError(
                f"Not enough PREFILL batches to fit (need ≥4, got {n_total_seen})."
            )

        # Eager fit (4 params)
        if len(eager_T) >= 4:
            S = np.asarray(eager_S); Tin = np.asarray(eager_Tin)
            Tft = np.asarray(eager_Tft); T = np.asarray(eager_T)
            X = np.column_stack([S, Tin, Tft, np.ones_like(T)])
            a, b, g, c = self._linfit(X, T)
            self._eager_params = PrefillParams(
                alpha=float(a), beta=float(b), gamma=float(g), c=float(c))
            preds = X @ np.array([a, b, g, c])
            self.eager_fit_rmse = float(np.sqrt(np.mean((preds - T) ** 2)))
        else:
            print(f"[PrefillEstimator] eager regime has {len(eager_T)} samples (<4); keeping previous eager params")

        # Graph fit (3 params: γ omitted, no FT in graph-regime data)
        if len(graph_T) >= 4:
            S = np.asarray(graph_S); Tin = np.asarray(graph_Tin); T = np.asarray(graph_T)
            X = np.column_stack([S, Tin, np.ones_like(T)])
            a, b, c = self._linfit(X, T)
            self._graph_params = PrefillParams(
                alpha=float(a), beta=float(b), gamma=0.0, c=float(c))
            preds = X @ np.array([a, b, c])
            self.graph_fit_rmse = float(np.sqrt(np.mean((preds - T) ** 2)))
        else:
            if len(graph_T) > 0:
                print(f"[PrefillEstimator] graph regime has {len(graph_T)} samples (<4); keeping previous graph params")

        print(
            f"[PrefillEstimator] fit: eager={len(eager_T)} samples "
            f"(rmse={self.eager_fit_rmse}), graph={len(graph_T)} samples "
            f"(rmse={self.graph_fit_rmse})"
        )

        return self._eager_params, self._graph_params
        

@dataclass
class DecodeParams:
    delta: float = 0.0   # per-request term
    epsilon: float = 0.0 # per-KV token term
    d: float = 0.0       # constant overhead

class DecodeExecutionEstimator:
    """
    Execution time model (decode per step), per regime:

        T_decode ≈ δ * B_t + ε * K_t + d

    where:
        B_t = number of active inference requests
        K_t = total KV-cache tokens across requests

    Two regimes (graph / eager) — exactly the same split as prefill.
    Decode is always inference-only (no FT) so the only signal that
    determines regime is whether the (B, max_len_bucket) is captured.
    """

    def __init__(self) -> None:
        self._graph_params = DecodeParams()
        self._eager_params = DecodeParams()
        self.graph_fit_rmse = None
        self.eager_fit_rmse = None

    @property
    def fit_rmse(self):
        return self.eager_fit_rmse if self.eager_fit_rmse is not None else self.graph_fit_rmse

    @fit_rmse.setter
    def fit_rmse(self, value):
        self.eager_fit_rmse = value

    @staticmethod
    def _as_np_1d(x: Iterable[float]) -> np.ndarray:
        arr = np.asarray(list(x), dtype=float)
        if arr.ndim != 1:
            raise ValueError("Expected 1D array-like")
        return arr

    @staticmethod
    def _linfit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return coef

    # ----------------------------------------------------------------------
    # Fitting from raw arrays
    # ----------------------------------------------------------------------
    def fit(
        self,
        total_tokens: Sequence[float],  # K_t
        batch_sizes: Sequence[float],   # B_t
        times: Sequence[float],         # measured decode times
    ) -> DecodeParams:
        K = self._as_np_1d(total_tokens)
        B = self._as_np_1d(batch_sizes)
        T = self._as_np_1d(times)

        X = np.column_stack([B, K, np.ones_like(B)])
        delta, epsilon, d = self._linfit(X, T)

        # Legacy `fit()` has no eligibility info — treat as eager regime.
        self._eager_params = DecodeParams(delta=float(delta), epsilon=float(epsilon), d=float(d))

        preds = X @ np.array([delta, epsilon, d])
        self.eager_fit_rmse = float(np.sqrt(np.mean((preds - T) ** 2)))
        return self._eager_params

    # ----------------------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------------------
    def _select_params(self, will_use_graph: bool) -> DecodeParams:
        if will_use_graph and self._graph_params.delta != 0.0:
            return self._graph_params
        if not will_use_graph and self._eager_params.delta != 0.0:
            return self._eager_params
        # Cold-start fallback: pick whichever is fitted.
        return self._graph_params if self._graph_params.delta != 0.0 else self._eager_params

    @staticmethod
    def _params_fitted(p: "DecodeParams") -> bool:
        # `delta != 0.0` is the existing fitted-sentinel used by
        # `_select_params`; preserve that convention.
        return p.delta != 0.0 or p.epsilon != 0.0 or p.d != 0.0

    def predict(
        self,
        total_tokens: float,
        batch_size: float,
        will_use_graph: bool = False,
        will_capture: bool = False,
    ) -> float:
        """Three regimes, mirroring `PrefillExecutionEstimator`.

        Decode `capture()` does 3 warmup forwards + 1 forward inside the
        graph context, with no post-capture replay
        (`cuda_graph_runner.py:249-260`). So capture cost ≈ 4 × eager;
        the graph replay term doesn't appear here.
        """
        if will_capture:
            ep = self._eager_params
            gp = self._graph_params
            eager_pred = (
                ep.delta * batch_size + ep.epsilon * total_tokens + ep.d
                if self._params_fitted(ep) else None
            )
            graph_pred = (
                gp.delta * batch_size + gp.epsilon * total_tokens + gp.d
                if self._params_fitted(gp) else None
            )
            if eager_pred is None and graph_pred is None:
                return 0.0
            if eager_pred is None:
                eager_pred = graph_pred
            return float(4.0 * eager_pred)

        p = self._select_params(will_use_graph)
        pred = p.delta * batch_size + p.epsilon * total_tokens + p.d
        return float(pred)

    def verify(self, total_tokens: float, batch_size: float, actual_time: float) -> float:
        pred = self.predict(total_tokens, batch_size)
        err = abs(pred - actual_time) / max(actual_time, 1e-9)
        print(f"[verify_decode] pred {pred:.6f}s vs actual {actual_time:.6f}s (err={err:.2%})")
        self.fit_rmse = max(self.fit_rmse or 0.0, err)
        return err

    # ----------------------------------------------------------------------
    # Fitting directly from the tracker
    # ----------------------------------------------------------------------
    def data_fit(self, tracker: BatchExecutionTracker, eligibility=None):
        """
        Fit using DECODE batches from tracker, partitioned by graph
        eligibility. If `eligibility` is None or unfitted, all samples
        go into eager (back-compat).
        """
        eager_B, eager_K, eager_T = [], [], []
        graph_B, graph_K, graph_T = [], [], []

        for inf_tokens_per_batch, ft_tokens_per_batch, exec_type, duration, stamped_was_graph in zip(
            tracker.inference_tokens_list,
            tracker.finetuning_tokens_list,
            tracker.execution_type_list,
            tracker.execution_duration_list,
            tracker.was_graph_list,
        ):
            if exec_type != BatchExecutionType.DECODE:
                continue

            try:
                n_inf = np.asarray([sum(toks) for toks in inf_tokens_per_batch], dtype=float)
            except Exception:
                n_inf = np.asarray(inf_tokens_per_batch, dtype=float)

            if len(n_inf) == 0:
                continue

            B = len(n_inf)
            K = float(np.sum(n_inf))

            # Decide regime: prefer the at-execution-time stamp; fall back
            # to the current mirror only for samples without one.
            max_len = int(n_inf.max()) if n_inf.size > 0 else 0
            if stamped_was_graph is not None:
                is_graph = bool(stamped_was_graph)
            elif eligibility is not None:
                is_graph = bool(eligibility.will_decode_use_graph(B, max_len))
            else:
                is_graph = False

            if is_graph:
                graph_B.append(B); graph_K.append(K); graph_T.append(duration)
            else:
                eager_B.append(B); eager_K.append(K); eager_T.append(duration)

        n_total = len(eager_T) + len(graph_T)
        if n_total < 3:
            raise ValueError(f"Not enough DECODE batches (need ≥3, got {n_total}).")

        if len(eager_T) >= 3:
            B = np.asarray(eager_B, dtype=float); K = np.asarray(eager_K, dtype=float); T = np.asarray(eager_T, dtype=float)
            X = np.column_stack([B, K, np.ones_like(B)])
            d, e, off = self._linfit(X, T)
            self._eager_params = DecodeParams(delta=float(d), epsilon=float(e), d=float(off))
            preds = X @ np.array([d, e, off])
            self.eager_fit_rmse = float(np.sqrt(np.mean((preds - T) ** 2)))
        else:
            print(f"[DecodeEstimator] eager regime has {len(eager_T)} samples (<3); keeping previous eager params")

        if len(graph_T) >= 3:
            B = np.asarray(graph_B, dtype=float); K = np.asarray(graph_K, dtype=float); T = np.asarray(graph_T, dtype=float)
            X = np.column_stack([B, K, np.ones_like(B)])
            d, e, off = self._linfit(X, T)
            self._graph_params = DecodeParams(delta=float(d), epsilon=float(e), d=float(off))
            preds = X @ np.array([d, e, off])
            self.graph_fit_rmse = float(np.sqrt(np.mean((preds - T) ** 2)))
        else:
            if len(graph_T) > 0:
                print(f"[DecodeEstimator] graph regime has {len(graph_T)} samples (<3); keeping previous graph params")

        print(
            f"[DecodeEstimator] fit: eager={len(eager_T)} samples "
            f"(rmse={self.eager_fit_rmse}), graph={len(graph_T)} samples "
            f"(rmse={self.graph_fit_rmse})"
        )
        return self._eager_params, self._graph_params