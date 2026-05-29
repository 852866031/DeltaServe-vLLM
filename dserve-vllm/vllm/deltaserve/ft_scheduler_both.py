# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Sibling scheduler that admits FT on ANY step composition (prefill, decode,
mixed, idle) under the SLO estimator's gate — NOT just prefill-carrying steps.

Selected via ``scheduler_config.scheduler_cls`` when
``finetune.coserving_admission_phase == "both"`` (see ``vllm/config/vllm.py``
__post_init__). The proportional cap
``ft_tokens_admission_constrain_factor`` is prefill-relative so it's
incompatible with this mode — selection soft-falls back to ``FinetuneScheduler``
with a warning when both are set.

Two overrides vs the parent ``FinetuneScheduler``:

  * ``_initial_ft_budget`` — drops the ``decode_only → 0`` short-circuit so
    every step composition runs through the SLO budget computation.
  * ``_slo_ft_budget`` — when the upcoming step is decode-only, scales
    ``max_tbt_slo`` by ``finetune.decode_only_ft_safety_margin`` (default 0.7)
    before computing the TBT-derived FT-token cap. Tighter margin is justified
    by: (a) the eager penalty from losing the CUDA-graph fast path can
    dominate the sub-5ms decode-only step time, (b) the SLO estimator's γ
    coefficient hasn't seen many decode-only + FT samples until online refit
    has accumulated enough of them.

Inherits everything else from ``FinetuneScheduler``: injection, coordinator
wiring, backward triggers, async-safety (reserve-at-inject), the
``match_prefill_workload_factor`` leaky-bucket gate, the
``_unspent_prefill`` counter, etc.

``match_prefill_workload_factor`` semantics in "both" mode: the counter still
only accumulates on prefill-carrying steps (self-gated on ``feats.t_in > 0``
in the parent). On long decode-only stretches no new credit accrues — but
credit banked from prior prefill bursts can still release FT samples onto
decode-only steps as long as the leaky-bucket trigger condition fires. This
gives natural rate limiting without a separate "match decode" knob.
"""

import time

from vllm.deltaserve import dprint
from vllm.deltaserve.estimator import StepFeatures
from vllm.deltaserve.ft_scheduler import FinetuneScheduler


class BothPhaseFinetuneScheduler(FinetuneScheduler):
    """FT scheduler that admits to any step composition. See module docstring."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        ft_cfg = self.vllm_config.finetune_config
        self._decode_only_safety = float(getattr(
            ft_cfg, "decode_only_ft_safety_margin", 0.7))
        dprint(
            f"[ft-sched] BothPhaseFinetuneScheduler active | "
            f"decode_only_safety={self._decode_only_safety} | "
            f"max_tbt_slo={self._max_tbt_slo}s "
            f"(effective on decode-only: "
            f"{self._max_tbt_slo * self._decode_only_safety:.4f}s)"
        )

    # ─── FT admission gate: ANY step composition ──────────────────────────
    def _initial_ft_budget(self, feats: StepFeatures,
                           earliest_arrival: float | None) -> int:
        """Drop the parent's ``decode_only → 0`` short-circuit — every step
        composition runs through the SLO budget computation. The decode-only
        safety margin is applied inside ``_slo_ft_budget``."""
        return self._slo_ft_budget(feats, earliest_arrival)

    # ─── SLO budget with decode-only safety margin ────────────────────────
    def _slo_ft_budget(self, feats: StepFeatures,
                       earliest_arrival: float | None) -> int:
        """Same SLO math as the parent, but scales ``max_tbt_slo`` by
        ``decode_only_ft_safety_margin`` on decode-only steps (see class
        docstring for the rationale).

        Kept structurally close to ``FinetuneScheduler._slo_ft_budget`` so a
        future refactor that pulls the common body into a helper can land
        without churn here."""
        coord_budget = self._coord.next_ft_budget()
        if coord_budget <= 0 or not self._estimator.is_ready:
            return coord_budget
        # Baseline step time WITHOUT the new FT (eager: adding FT forces eager).
        t_current = self._estimator.predict(feats, will_use_graph=False)
        now = time.time()
        # Async-pipelined: queue behind the in-flight forward.
        queue_wait = self._last_step_pred_s if self._async else 0.0
        # Decode-only safety: scale TBT budget down so a wrong estimator
        # prediction has less blast radius.
        decode_only = feats.b_d > 0 and feats.t_in == 0
        effective_tbt_slo = (self._max_tbt_slo * self._decode_only_safety
                             if decode_only else self._max_tbt_slo)
        budgets_s: list[float] = []
        if earliest_arrival is not None and feats.t_in > 0:
            budgets_s.append(
                earliest_arrival + 0.9 * self._ttft_slo
                - now - queue_wait - t_current)
        if feats.b_d > 0:
            budgets_s.append(effective_tbt_slo - t_current)
        if not budgets_s:
            # No inference constraint this step (idle but for FT) → buffer cap.
            return coord_budget
        slo_x = min(self._estimator.max_next_ft_tokens(b) for b in budgets_s)
        return max(0, min(coord_budget, slo_x))
