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

After the three-regime estimator + iterative admission redesign
(see ``SLO_ESTIMATOR_REDESIGN.md``), this class is essentially a marker:

  * The phase gate (``coserving_admission_phase == "prefill"`` skips FT on
    decode-only steps) lives inside the parent's ``admit_ft_to_step`` and
    reads the config directly. No method override needed.
  * The decode-only safety margin (``decode_only_ft_safety_margin``, default
    0.7) is also applied inside ``admit_ft_to_step`` and is active only when
    ``phase == "both"`` AND the step is decode-only. Cold-start conservatism
    until the EAGER regime accumulates records for the
    ``(T_in=0, T_ft>0, B_d>0)`` shape; tune toward 1.0 after the offline
    profiling pass + ~256 online samples for that shape have landed.

Inherits everything else from ``FinetuneScheduler`` unchanged: injection,
coordinator wiring, backward triggers, async-safety (reserve-at-inject),
admission shapers (``match_prefill_workload_factor``,
``ft_tokens_admission_constrain_factor``), the ``_unspent_prefill`` counter,
the iterative ``admit_ft_to_step`` algorithm.

``match_prefill_workload_factor`` semantics in "both" mode: the counter still
only accumulates on prefill-carrying steps. On long decode-only stretches no
new credit accrues — but credit banked from prior prefill bursts can still
release FT samples onto decode-only steps as long as the leaky-bucket
trigger condition fires. Natural rate limiting without a separate "match
decode" knob.
"""

from vllm.deltaserve import dprint
from vllm.deltaserve.ft_scheduler import FinetuneScheduler


class BothPhaseFinetuneScheduler(FinetuneScheduler):
    """FT scheduler that admits to any step composition. See module docstring.

    Marker subclass of ``FinetuneScheduler`` — the parent's
    ``admit_ft_to_step`` handles the phase gate and decode-only safety
    margin internally by reading
    ``vllm_config.finetune_config.coserving_admission_phase`` and
    ``decode_only_ft_safety_margin``. This subclass exists so the config-time
    scheduler-class selection (in ``vllm/config/vllm.py``) can route to a
    distinct class name, keeping the existing wiring intact.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        ft_cfg = self.vllm_config.finetune_config
        margin = float(getattr(ft_cfg, "decode_only_ft_safety_margin", 0.7))
        dprint(
            f"[ft-sched] BothPhaseFinetuneScheduler active | "
            f"decode_only_safety={margin} | "
            f"max_tbt_slo={self._max_tbt_slo}s "
            f"(effective on decode-only: "
            f"{self._max_tbt_slo * margin:.4f}s)"
        )
