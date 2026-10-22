# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Engine-config defaults a co-serving run needs, applied once from
``VllmConfig.__post_init__`` (before the platform fills in its own defaults).

- the worker is ``DeltaServeWorker`` (the stock GPU worker + backward child,
  activation buffers, served-LoRA sharing);
- the scheduler is the FT-injecting ``FinetuneScheduler``, or its
  ``BothPhaseFinetuneScheduler`` sibling for ``coserving_admission_phase: both``;
- async scheduling defaults ON (the coordinator reserves activation-buffer rows
  at injection, so a step scheduled before the previous forward completes can
  neither overflow nor overlap the buffer) and is forced OFF for
  ``validate_estimator`` (its per-batch actuals need settled timing events).

Explicit user choices for the worker / scheduler class and async scheduling
are respected, except that ``validate_estimator`` always forces sync.
"""

from vllm.logger import init_logger

logger = init_logger(__name__)

WORKER_CLS = "vllm.deltaserve.worker.DeltaServeWorker"
SCHEDULER_CLS = "vllm.deltaserve.ft_scheduler.FinetuneScheduler"
BOTH_PHASE_SCHEDULER_CLS = (
    "vllm.deltaserve.ft_scheduler_both.BothPhaseFinetuneScheduler")


def configure_vllm_config(vllm_config) -> None:
    ft = vllm_config.finetune_config
    if not ft.enable_finetuning:
        return
    parallel = vllm_config.parallel_config
    if parallel.worker_cls == "auto":
        parallel.worker_cls = WORKER_CLS

    sched = vllm_config.scheduler_config
    if sched.scheduler_cls is None:
        # The proportional cap ft_tokens_admission_constrain_factor is defined
        # relative to prefill tokens, so it cannot drive decode-only admission:
        # with both set, fall back to prefill-only admission.
        phase = ft.coserving_admission_phase
        cap_factor = float(ft.ft_tokens_admission_constrain_factor)
        if phase == "both" and cap_factor != -1.0:
            logger.warning(
                "[deltaserve] coserving_admission_phase='both' is incompatible "
                "with ft_tokens_admission_constrain_factor=%s (must be -1); "
                "forcing 'prefill'.", cap_factor)
            phase = "prefill"
        sched.scheduler_cls = (BOTH_PHASE_SCHEDULER_CLS if phase == "both"
                               else SCHEDULER_CLS)

    if ft.validate_estimator:
        if sched.async_scheduling:
            logger.info("[deltaserve] validate_estimator=True — forcing "
                        "async_scheduling OFF for accurate per-batch "
                        "estimator validation.")
        sched.async_scheduling = False
    elif sched.async_scheduling is None:
        sched.async_scheduling = True
