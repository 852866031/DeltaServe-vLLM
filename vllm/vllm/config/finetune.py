# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Configuration for DeltaServe co-serving (inference + LoRA finetuning).

This is net-new on our fork. It is the analogue of DeltaServe's ``finetune.*``
YAML section. Phase 1 only needs ``enable_finetuning`` (the master gate that
decides whether the worker spawns the backward process). Later phases add the
data path, the dedicated FT LoRA adapter path, optimizer hyperparameters, the
activation-buffer token budget, and the SLO knobs.
"""

from vllm.config.utils import config


@config
class FinetuneConfig:
    """Knobs for the DeltaServe co-serving finetuning layer."""

    enable_finetuning: bool = False
    """Master switch for co-serving. When True, the GPU worker spawns the
    backward (SFT) process and sets up the shared activation buffers. When
    False (default), vLLM behaves exactly like upstream — no extra process,
    no overhead."""

    backward_mps_percentage: int = 10
    """CUDA MPS active-thread percentage granted to the backward (SFT) process.
    Applied as CUDA_MPS_ACTIVE_THREAD_PERCENTAGE only while spawning the child,
    so it inherits a constrained MPS partition and inference keeps the rest.
    Mirrors DeltaServe's model_rpc.py (=10). Requires the MPS daemon to be
    running to take effect."""

    finetuning_lora_path: str | None = None
    """Path to the dedicated finetuning LoRA adapter (PEFT format dir). This is
    the adapter the backward process trains. Its weights are loaded fp32 and
    shared with the backward process. Analogue of DeltaServe's
    finetune.lora_path / finetuning_adapter (model_rpc.py:134-146)."""

    data_path: str | None = None
    """Path to the finetuning corpus: one tokenizable sample per non-empty line.
    Loaded + tokenized at startup by the FinetuningStore."""

    num_epochs: int = 1
    """Number of passes over the finetuning corpus (FinetuningStore.advance_epoch)."""

    learning_rate: float = 1e-4
    """AdamW learning rate for the LoRA SFT backward (Phase 3). DeltaServe default 1e-4;
    the llama3 config sets 1e-5."""

    weight_decay: float = 0.0
    """AdamW weight decay for the LoRA SFT backward."""

    gamma: float = 1.0
    """StepLR multiplicative decay applied once per finetuning epoch (step_size=1)."""

    backward_fp32: bool = False
    """Run the backward's bulk matmuls (FFN-bwd, LoRA-grad, rope/proj-bwd) in fp32.
    Default False = the model dtype (bf16 for llama3), matching DeltaServe — lighter
    on the co-serving backward process. The load-bearing fp32 ops (attention scores/
    softmax, RMSNorm, LM head / final norm, fp32 LoRA master) are always fp32 either
    way. Set True for strictly more accurate gradients at ~2x backward matmul cost."""

    max_prepare: int | None = None
    """Cap on how many corpus lines to load (None = all). DeltaServe's
    finetune.prepare_size."""

    max_saved_finetuning_tokens: int = 256
    """Per-backward FT token budget. Also sizes the shared activation buffers and
    bounds length-bucketed sample selection."""

    save_activations: bool = True
    """Whether to actually accumulate FT activations into the shared buffers.
    Set False to measure the activation-save overhead: FT samples are still
    injected and the batch still runs eager, but the per-layer hook copies +
    final-hidden/targets copies are skipped (the coordinator still advances, so
    the control flow is identical). For A/B benchmarking only."""

    backward_sleep_seconds: float = 2.0
    """How long the (stub) backward process sleeps to simulate a backward pass
    after it consumes + cleans the activation buffer. Larger = easier to observe
    inference running while the backward is busy."""

    # --- Phase 4: SLO-aware admission + execution-time estimator ---
    ttft_slo: float = 1.0
    """Time-to-first-token SLO (seconds). The admission gate keeps the predicted
    step time within this for the earliest waiting inference request. Analogue of
    DeltaServe slo.ttft_slo."""

    avg_tbt_slo: float = 0.1
    """Average time-between-tokens SLO (seconds) for running decode requests."""

    max_tbt_slo: float = 0.2
    """Max time-between-tokens SLO (seconds): a single step's predicted time must
    not exceed this while FT tokens are admitted."""

    ft_tokens_admission_constrain_factor: float = -1.0
    """Cap FT tokens admitted per step relative to that step's INFERENCE PREFILL
    tokens. When > 0 and the upcoming step carries prefill (t_in > 0), admit at
    most ``t_in * factor`` FT tokens (on top of the SLO + buffer-space caps), so
    FT prefill can't dwarf the inference prefill it rides along with. When -1
    (default), this constraint is ignored — FT fills up to whatever the SLO
    estimator and buffer space allow. No effect on prefill-free steps (idle /
    FT-only fills), where prefill is zero."""

    profile_on_launch: bool = True
    """Run the offline execution-time profiling pass at launch (before serving)
    to seed the SLO estimator. When False, the estimator starts cold and relies
    on online refit."""

    start_on_launch: bool = True
    """Whether FT admission is open as soon as serving begins. True (default)
    keeps the original behaviour. False holds FT admission closed until a
    POST /start_finetuning HTTP call flips it on — used by the eval harness to
    start finetuning after warmup. (Independent of the launch profiling pass,
    which always runs and is unaffected.)"""

    profile_num_repeats: int = 2
    """Recorded passes per profiling shape (after one unrecorded warmup pass)."""

    batch_prediction_stats_path: str | None = None
    """If set, dump per-step predicted-vs-actual execution times to this CSV on
    FT exit (for offline estimator analysis). None disables."""

    bwd_log_path: str | None = None
    """If set, append one row per completed backward to this CSV (timestamp,
    epoch, batch_idx, batch_tokens, batch_loss, total_processed_tokens) — the
    finetune-throughput log the eval harness (eval/auto_plot.py) reads. None
    disables. Written from the coordinator when a backward reports done."""

    # --- debug / observability ---
    print_weight_hash: bool = False
    """Print + cross-process compare the base/FT-adapter weight hashes at startup."""

    print_activation_hash: bool = False
    """Print + cross-process compare the captured FT activation hashes (one-shot)."""

    print_step_mode: bool = False
    """Print each step's (has_ft, cudagraph_mode) when it changes — used to verify
    that non-finetuning batches run under a CUDA graph while FT batches run eager."""
