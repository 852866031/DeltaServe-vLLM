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

    backward_cuda_graph: bool = False
    """Capture the per-layer forward-recompute, FFN-backward, and padded-
    attention-backward sub-regions of each layer's manual SFT backward as CUDA
    graphs (forward + FFN bwd are one graph each per layer; attn bwd is ONE
    shared graph reused across all layers), keyed at the fixed activation-
    buffer width (max_saved_finetuning_tokens). Cuts host dispatch overhead on
    Llama-3's 32-layer backward. Eager fallback per-layer on capture or
    shape-fit failure; gradient values are bit-identical to the eager path
    (same math, just replayed). The forward graph falls back to eager for any
    layer whose saved gate||up is absent (e.g. ``save_activations: false``).
    _maybe_pause() is preserved and called between the FFN-bwd and
    attn-bwd graphs each layer, so the GPU-yield contract for the Phase-5
    pause and Phase-6 forward_interruptible paths is unaffected. Default
    off; mirrors DeltaServe's SFT_service_graph.py."""

    backward_cuda_graph_attn_bn_max: int = 8
    """[backward_cuda_graph] Padded-attention max distinct samples per backward.
    Batches with more samples silently fall back to the eager per-sample attention
    loop for that one backward. Mirrors DeltaServe's ATTN_BN_MAX (=8 default)."""

    backward_cuda_graph_attn_l_max: int = 64
    """[backward_cuda_graph] Padded-attention max per-sample sequence length.
    Batches whose max sample length exceeds this fall back to eager attention.
    Quadratic in compute (scores tensor is [bn_max, Hq, l_max, l_max]); oversizing
    tanks throughput. Mirrors DeltaServe's ATTN_L_MAX (=64 default)."""

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

    save_attn_qkv: bool = False
    """Also save the post-RoPE per-layer attention q/k/v tensors during the FT
    forward (one ``forward_pre_hook`` on each ``self_attn.attn`` module reads
    its ``(q, k, v)`` args). The backward then SKIPS the per-layer Q/K/V
    projection + RoPE recompute (the largest chunk of the attention-block
    recompute on Llama-3-8B — ~13 GFLOPs/layer, ~400-500 GFLOPs / backward at
    s_max=256). RMSNorm in_ln stays a recompute (cheap, needed for Q/K/V
    LoRA-A grad). Memory cost: ~99 MB at s_max=256 (qh [s_max, q_size] +
    kh/vh [s_max, kv_size] bf16 × 32 layers). Best perf/MB candidate of the
    remaining activation-save optimizations (see ``INTEGRATION_PROGRESS.md``
    Phase 5 "future activation-save optimization" section). Default off."""

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
    FT-only fills), where prefill is zero.

    Mutually exclusive with ``match_prefill_workload_factor`` — when the
    latter is > 0, this factor is ignored (the leaky-bucket strategy fully
    replaces the per-step proportional cap)."""

    match_prefill_workload_factor: float = 0.0
    """Alternate FT admission strategy: a leaky-bucket gate that defers FT
    admission until accumulated inference-prefill credit (scaled by this
    factor) covers the next FT sample's length.

    When ``> 0``, the scheduler maintains an internal counter
    ``_unspent_prefill``. On every prefill-carrying step (``t_in > 0``):
      * If ``(_unspent_prefill + t_in) * factor >= peek_next_ft_sample.input_len``
        AND the SLO budget is non-zero → admit that one FT sample (size capped
        by the SLO budget), reset the counter to 0.
      * Else → don't admit FT this step, accumulate
        ``_unspent_prefill += t_in``.
    FT-only / idle steps and decode-only steps follow the existing flow
    (full buffer fill on idle, skip on decode-only).

    The factor scales how much credit is "earned" per prefill token:
      * ``factor=1.0`` — 1 prefill token earns 1 FT-token of credit
        (reproduces the previous ``match_with_prefill_workload: True``
        behaviour exactly).
      * ``factor>1`` — more aggressive FT admission (each prefill token
        earns >1 credit). E.g. ``factor=2.0`` admits FT samples after only
        half the prior accumulation.
      * ``factor<1`` — more conservative (each prefill token earns <1
        credit). E.g. ``factor=0.5`` requires 2× the sample's length in
        accumulated prefill before admitting.
      * ``factor=0`` — disabled (default). ``0 * anything = 0`` never
        meets a positive sample length, so the trigger never fires.

    Any successful FT admission (this gate, FT-only fill, or any future
    path) resets the counter to 0, so credit is consumed atomically and
    can't be double-spent.

    The SLO gate still has the final say: when triggered, the admit budget
    is ``min(slo_budget, peek_sample.input_len)``. If SLO predicts no
    headroom (budget=0), the trigger fires but admission is 0; the
    counter retains its credit for a later step.

    Replaces ``ft_tokens_admission_constrain_factor`` when ``> 0`` — the
    proportional factor cap is bypassed since this gate controls the
    FT-per-prefill ratio directly. Default 0.0 keeps the previous
    factor-based behaviour."""

    coserving_admission_phase: str = "prefill"
    """Which step compositions are eligible for FT admission. Two values:

      * ``"prefill"`` (default) — FT only admits to prefill-carrying steps.
        Today's behaviour: decode-only steps short-circuit to budget=0 in
        ``FinetuneScheduler.schedule()``.
      * ``"both"`` — FT can admit to ANY step (prefill, decode, mixed,
        idle) subject to the SLO estimator's TTFT/max_TBT gate AND
        ``match_prefill_workload_factor`` (which still controls the
        FT-per-prefill ratio — the counter only accumulates on prefill-
        carrying steps, so long decode-only stretches naturally rate-limit
        FT). Selects ``BothPhaseFinetuneScheduler`` instead of the default.

    Forced to ``"prefill"`` (with warning at startup) when
    ``ft_tokens_admission_constrain_factor != -1`` — the proportional cap
    is defined relative to prefill tokens and doesn't apply to decode-only
    admission. Set the factor to ``-1`` to use ``"both"``."""

    decode_only_ft_safety_margin: float = 0.7
    """[both mode only] Multiplier applied to ``max_tbt_slo`` when computing
    the FT budget on DECODE-ONLY steps. Tighter than the prefill-carrying
    path because: (a) the eager penalty from losing the CUDA-graph fast path
    can dominate the sub-5ms decode-only step time, (b) the SLO estimator's
    γ coefficient hasn't seen many decode-only + FT samples until the online
    refit accumulates them. 0.7 (default) = the FT budget on a decode-only
    step is computed against 70% of the configured ``max_tbt_slo``. Set to
    1.0 to disable the extra margin (use the same TBT budget as prefill-
    carrying steps). Has no effect when ``coserving_admission_phase ==
    "prefill"`` (today's behaviour). Independent of the standard
    0.9 × ttft_slo TTFT margin, which still applies everywhere."""

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

    # --- Inference pre-emption of FT-only stepping (A + B + C tiers) ---
    forward_interruptible: bool = False
    """Master switch for the inference-preemption pipeline. When True, late-
    arriving inference requests can pre-empt FT-only stepping at three points:
    (A) before ``schedule()`` commits — a small blocking grace poll on the
    engine input queue when the next step would be FT-only; (B) between
    ``schedule()`` and ``execute_model()`` — if the produced batch is FT-only
    and an arrival landed in the meantime, the FT scheduling is rolled back
    and re-tried; (C) during the FT-only forward itself — the input-socket
    thread sets an abort event on each ADD while an FT-only batch is in
    flight, and the per-layer activation hooks raise a sentinel exception
    that ``execute_model`` catches and treats as an empty step. When False
    (default), every hook short-circuits and behaviour is bit-identical to
    today."""

    ft_only_admission_grace_ms: float = 2.0
    """[forward_interruptible] Tier-A grace window (ms) the engine main loop
    blocks on the input queue before scheduling what would be an FT-only
    step. Catches inference arrivals that landed in the window between the
    previous busy-loop drain and the upcoming schedule call. Cost is roughly
    ``ms / ft_forward_ms`` of FT throughput during idle; benefit scales with
    inference QPS. Set 0 to disable A while keeping B and C."""

    # --- debug / observability ---
    print_weight_hash: bool = False
    """Print + cross-process compare the base/FT-adapter weight hashes at startup."""

    print_activation_hash: bool = False
    """Print + cross-process compare the captured FT activation hashes (one-shot)."""

    print_step_mode: bool = False
    """Convenience master switch: when True, enables all four per-batch /
    per-request debug prints below (``print_scheduler_add``,
    ``print_engine_batch_exec``, ``print_engine_batch_done``,
    ``print_engine_req_recv``). Existing configs that set this flag continue
    to get the same combined output. Set the individual flags below instead
    if you want to enable just one or two of them."""

    print_scheduler_add: bool = False
    """[batch-sched] log: emitted right after ``scheduler.schedule()`` returns
    a non-decode-only batch (about to be handed to the executor). One per
    scheduled non-decode-only batch."""

    print_engine_batch_exec: bool = False
    """[batch] log: emitted inside the model runner's ``execute_model`` right
    before kernel launch / cudagraph replay, for non-decode-only batches.
    Includes per-sample decode KV sizes and whether the step ran as a graph."""

    print_engine_batch_done: bool = False
    """[batch-done] log: emitted on the engine main thread right after
    ``future.result()`` unblocks on GPU completion of a non-decode-only batch.
    Pairs with [batch-sched] / [batch] so per-batch GPU latency is visible."""

    print_engine_req_recv: bool = False
    """[engine-recv] log: emitted when the engine's main loop pulls a client
    request (ADD) off the input queue and adds it to the scheduler. Timestamp
    is when the MAIN thread drained the queue, not when the engine process
    received the ZMQ message."""
