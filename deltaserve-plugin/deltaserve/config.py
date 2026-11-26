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

    backward_mps_percentage: int = 0
    """CUDA MPS active-thread percentage granted to the backward (SFT) process,
    or 0 (default) for no MPS: the child is spawned as an ordinary CUDA context
    that the driver time-slices against inference, and the GPU-yield contract
    (``_maybe_pause`` + ``pause_until_prefill_done``) is what protects prefill
    latency. A value > 0 is applied as CUDA_MPS_ACTIVE_THREAD_PERCENTAGE only
    while spawning the child (inference keeps the rest) and takes effect only
    if an MPS control daemon is running (``nvidia-cuda-mps-control -d``) — it
    is a hard SM cap on the child even when inference is idle. DeltaServe's
    model_rpc.py used 10 under MPS; this project runs without a daemon."""

    finetuning_lora_path: str | None = None
    """Path to the dedicated finetuning LoRA adapter (PEFT format dir). This is
    the adapter the backward process trains. Its weights are loaded fp32 and
    shared with the backward process. Analogue of DeltaServe's
    finetune.lora_path / finetuning_adapter (model_rpc.py:134-146)."""

    adapter_save_path: str | None = None
    """Directory trained-adapter checkpoints are written under (PEFT format:
    ``adapter_model.safetensors`` with the fp32 masters, the source adapter's
    ``adapter_config.json``, and ``trainer_state.json``). The periodic saves
    below go to ``<adapter_save_path>/backward-<N>`` / ``epoch-<E>``; POST
    /save_adapter without a ``path`` to ``<adapter_save_path>/manual-<time>``.
    None (default): no periodic saves; POST /save_adapter needs a ``path``."""

    adapter_save_every_backwards: int = 0
    """Save the trained adapter after every N-th backward cycle (0 = off).
    Written after the cycle's done-reply, so FT admission is not held for it."""

    adapter_save_every_epoch: bool = False
    """Save the trained adapter at every finetuning-epoch boundary (the state
    after the last backward of the epoch, written as ``epoch-<E>``)."""

    adapter_save_keep_last: int = 0
    """Keep only the newest N periodic ``backward-*`` checkpoints (0 = keep
    all). Epoch and manual saves are never pruned."""

    data_path: str | None = None
    """Path to the finetuning corpus: one tokenizable sample per non-empty line.
    Loaded + tokenized at startup by the FinetuningStore."""

    ft_job: dict | None = None
    """Finetuning-job metadata (the ``ft_job:`` YAML section minus the keys
    that set a flat field above): ``name``, ``stages``, ``auto_routing``,
    ``adapter_init``. None = the implicit job, stages ``[init, training]``.
    Parsed by ``deltaserve.ft_job.FinetuneJobSpec``; not part of the
    compile-cache hash."""

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
    pause and Phase-6 forward_interruptible paths is unaffected. TP-safe
    (Phase 7 / M5): no collective is ever captured — the forward graph is
    split at the o-proj output and that all-reduce, plus the six
    backward-side reduces, run eagerly between the replays. Default
    off; mirrors DeltaServe's SFT_service_graph.py."""

    backward_cuda_graph_attn_bn_max: int | str = "auto"
    """[backward_cuda_graph] Padded-attention max distinct samples per backward.
    A backward with more samples runs its attention EAGER for that cycle (the
    rest stays graphed) — several thousand Python-issued kernels instead of one
    replay. ``auto`` (default) picks it, with ``_l_max``, from the finetuning
    dataset at launch (``deltaserve/bwd_budget.py``: the cheapest shape that
    holds 99 % of the epoch's backward batches); a number is taken as it is,
    and the launch log says how much of the data it holds. (The old default
    8 x 64 held no cycle of a 512-token buffer of ~20-token samples.)"""

    backward_cuda_graph_attn_l_max: int | str = "auto"
    """[backward_cuda_graph] Padded-attention max per-sample sequence length;
    ``auto`` as above. A backward with a longer sample falls back to eager
    attention. Quadratic in compute (the scores tensor is
    [bn_max, Hq, l_max, l_max])."""

    backward_cuda_graph_attn_shapes: int = 1
    """[backward_cuda_graph] How many padded-attention shapes to capture. One
    (the default) is the budget above alone: every backward pays for its whole
    padding (15 x 57 on the alpaca corpus with 256-token batches, when a batch
    is 13 samples of 19 tokens at the start of an epoch and 4 of 63 at its
    end). With more, smaller shapes INSIDE the budget are chosen from the
    dataset and a backward replays the cheapest one that holds it. The
    attention graph is shared by all layers, so a shape costs its input slabs
    (a few MB) and one graph, not a graph per layer. Qwen3-14B TP=2: the
    attention backward of a cycle 8.6 ms -> ~4.2 ms."""

    backward_disable_torch_native: bool = True
    """Start the backward child with ``TORCH_DISABLE_NATIVE_JIT=1``. torch 2.14
    added ``torch/_native``: Python-level overrides of bmm / sum / rmsnorm /
    scatter_add / topk ... on CUDA that run a dispatch check before every
    call. The eager parts of the backward issue thousands of small calls per
    cycle; on a host with a slow CPU that check cost 13-18 % per token (A100
    cloud VM, Qwen3-14B TP=4: 2111 -> 1764 µs/token with it off, vs 1801 on
    torch 2.11). Only the child is affected — the variable is set around its
    spawn, the serving processes keep torch's default — and for the child it
    is the torch <= 2.13 behaviour. False: leave torch's default."""

    max_prepare: int | None = None
    """Cap on how many corpus lines to load (None = all). DeltaServe's
    finetune.prepare_size."""

    max_saved_finetuning_tokens: int = 256
    """Per-backward FT token budget. Also sizes the shared activation buffers and
    bounds length-bucketed sample selection. The single-size form of
    ``ft_step_size`` / ``bwd_batch_size`` below: with neither set, both are this
    value (one backward per optimizer step, as before they existed)."""

    ft_step_size: int | None = None
    """FT tokens per optimizer step (None = ``max_saved_finetuning_tokens``).
    Sizes the shared activation buffers: the forward collects a whole step's
    activations, the backward child works through them ``bwd_batch_size`` tokens
    at a time, accumulating gradients, and updates + publishes the adapter once
    per step. The gradients are normalised over the step's tokens, so k
    backward batches equal one backward of k times the size."""

    bwd_batch_size: int | None = None
    """FT tokens per backward batch (None = the step size: one backward per
    step). Must divide ``ft_step_size``. It sets how long the backward child
    holds the GPU between two points where it can yield (and how much it slows
    the decode steps it overlaps), the size of the backward graphs, and the
    longest sample that can be trained. With several batches per step FT
    admission stays open while the earlier batches run (the weights do not
    change until the step's update), and a step that carries FT tokens pauses
    the child like an inference prefill does."""

    ft_buffer_size: int | None = None
    """Rows of the activation buffer when it is SMALLER than the step (None =
    ``ft_step_size``). A multiple of ``bwd_batch_size``; needs several
    backward batches per step. The buffer then is a ring of batch slots: batch
    j of a step is filled in slot j mod (ft_buffer_size / bwd_batch_size), and
    a slot is given to a later batch once its backward is acked — nothing is
    allocated or freed while serving, the forward waits when every slot is
    still being trained. The buffer's memory follows this size, not the step:
    Qwen3-14B TP=2, 2.70 MB per token per GPU, so a 1024-token step in a
    512-token buffer is 1.3 GiB instead of 2.6, and a 4096-token step costs no
    more. Not together with ``bwd_batch_size_idle`` (a larger backward needs
    its rows in one piece).
    Measured (2026-11-26, production timeline, 0.88): 1024 / 256 with the whole
    step buffered -> in 512 rows: FT 624 -> 659 tok/s, FT only 1276 -> 1286,
    KV cache 53,248 tokens; 2048 / 256 in 512 rows: 670 tok/s, the same KV.
    Off by default (the buffer holds the step), as before the key."""

    bwd_batch_size_idle: int | None = None
    """FT tokens per backward batch while NO inference request is running or
    waiting (None = ``bwd_batch_size``: off). Up to ``ft_step_size``. A larger
    backward costs less per token; with this the child keeps the small batch
    (a short yield, a small share of a decode step's time) whenever inference
    is present and takes what the step holds, up to this size, when it is not.
    The backward graphs are padded to one batch size, so the child captures a
    second set for this size and needs the memory of the larger backward
    (Qwen3-14B TP=2: ~2.25 GiB for 256, ~5.2 GiB for 1024) on top.
    OFF BY DEFAULT: MEASURED, WORSE THAN ONE LARGE BACKWARD ALWAYS. Qwen3-14B
    TP=2, 1024 / 256 without -> with an idle size of 1024: FT only 1276 -> 1619
    tok/s, production 624 -> 708, loose-half 882 — but
    ``max_saved_finetuning_tokens: 1024`` with ``ft_tokens_per_step: 256``
    (preset ..._tp2_step1024) gives 1877 / 795 / 922 with a KV cache of 17,408
    tokens, against 6,144 here (two graph sets in 32 GB). For a deployment
    that must keep the small backward while inference is present."""

    ft_tokens_per_step: int | None = None
    """FT tokens ONE forward step may carry (None = ``bwd_batch_size``: a
    step fills a whole backward batch when nothing else limits it). Smaller
    than the backward batch, it separates the two costs of a large batch: the
    backward of 1024 tokens is far cheaper per token than four of 256 (bigger
    GEMMs, one LM head, one publish), and it yields between its per-layer
    graphs anyway, while a FORWARD step of 1024 FT tokens is 200 ms that an
    arriving request waits behind or has to abort. With ``ft_step_size`` =
    ``bwd_batch_size`` = 1024 and this at 256, the buffer fills over four
    steps and trains in one backward."""

    ft_only_tokens_per_step: int | None = None
    """FT tokens a step WITHOUT inference tokens may carry (None =
    ``ft_tokens_per_step``; at most a backward batch). An FT-only forward is
    abandoned when a request arrives (``forward_interruptible``), so it can be
    large — a 1024-token one costs less per token than four of 256 — while a
    step that also carries inference tokens keeps the smaller cap."""

    bwd_whole_batches: bool = False
    """With several backward batches per step: end the step when its
    ``ft_step_size / bwd_batch_size`` batches are full, not when its rows are.
    A batch is whole samples, so filling the rows leaves a remainder (1024 rows
    of ~19-token samples: four batches of 247 and 36 rows over) that becomes a
    fifth backward of a few dozen tokens at the price of a whole one — the
    backward graphs are padded to the batch size. On: no such batch; a step
    trains slightly fewer tokens. False: the behaviour before this key."""

    @property
    def step_tokens(self) -> int:
        """Tokens per optimizer step = the activation buffer's capacity."""
        return max(1, int(self.ft_step_size if self.ft_step_size is not None
                          else self.max_saved_finetuning_tokens))

    @property
    def bwd_tokens(self) -> int:
        """Tokens per backward batch."""
        return max(1, int(self.bwd_batch_size if self.bwd_batch_size is not None
                          else self.step_tokens))

    def check_step_sizes(self) -> None:
        step, bwd = self.step_tokens, self.bwd_tokens
        if bwd > step or step % bwd:
            raise ValueError(
                f"finetune_param.ft_step_size ({step}) must be a multiple of "
                f"finetune_param.bwd_batch_size ({bwd})")
        buf = self.ft_buffer_size
        if buf is not None and (int(buf) < bwd or int(buf) % bwd
                                or int(buf) > step):
            raise ValueError(
                f"finetune_param.ft_buffer_size ({buf}) must be a multiple of "
                f"bwd_batch_size ({bwd}) and at most ft_step_size ({step})")
        if (buf is not None and int(buf) < step
                and self.bwd_batch_size_idle not in (None, bwd)):
            raise ValueError(
                "finetune_param.ft_buffer_size below the step and "
                "bwd_batch_size_idle cannot be combined")
        idle = self.bwd_batch_size_idle
        if idle is not None and not (bwd <= int(idle) <= step):
            raise ValueError(
                f"finetune_param.bwd_batch_size_idle ({idle}) must lie between "
                f"bwd_batch_size ({bwd}) and ft_step_size ({step})")

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
    remaining activation-save optimizations (see ``INTEGRATION_PROGRESS.md`` on
    the ``legacy`` branch, Phase 5 "future activation-save optimization"). Families with a
    per-head q/k transform before RoPE (Qwen3's q/k-norm) capture q/k at the
    transform's INPUT instead (pre-hooks on ``self_attn.{q,k}_norm``) so the
    norm backward has what it needs; the backward re-applies only the
    elementwise norm + RoPE. Default off."""

    save_attn_ctx: bool = False
    """Also save the per-layer attention CONTEXT output (the input to
    ``o_proj``, i.e. the result of ``self_attn.attn(q, k, v)``) during the FT
    forward — one ``forward_hook`` on each ``self_attn.attn`` module reads its
    output. The backward then SKIPS the attention-forward recompute (the
    per-sample scores/softmax/AV loop, or the captured padded-attention forward
    in Graph C) and reads the saved ``ctx`` directly as the O-projection
    backward input. The Q/K/V tensors are still needed by the attention
    BACKWARD, so this composes with ``save_attn_qkv``: with both on, the
    per-layer forward recompute collapses to RMSNorm in_ln + O-proj + residual
    (everything else is read from saved buffers). Memory cost: ~33 MB at
    s_max=256 (ctx [s_max, q_size] bf16 × 32 layers). Default off."""

    save_resid_mid: bool = False
    """Also save the per-layer POST-ATTENTION residual (``layer_in + o_proj
    output`` = the FFN block's input, ``resid_mid`` in the backward) during
    the FT forward — one ``forward_pre_hook`` on each
    ``post_attention_layernorm`` reads its ``(hidden, residual)`` args, the
    same fused add-norm idiom the ``layer_in`` hooks use. The backward then
    SKIPS the O-projection recompute (base + LoRA GEMM, ~8.6 GFLOPs/layer on
    Llama-3-8B at s_max=256) and the residual add; the attention context is
    still produced for the O-proj BACKWARD. **Under tensor parallelism this
    also removes the forward remat's ``o`` all-reduce** — the only collective
    in the forward recompute — so the per-layer cross-rank traffic drops from
    7 to 6 collectives (the saved value is the already-reduced full residual
    vLLM's RowParallelLinear produced). With ``save_attn_qkv`` + ``save_attn_ctx``
    + this on, the per-layer forward recompute is RMSNorm in_ln alone. Memory
    cost: one [s_max, hidden] buffer per layer — ~64 MB at s_max=256 on
    Llama-3-8B (bf16 × 32 layers), ~100 MB on Qwen3-14B. Default off."""

    # --- Phase 4: SLO-aware admission + execution-time estimator ---
    ttft_slo: float = 1.0
    """Time-to-first-token SLO (seconds). The admission gate keeps the predicted
    step time within this for the earliest waiting inference request. Analogue of
    DeltaServe slo.ttft_slo."""

    avg_tbt_slo: float = 0.1
    """Average time-between-tokens SLO (seconds) for running decode requests:
    THE token-latency target. Enforced over a sliding window
    (``avg_tbt_window_s``), and what the TBT attainment is measured against (a
    request meets it when the mean of its own token gaps is within it)."""

    avg_tbt_window_s: float = 1.0
    """How ``avg_tbt_slo`` is enforced: FT is admitted to a step with decode
    requests only while the average time between tokens over the last
    ``avg_tbt_window_s`` seconds, that step included, stays within
    ``avg_tbt_slo`` (``deltaserve/tbt_window.py``: every decode token of the
    window, averaged). Default 1 s. 0 = off: only the per-step ``max_tbt_slo``
    bound runs and ``avg_tbt_slo`` is not enforced."""

    avg_tbt_pause_backward: bool = True
    """Keep ``avg_tbt_slo`` against the backward too. A decode-only step that
    shares the GPU with a running backward takes about twice as long, and FT
    admission has no say in it: the backward was triggered earlier. With this
    on, the scheduler predicts such a step WITH the overlap (the estimator's
    decode-with-backward regime) and, if the window's average time between
    tokens would then pass ``avg_tbt_headroom`` × ``avg_tbt_slo``, the step
    pauses the backward child, exactly as an inference prefill does. The
    backward so uses only the slack the target leaves and gives way when it is
    gone. Needs ``avg_tbt_window_s`` > 0. False: the backward overlaps decode
    steps freely (the behaviour before this key)."""

    avg_tbt_headroom: float = 0.9
    """The fraction of ``avg_tbt_slo`` the backward gate above steers the
    window average to. Below 1 because the SLO is met per request: with the
    window average ON the target, about half the requests' own averages are
    above it."""

    profile_extra_shapes: bool = False
    """Add to the launch profiling pass the shapes served traffic has and the
    base sweep leaves out: one short prompt prefilled alone (16-48 tokens) and
    many decode requests with short contexts (8-64 requests of 48 / 160
    tokens). ~10 shapes, 2 s. OFF BY DEFAULT: MEASURED, NO GAIN — the estimator
    is refit on live steps while serving, so what it predicts at run time does
    not depend on the launch sweep (Qwen3-14B TP=2, actual / predicted with
    and without: prefill + decode steps 1.02-1.08 vs 0.98-1.04, decode-only
    with 17-48 requests 0.82 vs 0.84, a lone short prefill 1.13 vs 1.10). The
    base sweep is enough; what is left is the model's form, not its data.
    False: the sweep before this key."""

    overload_hold_s: float = 0.0
    """FT stands down while inference is at or over its SLO. When the window's
    average time between tokens is above ``avg_tbt_slo``, or the oldest
    waiting inference request has waited 0.9 x ``ttft_slo``, no FT is admitted
    and the backward stays paused until this many seconds after the last such
    moment. The per-step checks only ask whether one more step fits, so
    without this FT keeps a share of a system that cannot meet its SLO even
    alone (Qwen3-14B TP=2 at an overloading request rate: TTFT attainment 96 %
    inference only, 81.5 % with FT at 275 tok/s). 0 = off, the behaviour
    before this key."""

    avg_tbt_admission_headroom: float = 1.0
    """The fraction of ``avg_tbt_slo`` FT ADMISSION steers the window average
    to (the check that decides how many FT tokens join a step with decode
    requests). 1.0 = the target itself (the behaviour before this key); below
    1 for the same reason as ``avg_tbt_headroom``."""

    max_tbt_slo: float = 0.1
    """Upper bound on ONE step that carries decode requests and FT tokens
    (seconds, predicted). A backstop, not the target: a step that carries an
    inference prefill next to decodes is slow whatever FT does, so a tight
    value here mostly shuts FT out of the steps it rides best. The average is
    what ``avg_tbt_slo`` keeps."""

    ft_tokens_admission_constrain_factor: float = -1.0
    """Cap FT tokens admitted per step relative to that step's INFERENCE PREFILL
    tokens. When > 0 and the upcoming step carries prefill (t_in > 0), admit at
    most ``t_in * factor`` FT tokens (on top of the SLO + buffer-space caps), so
    FT prefill can't dwarf the inference prefill it rides along with. When -1
    (default), this constraint is ignored — FT fills up to whatever the SLO
    estimator and buffer space allow. No effect on prefill-free steps (idle /
    FT-only fills), where prefill is zero.

    Three-regime estimator interaction: under the iterative
    ``admit_ft_to_step`` algorithm, this factor is a hard upper bound on
    ``T_ft`` per step. The loop's break condition becomes
    ``tentative.T_ft + candidate.input_len > min(buffer_cap, t_in * factor)``.

    Composes with ``match_prefill_workload_factor``. When both shapers
    are enabled (factor here > 0 AND match factor > 0), the leaky
    bucket decides WHEN to admit (a triggered step admits exactly one
    sample) and this proportional factor decides WHETHER the chosen
    sample's ``input_len`` fits under ``t_in * factor`` BEFORE it
    actually lands in the batch. If it doesn't fit, the leaky-bucket
    trigger fires but admission is 0 for the step; credit isn't
    consumed (the counter resets only on a *successful* admit), so a
    later step with a larger ``t_in`` cap can carry the sample
    through."""

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

    Composes with ``ft_tokens_admission_constrain_factor``. When both
    are enabled the leaky bucket fires the WHEN (trigger when credit
    is enough) and the proportional factor fires the WHETHER (the
    triggered sample must fit under ``t_in * factor``). Earlier
    versions made these mutually exclusive — the proportional cap
    was bypassed when the leaky bucket triggered — which let a
    triggered sample that was much larger than the step's prefill
    inflate the batch beyond the proportional rule. The current
    behaviour gates the triggered sample on the proportional cap
    too; if it doesn't fit, the step admits 0 and the bucket
    retains its credit for a later (larger-prefill) step.

    Three-regime estimator interaction: under the iterative
    ``admit_ft_to_step`` algorithm, the leaky bucket gates entry to the
    per-sample loop. If the trigger condition holds, the loop is capped at
    a single iteration (admit exactly one sample sized to the SLO budget).
    If the trigger condition fails, the loop is skipped entirely and
    ``_unspent_prefill`` accumulates by ``t_in``. The counter resets on
    any successful admit."""

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
    """[both mode only] Multiplier applied to ``max_tbt_slo`` when the
    iterative ``admit_ft_to_step`` evaluates a per-sample candidate on a
    DECODE-ONLY step. The per-sample SLO check uses ``max_tbt_slo * margin``
    instead of ``max_tbt_slo`` for that step's acceptance gate.

    Cold-start conservatism: the three-regime EAGER coefficient set predicts
    the FULL→eager transition cost more accurately than the previous
    closed-form solve (it sees the full step-with-FT directly), but the margin
    gives a safety buffer while the EAGER regime accumulates records for the
    ``(T_in=0, T_ft>0, B_d>0)`` shape. The offline profiling pass now sweeps
    this shape (see ``_ft_on_decode_shapes`` in
    ``profiling_batch_generator.py``), so the margin can be tuned toward 1.0
    once GPU A/B runs validate the estimator's per-regime RMSE on this shape.

    0.7 (default) = the FT budget on a decode-only step is computed against
    70% of the configured ``max_tbt_slo``. Set to 1.0 to disable the extra
    margin entirely. Has no effect when ``coserving_admission_phase ==
    "prefill"`` (decode-only steps short-circuit to no FT regardless).
    Independent of the standard 0.9 × ttft_slo TTFT margin, which still
    applies everywhere."""

    admission_margin: str = "relative"
    """How the step-time prediction is inflated before the SLO admission checks.

    ``"relative"`` (default): ×(1 + ``admission_margin_k`` · relRMSE), relRMSE
    the regime's RMS relative error on its fit data (e.g. 0.033 = 3.3 %): the
    headroom scales with the step. On Qwen3-14B TP=2 it cut FT-carrying steps
    over ``max_tbt_slo`` from 10 % to 2 % for -1 % FT (README, "The admission
    margin"; ``eval-tp/margin_ab.py``).
    ``"legacy"``: ×(1 + 1.5·RMSE) with the RMSE in seconds, as in the original
    DeltaServe tracker. With the RMSEs seen in practice (a few ms) this is a
    0.2-0.8 % factor, i.e. effectively no margin. The presets pin it, so they
    reproduce the published runs.
    ``"absolute"``: + ``admission_margin_k`` · RMSE seconds, the same headroom
    for every step.
    Applies to every margined prediction (the baseline-without-FT checks and
    each candidate in the iterative admission loop)."""

    admission_margin_k: float = 1.5
    """Multiplier of the error in ``admission_margin`` = relative / absolute
    (ignored by ``legacy``, which keeps its fixed 1.5)."""

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

    validate_estimator: bool = False
    """Estimator-accuracy validation mode. When True: (1) forces async
    scheduling OFF (so the deferred CUDA-event ring's prior slot is always
    settled → accurate per-batch actuals), and (2) appends one
    (predicted, actual, features) row per scheduled batch to
    ``estimator_validation_path`` as the drain loop pairs each step's features
    with its measured GPU time. The mode-"w" end-dump to
    ``batch_prediction_stats_path`` is SKIPPED in this mode (it would clobber
    the appended file). The logged ``predicted_duration`` is the RAW model
    prediction — the ×(1 + 1.5·RMSE) safety margin is NOT applied (admission
    still uses the margined value); this scores the model itself. Zero added
    runtime cost when False (a single bool check on the drain path). Requires
    ``enable_finetuning=True``. Note: the last ~RING (=4) batches of a run are
    never drained, so they're absent from the CSV — a sub-1% tail with no
    effect on accuracy stats."""

    estimator_validation_path: str | None = None
    """Output CSV for ``validate_estimator`` — per-batch rows appended (header
    on first write). Falls back to ``batch_prediction_stats_path`` when None;
    if both are None, validation rows are dropped (warned once). Intended
    location: eval/estimator/."""

    step_trace_path: str | None = None
    """Per-step predicted-vs-actual trace for estimator validation. When set,
    the FT scheduler writes one CSV row per timed step (composition features,
    the RAW and margined predictions, the admission-time baseline / accepted
    prediction, the CUDA-event GPU time, the host dispatch time, the
    backward-in-flight / paused flags, scheduler occupancy) plus one row per
    rolled-back FT-only step. Unlike ``validate_estimator`` it does NOT change
    the system under test (async scheduling stays as configured) and it costs
    nothing on the hot path beyond a few float formats per step: rows are
    formatted where the timing sample is drained (already off the forward's
    critical path) and handed to a daemon writer thread that owns the file.
    None (default) disables everything (one ``is None`` check per step).
    Analyse with ``eval-tp/analyze_step_trace.py``."""

    bwd_log_path: str | None = None

    metrics_dir: str | None = None
    """[telemetry] Record system metrics into this directory (see
    ``deltaserve/metrics.py``): ``requests.csv`` (per request TTFT, TBT,
    latency, tokens), ``backward.csv`` (per LoRA backward GPU / wall time,
    tokens, loss, child memory), ``summary.jsonl`` (every
    ``metrics_interval_s``: TTFT / TBT percentiles + SLO attainment, inference
    and FT throughput, GPU time by step kind, memory) and
    ``memory_rank<r>.json`` (per GPU: weights, KV cache, FT buffers, other).
    Also served live by GET /system_metrics. None (default): off, nothing
    recorded."""

    metrics_interval_s: float = 10.0
    """[telemetry] Seconds per ``summary.jsonl`` window (``metrics_dir``)."""

    steps_log_max_mb: float = 256.0
    """[telemetry] Size cap of the per-step log ``steps.<n>.jsonl.gz`` in
    ``metrics_dir`` (every step's composition, GPU time and FT admission,
    plain decode runs folded): rolling segments, the oldest deleted beyond
    the cap. 0 = no step log (the dashboard's live stream still works)."""

    estimator_state_save_path: str | None = None
    """[estimator] JSON file the SLO estimator's state (fitted per-regime
    coefficients + the recorded step samples it refits from) is written to at
    shutdown. Lets a long trace be replayed as separate runs — each run
    continues from the estimator state the previous one ended with."""

    estimator_state_load_path: str | None = None
    """[estimator] JSON file written via ``estimator_state_save_path`` to
    restore at launch. The launch-time profiling pass still runs (it also warms
    up the FT paths), then the restored state REPLACES what it fitted — the
    saved state already contains the profiling samples of the run it came
    from. Ignored with a warning if the file does not exist (first part)."""
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
    today. Under tensor parallelism tier C is rank-symmetric: the engine
    publishes an arrival counter in shared memory and every rank MAX-reduces
    its local decision over the TP group's gloo group at entry and at each
    layer boundary, so all ranks leave the FT-only forward at the same layer
    (one CPU collective per layer; never for co-serving batches)."""

    ft_only_admission_grace_ms: float = 2.0
    """[forward_interruptible] Tier-A grace window (ms) the engine main loop
    blocks on the input queue before scheduling what would be an FT-only
    step. Catches inference arrivals that landed in the window between the
    previous busy-loop drain and the upcoming schedule call. Cost is roughly
    ``ms / ft_forward_ms`` of FT throughput during idle; benefit scales with
    inference QPS. Set 0 to disable A while keeping B and C."""

    backward_run_ahead_boundaries: int = 2
    """[pause] How far the backward child's CPU may run ahead of its GPU, in
    ``_maybe_pause`` boundaries (roughly half a layer each on the graphed
    path). At every boundary the child records a CUDA event and waits for the
    event recorded this many boundaries earlier, so at most that much work is
    queued on the GPU when a pause request lands — the pause then takes
    effect within ~1-2 boundaries (3-6 ms) instead of after everything the
    child had already enqueued (the graphed loop launches a whole cycle in a
    few ms; measured on Qwen3-14B TP=2: paused prefills still ran 1.15-2.3×).
    0 restores the unbounded behaviour. No GPU idle as long as a boundary's
    GPU time exceeds its launch cost, which holds for every real model."""

    pause_on_arrival: bool = False
    """[pause] The backward child yields the GPU as soon as an inference
    request ARRIVES during a cycle, not only once its prefill is dispatched.
    Without it, decode-only steps share the GPU with the child (by design) and
    run 10-20x slower right after a cycle starts (17 → 180-330 ms on Qwen3-14B
    TP=2); a request arriving then waits behind one or two of them before its
    prefill can launch (production: the same burst request at 388-419 ms, around
    the 0.4 s TTFT SLO). With it the child pauses at its next boundary (the
    run-ahead bound keeps that within a few ms) and resumes once the worker's
    usual prefill pause for that request is over — or after 100 ms if no
    prefill follows. Reads the engine's arrival counter (shared memory, bumped
    by the input thread on every ADD); rank-symmetric under TP like every
    pause."""

    def compute_hash(self) -> str:
        """[mixed-fwd-cuda-graph] The finetuning fields that change the traced
        forward: with ``graph_ft_batches`` the accumulator's save-point hooks
        trace to the save op (one per enabled save), without it they trace to
        nothing — same source, different graph. vLLM's compile cache is keyed
        on this hash (``VllmConfig.compute_hash``); without it a run with the
        feature toggled loaded the other configuration's AOT-compiled graph
        (KeyError on the marker buffer at best, silently missing saves at
        worst)."""
        import hashlib

        factors = [
            bool(self.enable_finetuning), bool(self.graph_ft_batches),
            bool(self.save_activations), bool(self.save_attn_qkv),
            bool(self.save_attn_ctx), bool(self.save_resid_mid),
        ]
        return hashlib.sha256(repr(factors).encode()).hexdigest()

    graph_ft_batches: bool = False
    """[mixed-fwd-cuda-graph] Let MIXED batches (inference requests + FT
    samples) run vLLM's compiled forward and replay piecewise CUDA graphs
    instead of the Python eager path. The activation saves then go through
    the graph-capturable ``dserve_save_rows`` op (``deltaserve/ft_save_op.py``)
    fed by persistent index tensors, and a second piecewise capture set keyed
    ``has_ft`` is captured at startup. FT-ONLY batches stay eager (they are
    the ones the mid-forward abort applies to; ``graph_ft_only_batches`` puts
    them on the graphs too). The op is issued by the accumulator's hooks, so
    it works for every family (Qwen3 and Llama-3 are both gated:
    ``tests/test_phase1_m1.py``, ``tests/test_llama3_inference_leak.py``).
    Off (default) is bit-identical to before."""

    graph_ft_max_tokens: int = 512
    """[mixed-fwd-cuda-graph] Largest padded batch size to capture the
    ``has_ft`` piecewise graphs for (capped by the compilation config's
    ``max_cudagraph_capture_size``). Mixed FT steps above it run the compiled
    forward without a graph."""

    ft_breakable_cudagraph: bool = False
    """Serve on vLLM's breakable CUDA graph (what ``VLLM_USE_BREAKABLE_CUDAGRAPH=1``
    selects: the forward is recorded as it runs, no ``torch.compile`` split)
    and capture the ``has_ft`` graph set on it — the hooks' save op is executed
    during the capture pass, so no Dynamo trace is involved, and the tier-C
    abort poll runs between the replayed segments.
    OFF BY DEFAULT: MEASURED, NO GAIN. Qwen3-14B TP=2, compiled -> breakable:
    production FT 595 -> 589 tok/s, TTFT p50 / p99 72.9 / 112.5 -> 74.5 / 113.2
    ms, FT-only step 53.5 -> 53.2 ms, graphs 4.18 -> 4.26 GiB (2026-11-26). It
    runs the same kernels; kept for a vLLM that makes the breakable graph the
    default for these models. With this off and the env variable set by hand,
    vLLM serves on the breakable graph and FT batches run without a graph
    (the behaviour before this key)."""

    fill_graph_padding: bool = False
    """A step that replays a ``has_ft`` graph is padded to a captured size and
    pays for all of its rows. With this on, an FT sample the SLO check would
    refuse is admitted anyway while it fits the rows of the size the step
    already runs at (inference tokens + the FT admitted so far, rounded up to
    the next captured size): it adds no kernel work, only its activation
    saves. Applies once the step carries FT (the first sample still has to
    pass the SLO check). Requires ``graph_ft_batches``. Off: the behaviour
    before this key."""

    graph_ft_only_batches: bool = False
    """FT-ONLY steps (no inference tokens) replay the same ``has_ft`` graphs as
    mixed steps instead of running eager, where they are host-bound (~63 ms of
    dispatch for a ~66 ms step on Qwen3-14B TP=2). Requires ``graph_ft_batches``.
    With ``forward_interruptible`` the tier-C abort still lands mid-forward: the
    abort poll runs between the piecewise graph segments (one per layer) and the
    poller's launch-ahead bound keeps the CPU within a few layers of the GPU, so
    a decision reflects GPU progress."""

    pause_prefill_always: bool = False
    """[pause] Clear the backward child's GPU grant on EVERY prefill-carrying
    step, not only while a backward is outstanding. Closes the TP trigger
    race (the relayed trigger fires while the previous step is still on the
    GPU; that step decided "nothing to pause" and ran 2-3x against the fresh
    child). Measured on Qwen3-14B TP=2 with mixed FT batches graphed: the
    prefills after a mixed step are fixed (0 slow), but the child then only
    runs in decode-only windows during bursts — backward cycles 150 → 540-630
    ms median (max 10-25 s) and FT throughput 2-3x lower — because the race
    had been giving the child its first ~100 ms of every cycle unpaused.
    Off (default) keeps that behaviour; on protects the in-flight prefill at
    the cost of FT throughput until the child has a deliberate GPU share."""

    pause_until_prefill_done: bool = True
    """Keep the backward child paused until the inference prefill that paused
    it has actually COMPLETED on the GPU (default). ``False`` re-sets the GPU
    grant right after the prefill kernels are *enqueued* — fire-and-forget,
    only adequate under MPS where the child's SM share is bounded anyway.
    Without an MPS daemon (the default here) the child would then resume while
    the prefill is still executing and the two contexts time-slice, roughly
    doubling the prefill's TTFT (measured on the tight trace: burst-start TTFT
    80-200 ms vs 40 ms inference-only; 55-67 ms with this on).
    When True the runner records a CUDA event after the forward and resumes
    the child only once that event has completed (checked without blocking on
    the next execute_model / sample_tokens call; idle steps keep the engine
    stepping while a backward is outstanding, so the check always runs). Cost:
    the child yields for the whole prefill instead of one layer."""

    model_runner: str = "auto"
    """Which vLLM model runner hosts the co-serving hooks: ``v1``
    (``gpu_model_runner.py``; ``deltaserve/model_runner.py``), ``v2``
    (``vllm/v1/worker/gpu``; ``deltaserve/model_runner_v2.py``) or ``auto``
    (default): v1 where vLLM still ships it (up to v0.33), else v2. vLLM
    removed the v1 runner after v0.33, so v2 is the only choice from v0.34 on;
    on v0.33 both exist, which is what lets the two be compared on one vLLM."""

    # --- fwd-token throttle (extends the prefill-gated pause) ---
    fwd_token_throttle_enable: bool = False
    """Master switch for the fwd-token-throttle pause source. When False
    (default), pause decisions stay prefill-only (today's behaviour:
    pause when ``feats.t_in > 0`` and a backward is pending). When True,
    the pause condition is extended to ALSO fire when the current
    inference batch's total token count reaches ``fwd_token_throttle``
    — see that field for the math. The threshold is read every step,
    so toggling the bool at runtime takes effect on the next batch."""

    fwd_token_throttle: int = 1024
    """[fwd_token_throttle_enable] Token-count threshold for the new
    pause source. The pause decision becomes::

        pause = pending_backward and (
            feats.t_in > 0                           # today's rule
            or (enable and total >= fwd_token_throttle)  # new rule
        )

    where ``total = prefill + ft + len(decode_kv)`` — the same value
    the engine's ``[batch …] (total=N)`` log line shows for each
    step. Resume happens for free as soon as both conditions go
    False (no prefill AND total at-or-below the threshold), so a
    transient burst un-pauses backward within microseconds of the
    next step that drops back under the bound. Ignored when
    ``fwd_token_throttle_enable`` is False; the field is still read
    so the value can sit in a YAML and be flipped on/off via the
    bool without editing the number."""

    # --- rps-throttle (closes FT admission under bursts) ---
    rps_throttle_enable: bool = False
    """Master switch for the inference-RPS-based admission gate.
    When True, the FT scheduler closes ``coord.admission_open`` once
    the inference arrival rate (sliding window) crosses
    ``rps_throttle_close_rps``, and re-opens it once the rate drops
    below ``rps_throttle_open_rps``. Distinct from
    ``fwd_token_throttle_enable``: that one pauses the running
    backward; this one denies NEW FT samples from entering the
    activation buffer. The two stack — together they form 'back off
    FT in all directions during bursts'."""

    rps_throttle_close_rps: float = 50.0
    """[rps_throttle_enable] Inference arrival RPS above which the
    throttle ENGAGES (sets ``admission_open = False``). Measured as
    the count of requests entering the engine's input_queue in the
    last ``rps_throttle_window_s`` seconds, divided by the window
    width."""

    rps_throttle_open_rps: float = 30.0
    """[rps_throttle_enable] Inference arrival RPS below which a
    currently-engaged throttle RELEASES (sets ``admission_open =
    True``). Must be strictly less than ``rps_throttle_close_rps``
    so the gap acts as hysteresis — RPS sitting between the two
    thresholds leaves the gate in its current state, which avoids
    flapping at the boundary."""

    rps_throttle_window_s: float = 1.0
    """[rps_throttle_enable] Sliding-window width for the arrival-
    rate measurement. Too short (e.g. 0.1s) = noisy: a single-burst
    spike could flap the throttle. Too long (e.g. 10s) = laggy:
    the throttle stays engaged seconds after the burst is over.
    1-2s is the sensible range for typical workloads."""

    rps_throttle_close_time: float = 0.0
    """[rps_throttle_enable] Minimum duration (seconds) the throttle
    must stay engaged once it has closed admission, even if the
    arrival rate immediately drops below ``rps_throttle_open_rps``.
    The release check is deferred until ``close_time`` seconds have
    elapsed since the engage moment.

    Acts as TEMPORAL hysteresis layered on top of the existing
    SPATIAL hysteresis (the ``close_rps``/``open_rps`` gap):

      * Spatial hysteresis stops flapping at the rate boundary —
        the RPS must drop noticeably below the close threshold to
        re-open.
      * Temporal hysteresis (this knob) stops flapping in time —
        even if the RPS briefly dips, the throttle stays engaged
        for at least ``close_time`` seconds. Useful when bursts
        are short (sub-second) but you want the FT pause to be
        sticky for longer so the backward isn't dragged through
        a flapping pause/resume cycle.

    Default ``0.0`` disables — the throttle releases the moment
    the RPS drops below ``open_rps`` (today's behaviour). Set to
    e.g. ``2.0`` to require the throttle to stay engaged for at
    least 2 seconds after each engage.

    Idle bypass: when the sliding-window RPS reads EXACTLY 0
    (no arrivals in the last ``rps_throttle_window_s`` seconds),
    the close_time hold is short-circuited and the throttle
    releases immediately. Rationale: a fully-quiet inference
    window means there is no inference work to protect from, so
    holding the bwd paused any longer just wastes FT progress.
    Pick ``rps_throttle_window_s`` to define the "for a window"
    duration the idle bypass requires (e.g. ``window_s = 1.0``
    means rps=0 must hold for at least 1s before bypass fires)."""

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


@config
class AdapterRoutingConfig:
    """Knobs for adapter routing: requests that ask for it (``model`` equal to
    ``route_model_name``) get a LoRA adapter chosen from their content, by the
    ARIADNE recipe — a frozen sentence encoder, one centroid per adapter built
    from the adapter's training inputs ("indexing"), cosine top-1.

    Its own switch, independent of ``FinetuneConfig.enable_finetuning``. While
    ``enable_adapter_routing`` is False nothing under ``deltaserve.adapter_routing`` is
    imported and no routing process, middleware, pool or route exists.

    Travels as ``additional_config["deltaserve_adapter_routing"]`` (the ``adapter_routing:``
    section of a serving YAML). None of it changes the traced forward, so it is
    left out of the compile-cache hash."""

    enable_adapter_routing: bool = False
    """Master switch. False (default) = the server behaves exactly as without
    this feature."""

    route_model_name: str = "auto"
    """A request is routed only when its ``model`` field equals this name. A
    request naming an adapter or the base model is never touched."""

    router_device: str = "auto"
    """Where the route encoder runs. ``"auto"`` (default): on the GPU, inside
    the vLLM worker of the last tensor-parallel rank — in the model's own CUDA
    context, its memory (about 1.3 GiB of weights, fp32) part of what vLLM
    measures before it sizes the KV cache. ``"cpu"``: inside that worker on
    the CPU — every routed request then waits for a CPU encode (50-130 ms)
    inside an engine step; for experiments only, and nothing indexes on a GPU."""

    router_gpu: int | None = None
    """The tensor-parallel rank whose worker holds the route encoder (0 ..
    tp-1), when the default — the last rank — is not wanted. A value outside
    the model's ranks falls back to the last rank."""


    route_encode_max_tokens: int = 1024
    """Most tokens (requests x the longest one, as padded) one encode step
    carries: about 3.5 ms + 0.014 ms per token on an RTX 5090, so the default
    bounds an encode step at about 17 ms. Requests beyond it are encoded by
    the next step. One request is always taken, whatever its length."""

    adapter_copy_ms: float = 4.5
    """What a step pays to copy one adapter from host memory into a GPU slot
    (the adapter was not in a slot when the step was issued). Measured on
    Qwen3-14B TP=2 (rank 16, q/k/v/o): 4.3 ms. The SLO estimator adds it to a
    step's predicted time and takes it off the measured one."""

    adapter_read_ms: float = 13.5
    """What a step pays, on top of ``adapter_copy_ms``, to read one adapter
    from disk and register it — only when a cold adapter is NOT preloaded
    (nothing else was running: the request was scheduled at once). Measured on
    Qwen3-14B TP=2 with the file in the page cache: 11.7 + 1.5 ms."""




    encoder: str = "intfloat/e5-large-v2"
    """Sentence encoder for both the adapter centroids and the request
    embedding (they must match; the index records which one built it)."""

    encoder_query_prefix: str = "query: "
    """Text prepended to every encoded input (e5 convention)."""

    encoder_max_tokens: int = 512
    """Inputs are truncated to this many encoder tokens."""

    route_keep_percent: float = 90.0
    """The confidence threshold, as the share of the pool's REFERENCE samples
    (training data) that pass it; requests below it fall back to the base
    model. The pool computes the threshold from its reference samples (see
    ``adapter_routing/decision.py``) and recomputes it when the pool changes.
    Real requests are less like the training data than the training data is,
    so fewer of them pass: on the adapter-route benchmark 90 routed 78 % of the
    requests that had a matching adapter (no loss of task score on those) and
    sent 58 % of the unmatched ones to the base model; 80 routed 59 % / sent
    79 % to base. 100 = never fall back on confidence."""

    route_confidence_slack: float = 0.0
    """Lowers the confidence threshold by this many confidence units (the sum
    of two z-scores). Real requests are less typical than the training samples
    the threshold is computed from; with small or templated training sets the
    gap is large and genuine requests fall back unless this is raised. Set it
    from the decision log."""

    route_close_band: float = 0.05
    """Adapters whose (centred) score is within this of the best are "close":
    the request goes to one of them, preferring one already in a GPU slot,
    then the most used."""

    route_min_pool: int = 4
    """Fewer adapters than this in the pool: centring is unreliable, the
    confidence falls back to the raw score (a weaker signal)."""

    route_reference_samples: int = 64
    """Training-sample embeddings kept per adapter during indexing, to
    calibrate the routing confidence (about 256 KB each)."""

    route_decision_log: bool = False
    """Append every routing decision (no request text) to
    ``<index dir>/decisions.jsonl`` for offline calibration."""

    fallback_score_threshold: float = 0.0
    """A manual, absolute floor on the raw cosine of the best adapter, applied
    before the confidence check (0.0 = off). Mainly for pools without
    reference samples; e5 cosines sit in a narrow high band, so prefer
    ``route_keep_percent``."""

    max_active_adapters: int = 32
    """Most ACTIVE adapters — the ones auto-routing chooses among. A quality
    knob, not a hard limit of vLLM's: the more adapters compete, the more often
    a neighbour is picked. Retired adapters are not counted."""

    when_full: str = "retire_least_used"
    """What admission does when the active set is full:
    ``retire_least_used`` (the least used adapter that is not pinned and past
    its grace period is retired; if there is none the newcomer is admitted
    retired), ``admit_retired`` (the newcomer is admitted retired) or
    ``reject``."""

    retire_grace_s: float = 600.0
    """A newly admitted adapter cannot be auto-retired for this long (it has
    no usage yet)."""

    admission_min_self_recall: float = 0.0
    """Admit as retired (servable by name only) when fewer than this share of
    the candidate's own reference samples would be routed to it. 0 = never
    (the report is still recorded on the job)."""

    admission_max_stolen: float = 1.0
    """Admit as retired when the candidate would take more than this share of
    an existing adapter's reference samples. 1 = never. Adapters that are
    merely similar are admitted; routing treats them as close candidates."""

    pool_protected_slots: int = 1
    """How many of the most-used adapters in LoRA GPU slots can never be
    evicted to make room for another one."""

    pool_hit_half_life_s: float = 300.0
    """Half-life of an adapter's hit count: GPU slots are kept by this decayed
    frequency (0 = never decays, i.e. plain hit counts)."""

    adapters: list | None = None
    """Existing adapters to make auto-routable without training, each a mapping
    ``{path, name?, data_path?, centroid_path?, prewarm?}``: ``data_path`` is
    the dataset its centroid is built from (a job with stages
    ``[init, indexing, ready]``), ``centroid_path`` a ready-made centroid
    (.npy) instead. Without either it is servable by name only."""

    index_path: str | None = None
    """Directory of a centroid index built offline (adapters registered
    statically, without a finetuning job). None = start with an empty pool."""

    indexing_max_samples: int = 2500
    """At most this many training inputs are encoded per adapter centroid."""

    indexing_input_until: str | None = None
    """A marker that separates a training sample's input from its target, e.g.
    ``"### Response:"``. When set, only the text before it is indexed. Requests
    carry the input alone, so a centroid built from input + target matches
    them worse; set this whenever the corpus lines contain the target."""

    indexing_chunk_samples: int = 16
    """Samples per indexing chunk. The job checkpoints after every chunk and a
    chunk is the unit that moves between CPU and GPU (or is redone if dropped)."""

    indexing_cpu_max_threads: int = 8
    """Threads of the CPU indexer while the engine has no inference request.
    The indexer process runs in the lowest scheduling class either way (it
    only gets CPU time nothing else wants)."""

    indexing_cpu_busy_threads: int = 2
    """Threads of the CPU indexer while inference requests are present: what
    bounds its memory-bandwidth and cache pressure on the engine. The
    scheduler switches between the two when the engine's state changes."""



    indexing_cpus: str | None = None
    """CPU list the CPU indexer is pinned to, e.g. ``"8-15"``; keep it disjoint
    from the cores the vLLM processes use. None = no pinning."""

    indexing_on_gpu: bool = True
    """Let indexing use the GPU while the engine has no inference request: a
    process of its own that shares the route encoder's weights with the worker
    (no second copy; one more CUDA context, a few hundred MiB outside
    ``gpu_memory_utilization``). False = indexing stays on the CPU."""

    indexing_gpu_chunk_samples: int = 8
    """Samples per indexing chunk on the GPU. A chunk interrupted by an
    arriving request (at its next encoder layer) is redone, so small chunks
    lose less."""

    indexing_gpu_after_quiet_s: float = 3.0
    """The engine counts as free of inference — indexing may use the GPU, and
    the CPU indexer its full thread count — only after no inference request
    has been present for this long. It stops counting the moment one arrives.
    Keeps the gaps between requests of a steady load from flipping the state."""

    def __post_init__(self) -> None:
        if not self.route_model_name:
            raise ValueError("adapter_routing.route_model_name must not be empty")
        if not self.encoder:
            raise ValueError("adapter_routing.encoder must not be empty")
        if self.when_full not in ("retire_least_used", "admit_retired", "reject"):
            raise ValueError(
                "adapter_routing.when_full is one of retire_least_used, "
                "admit_retired, reject")
        for name in ("encoder_max_tokens", "max_active_adapters",
                     "route_encode_max_tokens",
                     "indexing_max_samples", "indexing_chunk_samples",
                     "indexing_cpu_max_threads", "indexing_cpu_busy_threads",
                     "indexing_gpu_chunk_samples",
                     "route_min_pool"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"adapter_routing.{name} must be >= 1")
        for name in (
                     "indexing_gpu_after_quiet_s", "pool_protected_slots",
                     "pool_hit_half_life_s",
                     "route_close_band",
                     "route_confidence_slack", "retire_grace_s",
                     "route_reference_samples"):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"adapter_routing.{name} must be >= 0")
        if not 0.0 <= float(self.route_keep_percent) <= 100.0:
            raise ValueError("adapter_routing.route_keep_percent is in [0, 100]")
        seen = set()
        for a in self.adapters or []:
            if not isinstance(a, dict) or not a.get("path"):
                raise ValueError(
                    "adapter_routing.adapters: each entry needs a 'path'")
            unknown = set(a) - {"path", "name", "data_path", "centroid_path",
                                "prewarm"}
            if unknown:
                raise ValueError(f"adapter_routing.adapters: unknown key(s) "
                                 f"{sorted(unknown)}")
            if a.get("name"):
                if a["name"] in seen:
                    raise ValueError(f"adapter_routing.adapters: duplicate name "
                                     f"'{a['name']}'")
                seen.add(a["name"])
        if self.router_gpu is not None and int(self.router_gpu) < 0:
            raise ValueError("adapter_routing.router_gpu must be a GPU index >= 0")
        if not -1.0 <= float(self.fallback_score_threshold) <= 1.0:
            raise ValueError(
                "adapter_routing.fallback_score_threshold is a cosine in [-1, 1]")
