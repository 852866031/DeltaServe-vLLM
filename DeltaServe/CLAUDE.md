# DeltaServe — Project Context

## What this project is

DeltaServe is an **LLM co-serving framework** that runs **inference** and **LoRA
fine-tuning** concurrently on the **same GPU**. The research goal is to
interleave a backward pass (SFT on LoRA adapters) with ongoing inference
serving so the GPU stays saturated, without letting backward work tank
inference TTFT/latency.

Two models are supported:
- `llama` (Llama-1/2 style) — the reference / known-good path.
- `llama3` (Llama-3 style, GQA) — the active target for most optimization work.

Only two finetune types exist now: **`"SFT"`** (the live path) and
**`"SFT Profile"`** (a profiling-only variant of SFT used to fit cost models).
Alignment / DPO code was removed in full — do not look for `setup_alignment`,
`reference_lora_path`, `live_alignment`, `feedback_collector`, `alpha`/`beta`/
`lambdas`, `print_reset_log`, etc.; none of those exist anymore.

## High-level architecture

```
 api_server.py                   ── HTTP + ServerConfig load
      │
      ▼
 router/manager.py               ── scheduler; picks inference batches,
                                    interleaves backward micro-batches,
                                    enforces co-serving policy
      │
      ▼
 router/model_infer/model_rpc.py ── owns GPU process; constructs the
                                    forward runner + the backward service
      │
      ├── forward/inference runner (graph-captured decode optional)
      └── backward service (SFT)  ← most active development surface
```

Backward services per model:

- `dserve/models/llama/SFT_service.py`   — base class `LlamaSFTBackwardService`.
- `dserve/models/llama3/SFT_service.py`  — subclass `Llama3SFTBackwardService`
  (GQA-aware attention backward).
- `dserve/models/llama/SFT_service_graph.py` — `GraphedBackwardRunner`: CUDA
  graph capture layer for the backward pass. Works for both models.

## Configuration system

Everything runtime-configurable lives in a single YAML loaded at startup. There
is **no flag soup**: `api_server.py` exposes only four CLI args.

### Files

```
dserve/server/config.py                                    ← ServerConfig dataclass +
                                                             YAML loader + override parser
eval/llama3/config/serving_config_finetuning.yaml          ← llama3 SFT-enabled (alpaca-1000 + packed_kv defaults)
eval/llama3/config/serving_config_no_finetuning.yaml       ← llama3 inference-only
eval/llama/config/serving_config_finetuning.yaml           ← llama1 SFT-enabled
eval/llama/config/serving_config_no_finetuning.yaml
```

The previous `serving_config_finetuning_packed.yaml` and
`serving_config_finetuning_alpaca.yaml` variants were folded into the
default `serving_config_finetuning.yaml` (alpaca-1000 corpus + packed_kv
allocator are now the baked-in defaults). To exercise other
combinations, pass `--override memory.allocator=unified` /
`--override finetune.data_path=...` rather than maintaining parallel
YAMLs.

### CLI

`api_server.py` accepts:
- `--config <path>` (required) — YAML to load.
- `--override section.field=value` (repeatable) — YAML-parsed value
  (`true`/`null`/`[a,b,c]` all work).
- `--port N` and `--rank_id N` — direct shortcuts for `server.port` /
  `server.rank_id` (the two knobs that vary every launch).

### Sections (see `dserve/server/config.py` for the dataclasses)

`server`, `model`, `serving`, `lora`, `scheduler`, `memory`, `cuda_graph`,
`finetune`, `slo`, `debug`. Strict validation rejects unknown sections/fields
with a helpful error.

### How cfg flows downstream

- `api_server.main()` builds `cfg`, then spawns subprocesses with `args` (which
  carries `args.cfg`).
- `start_router_process(args, ...)` calls `set_active_config(args.cfg)` and
  builds `InputParams(cfg)`.
- Each model RPC subprocess calls `set_active_config(input_params.cfg)` at the
  top of `exposed_init_model`.
- Hot-path code reads from `dserve/common/configs/config.py`'s
  `get_active_config()` (e.g., `infer_batch.py` for `max_req_total_len`).
- `InputParams` (`dserve/server/input_params.py`) is a **thin view** over cfg
  with legacy attribute names (`self.scheduler`, `self.finetuning_params.X`).
  ~50 downstream call sites still read it; new code should prefer
  `cfg.section.field` directly.

### Launchers (`eval/{llama,llama3}/launch_*.py`)

User-facing launchers expose a handful of flags (`--enable-finetuning`,
`--enable-cuda-graph`, `--enable-prefill-cuda-graph`,
`--enable-bwd-cuda-graph`, `--occupancy_log <path>`, `--port`,
`--rank_id`, `--ft_log_path`) and translate them into
`--config <yaml> --override ...` arguments to `api_server.py`. YAML
selection is binary: `--enable-finetuning` →
`serving_config_finetuning.yaml`, else `serving_config_no_finetuning.yaml`.
The previous `--packed-kv` / `--alpaca` flags were removed when the
packed_kv allocator and alpaca-1000 corpus became the baked-in defaults
of the finetuning YAML; override `memory.allocator` or
`finetune.data_path` via `--override` for non-default combinations.
`--occupancy_log` becomes a
`memory.unified_mem_manager_log_path=...` override and activates the
allocator's CSV tracker. The launcher resolves relative `lora.adapter_dirs`
entries to absolute paths so they match what the benchmark client sends as
`lora_dir` (the server stores the dir string verbatim in `lora_ranks` —
mismatched keys raise `KeyError` in `mixed_req_queue._can_add_new_req`).
It also resolves `finetune.data_path` against `eval/llama3/data/` (not
`config/`) — finetuning corpora and their prep scripts live under `data/`,
config YAMLs under `config/`.

## Backward pass: what is and isn't graph-captured

Per transformer layer the backward has two regions:

1. **Post-layer + FFN** — shape-stable after padding by
   `cfg.finetune.max_saved_finetuning_tokens`. **Graph-captured** per layer,
   one graph per layer at the single fixed size. Warmed up once at startup via
   `_warmup_full_backward`.

2. **Attention** — sample-boundary-aware, naturally variable shape. Two paths:
   - **Monolithic** (`_backpop_attention`): per-sample Python loop,
     shape-varying, eager. Always available as a fallback.
   - **Padded** (`_backpop_attention_padded` / `_padded_core`): pads to
     `[ATTN_BN_MAX, ATTN_L_MAX]`, uses a key-padding + causal mask, uses
     `index_put_(accumulate=True)` for shape-stable scatter of per-sample
     grads. **Graph-captured** when enabled. Falls back to monolithic if the
     batch exceeds `ATTN_BN_MAX` or any sample exceeds `ATTN_L_MAX`.

Padded-attention sizing comes from `cfg.cuda_graph`:
- `cuda_graph.use_graphed_bwd_attention` — gates dispatch into the padded
  path. Set on `self.USE_GRAPHED_ATTENTION` in the SFT service `__init__`.
- `cuda_graph.attn_bn_max` — padded batch dimension (distinct samples per
  bwd). Set on `self.ATTN_BN_MAX`.
- `cuda_graph.attn_l_max` — padded sequence length. Set on `self.ATTN_L_MAX`.

(These were class constants in earlier revisions; they're now per-instance
attributes set from cfg, but the attribute names on `self` are unchanged so
existing callsites work.)

Padded-path compute is **quadratic in `L_max`**, so oversizing `L_max` tanks
throughput. Use `eval/llama3/analyze_finetuning_data.py` and
`eval/llama3/keep_p95.py` to size them against the dataset.

## Co-serving contract (load-bearing, do not remove)

- `_maybe_pause()` is called at **every layer boundary** in the backward path.
  It is how backward yields the GPU back to inference. Removing or skipping it
  breaks the whole point of the framework.
- Backward runs on its own CUDA stream (`self.bwd_stream`). Timing must use
  `torch.cuda.synchronize()` before reading the wall clock — otherwise you
  measure host dispatch, not GPU completion (this trap has bitten us).
- MPS partitioning is the mechanism for true concurrent execution. The model
  RPC startup briefly sets `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=10` and
  `CUDA_DEVICE_MAX_CONNECTIONS=1` while spawning the backward subprocess.
- **Co-serving prefill is always eager** — the prefill CUDA graph is hard-gated
  off for any batch with FT tokens at `lora_unordered_batch_mixed.py:171-177`
  (`not has_ft`). The scheduler, estimator, and eligibility mirror all
  encode this as an invariant; do not try to predict graph-mode for a
  co-serve batch.

## SLO-aware scheduling & graph-aware estimators

The scheduler in `router/mixed_req_queue.py` decides every iteration
whether to start a new prefill batch or admit more FT tokens into the
running batch, gated by the configured TTFT / avg-TBT / max-TBT SLOs.
Decisions ride on two execution-time predictors:

```
T_prefill ≈ α·Σnᵢ² + β·Σnᵢ + γ·T_ft + c     # PrefillExecutionEstimator
T_decode  ≈ δ·B    + ε·K     + d            # DecodeExecutionEstimator
```

(`router/tracker.py`). Both are linear lstsq fits over rolling stats.

### Three regimes per estimator

CUDA-graph replay is qualitatively faster than the eager forward, **and**
the first-touch capture is qualitatively slower than either (warmup +
record + post-capture replay). So each estimator distinguishes three
regimes when it predicts a batch's execution time:

- **graph** — bucket is already in the captured set; runtime will hit
  `prefill_replay` / `replay`. Fast and near-constant per bucket.
- **eager** — bucket is not in the captured set and won't be captured
  (config disabled, batch too big, has_ft, etc.); runtime runs an eager
  forward (`_context_forward` / `_token_forward`).
- **capture** — bucket is not in the captured set but the runtime will
  capture it on first hit. Cost ≈ 2×eager + 1×graph for prefill
  (`prefill_capture` runs warmup + capture-pass + replay), 4×eager for
  decode (3 warmup iters + capture-pass, no post-capture replay — see
  `cuda_graph_runner.capture`).

Two parameter sets are *fitted* (`_graph_params`, `_eager_params`); the
**capture regime is computed analytically** as a combination of the
other two at predict time. We don't fit a third regime because capture
events are rare (once per bucket per process lifetime) and the analytic
combination is more stable than a 4-sample fit.

- `_graph_params` — fitted only on inf-only batches whose
  `(bs_bucket, T_bucket)` is in the captured set. Replay time is near-
  constant per bucket; γ is irrelevant in this regime (FT batches always
  go eager) so the prefill graph fit is 3-param (α, β, c) instead of 4.
- `_eager_params` — everything else: uncaptured inf-only + all co-serve.
  The full 4-param prefill model. γ is identifiable here because all
  co-serve batches land in this regime.
- Capture-regime prediction (in `predict_inference` / `predict`): when
  `will_capture=True`, returns `2 * _eval(_eager_params) +
  _eval(_graph_params)` for prefill, or `4 * _eval(_eager_params)` for
  decode. Capture samples themselves are *not* added to the graph fit —
  `was_graph` stamping at dispatch in `_prefill_batch` / `_decode_batch`
  intentionally labels first-touch samples as eager so they don't
  pollute the graph regime's replay-cost distribution.

Cold-start safety: if a regime has fewer than 4 prefill / 3 decode
samples, `data_fit` keeps its previous params and `predict_*` falls back
to the other regime. Predictions are pessimistic-safe by construction.

### `GraphEligibility` — two-place dual structure

The captured-bucket state lives in **two places** that must stay in
sync. Treat this as one logical structure with a "source of truth" and
a "mirror":

```
        ┌─ runner (model_rpc subprocess) ──────────┐
        │   _prefill_cache: {(bs_b, T_b) → graph}  │   ← source of truth
        │   _cache:         {(bs, ml_b) → graph}   │     (the actual graphs)
        │   _pending_prefill_captures: list        │
        │   _pending_decode_captures:  list        │
        └──────────────────────────────────────────┘
                          │
                          │  pop_pending_*_captures()  (drain on every
                          ▼                             scheduler tick)
        ┌─ manager (router process) ───────────────┐
        │   GraphEligibility                       │   ← mirror
        │     _prefill_buckets: Set[(bs_b, T_b)]   │     (just the keys)
        │     _decode_buckets:  Set[(bs, ml_b)]    │
        └──────────────────────────────────────────┘
```

**Runner side (`dserve/common/cuda_graph_runner.py`)** — owns the actual
captured `torch.cuda.CUDAGraph` objects, their static input/output
buffers, and a per-bucket byte-cost record. Two caches, keyed
identically on both sides:

- Prefill: `_prefill_cache[(bs_bucket, T_bucket)] = (graph, bufs,
  static_output, T_bucket)` with bucketing
  `bs_bucket = get_prefill_bs_bucket(bs)` (rounds up to `{1,2,4,8,16,
  32,64}`) and `T_bucket = get_prefill_token_bucket(total_tokens)`
  (rounds up to multiples of `PREFILL_TOKEN_BUCKET_SIZE = 128`).
- Decode: `_cache[(batch_size, ml_bucket)] = (graph, bufs, output,
  ml_bucket)` with `ml_bucket = get_max_len_bucket(max_len_in_batch)`
  (rounds up to multiples of `MAX_LEN_BUCKET_SIZE = 128`). Decode uses
  *exact* batch_size (not bucketed) because LoRA dispatch needs an
  exact adapter mapping.

On every successful `capture()` / `prefill_capture()`, the key is
appended to `_pending_decode_captures` / `_pending_prefill_captures`.
These lists are *push-only inboxes*; the manager drains them.

**Mirror side (`dserve/server/router/graph_eligibility.py`)** — a pure
Python set of keys, no tensor refs. The manager (which runs in a
separate process from the runner) needs `(bucket_key, captured?)` for
every scheduling decision, but it can't reach across the RPC boundary
on the hot path. So we mirror just the set of keys and consult that
locally.

The mirror imports `CudaGraphRunner.get_prefill_bs_bucket`,
`get_prefill_token_bucket`, `get_max_len_bucket`,
`PREFILL_BS_BUCKETS` — **the bucket helpers themselves**, not a copy.
Any change to bucketing in the runner immediately changes the mirror's
predicate evaluation, so the two sides can't drift on bucketing math.

Predicates exposed to the scheduler (`mixed_req_queue.py`):

```python
elig.will_prefill_use_graph(has_ft, bs, total)        # → True iff cached
elig.will_decode_use_graph(bs, max_len)
elig.will_prefill_capture_on_hit(has_ft, bs, total)   # → True iff captureable but not yet cached
elig.will_decode_capture_on_hit(bs, max_len)
```

The `will_*_use_graph` predicate returns True iff the key is in the
mirror's set AND the runtime gates are satisfied (`not has_ft`,
`prefill_enabled`, `bs ≤ PREFILL_BS_BUCKETS[-1]` for prefill).
`will_*_capture_on_hit` returns True iff the runtime *would capture*
on next hit — gates pass but key is not yet in the set.

**Sync lifecycle**:

1. **Seeded once** after `estimate_finetuning_overhead()` via
   `model_rpc.get_all_captured_buckets()` (returns
   `r.all_decode_buckets()` + `r.all_prefill_buckets()`). After this,
   the mirror reflects everything captured during offline profiling.
2. **Updated every scheduler iteration** at the top of
   `_co_serving_step` via `model_rpc.pop_pending_captures()`. The
   runner appends to `_pending_*_captures` on each successful capture;
   the manager drains those lists across all TP ranks (captures are
   lock-step under TP) and `note_pending`s the new keys. Lag is at
   most one scheduler tick (~ms).
3. **Consulted at every prediction site** in `mixed_req_queue.py`. The
   scheduler passes `will_use_graph=...` and `will_capture=...` into
   `predict_inference` / `predict` so the right param set / formula is
   selected. Same eligibility is also read at dispatch in
   `_prefill_batch` / `_decode_batch` so the tracker can stamp
   `was_graph` against the regime that was actually in effect.

**Mismatch surface**: the runner has one extra gate the mirror doesn't
model — `prefill_interrupt_event is not None` in
`lora_unordered_batch_mixed._prefill`. That's only set for FT-only
prefill batches (`manager.py:306`), so it doesn't fire for inference
batches that the scheduler reports as `(graph)`. If a real
scheduler-vs-runtime mismatch appears on inference batches, the first
thing to verify is whether the runtime actually entered
`_prefill_with_cuda_graph` and which of its three branches (REPLAY /
CAPTURE / `_prefill_padded_no_graph` after cap-refusal) it took. A
short-lived `_DEBUG_PREFILL_DISPATCH` print in
`_prefill_with_cuda_graph` is the standard way to pin it down.

`predict_coserving` and `max_next_ft_tokens` ignore the eligibility flag
entirely — they're always eager-regime by the co-serving invariant.

### Decode eligibility uses inf-only counts

The decode kernel forwards inference-only requests; FT requests sit in
the running batch until backward consumes them but don't go through
decode. So `_will_decode_use_graph(B, max_len)` calls in `mixed_req_queue`
use `_decode_active_count(batch)` and `_decode_max_len(batch)` (both
filter out `is_finetuning`), not `len(batch.reqs)` and
`batch.input_tokens()`.

### Refit cadence

Live refit fires every 256 batches (`tracker.py:check_refit`). At refit:
1. The mirror is already current (push model).
2. `data_fit(tracker, eligibility)` walks all historical samples, labels
   each by regime using the *current* mirror, and refits both regimes.
3. The lstsq is over the full tracker history, not a sliding window.
   Expected — eager-regime samples don't go stale because the linear
   model isn't workload-dependent at this granularity.

### Offline profiling — why it matters here

`profiling_batch_generator.py` populates the tracker before live serving
and seeds the eligibility mirror by triggering captures during a warmup
pass. For each unique shape, the **first** run is in `warmup_batches`
(no stats recorded — captures the graph) and the next `num_repeats`
runs are recorded (replay timing). This stops the eager-skew that
otherwise makes the offline fit inflate-then-collapse at the first
online refit. Coverage:
- inf-only token sweep (geometric) + decomposition variants for fitting α
- (inf, ft) coserve grid for fitting γ
All shapes respect `batch_max_tokens`, `max_finetuning_tokens`, and
`max_req_total_len`.

### File map

| Concern | File |
|---|---|
| Scheduler decisions, SLO gates | `router/mixed_req_queue.py` |
| Three-regime estimator + tracker | `router/tracker.py` |
| Manager-side eligibility mirror (just the keys) | `router/graph_eligibility.py` |
| Runner-side capture/replay + cache (the actual graphs) | `common/cuda_graph_runner.py` (`_prefill_cache`, `_cache`, `_pending_*_captures`) |
| RPC surface (`get_all_captured_buckets`, `pop_pending_captures`, `get_graph_mem_stats`) | `server/router/model_infer/model_rpc.py` |
| Pipeline wiring (seed, drain, fit) | `router/manager.py` (`_co_serving_step`, `estimate_finetuning_overhead`) |
| Offline profiling targets | `router/profiling_batch_generator.py` |
| Graph memory cap (`max_graph_memory_gb`) | `common/cuda_graph_runner.py` (`can_capture_more`, `note_capture_refused`); dispatch fallback in `models/peft/lora_unordered_batch_mixed.py` |

## Memory / pool gotchas (also load-bearing)

- CUDA graph capture uses a **graph memory pool** (`graph_pool_handle`). Any
  tensor allocated during capture lives in that pool and **can be reused /
  aliased** by later captured graphs sharing the pool.
- LoRA `.grad` buffers and padded-attention context tensors (the `ctx`) MUST
  be allocated **outside** the graph pool as **persistent buffers** before
  capture, then referenced inside the captured graph. Otherwise the optimizer
  reads grads that a later graph has already overwritten — you get NaNs in
  inference softmax that look unrelated. (`_persistent_lora_grads[layer_id]`
  and the persistent attention `ctx` are there for this reason.)
- Before each `run()`, `.grad` is re-attached to the persistent buffer for
  every LoRA weight, because `zero_grad(set_to_none=True)` elsewhere will
  otherwise detach it.
- `UnifiedMemoryAllocator.max_finetuning_tokens` (from
  `cfg.memory.max_finetuning_tokens`) sizes the shared activation / logit /
  input-id buffers. Do not confuse it with
  `cfg.finetune.max_saved_finetuning_tokens`, which is the per-backward token
  budget and the FFN graph capture size.
- **External callers MUST go through pool accessors**, not
  `mem_manager.gpu_pools[layer]` directly. The accessors are
  `get_kv_pool(layer)`, `get_adapter_pool(layer)`, `get_activation_pool(layer)`
  on `UnifiedMemoryAllocator`. Subclasses (e.g. `PackedKVMemoryAllocator`)
  return a different view from `get_kv_pool` while keeping the others on the
  original layout, so the call site stays oblivious to which allocator is in
  use. A grep for `mem_manager.gpu_pools[` in `dserve/` should return zero
  hits — keep it that way.
- **KV freers MUST call `free_kv(sub_ids)`**, not `free(page_ids)`. The base
  allocator aliases `free_kv → free` so this works uniformly. The packed
  allocator implements sub-slot semantics in `free_kv`. Mis-routing KV ids
  through `free()` corrupts the bitmap silently because sub-slot ids in
  `[0, tot_size)` collide with page ids and cannot be disambiguated by range.
  The three KV freers in `infer_batch.py` (`free_self`, the two filter paths)
  already use `free_kv`; preserve that.

### KV-leak surfaces — what to know before touching KV alloc/free

Three independent KV leak sources existed in this codebase; two are
fixed, one is open. Re-introducing any of them will manifest as KV
pages monotonically growing under steady-state load. The `_DEBUG_FREE`
toggle in `packed_kv_mem_allocator.py` (one-line trace per free call
with per-PageType breakdown) is the canonical instrument to spot
recurrence.

1. **Prefill scratch slots — FIXED in `lora_unordered_batch_mixed._prefill`.**
   When `use_prefill_cg=True`, `prefill_mem_index_key/value` is
   allocated at `T_bucket` size, but `init_bloc` only registers the
   first `total_token_num` slot ids into `b_loc_key/value`. The
   remaining `T_bucket − total_token_num` "padding scratch" slots back
   the padded forward's K/V writes but never reach the per-request
   free path. The fix `free_kv`s them at the end of `_prefill` before
   returning logits. **This was the dominant leak** — hundreds of
   pages per second under load, since every captured-graph prefill
   leaked. The captured graph doesn't pin those specific slot ids
   (the static `prefill_mem_index_key` buffer is overwritten on every
   `prefill_replay`), so freeing them is safe.

2. **Off-by-one in KV free slice — FIXED in `infer_batch.py`.**
   `free_self`, `filter`, and `clip` used to slice
   `b_loc_key[idx, max_len − seq_len : max_len − 1]`, length
   `seq_len − 1`, dropping the slot at column `max_len − 1`. That
   column is real: prefill's `init_bloc` writes into
   `[max_len − seq_len : max_len]` (inclusive), and decode writes the
   freshly allocated slot into `[max_len_in_batch − 1]`. Fix is the
   inclusive slice `[max_len − seq_len : max_len]`. Cost was 2
   sub-slots per finished request (1 K + 1 V).

3. **`InferBatch.merge` last-column drop — OPEN (small).** The merge
   copies `source.b_loc_key[:, :source_max − 1]` into the destination
   and writes `arange(...)` placeholders into
   `dest_b_loc_key[:, dest_max − 1]`. The source's column
   `source_max − 1` (= last real KV slot allocated) is silently
   discarded; the arange values are never valid sub-slot ids under
   `packed_kv` and don't correspond to allocated KV pages. The next
   decode after merge overwrites column `dest_max − 1` with a fresh
   alloc before attention reads it, so this is *visually* dead code
   on the inference path — but the source's discarded slot is
   leaked. Cost is ~2 sub-slots per request per merge call. Toggle
   `_DEBUG_MERGE = True` at the top of `infer_batch.py` to log
   per-merge `residual = allocated_KV_pages − in_flight_demand`. With
   leaks (1) and (2) fixed, residual should plateau within partial-
   page slack (≤ `F − 1` per active alloc).

4. **One-time accounting check on suspected leaks.** Set
   `_DEBUG_FREE = True` in `packed_kv_mem_allocator.py` and watch the
   `[mem_free] free_kv: … [KV=N ADP=M …] free=…` lines. After every
   finished batch, KV should return to the level needed for whatever's
   still in flight. KV climbing without a matching FREE event from
   the surrounding batch lifecycle = the new leak. Re-enable
   `_DEBUG_MERGE` to correlate with merge events. Both toggles are
   off by default — flipping to True adds one `print()` + one GPU
   sync per free call (only on the debug path), zero overhead off.

## Memory allocator selection (`unified` vs `packed_kv`)

Selected at startup via `cfg.memory.allocator` and dispatched through
`dserve/common/allocator_factory.py`. Both implementations share the bitmap +
`free_bitmap` + `page_type_map` machinery from
`dserve/common/unified_mem_allocator.py`.

- **`auto`** (default): `make_allocator` inspects the model's GQA factor
  `F = num_attention_heads / num_key_value_heads` and picks `packed_kv`
  when F > 1, `unified` when F == 1 (`_resolve_auto` in
  `allocator_factory.py`). Llama-3 lands on `packed_kv`; Llama-1/2 on
  `unified`. Logged at startup as
  `[allocator_factory] memory.allocator='auto' → selected '...'`.
- **`unified`**: one page = one KV slot. For Llama-3 GQA each page
  holds `[num_attention_heads=32, head_dim=128]` but only the first 8 head
  rows carry K (or V) data — the other 24 rows are zero padding. Simple, but
  the KV portion of the pool is ~4× oversized for GQA. Pick this explicitly
  for legacy-comparison runs.
- **`packed_kv`** (`dserve/common/packed_kv_mem_allocator.py`): subclass of
  `UnifiedMemoryAllocator` that packs `F = num_attention_heads /
  num_key_value_heads` (= 4 for Llama-3) KV sub-slots into each page. The
  trick is a zero-copy reshape view exposed only to KV-reading kernels:

  ```
  gpu_pools[l]      shape [tot_size, 32, 128]   — adapter / activation kernels
  gpu_pools_kv[l]   shape [tot_size * F, 8, 128] — KV kernels (same memory)
  ```

  KV kernel call sites all do `pool[idx, :kv_heads, :]`; on the reshape view
  the middle dim is already `kv_heads`, so the slice is identity and the
  kernel binary doesn't notice. Zero kernel edits.

- **Allocation strategy in `packed_kv`** is *whole-page-per-call* — no
  partial-page draining: `_alloc_kv(n)` calls `super().alloc(ceil(n/F),
  KV_CACHE)` and derives sub-slot ids arithmetically. `alloc_contiguous_kv`
  similarly delegates to the parent with a smaller page count. Each page's
  `kv_refcount` (int64, set on alloc) tracks how many sub-slots the caller
  holds; `free_kv` decrements via `torch.bincount(page_ids,
  minlength=tot_size)` and a global-mask reconcile, page returns to FREE on
  refcount=0. Trade-off: a partial last page's unused sub-slots are
  "wasted" until the page fully drains — fine for low-occupancy workloads,
  worth re-evaluating if pool occupancy ever climbs past ~50%.

- **`free_kv` is the bottleneck-shaped op**. Use `bincount` (1 sync) rather
  than `torch.unique` + boolean indexing (2 syncs). A "global-mask" rewrite
  scanning the whole `kv_sub_mask` was tried and was slower under MPS
  contention because the larger kernel time on the inference stream stalled
  the backward subprocess — the per-call sync cost is less important than
  GPU stream contention with the backward graphs. Keep this in mind before
  optimizing.

- **F=1 fast paths exist in every override** (`_alloc_kv`,
  `alloc_contiguous_kv`, `free_kv`). For Llama-1 MHA the subclass is
  byte-for-byte identical to the parent; any divergence with `packed_kv` on
  Llama-1 is a pure subclass bug. That's the regression-test canary.

- **CPU-side `_used_pages` counter**, maintained on every alloc/free path in
  both allocators. Source of truth for the occupancy tracker (see below) and
  for any allocator query that wants `used_pages` without a GPU sync.

- **Occupancy tracker** (`_OccupancyTracker` in `unified_mem_allocator.py`)
  is a daemon thread that samples `(used_pages, tot_size, occupancy_pct)`
  to a CSV at `cfg.memory.unified_mem_manager_log_path`. Reads
  `self._used_pages` only — **CPU-only**, never `free_bitmap.sum()`. GPU
  work in a background thread can interfere with CUDA-graph capture/replay
  on the inference thread (caching-allocator pool aliasing), causing silent
  NaN logits later. Don't add GPU ops to the tracker.

- The allocator's `log_path` plumbing flows from
  `cfg.memory.unified_mem_manager_log_path` through `manager.py:673` →
  `model_rpc` → model `_init_mem_manager` → `make_allocator(...)`. Setting
  the YAML field activates the tracker; leaving it `null` keeps the daemon
  thread uncreated and the counter overhead negligible (one Python int
  add/sub per alloc/free under the existing `page_table_lock`).

## Precision rule for llama3 attention

GQA attention backward recomputes softmax. Forward and backward **must match
bit-for-bit on `scores`**. In `_backpop_attention` the forward path computes:

```python
scores = (q_blk.float() @ k_rep.float().transpose(-1, -2)) * scale  # fp32 matmul
```

**Do not** downgrade this to a fp16 matmul followed by `.float()` — that was
the bug that caused llama3 loss to plateau while llama1 trained fine.

## Dead code / do not touch

- `dserve/models/llama/SFT_service_backup.py` — old reference implementation.
- `dserve/models/llama3/SFT_service_backup.py` — old reference implementation.
- `_backpop_attention_autograd` (if you see it) — experimental, not on any
  live path.
- `eval/llama3/data/emotion_original.txt` — raw dataset; the filtered
  `data/emotion.txt` is what the launcher actually loads when emotion is
  the dataset. (Both moved from `config/` to `data/` alongside the alpaca
  switch — see the data-prep tooling below.)
- `eval/{llama,llama3}/config/finetuning_config.json` /
  `no_finetuning_config.json` — legacy JSON configs from before the YAML
  migration. No code reads them anymore; safe to ignore.

## Eval / analysis tooling (`eval/llama3/`)

Layout (post-reorg):
- `eval/llama3/` — entry-point scripts (`launch_llama3.py`,
  `auto_benchmark.py`, `auto_plot.py`, `simple_test.py`,
  `analyze_finetuning_data.py`, `keep_p95.py`, `init_adapters.py`).
- `eval/llama3/scripts/` — comparison-plot helpers and the allocator
  correctness harness (`bwd_graph_plot.py`, `compare_graphs_plot.py`,
  `compare_kv_plot.py`, `compare_occupancy_plot.py`,
  `compare_emotion_alpaca_plot.py`, `compare_allocators.py`).
  All `_HERE`-relative paths in these helpers go up one level
  (`os.path.join(_HERE, "..", ...)`) to reach `output/`, `plots/`,
  `config/`.
- `eval/llama3/timelines/<gpu>/` — request schedules. `<gpu>` is one of
  `5090` / `A100`, auto-detected at script load via `nvidia-smi`. Use
  `--timeline-gpu` (auto_benchmark) or `--timeline-csv-dir` (auto_plot)
  to override on cross-GPU runs. Per-GPU files:
  `timeline_{live,loose,tight,nutanix}.csv`. The nutanix variant is
  proprietary and gitignored.
- `eval/llama3/data/` — finetuning corpora (`alpaca_1000_p95.txt` is
  the default; `emotion.txt` is the legacy alternative) plus the prep
  scripts `load_alpaca.py` / `load_emotion.py`.
- `eval/llama3/timelines/timeline_scaling.py` — hardcoded-config
  utility that rescales a timeline's RPS. Down-scaling drops a random
  subset (keeping original timestamps so the burst pattern is
  preserved); up-scaling compresses the time axis. Writes a CSV +
  comparison PNG next to the source.

Per-script notes:

- `launch_llama3.py` — picks `serving_config_finetuning.yaml` (under
  `--enable-finetuning`) or `serving_config_no_finetuning.yaml`,
  resolves relative paths to absolute, and execs `api_server.py` with
  `--config + --override`. The llama1 equivalent is
  `eval/llama/launch_server.py` (also handles offline HF-cache mode +
  MPS daemon check).
- `auto_benchmark.py` — drives inference load against a running
  server, logs per-request timings and periodic finetuning-token
  counters. Spawns `launch_llama3.py` itself; sends adapter as an
  absolute path. Output filenames are tagged by which knobs are on —
  order: `_decode_prefill_bwd_<tight|loose|nutanix>` (graph tags
  followed by the schedule-shape tag — see `auto_benchmark.py` around
  the `tags = []` block). The previous `_kv` and `_alpaca` tags were
  dropped when packed_kv + alpaca became defaults. Flags:
  - `--decode_graph` / `--prefill_graph` / `--bwd_graph` — append
    `_decode` / `_prefill` / `_bwd` to the suffix and pass
    corresponding overrides to the launcher.
  - `--graphs` — convenience: turns on all three graph flags above.
  - `--track_occupancy` — write `output/occupancy<suffix>.csv` (1 Hz
    sampling); passes `--occupancy_log` to the launcher.
  - `--tight` / `--loose` / `--nutanix` — pick the corresponding
    `timeline_<shape>.csv` from the resolved `timelines/<gpu>/` dir
    instead of `timeline_live.csv`; appends the matching `_<shape>`
    to the suffix. Mutually exclusive.
  - `--timeline-gpu` — override the auto-detected GPU subdir.

  **FT lifecycle under `--co`** (matters for interpreting the bwd log):
  finetuning is started *before* warmup so the FT loop is live during
  both the warmup window and the scheduled-timeline window. At the
  instant the scheduler arms the first timeline request, the script
  captures a wall-clock anchor `T = datetime.now()`. After
  `exit_finetuning` flushes the server's `bwd_log<suffix>_<mode>.csv`,
  `trim_bwd_log_before(path, T)` rewrites that file in place, dropping
  every row whose ISO-8601-second `timestamp` is strictly before `T`
  (rows in the same second as `T` are kept — the server records
  bwd_log timestamps at second precision). The final bwd_log therefore
  contains only the timeline-phase backward batches, which is what
  `auto_plot.py` (and the comparison plots under `scripts/`) treat as
  the FT contribution. If you're inspecting the raw bwd_log yourself,
  remember the warmup batches have already been stripped — the first
  row's timestamp ≈ `T`, not server-start time.
- `scripts/bwd_graph_plot.py` — compares two CSV runs (eager vs.
  graphed) on a 1×3 layout: TTFT CDF, E2E latency over time (with avg
  annotation), cumulative finetuning tokens (with tok/s). Uses
  `drop_duplicates(subset="timestamp", keep="last")` so the cumulative
  line can't backtrack.
- `scripts/compare_graphs_plot.py` — graph-ablation comparison plots,
  four-panel layout (request timeline + the three from
  `bwd_graph_plot`). Also exports `plot_request_timeline` which is
  reused by `auto_plot.py` (both Y-axes pinned to start at 0).
- `scripts/compare_kv_plot.py` — `unified` vs `packed_kv` comparison.
  Looks up `<suffix>` (unified) and `<suffix>_kv` (packed) under
  `output/`. Note: with packed_kv now the default and no `_kv` tag in
  the suffix, comparison runs need explicit
  `--override memory.allocator=unified` to produce a non-`_kv` file.
- `scripts/compare_occupancy_plot.py` — `(used_pages / total_pages)`
  over time, unified vs packed_kv, drives off the
  `occupancy<suffix>.csv` files emitted by `--track_occupancy`.
- `scripts/compare_emotion_alpaca_plot.py` — emotion vs alpaca
  comparison. Reads `_decode_prefill_bwd_kv_<shape>` (emotion) and
  `_decode_prefill_bwd_kv_alpaca_<shape>` (alpaca) for `<shape>` in
  `{tight, loose}` — these tags are *historical*; current runs no
  longer emit `_kv` / `_alpaca` tags, so this script needs runs
  produced under the previous suffix scheme or a small adapter.
- `scripts/compare_allocators.py` — sequential, end-to-end correctness
  check. Spawns the server twice on the same YAML with
  `memory.allocator=unified` vs `memory.allocator=packed_kv` (was two
  separate YAMLs before consolidation), sends N greedy prompts
  concurrently via `asyncio.gather`, diffs generated text + token
  counts. All three CUDA graphs are enabled in both runs so the
  comparison is under the production-realistic config. Use to verify
  packed_kv produces the same outputs as unified after any allocator
  change.
- `auto_plot.py` — single-trace 4-panel plots for one configuration
  across every workload-shape variant (no comparison overlay).
  Default `--suffix _decode_prefill_bwd` matches `--co --graphs`;
  emits `plots/{loose,tight,nutanix}_<config>.png` from the
  corresponding `output/timeline_results<suffix>_<mode>.csv` and
  `output/bwd_log<suffix>_<mode>.csv`. Subplots:
  1. Scheduled request timeline (dual y-axis: req/s bars + output
     tokens/s line; both axes pinned to 0).
  2. Per-request E2E latency vs time (scatter + mean line).
  3. Throughput tokens/s — stacked shading: inference contribution
     band + finetune contribution band, with a rolling-mean
     smoothing window auto-scaled to workload duration
     (`_auto_throughput_window_s`, clipped to `[5, 60]s`). Raw
     per-second values draw as a faint background. Reported avg
     values in the legend come from raw data, not the smoothed
     series. Override the window with `--throughput-window N`
     (`0` disables smoothing).
  4. Rolling TTFT SLO satisfaction rate vs 95% target. SLO threshold
     defaults to `slo.ttft_slo` from `serving_config_finetuning.yaml`
     (the YAML-resolver function lives in `_config_yaml_for_suffix`
     but now always returns that single YAML — the alpaca/packed
     branches were removed with the YAML consolidation). Override
     with `--slo` / `--config-yaml`. `--window` tunes the
     satisfaction-rate window.
  Modes whose CSVs are missing are skipped with a warning.
- `analyze_finetuning_data.py` — tokenizes a dataset, prints percentiles +
  histogram + worst-case greedy-packed distinct-sample count; recommends
  `attn_bn_max` / `attn_l_max` and estimates padding blowup.
- `keep_p95.py` — drops the top 5% longest samples (configurable) so you can
  tighten `attn_l_max` without ever hitting the monolithic fallback.
- `data/load_emotion.py` / `data/load_alpaca.py` — dataset prep scripts.
  Both write the same line-based `.txt` format `finetuning_store.load()`
  expects (one tokenizable sample per non-empty line). `load_alpaca.py`
  pulls `tatsu-lab/alpaca`, applies char-length filters, randomly
  subsamples N (default 1000, seeded), renders the Stanford prompt
  template, and flattens newlines so each rendered prompt is one line.
  Workflow for a new dataset: run the loader, then `keep_p95.py` to size
  `attn_l_max` precisely, then point `finetune.data_path` at the
  `_p95.txt` output.

`MEMORY_ANALYSIS.md` (project root) walks through where every GB of GPU
residency goes for the `--co --decode_graph --prefill_graph --bwd_graph`
workload, identifies the wastes (GQA KV oversizing, activation
double-buffer), and quantifies the right pool sizing for a given
throughput. Read it first if you're investigating memory.

## Important knobs (all in YAML)

| Knob | YAML path | What it controls |
|---|---|---|
| Enable backward graph | `cuda_graph.enable_bwd_cuda_graph` | FFN graph capture in backward |
| Enable decode graph | `cuda_graph.enable_decode_cuda_graph` | Forward decode graph capture |
| Enable prefill graph | `cuda_graph.enable_prefill_cuda_graph` | Forward prefill graph capture (piecewise; co-serve always eager regardless) |
| Padded-attn dispatch | `cuda_graph.use_graphed_bwd_attention` | Padded vs monolithic attn bwd |
| Padded-attn shape | `cuda_graph.attn_bn_max`, `cuda_graph.attn_l_max` | Hard-fail-to-fallback thresholds |
| Prefill profiling cap | `cuda_graph.prefill_sweep_max_tokens` | Upper bound on offline prefill sweep T_buckets. `null` = use INF_CAP. Lower it when GPU memory is tight; runtime batches above the cap lazily capture on first hit. |
| Graph memory cap | `cuda_graph.max_graph_memory_gb` | Cap on combined decode+prefill graph memory (GB). When `total_graph_bytes() ≥ cap`, the runtime refuses to capture additional buckets and falls back to eager (`_prefill_padded_no_graph` for prefill, `_token_forward` for decode). Already-captured buckets keep replaying. `null` or `-1` disables. One-time warning at first refusal via `note_capture_refused()`. |
| FFN graph size / token budget | `finetune.max_saved_finetuning_tokens` | The single fixed FFN graph shape |
| Allocator buffer size | `memory.max_finetuning_tokens` | Shared activation/logit buffers |
| Unified mem pool | `memory.unified_mem_manager_max_size_gb` | KV+activation pool capacity |
| Allocator implementation | `memory.allocator` | `"auto"` (default — `packed_kv` for GQA, `unified` for MHA), `"unified"` (page=1 KV slot, GQA-unaware), `"packed_kv"` (page=F KV sub-slots) |
| Occupancy tracker | `memory.unified_mem_manager_log_path` | If non-null, allocator daemon writes 1 Hz CSV of (used_pages, occupancy_pct). CPU-only, safe under graph capture. |
| SLOs | `slo.{ttft_slo, avg_tbt_slo, max_tbt_slo}` | Scheduler trade-off thresholds (drive admission gates in `mixed_req_queue.py`) |
| Scheduler | `scheduler.name` | Currently always `"dserve"` |
| Scheduler stats dump | `scheduler.batch_prediction_stats_path` | Where `BatchExecutionTracker` writes per-batch decisions (predicted vs. actual duration, batch composition) at FT exit. Default `output/scheduler/batch_prediction_stats.csv`; relative paths resolve to CWD; `null` disables the dump. |

`PROFILE_EVERY` on `GraphedBackwardRunner` is still a class constant
(`SFT_service_graph.py`); flip it in source if you want different profiling
cadence.

## When opening a fresh session

Good first reads, in order:
1. This file.
2. `dserve/server/config.py` — the `ServerConfig` shape; every knob lives
   here.
3. `eval/llama3/config/serving_config_finetuning.yaml` — concrete defaults.
4. `dserve/server/api_server.py` + `dserve/server/router/manager.py` — request
   flow and how cfg fans out to subprocesses.
5. `dserve/server/router/mixed_req_queue.py` + `dserve/server/router/tracker.py`
   — SLO-aware scheduling and the two-regime (graph / eager) estimator. Read
   alongside `dserve/server/router/graph_eligibility.py`.
6. `dserve/server/router/profiling_batch_generator.py` — offline profiling
   targets that seed the estimator and the eligibility mirror.
7. `dserve/models/llama/SFT_service.py` — base backward contract,
   `_maybe_pause()`, the LoRA grad lifecycle.
8. `dserve/models/llama3/SFT_service.py` — active GQA path
   (`_backpop_attention`, `_backpop_attention_padded*`).
9. `dserve/models/llama/SFT_service_graph.py` — graph runner and the
   capture/replay lifecycle, persistent-buffer rationale.
10. `dserve/common/cuda_graph_runner.py` — forward-graph runner shared by
    decode and piecewise prefill; bucket helpers (`get_*_bucket`) are the
    canonical bucketing logic mirrored by `GraphEligibility`.
11. `dserve/common/unified_mem_allocator.py` + `packed_kv_mem_allocator.py`
    + `allocator_factory.py` — pool layout, accessors, the unified vs
    packed_kv split, occupancy tracker.
12. `MEMORY_ANALYSIS.md` (project root) — what consumes GPU memory and
    why; read before any memory-side work.
