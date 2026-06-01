# SLO estimator + FT admission redesign

A spec for refining `deltaserve/estimator.py` and the FT admission gate in
`deltaserve/ft_scheduler.py` / `deltaserve/ft_scheduler_both.py`.

**Headline change in one sentence:** add a third coefficient set for
decode-only steps, replace the closed-form quadratic admission solve with
an iterative per-sample loop that uses the right regime for each
prediction.

The redesign is targeted — no upstream-vLLM changes, no IPC changes, no
backward-subprocess changes. Just the estimator's regime structure and the
admission gate's algorithm.

This file is a self-contained brief for the implementer. Read sections
in order. §1 is the current state (verified against the live tree). §2
is the proposed design. §3 enumerates the code changes.

---

## 1. Current state (verified against code)

### 1.1 Formula

One formula, fit per regime:

```
T_step  ≈  α·S  +  β·T_in  +  γ·T_ft  +  δ·B_d  +  ε·K  +  c
```

Features (`StepFeatures` in `deltaserve/estimator.py:58-87`):

| Symbol | Meaning |
|---|---|
| `T_in` | total prefill tokens this step (**FT tokens are a subset** — see §1.2) |
| `T_ft` | FT prefill tokens this step (subset of `T_in`) |
| `B_d` | number of decode requests this step |
| `K` | total decode context tokens across all decode requests |
| `P` | number of prefill samples this step |
| `S` | `Σ nᵢ²` (prefill self-attention work). Exact when `prefill_lens` is given; `T_in²/P` proxy otherwise |
| `c` | constant per-step overhead |

The design matrix row is `[S, T_in, T_ft, B_d, K, 1]`.

### 1.2 Subset convention (load-bearing — keep)

`T_ft` is a **subset** of `T_in`, not a disjoint column. Verified in two
places:

- `_features_from_output` in `ft_scheduler.py:230-265` is the recording
  path (it stamps the realized batch's features for the tracker). At
  lines 254-258: every prefill request — inference OR finetuning —
  contributes to `T_in`; the `is_finetuning` check **adds to `T_ft` in
  addition**, never instead. So the fit data uses subset semantics.
- The admission inverse solve `α·x² + (β+γ)·x = budget` at
  `estimator.py:236-258` is **correct under subset semantics**: adding
  one FT sample of `x` tokens increases `T_in` by `x` AND `T_ft` by `x`,
  so the marginal cost is `α·Δs + (β+γ)·x`.

`β` is the per-prefill-token cost paid by any prefill token (inference or
FT). `γ` is the **extra** cost paid specifically by FT tokens because
they incur activation-capture hook copies. Total cost per FT token =
`β + γ`. Both have clean physical interpretations.

`_current_step_features` in `ft_scheduler.py:148-178` builds features
for the baseline step **before any FT is added** — that's why it sets
`t_ft = 0.0`. This isn't a different convention; it's just "what would
the step cost without FT?" The subset convention is what the recorded
data and the inverse solve actually use.

### 1.3 Regime partition

Two regimes, partitioned by `was_graph`:

| Regime | Selected when | Fit on |
|---|---|---|
| `_eager` | `will_use_graph == False` at predict time | tracker records where `was_graph` is False OR `f.has_ft` is True |
| `_graph` | `will_use_graph == True` at predict time | tracker records where `was_graph` is True AND `f.has_ft` is False |

`was_graph` is stamped by `_will_use_graph()` in `ft_scheduler.py:267`,
which queries vLLM's `CudagraphDispatcher`. Binary flag — does the step
run as a CUDA graph (any kind: FULL or PIECEWISE) or fully eager?

The partition in `data_fit` (lines 261-284) forces a record into the
eager bucket when `f.has_ft` is True regardless of `was_graph`:

```python
is_graph = bool(wg) and not f.has_ft
```

The live admission path always calls `predict(feats, will_use_graph=False)`
(`ft_scheduler.py:190`, `ft_scheduler_both.py:84`) — it asks "what would
this step cost with FT added?" and FT always forces eager. So the graph
regime is only used for predicted-stat tracking, not live decisions.

### 1.4 What's broken

The single per-regime formula doesn't reflect that the kernel dispatch
envelope changes with step composition. vLLM v1 with the default
`cudagraph_mode = FULL_AND_PIECEWISE` (verified at
`vllm/config/compilation.py:598`) has three distinct runtime modes:

- **`FULL`** for decode-only batches.
- **`PIECEWISE`** for prefill and mixed prefill+decode batches.
- **`NONE`** (fully eager) for anything carrying FT (we force it via
  `force_eager` in the runner).

Each mode has substantially different per-token kernel-launch overhead.
Fitting one set of coefficients across rows from different modes
produces a compromise fit where each mode's predictions are biased — the
direction of the bias depends on which mode dominated the training data
for that regime.

The most visible failure: **decode-only steps under the current pooled
`_graph` regime get systematically wrong baseline predictions** because
the regime's training data also includes PIECEWISE prefill+decode
records (also `was_graph=True` in the dispatcher's view). Decode-only
steps run faster (FULL graph mode); the pooled fit overestimates their
cost.

The compensating bias accidentally helps the unified-phase scheduler's
admission safety (the over-prediction of baseline leaves less slack
budget for FT, so it can't over-admit). But it's wrong for everything
else — predicted-vs-actual stats, observability, and any future
admission logic that needs an accurate baseline.

### 1.5 Other things to know

- **`max_next_ft_tokens` quadratic** at `estimator.py:236-258` solves
  `α·x² + (β+γ)·x ≤ budget` against `_eager` params. Closed-form, single
  call per admission decision. This will be replaced by an iterative
  loop in the redesign (see §2.4).
- **`will_use_graph` boolean** on `StepExecutionTracker` records
  (`estimator.py:126`) is stamped at schedule time. After the redesign,
  it becomes vestigial — kept for sanity checking only, not used for
  selection.
- **`MIN_FIT_SAMPLES = 8`, `REFIT_EVERY = 256`** cadences stay unchanged.

---

## 2. Proposed redesign

### 2.1 Three composition-based regimes

Partition by the upcoming step's composition. Each regime maps 1:1 to
vLLM's runtime CUDA-graph mode.

| Regime | Selection rule | Runtime mode | Design-matrix columns |
|---|---|---|---|
| `INF_PREFILL` | `T_ft == 0 AND T_in > 0` | PIECEWISE | `[S, T_in, B_d, K, 1]` (5 cols, no γ) |
| `EAGER` | `T_ft > 0` | NONE (forced eager) | `[S, T_in, T_ft, B_d, K, 1]` (full 6 cols) |
| `DECODE_ONLY` | `T_in == 0 AND T_ft == 0 AND B_d > 0` | FULL | `[B_d, K, 1]` (3 cols, no prefill) |

Three things to note:

- **`EAGER` absorbs the FT-only-on-idle case** (`B_d == 0`, `T_ft > 0`,
  `T_in > 0` because FT tokens are subset of T_in). The linear formula
  handles `B_d == 0` and `K == 0` as the degenerate cases (`δ·0 + ε·0
  = 0`). No separate "pure FT" regime needed.
- **`INF_PREFILL` covers both prefill-only and prefill+decode** when no
  FT is present. The decode contribution (`δ·B_d + ε·K`) is in the
  formula; pure prefill-only inference is the `B_d == 0` degenerate
  case.
- **`was_graph` field is no longer used for selection.** The regime
  label is derived from composition. Keep `was_graph` in tracker
  records as a sanity-check (assert regime ↔ was_graph correlation at
  fit time), or drop it.

### 2.2 Subset convention — keep

`T_in` continues to include FT tokens as a subset; `T_ft` is the FT
subset. No change to the recording path or to feature semantics. This
keeps the per-coefficient physical interpretation intact:
- `β` = per-prefill-token cost paid by any prefill token (inference or FT)
- `γ` = extra per-FT-token cost from activation-capture hook overhead
- `β + γ` = total cost per FT token

### 2.3 Selection logic

```python
def _select(features: StepFeatures) -> StepParams:
    if features.t_ft > 0:
        regime = EAGER
    elif features.t_in > 0:
        regime = INF_PREFILL
    else:
        regime = DECODE_ONLY
    return self._params[regime]
```

`predict()` no longer takes a `will_use_graph` argument. Regime is
composition-derived.

Cold-start fallback: if the selected regime isn't fitted yet, fall
back to any fitted regime (in priority order EAGER → INF_PREFILL →
DECODE_ONLY) so admission can still make best-effort decisions.

### 2.4 Admission algorithm — iterative greedy per-sample

Replace the closed-form `max_next_ft_tokens(slack)` quadratic with a
per-sample loop. For each candidate FT sample (smallest-first):

1. Build hypothetical features including this sample.
2. Predict the step cost using the **`EAGER` regime** (because admitting
   any FT forces eager).
3. Check TBT and TTFT SLOs against the predicted cost.
4. If all SLO constraints hold, admit; otherwise stop.

Smallest-first lets the loop stop early without leaving headroom
unused. The baseline-before-FT prediction uses the regime matching the
upcoming step's actual CUDA-graph mode (DECODE_ONLY or INF_PREFILL),
giving an accurate headroom check.

The full pseudo-code:

```
FUNCTION admit_ft_to_step(inference_batch):
    # ────────────────────────────────────────────────────────────
    # STAGE 0 — Hard preconditions.
    # ────────────────────────────────────────────────────────────
    IF NOT coord.ft_started:                          # /start_finetuning not POSTed
        RETURN []
    IF coord.pending_backward:                        # backward in flight
        RETURN []
    IF coord.space_remaining() <= 0:                  # activation buffer full
        RETURN []
    IF store.is_empty_and_epoch_drained():
        RETURN []

    # ────────────────────────────────────────────────────────────
    # STAGE 1 — Extract features for the upcoming step (no FT yet).
    # ────────────────────────────────────────────────────────────
    features = features_from_inference_batch(inference_batch)
    # features = {T_in, P, prefill_lens, B_d, K, T_ft=0}

    is_decode_only = (features.T_in == 0 AND features.B_d > 0)
    is_idle        = (features.T_in == 0 AND features.B_d == 0)
    has_prefill    = (features.T_in > 0)

    # ────────────────────────────────────────────────────────────
    # STAGE 2 — Scheduler phase gate.
    # ────────────────────────────────────────────────────────────
    IF config.coserving_admission_phase == "prefill" AND is_decode_only:
        RETURN []

    # ────────────────────────────────────────────────────────────
    # STAGE 3 — Baseline-without-FT prediction + SLO headroom check.
    # Pick the regime matching the step's actual CUDA-graph mode
    # if no FT were added.
    # ────────────────────────────────────────────────────────────
    IF is_decode_only:
        T_baseline = estimator.predict(features, regime=DECODE_ONLY)
    ELSE IF is_idle:
        T_baseline = 0      # no inference cost; only FT will run
    ELSE:
        T_baseline = estimator.predict(features, regime=INF_PREFILL)

    # TBT headroom (when there are decodes)
    IF features.B_d > 0:
        IF T_baseline >= max_tbt_slo:
            RETURN []       # already over TBT budget

    # TTFT headroom (when there's a waiting request)
    IF has_prefill:
        earliest_arrival = inference_batch.earliest_waiting_arrival_time
        queue_wait = (last_scheduled_step.predicted_time IF async_scheduling
                      ELSE 0)
        ttft_deadline = earliest_arrival + 0.9 * ttft_slo
        IF (ttft_deadline - now - queue_wait - T_baseline) <= 0:
            RETURN []       # already over TTFT budget

    # Buffer-space cap (always applies)
    token_budget_hard_cap = coord.space_remaining()

    # ────────────────────────────────────────────────────────────
    # STAGE 4 — Iterative greedy admission, SLO check per sample.
    # Each iteration uses the EAGER formula because any admitted FT
    # sample forces the step eager (even if baseline was FULL graph).
    # ────────────────────────────────────────────────────────────
    admitted = []
    tentative_features = COPY(features)

    LOOP:
        IF tentative_features.T_ft >= token_budget_hard_cap:
            BREAK

        candidate = store.peek_next_smallest(excluding=admitted)
        IF candidate IS None:
            BREAK                            # corpus exhausted (this epoch)

        IF tentative_features.T_ft + candidate.input_len > token_budget_hard_cap:
            BREAK                            # next sample is too big for buffer

        # Build hypothetical features with this sample added.
        # Subset convention: T_in includes FT prefill, T_ft is the FT subset.
        hypothetical = tentative_features.WITH_FT_SAMPLE_ADDED(candidate)
        # hypothetical.T_ft += candidate.input_len
        # hypothetical.T_in += candidate.input_len
        # hypothetical.prefill_lens += [candidate.input_len]
        # hypothetical.P += 1
        # hypothetical.S recomputed from prefill_lens

        # Predict step cost with this sample admitted (forces eager).
        T_with_ft = estimator.predict(hypothetical, regime=EAGER)

        # SLO checks on the hypothetical-with-FT cost.
        IF features.B_d > 0:
            IF T_with_ft > max_tbt_slo:
                BREAK
        IF has_prefill:
            IF (earliest_arrival + 0.9*ttft_slo - now - queue_wait - T_with_ft) <= 0:
                BREAK

        # Sample fits — admit it.
        admitted.APPEND(candidate)
        tentative_features = hypothetical

    # ────────────────────────────────────────────────────────────
    # STAGE 5 — Commit admission.
    # ────────────────────────────────────────────────────────────
    IF LEN(admitted) == 0:
        RETURN []

    store.claim(admitted)                    # remove from selectable pool
    coord.reserve(SUM(s.input_len FOR s IN admitted))    # async-safe buffer slot

    RETURN [build_ft_request(s) FOR s IN admitted]
```

### 2.5 Why iterative instead of the closed-form quadratic

- **Naturally regime-aware.** The baseline uses one regime, the
  per-sample iteration uses another. The eager-penalty cost (the cost
  of the step going from FULL graph to NONE eager when FT is admitted)
  is **implicit in the difference between what the two formulas predict
  for the same `(B_d, K)`** — no separate term, no manual safety
  margin needed.
- **Exact `S` per iteration.** The closed-form quadratic approximated
  `S` via the `T_in²/P` proxy. The iterative version recomputes `S =
  Σ nᵢ²` exactly from each hypothetical's prefill_lens list. More
  accurate.
- **Cheap.** Each iteration is one `predict()` call (a handful of
  floating-point ops). Typical admission considers 1–10 samples per
  step. Negligible overhead.
- **Honest about the regime transition cost.** The current closed-form
  solve assumes "FT adds α·x² + (β+γ)·x on top of whatever baseline
  was." That's true if baseline was already in the eager regime, but
  not if baseline was in a graph regime (decode-only or PIECEWISE
  prefill). The iterative version compares apples to apples — each SLO
  check is "would the full eager prediction with this sample admitted
  exceed the SLO?"

### 2.6 The `decode_only_ft_safety_margin` knob can drop to 1.0

Today's `decode_only_ft_safety_margin: 0.7` exists because the current
admission solve doesn't model the FULL→eager transition cost — the
margin compensates by tightening the TBT budget on decode-only steps.

After the redesign, the iterative loop's per-sample EAGER prediction
already captures that cost. The margin becomes structurally
unnecessary. Keep the knob with default 1.0 for cold-start
conservatism (in case the EAGER regime's predictions on
`(T_in=0, T_ft>0, B_d>0)` shapes are noisy until enough data
accumulates).

### 2.7 Cold-start data sparsity per regime

| Regime | Data source | Cold-start fill |
|---|---|---|
| `INF_PREFILL` | every prefill-carrying inference step where FT is NOT admitted | well-populated during prefill bursts and during steps where the SLO gate denies FT |
| `EAGER` | every step where FT is admitted | well-populated once `/start_finetuning` opens admission and stays open |
| `DECODE_ONLY` | every decode-only step (no inference prefill, no FT) | well-populated during long generation phases |

All three regimes have a natural population path under steady-state.

The one sparsity concern is **`EAGER` regime with `T_in == 0`** (FT
admitted onto a decode-only step under the unified-phase scheduler).
This shape is rare under the default scheduler and only appears when
the unified-phase mode is on AND the SLO gate admits FT on a decode-
only step. The iterative loop's first few admissions in this case
will use EAGER-regime coefficients that haven't seen the exact shape
much.

**Mitigation**: extend the offline profiling pass to cover
`(B_d > 0, K > 0, T_in = 0, T_ft > 0)` shapes so the EAGER regime has
representative data from step 0. Cheap to add (a few extra shape
entries in `profiling_batch_generator.py`).

### 2.8 What stays the same

- `StepFeatures` dataclass shape and subset convention.
- `StepExecutionTracker` interface (records still carry features +
  duration + predicted; `was_graph` field stays for backward
  compatibility / sanity check).
- `MIN_FIT_SAMPLES = 8`, `REFIT_EVERY = 256` cadences.
- The pessimistic `pred *= 1.0 + 1.5 * rmse` safety multiplier on
  predictions (apply per regime).
- The prediction-stats CSV format (just add a `regime` column).
- The forward-recompute / FFN-bwd / attn-bwd CUDA graphs in the
  backward subprocess — entirely separate concern.
- The `_maybe_pause` GPU-yield contract — entirely separate concern.
- The `match_prefill_workload_factor` and
  `ft_tokens_admission_constrain_factor` admission shapers — left as
  outer wrappers around the iterative loop, or removed entirely (see
  §3.2 — implementer's choice).

---

## 3. Concrete code changes

### 3.1 `deltaserve/estimator.py`

**Replace the two-attr `_eager`/`_graph` pair with a dict keyed by
regime.**

```python
REGIMES = ("inf_prefill", "eager", "decode_only")

class MergedExecutionEstimator:
    def __init__(self) -> None:
        self._params = {r: StepParams() for r in REGIMES}
        self._rmse   = {r: None for r in REGIMES}
        self._warned_unfitted = False
```

**Replace `_select(will_use_graph)` with a composition-based selector.**

```python
def _select(self, features: StepFeatures) -> tuple[str, StepParams]:
    if features.t_ft > 0:
        regime = "eager"
    elif features.t_in > 0:
        regime = "inf_prefill"
    else:
        regime = "decode_only"
    p = self._params[regime]
    if p.is_fitted:
        return regime, p
    # Cold-start fallback: try fitted regimes in priority order.
    for r in ("eager", "inf_prefill", "decode_only"):
        if self._params[r].is_fitted:
            return r, self._params[r]
    return regime, p  # still unfitted; predict returns 0
```

**`predict()` drops the `will_use_graph` parameter.** Make it accept an
optional explicit `regime` override for callers that need it (the
admission loop's per-sample step always wants EAGER regardless of
what `_select` would choose).

```python
def predict(self, features: StepFeatures,
            regime: str | None = None) -> float:
    if not self.is_ready:
        ...
    if regime is not None:
        p = self._params[regime]
        if not p.is_fitted:
            # Cold-start fallback for explicit-regime requests too.
            _, p = self._select(features)
    else:
        regime, p = self._select(features)
    pred = p.eval(features)
    rmse = self._rmse[regime]
    if rmse:
        pred *= 1.0 + 1.5 * rmse
    return max(0.0, float(pred))
```

**`StepParams.eval()` needs per-regime column awareness.** Simplest
approach: store unused coefficients as 0.0 (not None) in each regime,
so the formula `α·S + β·T_in + γ·T_ft + δ·B_d + ε·K + c` evaluates
correctly for any regime — the unused columns contribute nothing.

The per-regime fit (§3.2 below) produces only the non-zero coefficients
via reduced design matrices; the others are filled with 0.0 at fit
time.

**Replace `max_next_ft_tokens(budget_s)` with iterative-friendly
helpers.** Actually — just delete it. The iterative admission loop
calls `predict()` directly, no quadratic solve needed.

**`data_fit()` partitioning** — partition tracker records by the
composition predicate, fit each regime on its reduced design matrix:

```python
def data_fit(self, tracker: StepExecutionTracker) -> dict[str, StepParams]:
    buckets = {r: ([], []) for r in REGIMES}
    for f, dur in zip(tracker.features, tracker.durations):
        if f.t_ft > 0:
            regime = "eager"
        elif f.t_in > 0:
            regime = "inf_prefill"
        else:
            regime = "decode_only"
        X, y = buckets[regime]
        X.append(self._row_for_regime(f, regime))
        y.append(dur)

    for r in REGIMES:
        self._params[r], self._rmse[r] = self._fit_regime(
            buckets[r][0], buckets[r][1], self._params[r], r)
    return self._params

def _row_for_regime(self, f: StepFeatures, regime: str) -> list[float]:
    if regime == "inf_prefill":
        return [f.s, f.t_in, f.b_d, f.k, 1.0]            # 5 cols, no γ
    elif regime == "eager":
        return [f.s, f.t_in, f.t_ft, f.b_d, f.k, 1.0]    # full 6 cols
    else:  # decode_only
        return [f.b_d, f.k, 1.0]                          # 3 cols, no prefill
```

`_fit_regime` scatters the fitted coefficient vector back into the
right `StepParams` fields based on regime (fill unused fields with
0.0).

### 3.2 `deltaserve/ft_scheduler.py` (and `ft_scheduler_both.py`)

**Replace `_slo_ft_budget` with `admit_ft_to_step`** (§2.4 pseudo-
code). The function name change reflects the algorithm change — no
longer "what's the budget?", now "which samples should we admit?".

**Wire it into `schedule()`** in place of the current
`_initial_ft_budget` → `_slo_ft_budget` → `next_ft_requests(budget)`
chain. The new function returns the list of FT requests directly.

**`_will_use_graph()`** at `ft_scheduler.py:267` and the `was_graph`
stamping at `ft_scheduler.py:593` become vestigial. Either remove
them, or keep them as observability (so the prediction-stats CSV can
correlate composition-regime with actual runtime mode for sanity
checking).

**`_features_from_output()`** (the post-schedule realized-features
stamp) is unchanged — it still records what actually ran, subset
convention intact.

**Admission shapers** (`match_prefill_workload_factor`,
`ft_tokens_admission_constrain_factor`): out of scope for the
estimator redesign. Two options for the implementer:

- (a) Wrap the new `admit_ft_to_step` with the existing shapers as
  outer pre-filters (e.g. the leaky-bucket caps the loop iteration
  count to 1). Backward-compatible with current configs.
- (b) Remove the shapers entirely. The iterative SLO-driven admission
  is more principled than either shaper; the shapers were workarounds
  for the closed-form solve's blind spots.

Pick based on whether existing configs in the wild are relying on the
shapers.

### 3.3 `deltaserve/ft_scheduler_both.py`

Drop the `decode_only_ft_safety_margin` multiplier from the SLO
checks, or default it to 1.0. The iterative loop's per-sample EAGER
prediction now captures the eager-penalty cost; the manual margin is
redundant.

If keeping the knob: leave it at 1.0 by default, document it as
"cold-start conservatism — leave at 1.0 once the EAGER regime has
seen ≥256 records with `T_in == 0, T_ft > 0`."

### 3.4 `profiling_batch_generator.py`

Extend the shape sweep to cover EAGER regime's new shapes that the
default scheduler doesn't exercise:

- `(B_d > 0, K > 0, T_in = 0, T_ft > 0)` — FT on decode-only step
  (unified-phase scheduler's new admission case).
- `(T_in == 0, T_ft > 0, B_d == 0)` — pure FT-fill on idle (already
  exercised by FT-only periods, but worth sweeping explicitly for
  cold-start).

`(B_d > 0, K > 0, T_in > 0, T_ft > 0)` — FT on mixed step — is
already exercised under the default scheduler. Keep.

### 3.5 Migration / backward compatibility

- The CSV format for `batch_prediction_stats_path` should gain a
  `regime` column (the composition-derived regime label, not
  `was_graph`).
- `eager_rmse` / `graph_rmse` properties on `MergedExecutionEstimator`
  can become legacy aliases for `_rmse["eager"]` / `_rmse["decode_only"]`,
  or just drop them (they're only used in the estimator's own log
  line).

---

## 4. Testing

### 4.1 Unit tests

`tests/test_merged_estimator.py` (~21 tests) needs updating:

- Tests exercising `predict(..., will_use_graph=True)` switch to
  constructing features that route to a specific regime (or pass
  `regime=` explicitly).
- Tests checking `_eager.is_fitted` / `_graph.is_fitted` switch to
  `_params["eager"].is_fitted` etc.
- Tests fitting synthetic data on the two-regime layout regenerate
  data for the three composition-based regimes.

New tests:

- `_select` routes to the right regime on representative features
  (one per composition shape).
- `data_fit` partitions correctly (synthetic tracker → verify each
  regime's bucket).
- Cold-start fallback (only one regime fitted → other regimes' predict
  falls back to the fitted one).
- The iterative admission loop's stop conditions (corpus exhausted,
  buffer cap hit, TBT exceeded, TTFT exceeded — one test per).

### 4.2 Live GPU validation

Run `eval/auto_benchmark.py` before and after on the same workload +
same scheduler (default `prefill` mode). Confirm:

- Each regime's RMSE is lower than the old pooled-regime RMSE.
- Admission decisions don't dramatically shift on the default
  scheduler (the new model should make similar decisions in the
  regime distribution the default scheduler exercises).
- Predicted-vs-actual scatter, broken out by composition-regime,
  shows no systematic bias per regime.

Then enable the unified-phase scheduler
(`coserving_admission_phase: both`) and confirm:

- EAGER regime sees decode-only-with-FT records and converges within
  a few hundred steps.
- With `decode_only_ft_safety_margin: 1.0`, TBT SLO satisfaction stays
  within target. If it drifts above SLO consistently, tighten the
  margin (or add more shapes to the offline profiling pass).

### 4.3 Headline check

The `eval/auto_plot_schedulers.py` two-PNG output is the headline. The
`both_vs_prefill` figure should show the unified-phase scheduler
admitting more FT confidently (no longer over-conservative due to the
safety margin) while keeping TBT SLO compliance ≥ the default
scheduler's.

---

## 5. Out of scope

Listed so the implementer doesn't accidentally pull them in:

- The `match_prefill_workload_factor` and
  `ft_tokens_admission_constrain_factor` admission shapers — orthogonal;
  keep wrapping or remove, see §3.2.
- The forward CUDA graphs in the backward subprocess — unrelated to
  the SLO estimator.
- The `_maybe_pause` GPU-yield contract — unrelated.
- The activation-save subset (`save_attn_qkv` / F1) — unrelated.
- The forward_interruptible 3-tier pre-emption — unrelated.

---

## 6. Side-by-side diff

| Aspect | Current (verified in code) | Proposed |
|---|---|---|
| Number of coefficient sets | 2 (`_eager`, `_graph`) | 3 (`inf_prefill`, `eager`, `decode_only`) |
| Partition key | `was_graph` boolean | Composition: `(T_ft > 0)` / `(T_in > 0)` / decode-only |
| Selection arg to `predict()` | `will_use_graph: bool` | Derived from `features`; optional `regime=` override |
| Coefficients per regime | 6 (full formula) for both | inf_prefill: 5 (no γ), eager: 6, decode_only: 3 (no prefill) |
| Subset convention (`T_ft ⊆ T_in`) | yes (correct, keep) | yes (unchanged) |
| Admission algorithm | Closed-form quadratic solve `α·x² + (β+γ)·x ≤ budget` against `_eager` | Iterative per-sample loop; baseline uses DECODE_ONLY/INF_PREFILL, per-iteration uses EAGER |
| Eager-penalty on decode-only FT admit | Hidden in `_eager` regime's bias on out-of-distribution queries | Explicit: emerges as the difference between DECODE_ONLY baseline and EAGER per-sample prediction |
| `decode_only_ft_safety_margin` need | Required (default 0.7) to compensate for unmodeled eager penalty | Unnecessary (default 1.0); keep as cold-start conservatism knob |
| `was_graph` field role | Drives selection AND partition | Vestigial; kept for observability/sanity-check only |
| Cold-start data path | Online refit only | Online refit + profiling-pass extension for `(B_d>0, T_in=0, T_ft>0)` shapes |
| `max_next_ft_tokens` quadratic solve | Used by every admission decision | Deleted (replaced by iterative loop's per-sample predict calls) |

---

*Self-contained spec. Implementer needs read access to
`deltaserve/estimator.py`, `deltaserve/ft_scheduler.py`,
`deltaserve/ft_scheduler_both.py`, `config/finetune.py`,
`profiling_batch_generator.py`, and `tests/test_merged_estimator.py`.*
