# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""What the EngineCore process does for co-serving, behind a handful of
one-line hooks in v1/engine/core.py (EngineHooks; None when
finetuning is off, so a plain deployment pays one is None check per hook).

- **Arrival signal (tier C under TP).** The workers are separate processes, so
  the tp=1 abort threading.Event cannot reach them: a shared-memory arrival
  counter is created before the executor spawns the workers (its name travels in
  their environment) and bumped by the input thread on every ADD. At tp=1 the
  input thread sets the coordinator's abort event instead while an FT-only
  forward is in flight.
- **Engine inputs for tiers A / B.** The scheduler's grace poll (tier A) and
  late-arrival roll-back (tier B) admit requests that landed on the engine's
  input queue; EngineHooks hands it a narrow view of that queue.
- **Launch-time profiling pass** that seeds the SLO estimator, before the busy loop.
- **Estimator state** saved at the first sign of a shutdown, and on request
  (deltaserve_save_estimator_state utility).
"""

import queue

from vllm.deltaserve import dprint
from vllm.logger import init_logger

logger = init_logger(__name__)


def attach_engine_hooks(engine, vllm_config) -> "EngineHooks | None":
    """Called from EngineCore.__init__ before the executor is built."""
    if not vllm_config.finetune_config.enable_finetuning:
        return None
    return EngineHooks(engine, vllm_config)


class EngineInputs:
    """The scheduler's view of the engine's input queue (tiers A / B)."""

    def __init__(self, engine) -> None:
        self._engine = engine

    def pending(self) -> bool:
        return not self._engine.input_queue.empty()

    def poll(self, timeout_s: float) -> bool:
        """Wait up to timeout_s for a request; if one lands, handle it and
        any that arrived with it (a burst rides the same step). True if any."""
        try:
            req = self._engine.input_queue.get(timeout=timeout_s)
        except queue.Empty:
            return False
        self._engine._handle_client_request(*req)
        self.drain()
        return True

    def drain(self) -> None:
        while True:
            try:
                req = self._engine.input_queue.get_nowait()
            except queue.Empty:
                return
            self._engine._handle_client_request(*req)


class EngineHooks:
    def __init__(self, engine, vllm_config) -> None:
        self.engine = engine
        self.arrival_signal = None
        ft = vllm_config.finetune_config
        if (ft.forward_interruptible
                and vllm_config.parallel_config.tensor_parallel_size > 1):
            import os

            from vllm.deltaserve.coordinator import FtArrivalSignal
            self.arrival_signal = FtArrivalSignal.create()
            os.environ[FtArrivalSignal.ENV] = self.arrival_signal.name
            dprint("[deltaserve] forward_interruptible tier C under TP: arrival "
                   f"signal {self.arrival_signal.name} (rank-symmetric abort)")
        self._coord = None

    def _coordinator(self):
        if self._coord is None:
            self._coord = getattr(getattr(self.engine, "scheduler", None),
                                  "_coord", None)
        return self._coord

    def on_input_add(self) -> None:
        """Input thread, for every ADD: signal a mid-forward abort if an
        FT-only batch is on the GPU (tp=1: the coordinator's Event; TP: the
        arrival counter every rank polls at its layer boundaries)."""
        coord = self._coordinator()
        if coord is not None and coord.ft_only_in_flight:
            coord.ft_abort_event.set()
        if self.arrival_signal is not None:
            self.arrival_signal.bump()

    def before_busy_loop(self) -> None:
        """After the engine is built and connected, before serving."""
        sched = self.engine.scheduler
        if hasattr(self.engine, "input_queue") and hasattr(sched, "set_engine_inputs"):
            sched.set_engine_inputs(EngineInputs(self.engine))
        profile_execution_model(self.engine)

    def save_estimator_state(self) -> None:
        save = getattr(self.engine.scheduler, "save_estimator_state", None)
        if save is not None:
            try:
                save()
            except Exception as e:
                logger.warning("estimator state save failed: %s", e)

    def on_shutdown(self) -> None:
        # Save the estimator state FIRST: tearing down the executor (TP workers
        # + backward children) can outlast the grace period of whoever is
        # stopping the server, and the scheduler's own shutdown hook only runs
        # after it.
        self.save_estimator_state()

    def after_shutdown(self) -> None:
        if self.arrival_signal is not None:
            self.arrival_signal.close()
            self.arrival_signal = None


def _profiling_step(engine) -> None:
    """One synchronous schedule→execute→update step for the launch profiler.

    Unlike `step()`, this materializes the model output via `sample_tokens`,
    so it works whether or not async scheduling is enabled (under async,
    `execute_model` returns a deferred AsyncModelRunnerOutput that must be
    resolved by `sample_tokens`). The profiler always drives one step at a
    time — no pipelining — so reserve/commit reconciles within each step.
    """
    if not engine.scheduler.has_requests():
        return
    scheduler_output = engine.scheduler.schedule()
    exec_future = engine.model_executor.execute_model(
        scheduler_output, non_block=True)
    if scheduler_output.total_num_scheduled_tokens == 0:
        model_output = exec_future.result()
    else:
        # Materialize via the non_block future's .result(): under async that
        # is an AsyncOutputFuture whose result() calls get_output() to turn
        # the deferred AsyncModelRunnerOutput into a real ModelRunnerOutput
        # (matches step_with_batch_queue). A plain sample_tokens() call would
        # hand back the unresolved async wrapper.
        grammar_output = engine.scheduler.get_grammar_bitmask(scheduler_output)
        sample_future = engine.model_executor.sample_tokens(
            grammar_output, non_block=True)
        model_output = sample_future.result()
    engine.scheduler.update_from_output(scheduler_output, model_output)

def save_estimator_state(engine, path: str | None = None) -> dict:
    """[estimator] Utility call behind POST /save_estimator_state: write
    the SLO estimator's state now. ``path`` defaults to
    ``finetune.estimator_state_save_path``."""
    save = getattr(engine.scheduler, "save_estimator_state", None)
    if save is None:
        return {"saved": False, "error": "finetuning is not enabled"}
    path = path or getattr(engine.vllm_config.finetune_config,
                           "estimator_state_save_path", None)
    if not path:
        return {"saved": False, "error": "no path given and "
                "finetune.estimator_state_save_path is not set"}
    save(path)
    return {"saved": True, "path": path,
            "samples": engine.scheduler._tracker.size()}

def profile_execution_model(engine) -> None:
    """[Phase 4] Offline pass that seeds the SLO execution-time estimator.

    Runs synthetic batches through the live FinetuneScheduler before serving
    starts, recording realized (features, GPU-timed duration) per step and
    fitting the merged estimator. The scheduler's own per-step recording
    (Part C) does the bookkeeping; here we just drive controlled shapes and
    discard their outputs (no frontend is consuming yet).
    """
    import random

    ft_cfg = engine.vllm_config.finetune_config
    if not ft_cfg.enable_finetuning:
        return
    if not ft_cfg.profile_on_launch:
        # No profiling pass: a carried-over estimator state is all there is.
        if hasattr(engine.scheduler, "load_estimator_state"):
            engine.scheduler.load_estimator_state()
        return
    sched = engine.scheduler
    # Only the FinetuneScheduler carries the estimator/tracker + hooks.
    if not hasattr(sched, "_estimator"):
        return
    coord = sched._coord

    from vllm.deltaserve import dprint
    from vllm.deltaserve.profiling_batch_generator import (
        ProfilingShapeGenerator,
    )

    try:
        vocab = engine.vllm_config.model_config.get_vocab_size()
    except Exception:
        vocab = 1000
    vocab = max(16, min(int(vocab), 30000))
    rng = random.Random(1234)

    def rand_ids(n: int) -> list[int]:
        return [rng.randint(1, vocab - 1) for _ in range(max(1, n))]

    gen = ProfilingShapeGenerator(
        batch_max_tokens=engine.vllm_config.scheduler_config.max_num_batched_tokens,
        max_saved_finetuning_tokens=ft_cfg.max_saved_finetuning_tokens,
        max_req_len=engine.vllm_config.model_config.max_model_len,
    )
    warmup, recorded = gen.prepare()
    n_repeats = max(1, ft_cfg.profile_num_repeats)
    total_shapes = len(warmup) + n_repeats * len(recorded)
    dprint(f"[ft-profile] starting: {len(warmup)} warmup + "
           f"{len(recorded)} recorded shapes x {n_repeats}")

    from tqdm import tqdm
    pbar = tqdm(total=total_shapes, desc="[ft-profile]", unit="shape",
                dynamic_ncols=True)

    # Save + neutralize state that must not fire during profiling.
    saved_bp = coord.backward_process
    coord.backward_process = None
    sched._profiling_mode = True
    coord.profiling = True

    def reset_coord():
        coord.fill_count = 0
        coord.reserved_fill = 0
        coord.sample_lens = []
        coord.admission_open = True
        coord.pending_backward = False
        coord.epoch_flush_pending = False

    def run_shape(shape) -> None:
        if shape.kind == "prefill":
            for L in shape.inf_lens:
                sched.enqueue_profiling_request(
                    sched.new_profiling_request(rand_ids(L), 1, is_ft=False))
            _profiling_step(engine)
        elif shape.kind == "coserve":
            for L in shape.inf_lens:
                sched.enqueue_profiling_request(
                    sched.new_profiling_request(rand_ids(L), 1, is_ft=False))
            for L in shape.ft_lens:
                sched.enqueue_profiling_request(
                    sched.new_profiling_request(rand_ids(L), 1, is_ft=True))
            _profiling_step(engine)
        elif shape.kind == "decode":
            for L in shape.inf_lens:
                sched.enqueue_profiling_request(
                    sched.new_profiling_request(
                        rand_ids(L), 1 + shape.n_decode, is_ft=False))
            if shape.ft_lens:
                # FT-on-decode: prefill inf reqs (wave 1), then inject FT
                # alongside the first decode step → the recorded step
                # carries (T_in=0, B_d>0, K>0, T_ft>0) — the EAGER regime
                # shape required by the unified-phase scheduler.
                _profiling_step(engine)  # wave-1 prefill
                for L in shape.ft_lens:
                    sched.enqueue_profiling_request(
                        sched.new_profiling_request(
                            rand_ids(L), 1, is_ft=True))
                _profiling_step(engine)  # wave-1 decode + FT (recorded shape)
                # Drain remaining decode steps (one already consumed above).
                for _ in range(max(0, shape.n_decode - 1)):
                    _profiling_step(engine)
            else:
                for _ in range(1 + shape.n_decode):
                    _profiling_step(engine)
        elif shape.kind == "mixed":
            for L in shape.inf_lens:
                sched.enqueue_profiling_request(
                    sched.new_profiling_request(
                        rand_ids(L), 1 + shape.n_decode, is_ft=False))
            _profiling_step(engine)  # wave-1 prefill
            for L in shape.mix_lens:
                sched.enqueue_profiling_request(
                    sched.new_profiling_request(rand_ids(L), 1, is_ft=False))
            # Optional FT alongside wave-2 prefill — gives the EAGER
            # regime ``(T_in>0, B_d>0, K>0, T_ft>0)`` shape data.
            for L in shape.ft_lens:
                sched.enqueue_profiling_request(
                    sched.new_profiling_request(
                        rand_ids(L), 1, is_ft=True))
            _profiling_step(engine)  # wave-1 decode + wave-2 prefill (mixed) [+ FT]
            # drain the rest
            for _ in range(shape.n_decode):
                _profiling_step(engine)
        # Clean up any leftover synthetic requests. Prefix-cache hits are
        # avoided by using distinct random token ids per request, so no
        # per-shape cache reset is needed (it floods the log + the bar).
        sched.purge_profiling_requests()
        # Clear the activation-buffer accounting after EVERY shape so the
        # next shape starts with an empty buffer — otherwise FT rows from
        # earlier shapes accumulate and the buffer-full gate would throttle
        # later co-serve shapes (and shift their write offsets). The backward
        # is detached, so nothing consumes the buffer during profiling.
        reset_coord()

    try:
        # Warmup: don't record (the runner's deferred timer reads this gate
        # at forward time, so warmup samples are never pushed to the tracker).
        coord.record_timing = False
        for shape in warmup:
            run_shape(shape)
            pbar.update(1)
        coord.record_timing = True
        pbar.set_description("[ft-profile] recording")
        for _ in range(n_repeats):
            for shape in recorded:
                run_shape(shape)
                pbar.update(1)
        # Final drain of the coordinator queue into the tracker (samples
        # pushed during the last step, after that step's schedule() drain).
        # The last few steps' events may still sit in the runner's ring and
        # are dropped — negligible against the full sweep.
        for feats, dur, wg, pred, *_ in coord.drain_completed_samples():
            if dur > 0:
                sched._tracker.add(feats, dur, predicted=pred, was_graph=wg)
        sched._estimator.data_fit(sched._tracker)
        dprint(f"[ft-profile] done: {sched._tracker.size()} samples | "
               f"eager_rmse={sched._estimator.eager_rmse} "
               f"graph_rmse={sched._estimator.graph_rmse}")
    finally:
        pbar.close()
        sched._profiling_mode = False
        coord.record_timing = True
        coord.profiling = False
        coord.backward_process = saved_bp
        reset_coord()
        sched.purge_profiling_requests()
        # Single prefix-cache reset so live serving starts clean.
        try:
            engine.scheduler.reset_prefix_cache()
        except Exception:
            pass
        # [estimator_state_load_path] Continue from the state a previous
        # run ended with: it replaces what this pass just fitted (the pass
        # still ran, for its warm-up of the FT paths).
        try:
            sched.load_estimator_state()
        except Exception as e:
            dprint(f"[ft-profile] estimator state restore failed: {e}")
