#!/usr/bin/env python
"""M4.2 — the SLO-estimator timing relay + backward-ack agreement under TP.

Under the multiproc executor the worker's and the scheduler's coordinators are
different objects in different processes. This test drives the two halves of
the relay on the CPU, exactly as the runner and the scheduler do:

  worker coord  --push_sample-->  drain  --pickle (ModelRunnerOutput)-->
  scheduler coord  --push_sample-->  drain  -->  tracker  -->  estimator ready

plus the record-timing gate on ``SchedulerOutput`` and the all-rank ack
agreement (``relay_backward_outstanding`` / ``poll_own_backward_ack`` /
``take_relay_ack``) with a fake MIN all-reduce across two "ranks".

    python tests/test_tp_timing_relay.py
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bwd_harness as H  # noqa: E402

sys.path[:] = [p for p in sys.path
               if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]

from vllm.deltaserve.coordinator import FinetuneCoordinator  # noqa: E402
from vllm.deltaserve.estimator import (  # noqa: E402
    MIN_FIT_SAMPLES, REGIME_EAGER, REGIME_INF_PREFILL, MergedExecutionEstimator,
    StepExecutionTracker, StepFeatures)
from vllm.v1.core.sched.output import SchedulerOutput  # noqa: E402
from vllm.v1.outputs import ModelRunnerOutput  # noqa: E402

C = H.Checker(tol=1e-9)


def _coord(relay: bool) -> FinetuneCoordinator:
    c = FinetuneCoordinator(256, 256)
    c.relay_mode = relay
    return c


def test_timing_round_trip():
    print("test_timing_round_trip (worker coord → pickle → scheduler coord → estimator):")
    worker, sched = _coord(True), _coord(True)

    # The runner pushes (features, duration, was_graph, predicted) per step.
    # 12 inference-prefill + 12 co-serving (eager) steps with a known linear cost.
    def dur(f):
        return 1e-4 * f.t_in + 5e-4 * f.t_ft + 2e-3

    for i in range(12):
        f = StepFeatures(t_in=64 + 8 * i, p=2, t_ft=0, b_d=0, k=0,
                         prefill_lens=[32 + 4 * i, 32 + 4 * i])
        worker.push_sample(f, dur(f), False, None)
        g = StepFeatures(t_in=96 + 8 * i, p=3, t_ft=32, b_d=0, k=0,
                         prefill_lens=[32 + 4 * i, 32 + 4 * i, 32])
        worker.push_sample(g, dur(g), False, 0.01)

    # sample_tokens: drain onto the output; the executor pickles it across.
    out = ModelRunnerOutput(req_ids=[], req_id_to_index={}, sampled_token_ids=[],
                            logprobs=None, prompt_logprobs_dict={}, pooler_output=[])
    C.ok("ModelRunnerOutput.finetune_timing defaults None", out.finetune_timing is None)
    out.finetune_timing = worker.drain_completed_samples() or None
    C.ok("worker queue drained", not worker._completed_samples)
    wire = pickle.loads(pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL))
    C.ok("24 samples survive pickling", len(wire.finetune_timing) == 24)

    # update_from_output: push into the scheduler coord; schedule(): drain → tracker.
    for t in wire.finetune_timing:
        sched.push_sample(*t)
    tracker = StepExecutionTracker()
    for feats, d, was_graph, predicted, *_ in sched.drain_completed_samples():
        if d > 0:
            tracker.add(feats, d, predicted=predicted, was_graph=was_graph)
    C.ok("tracker holds every relayed sample", tracker.size() == 24)
    est = MergedExecutionEstimator()
    C.ok("estimator cold before fit", not est.is_ready)
    est.data_fit(tracker)
    C.ok("estimator ready after relay", est.is_ready)
    f = StepFeatures(t_in=200, p=2, t_ft=64, b_d=0, k=0, prefill_lens=[100, 100])
    pred = est.predict(f, regime=REGIME_EAGER, apply_margin=False)
    C.check("eager prediction recovers the cost", H.torch.tensor(pred), H.torch.tensor(dur(f)), tol=5e-2)
    C.ok("both regimes fitted", est._params[REGIME_EAGER].is_fitted
         and est._params[REGIME_INF_PREFILL].is_fitted and 12 >= MIN_FIT_SAMPLES)


def test_record_timing_gate():
    print("test_record_timing_gate (SchedulerOutput.finetune_record_timing):")
    so = SchedulerOutput.make_empty()
    C.ok("defaults to True (record)", so.finetune_record_timing is True)
    so.finetune_record_timing = False
    wire = pickle.loads(pickle.dumps(so))
    C.ok("False survives the broadcast pickle", wire.finetune_record_timing is False)
    # The runner reads it with a fallback to the local coordinator flag.
    c = _coord(True)
    c.record_timing = True
    C.ok("runner gate prefers the stamped value",
         bool(getattr(wire, "finetune_record_timing", c.record_timing)) is False)


class _FakeChild:
    """Stands in for BackwardProcess: acks on the poll number given; records
    the corpus-meta / work messages in the order they were sent."""

    def __init__(self, ack_at: int) -> None:
        self.ack_at = ack_at
        self.polls = 0
        self.sent: list = []

    def set_corpus_meta(self, total):
        self.sent.append(("set_corpus_meta", int(total)))

    def notify_buffer_full(self, n, sleep_s=2.0, sample_lens=None, epoch=0):
        self.sent.append(("process_activations", int(n)))

    def poll_response(self):
        self.polls += 1
        return {"event": "activations_processed", "n": 7, "loss": 1.0} \
            if self.polls >= self.ack_at else None


def _min_all_reduce(flags):
    return min(flags)


def test_ack_agreement():
    print("test_ack_agreement (ack relayed only when every rank's child is done):")
    ranks = [_coord(True), _coord(True)]
    ranks[0].backward_process = _FakeChild(ack_at=1)   # rank 0 finishes first
    ranks[1].backward_process = _FakeChild(ack_at=3)   # rank 1 three steps later
    for c in ranks:
        c.pending_backward = True   # what execute_trigger sets on every rank
    C.ok("outstanding on both after trigger",
         all(c.relay_backward_outstanding() for c in ranks))

    relayed = {0: None, 1: None}
    for step in range(1, 5):
        mine = [c.poll_own_backward_ack() for c in ranks]
        all_done = _min_all_reduce([1 if m else 0 for m in mine]) == 1
        for r, c in enumerate(ranks):
            ack = c.take_relay_ack() if all_done else None
            if ack is not None:
                relayed[r] = (step, ack)
        if step < 3:
            C.ok(f"step {step}: rank0 done, rank1 not, nothing relayed",
                 mine == [True, False] and all(v is None for v in relayed.values()))
            C.ok(f"step {step}: still outstanding on both",
                 all(c.relay_backward_outstanding() for c in ranks))
    C.ok("relayed on step 3 on both ranks",
         relayed[0] is not None and relayed[1] is not None
         and relayed[0][0] == 3 and relayed[1][0] == 3)
    C.ok("rank 0 held its ack until agreement (not lost)",
         relayed[0][1]["n"] == 7)
    C.ok("cleared on both after relay",
         not any(c.relay_backward_outstanding() for c in ranks)
         and all(c._relay_ack is None and not c.pending_backward for c in ranks))

    # Single-rank semantics of the legacy helper are unchanged.
    solo = _coord(True)
    solo.backward_process = _FakeChild(ack_at=2)
    solo.pending_backward = True
    C.ok("poll_backward_relay: None before ack", solo.poll_backward_relay() is None)
    C.ok("poll_backward_relay: payload on ack", solo.poll_backward_relay()["n"] == 7)
    C.ok("poll_backward_relay: cleared", not solo.relay_backward_outstanding())


def test_corpus_meta_relay():
    """The corpus total rides on the relayed trigger and reaches each rank's
    child exactly once, before the first work signal (fixes the ``N/?`` meter)."""
    print("test_corpus_meta_relay (scheduler total → trigger → worker child, once):")
    sched = _coord(True)
    sched.corpus_total_tokens = 123456      # what FinetuneScheduler stores
    sched.backward_sleep_s = 0.0
    # Two triggers from the scheduler side (buffer-full twice).
    cmds = []
    for _ in range(2):
        sched.fill_count, sched.sample_lens = 200, [100, 100]
        sched._trigger_backward()
        cmd = sched.take_trigger_cmd()
        cmds.append(pickle.loads(pickle.dumps(cmd)))   # SchedulerOutput round trip
    C.ok("trigger carries the corpus total",
         all(c["total_tokens_per_epoch"] == 123456 for c in cmds))
    C.ok("nothing pending between triggers", sched.take_trigger_cmd() is None)

    worker = _coord(True)
    worker.backward_process = _FakeChild(ack_at=1)
    worker.capture_done_evt = None
    import torch
    _sync = torch.cuda.synchronize
    torch.cuda.synchronize = lambda *a, **k: None      # no GPU in this test
    try:
        for c in cmds:
            worker.execute_trigger(c)
        # tp=1-style coordinator (no relay): a total on the command is harmless.
        solo = _coord(False)
        solo.backward_process = _FakeChild(ack_at=1)
        solo.capture_done_evt = None
        solo.execute_trigger({"n": 1, "total_tokens_per_epoch": 5})
    finally:
        torch.cuda.synchronize = _sync
    sent = worker.backward_process.sent
    C.ok("corpus meta sent once, before the first work signal",
         sent[0] == ("set_corpus_meta", 123456)
         and sum(1 for k, _ in sent if k == "set_corpus_meta") == 1
         and [k for k, _ in sent[1:]] == ["process_activations", "process_activations"])
    C.ok("execute_trigger with a total is harmless on a non-relay coord",
         solo.backward_process.sent[0] == ("set_corpus_meta", 5))


if __name__ == "__main__":
    test_timing_round_trip()
    test_record_timing_gate()
    test_ack_agreement()
    test_corpus_meta_relay()
    C.finish()
