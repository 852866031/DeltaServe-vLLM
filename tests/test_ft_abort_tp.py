#!/usr/bin/env python
"""forward_interruptible tier C under TP — the rank-symmetric abort (CPU, gloo).

The hazard: under TP each rank runs the FT-only forward on its own GPU and the
ranks meet in a collective every layer. If one rank aborts at layer k while the
other keeps going, the other blocks in layer k's collective forever. So the
abort must be a joint decision. This test drives ``FtArrivalSignal`` +
``FtAbortPoller`` through a 2-process gloo group exactly as the runner and the
accumulator hooks do, with deliberately ASYMMETRIC observations, and checks:

  1. no arrival → neither entry nor any layer aborts, on both ranks;
  2. an arrival that only rank 0 observes (rank 1's read is stale) → BOTH ranks
     abort at the same layer, the one where rank 0 saw it;
  3. an arrival before the forward starts → both abort at entry; after
     ``finish`` / ``note_served`` the same counter value no longer aborts;
  4. ``hook_check(False)`` (the non-boundary hooks) never runs a collective —
     the per-rank collective counts stay equal (one per boundary + one at entry);
  5. the sentinel ``ModelRunnerOutput.finetune_aborted`` survives pickling,
     which is how it crosses the worker → engine hop under TP.

    python tests/test_ft_abort_tp.py
"""

from __future__ import annotations

import os
import pickle
import sys
import tempfile
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bwd_harness as H  # noqa: E402

sys.path[:] = [p for p in sys.path
               if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]

from vllm.deltaserve.coordinator import FtAbortPoller, FtArrivalSignal  # noqa: E402

PORT = 29730
NL = 6


class _StaleSignal:
    """Rank 1's view: always reads the value it saw at construction — models a
    rank whose read happens just before the engine's bump lands."""

    def __init__(self, real: FtArrivalSignal) -> None:
        self._v = real.read()

    def read(self) -> int:
        return self._v


class _EarlySignal:
    """Rank 0's view: sees the bump from layer ``at`` onwards."""

    def __init__(self, real: FtArrivalSignal, at: int) -> None:
        self._real = real
        self._at = at
        self.calls = 0

    def read(self) -> int:
        self.calls += 1
        # Reads: 1 at construction (seen), 1 at entry(), then one per layer k
        # (call k+3). Return the bumped value from call index ``at`` on, so
        # the arrival becomes visible at layer ``at - 2``.
        return self._real.read() + (1 if self.calls - 1 >= self._at else 0)


def _fake_forward(poller):
    """The runner + hooks: entry check, then per layer the boundary hook plus
    four non-boundary hooks. Returns ('entry' | layer index | None)."""
    if poller.entry():
        return "entry"
    for k in range(NL):
        if poller.hook_check(True):
            return k
        for _ in range(4):
            assert not poller.hook_check(False)
    return None


def _worker(rank, shm_name, outdir, ready1, bumped1, ready2, bumped2):
    try:
        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{PORT}",
                                rank=rank, world_size=2)
        real = FtArrivalSignal.attach(shm_name)
        res = {}
        # 0) not between entry() and finish() (a co-serving batch with inference
        #    tokens): the hooks' check is inert and runs NO collective — the
        #    two ranks would otherwise desync on the collective count.
        poller = FtAbortPoller(real, dist.group.WORLD)
        res["inactive"] = [poller.hook_check(True) for _ in range(3)] + [poller.hook_check(False)]
        res["inactive_reduces"] = poller.reduces
        # 1) quiet forward.
        res["quiet"] = _fake_forward(poller)
        res["quiet_reduces"] = poller.reduces
        poller.finish()
        # 2) asymmetric observation: rank 0 sees an arrival from layer 3 on,
        #    rank 1 never does → both must abort at layer 3.
        sig = _EarlySignal(real, at=5) if rank == 0 else _StaleSignal(real)   # → layer 3
        poller2 = FtAbortPoller(sig, dist.group.WORLD)
        res["asym"] = _fake_forward(poller2)
        res["asym_reduces"] = poller2.reduces
        # 3) a real arrival before the forward → entry abort on both ranks;
        #    finish() accounts for it; the next forward is quiet again.
        dist.barrier()
        if rank == 0:
            ready1.set()                     # both ranks are past the asym test
        bumped1.wait()                       # main has bumped
        poller3 = FtAbortPoller(real, dist.group.WORLD)
        poller3.seen = poller.seen           # what this rank had accounted for
        res["entry"] = _fake_forward(poller3)
        poller3.finish()
        res["after_finish"] = _fake_forward(poller3)
        # note_served semantics: an arrival then an inference batch → not pending.
        dist.barrier()
        if rank == 0:
            ready2.set()
        bumped2.wait()
        poller3.note_served()
        res["after_note_served"] = _fake_forward(poller3)
        torch.save(res, os.path.join(outdir, f"rank{rank}.pt"))
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)


def main():
    C = H.Checker(tol=0.0)
    print("test_ft_abort_tp (2-process gloo, rank-symmetric tier-C decision):")
    sig = FtArrivalSignal.create()
    tmp = tempfile.mkdtemp(prefix="ft_abort_")
    ctx = mp.get_context("spawn")
    ready1, bumped1, ready2, bumped2 = (ctx.Event() for _ in range(4))
    procs = [ctx.Process(target=_worker,
                         args=(rk, sig.name, tmp, ready1, bumped1, ready2, bumped2))
             for rk in range(2)]
    for p in procs:
        p.start()
    # Handshake with the workers: bump only once both ranks are waiting.
    assert ready1.wait(60); sig.bump(); bumped1.set()     # before poller3's forward
    assert ready2.wait(60); sig.bump(); bumped2.set()     # before the note_served check
    for p in procs:
        p.join()
    if any(p.exitcode != 0 for p in procs):
        C.ok("workers exited cleanly", False)
        C.finish()
    r = [torch.load(os.path.join(tmp, f"rank{rk}.pt")) for rk in range(2)]
    C.ok("inactive poller (mixed batch): never aborts, no collective",
         r[0]["inactive"] == [False] * 4 and r[1]["inactive"] == [False] * 4
         and r[0]["inactive_reduces"] == 0 == r[1]["inactive_reduces"])
    C.ok("quiet forward: no abort on either rank",
         r[0]["quiet"] is None and r[1]["quiet"] is None)
    C.ok("quiet forward: 1 entry + NL boundary collectives per rank",
         r[0]["quiet_reduces"] == NL + 1 == r[1]["quiet_reduces"])
    C.ok("asymmetric observation: both ranks abort at layer 3",
         r[0]["asym"] == 3 and r[1]["asym"] == 3, f"got {r[0]['asym']} / {r[1]['asym']}")
    C.ok("asymmetric observation: equal collective counts (no desync)",
         r[0]["asym_reduces"] == r[1]["asym_reduces"] == 1 + 4)
    C.ok("arrival before the forward: both abort at entry",
         r[0]["entry"] == "entry" and r[1]["entry"] == "entry")
    C.ok("after finish(): the same arrival no longer aborts",
         r[0]["after_finish"] is None and r[1]["after_finish"] is None)
    C.ok("after note_served(): a served arrival no longer aborts",
         r[0]["after_note_served"] is None and r[1]["after_note_served"] is None)
    sig.close()

    # 5) sentinel field round-trips the worker → engine serialisation.
    from vllm.v1.outputs import ModelRunnerOutput
    out = ModelRunnerOutput(req_ids=[], req_id_to_index={})
    out.finetune_aborted = True
    back = pickle.loads(pickle.dumps(out))
    C.ok("ModelRunnerOutput.finetune_aborted survives pickle", back.finetune_aborted is True)
    C.ok("default is False", ModelRunnerOutput(req_ids=[], req_id_to_index={}).finetune_aborted is False)
    C.finish()


if __name__ == "__main__":
    main()
