#!/usr/bin/env python
"""The co-serving hooks on vLLM's v2 model runner (``vllm/v1/worker/gpu``,
the only runner from vLLM v0.34 on). CPU: the stock runner's GPU work is
replaced by a stand-in that makes the same calls in the same order
(``gather_batch_req_state`` -> ``cudagraph_manager.dispatch`` ->
``step_timing.forward_start`` -> the model), so what is checked is where the
hooks sit and what they do, not vLLM.

  A. ``cudagraph_v2.FtCudaGraphManager`` on a real ``ModelCudaGraphManager``:
     the ``has_ft`` key set, dispatch for no-FT / eager / graph steps, the
     scheduler's ``will_use_graph`` query (no shared state touched), the
     forward-context descriptor swap in ``run_pw_graph``, and the second
     capture pass (its own descriptors, the save op live, state restored).
  B. ``model_runner_v2.DeltaServeModelRunnerV2``: an inference step, an
     FT-only eager step (``skip_compiled`` inside the forward context, hook
     arming, the saved rows committed), a mixed graph step (save op armed for
     the forward only), tier-C aborts at entry and mid-forward (roll-back
     sentinel, ``sample_tokens`` answers with it), a failing forward, the TP
     relay fields on idle and sampled outputs, retired FT requests, dummy
     runs.
  C. ``finetune_features.model_runner``: auto / v1 / v2, the config key, and
     the runner a real ``VllmConfig`` then selects.

    python tests/test_runner_v2.py
"""

import os
import sys
import threading
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bwd_harness as H  # noqa: E402

sys.path[:] = [p for p in sys.path
               if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]

import inspect  # noqa: E402

# The checks below are about the config key; the variable would override it.
os.environ.pop("DSERVE_MODEL_RUNNER", None)

from vllm.config import CUDAGraphMode  # noqa: E402
from vllm.forward_context import BatchDescriptor  # noqa: E402
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput  # noqa: E402
from vllm.v1.worker.gpu import cudagraph_utils as cgu  # noqa: E402
from vllm.v1.worker.gpu import model_runner as mr_mod  # noqa: E402
from vllm.v1.worker.gpu.async_utils import StepTimingCollector  # noqa: E402

import deltaserve.cudagraph_v2 as cg2  # noqa: E402
import deltaserve.ft_save_op as save_op  # noqa: E402
import deltaserve.model_runner_v2 as mr2  # noqa: E402
from deltaserve.config import FinetuneConfig  # noqa: E402
from deltaserve.coordinator import FTAborted, LocalAbortPoller  # noqa: E402
from deltaserve.cudagraph_common import FtBatchDescriptor  # noqa: E402
from deltaserve.runner_common import _empty_output  # noqa: E402

C = H.Checker(tol=1e-6)
NS = types.SimpleNamespace
NONE, PW, FULL = CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL
MAX_LORAS = 2
LORA_CASE = MAX_LORAS + 1            # what a LoRA batch dispatches to


# ─── a real graph manager on the CPU ────────────────────────────────────────
def make_vllm_config(**ft):
    ft_cfg = FinetuneConfig(enable_finetuning=True, **ft)
    comp = NS(cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32],
              max_cudagraph_capture_size=32,
              max_decode_cudagraph_capture_tokens=None,
              prefill_cudagraph_capture_token_sizes=[1, 2, 4, 8, 16, 32],
              decode_cudagraph_capture_request_sizes=[1, 2, 4, 8],
              use_breakable_cudagraph=False)
    return NS(finetune_config=ft_cfg, compilation_config=comp,
              scheduler_config=NS(max_num_seqs=8),
              parallel_config=NS(tensor_parallel_size=1, data_parallel_size=1),
              cache_config=NS(use_kda_recoverssm=False),
              speculative_config=None, num_speculative_tokens=0,
              uniform_decode_query_len=1,
              get_decode_cudagraph_query_lens=lambda q: {q: 8})


def make_manager(vcfg, mode=CUDAGraphMode.FULL_AND_PIECEWISE):
    cgu.get_pp_group = lambda: NS(is_first_rank=True, is_last_rank=True)
    cgu.current_platform = NS(get_global_graph_pool=lambda: None)
    if hasattr(cgu, "is_breakable_cudagraph_enabled"):
        cgu.is_breakable_cudagraph_enabled = lambda *a, **k: False
    params = inspect.signature(cgu.ModelCudaGraphManager.__init__).parameters
    kw = ({"decode_query_lens": {1: 8}} if "decode_query_lens" in params
          else {"decode_query_len": 1})
    return cgu.ModelCudaGraphManager(
        vcfg, torch.device("cpu"), mode,
        lora_capture_cases=[0, LORA_CASE], **kw)


print("A. FtCudaGraphManager")
vcfg = make_vllm_config(graph_ft_batches=True, graph_ft_max_tokens=16,
                        forward_interruptible=True)
ACC = NS(capture_state=lambda: "capture-state")
mgr = cg2.FtCudaGraphManager.adopt(make_manager(vcfg), lambda: ACC)
C.ok("adopted by class swap", type(mgr) is cg2.FtCudaGraphManager)
C.ok("has_ft keys: LoRA sizes <= ft_max",
     mgr._ft_keys == {(n, LORA_CASE) for n in (1, 2, 4, 8, 16)},
     str(sorted(mgr._ft_keys)))

d = mgr.dispatch(3, 5, None, 1)
C.ok("nothing captured yet -> eager", d.cg_mode == NONE)
mgr._graphs_captured = True
d = mgr.dispatch(3, 5, None, 1)
C.ok("no FT, mixed batch -> PIECEWISE(8)",
     d.cg_mode == PW and d.num_tokens == 8 and d.num_active_loras == LORA_CASE,
     str(d))
C.ok("last_desc is the runner's dispatch", mgr.last_desc is d)
d = mgr.dispatch(4, 4, 1, 0, 1)
C.ok("no FT, uniform decode -> FULL", d.cg_mode == FULL, str(d))

mgr.ft_mode = "eager"
d = mgr.dispatch(3, 5, None, 1)
C.ok("eager FT step -> NONE, size kept",
     d.cg_mode == NONE and d.num_tokens == 5 and d.num_reqs == 3, str(d))
d = mgr.dispatch(4, 4, 1, 1, 1)
C.ok("eager FT step, decode shape -> NONE", d.cg_mode == NONE)

mgr.ft_mode = "graph"
d = mgr.dispatch(3, 5, None, 1)
C.ok("graph FT step -> PIECEWISE(8)", d.cg_mode == PW and d.num_tokens == 8)
d = mgr.dispatch(4, 4, 1, 1, 1)
C.ok("graph FT step never takes a FULL decode graph",
     d.cg_mode == PW and d.num_tokens == 4, str(d))
d = mgr.dispatch(3, 20, None, 1)
C.ok("graph FT step above graph_ft_max_tokens -> NONE",
     d.cg_mode == NONE and d.num_tokens == 20, str(d))
d = mgr.dispatch(3, 5, None, 0)
C.ok("graph FT step without an adapter -> NONE", d.cg_mode == NONE)

mgr.ft_mode, mgr.last_desc = None, "untouched"
C.ok("will_use_graph: no FT, mixed", mgr.will_use_graph(5, 3, False, 1, None))
C.ok("will_use_graph: no FT, decode", mgr.will_use_graph(4, 4, True, 0, None))
C.ok("will_use_graph: mixed FT on the save op",
     mgr.will_use_graph(5, 3, False, 1, True))
C.ok("will_use_graph: mixed FT too large",
     not mgr.will_use_graph(20, 3, False, 1, True))
C.ok("will_use_graph: eager FT", not mgr.will_use_graph(5, 3, False, 1, False))
C.ok("will_use_graph leaves the runner's state alone",
     mgr.ft_mode is None and mgr.last_desc == "untouched")

CTX = NS(batch_descriptor=None, skip_compiled=False)
cg2.get_forward_context = lambda: CTX
plain = BatchDescriptor(num_tokens=8, num_active_loras=LORA_CASE)
CTX.batch_descriptor = plain
mgr.run_pw_graph(lambda **kw: None, {})
C.ok("run_pw_graph: no FT keeps the plain descriptor",
     CTX.batch_descriptor is plain)
mgr.ft_mode = "graph"
mgr.run_pw_graph(lambda **kw: None, {})
ftd = CTX.batch_descriptor
C.ok("run_pw_graph: graph FT step swaps in FtBatchDescriptor",
     isinstance(ftd, FtBatchDescriptor) and ftd != plain
     and ftd.num_tokens == 8 and ftd.num_active_loras == LORA_CASE, repr(ftd))
mgr.ft_mode = None

passes = []
orig_capture = cgu.ModelCudaGraphManager.capture


def fake_capture(self, *args, **kwargs):
    CTX.batch_descriptor = BatchDescriptor(num_tokens=4,
                                           num_active_loras=LORA_CASE)
    self.run_pw_graph(lambda **kw: None, {})
    passes.append(NS(descs=dict(self._capture_descs), state=save_op._ACTIVE,
                     ft=isinstance(CTX.batch_descriptor, FtBatchDescriptor),
                     name=kwargs.get("progress_bar_desc")))
    if kwargs.get("fail") and len(passes) == 2:
        raise RuntimeError("capture failed")


cgu.ModelCudaGraphManager.capture = fake_capture
stock_descs = mgr._capture_descs
mgr.capture("model")
C.ok("capture: two passes", len(passes) == 2, str(len(passes)))
C.ok("pass 1 is upstream's own (plain descriptors, op off)",
     passes[0].descs is not None and passes[0].state is None
     and not passes[0].ft and set(passes[0].descs) == set(stock_descs))
C.ok("pass 2: the has_ft sizes only, save op live, own graph table",
     list(passes[1].descs) == [PW]
     and {(x.num_tokens, x.num_active_loras) for x in passes[1].descs[PW]}
     == mgr._ft_keys
     and passes[1].state == "capture-state" and passes[1].ft
     and passes[1].name == cg2._PASS_NAME)
C.ok("capture: state restored",
     mgr._capture_descs is stock_descs and not mgr._ft_capturing
     and save_op._ACTIVE is None)
passes.clear()
try:
    mgr.capture("model", fail=True)
    raised = False
except RuntimeError:
    raised = True
C.ok("capture: a failing has_ft pass restores the state too",
     raised and mgr._capture_descs is stock_descs and not mgr._ft_capturing
     and save_op._ACTIVE is None)
cgu.ModelCudaGraphManager.capture = orig_capture

off = cg2.FtCudaGraphManager.adopt(
    make_manager(make_vllm_config(graph_ft_batches=False)), lambda: ACC)
C.ok("graph_ft_batches off: no has_ft set", not off._ft_descs)
off._graphs_captured = True
off.ft_mode = "graph"
C.ok("... and a graph-mode dispatch is eager",
     off.dispatch(3, 5, None, 1).cg_mode == NONE)


# ─── the runner, with a stand-in for the stock one's GPU work ───────────────
print("B. DeltaServeModelRunnerV2")


class FakeEvent:
    def __init__(self, *a, **k):
        self.recorded = False

    def record(self):
        self.recorded = True

    def query(self):
        return True

    def elapsed_time(self, other):
        return 1.0


torch.cuda.Event = FakeEvent
LOG: list = []                       # the order things happen in
FORWARD = {"fn": None}               # what the stand-in forward does


def base_init(self, vllm_config, device):
    self.vllm_config, self.device = vllm_config, device
    self.step_timing = StepTimingCollector()
    self.cudagraph_manager = None
    self.execute_model_state = None
    self.is_pooling_model = False
    self.model = torch.nn.Identity()
    self.removed = []


def base_gather(self, scheduler_output, dummy_run):
    if dummy_run:
        return None, None
    ids = sorted(scheduler_output.num_scheduled_tokens)      # "batch order"
    toks = np.array([scheduler_output.num_scheduled_tokens[i] for i in ids],
                    dtype=np.int32)
    return NS(req_ids=ids, num_scheduled_tokens=toks), None


def base_execute(self, scheduler_output, intermediate_tensors=None,
                 dummy_run=False, **kwargs):
    if dummy_run:
        LOG.append("dummy")
        self.step_timing.forward_start()
        return None
    self.finish_requests(scheduler_output)
    total = scheduler_output.total_num_scheduled_tokens
    if total == 0:
        return EMPTY_MODEL_RUNNER_OUTPUT
    state, _ = self.gather_batch_req_state(scheduler_output, dummy_run)
    desc = self.cudagraph_manager.dispatch(len(state.req_ids), total, None, 1)
    LOG.append(("dispatch", self.cudagraph_manager.ft_mode, desc.cg_mode))
    self.step_timing.forward_start()
    hidden = torch.arange(desc.num_tokens * 4, dtype=torch.float32).view(-1, 4)
    ids = torch.arange(100, 100 + desc.num_tokens)
    CTX.skip_compiled = False
    self.model(hidden)                       # inside the "forward context"
    LOG.append(("forward", CTX.skip_compiled, save_op._ACTIVE,
                self._ds_skip_compiled))
    if FORWARD["fn"] is not None:
        FORWARD["fn"](self)
    self.execute_model_state = NS(hidden_states=hidden,
                                  input_batch=NS(input_ids=ids))
    return None


def base_sample(self, grammar_output):
    if self.execute_model_state is None:
        return None
    self.execute_model_state = None
    return NS(model_runner_output=_empty_output())


def base_finish(self, scheduler_output):
    LOG.append("finish_requests")


def base_remove(self, req_id):
    self.removed.append(req_id)
    return True


G = mr_mod.GPUModelRunner
G.__init__ = base_init
G.gather_batch_req_state = base_gather
G.execute_model = base_execute
G.sample_tokens = base_sample
G.finish_requests = base_finish
G._remove_request = base_remove
G.initialize_kv_cache = lambda self, *a, **k: setattr(
    self, "cudagraph_manager", make_manager(self.vllm_config))
mr2.get_forward_context = lambda: CTX


class Coord:
    def __init__(self, relay=False):
        self.relay_mode = relay
        self.pending_backward = False
        self.fwd_throttle_active = False
        self.cudagraph_dispatcher = None
        self.ft_abort_event = threading.Event()
        self.ft_abort_poller = LocalAbortPoller(self.ft_abort_event)
        self.ft_only_in_flight = False
        self.record_timing = False
        self.capture_done_evt = None
        self.ft_started = True
        self.profiling = False
        self.captured = []
        self.samples = []
        self.acks = []

    def gpu_pause_backward(self):
        LOG.append("pause")

    def gpu_resume_backward(self):
        LOG.append("resume")

    def execute_trigger(self, cmd):
        LOG.append(("trigger", cmd))

    def push_sample(self, *a):
        self.samples.append(a)

    def current_offset(self):
        return 7

    def record_capture(self, n, lens):
        self.captured.append((n, lens))

    def relay_backward_outstanding(self):
        return bool(self.acks)

    def poll_own_backward_ack(self):
        return True

    def take_relay_ack(self):
        return self.acks.pop()

    def drain_completed_samples(self):
        out, self.samples = self.samples, []
        return out


class Acc:
    def __init__(self, op_ready):
        self.op_ready = op_ready
        self.calls = []

    def begin_step(self, mask, n, offset, start=0, contiguous=False):
        self.calls.append(("begin_step", mask.tolist(), n, offset, start,
                           contiguous))

    def arm_op_step(self, positions, n, offset):
        self.calls.append(("arm_op_step", list(positions), n, offset))
        return "step-state"

    def accumulate_final(self, hidden, ids, mask, n, offset, start=0,
                         contiguous=False):
        self.calls.append(("accumulate_final", tuple(hidden.shape),
                           ids.tolist(), n, offset))

    def end_step(self):
        self.calls.append(("end_step",))

    def zero_offset_range(self, off, n):
        self.calls.append(("zero", off, n))

    def capture_state(self):
        return "capture-state"

    def names(self):
        return [c[0] for c in self.calls]


def make_runner(relay=False, op_ready=False, **ft):
    ft.setdefault("forward_interruptible", True)
    ft.setdefault("graph_ft_batches", True)
    ft.setdefault("graph_ft_max_tokens", 16)
    r = mr2.DeltaServeModelRunnerV2(make_vllm_config(**ft), torch.device("cpu"))
    r.initialize_kv_cache(None)
    r.cudagraph_manager._graphs_captured = True
    r._ft_coordinator = Coord(relay)
    r._finetune_accumulator = Acc(op_ready)
    return r


def sched(tokens: dict, ft=(), **extra):
    return NS(num_scheduled_tokens=dict(tokens),
              total_num_scheduled_tokens=sum(tokens.values()),
              finetune_req_ids=set(ft), **extra)


def step(r, so, fwd=None):
    LOG.clear()
    FORWARD["fn"] = fwd
    return r.execute_model(so)


r = make_runner()
C.ok("constructed through the stock __init__",
     type(r.step_timing) is mr2._FtStepTiming and r._ft_has is False
     and type(r.cudagraph_manager) is cg2.FtCudaGraphManager
     and r.cudagraph_dispatcher is r.cudagraph_manager)

# 1. inference step, with a backward pending: the child is paused before the
#    forward and the resume waits for the prefill (pause_until_prefill_done).
r._ft_coordinator.pending_backward = True
out = step(r, sched({"a": 3, "b": 2},
                    _ft_step_features=NS(t_in=5, t_ft=0, b_d=0)))
acc = r._finetune_accumulator
C.ok("inference step: stock output, plain dispatch",
     out is None and ("dispatch", None, PW) in LOG, str(LOG))
C.ok("... backward paused before the forward, not resumed by it",
     LOG.index("pause") < [i for i, x in enumerate(LOG)
                           if isinstance(x, tuple) and x[0] == "forward"][0]
     and "resume" not in LOG and r._ft_resume_evt is not None, str(LOG))
C.ok("... no FT work", acc.calls == [] and r._ft_coordinator.captured == [])
C.ok("... the scheduler is handed the graph manager",
     r._ft_coordinator.cudagraph_dispatcher is r.cudagraph_manager)
C.ok("... per-step state cleared",
     r.cudagraph_manager.ft_mode is None and r._ds_sched is None
     and r._ds_fwd is None and not r._ds_skip_compiled)
r.sample_tokens(None)
C.ok("... sample_tokens resumes it once the prefill is done",
     LOG[-1] == "resume" and r._ft_resume_evt is None, str(LOG))
r._ft_coordinator.pending_backward = False

# 2. FT-only step on the hook path: eager, skip_compiled inside the forward.
out = step(r, sched({"__ft__1": 3, "__ft__0": 2}, ft={"__ft__0", "__ft__1"},
                    _ft_write_offset=40))
fw = [x for x in LOG if isinstance(x, tuple) and x[0] == "forward"][0]
C.ok("FT-only step: dispatched eager", ("dispatch", "eager", NONE) in LOG)
C.ok("... skip_compiled set inside the forward context, save op off",
     fw[1] is True and fw[2] is None and fw[3] is True, str(fw))
C.ok("... hooks armed with the batch-order mask at the reserved offset",
     acc.calls[0] == ("begin_step", [True] * 5, 5, 40, 0, True),
     str(acc.calls[0]))
C.ok("... input ids + rows saved after the forward, then disarmed",
     acc.calls[1] == ("accumulate_final", (5, 4), [100, 101, 102, 103, 104],
                      5, 40) and acc.calls[2] == ("end_step",),
     str(acc.calls[1:]))
C.ok("... rows committed with per-sample lengths in batch order",
     r._ft_coordinator.captured == [(5, [2, 3])],
     str(r._ft_coordinator.captured))
C.ok("... the in-flight marker is cleared",
     not r._ft_coordinator.ft_only_in_flight and not r._ds_skip_compiled)
out = step(r, sched({"a": 3}))
fw = [x for x in LOG if isinstance(x, tuple) and x[0] == "forward"][0]
C.ok("the next inference step runs compiled again", fw[1] is False, str(fw))

# 3. mixed step on the save-op path: has_ft graph, op armed for the forward.
r = make_runner(op_ready=True)
acc = r._finetune_accumulator
out = step(r, sched({"a": 2, "__ft__0": 3, "b": 1}, ft={"__ft__0"}))
fw = [x for x in LOG if isinstance(x, tuple) and x[0] == "forward"][0]
C.ok("mixed step: dispatched into the has_ft set",
     ("dispatch", "graph", PW) in LOG, str(LOG))
C.ok("... save op armed with the FT rows, live during the forward only",
     acc.calls[0] == ("arm_op_step", [0, 1, 2], 3, 7)
     and fw[2] == "step-state" and fw[1] is False
     and save_op._ACTIVE is None, f"{acc.calls[0]} {fw}")
C.ok("... padded output: ids sliced to the unpadded batch",
     acc.calls[1][0] == "accumulate_final" and acc.calls[1][1] == (8, 4)
     and len(acc.calls[1][2]) == 6, str(acc.calls[1]))
C.ok("... never aborted (inference tokens present)",
     not r._ft_coordinator.ft_only_in_flight
     and r._ft_coordinator.captured == [(3, [3])])

# 4. tier C: a late arrival aborts an FT-only forward mid-way.
r = make_runner()
acc, coord = r._finetune_accumulator, r._ft_coordinator


def late_arrival(runner):
    assert runner._ft_coordinator.ft_only_in_flight
    raise FTAborted()


out = step(r, sched({"__ft__0": 4}, ft={"__ft__0"}), fwd=late_arrival)
C.ok("mid-forward abort: the roll-back sentinel",
     isinstance(out, ModelRunnerOutput)
     and getattr(out, "finetune_aborted", False) and out.req_ids == [])
C.ok("... hooks disarmed, partial rows zeroed, nothing committed",
     acc.names() == ["begin_step", "end_step", "zero"]
     and acc.calls[-1] == ("zero", 7, 4) and coord.captured == [],
     str(acc.calls))
C.ok("... marker cleared, per-step state cleared",
     not coord.ft_only_in_flight and r.cudagraph_manager.ft_mode is None
     and not r._ds_skip_compiled and save_op._ACTIVE is None)
C.ok("... sample_tokens answers with the sentinel, once",
     getattr(r.sample_tokens(None), "finetune_aborted", False)
     and r.sample_tokens(None) is None)

# 5. tier C at entry: the flag was set before this FT-only batch started.
coord.ft_abort_event.set()
acc.calls.clear()
out = step(r, sched({"__ft__0": 4}, ft={"__ft__0"}))
C.ok("entry abort: sentinel, the forward never ran",
     getattr(out, "finetune_aborted", False)
     and not any(isinstance(x, tuple) and x[0] == "forward" for x in LOG),
     str(LOG))
C.ok("... state settled", not coord.ft_only_in_flight
     and "end_step" in acc.names() and r._ds_fwd is None)
coord.ft_abort_event.clear()
coord.ft_abort_poller.finish()

# 6. a forward that fails for another reason propagates, state settled.
r = make_runner()


def boom(runner):
    raise RuntimeError("cuda died")


try:
    step(r, sched({"__ft__0": 4}, ft={"__ft__0"}), fwd=boom)
    raised = False
except RuntimeError:
    raised = True
C.ok("failing forward: raised to the engine", raised)
C.ok("... state settled, nothing committed",
     r.cudagraph_manager.ft_mode is None and r._ds_fwd is None
     and not r._ds_skip_compiled and not r._ft_coordinator.ft_only_in_flight
     and r._ft_coordinator.captured == [] and not r._ft_pending_abort)

# 7. TP relay: idle steps and sampled outputs carry the relay fields.
import vllm.distributed.parallel_state as ps  # noqa: E402

ps.get_tp_group = lambda: NS(world_size=1)   # one rank: no ack all-reduce
r = make_runner(relay=True)
coord = r._ft_coordinator
coord.acks.append({"ack": 1})
out = step(r, sched({}, finetune_backward_trigger="go"))
C.ok("idle step: a private copy of EMPTY with the relay fields",
     out is not EMPTY_MODEL_RUNNER_OUTPUT and out.finetune_backward_done
     == {"ack": 1} and out.finetune_ft_started is True
     and not hasattr(EMPTY_MODEL_RUNNER_OUTPUT, "finetune_backward_done"))
C.ok("... and the relayed trigger fired", ("trigger", "go") in LOG, str(LOG))
out = step(r, sched({"__ft__0": 3}, ft={"__ft__0"}))
C.ok("FT step under TP: rows relayed, not committed here",
     out is None and coord.captured == [])
res = r.sample_tokens(None)
C.ok("... on the output the scheduler receives",
     res.model_runner_output.finetune_saved == (3, [3])
     and res.model_runner_output.finetune_backward_done is None)

# 8. retired FT requests are freed before the stock finished path, sorted.
r = make_runner()
step(r, sched({"a": 1}, finetune_retired_req_ids=["__ft__9", "__ft__10",
                                                    "__ft__2"]))
C.ok("retired FT requests removed (sorted), then the stock path",
     r.removed == ["__ft__10", "__ft__2", "__ft__9"]
     and LOG[0] == "finish_requests", f"{r.removed} {LOG[:1]}")

# 9. dummy runs (warmup / capture) pass straight through.
LOG.clear()
r._ft_coordinator.pending_backward = True
r.execute_model(sched({"a": 4}), dummy_run=True)
C.ok("dummy run: none of the hooks", LOG == ["dummy"] and r._ds_fwd is None,
     str(LOG))


# ─── the config key ─────────────────────────────────────────────────────────
print("C. finetune_features.model_runner")
import deltaserve.runner_select as rs  # noqa: E402
from deltaserve.config_loader import FINETUNE_SECTIONS  # noqa: E402

C.ok("default is auto", FinetuneConfig().model_runner == "auto")
C.ok("listed under finetune_features",
     "model_runner" in FINETUNE_SECTIONS["finetune_features"])
has_v1 = rs.v1_runner_available()
C.ok("auto: v1 where vLLM ships it, else v2",
     rs.wants_v2(NS(model_runner="auto")) == (not has_v1), f"has_v1={has_v1}")
C.ok("v2 is v2", rs.wants_v2(NS(model_runner="v2")))
rs.v1_runner_available = lambda: True
C.ok("v1 on a vLLM with it", not rs.wants_v2(NS(model_runner="v1")))
rs.v1_runner_available = lambda: False
C.ok("auto without a v1 runner is v2", rs.wants_v2(NS(model_runner="auto")))
try:
    rs.wants_v2(NS(model_runner="v1"))
    refused = False
except ValueError:
    refused = True
C.ok("v1 without a v1 runner is refused", refused)
try:
    rs.wants_v2(NS(model_runner="v3"))
    refused = False
except ValueError:
    refused = True
C.ok("an unknown runner is refused", refused)

# Through the real VllmConfig: vLLM <= 0.33 picks a runner per launch and the
# plugin steers it; later versions have no such choice (v2 only).
import deltaserve.plugin as plugin  # noqa: E402
from vllm.config import VllmConfig  # noqa: E402

rs.v1_runner_available = lambda: has_v1
plugin.register()
if hasattr(VllmConfig, "use_v2_model_runner"):
    def launch(choice):
        return VllmConfig(additional_config={"deltaserve": {
            "enable_finetuning": True, "model_runner": choice}})

    C.ok("VllmConfig: auto keeps the v1 runner",
         launch("auto").use_v2_model_runner is False)
    C.ok("VllmConfig: v1 keeps the v1 runner",
         launch("v1").use_v2_model_runner is False)
    c = launch("v2")
    C.ok("VllmConfig: v2 selects the v2 runner, same worker class",
         c.use_v2_model_runner is True
         and c.parallel_config.worker_cls == "deltaserve.worker.DeltaServeWorker")
else:
    print("  (this vLLM has only the v2 runner)")

C.finish()
