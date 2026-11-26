#!/usr/bin/env python
"""Several backward batches per optimizer step (``finetune_param.ft_step_size``
+ ``bwd_batch_size``). CPU.

  A. Trainer: a step trained as k backward batches (gradients accumulated,
     one clip + update at the end) gives the gradient and the weights of ONE
     backward over the whole step — for Qwen3 and Llama-3, with the batches
     taken as views at their buffer offset. The weights do not move before the
     step's last batch. A step can also end with an empty final batch.
  B. The child's handler (``_handle_process_activations``): the same step driven
     by ``start`` / ``first`` / ``final`` commands over one shared buffer; the
     buffer is not zero-filled between or behind the batches, the ack says
     ``final``.
  C. Coordinator: when batches go out, that admission stays open while a
     non-final batch runs and closes for the final one, the reset after it,
     an epoch flush, the empty final batch, the TP relay command; and that a
     buffer of one batch behaves as before the feature.
  D. Config: the two keys, the single-size alias, the multiple rule; the save
     op moves one backward batch's rows, not the buffer's.
  E. The runner pauses the child on a step that carries FT tokens.

    python tests/test_grad_accum.py
"""

import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bwd_harness as H  # noqa: E402

sys.path[:] = [p for p in sys.path
               if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]

from deltaserve.bwd_services.llama3 import Llama3BackwardService  # noqa: E402
from deltaserve.bwd_services.qwen3 import Qwen3BackwardService  # noqa: E402
from deltaserve.coordinator import FinetuneCoordinator  # noqa: E402

C = H.Checker(tol=2e-5)
# The weights after an AdamW step: the update is lr * g / (|g| + eps) on the
# first step, which magnifies a last-bit difference in a small gradient.
W_TOL = 2e-4
NS = types.SimpleNamespace
VOCAB = 40
SEQ = [5, 4, 6, 3, 7, 5, 4, 6]            # 40 rows
GROUPS = [(0, 3), (3, 5), (5, 8)]         # batches of 15, 10, 15 rows


def build(cls, seed=3):
    svc, embed = H.build_synthetic_service(
        cls, hidden=48, n_layers=2, Hq=4, Hkv=2, Hd=16, inter=128, vocab=VOCAB,
        r=8, lr=5e-3, theta=1e6, eps=1e-6, seed=seed)
    # B = 0 at init makes every A gradient zero: start from a trained-looking
    # adapter so all eight factors of a layer carry gradient.
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for ld in svc.lora.values():
            for ab in ld.values():
                ab["B"].copy_(torch.randn(ab["B"].shape, generator=g) * 0.05)
    return svc, embed


def params(svc):
    return [p for ld in svc.lora.values() for ab in ld.values() for p in ab.values()]


def snapshot(svc):
    return torch.cat([p.detach().flatten().clone() for p in params(svc)])


def grads(svc):
    return torch.cat([p.grad.detach().flatten().clone() for p in params(svc)])


def capture(svc, embed):
    torch.manual_seed(5)
    ids = torch.randint(0, VOCAB, (sum(SEQ),))
    b_start, _, _ = H.seq_layout(SEQ, 16, 1e6)
    return H.forward_capture(svc, embed, ids, SEQ, b_start)


def view(acts, start):
    return {k: ([t[start:] for t in v] if isinstance(v, list) else v[start:])
            for k, v in acts.items()}


def run_batches(svc, acts, last_is_final=True):
    row = 0
    for gi, (a, b) in enumerate(GROUPS):
        lens = SEQ[a:b]
        svc._step_first = gi == 0
        svc._step_final = last_is_final and gi == len(GROUPS) - 1
        svc.process_backward(view(acts, row), lens, sum(lens), epoch=0)
        row += sum(lens)


# ─── A. the trainer ─────────────────────────────────────────────────────────
print("A. trainer: k batches == one backward over the step")
for cls in (Qwen3BackwardService, Llama3BackwardService):
    name = cls.family.name
    one, embed = build(cls)
    acts = capture(one, embed)
    w0 = snapshot(one)
    one.process_backward(acts, SEQ, sum(SEQ), epoch=0)
    g_one, w_one = grads(one), snapshot(one)

    acc, embed2 = build(cls)
    acts2 = capture(acc, embed2)
    C.check(f"{name}: same start", snapshot(acc), w0, tol=1e-12)
    # stop before the last batch: nothing may have moved yet
    row = 0
    for gi, (a, b) in enumerate(GROUPS[:-1]):
        acc._step_first, acc._step_final = gi == 0, False
        acc.process_backward(view(acts2, row), SEQ[a:b], sum(SEQ[a:b]), epoch=0)
        row += sum(SEQ[a:b])
    C.ok(f"{name}: weights unchanged before the step's last batch",
         torch.equal(snapshot(acc), w0))
    a, b = GROUPS[-1]
    acc._step_first, acc._step_final = False, True
    acc.process_backward(view(acts2, row), SEQ[a:b], sum(SEQ[a:b]), epoch=0)
    C.ok(f"{name}: the step moved the weights", not torch.equal(snapshot(acc), w0))
    C.check(f"{name}: step gradient (after the clip)", grads(acc), g_one)
    C.check(f"{name}: weights after the step", snapshot(acc), w_one, tol=W_TOL)

    # every batch non-final, then the update alone (an empty final batch)
    late, embed3 = build(cls)
    run_batches(late, capture(late, embed3), last_is_final=False)
    C.ok(f"{name}: all batches non-final: still unchanged",
         torch.equal(snapshot(late), w0))
    late.finish_step(epoch=0)
    C.check(f"{name}: finish_step applies the same update", snapshot(late), w_one,
            tol=W_TOL)
    w_late = snapshot(late)
    late.finish_step(epoch=0)
    C.ok(f"{name}: a second finish_step is a no-op",
         torch.equal(snapshot(late), w_late))

    # a second step starts clean (no gradient carried over)
    acts_b = capture(one, embed)
    one.process_backward(acts_b, SEQ, sum(SEQ), epoch=0)
    run_batches(acc, capture(acc, embed2))
    C.check(f"{name}: second step matches too", snapshot(acc), snapshot(one),
            tol=10 * W_TOL)


# ─── B. the child's handler over one shared buffer ──────────────────────────
print("B. child handler: start / first / final over the shared buffer")


class Conn:
    def __init__(self):
        self.sent = []

    def send(self, msg):
        self.sent.append(msg)


ref, embed = build(Qwen3BackwardService)
acts = capture(ref, embed)
ref.process_backward(acts, SEQ, sum(SEQ), epoch=0)

svc, embed = build(Qwen3BackwardService)
buf = {k: ([t.clone() for t in v] if isinstance(v, list) else v.clone())
       for k, v in capture(svc, embed).items()}
svc.activations = buf
conn = Conn()
row = 0
for gi, (a, b) in enumerate(GROUPS):
    lens = SEQ[a:b]
    final = gi == len(GROUPS) - 1
    svc._handle_process_activations(conn, {
        "cmd": "process_activations", "n": sum(lens), "sample_lens": lens,
        "epoch": 0, "start": row, "first": gi == 0, "final": final})
    row += sum(lens)
    if not final:
        C.ok(f"batch {gi}: buffer left alone (the forward is still writing)",
             float(buf["final_in"].detach().abs().sum()) > 0
             and conn.sent[-1]["final"] is False)
C.ok("final batch: ack says final; the buffer is not zero-filled behind it "
     "(the next step's forward may already be writing)",
     conn.sent[-1]["final"] is True
     and float(buf["final_in"].detach().abs().sum()) > 0)
C.ok("one ack per batch with its row count",
     [m["n"] for m in conn.sent] == [15, 10, 15], str([m["n"] for m in conn.sent]))
C.check("handler-driven step == one backward", snapshot(svc), snapshot(ref),
        tol=W_TOL)

svc, embed = build(Qwen3BackwardService)
svc.activations = {k: ([t.clone() for t in v] if isinstance(v, list) else v.clone())
                   for k, v in capture(svc, embed).items()}
conn, row = Conn(), 0
for gi, (a, b) in enumerate(GROUPS):
    lens = SEQ[a:b]
    svc._handle_process_activations(conn, {
        "n": sum(lens), "sample_lens": lens, "epoch": 0, "start": row,
        "first": gi == 0, "final": False})
    row += sum(lens)
svc._handle_process_activations(conn, {
    "n": 0, "sample_lens": [], "epoch": 0, "start": row, "first": False,
    "final": True})
C.check("empty final batch through the handler == one backward",
        snapshot(svc), snapshot(ref), tol=W_TOL)

svc, embed = build(Qwen3BackwardService)
svc.activations = capture(svc, embed)
conn = Conn()
svc._handle_process_activations(conn, {
    "cmd": "process_activations", "n": sum(SEQ), "sample_lens": SEQ, "epoch": 0})
C.check("a command without the new fields is one whole-buffer step",
        snapshot(svc), snapshot(ref), tol=1e-12)
C.ok("... and that one zero-fills the buffer, as before",
     float(svc.activations["final_in"].detach().abs().sum()) == 0)


# ─── C. the coordinator ─────────────────────────────────────────────────────
print("C. coordinator: batches inside a step")


class Child:
    def __init__(self):
        self.calls = []
        self._resp = None

    def notify_buffer_full(self, n, sample_lens=None, epoch=0, start=0,
                           first=True, final=True):
        self.calls.append(NS(n=n, lens=list(sample_lens or []), start=start,
                             first=first, final=final))

    def poll_response(self):
        r, self._resp = self._resp, None
        return r

    def done(self):
        c = self.calls[-1]
        self._resp = {"event": "activations_processed", "n": c.n, "loss": 1.0,
                      "final": c.final}

    def set_pause(self, paused):
        pass


class Evt:
    def synchronize(self):
        pass


def coord(capacity=48, bwd=16, relay=False):
    c = FinetuneCoordinator(capacity, per_step_budget=bwd)
    c.bwd_batch = bwd
    c.min_sample_len = 4
    c.capture_done_evt = Evt()
    c.trained = []
    c.on_backward_done = c.trained.append
    if relay:
        c.relay_mode = True
    else:
        c.backward_process = Child()
    return c


def forward(c, lens):
    """One FT step: reserve, then the runner's commit."""
    samples = [NS(request_id=f"s{id(c)}-{len(c.buffer_samples) + i}")
               for i in range(len(lens))]
    c.note_injection(4, admitted_now=sum(lens))
    c.reserve(sum(lens), samples)
    c.record_capture(sum(lens), lens)


c = coord()
ch = c.backward_process
C.ok("multi_batch with a batch below the capacity", c.multi_batch)
forward(c, [6, 6])
C.ok("12 of 16 rows: no batch yet, admission open",
     not ch.calls and c.next_ft_budget() == 16)
forward(c, [6, 5])
b = ch.calls[-1]
C.ok("23 rows: first batch = the samples that fit 16 (two of 6), first, not final",
     len(ch.calls) == 1 and (b.n, b.lens, b.start, b.first, b.final)
     == (12, [6, 6], 0, True, False), str(vars(b)))
C.ok("... admission stays open while it runs (the weights do not change)",
     c.pending_backward and c.next_ft_budget() == 16 and c.admission_open)
forward(c, [6, 6, 6])
C.ok("more rows while it runs: no second batch at once", len(ch.calls) == 1)
ch.done()
c.poll_backward()
b = ch.calls[-1]
C.ok("its ack sends the next: rows 12.., [6, 5], not first, not final",
     len(ch.calls) == 2 and (b.n, b.lens, b.start, b.first, b.final)
     == (11, [6, 5], 12, False, False), str(vars(b)))
C.ok("... nothing committed as trained yet, buffer kept",
     c.trained == [] and c.fill_count == 41)
forward(c, [4])                      # 45 of 48: no sample (>= 4) fits any more
C.ok("the buffer cannot grow: admission closed", c.next_ft_budget() == 0)
ch.done()
c.poll_backward()
b = ch.calls[-1]
C.ok("third batch [6, 6], more left: not final",
     (b.n, b.lens, b.start, b.final) == (12, [6, 6], 23, False), str(vars(b)))
ch.done()
c.poll_backward()
b = ch.calls[-1]
C.ok("last batch [6, 4] empties the closed buffer: FINAL",
     (b.n, b.lens, b.start, b.first, b.final) == (10, [6, 4], 35, False, True),
     str(vars(b)))
C.ok("... admission closed for the batch that ends in the update",
     c.next_ft_budget() == 0 and not c.admission_open)
c.note_injection(4)                  # a schedule() while the final batch runs
ch.done()
c.poll_backward()
C.ok("... a schedule during the final batch does not mark the NEXT step closed",
     not c.epoch_flush_pending)
C.ok("final ack: the step's samples committed once, buffer reset, open again",
     len(c.trained) == 1 and len(c.trained[0]) == 8 and c.fill_count == 0
     and c._bwd_rows == 0 and c._bwd_samples == 0 and not c.pending_backward
     and c.next_ft_budget() == 16)
C.ok("four batches for one step", len(ch.calls) == 4)
forward(c, [6, 6])
C.ok("the next step starts like the first: no early final batch",
     len(ch.calls) == 4 and c.next_ft_budget() == 16)
C.ok("backward_blocks_admission: only for a batch that ends in the update",
     not c.backward_blocks_admission)

# epoch flush on a partial buffer, with one batch already running
c = coord()
ch = c.backward_process
forward(c, [6, 6, 6])                # batch [6, 6] goes out, 6 rows stay
c.request_epoch_flush()
c.try_epoch_flush()
C.ok("epoch flush while a batch runs: nothing more yet", len(ch.calls) == 1)
ch.done()
c.poll_backward()
b = ch.calls[-1]
C.ok("its ack sends the tail as the FINAL batch",
     (b.n, b.lens, b.start, b.final) == (6, [6], 12, True), str(vars(b)))
ch.done()
c.poll_backward()
C.ok("... and the flush flag is consumed",
     not c.epoch_flush_pending and c.fill_count == 0)

# every row consumed before the step closes: an empty final batch
c = coord()
ch = c.backward_process
forward(c, [6, 6, 4])                # exactly 16: one full batch, nothing left
C.ok("exactly one batch's rows: a non-final batch",
     (ch.calls[-1].n, ch.calls[-1].final) == (16, False))
ch.done()
c.poll_backward()
C.ok("step still open: no further batch", len(ch.calls) == 1)
c.request_epoch_flush()
c.try_epoch_flush()
b = ch.calls[-1]
C.ok("the epoch ends there: an EMPTY final batch (the update alone)",
     len(ch.calls) == 2 and (b.n, b.lens, b.start, b.first, b.final)
     == (0, [], 16, False, True), str(vars(b)))
ch.done()
c.poll_backward()
C.ok("... which resets the step", c.fill_count == 0 and c.next_ft_budget() == 16)

# an FT-only step that tier C may still abort: nothing is reserved behind it
c = coord()
ch = c.backward_process
c.reserve(6, [NS(request_id="a")], abortable=True)
C.ok("abortable FT-only step in flight: no FT for the next step",
     c.abortable_in_flight and c.next_ft_budget() == 0)
c.release_reserve(6, [NS(request_id="a")])           # tier C aborted it
C.ok("aborted and rolled back: admission again, from the same rows",
     not c.abortable_in_flight and c.next_ft_budget() == 16
     and c.current_offset() == 0)
c.reserve(6, [NS(request_id="b")], abortable=True)
c.record_capture(6, [6])
C.ok("committed: the next step may carry FT, behind it",
     c.next_ft_budget() == 16 and c.current_offset() == 6)
c.reserve(6, [NS(request_id="c")], abortable=False)   # a mixed step: never aborted
C.ok("a mixed step in flight does not hold the next one back",
     c.next_ft_budget() == 16 and c.current_offset() == 12)
c.record_capture(6, [6])
one = coord(capacity=16, bwd=16)
one.reserve(6, [NS(request_id="d")], abortable=True)
C.ok("one backward per step: the rule is not applied (as before)",
     one.next_ft_budget() == 10)

# TP relay: the scheduler-side coordinator emits the batch on the command
c = coord(relay=True)
forward(c, [6, 6, 6])
cmd = c.take_trigger_cmd()
C.ok("relay command carries start / first / final",
     cmd is not None and (cmd["n"], cmd["sample_lens"], cmd["start"],
                          cmd["first"], cmd["final"]) == (12, [6, 6], 0, True, False),
     str(cmd))
c.apply_relayed_done({"event": "activations_processed", "n": 12, "final": False})
C.ok("relayed ack of a non-final batch: buffer kept, child free",
     c.fill_count == 18 and c._bwd_rows == 12 and not c.pending_backward)
w = FinetuneCoordinator(48, per_step_budget=16)
w.backward_process, w.capture_done_evt = Child(), Evt()
w.execute_trigger({"n": 12, "sample_lens": [6, 6], "epoch": 0, "start": 0,
                   "first": True, "final": False})
b = w.backward_process.calls[-1]
C.ok("worker side passes them to its child",
     (b.n, b.start, b.first, b.final) == (12, 0, True, False))

# one batch per step: exactly the behaviour before the feature
c = coord(capacity=16, bwd=16)
ch = c.backward_process
C.ok("batch == capacity: not multi_batch", not c.multi_batch)
forward(c, [6, 6])
C.ok("partial buffer: no backward", not ch.calls and c.next_ft_budget() == 4)
forward(c, [4])
b = ch.calls[-1]
C.ok("full buffer: ONE backward of everything, first and final",
     len(ch.calls) == 1 and (b.n, b.lens, b.start, b.first, b.final)
     == (16, [6, 6, 4], 0, True, True), str(vars(b)))
C.ok("... admission closed for all of it",
     c.next_ft_budget() == 0 and not c.admission_open)
ch.done()
c.poll_backward()
C.ok("... its ack resets and reopens", c.fill_count == 0 and c.admission_open
     and len(c.trained) == 1 and len(c.trained[0]) == 3)


# -- [bwd_whole_batches] the step ends when its batches are full --------------
def run_step(c, sample_len, child=None, max_steps=50):
    """The scheduler's side of one optimizer step with samples of one length:
    ask for the cap, admit what fits, tell the coordinator, reserve, commit.
    Acks every batch at once. Returns the rows admitted per engine step."""
    took = []
    for _ in range(max_steps):
        if c.next_ft_budget() <= 0:
            break
        cap = min(c.space_remaining(sample_len), c.per_step_budget)
        k = cap // sample_len
        if k:
            lens = [sample_len] * k
            samples = [NS(request_id=f"w{id(c)}-{len(c.buffer_samples) + i}",
                          input_len=sample_len) for i in range(k)]
        else:
            lens, samples = [], []
        c.note_injection(sample_len, admitted_now=sum(lens))
        if lens:
            c.reserve(sum(lens), samples)
            c.record_capture(sum(lens), lens)
            took.append(sum(lens))
        else:
            c.try_epoch_flush()
        while child is not None and c.pending_backward:
            child.done()
            c.poll_backward()
        if child is not None and child.calls and child.calls[-1].final \
                and c.fill_count == 0:
            break                     # the step's last batch was acked
    return took


c = coord()                           # 48 rows, batches of 16, samples of 6
ch = c.backward_process
took = run_step(c, 6, ch)
C.ok("rows-full step (default): 48 rows of 6-token samples = 4 backwards, the "
     "last of them the remainder",
     [b.n for b in ch.calls] == [12, 12, 12, 12] and ch.calls[-1].final
     and sum(took) == 48, str([b.n for b in ch.calls]))
c = coord(capacity=48, bwd=16)
c.whole_batches = True
ch = c.backward_process
took = run_step(c, 6, ch)
C.ok("whole batches: the step ends with its 3 batches full (36 rows), no "
     "fourth backward",
     [b.n for b in ch.calls] == [12, 12, 12] and ch.calls[-1].final
     and [b.final for b in ch.calls] == [False, False, True]
     and sum(took) == 36, str([(b.n, b.final) for b in ch.calls]))
C.ok("... one admission stays inside one batch", all(t <= 12 for t in took),
     str(took))
C.ok("... and the next step starts clean",
     c.fill_count == 0 and c.admission_open and c._step_lens == []
     and c.space_remaining(6) == 16)
c = coord(capacity=64, bwd=16)
c.whole_batches = True
ch = c.backward_process
took = run_step(c, 5, ch)             # 3 samples of 5 per batch
C.ok("whole batches, 64 / 16 with 5-token samples: 4 batches of 15",
     [b.n for b in ch.calls] == [15, 15, 15, 15] and ch.calls[-1].final,
     str([b.n for b in ch.calls]))
c = coord(capacity=48, bwd=16)
c.whole_batches = True
s1 = [NS(request_id="r1", input_len=6), NS(request_id="r2", input_len=6)]
c.reserve(12, s1)
s2 = [NS(request_id="r3", input_len=6)]
c.reserve(6, s2, abortable=True)
c.release_reserve(6, s2)
C.ok("an aborted reservation gives its place in the batch back",
     c._step_lens == [6, 6] and c.space_remaining(4) == 4
     and c.space_remaining(6) == 16)
c = coord(capacity=16, bwd=16)
c.whole_batches = True
C.ok("one batch per step: the key changes nothing",
     c.space_remaining(6) == 16 and not c.multi_batch)

# -- [bwd_batch_size_idle] one larger backward while no inference is around ---
def fill(c, lens_per_step):
    for lens in lens_per_step:
        samples = [NS(request_id=f"i{id(c)}-{len(c.buffer_samples) + i}",
                      input_len=l) for i, l in enumerate(lens)]
        c.note_injection(6, admitted_now=sum(lens))
        c.reserve(sum(lens), samples)
        c.record_capture(sum(lens), lens)


c = coord(capacity=48, bwd=16)
c.bwd_batch_idle = 48
c.idle_now = True
ch = c.backward_process
fill(c, [[6, 6], [6, 6], [6, 6]])
C.ok("idle: no small batch goes out while the step is still filling",
     not ch.calls and c.next_ft_budget() > 0)
fill(c, [[6, 6]])                      # 48 rows: the step is full
b = ch.calls[-1]
C.ok("idle: the whole step in ONE backward, first and final",
     len(ch.calls) == 1 and (b.n, b.start, b.first, b.final) == (48, 0, True, True)
     and c.idle_batches == 1, str(vars(b)))
ch.done()
c.poll_backward()
c.idle_now = False                     # a request arrived
fill(c, [[6, 6], [6, 5]])
b = ch.calls[-1]
C.ok("inference present: batches of the normal size again",
     len(ch.calls) == 2 and (b.n, b.first, b.final) == (12, True, False), str(vars(b)))
c.idle_now = True                      # it left again, mid-step
ch.done()
c.poll_backward()
fill(c, [[6, 6], [6, 6]])              # 47 of 48 rows, nothing (>= 4) fits
b = ch.calls[-1]
C.ok("idle again mid-step: the rest of the step in one backward, final",
     len(ch.calls) == 3 and (b.n, b.start, b.first, b.final) == (35, 12, False, True),
     str(vars(b)))
c = coord(capacity=48, bwd=16)
c.idle_now = True
ch = c.backward_process
fill(c, [[6, 6], [6, 5]])
C.ok("key unset (idle size = batch size): idle changes nothing",
     len(ch.calls) == 1 and ch.calls[-1].n == 12 and c.idle_batches == 0)
from deltaserve.ft_sizes import idle_bwd_tokens_of  # noqa: E402
from deltaserve.config import FinetuneConfig as _FC  # noqa: E402
C.ok("bwd_batch_size_idle: unset = the batch; set = itself, at most the step; "
     "an FT-only step may then carry as much",
     idle_bwd_tokens_of(_FC(ft_step_size=1024, bwd_batch_size=256)) == 256
     and idle_bwd_tokens_of(_FC(ft_step_size=1024, bwd_batch_size=256,
                                bwd_batch_size_idle=1024)) == 1024
     and idle_bwd_tokens_of(_FC(ft_step_size=1024, bwd_batch_size=256,
                                bwd_batch_size_idle=4096)) == 1024)
try:
    _FC(ft_step_size=1024, bwd_batch_size=256, bwd_batch_size_idle=128).check_step_sizes()
    _bad = False
except ValueError:
    _bad = True
C.ok("an idle size below the batch is refused", _bad)

# ─── D. config ──────────────────────────────────────────────────────────────
print("D. config")
from deltaserve.accumulate import FinetuneAccumulator  # noqa: E402
from deltaserve.config import FinetuneConfig  # noqa: E402
from deltaserve.ft_sizes import bwd_tokens_of, step_tokens_of  # noqa: E402
from deltaserve.config_loader import FINETUNE_SECTIONS  # noqa: E402

cfg = FinetuneConfig()
C.ok("neither key: both sizes are max_saved_finetuning_tokens",
     (cfg.step_tokens, cfg.bwd_tokens) == (256, 256))
cfg = FinetuneConfig(max_saved_finetuning_tokens=600)
C.ok("the single-size key still sets both",
     (cfg.step_tokens, cfg.bwd_tokens) == (600, 600))
cfg = FinetuneConfig(ft_step_size=2048, bwd_batch_size=512)
cfg.check_step_sizes()
C.ok("ft_step_size 2048 / bwd_batch_size 512",
     (cfg.step_tokens, cfg.bwd_tokens) == (2048, 512))
cfg = FinetuneConfig(ft_step_size=1024)
C.ok("ft_step_size alone: one backward per step",
     (cfg.step_tokens, cfg.bwd_tokens) == (1024, 1024))
for bad in (dict(ft_step_size=1000, bwd_batch_size=512),
            dict(ft_step_size=256, bwd_batch_size=512)):
    try:
        FinetuneConfig(**bad).check_step_sizes()
        refused = False
    except ValueError:
        refused = True
    C.ok(f"{bad} is refused", refused)
C.ok("both keys listed under finetune_param",
     {"ft_step_size", "bwd_batch_size"} <= set(FINETUNE_SECTIONS["finetune_param"]))
from deltaserve.ft_sizes import fwd_tokens_of  # noqa: E402
C.ok("ft_tokens_per_step: unset = the backward batch; set = itself, never "
     "above the batch",
     fwd_tokens_of(FinetuneConfig(ft_step_size=1024, bwd_batch_size=256)) == 256
     and fwd_tokens_of(FinetuneConfig(max_saved_finetuning_tokens=1024,
                                      ft_tokens_per_step=256)) == 256
     and fwd_tokens_of(FinetuneConfig(max_saved_finetuning_tokens=256,
                                      ft_tokens_per_step=999)) == 256
     and FinetuneConfig().ft_tokens_per_step is None
     and "ft_tokens_per_step" in FINETUNE_SECTIONS["finetune_param"])
from deltaserve.ft_sizes import ft_only_tokens_of  # noqa: E402
_c = FinetuneConfig(max_saved_finetuning_tokens=1024, ft_tokens_per_step=256)
C.ok("ft_only_tokens_per_step: unset = ft_tokens_per_step; set = itself, "
     "between that and the backward batch",
     ft_only_tokens_of(_c) == 256
     and ft_only_tokens_of(FinetuneConfig(
         max_saved_finetuning_tokens=1024, ft_tokens_per_step=256,
         ft_only_tokens_per_step=1024)) == 1024
     and ft_only_tokens_of(FinetuneConfig(
         max_saved_finetuning_tokens=1024, ft_tokens_per_step=256,
         ft_only_tokens_per_step=64)) == 256
     and ft_only_tokens_of(FinetuneConfig(
         max_saved_finetuning_tokens=512, ft_only_tokens_per_step=4096)) == 512
     and "ft_only_tokens_per_step" in FINETUNE_SECTIONS["finetune_param"])
c = FinetuneCoordinator(64, per_step_budget=16)      # one backward of 64, steps of 16
c.ft_started = True
c.min_sample_len = 4
C.ok("one backward per step, filled over several forward steps: not "
     "multi_batch, 16 tokens a step", not c.multi_batch and c.next_ft_budget() == 16)
c.reserve(16, [NS(request_id="a", input_len=16)], abortable=True)
C.ok("... and nothing is reserved behind an FT-only step that may still be "
     "aborted (its rows would leave a hole)", c.next_ft_budget() == 0)
c.record_capture(16, [16])
C.ok("... its commit reopens the budget", c.next_ft_budget() == 16)
old = NS(max_saved_finetuning_tokens=16)
C.ok("a config object with only the old key resolves",
     (step_tokens_of(old), bwd_tokens_of(old)) == (16, 16))

model = torch.nn.Module()
accu = FinetuneAccumulator(model, max_saved=1024, hidden_size=8,
                           device="cpu", dtype=torch.float32, op_rows=256)
C.ok("save op moves one backward batch's rows; the buffer holds the step",
     accu.ft_src.numel() == 256 and accu.ft_dst.numel() == 256
     and accu.concat_input_ids.numel() == 1024
     and int(accu.ft_dst[0]) == 1024)
accu = FinetuneAccumulator(model, max_saved=256, hidden_size=8,
                           device="cpu", dtype=torch.float32)
C.ok("without op_rows: the buffer's size, as before", accu.ft_src.numel() == 256)


# ─── E. the runner's pause rule ─────────────────────────────────────────────
print("E. runner: a step with FT tokens pauses the child")
from deltaserve.runner_common import FtRunnerMixin  # noqa: E402

pb = FtRunnerMixin._ft_pauses_backward
C.ok("several batches per step + FT tokens: pause",
     pb(NS(t_ft=40, t_in=40, b_d=0), NS(multi_batch=True)))
C.ok("... not for a step without FT tokens",
     not pb(NS(t_ft=0, t_in=40, b_d=0), NS(multi_batch=True)))
C.ok("one batch per step: never (as before)",
     not pb(NS(t_ft=40, t_in=40, b_d=0), NS(multi_batch=False)))


class R(FtRunnerMixin):
    pass


r = R()
r._ds_init()
r.vllm_config = NS(finetune_config=FinetuneConfig(enable_finetuning=True))
paused = []
co = NS(pending_backward=True, multi_batch=True, fwd_throttle_active=False,
        gpu_pause_backward=lambda: paused.append(1),
        gpu_resume_backward=lambda: paused.append(0))
so = NS(_ft_step_features=NS(t_in=40, t_ft=40, b_d=0))
C.ok("pause decision: FT-only step with a batch in flight pauses it",
     r._ft_pause_decision(so, co) == (True, False) and paused == [1])
co.multi_batch = False
paused.clear()
C.ok("... and does not when a step is one backward",
     r._ft_pause_decision(so, co) == (False, False) and paused == [])

C.finish()
