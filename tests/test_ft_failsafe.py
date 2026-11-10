#!/usr/bin/env python
"""Finetuning fails loudly or degrades — never silently, never a hang. CPU.

  A. trainer state build: a failing ``_build_state`` in a REAL spawned
     backward child comes back in the ``weights_received`` reply (the worker
     raises on it and the launch aborts); a good build replies ``error=None``.
  B. ``disable_finetuning`` is sticky: ``start_finetuning`` refuses, and the TP
     mirror of the workers' ``ft_started`` flag (``_apply_relay``) cannot
     reopen admission — before, a "backward child DEAD → FINETUNING DISABLED"
     was undone on the next TP step. The stuck detector goes through it.
  C. a failing launch profiling pass (``DSERVE_TEST_FAIL_PROFILING``) is
     caught in ``before_busy_loop``: finetuning disabled with the reason,
     nothing raised (the engine serves inference).
  D. status: backward acks are booked on both paths (tp=1 ``poll_backward``,
     TP ``apply_relayed_done``); ``FinetuneScheduler.finetuning_status`` and
     the EngineCore utility.
  E. HTTP: POST /save_adapter (default path from ``adapter_save_path``, 400
     without one, 200 on the child's result file, 500 on its error file, 202
     without ``wait`` or on timeout, 409 when no child took it) and GET
     /finetuning_status, against a fake engine client.

    python tests/test_ft_failsafe.py
"""

import asyncio
import json
import os
import sys
import tempfile
import threading
import time
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bwd_harness as H  # noqa: E402

sys.path[:] = [p for p in sys.path
               if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]

from deltaserve.adapter_io import write_peft_adapter, write_save_error  # noqa: E402
from deltaserve.coordinator import FinetuneCoordinator  # noqa: E402

C = H.Checker(tol=1e-5)


def synthetic_weights():
    from deltaserve.bwd_services.llama3 import Llama3BackwardService
    svc, _ = H.build_synthetic_service(
        Llama3BackwardService, hidden=32, n_layers=2, Hq=4, Hkv=2, Hd=8,
        inter=64, vocab=40, r=4, lr=1e-3)
    return svc.shared


def part_a():
    print("A. trainer state build errors reach the worker")
    from deltaserve.backward_process import BackwardProcess

    shared = synthetic_weights()
    broken = {k: v for k, v in shared["base"].items()
              if "layers.1.mlp.down_proj" not in k}
    for label, base, meta, want_err in [
            ("good weights → error=None", shared["base"], shared["meta"], None),
            ("missing base weight → error", broken, shared["meta"], "down_proj"),
            ("bad TP geometry → error, before any rendezvous", shared["base"],
             dict(shared["meta"], tp_size=3), "divisible"),
            ("missing optimizer setting → error", shared["base"],
             {k: v for k, v in shared["meta"].items() if k != "learning_rate"},
             "learning_rate")]:
        bp = BackwardProcess(service_name="llama3")
        bp.start()
        try:
            t0 = time.monotonic()
            reply = bp.share_weights(base, shared["ft"], meta=meta, timeout=60)
            dt = time.monotonic() - t0
        finally:
            bp.shutdown()
        err = reply.get("error")
        ok = (err is None) if want_err is None else (err is not None and want_err in err)
        C.ok(label, ok and reply.get("event") == "weights_received",
             f"{dt:.1f}s error={str(err)[:60]}")
    # The worker turns a non-empty error into a RuntimeError (launch aborts).
    src = open(os.path.join(os.path.dirname(H.__file__), "..", "deltaserve-plugin",
                            "deltaserve", "worker.py")).read()
    C.ok("worker raises on the reply's error",
         'if summary.get("error"):' in src and "could not" in src)


def part_b():
    print("\nB. disable_finetuning is sticky (tp=1 and the TP mirror)")
    from deltaserve.ft_scheduler import FinetuneScheduler

    c = FinetuneCoordinator(capacity=64)
    C.ok("start before disable → True", c.start_finetuning() is True)
    c.disable_finetuning("test reason")
    C.ok("disabled closes admission", not c.ft_started and not c.admission_open)
    C.ok("start after disable → False", c.start_finetuning() is False
         and not c.ft_started)
    c.disable_finetuning("second reason")
    C.ok("first reason kept", c.ft_disabled_reason == "test reason")

    def relay(coord, started):
        fake = types.SimpleNamespace(_coord=coord)
        out = types.SimpleNamespace(finetune_ft_started=started,
                                    finetune_timing=None,
                                    finetune_backward_done=None,
                                    finetune_saved=None)
        FinetuneScheduler._apply_relay(fake, out)

    c.relay_mode = True
    relay(c, True)
    C.ok("TP mirror cannot reopen a disabled run", not c.ft_started)
    c2 = FinetuneCoordinator(capacity=64)
    c2.relay_mode, c2.ft_started = True, False
    relay(c2, True)
    C.ok("TP mirror still starts a healthy run", c2.ft_started)
    relay(c2, False)
    C.ok("TP mirror still stops it", not c2.ft_started)

    c3 = FinetuneCoordinator(capacity=64)
    c3.pending_backward = True
    c3.pending_backward_warn_s = 0.0
    c3._pending_backward_t0 = time.monotonic() - 1.0
    c3._check_backward_stuck(alive=False)
    C.ok("stuck detector disables with its reason",
         not c3.ft_started and "DEAD" in (c3.ft_disabled_reason or ""),
         str(c3.ft_disabled_reason))
    c3.relay_mode = True
    relay(c3, True)
    C.ok("... and the TP mirror keeps it off", not c3.ft_started)


def fake_engine(profile_on_launch=True):
    coord = FinetuneCoordinator(capacity=64)
    sched = types.SimpleNamespace(
        _coord=coord, _estimator=object(), load_estimator_state=lambda: None,
        set_engine_inputs=lambda x: None)
    ft = types.SimpleNamespace(enable_finetuning=True,
                               profile_on_launch=profile_on_launch)
    eng = types.SimpleNamespace(
        scheduler=sched, input_queue=None,
        vllm_config=types.SimpleNamespace(finetune_config=ft))
    return eng, coord


def part_c():
    print("\nC. a failing launch profiling pass degrades to inference-only")
    from deltaserve.engine_core import EngineHooks

    eng, coord = fake_engine()
    hooks = EngineHooks.__new__(EngineHooks)
    hooks.engine, hooks.arrival_signal, hooks._coord = eng, None, None
    os.environ["DSERVE_TEST_FAIL_PROFILING"] = "1"
    try:
        hooks.before_busy_loop()
        raised = None
    except Exception as e:  # noqa: BLE001
        raised = e
    finally:
        del os.environ["DSERVE_TEST_FAIL_PROFILING"]
    C.ok("before_busy_loop does not raise", raised is None, repr(raised))
    C.ok("finetuning disabled with the reason",
         not coord.ft_started and "launch profiling pass failed"
         in (coord.ft_disabled_reason or ""), str(coord.ft_disabled_reason))
    eng2, coord2 = fake_engine(profile_on_launch=False)
    h2 = EngineHooks.__new__(EngineHooks)
    h2.engine, h2.arrival_signal, h2._coord = eng2, None, None
    h2.before_busy_loop()
    C.ok("a passing launch leaves finetuning on",
         coord2.ft_started and coord2.ft_disabled_reason is None)


def part_d():
    print("\nD. status bookkeeping")
    from deltaserve.engine_core import finetuning_status
    from deltaserve.ft_scheduler import FinetuneScheduler

    c = FinetuneCoordinator(capacity=64)

    class BP:
        def __init__(self, resp):
            self.resp = resp

        def poll_response(self):
            r, self.resp = self.resp, None
            return r

    for loss in (3.0, 2.0):
        c.pending_backward = True
        c.backward_process = BP({"event": "activations_processed", "n": 50,
                                 "loss": loss})
        c.poll_backward()
    t = FinetuneCoordinator(capacity=64)
    t.relay_mode = True
    t.pending_backward = True
    t.apply_relayed_done({"event": "activations_processed", "n": 30, "loss": 1.5})
    C.ok("tp=1 acks booked", c.backwards_done == 2 and c.tokens_trained == 100
         and c.last_loss == 2.0)
    C.ok("TP relayed ack booked", t.backwards_done == 1 and t.tokens_trained == 30
         and t.last_loss == 1.5)
    st = c.status()
    C.ok("coordinator status", st["mean_loss_last_20"] == 2.5
         and st["started"] and st["disabled_reason"] is None
         and st["buffer_capacity"] == 64, json.dumps(st)[:80])
    store = types.SimpleNamespace(total_epochs=3, current_epoch=1,
                                  trained=[True, False, True], __len__=None)
    store = type("S", (), {"total_epochs": 3, "current_epoch": 1,
                           "trained": [True, False, True],
                           "__len__": lambda self: 3})()
    fake = types.SimpleNamespace(
        _coord=c, _ft_injector=types.SimpleNamespace(store=store),
        _estimator=types.SimpleNamespace(eager_rmse=0.01))
    fs = FinetuneScheduler.finetuning_status(fake)
    C.ok("scheduler status adds store progress",
         fs["corpus_samples"] == 3 and fs["samples_trained_this_epoch"] == 2
         and fs["total_epochs"] == 3 and fs["estimator_fitted"]
         and fs["backwards_done"] == 2)
    eng = types.SimpleNamespace(scheduler=types.SimpleNamespace(
        finetuning_status=lambda: {"backwards_done": 7}))
    C.ok("utility: enabled", finetuning_status(eng) == {"enabled": True,
                                                       "backwards_done": 7})
    C.ok("utility: plain scheduler",
         finetuning_status(types.SimpleNamespace(scheduler=object()))["enabled"] is False)


class FakeEngineClient:
    """collective_rpc → what the backward child would do (asynchronously)."""

    def __init__(self, mode):
        self.mode, self.calls = mode, []
        outer = self

        class Core:
            async def call_utility_async(self, method, *args):
                outer.calls.append((method, args))
                return {"enabled": True, "backwards_done": 3}
        self.engine_core = Core()

    async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        self.calls.append((method, args))
        path, save_id = args

        def child():
            time.sleep(0.3)
            if self.mode == "ok":
                write_peft_adapter(path, {"x.q_proj.lora_A.weight": torch.ones(2, 2)},
                                   source_dir=None, scaling=1.0,
                                   state={"save_id": save_id, "reason": "manual"})
            elif self.mode == "error":
                write_save_error(path, save_id, "disk full")
        if self.mode in ("ok", "error"):
            threading.Thread(target=child, daemon=True).start()
        return [self.mode != "nochild", self.mode != "nochild"]


def part_e():
    print("\nE. HTTP routes")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from deltaserve import api

    with tempfile.TemporaryDirectory() as root:
        def client(mode, save_root):
            app = FastAPI()
            ec = FakeEngineClient(mode)
            ac = {"deltaserve": {"enable_finetuning": True,
                                 **({"adapter_save_path": save_root} if save_root else {})}}
            app.state.args = types.SimpleNamespace(additional_config=json.dumps(ac))
            app.state.engine_client = ec
            api.attach_router(app)
            return TestClient(app), ec

        cl, ec = client("ok", os.path.join(root, "ck"))
        r = cl.post("/save_adapter")
        body = r.json()
        C.ok("default path under adapter_save_path → 200",
             r.status_code == 200 and body["saved"] is True
             and os.path.dirname(body["path"]) == os.path.join(root, "ck")
             and os.path.basename(body["path"]).startswith("manual-"),
             f"{r.status_code} {body.get('path')}")
        C.ok("RPC got (path, save_id)", ec.calls[0][0] == "deltaserve_save_adapter"
             and ec.calls[0][1][1] == body["save_id"])
        r = cl.post("/save_adapter", json={"path": os.path.join(root, "explicit")})
        C.ok("explicit path → 200", r.status_code == 200
             and r.json()["path"] == os.path.join(root, "explicit"))
        cl0, _ = client("ok", None)
        r = cl0.post("/save_adapter")
        C.ok("no path and no adapter_save_path → 400", r.status_code == 400,
             r.json().get("error", ""))
        cle, _ = client("error", None)
        r = cle.post("/save_adapter", json={"path": os.path.join(root, "e")})
        C.ok("child error → 500", r.status_code == 500
             and r.json()["error"] == "disk full")
        r = cl.post("/save_adapter", json={"path": os.path.join(root, "nw"),
                                          "wait": False})
        C.ok("wait=false → 202 scheduled", r.status_code == 202
             and r.json()["scheduled"] is True)
        cls, _ = client("silent", None)
        r = cls.post("/save_adapter", json={"path": os.path.join(root, "s"),
                                           "timeout_s": 0.5})
        C.ok("no result before timeout → 202 pending", r.status_code == 202
             and r.json()["pending"] is True)
        cln, _ = client("nochild", None)
        r = cln.post("/save_adapter", json={"path": os.path.join(root, "n")})
        C.ok("no live child → 409", r.status_code == 409)
        r = cl.get("/finetuning_status")
        C.ok("GET /finetuning_status", r.status_code == 200
             and r.json() == {"enabled": True, "backwards_done": 3})
        r = cl.post("/start_finetuning")   # existing route still mounted
        C.ok("existing routes still mounted", r.status_code != 404, str(r.status_code))


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
    part_d()
    part_e()
    C.finish()
