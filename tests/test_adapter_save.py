#!/usr/bin/env python
"""Saving a finetuning job's trained adapter (deltaserve.adapter_save).

CPU-only, tiny synthetic weights; the trainer export runs on the same synthetic
backward service the gradcheck tests use.

    python tests/test_adapter_save.py

Checks
  * merging the TP ranks' shards is the exact inverse of ``lora_shard_slice``
    (tp = 1, 2, 4), and a replicated factor that differs between ranks is an
    error;
  * the written adapter has the source's tensor names / shapes / side files and
    the trained values, is written atomically, and unmerged shards, unknown
    tensors and an existing directory are refused;
  * the trainer exports its masters under the PEFT names it was built from, and
    the backward child answers the ``export_lora`` message with them;
  * the engine-side export orders the ranks, refuses a missing one, and the
    'ready' logic saves to ``ft_job.save_path``.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import bwd_harness as H   # before any sys.path change (lives in tests/)
import torch

_passed = 0
_failed = 0
_TTY = sys.stdout.isatty()
_GREEN = "\033[92m" if _TTY else ""
_RED = "\033[91m" if _TTY else ""
_RESET = "\033[0m" if _TTY else ""

HID, Q, KV, R, LAYERS = 32, 32, 16, 4, 2      # q_size = hidden, kv = hidden / 2


def green(msg):
    print(f"{_GREEN}{msg}{_RESET}")


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  {_GREEN}[PASS]{_RESET} {name}")
    else:
        _failed += 1
        print(f"  {_RED}[FAIL]{_RESET} {name}")


def error_of(fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"
    return None


def full_state(seed=0):
    g = torch.Generator().manual_seed(seed)
    st = {}
    for i in range(LAYERS):
        pre = f"base_model.model.model.layers.{i}.self_attn."
        for proj, out, inp in [("q", Q, HID), ("k", KV, HID), ("v", KV, HID),
                               ("o", HID, Q)]:
            st[pre + f"{proj}_proj.lora_A.weight"] = torch.randn(R, inp, generator=g)
            st[pre + f"{proj}_proj.lora_B.weight"] = torch.randn(out, R, generator=g)
    return st


def shards(state, tp):
    from deltaserve.adapter_save import LORA_KEY_RE
    from deltaserve.bwd_services.common.tp import lora_shard_slice

    out = []
    for r in range(tp):
        d = {}
        for k, t in state.items():
            m = LORA_KEY_RE.search(k)
            d[k] = lora_shard_slice(m.group(2), m.group(3), t, r, tp,
                                    Q // tp, KV // tp).clone()
        out.append(d)
    return out


def make_source(d, state=None):
    from safetensors.torch import save_file

    src = Path(d) / "source-adapter"
    src.mkdir()
    save_file(state or full_state(seed=1), str(src / "adapter_model.safetensors"))
    (src / "adapter_config.json").write_text(json.dumps(
        {"peft_type": "LORA", "r": R, "lora_alpha": 8,
         "base_model_name_or_path": "org/base"}))
    (src / "tokenizer_config.json").write_text("{}")
    return str(src)


def same(a, b):
    return set(a) == set(b) and all(torch.equal(a[k].float(), b[k].float()) for k in a)


def test_merge():
    green("merging TP rank shards:")
    from deltaserve.adapter_save import merge_rank_shards

    full = full_state()
    for tp in (1, 2, 4):
        parts = shards(full, tp)
        if tp > 1:
            k = next(k for k in full if "q_proj.lora_B" in k)
            assert parts[0][k].shape[0] == Q // tp       # really sharded
        check(f"tp={tp}: merge(slice(full)) == full, bit for bit",
              same(merge_rank_shards(parts), full))
    parts = shards(full, 2)
    k = next(k for k in full if "q_proj.lora_A" in k)
    parts[1][k] = parts[1][k] + 0.5
    err = error_of(lambda: merge_rank_shards(parts))
    check("a replicated factor that differs between ranks is an error",
          err is not None and "replicated" in err and "rank 1" in err)
    parts = shards(full, 2)
    parts[1].pop(k)
    check("ranks exporting different tensors is an error",
          error_of(lambda: merge_rank_shards(parts)) is not None)
    check("nothing to merge is an error",
          error_of(lambda: merge_rank_shards([])) is not None)


def test_write():
    green("writing the adapter directory:")
    from safetensors.torch import load_file

    from deltaserve.adapter_save import write_adapter

    trained = full_state(seed=7)
    with tempfile.TemporaryDirectory() as d:
        src = make_source(d)
        dst = write_adapter(src, str(Path(d) / "out" / "dserve-abc"), trained)
        back = load_file(os.path.join(dst, "adapter_model.safetensors"))
        check("trained values on disk, same names, fp32",
              same(back, trained) and all(v.dtype == torch.float32
                                          for v in back.values()))
        check("adapter_config.json and side files copied from the source",
              json.load(open(os.path.join(dst, "adapter_config.json")))["r"] == R
              and os.path.exists(os.path.join(dst, "tokenizer_config.json")))
        check("no temp directory left behind",
              os.listdir(Path(d) / "out") == ["dserve-abc"])
        check("the source adapter is untouched",
              same(load_file(os.path.join(src, "adapter_model.safetensors")),
                   full_state(seed=1)))
        err = error_of(lambda: write_adapter(src, dst, trained))
        check("an existing directory is refused",
              err is not None and err.startswith("FileExistsError"))
        newer = full_state(seed=9)
        write_adapter(src, dst, newer, overwrite=True)
        check("overwrite=True replaces it",
              same(load_file(os.path.join(dst, "adapter_model.safetensors")), newer))

        half = shards(trained, 2)[0]
        err = error_of(lambda: write_adapter(src, str(Path(d) / "x1"), half))
        check("an unmerged TP shard is refused (shape differs from the source)",
              err is not None and "shards not merged" in err
              and not os.path.exists(Path(d) / "x1"))
        bad = dict(trained)
        bad["something.else"] = torch.zeros(2)
        check("a tensor the source does not have is refused",
              "not in the source" in (error_of(
                  lambda: write_adapter(src, str(Path(d) / "x2"), bad)) or ""))
        partial = {k: v for k, v in trained.items() if "layers.0." in k}
        dst3 = write_adapter(src, str(Path(d) / "x3"), partial)
        back = load_file(os.path.join(dst3, "adapter_model.safetensors"))
        srcw = full_state(seed=1)
        check("tensors that were not trained are carried over from the source",
              all(torch.equal(back[k], trained[k] if "layers.0." in k else srcw[k])
                  for k in srcw))
        check("a source without weights is an error",
              error_of(lambda: write_adapter(d, str(Path(d) / "x4"), trained))
              is not None)


class FakeConn:
    def __init__(self, msgs):
        self.msgs, self.sent = list(msgs), []

    def recv(self):
        if not self.msgs:
            raise EOFError
        return self.msgs.pop(0)

    def send(self, m):
        self.sent.append(m)

    def close(self):
        pass


def test_trainer_export():
    green("trainer export (synthetic backward service):")
    from deltaserve.bwd_services.llama3 import Llama3BackwardService

    svc, _ = H.build_synthetic_service(
        Llama3BackwardService, hidden=HID, n_layers=LAYERS, Hq=4, Hkv=2, Hd=8,
        inter=48, vocab=50, r=R, lr=1e-3)
    ft = svc.shared["ft"]
    state = svc.export_lora_state()
    check("exports every LoRA factor under the PEFT name it was built from",
          set(state) == set(ft) and len(state) == LAYERS * 8)
    check("freshly built: the export equals the adapter it was initialised from",
          same(state, ft) and all(v.device.type == "cpu" for v in state.values()))
    key = next(k for k in ft if "layers.1.self_attn.v_proj.lora_B" in k)
    with torch.no_grad():
        svc.lora[1]["v"]["B"].add_(1.25)
    state = svc.export_lora_state()
    check("a trained master shows up in the export (and only that one changed)",
          torch.allclose(state[key], ft[key] + 1.25)
          and all(torch.equal(state[k], ft[k]) for k in ft if k != key))
    state[key].zero_()
    check("the export is a copy: editing it does not touch the master",
          float(svc.lora[1]["v"]["B"].detach().abs().sum()) > 0)

    conn = FakeConn([{"cmd": "export_lora"}, {"cmd": "shutdown"}])
    svc.run(conn)
    replies = [m for m in conn.sent if m.get("event") == "lora_state"]
    check("the backward child answers the 'export_lora' message with the state",
          len(replies) == 1 and set(replies[0]["state"]) == set(ft)
          and torch.allclose(replies[0]["state"][key], ft[key] + 1.25))

    with tempfile.TemporaryDirectory() as d:
        from safetensors.torch import load_file

        from deltaserve.adapter_save import merge_rank_shards, write_adapter

        src = make_source(d, {k: v.clone() for k, v in ft.items()})
        dst = write_adapter(src, str(Path(d) / "trained"),
                            merge_rank_shards([svc.export_lora_state()]))
        back = load_file(os.path.join(dst, "adapter_model.safetensors"))
        check("trainer → merge → write → load gives the trained weights",
              torch.allclose(back[key], ft[key] + 1.25)
              and tuple(back[key].shape) == tuple(ft[key].shape))


def test_switch_adapter():
    green("switching the trainer to the next job's adapter, in place:")
    from deltaserve.bwd_services.llama3 import Llama3BackwardService

    svc, _ = H.build_synthetic_service(
        Llama3BackwardService, hidden=HID, n_layers=LAYERS, Hq=4, Hkv=2, Hd=8,
        inter=48, vocab=50, r=R, lr=1e-3)
    ft = svc.shared["ft"]
    masters = {k: svc.lora[k[0]][k[1]][k[2]] for k in svc._lora_keys}
    old_opt = svc.optimizer
    svc.current_epoch, svc._epoch_processed_tokens = 3, 999
    g = torch.Generator().manual_seed(5)
    new = {k: torch.randn(v.shape, generator=g) for k, v in ft.items()}
    hp = {"learning_rate": 5e-4, "weight_decay": 0.02, "gamma": 0.5,
          "lora_scaling": svc.scaling, "total_tokens_per_epoch": 1234}
    out = svc.switch_adapter(new, hp)
    check("the new weights are in the masters", out["tensors"] == LAYERS * 8
          and same(svc.export_lora_state(), new))
    check("the masters are the SAME tensors (graphs and the served slot stay valid)",
          all(svc.lora[k[0]][k[1]][k[2]] is p for k, p in masters.items()))
    check("fresh optimizer with the job's learning rate and weight decay",
          svc.optimizer is not old_opt
          and svc.optimizer.param_groups[0]["lr"] == 5e-4
          and svc.optimizer.param_groups[0]["weight_decay"] == 0.02
          and svc.scheduler.gamma == 0.5)
    check("epoch counters reset, corpus size taken",
          svc.current_epoch == 0 and svc._epoch_processed_tokens == 0
          and svc._total_tokens_per_epoch == 1234)
    check("no stale gradients", all(p.grad is None for p in masters.values()))

    before = svc.export_lora_state()
    bad_rank = {k: torch.randn(v.shape[0], 2 * R) if "lora_B" in k
                else torch.randn(2 * R, v.shape[1]) for k, v in ft.items()}
    err = error_of(lambda: svc.switch_adapter(bad_rank, hp))
    check("another rank is refused, nothing changed",
          err is not None and "rank" in err and same(svc.export_lora_state(), before))
    fewer = {k: v for k, v in new.items() if "o_proj" not in k}
    err = error_of(lambda: svc.switch_adapter(fewer, hp))
    check("other target modules are refused",
          err is not None and "target modules" in err
          and same(svc.export_lora_state(), before))
    out = svc.switch_adapter(ft, {**hp, "dry_run": True})
    check("dry run: the adapter is checked, nothing is changed",
          out.get("dry_run") is True and same(svc.export_lora_state(), before))
    check("dry run of a bad adapter reports the same refusal",
          "rank" in (error_of(lambda: svc.switch_adapter(
              bad_rank, {**hp, "dry_run": True})) or ""))
    err = error_of(lambda: svc.switch_adapter(new, {**hp, "lora_scaling": 9.0}))
    check("another alpha/r is refused (it is baked into captured graphs)",
          err is not None and "alpha/r" in err)

    conn = FakeConn([{"cmd": "switch_adapter", "ft": bad_rank, "hparams": hp},
                     {"cmd": "switch_adapter", "ft": ft, "hparams": hp},
                     {"cmd": "shutdown"}])
    svc.run(conn)
    ev = [m.get("event") for m in conn.sent]
    check("the backward child reports a refused adapter and stays alive for "
          "the next one", ev.count("error") == 1 and ev.count("adapter_switched") == 1
          and same(svc.export_lora_state(), ft))


def test_publish_keeps_no_graph():
    green("publishing to the served slot builds no autograd graph:")
    import inspect

    from deltaserve.bwd_services.common.trainer import LoraSftTrainerService
    from deltaserve.bwd_services.llama3 import Llama3BackwardService

    svc, _ = H.build_synthetic_service(
        Llama3BackwardService, hidden=HID, n_layers=LAYERS, Hq=4, Hkv=2, Hd=8,
        inter=48, vocab=50, r=R, lr=1e-3)
    layers = {}
    for i in range(LAYERS):
        layers[i] = {}
        for proj in "qkvo":
            pa, pb = svc.lora[i][proj]["A"], svc.lora[i][proj]["B"]
            layers[i][proj] = {
                "a": torch.zeros(2, 1, pa.shape[0], pa.shape[1], dtype=torch.float16),
                "b": torch.zeros(2, 1, pb.shape[0], pb.shape[1], dtype=torch.float16)}
    svc.lora_buffers = {"slot": 1, "layers": layers}
    with torch.no_grad():
        svc.lora[1]["q"]["B"].add_(0.5)
    assert torch.is_grad_enabled()
    for _ in range(3):
        svc._publish_to_served()
    bufs = [t for d in layers.values() for p in d.values() for t in p.values()]
    check("the masters still require grad (they are what is trained)",
          all(svc.lora[i][p][ab].requires_grad for i in range(LAYERS)
              for p in "qkvo" for ab in "AB"))
    check("after publishing, no served buffer carries a grad_fn or requires grad",
          all(t.grad_fn is None and not t.requires_grad for t in bufs))
    b = layers[1]["q"]["b"][1, 0]
    want = (svc.lora[1]["q"]["B"].detach() * svc.scaling).to(torch.float16)
    check("...and the published values are the masters' (alpha/r baked into B)",
          torch.allclose(b.float(), want.float(), atol=1e-3)
          and float(layers[1]["q"]["b"][0].abs().sum()) == 0.0)
    src = inspect.getsource(LoraSftTrainerService)
    i = src.index("def _publish_to_served")
    check("_publish_to_served is decorated with torch.no_grad()",
          src[:i].rstrip().endswith("@torch.no_grad()"))
    new = {k: v.clone() + 0.25 for k, v in svc.shared["ft"].items()}
    svc.switch_adapter(new, {"learning_rate": 1e-3, "weight_decay": 0.0,
                             "gamma": 1.0, "lora_scaling": svc.scaling})
    check("switching adapters publishes without a graph either",
          all(t.grad_fn is None and not t.requires_grad for t in bufs))


def test_engine_export_and_ready():
    green("engine-side export + what 'ready' does:")
    from safetensors.torch import load_file

    from deltaserve.adapter_save import export_trained_adapter, save_job_adapter
    from deltaserve.ft_job import FinetuneJob, FinetuneJobSpec

    trained = full_state(seed=3)

    def engine(results):
        calls = []

        def rpc(method, *a, **k):
            calls.append(method)
            return results

        return SimpleNamespace(model_executor=SimpleNamespace(collective_rpc=rpc),
                               calls=calls)

    with tempfile.TemporaryDirectory() as d:
        src = make_source(d)
        s = shards(trained, 2)
        eng = engine([(1, s[1]), (0, s[0])])           # ranks out of order
        dst = export_trained_adapter(eng, src, str(Path(d) / "a"))
        check("ranks are ordered before merging; the worker RPC is the export one",
              same(load_file(os.path.join(dst, "adapter_model.safetensors")), trained)
              and eng.calls == ["deltaserve_export_ft_adapter"])
        check("a missing rank is an error",
              "missing a rank" in (error_of(lambda: export_trained_adapter(
                  engine([(1, s[1])]), src, str(Path(d) / "b"))) or ""))
        check("a worker that reports an error (returned, not raised) fails the export",
              "export failed on a worker" in (error_of(lambda: export_trained_adapter(
                  engine([(0, s[0]), {"error": "OSError: pipe closed"}]), src,
                  str(Path(d) / "e"))) or ""))
        check("no worker with a backward child is an error",
              error_of(lambda: export_trained_adapter(
                  engine([None, None]), src, str(Path(d) / "c"))) is not None)

        job = FinetuneJob(spec=FinetuneJobSpec(
            adapter_path=src, save_path=str(Path(d) / "saved")), name="dserve-1")
        out = save_job_adapter(job, engine([(0, trained)]), routing=None)
        check("'ready' with ft_job.save_path: adapter written there, recorded "
              "on the job",
              out == job.saved_path == str(Path(d) / "saved")
              and same(load_file(os.path.join(out, "adapter_model.safetensors")),
                       trained) and job.describe()["saved_path"] == out)
        job2 = FinetuneJob(spec=FinetuneJobSpec(adapter_path=src), name="dserve-2")
        check("'ready' with nowhere to save: not saved, job not failed",
              save_job_adapter(job2, engine([(0, trained)]), None) is None
              and job2.saved_path is None)

        class Ctx:
            store = SimpleNamespace(data_path=None)
            coordinator = SimpleNamespace(pending_backward=False, fill_count=0,
                                          reserved_fill=0)

            def on_job_ready(self, j):
                save_job_adapter(j, engine([None]), None)

        job3 = FinetuneJob(spec=FinetuneJobSpec(
            stages=("init", "training", "ready"), adapter_path=src,
            save_path=str(Path(d) / "never")), name="dserve-3")
        job3.tick(Ctx())
        check("a failed save fails the job at 'ready' (not silently ready)",
              job3.failed and "ready" in job3.error
              and not os.path.exists(Path(d) / "never"))


def main():
    green("=== adapter save ===")
    test_merge()
    test_write()
    test_trainer_export()
    test_switch_adapter()
    test_publish_keeps_no_graph()
    test_engine_export_and_ready()
    print()
    if _failed:
        print(f"{_RED}{_failed} failed{_RESET}, {_passed} passed")
        sys.exit(1)
    green(f"all {_passed} checks passed")


if __name__ == "__main__":
    main()
