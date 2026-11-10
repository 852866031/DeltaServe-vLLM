#!/usr/bin/env python
"""End to end on the GPU: adapter checkpoints, status, fail-safes, RoPE scaling.

Each phase runs in its own process (fresh engine):

  1. http (``--tp N``, default 1): a real ``vllm serve`` co-serving
     Qwen3-0.6B. GET /finetuning_status shows training progress (and, under TP,
     the relayed acks); periodic ``backward-*`` checkpoints appear and are
     pruned; inference keeps answering; after POST /stop_finetuning,
     POST /save_adapter writes one PEFT adapter (merged across ranks under TP)
     whose cycle count equals the status's; it differs from the source.
  2. served (tp=1, offline): the trained FT slot vLLM is serving, the saved
     checkpoint loaded as an ordinary LoRA, and the checkpoint under HF PEFT
     give the same prompt logprobs; the untrained source adapter does not.
  3. failprofile: with the launch profiling pass forced to fail
     (``DSERVE_TEST_FAIL_PROFILING``) the engine still serves, and the status
     reports finetuning disabled with the reason.
  4. resume: ``finetuning_lora_path`` = the checkpoint launches and trains.
  5. llama32 (Llama-3.2-1B, rope_type llama3): with ``DSERVE_TP_DIAG`` the
     backward's rematerialized layers match what vLLM's forward captured; the
     same run with the RoPE scaling dropped (the old behaviour, patched in
     through a sitecustomize) does not.

    python tests/test_adapter_save_e2e.py            # phases 1-5, GPU 0
    python tests/test_adapter_save_e2e.py --tp 2     # phase 1 on both GPUs
"""

import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_CFG = _REPO / "configs" / "preset" / "serving_config_finetuning_qwen3_0.6b.yaml"
_PORT = 8013
_TTY = sys.stdout.isatty()
_G, _R, _X = ("\033[92m", "\033[91m", "\033[0m") if _TTY else ("", "", "")
_passed = _failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    print(f"  {_G + '[PASS]' if cond else _R + '[FAIL]'}{_X} {name}  {detail}", flush=True)
    _passed += bool(cond)
    _failed += not cond
    return cond


def base_env():
    env = dict(os.environ, HF_HUB_OFFLINE="1", VLLM_USE_FLASHINFER_SAMPLER="0")
    env.setdefault("HF_HOME", "/mnt/storage/huggingface")
    return env


def temp_yaml(out: Path, **finetune) -> Path:
    """The Qwen3-0.6B preset with absolute paths and ``finetune`` overrides."""
    import yaml

    raw = yaml.safe_load(_CFG.read_text())
    for sec in raw.values():
        if isinstance(sec, dict):
            for k, v in list(sec.items()):
                if "path" in k and isinstance(v, str) and not os.path.isabs(v):
                    sec[k] = str((_CFG.parent / v).resolve())
    raw["finetune_param"].update(start_on_launch=True, **finetune)
    raw["lora"]["max_loras"] = 4
    out.write_text(yaml.safe_dump(raw))
    return out


def load_st(d):
    from safetensors.torch import load_file
    return load_file(os.path.join(d, "adapter_model.safetensors"))


def max_diff(a, b):
    return max((a[k].float() - b[k].float()).abs().max().item() for k in a)


# --------------------------------------------------------------------------- #
# phase 1: HTTP
# --------------------------------------------------------------------------- #

def http(method, path, body=None, timeout=600):
    req = urllib.request.Request(
        f"http://127.0.0.1:{_PORT}{path}", method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def wait_status(pred, timeout):
    deadline, st = time.monotonic() + timeout, None
    while time.monotonic() < deadline:
        try:
            code, st = http("GET", "/finetuning_status", timeout=10)
            if code == 200 and pred(st):
                return st
        except Exception:
            pass
        time.sleep(1)
    return st


def phase_http(tmp: Path, tp: int) -> None:
    print(f"=== 1. HTTP: /finetuning_status + /save_adapter (tp={tp}) ===", flush=True)
    ck = tmp / "ck"
    cfg = temp_yaml(tmp / "http.yaml", adapter_save_path=str(ck),
                    adapter_save_every_backwards=3, adapter_save_keep_last=2,
                    learning_rate=2e-3)
    log = open(tmp / "http.log", "w")
    proc = subprocess.Popen(
        [sys.executable, str(_REPO / "eval-tp/launch_deltaserve.py"),
         "--config", str(cfg), "--tp", str(tp), "--port", str(_PORT)],
        stdout=log, stderr=subprocess.STDOUT, env=base_env(), start_new_session=True)
    try:
        deadline = time.monotonic() + 900
        while True:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{_PORT}/health",
                                            timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            if proc.poll() is not None or time.monotonic() > deadline:
                check(f"server healthy (see {tmp / 'http.log'})", False)
                return
            time.sleep(2)
        check("server healthy", True)
        st = wait_status(lambda s: s.get("backwards_done", 0) >= 7, 600)
        check("training progresses (status: ≥7 backwards)",
              st and st["backwards_done"] >= 7 and st["tokens_trained"] > 0
              and st["started"] and st["disabled_reason"] is None
              and st["last_loss"] is not None and math.isfinite(st["last_loss"]),
              json.dumps({k: st.get(k) for k in ("backwards_done", "tokens_trained",
                                                 "last_loss", "epoch")}) if st else "")
        _, models = http("GET", "/v1/models")
        code, out = http("POST", "/v1/completions", {
            "model": models["data"][0]["id"], "prompt": "The capital of France is",
            "max_tokens": 8, "temperature": 0})
        check("inference keeps answering", code == 200 and out["choices"][0]["text"],
              repr(out.get("choices", [{}])[0].get("text")))
        code, _ = http("POST", "/stop_finetuning")
        check("POST /stop_finetuning", code == 200)
        prev = None
        for _ in range(60):
            _, st = http("GET", "/finetuning_status")
            if not st["backward_in_flight"] and prev == st["backwards_done"]:
                break
            prev = st["backwards_done"]
            time.sleep(1)
        check("in-flight backward acked after the stop (engine kept polling)",
              not st["backward_in_flight"], json.dumps(st)[:120])
        manual = tmp / "manual"
        code, res = http("POST", "/save_adapter", {"path": str(manual)})
        check("POST /save_adapter → 200 saved", code == 200 and res.get("saved"),
              json.dumps(res)[:160])
        check("checkpoint cycles == status backwards",
              res.get("backward_cycles") == st["backwards_done"]
              and res.get("tp_size") == tp,
              f"{res.get('backward_cycles')} vs {st['backwards_done']}")
        src = load_st(cfg_ft_path(cfg))
        got = load_st(manual)
        check("merged keys/shapes == source adapter",
              got.keys() == src.keys()
              and all(got[k].shape == src[k].shape for k in src))
        d = max_diff(got, src)
        check("checkpoint is trained (≠ source)", d > 1e-4, f"max|Δ|={d:.2e}")
        cks = sorted(os.listdir(ck)) if ck.exists() else []
        want_last = 3 * (st["backwards_done"] // 3)
        check("periodic checkpoints: newest kept, pruned to 2",
              0 < len(cks) <= 2 and cks[-1] == f"backward-{want_last:06d}",
              f"{cks} (backwards={st['backwards_done']})")
        code, res = http("POST", "/save_adapter", {})
        check("default path under adapter_save_path",
              code == 200 and res["path"].startswith(str(ck / "manual-")), res.get("path"))
    finally:
        try:
            os.killpg(proc.pid, signal.SIGINT)
            proc.wait(60)
        except Exception:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        log.close()


def cfg_ft_path(cfg: Path) -> str:
    import yaml
    return yaml.safe_load(cfg.read_text())["finetune_param"]["finetuning_lora_path"]


# --------------------------------------------------------------------------- #
# offline workers (one engine per process)
# --------------------------------------------------------------------------- #

def make_llm(cfg: Path, **overrides):
    from vllm import LLM

    from deltaserve.config_loader import additional_config, load_yaml_config, split_config

    raw = load_yaml_config(str(cfg))
    engine_kwargs, _, _ = split_config(raw)
    engine_kwargs["gpu_memory_utilization"] = 0.6
    # Prompt logprobs are compared across runs: no prefix-cache hits.
    engine_kwargs["enable_prefix_caching"] = False
    ft = additional_config(raw, start_on_launch=True, **overrides)["deltaserve"]
    return LLM(**engine_kwargs, additional_config={"deltaserve": ft})


def status(llm):
    return llm.llm_engine.engine_core.call_utility("deltaserve_finetuning_status")


def wait_backwards(llm, n, timeout=600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        st = status(llm)
        if st.get("backwards_done", 0) >= n:
            return st
        time.sleep(1)
    return status(llm)


def quiesce(llm):
    llm.collective_rpc("deltaserve_stop_finetuning")
    prev = None
    for _ in range(60):
        st = status(llm)
        if not st["backward_in_flight"] and prev == st["backwards_done"]:
            return st
        prev = st["backwards_done"]
        time.sleep(1)
    return status(llm)


def save(llm, path, timeout=120):
    import uuid

    from deltaserve.adapter_io import read_save_result
    sid = uuid.uuid4().hex
    llm.collective_rpc("deltaserve_save_adapter", args=(str(path), sid))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = read_save_result(str(path), sid)
        if r:
            return r
        time.sleep(0.2)
    return {"saved": False, "error": "timeout"}


PROMPTS = ["Below is an instruction that describes a task. Write a response.\n"
           "### Instruction:\nName three primary colors.\n### Response:\n",
           "Explain why the sky is blue in one sentence:",
           "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"]


def prompt_logprobs(llm, lora):
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    outs = llm.generate(PROMPTS, sp, lora_request=lora, use_tqdm=False)
    res = []
    for o in outs:
        ids = o.prompt_token_ids
        res.append([o.prompt_logprobs[j][ids[j]].logprob for j in range(1, len(ids))])
    return res


def compare_slots(worker, id_a, id_b):
    """Worker-side (collective_rpc): elementwise compare two adapters' rows in
    vLLM's served LoRA buffers — what inference actually multiplies with."""
    am = worker.model_runner.lora_manager._adapter_manager
    sa, sb = am.lora_index_to_id.index(id_a), am.lora_index_to_id.index(id_b)
    ndiff = total = 0
    worst = 0.0
    for m in am.modules.values():
        for stacks in (getattr(m, "lora_a_stacked", ()), getattr(m, "lora_b_stacked", ())):
            for st in stacks:
                if st.shape[0] <= max(sa, sb):
                    continue
                x, y = st[sa].float(), st[sb].float()
                ndiff += int((x != y).sum())
                total += x.numel()
                worst = max(worst, (x - y).abs().max().item())
    return {"ndiff": ndiff, "total": total, "worst": worst}


def worker_served(tmp: Path, out: Path) -> None:
    from vllm.lora.request import LoRARequest

    from deltaserve.ft_injector import FT_LORA_INT_ID
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"   # callable RPC
    cfg = temp_yaml(tmp / "served.yaml", learning_rate=2e-3)
    llm = make_llm(cfg)
    st0 = wait_backwards(llm, 6)
    st = quiesce(llm)
    ck = tmp / "served-ckpt"
    res = save(llm, ck)
    src = cfg_ft_path(cfg)
    lp = {
        "ft_slot": prompt_logprobs(llm, LoRARequest("__finetune__", FT_LORA_INT_ID, src)),
        "ckpt": prompt_logprobs(llm, LoRARequest("ckpt", 7, str(ck))),
        "ckpt_again": prompt_logprobs(llm, LoRARequest("ckpt", 7, str(ck))),
        "source": prompt_logprobs(llm, LoRARequest("source", 8, src)),
        "base": prompt_logprobs(llm, None),
    }
    bufs = {"slot_vs_ckpt": llm.collective_rpc(compare_slots, args=(FT_LORA_INT_ID, 7))[0],
            "slot_vs_source": llm.collective_rpc(compare_slots, args=(FT_LORA_INT_ID, 8))[0]}
    json.dump({"status_before": st0, "status": st, "save": res, "lp": lp, "bufs": bufs,
               "ckpt": str(ck), "source": src}, open(out, "w"))


def worker_failprofile(tmp: Path, out: Path) -> None:
    from vllm import SamplingParams
    os.environ["DSERVE_TEST_FAIL_PROFILING"] = "1"
    llm = make_llm(temp_yaml(tmp / "fail.yaml"))
    o = llm.generate(["The capital of France is"],
                     SamplingParams(max_tokens=8, temperature=0), use_tqdm=False)
    time.sleep(3)
    json.dump({"text": o[0].outputs[0].text, "status": status(llm)}, open(out, "w"))


def worker_resume(tmp: Path, out: Path) -> None:
    ck = os.environ["DSERVE_E2E_CKPT"]
    llm = make_llm(temp_yaml(tmp / "resume.yaml", finetuning_lora_path=ck))
    st = wait_backwards(llm, 2, timeout=300)
    st = quiesce(llm)
    res = save(llm, tmp / "resumed")
    json.dump({"status": st, "save": res}, open(out, "w"))


def make_llama_adapter(d: Path) -> Path:
    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig
    import glob
    snap = glob.glob(f"{os.environ['HF_HOME']}/hub/models--meta-llama--Llama-3.2-1B/snapshots/*")[0]
    c = AutoConfig.from_pretrained(snap)
    q, kv, h, r = c.num_attention_heads * c.head_dim, c.num_key_value_heads * c.head_dim, \
        c.hidden_size, 8
    g = torch.Generator().manual_seed(0)
    t = {}
    for i in range(c.num_hidden_layers):
        p = f"base_model.model.model.layers.{i}.self_attn."
        for proj, o, n in [("q", q, h), ("k", kv, h), ("v", kv, h), ("o", h, q)]:
            t[p + f"{proj}_proj.lora_A.weight"] = torch.randn(r, n, generator=g) * 0.01
            t[p + f"{proj}_proj.lora_B.weight"] = torch.randn(o, r, generator=g) * 0.01
    d.mkdir(parents=True, exist_ok=True)
    save_file(t, str(d / "adapter_model.safetensors"))
    (d / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": r, "lora_alpha": 16,
        "lora_dropout": 0.0, "bias": "none", "base_model_name_or_path":
        "meta-llama/Llama-3.2-1B",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}))
    return d


def worker_llama32(tmp: Path, out: Path) -> None:
    from vllm import LLM
    adapter = make_llama_adapter(tmp / "llama32-ft")
    # Long samples (~1.5k tokens): the llama3 rescale only touches the
    # low-frequency dims, which barely rotate over a short alpaca sample —
    # the remat check can only tell the RoPEs apart at long positions.
    lines = [l.strip() for l in open(_REPO / "data" / "alpaca_1000.txt") if l.strip()]
    corpus = tmp / "long_corpus.txt"
    with open(corpus, "w") as f:
        for i in range(12):
            f.write(" ".join(lines[i * 40:(i + 1) * 40])[:5500] + "\n")
    ft = dict(enable_finetuning=True, start_on_launch=True,
              finetuning_lora_path=str(adapter), data_path=str(corpus),
              max_saved_finetuning_tokens=2048, num_epochs=3, learning_rate=1e-5,
              profile_on_launch=False)
    llm = LLM(model="meta-llama/Llama-3.2-1B", enable_lora=True, max_lora_rank=8,
              max_loras=2, max_model_len=4096, gpu_memory_utilization=0.6,
              enforce_eager=True, additional_config={"deltaserve": ft})
    st = wait_backwards(llm, 5, timeout=400)
    json.dump({"status": st}, open(out, "w"))


WORKERS = {"served": worker_served, "failprofile": worker_failprofile,
           "resume": worker_resume, "llama32": worker_llama32}


def spawn(name, tmp: Path, extra_env=None, timeout=1200, tag=None):
    tag = tag or name
    out = tmp / f"{tag}.json"
    env = base_env()
    env.update(extra_env or {}, DSERVE_E2E_OUT=str(out))
    with open(tmp / f"{tag}.log", "w") as log:
        subprocess.run([sys.executable, __file__, "--worker", name, "--tmp", str(tmp)],
                       stdout=log, stderr=subprocess.STDOUT, env=env, timeout=timeout)
    text = (tmp / f"{tag}.log").read_text()
    if not out.exists():
        print(text[-3000:])
        return None, text
    return json.load(open(out)), text


# --------------------------------------------------------------------------- #

def phase_served(tmp):
    print("=== 2. the checkpoint is what vLLM serves ===", flush=True)
    r, _ = spawn("served", tmp)
    if not check("worker finished", r is not None):
        return None
    check("trained ≥6 backwards, saved", r["status_before"]["backwards_done"] >= 6
          and r["save"]["saved"], json.dumps(r["save"])[:120])
    check("save cycles == status", r["save"]["backward_cycles"] == r["status"]["backwards_done"])
    lp = r["lp"]

    def dmean(a, b):
        d = [abs(x - y) for sa, sb in zip(lp[a], lp[b]) for x, y in zip(sa, sb)]
        return sum(d) / len(d)
    b = r["bufs"]
    check("served FT slot == checkpoint, bit for bit (vLLM's LoRA buffers)",
          b["slot_vs_ckpt"]["ndiff"] == 0 and b["slot_vs_ckpt"]["total"] > 10**6,
          json.dumps(b["slot_vs_ckpt"]))
    check("control: the untrained source's buffers differ",
          b["slot_vs_source"]["ndiff"] > 0, json.dumps(b["slot_vs_source"]))
    # vLLM's prompt logprobs are not bit-reproducible run to run (the same
    # request twice moves by up to ~0.3 nats at single positions), so compare
    # means against that noise floor.
    # (Requests on the FT adapter's internal id run as FT traffic on another
    # forward path, so their logprobs are not compared — the buffers above are.)
    noise = dmean("ckpt", "ckpt_again")
    d_src = dmean("ckpt", "source")
    # The same request twice is not bit-reproducible in vLLM (seen 0.00-0.05
    # nats mean): measure that noise and require the source to sit far outside.
    check("logprobs: the untrained source is far outside the run-to-run noise",
          d_src > 10 * max(noise, 0.01),
          f"source mean|Δ|={d_src:.3f}, same-request noise {noise:.4f}")
    # HF PEFT with the checkpoint, fp32, on the same prompts.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base",
                                             torch_dtype=torch.float32).cuda()
    m = PeftModel.from_pretrained(m, r["ckpt"]).eval()
    diffs = []
    with torch.no_grad():
        for prompt, vl in zip(PROMPTS, lp["ckpt"]):
            ids = tok(prompt, return_tensors="pt").input_ids.cuda()
            logp = torch.log_softmax(m(ids).logits[0].float(), -1)
            hf = logp[:-1].gather(1, ids[0, 1:, None])[:, 0].tolist()
            if len(hf) != len(vl):
                diffs.append(float("inf"))
                break
            diffs += [abs(a - b) for a, b in zip(hf, vl)]
    hf_mean = sum(diffs) / len(diffs)
    check("HF PEFT(checkpoint) ≈ vLLM(checkpoint) (fp32 vs bf16)",
          hf_mean <= 0.15 and hf_mean < d_src / 5,
          f"mean|Δ|={hf_mean:.3f} max={max(diffs):.3f}")
    return r["ckpt"]


def phase_failprofile(tmp):
    print("=== 3. a failing launch profiling pass → inference only ===", flush=True)
    r, log = spawn("failprofile", tmp)
    if not check("engine came up and served", r is not None and r["text"]):
        return
    st = r["status"]
    check("status: finetuning disabled with the reason",
          not st["started"] and "launch profiling pass failed" in (st["disabled_reason"] or ""),
          str(st["disabled_reason"]))
    check("no backward ran", st["backwards_done"] == 0)
    check("the failure is logged", "launch profiling pass failed" in log)


def phase_resume(tmp, ckpt):
    print("=== 4. resume training from a checkpoint ===", flush=True)
    r, _ = spawn("resume", tmp, {"DSERVE_E2E_CKPT": ckpt})
    if not check("worker finished", r is not None):
        return
    check("launched from the checkpoint and trained", r["status"]["backwards_done"] >= 2
          and r["save"]["saved"] and r["save"]["source_adapter"] == ckpt,
          json.dumps(r["save"])[:120])


_DIAG_RE = re.compile(r"\[tpdiag\] rank=0 remat-vs-captured rel\|d\|: (.*)")
_RES_RE = re.compile(r"L\d+=([0-9.e+-]+)")
_GU_RE = re.compile(r"gu([0-9.e+-]+)")


def diag_stats(log):
    """Geometric means of the remat-vs-captured errors: the residual stream
    after each layer (``L<i>=``) and the MLP gate input right after attention
    (``gu``, not diluted by the residual), over every [tpdiag] line."""
    lines = _DIAG_RE.findall(log)

    def gmean(rx):
        v = [float(x) for line in lines for x in rx.findall(line)]
        return math.exp(sum(math.log(max(x, 1e-12)) for x in v) / len(v)) if v else None
    return {"resid": gmean(_RES_RE), "gate": gmean(_GU_RE), "lines": len(lines)}


def phase_llama32(tmp):
    print("=== 5. Llama-3.2-1B (rope_type llama3): remat == served forward ===", flush=True)
    env = {"DSERVE_TP_DIAG": "1"}
    r, log = spawn("llama32", tmp, env)
    check("trained (≥5 backwards on ~1.5k-token samples)",
          r is not None and r["status"]["backwards_done"] >= 5,
          json.dumps(r["status"])[:100] if r else "")
    good = diag_stats(log)
    # negative control: the old behaviour (scaling dropped) via sitecustomize
    sc = tmp / "sitecustom"
    sc.mkdir(exist_ok=True)
    (sc / "sitecustomize.py").write_text(
        "try:\n    import deltaserve.ft_meta as m\n"
        "    m.rope_scaling_of = lambda cfg: None\nexcept Exception:\n    pass\n")
    env2 = dict(env, PYTHONPATH=f"{sc}:{os.environ.get('PYTHONPATH', '')}")
    _, log2 = spawn("llama32", tmp, env2, tag="llama32-noscaling")
    bad = diag_stats(log2)
    check("rope_scaling reached the backward", "rope=llama3(theta=500000)" in log)
    check("control run really trained with plain RoPE",
          "rope=default" in log2 and "rope=llama3" in log)
    ok = good["lines"] >= 3 and bad["lines"] >= 3
    check("remat matches the served forward (bf16 noise)",
          ok and good["gate"] < 0.02 and good["resid"] < 2e-3, json.dumps(good))
    check("control: without the scaling it does not (≥3x on gate and residual)",
          ok and bad["gate"] > 3 * good["gate"] and bad["resid"] > 3 * good["resid"],
          json.dumps(bad))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--worker")
    ap.add_argument("--tmp")
    ap.add_argument("--phases", default="1,2,3,4,5")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    if a.worker:
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        tmp = Path(a.tmp)
        WORKERS[a.worker](tmp, Path(os.environ.get("DSERVE_E2E_OUT",
                                                   tmp / f"{a.worker}.json")))
        return
    tmp = Path(tempfile.mkdtemp(prefix="ds-e2e-"))
    print(f"(work dir {tmp})", flush=True)
    phases = set(a.phases.split(","))
    try:
        if "1" in phases:
            phase_http(tmp, a.tp)
        if a.tp == 1:
            ckpt = phase_served(tmp) if "2" in phases else None
            if "3" in phases:
                phase_failprofile(tmp)
            if "4" in phases and ckpt:
                phase_resume(tmp, ckpt)
            if "5" in phases:
                phase_llama32(tmp)
    finally:
        if not a.keep and not _failed:
            shutil.rmtree(tmp, ignore_errors=True)
    color = _R if _failed else _G
    print(f"\n{color}{_passed} passed, {_failed} failed{_X}")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
