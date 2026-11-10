#!/usr/bin/env python
"""Milestone 1 verification: FT injection must not perturb real inference.

Every run serves the SAME prompts (greedy, base model, no LoRA on the real
requests) on the v1 model runner. Two prompts start at once; the other two
arrive _ARRIVALS later, while the first two decode, so prefills land at
different points of the backward cycle.

  - baseline / baseline-graph: enable_finetuning=False (eager / CUDA graphs).
  - ft-* runs: enable_finetuning=True. FT samples are injected as prefill-only
    requests routed to the FT adapter and retired the same step. The FT adapter
    is a copy of the toy one with lora_B scaled by the profile's amplify factor, so any leak of FT
    state into a real request (mask misalignment routing real tokens through
    the FT adapter, FT KV reused by a real request, a wrong LoRA mapping in a
    mixed batch, a save op writing into live activations) moves its logits by
    tens of nats instead of hiding in bf16 noise. (The unscaled toy adapter is
    too weak for that: applying it moves the first token by 0.09-0.45 nats.)
      ft / ft-graph            admission phase "prefill" (the default): FT rides
                               steps that carry a real prefill (+ any decodes).
      ft-both / ft-both-graph  phase "both": FT also rides decode-only steps.
    The -graph runs replay the has_ft piecewise CUDA graphs for mixed batches
    (graph_ft_batches) and for FT-only steps (graph_ft_only_batches).
  - control: enable_finetuning=False, the amplified adapter APPLIED to the real
    requests. A positive control: proves that leaking it would be unmistakable.

For each (baseline, ft) pair the ft run must actually co-batch FT with the step
kind its phase targets (real prefill / decode-only), read from its per-step
batch log. Its outputs must match the baseline's, except that bf16 results
depend on the batch they ran in. So, up to and including the first divergence,
the baseline token's logprob may move by at most _DRIFT_NATS, and a divergence
is accepted only where the ft run's top two candidates are within _TIE_NATS.
Each run must also return exactly one output per prompt (no FT output leaks to
the frontend, no hang).

    python tests/test_phase1_m1.py            # driver: spawns the workers, diffs
    python tests/test_phase1_m1.py --worker <mode>   # one worker (modes in _MODES)

The ft worker also runs standalone (test_phase1_m2.py does): without
DSERVE_M1_ADAPTER it trains the YAML's own FT adapter.

Run on the 5090 (uses VLLM_USE_FLASHINFER_SAMPLER=0 to dodge the sm_120 sampler JIT).
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "The opposite of hot is",
    "Once upon a time, there was a",
]
# Long enough that mixed steps land mid-generation even with graphs (~0.5 s of
# decode vs ~110 ms backward cycles): the real requests' KV written in a mixed
# step feeds the later tokens.
_MAX_TOKENS = 128
# Batch-shape noise with FT co-batched, measured: <= 0.18 nats on the baseline
# token. A leak of the amplified adapter moves it by tens of nats.
_DRIFT_NATS = 0.5
# A divergence is numerical noise when the ft run's two choices were this close.
# Logits are bf16 (steps of 0.125 at these magnitudes): 0.4 admits 3 steps, still
# 2x the measured drift and far below a leak's tens of nats.
_TIE_NATS = 0.4
_CONTROL_MIN_NATS = 5.0
# Profiles (DSERVE_M1_PROFILE; the worker subprocesses inherit it):
#   tp1  Qwen3-0.6B on one GPU, eager + graph, a minimal FT config (the default).
#   tp2  Qwen3-14B TP=2 with the production preset's FT config unchanged (graphs,
#        rank-symmetric tier-C aborts, pause_on_arrival, every activation save),
#        graph mode only. tests/test_tp_inference_leak.py runs it.
_PROFILES = {
    "tp1": dict(
        config="serving_config_finetuning_qwen3_0.6b.yaml",
        preset=False,
        # lora_B scale for the FT adapter. Measured: at 10x, applying the adapter
        # pushes the baseline's first token out of the top 5 on every prompt.
        amplify=10.0,
        # seconds after the start at which each prompt is submitted
        arrivals=[0.0, 0.0, 0.25, 0.5],
        timeout=600,
        modes=("baseline", "ft", "ft-both", "control",
               "baseline-graph", "ft-graph", "ft-both-graph"),
        # (label, baseline mode, ft mode, step kind the ft run must co-batch with)
        pairs=(("eager / prefill", "baseline", "ft", "prefill"),
               ("eager / both", "baseline", "ft-both", "decode"),
               ("graph / prefill", "baseline-graph", "ft-graph", "prefill"),
               ("graph / both", "baseline-graph", "ft-both-graph", "decode")),
    ),
    "tp2": dict(
        config="serving_config_finetuning_qwen3_14b_tp2.yaml",
        preset=True,
        # Qwen3-14B is less sensitive (lora_A half the 0.6B's scale): at 10x the
        # control moved a first token by only 0.4 nats
        amplify=100.0,
        arrivals=[0.0, 0.0, 0.6, 1.2],
        timeout=1500,
        modes=("baseline-graph", "ft-graph", "ft-both-graph", "control-graph"),
        pairs=(("TP=2 graph / prefill", "baseline-graph", "ft-graph", "prefill"),
               ("TP=2 graph / both", "baseline-graph", "ft-both-graph", "decode")),
    ),
}
_PROFILE_NAME = os.environ.get("DSERVE_M1_PROFILE", "tp1")
_PROFILE = _PROFILES[_PROFILE_NAME]
_CONFIG = _REPO / "configs" / "preset" / _PROFILE["config"]
_ARRIVALS = _PROFILE["arrivals"]
_MODES = _PROFILE["modes"]
_PAIRS = _PROFILE["pairs"]
_BATCH_RE = re.compile(r"\[batch [^\]]*\] prefill=(\d+) ft=(\d+) decode=\[([^\]]*)\]")

_TTY = sys.stdout.isatty()
_GREEN = "\033[92m" if _TTY else ""
_RED = "\033[91m" if _TTY else ""
_RESET = "\033[0m" if _TTY else ""


def _amplified_adapter(src: Path, dst: Path) -> Path:
    from safetensors.torch import load_file, save_file

    shutil.copytree(src, dst)
    w = load_file(str(src / "adapter_model.safetensors"))
    w = {k: (v * _PROFILE["amplify"] if "lora_B" in k else v) for k, v in w.items()}
    save_file(w, str(dst / "adapter_model.safetensors"))
    return dst


def _serve(llm, lora) -> list:
    """Submit _PROMPTS at their _ARRIVALS offsets, step until done, return the
    finished RequestOutputs in prompt order."""
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    eng = llm.llm_engine
    sp = SamplingParams(temperature=0.0, max_tokens=_MAX_TOKENS, logprobs=5,
                        output_kind=RequestOutputKind.FINAL_ONLY)
    done: dict[str, object] = {}
    nxt = 0
    t0 = time.monotonic()
    while len(done) < len(_PROMPTS):
        now = time.monotonic() - t0
        while nxt < len(_PROMPTS) and _ARRIVALS[nxt] <= now:
            eng.add_request(str(nxt), _PROMPTS[nxt], sp, lora_request=lora)
            nxt += 1
        if eng.has_unfinished_requests():
            for out in eng.step():
                if out.finished:
                    done[out.request_id] = out
        else:
            time.sleep(0.002)
    return [done[str(i)] for i in range(len(_PROMPTS))]


def run_worker(mode: str) -> None:
    """Build an LLM, serve the prompts, write token ids + logprobs."""
    # Every run on the v1 model runner. FT runs land there anyway (the plugin
    # routes finetuning to it), so the baselines must too: v2 differs in
    # numerics, and at the pinned vLLM commit v2 + LoRA + CUDA graphs fails at
    # warmup ("CUDA graphs must be captured on a non-default stream") even with
    # plugins disabled (VLLM_PLUGINS=) — an upstream issue, not ours.
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from deltaserve.config_loader import additional_config, load_yaml_config, split_config

    raw = load_yaml_config(str(_CONFIG))
    engine_kwargs, ft_cfg, _ = split_config(raw)
    graph = mode.endswith("-graph")
    engine_kwargs["enforce_eager"] = not graph
    adapter = os.environ.get("DSERVE_M1_ADAPTER", ft_cfg.finetuning_lora_path)
    enabled = mode.startswith("ft")
    common = dict(
        enable_finetuning=enabled,
        finetuning_lora_path=adapter,
        # enable the one-shot activation-hash check so the M2 test can observe it
        print_activation_hash=enabled,
        # per-step batch shape, parsed by the driver to prove co-batching
        print_engine_batch_exec=enabled,
        coserving_admission_phase="both" if "-both" in mode else "prefill",
    )
    if _PROFILE["preset"]:
        # the YAML's whole finetune / throttles / slo config, as served; only
        # admission opens at launch (no /start_finetuning offline)
        ft = additional_config(raw, start_on_launch=True, **common)["deltaserve"]
    else:
        ft = dict(
            data_path=ft_cfg.data_path,
            num_epochs=ft_cfg.num_epochs,
            max_saved_finetuning_tokens=ft_cfg.max_saved_finetuning_tokens,
            graph_ft_batches=graph,
            graph_ft_only_batches=graph,
            **common,
        )
    llm = LLM(**engine_kwargs, additional_config={"deltaserve": ft})
    lora = LoRARequest("amplified", 1, adapter) if mode.startswith("control") else None
    # Untimed warm-up generation (every mode, so the runs stay symmetric): the
    # first backward cycle is slow, and FT can only co-batch once it is done.
    llm.generate(["Warm-up:"], SamplingParams(temperature=0.0, max_tokens=_MAX_TOKENS))
    print("__SERVE_START__", flush=True)
    outs = _serve(llm, lora)
    print("__SERVE_END__", flush=True)
    result = {
        "mode": mode,
        "num_outputs": len(outs),
        "token_ids": [list(o.outputs[0].token_ids) for o in outs],
        # per position: [[token_id, logprob], ...] of the top candidates. A list,
        # not a dict: JSON turns int keys into strings, which once made every
        # near-tie lookup miss.
        "top": [[[[int(t), float(lp.logprob)] for t, lp in pos.items()]
                 for pos in o.outputs[0].logprobs] for o in outs],
    }
    # A file, not stdout: the backward child's prints share the stream and can
    # land in the middle of a long line.
    out_path = os.environ.get("DSERVE_M1_RESULT")
    if out_path:
        with open(out_path, "w") as f:
            json.dump(result, f)
    print("__RESULT__ written")


def _spawn(mode: str, adapter: Path) -> dict:
    env = dict(os.environ)
    env.setdefault("HF_HOME", "/mnt/storage/huggingface")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    fd, result_path = tempfile.mkstemp(suffix=f"_m1_{mode}.json")
    os.close(fd)
    os.unlink(result_path)
    env["DSERVE_M1_RESULT"] = result_path
    env["DSERVE_M1_ADAPTER"] = str(adapter)
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", mode],
        capture_output=True, text=True, env=env, timeout=_PROFILE["timeout"],
    )
    if not os.path.exists(result_path):
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        raise RuntimeError(f"worker '{mode}' produced no result (see above)")
    with open(result_path) as f:
        result = json.load(f)
    os.unlink(result_path)
    result["top"] = [[{t: lp for t, lp in pos} for pos in seq] for seq in result["top"]]
    # mixed steps of the served window only (the warm-up does not count)
    # under TP every rank prints the same batch line: keep rank 0's
    log = "\n".join(l for l in (proc.stdout + proc.stderr).splitlines()
                    if "Worker_TP" not in l or "Worker_TP0 " in l)
    a, b = log.find("__SERVE_START__"), log.find("__SERVE_END__")
    steps = [(int(p), int(f), bool(d.strip()))
             for p, f, d in _BATCH_RE.findall(log[a:b] if 0 <= a < b else "")]
    result["mixed"] = {
        "prefill": sum(1 for p, f, _ in steps if f and p),
        "decode": sum(1 for p, f, d in steps if f and not p and d),
    }
    return result


def _drift(base: dict, run: dict, i: int, upto: int) -> float:
    """Max |logprob change| of the baseline's token over positions 0..upto.
    A token missing from the run's top 5 counts as a 99-nat move."""
    worst = 0.0
    for j in range(upto + 1):
        t = base["token_ids"][i][j]
        worst = max(worst, abs(base["top"][i][j][t] - run["top"][i][j].get(t, -99.0)))
    return worst


def main():
    if "--worker" in sys.argv:
        run_worker(sys.argv[sys.argv.index("--worker") + 1])
        return

    print(f"{_GREEN}=== Milestone 1: FT injection leaves inference unchanged "
          f"[{_PROFILE_NAME}: {_CONFIG.name}] ==={_RESET}")
    from deltaserve.config_loader import load_yaml_config, split_config

    _, ft_cfg, _ = split_config(load_yaml_config(str(_CONFIG)))
    with tempfile.TemporaryDirectory() as tmp:
        adapter = _amplified_adapter(Path(ft_cfg.finetuning_lora_path), Path(tmp) / "ft")
        runs = {m: _spawn(m, adapter) for m in _MODES}

    passed = failed = 0

    def check(name, cond):
        nonlocal passed, failed
        mark = f"{_GREEN}[PASS]{_RESET}" if cond else f"{_RED}[FAIL]{_RESET}"
        print(f"  {mark} {name}")
        if cond:
            passed += 1
        else:
            failed += 1

    n = len(_PROMPTS)
    for m in _MODES:
        check(f"{m}: one output per prompt (no FT output leak, no hang)",
              runs[m]["num_outputs"] == n)
    ctl_m = next(m for m in _MODES if m.startswith("control"))
    ctl_base = "baseline-graph" if ctl_m.endswith("-graph") else "baseline"
    ctl = min(_drift(runs[ctl_base], runs[ctl_m], i, 0) for i in range(n))
    check(f"positive control: applying the amplified FT adapter moves the first "
          f"token by >= {_CONTROL_MIN_NATS} nats on every prompt (min {ctl:.1f})",
          ctl >= _CONTROL_MIN_NATS)

    for tag, base_m, ft_m, kind in _PAIRS:
        base, fton = runs[base_m], runs[ft_m]
        mixed = fton["mixed"]
        print(f"  -- {tag}: {base_m} vs {ft_m}  (mixed steps: {mixed['prefill']} "
              f"with a real prefill, {mixed['decode']} decode-only)")
        check(f"[{tag}] FT co-batched with real {kind} tokens", mixed[kind] > 0)
        for i, (b, f) in enumerate(zip(base["token_ids"], fton["token_ids"])):
            j = next((k for k in range(min(len(b), len(f))) if b[k] != f[k]), None)
            drift = _drift(base, fton, i, len(b) - 1 if j is None else j)
            ok_drift = drift <= _DRIFT_NATS
            if j is None:
                check(f"[{tag}] prompt[{i}]: identical output, baseline-token "
                      f"logprob drift {drift:.3f} <= {_DRIFT_NATS}", ok_drift)
                continue
            top = fton["top"][i][j]
            gap = top[f[j]] - top.get(b[j], float("-inf"))
            ok = gap < _TIE_NATS and ok_drift
            check(f"[{tag}] prompt[{i}]: diverges at token {j} on a near-tie "
                  f"(gap {gap:.3f} < {_TIE_NATS}), drift before it {drift:.3f} "
                  f"<= {_DRIFT_NATS}", ok)
            if not ok:
                print(f"    baseline={b}\n    ft      ={f}")

    color = _RED if failed else _GREEN
    print(f"\n{color}{passed} passed, {failed} failed{_RESET}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
