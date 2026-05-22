#!/usr/bin/env python
"""Milestone 1 verification: FT injection must not perturb real inference.

Runs two GPU subprocesses on the SAME prompts (greedy, no inference LoRA):
  - baseline: enable_finetuning=False  (no FT injection)
  - ft-on:    enable_finetuning=True   (FT samples injected as prefill-only,
              routed to the FT adapter, retired same step)
and asserts the real-request output token ids are byte-identical. This catches:
FT output leaking to the frontend, FT KV corrupting real requests, mask
misalignment routing real tokens through the FT adapter, and force-eager drift.
It also confirms the FT-on engine terminates (no hang) and returns exactly one
output per prompt (no FT leakage).

    python tests/test_phase1_m1.py            # driver: spawns both workers, diffs
    python tests/test_phase1_m1.py --worker {baseline|ft}   # internal

Run on the 5090 (uses VLLM_USE_FLASHINFER_SAMPLER=0 to dodge the sm_120 sampler JIT).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_CONFIG = _REPO / "configs" / "serving_config_finetuning_opt.yaml"
_PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "The opposite of hot is",
    "Once upon a time, there was a",
]
_MAX_TOKENS = 32

_TTY = sys.stdout.isatty()
_GREEN = "\033[92m" if _TTY else ""
_RED = "\033[91m" if _TTY else ""
_RESET = "\033[0m" if _TTY else ""


def run_worker(mode: str) -> None:
    """Build an LLM (FT on/off), greedy-generate the prompts, print token ids."""
    from vllm import LLM, SamplingParams
    from vllm.config import FinetuneConfig
    from vllm.deltaserve.config_loader import load_yaml_config, split_config

    engine_kwargs, ft_cfg, _ = split_config(load_yaml_config(str(_CONFIG)))
    engine_kwargs["model"] = "facebook/opt-125m"
    enabled = mode == "ft"
    ft = FinetuneConfig(
        enable_finetuning=enabled,
        finetuning_lora_path=ft_cfg.finetuning_lora_path,
        data_path=ft_cfg.data_path,
        num_epochs=ft_cfg.num_epochs,
        max_saved_finetuning_tokens=ft_cfg.max_saved_finetuning_tokens,
        # enable the one-shot activation-hash check so the M2 test can observe it
        print_activation_hash=enabled,
    )
    llm = LLM(**engine_kwargs, finetune_config=ft)
    # Real requests use the BASE model (no inference LoRA) — isolates the effect
    # of FT injection on ordinary inference.
    outs = llm.generate(
        _PROMPTS, SamplingParams(temperature=0.0, max_tokens=_MAX_TOKENS)
    )
    result = {
        "mode": mode,
        "num_outputs": len(outs),
        "token_ids": [list(o.outputs[0].token_ids) for o in outs],
    }
    print("__RESULT__" + json.dumps(result))


def _spawn(mode: str) -> dict:
    env = dict(os.environ)
    env.setdefault("HF_HOME", "/mnt/storage/huggingface")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", mode],
        capture_output=True, text=True, env=env, timeout=600,
    )
    line = next((ln for ln in proc.stdout.splitlines()
                 if ln.startswith("__RESULT__")), None)
    if line is None:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        raise RuntimeError(f"worker '{mode}' produced no result (see above)")
    return json.loads(line[len("__RESULT__"):])


def main():
    if "--worker" in sys.argv:
        run_worker(sys.argv[sys.argv.index("--worker") + 1])
        return

    print(f"{_GREEN}=== Milestone 1: FT injection leaves inference unchanged ==={_RESET}")
    baseline = _spawn("baseline")
    fton = _spawn("ft")

    passed = failed = 0

    def check(name, cond):
        nonlocal passed, failed
        mark = f"{_GREEN}[PASS]{_RESET}" if cond else f"{_RED}[FAIL]{_RESET}"
        print(f"  {mark} {name}")
        if cond:
            passed += 1
        else:
            failed += 1

    check("baseline returned one output per prompt",
          baseline["num_outputs"] == len(_PROMPTS))
    check("ft-on returned one output per prompt (engine terminated, no FT leak)",
          fton["num_outputs"] == len(_PROMPTS))
    identical = baseline["token_ids"] == fton["token_ids"]
    check("real-request output token ids byte-identical (FT vs baseline)", identical)
    if not identical:
        for i, (b, f) in enumerate(zip(baseline["token_ids"], fton["token_ids"])):
            if b != f:
                print(f"    prompt[{i}] baseline={b}\n             ft      ={f}")

    color = _RED if failed else _GREEN
    print(f"\n{color}{passed} passed, {failed} failed{_RESET}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
