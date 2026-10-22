#!/usr/bin/env python
"""Phase 1 / Step 1 test: the deltaserve plugin's config plumbing.

Co-serving is switched on through vLLM's ``additional_config``
(``{"deltaserve": {...}}``); the plugin parses it into a ``FinetuneConfig``,
selects the worker / scheduler classes and keys the compile cache on the
graph-affecting settings only. CPU-only, no model load.

    python tests/test_phase1_step1.py          # CPU-only plumbing checks
    python tests/test_phase1_step1.py --gpu    # also load Qwen3-0.6B and generate
Exits 0 if all checks pass, 1 otherwise.
"""

import sys

_passed = 0
_failed = 0

_TTY = sys.stdout.isatty()
_GREEN = "\033[92m" if _TTY else ""
_RED = "\033[91m" if _TTY else ""
_RESET = "\033[0m" if _TTY else ""


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


_MODEL = "Qwen/Qwen3-0.6B-Base"


def _ft(**kw):
    return {"deltaserve": {"enable_finetuning": True, **kw}}


def test_plugin_registered():
    green("plugin entry point:")
    from importlib.metadata import entry_points

    eps = {e.name: e.value for e in entry_points(group="vllm.general_plugins")}
    check("vllm.general_plugins has deltaserve", "deltaserve" in eps)
    check("points at deltaserve.plugin:register",
          eps.get("deltaserve") == "deltaserve.plugin:register")


def test_config_defaults():
    green("config defaults:")
    from deltaserve.config import FinetuneConfig

    check("FinetuneConfig() defaults enable_finetuning=False",
          FinetuneConfig().enable_finetuning is False)
    check("FinetuneConfig(enable_finetuning=True) honored",
          FinetuneConfig(enable_finetuning=True).enable_finetuning is True)


def _config(additional_config):
    from vllm.engine.arg_utils import EngineArgs

    return EngineArgs(model=_MODEL, additional_config=additional_config,
                      enforce_eager=True).create_engine_config()


def test_engine_config():
    green("engine config via additional_config:")
    import pickle

    from deltaserve.plugin import DeltaServeConfig

    off = _config({})
    check("no deltaserve entry → finetuning off",
          off.finetune_config.enable_finetuning is False)
    check("no deltaserve entry → stock worker",
          off.parallel_config.worker_cls != "deltaserve.worker.DeltaServeWorker")

    on = _config(_ft(ttft_slo=0.4))
    check("finetune_config parsed", on.finetune_config.enable_finetuning is True
          and on.finetune_config.ttft_slo == 0.4)
    check("additional_config is a DeltaServeConfig",
          isinstance(on.additional_config, DeltaServeConfig))
    check("worker_cls = DeltaServeWorker",
          on.parallel_config.worker_cls == "deltaserve.worker.DeltaServeWorker")
    check("scheduler_cls = FinetuneScheduler",
          "FinetuneScheduler" in str(on.scheduler_config.scheduler_cls))
    check("async scheduling defaults on", on.scheduler_config.async_scheduling)
    check("v1 model runner", not on.use_v2_model_runner)
    back = pickle.loads(pickle.dumps(on))
    check("finetune_config survives pickling",
          back.finetune_config.ttft_slo == 0.4)

    import vllm.kernels  # noqa: F401  (registers the IR op providers the hash reads)

    h = on.compute_hash()
    check("an SLO knob does not change the compile hash",
          _config(_ft(ttft_slo=0.2)).compute_hash() == h)
    check("graph_ft_batches does",
          _config(_ft(ttft_slo=0.4, graph_ft_batches=True)).compute_hash() != h)


def test_cli_parse():
    green("CLI parsing:")
    import json

    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    p = EngineArgs.add_cli_args(FlexibleArgumentParser())
    ns = p.parse_args(["--model", _MODEL, "--additional-config",
                       json.dumps(_ft())])
    ea = EngineArgs.from_cli_args(ns)
    check("--additional-config carries the deltaserve entry",
          ea.additional_config["deltaserve"]["enable_finetuning"] is True)


def test_gpu_runtime():
    green(f"GPU runtime (loads {_MODEL}):")
    from vllm import LLM

    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    llm = LLM(_MODEL, enforce_eager=True, gpu_memory_utilization=0.5,
              enable_lora=True, max_lora_rank=16,
              additional_config=_ft(
                  finetuning_lora_path=str(repo / "adapters/qwen3-0.6b-toy-lora-ft"),
                  data_path=str(repo / "alpaca_1000.txt")))
    out = llm.generate("Hello")
    check("LLM loaded and generated with finetuning on", len(out) > 0)


def main():
    run_gpu = "--gpu" in sys.argv
    green("=== Phase 1 / Step 1: plugin config plumbing ===")
    test_plugin_registered()
    test_config_defaults()
    test_engine_config()
    test_cli_parse()
    if run_gpu:
        test_gpu_runtime()
    else:
        print("(skipping GPU runtime check; pass --gpu to enable)")

    color = _RED if _failed else _GREEN
    print(f"\n{color}{_passed} passed, {_failed} failed{_RESET}")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
