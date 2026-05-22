#!/usr/bin/env python
"""Phase 1 / Step 1 test: the --enable-finetuning config flag.

Lightweight checks that the FinetuneConfig flag is plumbed end-to-end through
vLLM's config system and CLI, all CPU-only (no model load).

    python tests/test_phase1_step1.py          # CPU-only plumbing checks
    python tests/test_phase1_step1.py --gpu     # also load opt-125m and show
                                                # the worker's runtime print

Run inside the `dserve-vllm` conda env (see vllm_setup_5090.md).
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


def test_config_defaults():
    green("config defaults:")
    from vllm.config import FinetuneConfig, VllmConfig

    check("FinetuneConfig() defaults enable_finetuning=False",
          FinetuneConfig().enable_finetuning is False)
    check("FinetuneConfig(enable_finetuning=True) honored",
          FinetuneConfig(enable_finetuning=True).enable_finetuning is True)
    check("VllmConfig has 'finetune_config' field",
          "finetune_config" in VllmConfig.__dataclass_fields__)


def test_engine_args():
    green("EngineArgs plumbing:")
    from vllm.config import FinetuneConfig
    from vllm.engine.arg_utils import EngineArgs

    ea = EngineArgs(model="facebook/opt-125m")
    check("EngineArgs exposes finetune_config",
          isinstance(ea.finetune_config, FinetuneConfig))
    check("EngineArgs default enable_finetuning=False",
          ea.finetune_config.enable_finetuning is False)
    ea.finetune_config = FinetuneConfig(enable_finetuning=True)
    check("EngineArgs setter honored",
          ea.finetune_config.enable_finetuning is True)


def test_cli_parse():
    green("CLI parsing (FlexibleArgumentParser):")
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    p = FlexibleArgumentParser()
    p = EngineArgs.add_cli_args(p)
    ns = p.parse_args([
        "--model", "facebook/opt-125m",
        "--finetune-config.enable_finetuning=true",
    ])
    ea = EngineArgs.from_cli_args(ns)
    check("--finetune-config.enable_finetuning=true parsed True",
          ea.finetune_config.enable_finetuning is True)

    p2 = FlexibleArgumentParser()
    p2 = EngineArgs.add_cli_args(p2)
    ns2 = p2.parse_args(["--model", "facebook/opt-125m"])
    ea2 = EngineArgs.from_cli_args(ns2)
    check("omitting the flag defaults to False",
          ea2.finetune_config.enable_finetuning is False)


def test_gpu_runtime_print():
    green("GPU runtime print (loads facebook/opt-125m):")
    from vllm import LLM
    from vllm.config import FinetuneConfig

    # Watch stdout for: [deltaserve] Worker ... enable_finetuning=True
    llm = LLM("facebook/opt-125m",
              finetune_config=FinetuneConfig(enable_finetuning=True))
    out = llm.generate("Hello")
    check("LLM loaded and generated with enable_finetuning=True",
          len(out) > 0)
    print("  (scroll up for the [deltaserve] Worker ... line)")


def main():
    run_gpu = "--gpu" in sys.argv
    green("=== Phase 1 / Step 1: --enable-finetuning flag ===")
    test_config_defaults()
    test_engine_args()
    test_cli_parse()
    if run_gpu:
        test_gpu_runtime_print()
    else:
        print("(skipping GPU runtime check; pass --gpu to enable)")

    color = _RED if _failed else _GREEN
    print(f"\n{color}{_passed} passed, {_failed} failed{_RESET}")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
