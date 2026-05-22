#!/usr/bin/env python
"""Test the DeltaServe YAML config loader (vllm.deltaserve.config_loader).

CPU-only: builds EngineArgs from the YAML but never loads a model.

    python tests/test_config_loader.py

Run inside the `dserve-vllm` conda env. Exits 0 if all checks pass, 1 otherwise.
Run as a file (not `python -c ...`) so `vllm` resolves to the installed package
rather than the repo's ./vllm directory.
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_CONFIG = _REPO / "configs" / "serving_config_finetuning_opt.yaml"

_passed = 0
_failed = 0
_MISSING = object()  # sentinel: distinct from any real config value (incl. None)
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


def test_loads_and_maps():
    """Data-driven: every value in the YAML must appear, unchanged, in the right
    place in the parsed output. No hardcoded expectations — the YAML is the
    source of truth, so editing the config can't silently make this test stale.
    """
    green(f"load + map: {_CONFIG.relative_to(_REPO)}")
    from pathlib import Path

    from vllm.deltaserve.config_loader import engine_args_from_yaml, load_yaml_config

    # Compare against the *loaded* config (load_yaml_config resolves relative
    # adapter paths to absolute) so the expectations match the loader contract.
    raw = load_yaml_config(_CONFIG)
    engine_args, extras = engine_args_from_yaml(_CONFIG)

    passthrough = ("server", "adapters")
    special = ("finetune", "debug", *passthrough)

    # finetune + debug sections -> FinetuneConfig attributes (debug is folded in)
    ft = engine_args.finetune_config
    for section in ("finetune", "debug"):
        for key, want in (raw.get(section) or {}).items():
            check(f"{section}.{key} -> FinetuneConfig.{key} == {want!r}",
                  getattr(ft, key, _MISSING) == want)

    # passthrough sections -> extras[section] dict (verbatim)
    for section in passthrough:
        for key, want in (raw.get(section) or {}).items():
            check(f"{section}.{key} -> extras[{section!r}][{key!r}] == {want!r}",
                  extras.get(section, {}).get(key, _MISSING) == want)

    # every other section -> EngineArgs attributes (verbatim)
    for section, body in raw.items():
        if section in special:
            continue
        for key, want in (body or {}).items():
            check(f"{section}.{key} -> EngineArgs.{key} == {want!r}",
                  getattr(engine_args, key, _MISSING) == want)

    # relative adapter paths are resolved to absolute by the loader
    check("finetuning_lora_path resolved to absolute",
          Path(ft.finetuning_lora_path).is_absolute())
    check("adapters.lora_path_0 resolved to absolute",
          Path(extras["adapters"]["lora_path_0"]).is_absolute())


def test_error_handling():
    green("error handling:")
    import tempfile

    from vllm.deltaserve.config_loader import (
        engine_args_from_yaml,
        load_yaml_config,
    )

    # unknown finetune key -> FinetuneConfig rejects (extra=forbid)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("finetune:\n  enable_finetuning: true\n  bogus_key: 1\n")
        bad1 = f.name
    try:
        engine_args_from_yaml(bad1)
        check("unknown finetune key raises", False)
    except Exception:
        check("unknown finetune key raises", True)

    # non-dict section -> clear ValueError
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("model: just_a_string\n")
        bad2 = f.name
    try:
        load_yaml_config(bad2)
        check("non-dict section raises", False)
    except ValueError:
        check("non-dict section raises", True)

    # missing file -> FileNotFoundError
    try:
        load_yaml_config("/nonexistent/deltaserve.yaml")
        check("missing file raises", False)
    except FileNotFoundError:
        check("missing file raises", True)


def main():
    green("=== DeltaServe YAML config loader ===")
    test_loads_and_maps()
    test_error_handling()
    color = _RED if _failed else _GREEN
    print(f"\n{color}{_passed} passed, {_failed} failed{_RESET}")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
