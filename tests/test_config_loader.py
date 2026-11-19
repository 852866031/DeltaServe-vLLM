#!/usr/bin/env python
"""Test the DeltaServe YAML config loader (deltaserve.config_loader).

CPU-only: builds EngineArgs from the YAML but never loads a model.

    python tests/test_config_loader.py

Run inside the `dserve-vllm` conda env. Exits 0 if all checks pass, 1 otherwise.
Run as a file (not `python -c ...`) so `vllm` resolves to the installed package
rather than the repo's ./vllm directory.
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_CONFIG = _REPO / "configs" / "preset" / "serving_config_finetuning_qwen3_0.6b.yaml"

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

    from deltaserve.config_loader import engine_args_from_yaml, load_yaml_config

    # Compare against the *loaded* config (load_yaml_config resolves relative
    # adapter paths to absolute) so the expectations match the loader contract.
    raw = load_yaml_config(_CONFIG)
    engine_args, extras = engine_args_from_yaml(_CONFIG)

    passthrough = ("server", "adapters")
    from deltaserve.config_loader import ALL_FINETUNE_SECTIONS as ft_sections
    special = (*ft_sections, *passthrough)

    # the four finetune sections -> one flat FinetuneConfig, carried to vLLM in
    # additional_config["deltaserve"]
    from deltaserve.config import FinetuneConfig
    ft = FinetuneConfig(**engine_args.additional_config["deltaserve"])
    for section in ft_sections:
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
            got = getattr(engine_args, key, _MISSING)
            if key == "model" and isinstance(got, str) and got != want:
                # vLLM's arg_utils replaces a hub model id with its local
                # snapshot path under HF_HUB_OFFLINE; accept that form.
                ok = str(want).replace("/", "--") in got and got.startswith("/")
                check(f"{section}.{key} -> EngineArgs.{key} == {want!r} "
                      f"(or its local snapshot path)", ok)
                continue
            check(f"{section}.{key} -> EngineArgs.{key} == {want!r}", got == want)

    # relative adapter paths are resolved to absolute by the loader
    check("finetuning_lora_path resolved to absolute",
          Path(ft.finetuning_lora_path).is_absolute())
    check("adapters.lora_path_0 resolved to absolute",
          Path(extras["adapters"]["lora_path_0"]).is_absolute())


def test_error_handling():
    green("error handling:")
    import tempfile

    from deltaserve.config_loader import (
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

    # a key in two finetune sections -> clear ValueError
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("finetune:\n  ttft_slo: 0.3\nslo:\n  ttft_slo: 0.4\n")
        bad3 = f.name
    try:
        engine_args_from_yaml(bad3)
        check("key in two finetune sections raises", False)
    except ValueError:
        check("key in two finetune sections raises", True)

    # missing file -> FileNotFoundError
    try:
        load_yaml_config("/nonexistent/deltaserve.yaml")
        check("missing file raises", False)
    except FileNotFoundError:
        check("missing file raises", True)


def test_layout():
    """Every shipped YAML (templates + presets) loads, in the current layout:
    no old ``finetune`` section, every finetune key in its own section, and
    every finetune / adapter path that is set exists (A100 presets carry the
    A100 box's absolute paths and are only loaded)."""
    green("layout of every shipped YAML:")
    import contextlib
    import io
    import os

    import yaml

    from deltaserve.config_loader import (
        FINETUNE_SECTIONS,
        LEGACY_SECTIONS,
        SECTION_OF,
        load_yaml_config,
        split_config,
    )

    # each key has exactly one home section
    homes = [k for keys in FINETUNE_SECTIONS.values() for k in keys]
    check("no key listed in two sections", len(homes) == len(set(homes)))
    check("sections: finetune_param, finetune_features, throttles, slo, debug, "
          "telemetry", tuple(FINETUNE_SECTIONS) == (
              "finetune_param", "finetune_features", "throttles", "slo",
              "debug", "telemetry"))
    from deltaserve.config import FinetuneConfig
    check("every sectioned key is a FinetuneConfig field",
          all(k in FinetuneConfig.__dataclass_fields__ for k in homes))

    files = sorted((_REPO / "configs").glob("template_*.yaml")) + \
        sorted((_REPO / "configs" / "preset").glob("*.yaml"))
    check(f"found templates + presets ({len(files)})",
          len(list((_REPO / "configs").glob("template_*.yaml"))) >= 3
          and len(files) > 10)
    for path in files:
        name = path.relative_to(_REPO)
        raw_text = yaml.safe_load(path.read_text())
        misplaced = [f"{sec}.{k}" for sec in FINETUNE_SECTIONS
                     for k in (raw_text.get(sec) or {})
                     if SECTION_OF.get(k, sec) != sec]
        legacy = [s for s in LEGACY_SECTIONS if s in raw_text]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            raw = load_yaml_config(path, verbose=False)
            _, ft, extras = split_config(raw)
        missing = []
        if "A100" not in path.name:
            for k in ("finetuning_lora_path", "data_path"):
                v = getattr(ft, k)
                if v and not os.path.exists(v):
                    missing.append(k)
            for k, v in (extras.get("adapters") or {}).items():
                if not os.path.exists(v):
                    missing.append(k)
        check(f"{name}: loads, layout current, paths exist",
              not misplaced and not legacy and not missing
              and ft.enable_finetuning
              and "WARNING" not in out.getvalue(),
              ) if True else None
        if misplaced or legacy or missing:
            print(f"      misplaced={misplaced} legacy={legacy} missing={missing}")
    tmpl = [p for p in files if p.name.startswith("template_")]
    for path in tmpl:
        body = [ln for ln in path.read_text().splitlines()
                if ln.startswith("  ") and ":" in ln and not ln.strip().startswith("#")]
        uncommented = [ln.strip().split(":")[0] for ln in body if "#" not in ln]
        allowed = {"host", "port", "tokenizer_mode", "trust_remote_code", "model",
                   "ttft_slo"}
        check(f"{path.name}: every setting carries a comment",
              not (set(uncommented) - allowed), )
        if set(uncommented) - allowed:
            print(f"      uncommented: {sorted(set(uncommented) - allowed)}")


def test_overrides():
    """additional_config: an override replaces the YAML value; a None override
    (the launchers' default for bwd_log_path / step_trace_path) keeps it."""
    green("overrides:")
    from deltaserve.config_loader import additional_config
    cfg = {"telemetry": {"bwd_log_path": "/yaml.csv"},
           "finetune_param": {"enable_finetuning": True}}
    ds = additional_config(cfg, bwd_log_path=None, step_trace_path=None)["deltaserve"]
    check("None override keeps the YAML value", ds.get("bwd_log_path") == "/yaml.csv"
          and "step_trace_path" not in ds)
    ds = additional_config(cfg, bwd_log_path="/run.csv")["deltaserve"]
    check("a real override wins", ds["bwd_log_path"] == "/run.csv")


def test_legacy_layout():
    """The pre-2026-11 single ``finetune`` section still loads to the same
    FinetuneConfig, with one note (no per-key warnings); the converter splits
    it and the result loads identically."""
    green("legacy 'finetune' section:")
    import contextlib
    import io
    import subprocess
    import tempfile

    from deltaserve.config_loader import finetune_kwargs, load_yaml_config

    old = ("finetune:\n  enable_finetuning: true\n  learning_rate: 0.001\n"
           "  graph_ft_batches: true\n  save_attn_ctx: true\n"
           "debug:\n  bwd_log_path: /tmp/x.csv\n  print_scheduler_add: true\n"
           "model:\n  model: Qwen/Qwen3-0.6B-Base\n")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(old)
        p_old = f.name
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        kw_old = finetune_kwargs(load_yaml_config(p_old, verbose=False))
    log = out.getvalue()
    check("legacy section loads every key",
          kw_old == {"enable_finetuning": True, "learning_rate": 0.001,
                     "graph_ft_batches": True, "save_attn_ctx": True,
                     "bwd_log_path": "/tmp/x.csv", "print_scheduler_add": True})
    check("one note for the legacy section, no per-key warning for it",
          log.count("old layout") == 1
          and "'finetune." not in log)
    check("a key moved out of debug warns (bwd_log_path → telemetry)",
          "'debug.bwd_log_path' belongs in section 'telemetry'" in log)
    conv = subprocess.run([sys.executable, str(_REPO / "scripts/convert_config_layout.py"),
                           p_old], capture_output=True, text=True)
    new = conv.stdout
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(new)
        p_new = f.name
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        kw_new = finetune_kwargs(load_yaml_config(p_new, verbose=False))
    check("converter splits it (finetune_param / finetune_features / telemetry)",
          conv.returncode == 0 and "finetune_param:" in new
          and "finetune_features:" in new and "telemetry:" in new
          and "\nfinetune:" not in new)
    check("converted file loads identically, silently",
          kw_new == kw_old and "WARNING" not in out.getvalue()
          and "old layout" not in out.getvalue())


def main():
    green("=== DeltaServe YAML config loader ===")
    test_loads_and_maps()
    test_error_handling()
    test_layout()
    test_overrides()
    test_legacy_layout()
    color = _RED if _failed else _GREEN
    print(f"\n{color}{_passed} passed, {_failed} failed{_RESET}")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
